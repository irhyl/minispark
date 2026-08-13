# MiniSpark Architecture

This document explains the layering of MiniSpark and, for each layer, *why*
it exists as a separate thing rather than being folded into its neighbor.
It is updated as each milestone lands; sections describing unimplemented
subsystems are added when those subsystems exist, not before.

## Layering (target, from the build spec)

```
User API
   |
DataFrame / Dataset API
   |
Expression System
   |
Logical Plan
   |
Analyzer
   |
Query Optimizer
   |
Physical Plan
   |
DAG Builder
   |
Stage Planner
   |
Scheduler
   |
Task Execution
   |
Storage / Shuffle
```

**Milestone 5 status**: every layer above is implemented. `DataFrame`
actions (`collect`/`show`/`count`/`explain`) run, in order:
`logical/analyzer.py` (validates the plan), `optimizer/optimizer.py`
(rewrites it), `physical/planner.py` (translates it to a physical plan,
including turning `group_by(...).agg(...)` into a partial aggregate, a
shuffle exchange, and a final aggregate; `join(...)` into a shuffle hash
join or a broadcast join; `order_by(...)` into a local sort, a range
exchange, and a final sort), `execution/stages.py` (splits it into stages
at shuffle boundaries; one stage for a shuffle-free plan, two or more for
a plan with `group_by`/`join`/`order_by`, see `docs/execution-model.md`),
`execution/scheduler.py`'s `LocalScheduler` (runs each stage's `Task`s,
either sequentially or across a real `ProcessPoolExecutor` depending on
`local[N]`, moving data between stages through a real disk-backed
shuffle, see `docs/shuffle.md`). See `docs/execution-model.md` for the
full DAG/Stage/Task/Worker/Scheduler picture; this file stays focused on
package layout and design decisions.

`minispark/execution/executor.py` (Milestone 1's naive, single-process,
logical-plan interpreter) and `physical/operators.py`'s whole-Dataset
`execute()` (Milestone 2) are both no longer on the `DataFrame` action
path. Both are retained as correctness oracles: `tests/unit/
test_physical_plan.py`, `tests/unit/test_optimizer.py`, and
`tests/unit/test_worker.py` assert that later, more complex execution
paths produce the same rows a simpler earlier one would, on the
equivalent unoptimized plan.

## Package layout and why each package exists

- **`minispark/core/`** — `DataType`, `Schema`, `Field`, `Record`,
  `Partition`, `Dataset`. **This is a deviation from the structure sketched
  in the build spec**, which only lists a top-level `storage/` package and
  implies Dataset/Partition might live there. They were pulled into their
  own `core/` package instead because they are not a storage concern: the
  DataFrame API, the logical/physical planners, and the execution engine
  all need to agree on what a row/partition/dataset *is*, independent of
  where the data came from. `core/` has zero dependencies on any other
  MiniSpark package; every other package depends on it, never the reverse.

- **`minispark/expressions/`** — the expression tree (`Column`, `Literal`,
  comparison/arithmetic/boolean operators, `IsNull`/`IsNotNull`/`Not`,
  `Alias`). Operator overloads (`__gt__`, `__add__`, ...) live on the
  `Expression` base class, not only on `Column`, so composite expressions
  like `(col("a") + col("b")) > 10` build correctly. `Expression.evaluate()`
  is the row-at-a-time evaluation used by the naive executor; it is not
  vectorized and is expected to be replaced by batch evaluation over Arrow
  arrays when columnar execution (Milestone 7) lands — row-at-a-time
  evaluation should not be assumed to be the permanent execution strategy.

- **`minispark/logical/`**: `Scan`, `Filter`, `Project`, `Aggregate`,
  `Join`, and `Sort` plan nodes, an `explain()` pretty-printer
  (`plan.py`), and `analyzer.py`. Limit/Union/Repartition/Distinct nodes
  are intentionally not stubbed out empty here; they are added in the
  milestone that gives each one real behavior so that "the node exists"
  always means "the node does something." `Aggregate.group_by` and
  `Sort.sort_exprs` both require plain `Column`s, not computed
  expressions (each is evaluated as a shuffle-partitioning key, see
  `docs/shuffle.md`, not a general expression); `Join` requires its `on`
  columns to be present, by name, on both sides, and only supports
  `how="inner"` (see `Join`'s own docstring for exactly what is and is
  not supported, and why). `analyzer.py`'s `analyze()` validates every
  `Column` reference in a plan against its child schema before anything
  executes, raising `AnalysisException` with the offending column name
  and the available columns; it does not do type inference beyond "does
  this column exist" (see Key design decisions below).

- **`minispark/optimizer/`**: `rules.py` (five rules: `ConstantFolding`,
  `FilterSimplification`, `PredicatePushdown`, `ProjectionPruning`,
  `RedundantProjectionElimination`, every one of which has a case for
  every logical node type, `Aggregate`/`Join`/`Sort` included),
  `optimizer.py` (`Optimizer`, which runs the rule list to a
  text-comparison fixed point), `statistics.py` (`compute_statistics()`,
  exact full-scan statistics; used by `physical/planner.py` for
  `order_by()`'s range-partition boundaries, see Key design decisions
  below, still not consulted by any optimizer rule). Depends on
  `logical/` only, never on `physical/` or `execution/`: an optimizer
  rule rewrites plan shape, it never touches a Dataset or a Partition.

- **`minispark/physical/`**: `plan.py` (`ScanExec`, `FilterExec`,
  `ProjectExec`, `HashAggregateExec`, `HashJoinExec`, `SortExec`,
  `ExchangeExec`, `ShuffleWriteExec`, `ShuffleReadExec`, plus a `leaves()`
  helper for walking a multi-child plan's leaves), `planner.py`
  (`plan_physical()`; 1:1 for Scan/Filter/Project. `Aggregate` becomes a
  partial `HashAggregateExec` -> `ExchangeExec` -> final
  `HashAggregateExec` chain; `Join` becomes one `HashJoinExec` whose
  children are wrapped in `ExchangeExec`s differently depending on the
  `broadcast` hint; `Sort` becomes a local `SortExec` -> range
  `ExchangeExec` -> final `SortExec` chain, the one physical-planning
  case that touches data, see Key design decisions below. See
  `docs/shuffle.md` for all three), `operators.py` (`execute()`, a
  whole-Dataset oracle only; `execute_partition()`, what a Task actually
  runs, with a case per physical node type). `ExchangeExec`/
  `ShuffleWriteExec`/`ShuffleReadExec` are the seam that let
  `Aggregate`/`Join`/`Sort` each define a wide dependency without
  duplicating the stage-splitting or shuffle machinery; a future Sort-
  Merge join would extend the same seam rather than needing a new one.

- **`minispark/shuffle/`**: `partitioner.py` (`HashPartitioner`, used by
  `group_by`/`join`; `RangePartitioner`, used by `order_by()` as of
  Milestone 5), `writer.py`/`reader.py` (disk-backed, checksummed shuffle
  blocks), `manager.py` (`ShuffleManager`, driver-side bookkeeping of
  which blocks exist for which stage/partition). See `docs/shuffle.md`
  for the full picture, including how a broadcast join and a range-
  partitioned sort each reuse this same machinery. Depends on `core/`
  only.

- **`minispark/storage/`** — `DataSource` (abstract), `MemoryDataSource`,
  `CSVDataSource`. Depends only on `core/`. A `Scan` logical node holds an
  already-`.read()` `Dataset`, not a `DataSource` reference — so the
  logical-plan layer never imports the storage layer's I/O code, only the
  data model it produces.

- **`minispark/execution/`**: `executor.py` (Milestone 1's naive
  logical-plan interpreter, kept only as a correctness oracle), `dag.py`,
  `stages.py`, `tasks.py`, `worker.py`, `scheduler.py`. See
  `docs/execution-model.md` for what each of these does and how they fit
  together; that document, not this one, is the place to look for the
  full DAG/Stage/Task/Worker/Scheduler picture.

- **`minispark/api/`** — `DataFrame` (lazy; `filter`/`select`/`group_by`/
  `join`/`order_by` (alias `sort`) build plan nodes, `collect`/`show`/
  `count`/`explain` are the only things that trigger analysis/
  optimization/execution), `grouped.py` (`GroupedData`, the result of
  `group_by()` before `.agg()` turns it back into a `DataFrame`),
  `MiniSparkSession` (+ builder), `functions.py` (`col()`, `lit()`,
  `count()`, `sum()`, `avg()`, `min()`, `max()`). `DataFrame.join()`
  intentionally only accepts `on=` (common column names on both sides),
  matching `Join`'s own scope (see `logical/`, above). `explain(
  optimized=False)` (the default) prints the raw logical plan, matching
  Milestone 1's behavior exactly. `explain(optimized=True)` prints
  "Analyzed Logical Plan" (post-`analyze()`, pre-rewrite), "Optimized
  Logical Plan" (post-`Optimizer.optimize()`), "Physical Plan" (post-
  `plan_physical()`), and "Stages" (post-`build_stages()`, one section
  per stage, however many that turns out to be), so a user can see what
  each step changed.

- **`minispark/config/`** — `Config`/`EngineConfig`/`ExecutionConfig`/
  `MemoryConfig`/`OptimizerConfig` dataclasses matching the shape in the
  build spec, and structured logging setup (`log.py`). `engine.master`
  (via `EngineConfig.num_workers`) and `engine.max_task_retries` are read
  by `execution/scheduler.py`'s `LocalScheduler`; `optimizer.
  predicate_pushdown` / `optimizer.projection_pruning` are read by
  `optimizer/optimizer.py`'s `default_rules()`; `execution.
  shuffle_partitions` is read by `physical/planner.py` when translating
  an `Aggregate` or a non-broadcast `Join` (how many reduce-side
  partitions the shuffle fans out to) and as the target for `Sort`'s
  range partitioning, when a range split is possible (see
  `docs/shuffle.md`'s Sort section for when it falls back to one
  partition instead). `execution.partition_size_mb`, `execution.
  shuffle_compression`, and `memory` are still unread.

## Key Milestone-5 design decisions

**`Join` is the first multi-input logical/physical node, and it ripples.**
Every node before it (`Filter`, `Project`, `Aggregate`) has exactly one
child; `Join` has two. That single fact required updating almost every
generic tree-walker in the codebase: the four optimizer rules (each
needed a two-child recursion case), `execution/stages.py`'s `_split()`
(a `HashJoinExec` splits each side independently, either side may close
its own upstream stage(s)), and, biggest of all, `execution/tasks.py`'s
`Task.shuffle_blocks`, which changed from a flat `list[ShuffleBlockMeta]`
to a `dict[stage_id, list[ShuffleBlockMeta]]`, because a `HashJoinExec`-
rooted stage's task needs blocks from *two* different upstream stages
(see `docs/shuffle.md`'s "Reading from more than one prior stage"), and a
flat list has no way to say which blocks came from which side.
`physical/plan.py`'s `leaves()` helper (walks every leaf of a
possibly-multi-child tree, not just `children[0]`) is what makes
`execution/worker.py` and `execution/scheduler.py` able to find both
`ShuffleReadExec`s without hardcoding "there are exactly two."

**Broadcast join reuses the shuffle machinery instead of a separate
broadcast mechanism.** A broadcast is implemented as `ExchangeExec`/
`ShuffleWriteExec`/`ShuffleReadExec` with `num_partitions=1`: the small
side is "shuffled" to a single target partition (`HashPartitioner(1)`
sends everything there regardless of key, so no new partitioner was
needed), and every task in the consuming stage reads that same partition
in full, via `is_broadcast=True` on `ExchangeExec`/`ShuffleReadExec`,
consulted only by `execution/scheduler.py`'s task-building code. This
was a deliberate choice over inventing a separate "broadcast a value to
every worker" mechanism: it stays inside the disk-backed shuffle model
already built and tested in Milestone 4, at the cost of writing the
small side to disk even though it will be read back by every consumer
task, a real, accepted overhead for the architectural simplicity.

**`Sort` is the one place physical planning touches data, and it says so
loudly.** Range-partitioning `order_by()`'s shuffle needs boundary values
computed from the sort key's actual range before the shuffle that uses
them runs; there is no distributed sampling stage to get those without
looking at data early. `physical/planner.py`'s `_sort_range_boundaries()`
eagerly runs the child plan and calls `optimizer/statistics.py`'s
`compute_statistics()` right there, breaking "a plan is built without
touching data," true of every other node in this codebase since
Milestone 1. This is flagged in the function's own docstring, in
`docs/query-planning.md`, and in `docs/shuffle.md`, not silently done:
the alternative (a real sampling stage that runs through the scheduler
before the main sort stage) would preserve the invariant but is enough
additional machinery (a stage whose sole purpose is producing input to
another stage's *planning*, not its execution) that it was cut from this
milestone's scope, not attempted and abandoned.

**A descending sort needed a real bug fix, caught by testing, not
inspection.** The first version of range-partition boundary computation
did not account for direction: `RangePartitioner` always assigns
ascending target partitions, and the scheduler always merges partitions
back in id order, so a naive implementation produced a result that was
locally sorted (each partition correct on its own) but globally wrong
(partitions themselves not in the right order) for `ascending=False`. A
real-multiprocessing integration test caught it immediately by comparing
against Python's own `sorted(..., reverse=True)`. The fix negates the
partitioning key and the computed boundaries for a descending primary
sort key, rather than teaching `RangePartitioner` or the scheduler
anything about sort direction; see `docs/shuffle.md`'s Sort section and
`physical/planner.py`'s `_sort_range_boundaries()` docstring for exactly
how.

## Key Milestone-4 design decisions

**`HashPartitioner` cannot use Python's builtin `hash()`.** CPython
randomizes `hash()` for `str` per process (`PYTHONHASHSEED`) unless
disabled. Two worker processes computing a group key's target partition
with the builtin `hash()` could disagree on the same string key, silently
splitting one group's rows across two shuffle target partitions, a
correctness bug that would not show up in single-process testing.
`shuffle/partitioner.py` uses `hashlib.md5` over `repr(key)` instead,
which is stable across processes; verified in `tests/unit/
test_partitioner.py` by spawning a real second Python process.

**Grouping is not streaming; a shuffle write is.** `HashAggregateExec`
(physical/operators.py) has to see every row for a key (within one
partition) before it knows that group's final state, so it builds a
`dict` up front rather than yielding lazily like `FilterExec`/
`ProjectExec` do. `shuffle/writer.py`, by contrast, writes one record at
a time to whichever target file it belongs to, the only per-target state
kept in memory is one open file handle and a running checksum. Spilling
the aggregate's hash table to disk under memory pressure is not
implemented (Milestone 9); the architecture note is in `docs/shuffle.md`.

**Shuffle blocks are pickled records, not JSON lines.** `Avg`'s partial
state is a `(sum, count)` tuple (`expressions/aggregate.py`); JSON would
silently turn that into a list on the way back out, and cannot represent
`NaN` by default. Every worker process reading a shuffle block is already
a MiniSpark process willing to unpickle MiniSpark data (the same trust
boundary Milestone 3's Task/PhysicalPlan picklability already relies on),
so pickle costs nothing extra in trust and preserves exact types.

**`ExchangeExec` and `ShuffleWriteExec`/`ShuffleReadExec` are different
node types, not the same one used two ways.** `physical/planner.py`
leaves an abstract `ExchangeExec` marker at a shuffle boundary; only
`execution/stages.py`'s `build_stages()` rewrites it into the concrete
`ShuffleWriteExec` (ends the upstream stage)/`ShuffleReadExec` (starts
the downstream stage) pair a Task actually executes. This keeps "the
physical plan says a shuffle happens here" (visible in `explain(
optimized=True)`'s "Physical Plan" section) separate from "here is
concretely how that shuffle is split across two stages" (the "Stages"
section): a bare `ExchangeExec` reaching a worker is a stage-splitting
bug, not a valid state, and the type system distinguishes the two rather
than relying on a flag.

## Key Milestone-3 design decisions

**Partition row-data had to stop being closures.** `local[N]` with `N > 1`
sends `Task`s (which carry a whole `PhysicalPlan`, including its `Scan`
leaf's `Dataset`) to worker processes with the standard library `pickle`
module. A lambda, or a nested function closing over an enclosing method's
variables, is not picklable no matter what it captures. `storage/
memory.py`, `storage/csv.py`, and `core/dataset.py`'s `repartition()` were
rewritten to build `records_fn` with `functools.partial(iter, rows)` or
`functools.partial(a_module_level_function, ...)` instead, which pickles
correctly and preserves `Partition`'s public `records_fn: Callable[[],
Iterator[Record]]` contract exactly, so nothing that constructs a
`Partition` directly with a raw lambda (most unit tests, which never cross
a process boundary) needed to change.

**Retry decisions are made by the scheduler, not inside a worker.**
`execute_task` (execution/worker.py) already converts an exception into a
`FAILED` `TaskResult` instead of raising, so "should this be retried" is
always a plain inspection of a returned value, made in the scheduler's own
process, whether that value came back from a direct call (`local[1]`) or
from a `ProcessPoolExecutor` worker (`local[N>1]`). This also means a
`FAILED` result and a genuinely crashed/killed worker process are
distinguishable in principle (a crash would show up as the pool itself
raising, not as a returned `TaskResult`), which matters for honestly
scoping what Milestone 3's retry actually covers versus what Milestone 6's
lineage-based recomputation will need to cover.

**Stage splitting is real, even though it only ever produces one stage
right now.** `execution/stages.py`'s `build_stages()` routes through
`execution/dag.py`'s dependency classification and explicitly checks for a
wide dependency (raising `NotImplementedError` if it finds one) rather
than hardcoding "return one Stage." No physical node is wide until
Milestone 4's `Aggregate`, so today that check always passes and one stage
is always the right answer; the check exists so the day it stops being the
right answer, the code says so loudly instead of silently producing a
wrong single-stage plan for a query that actually needed a shuffle
boundary.

**The scheduler's task runner is an injectable constructor argument.**
`LocalScheduler(run_task=...)` defaults to `execute_task`, but tests can
pass a synchronous stub instead. This is what keeps
`tests/unit/test_scheduler.py`'s retry/state-tracking tests fast and
deterministic (no real subprocess spawn) without weakening what they
prove: scheduling logic is tested in isolation from the mechanism that
actually runs a task, and genuine multiprocessing gets its own dedicated
tests (`tests/integration/test_scheduler_multiprocessing.py`) that assert
on something a stub cannot fake, an observed worker process id different
from the driver's.

## Key Milestone-2 design decisions

**Fixed-point rule application compares explain() text, not plan equality.**
`Optimizer.optimize()` reruns its rule list until two passes produce
identical output, capped at 10 iterations. "Identical" is checked by
rendering both plans with `explain_string()` and comparing the strings,
not by `plan_a == plan_b`. `Expression.__eq__` is overloaded to build an
`Equal` expression node rather than return a bool (see the Milestone-1
decision below), so plan objects containing expressions cannot be compared
with `==` in the first place; adding a separate structural-equality method
only for this one use would be more machinery than the problem needs.

**Predicate pushdown pushes a Filter below a Project, not below a Join.**
As of Milestone 2, the textbook example (`Project -> Join -> Filter`
becomes `Project -> Join (one side wrapped in Filter)`) needs a Join node,
which does not exist yet. Pushing a Filter below a Project, when every
column the filter needs is present on the Project's child, is the
meaningful instance of the same idea available with today's node set:
rows get dropped before the (comparatively cheap) projection work runs,
instead of after. *(Milestone 5 adds the Join case this bullet describes
as missing; see Key Milestone-5 design decisions, above, and
`optimizer/rules.py`'s `PredicatePushdown` docstring.)*

**Projection pruning narrows columns in the plan, not bytes read from
disk.** `ProjectionPruning` inserts a Column-only `Project` directly above
`Scan` when the Scan's schema is wider than what is referenced anywhere
above it. This shrinks the `Record` dicts flowing through the rest of the
tree. `CSVDataSource.read()` still parses every column of every row
regardless: true source-level pruning (skip parsing unrequested CSV
columns, or a Parquet reader that only opens requested column chunks)
needs the storage layer to accept a "requested columns" hint, which does
not exist yet.

**Statistics are exact but unused, as of Milestone 2.**
`optimizer/statistics.py`'s `compute_statistics()` does one full scan and
returns exact row counts, null counts, min/max, and distinct counts
(`distinct_count` costs memory proportional to cardinality: it is a
Python `set`, not an approximate sketch like HyperLogLog). No optimizer
rule reads a `TableStatistics` value at this point; there is no
cost-based decision to make with them yet. Built now, ahead of that
dependency, so the Dataset-scanning code and its tests exist before
anything's correctness depends on them. *(Milestone 5's `physical/
planner.py` becomes the first real consumer, calling `compute_statistics()`
directly for `order_by()`'s range-partition boundaries, not through an
optimizer rule; join strategy selection is still an explicit hint, not a
statistics-driven decision, see `logical/nodes.py`'s `Join` docstring.)*

**The analyzer checks column existence, not types.** `analyze()` walks
every `Filter` condition and `Project` column via the new
`Expression.children` property and raises `AnalysisException` if a
`Column` name is not in the relevant child schema, or if `select()`
produces two outputs with the same name. It does not infer or check
result types for arithmetic expressions: `logical/nodes.py`'s
`_output_field()` still defaults computed columns to `STRING`/nullable,
exactly as it did in Milestone 1. A real type-inference pass is future
work, not an oversight.

## Key Milestone-1 design decisions

**Row-oriented, not columnar.** `Record = dict[str, Any]`. The build spec
explicitly warns against representing everything as a pandas DataFrame; the
simplest *correct* alternative is plain Python dicts. This keeps the naive
executor a straightforward tree-walk instead of requiring batch/vectorized
evaluation machinery before there is even an optimizer. Milestone 7 adds a
columnar (Arrow-backed) representation for the physical execution path;
`Record` remains useful afterward as the row-oriented interchange format
for `collect()`/`show()`.

**Partitions hold a factory, not materialized rows.** `Partition.__iter__`
calls a zero-argument `records_fn()` each time. This buys two things ahead
of when they're needed: (1) streaming — a partition never needs to fit in
memory as a materialized list; (2) re-computability — calling the factory
again re-derives the same rows from the same source, which is the seed of
lineage-based fault tolerance (Milestone 6). The known cost, not yet paid:
sources that can't honor "call the factory twice" (e.g. a network stream)
aren't supported. Only file/in-memory sources exist so far, so this is
free for now.

**CSV reads are two-pass but bounded-memory.** `CSVDataSource.read()` scans
the file once to infer a schema (from a sample) and count rows (to compute
partition boundaries), retaining no row data. Each partition's factory
re-opens the file and streams just its row range via `itertools.islice`.
This means a file larger than RAM can be processed, at the cost of
re-seeking past earlier rows once per partition — a production system
would instead record byte offsets per partition to seek directly; skipped
here as unneeded complexity for what Milestone 1 needs to demonstrate.

**`repartition()` is not streaming.** `Dataset.repartition(n)` currently
materializes every row in memory to redistribute them round-robin, because
knowing the new partition boundaries requires having seen every row first.
Flagged explicitly in the docstring rather than silently pretending this
scales to unbounded data — a streaming-friendly version (partition by row
index without full materialization) is possible but not implemented.

**No analyzer yet, as of Milestone 1.** `df.select("does_not_exist")` did
not fail until `collect()`/`show()`/`count()` actually evaluated a
`Column` against a row and got a `KeyError`. Milestone 2's
`logical/analyzer.py` closes this gap: the same call now raises
`AnalysisException` as soon as an action runs, before any row is read.
`expressions/column.py`'s `Column.evaluate()` still raises `KeyError` as a
fallback (e.g. if a plan is executed directly without going through
`analyze()`, as the naive-executor oracle tests do on purpose), but the
`DataFrame` action path always analyzes first.

**Expression equality is overloaded.** `Expression.__eq__` returns an
`Equal` expression node (so `col("x") == 1` builds a tree) rather than a
bool, following the same DSL idiom used by PySpark and SQLAlchemy. This
forfeits the default identity-based `__hash__`, so `Expression.__hash__`
falls back to `id()` explicitly — expression trees are not intended to be
used as dict/set keys for value equality.

## What's deliberately not here yet

Per the build spec's milestone breakdown: lineage-based fault recovery,
checkpointing, columnar execution, SQL, and benchmarking. Each has a
numbered section in the build spec and lands in the milestone assigned to
it, see `README.md`'s status section for the current cut line. Within
what Milestone 5 does cover: `Join` only supports `how="inner"` with
common-named `on=` columns (no left/right/full outer, no semi/anti, no
differently-named join keys); there is no sort-merge join, only hash join
(broadcast or shuffled); broadcast-vs-shuffle join selection is an
explicit hint, never automatic; `order_by()`'s range partitioning is
equal-width over the observed min/max, not equal-row-count from a real
sample, and only exists for numeric sort keys (a string key still sorts
correctly, just through a single, non-parallel shuffle partition). The
scheduler exists and retries individual task failures, and a real shuffle
exists (now used by three different operators), but nothing recomputes a
*lost* partition via lineage (Milestone 6), nothing spills an in-progress
aggregate's hash table or sort buffer to disk under memory pressure
(Milestone 9), and shuffle output is never compressed
(`ExecutionConfig.shuffle_compression` exists but is unread).
