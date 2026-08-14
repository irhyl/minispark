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

**Milestone 8 status**: every layer above is implemented. A `DataFrame`
is built either by chaining API calls or by `session.sql(...)` parsing
SQL text into the identical logical plan (`sql/parser.py`, see
`docs/sql.md`); from that point on the two are indistinguishable.
`DataFrame` actions (`collect`/`show`/`count`/`explain`) run, in order:
`logical/analyzer.py` (validates the plan), `optimizer/optimizer.py`
(rewrites it), `physical/planner.py` (translates it to a physical plan,
including turning `group_by(...).agg(...)` into a partial aggregate, a
shuffle exchange, and a final aggregate; `join(...)` into a shuffle hash
join or a broadcast join; `order_by(...)` into a local sort, a range
exchange, and a final sort; and, before any of that translation, a
scan-pushdown pass that re-reads a `Scan` with real column/predicate
hints when the query pattern and the source both allow it, see Key
Milestone-7 design decisions below and `docs/columnar-storage.md`),
`execution/stages.py` (splits it into stages at shuffle boundaries; one
stage for a shuffle-free plan, two or more for a plan with `group_by`/
`join`/`order_by`, see `docs/execution-model.md`), `execution/
scheduler.py`'s `LocalScheduler` (runs each stage's `Task`s, either
sequentially or across a real `ProcessPoolExecutor` depending on
`local[N]`, moving data between stages through a real disk-backed
shuffle, see `docs/shuffle.md`, and building a `QueryMetrics` summary
along the way, see Key Milestone-8 design decisions below and
`docs/execution-model.md`'s "Metrics and profiling"). See
`docs/execution-model.md` for the full DAG/Stage/Task/Worker/Scheduler
picture; this file stays focused on package layout and design
decisions.

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
  `ExchangeExec` -> final `SortExec` chain; and, as of Milestone 7, a
  scan-pushdown pass that runs once, before any of the above
  translation, re-reading a `Scan` with real column/predicate hints
  wherever the plan shape and the source both allow it. Two different,
  documented exceptions to "a plan is built without touching data," see
  Key design decisions below. See `docs/shuffle.md` for the shuffle-
  based three and `docs/columnar-storage.md` for scan pushdown),
  `operators.py` (`execute()`, a whole-Dataset oracle only;
  `execute_partition()`, what a Task actually runs, with a case per
  physical node type). `ExchangeExec`/`ShuffleWriteExec`/
  `ShuffleReadExec` are the seam that let `Aggregate`/`Join`/`Sort` each
  define a wide dependency without duplicating the stage-splitting or
  shuffle machinery; a future Sort-Merge join would extend the same seam
  rather than needing a new one.

- **`minispark/shuffle/`**: `partitioner.py` (`HashPartitioner`, used by
  `group_by`/`join`; `RangePartitioner`, used by `order_by()` as of
  Milestone 5), `writer.py`/`reader.py` (disk-backed, checksummed shuffle
  blocks), `manager.py` (`ShuffleManager`, driver-side bookkeeping of
  which blocks exist for which stage/partition). See `docs/shuffle.md`
  for the full picture, including how a broadcast join and a range-
  partitioned sort each reuse this same machinery. Depends on `core/`
  only.

- **`minispark/storage/`** — `DataSource` (abstract), `MemoryDataSource`,
  `CSVDataSource`, `CheckpointDataSource`/`write_checkpoint()` (Milestone
  6, reads back a `Dataset` durably materialized to local disk by
  `DataFrame.checkpoint()`, reusing the same pickled-record block format
  as a shuffle block, see `docs/shuffle.md`), and, as of Milestone 7,
  `ParquetDataSource`/`write_parquet_dataset()` (real, pyarrow-backed
  columnar storage; see `docs/columnar-storage.md`). `DataSource.read()`
  gained two optional pushdown hints this milestone, `columns` and
  `filter` (an `Expression`), which is a genuine, deliberate widening of
  this package's dependencies: it now depends on `expressions/` as well
  as `core/` (a documented deviation from the sketched package
  structure, not an oversight, see the build spec's "document every
  deviation" rule). Only `storage/parquet.py` additionally imports
  `pyarrow`, and only inside methods that are called lazily (never at
  module import time), since `pyarrow` is an optional extra (`pip
  install minispark[columnar]`, see pyproject.toml) and nothing outside
  that one file may require it to be installed. A `Scan` logical node
  holds an already-`.read()` `Dataset`, not a `DataSource` reference, so
  the logical-plan layer still never imports the storage layer's I/O
  code (see logical/nodes.py's `ScanSource` Protocol, Key Milestone-7
  design decisions below).

- **`minispark/execution/`**: `executor.py` (Milestone 1's naive
  logical-plan interpreter, kept only as a correctness oracle), `dag.py`,
  `stages.py`, `tasks.py`, `worker.py`, `scheduler.py`, and, as of
  Milestone 8, `metrics.py` (`StageMetrics`/`QueryMetrics`, aggregated
  from `TaskMetrics` across a whole `run_plan()` call). See
  `docs/execution-model.md` for what each of these does and how they fit
  together; that document, not this one, is the place to look for the
  full DAG/Stage/Task/Worker/Scheduler picture.

- **`minispark/sql/`** (Milestone 8, a SQL front-end, not a second
  execution engine). `tokenizer.py` (hand-written lexer) and `parser.py`
  (`parse_sql()`, a hand-written recursive-descent parser with
  precedence climbing for expressions) translate SQL text directly into
  the same `logical/nodes.py` nodes and `expressions/` trees the
  DataFrame API builds. Depends on `logical/` and `expressions/` only,
  never on `api/`: `api/session.py`'s `MiniSparkSession.sql()` is what
  resolves table names (against its own `_temp_views` registry) and
  wraps the resulting `LogicalPlan` back into a `DataFrame`, keeping
  `parse_sql()` itself testable with a plain `dict[str, LogicalPlan]`,
  no session required. See `docs/sql.md` for the supported grammar and
  every scope decision behind it.

- **`minispark/api/`** — `DataFrame` (lazy; `filter`/`select`/`group_by`/
  `join`/`order_by` (alias `sort`) build plan nodes, `collect`/`show`/
  `count`/`explain` are the only things that trigger analysis/
  optimization/execution; `checkpoint()`, as of Milestone 6, also
  triggers execution, eagerly, and returns a new `DataFrame` whose plan
  is a fresh `Scan` over the durably-materialized result, see Key
  Milestone-6 design decisions below; `write` (Milestone 7) returns a
  `DataFrameWriter`, `df.write.parquet(path)`, also eager, writing one
  `.parquet` file per partition, see `docs/columnar-storage.md`;
  `last_run_metrics`, Milestone 8, exposes the most recently collected
  `QueryMetrics`, `None` until an action has run, see Key Milestone-8
  design decisions below), `grouped.py` (`GroupedData`, the result of
  `group_by()` before `.agg()` turns it back into a `DataFrame`),
  `writer.py` (`DataFrameWriter`, the write-side mirror of `session.py`'s
  `DataFrameReader`), `MiniSparkSession` (+ builder; `sql()` and
  `create_or_replace_temp_view()`, Milestone 8, are the SQL entry point,
  see `minispark/sql/` above), `functions.py` (`col()`, `lit()`,
  `count()`, `sum()`, `avg()`, `min()`, `max()`).
  `DataFrameReader.parquet()`/`DataFrameWriter.parquet()` both import
  `storage/parquet.py` *inside* the method body, not at module top,
  since `pyarrow` is an optional extra and `import minispark.api.
  session` alone (to call `.csv()`, say) must never require it.
  `DataFrame.join()`
  intentionally only accepts `on=` (common column names on both sides),
  matching `Join`'s own scope (see `logical/`, above). `explain(
  optimized=False)` (the default) prints the raw logical plan, matching
  Milestone 1's behavior exactly. `explain(optimized=True)` prints
  "Analyzed Logical Plan" (post-`analyze()`, pre-rewrite), "Optimized
  Logical Plan" (post-`Optimizer.optimize()`), "Physical Plan" (post-
  `plan_physical()`), and "Stages" (post-`build_stages()`, one section
  per stage, however many that turns out to be), so a user can see what
  each step changed. `explain()` never executes anything, before or
  after Milestone 8; `last_run_metrics` is the separate mechanism for
  what a query actually did, not what it would do.

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

## Key Milestone-8 design decisions

**SQL is a translator into the existing logical plan, never a second
interpreter.** The build spec is explicit: "there must not be a
separate SQL execution engine." `sql/parser.py`'s `parse_sql()` builds
`logical/nodes.py` nodes directly, the same `Scan`/`Filter`/`Project`/
`Aggregate`/`Join`/`Sort` the DataFrame API builds; `MiniSparkSession.
sql()` hands the result to a plain `DataFrame`, which runs through the
exact same analyze/optimize/physical-plan/stage/schedule path any other
`DataFrame` does. Checked directly, not just asserted: `tests/
integration/test_sql_e2e.py` compares `explain_string()` output between
a SQL-built and an API-built `DataFrame` for the equivalent query and
requires them to be textually identical, both before and after the
scan-pushdown pass, not merely "produces the same rows."

**A hand-written tokenizer and recursive-descent parser, not a grammar
library.** The build spec's allowed-dependencies line permits "a
lightweight parser if needed for SQL." The supported grammar (`sql/
parser.py`'s module docstring: `SELECT`/`FROM`/one `JOIN`/`WHERE`/
`GROUP BY`/`HAVING`/`ORDER BY`, comparisons, boolean connectives,
arithmetic, five aggregate functions) is small and fixed enough that a
generated parser or a third-party grammar would be more machinery than
the problem needs, the same reasoning `optimizer/rules.py` gives for not
having a generic tree-visitor abstraction over six logical node types.

**SQL support is scoped to mirror the DataFrame API exactly, not to
add capability.** `Join`'s `on=` only supports a column with the same
name on both sides (see `logical/nodes.py`'s `Join` docstring); SQL's
`JOIN ... ON a = b` enforces the same restriction at parse time
(`SqlParseError`, not a confusing failure three layers downstream) via a
plain string-equality check on the two column names, once the query's
own qualifiers (`table.column`) have been stripped, matching-name
required. Grouping/aggregation similarly only supports what `Aggregate`
already supports: no `LIMIT`, no window functions, no subqueries, no
`UNION`. Adding SQL syntax for any of these without first adding the
underlying `LogicalPlan`/execution support would be exactly the
"separate execution engine" the build spec forbids, just spelled as new
grammar instead of a new interpreter.

**`HAVING`'s aggregate function calls resolve to `Column` references by
structural match, not by re-embedding the raw aggregate expression.** A
first version of `HAVING COUNT(*) >= 1` built `GreaterEqual(Count(None),
Literal(1))` directly, the literal parse of the clause, and it crashed:
`AggregateFunction.evaluate()` (expressions/aggregate.py) raises
`NotImplementedError` on purpose (an aggregate has no per-row value
until `HashAggregateExec` finalizes it), and `HAVING` runs as a plain
row-level `Filter` *after* the `Aggregate`, where the row already holds
the finalized value under its output alias, not the raw
`AggregateFunction` object. Caught immediately by actually running the
query, not just by unit-testing the parser's output shape. The fix,
`_substitute_aggregates_with_output_columns()`, rebuilds the `HAVING`
expression tree, replacing any `AggregateFunction` node with a `Column`
reference to the matching `SELECT`-list aggregate's output name,
matched by `repr()` equality (`Expression.__eq__` is overloaded to
build an `Equal` node, not compare for equality, the same reason
`Optimizer.optimize()`'s fixed-point check compares `explain_string()`
text instead of `==`). An aggregate referenced in `HAVING` but absent
from `SELECT` raises `SqlParseError` rather than silently adding a
second, hidden aggregate the way some SQL engines allow.

**Metrics are a plain scheduler attribute, not a change to `run_plan()`'s
return type.** `LocalScheduler.run_plan()` has returned a `Dataset` and
nothing else since Milestone 3; every caller depends on that. Rather
than widen it to a tuple (and touch every call site), `run_plan()` now
also sets `self.last_metrics: QueryMetrics`, read immediately afterward
by `api/dataframe.py`'s `DataFrame._collect_dataset()` and exposed as
`DataFrame.last_run_metrics`. Deliberately not threaded into `explain()`:
`explain()` has never executed anything, and folding "what happened"
into "what would happen" would change an already-stable, widely used
method's contract. A lineage-recomputed stage gets its own, second
`StageMetrics` entry (`recomputed=True`), not merged into the first:
both runs did real, separately measurable work, and merging them would
understate what a fault actually cost.

**Profiling reuses the pyarrow-style optional-dependency pattern, with
one necessary difference.** `psutil` (`cpu_time_seconds`/
`peak_memory_bytes` on `TaskMetrics`, filled in by `execution/
worker.py`) is optional like `pyarrow`, but the import cannot be
deferred to inside a method the way `storage/parquet.py`'s callers defer
theirs: `execute_task` runs unconditionally for *every* task, Parquet or
not, so there is no feature-specific method boundary to hide the import
behind. Instead, `worker.py` attempts `import psutil` once at module
load time inside a `try`/`except ImportError`, leaving the module-level
name `None` on failure, with every use guarded by `if _psutil is not
None`; both fields simply stay `None`, exactly their pre-Milestone-8
state, when `psutil` is not installed. `peak_memory_bytes` is
documented as this process's RSS at task completion, not a true,
continuously sampled peak (which would need a background thread polling
concurrently with the task, not implemented), rather than silently
overstating precision the measurement does not have.

**Benchmarks report what was actually measured, including results that
do not flatter the design.** `benchmarks/scaling.py`'s `local[1]` vs
`local[N]` comparison found `local[1]` faster at every tested size on
this development machine (`docs/benchmarks.md`), the opposite of what a
"more workers is faster" story would predict. Reported anyway, with the
`ProcessPoolExecutor`-on-Windows (`spawn`, not `fork`) and per-task
disk-backed-shuffle overhead named as the most likely explanation,
because the build spec's non-negotiables include "never claim
scalability without measurements," and a measurement that contradicts
the design's intent is still a measurement, not a bug to be quietly
tuned away in the writeup.

## Key Milestone-7 design decisions

**"Columnar execution" was scoped to the storage layer, not the whole
engine.** Milestone 7's build-spec bullet groups four things: columnar
execution, Parquet, predicate pushdown, projection pruning. The scope
decision made explicitly before implementing (see the conversation this
milestone was planned in) was: pyarrow reads Parquet with genuine column
pruning and genuine row-group-level predicate pushdown (real bytes not
read), and every physical operator downstream of a Scan, Filter,
Project, HashAggregateExec, HashJoinExec, SortExec, stays exactly the
row-at-a-time Python engine that exists today. Fully vectorizing Filter/
Project to operate on Arrow `RecordBatch`es end to end (falling back to
rows only at a row-based operator's boundary) was the alternative
considered and explicitly deferred: it would deliver a more complete
"columnar execution" claim, but touches `Partition`'s core contract,
`execute_partition()`, and a large share of existing physical-operator
tests, a substantially larger and riskier change than one milestone's
scope justifies here. `Record = dict[str, Any]` remains what every
physical operator sees; a Parquet-backed partition's `records_fn`
decodes straight to `Record` dicts (`RecordBatch.to_pylist()`) at the
partition boundary, and nothing past that point knows or cares the
source was columnar.

**Real pushdown needed `Scan` to hold onto its `DataSource`, which
needed a `Protocol`, not an import.** Before this milestone, `Scan`
(logical/nodes.py) held only an already-`.read()` `Dataset`: the read
happened once, at DataFrame-construction time (`session.read.csv(path)`
calls `.read()` immediately), long before the optimizer computes which
columns/predicates could actually be pushed. Making pushdown real (not
just plan-shape pruning, Milestone 2's version) requires re-reading with
those hints once they're known, at physical-planning time, which means
`Scan` has to keep a handle on the `DataSource` that produced it. But
`logical/` deliberately never imports `storage/` (see the Milestone-1-era
package layout note, still true). The fix: `logical/nodes.py` defines a
structural `ScanSource` `Protocol` (`read(columns=, filter=) -> Dataset`)
that `storage.datasource.DataSource` satisfies purely by having a
matching method, no inheritance, no registration, no import from
`logical/` to `storage/` needed at all. `Scan.source: ScanSource | None`
defaults to `None`, so every hand-built `Scan(dataset, name)` already in
the test suite keeps working unchanged; pushdown simply does not apply
to a `Scan` with no `source`.

**The scan-pushdown pass lives in `physical/planner.py`, runs once per
plan, not once per recursive call.** Like `Sort`'s range-boundary
computation (Milestone 5), re-reading a `Scan` with pushdown hints
touches real I/O, so it cannot live in `optimizer/rules.py`, whose rules
are held to "never touch data." The first implementation called the
pushdown pass at the top of the same function `plan_physical()` uses to
recurse (`_translate()` after this fix); that turned out to call
`DataSource.read()` a second, redundant time for any Filter/Scan chain
sitting directly under an `Aggregate`/`Join`/`Sort` (which each make
their own recursive `plan_physical(child, ...)`-style call on a
subtree), correct, since re-reading with the same hints is idempotent,
but wasteful, genuinely reading Parquet twice, not just re-walking a
tree. Caught before shipping, not after: `plan_physical()` is now a thin
public entry point that runs the pushdown pass exactly once, over the
whole plan, then delegates to a separate recursive `_translate()` that
every internal call site (`_plan_join`, `_plan_aggregate`, `_plan_sort`,
and the Filter/Project branches) uses instead of calling back into
`plan_physical()`.

**Pushdown only ever narrows what a Scan reads relative to the query's
true meaning, and the Filter/Project physical nodes always stay in the
plan regardless.** Two safety rules, both load-bearing: (1) `columns`
sent to a source is always at least the union of every Project's and
every Filter's referenced columns in the chain reaching that Scan
(`_reread_scan()` unions them defensively even though the recursive walk
already guarantees it by construction, belt-and-suspenders on purpose);
a source is free to ignore `columns`/`filter` entirely and still be
correct, since the row-level `FilterExec`/`ProjectExec` above it are
never elided, pushdown is always an optimization underneath them, never
a substitute for their own correctness (see storage/datasource.py's
`DataSource.read()` docstring). (2) A `Filter`'s condition is only
carried down through more `Filter`s, never through a `Project`, even a
plain-column one: a `Filter`'s condition is an `Expression` tied to
whichever namespace was valid where it was written, and a `Project` is
exactly a namespace boundary (its output names need not match its
input's). Carrying a filter expression past one, unchanged, risks
evaluating it against columns that mean something different underneath,
or do not exist at all. `columns`, by contrast, is recomputed fresh at
every `Project` (via `referenced_columns()`, which already walks into
`Alias`/computed sub-expressions correctly), so it stays correct across
renames at every level; only `filter_expr` resets to `None` at a
`Project` boundary. In practice this means predicate pushdown to Parquet
only fires along a `Filter`-directly-on-`Scan` chain, which is exactly
what `PredicatePushdown`'s own logical-level rule (Milestone 2) already
arranges whenever pushing a filter below a `Project` is namespace-safe;
a `Filter` that rule could not push that far is, correctly, one this
pass does not push to storage either.

**Predicate translation only ever narrows, via two different
correctness rules for `And` vs `Or`, and refuses to translate a
None-literal comparison at all.** `storage/parquet.py`'s
`translate_predicate()` may push just one side of an `And` (a safe
superset: the untranslated side still gets checked by the row-level
Filter that always remains), but must translate *both* sides of an `Or`
or neither, pushing only one side of an "or" could wrongly exclude rows
the untranslated side would have kept. A subtler bug, caught by testing
before it shipped: pyarrow's comparison operators implement SQL's
three-valued NULL logic (`x == null` never matches, even when `x` is
itself null), but MiniSpark's row engine evaluates `==`/`!=` as plain
Python equality, where `None == None` is `True`. Translating
`col("x") == None` the way pyarrow would evaluate it could therefore
wrongly *exclude* rows the row-level Filter would have kept, an
over-exclusion, not merely a missed optimization, the one pushdown
mistake this design otherwise guards against everywhere else. The fix:
a `None`-valued `Literal` is treated as untranslatable, not translated
to a pyarrow null scalar, so any comparison involving it is left
entirely to the row-level Filter. A related, accepted (not fixed)
inconsistency: a comparison against a genuinely null *column* value
(not a `None` literal) makes pyarrow exclude that row silently, while
the row-based engine, reached without pushdown, would raise `TypeError`
on the same comparison (`None > 18`, say), a pre-existing limitation of
row-at-a-time evaluation, not something this milestone introduces or
attempts to fix; documented in `docs/columnar-storage.md` rather than
hidden.

**Parquet is partitioned at row-group granularity, not by row range.**
Unlike CSV (which computes contiguous row ranges from a row count known
only after a full-file scan), a Parquet file already has a natural,
independently-readable physical unit: the row group. `ParquetDataSource.
read()` enumerates every row group across the dataset (via `pyarrow.
dataset`'s fragment API, `split_by_row_group()`) and assigns them to
`num_partitions` buckets; each partition's `records_fn` reads only its
own assigned row groups. This is what makes row-group-level predicate
skipping observable per-partition, not just in aggregate: a partition
whose row groups' statistics all fail the pushed filter reads zero rows,
directly, not merely "fewer" (see `tests/unit/test_parquet_source.py`'s
row-group-skip test). `ParquetFileFragment` and `pyarrow.dataset.
Expression` objects are carried directly in the `records_fn` closure
(not reconstructed from a `(path, row_group_index)` pair inside the
worker): both were confirmed picklable, including across a real
`ProcessPoolExecutor` round trip, before relying on that rather than
assuming it.

## Key Milestone-6 design decisions

**Retry and lineage-based recomputation are different mechanisms for
different failure classes, not one generalized "recover from failure"
system.** Milestone 3's task retry re-runs a task in place with the same
inputs; that is correct for a transient failure but cannot help when the
input itself is gone. Milestone 6 adds a second, narrower mechanism only
for that second case: `physical/operators.py` distinguishes "a shuffle
block is unreadable" (`MissingShuffleDataError`, wrapping a
`FileNotFoundError` or a `ShuffleChecksumError`) from every other
exception, threads that distinction through `TaskResult.
missing_shuffle_stage_id`, and `execution/scheduler.py`'s
`_try_recover_missing_shuffle` recomputes only the specific upstream
stage that produced the missing data before retrying, rather than
either (a) blindly retrying the same doomed read or (b) recomputing the
entire query from Scan. Keeping these as two separate, composable
mechanisms (ordinary retry still runs first/underneath; recomputation
only fires for this one specific, identifiable failure signature) was
chosen over one unified "just retry harder" loop because the two
failures need genuinely different responses, and conflating them would
mean either wasting work retrying an unrecoverable read forever, or
recomputing a whole upstream stage for a merely transient hiccup that a
plain retry would have fixed for free.

**Recomputation is stage-granular, matching what this architecture can
actually know, not the finest grain possible in principle.** A lost
target partition could in principle be recovered by re-running only the
specific source tasks that wrote to it (a real per-source-task
map-output tracker, closer to how a mature Spark deployment behaves).
`ShuffleManager` here only ever records "these blocks exist for this
stage," not "this specific source task, of this specific stage, wrote
to this specific target partition, and no other source task did," so
the information needed for finer-grained recovery is not tracked. Rather
than adding that tracking now, `_try_recover_missing_shuffle` recomputes
the *entire* upstream stage (every task, every target partition) and
re-registers all of it, a real, accepted cost (recomputing partitions
that were not actually lost) in exchange for not needing a second
bookkeeping structure alongside `ShuffleManager`'s existing one. Bounded
to at most one recompute per stage per `run_plan()` call (a `set` of
already-recomputed stage_ids threaded through the run) specifically so a
stage that is not actually recoverable (a permanently broken source, not
a one-off lost block) fails cleanly with `TaskExecutionError` instead of
the scheduler looping on it.

**There is no distinct "lost worker" failure domain to simulate, so this
milestone simulates a lost *block* instead.** In a real multi-machine
cluster, the motivating scenario for lineage-based recovery is an
executor process (and the shuffle data cached on its local disk) dying
outright. On one machine, every shuffle block for a query already lives
under one shared scratch directory (`ShuffleManager.root_dir`, a single
`tempfile.mkdtemp()`), not on a per-worker local disk the way a real
cluster's executors would have; there is no separate failure domain
"losing one worker" could correspond to here that "losing one file"
does not already cover. `tests/integration/test_lineage_recovery_e2e.py`
therefore simulates the failure the architecture can actually produce: a
real shuffle block file deleted from real disk mid-query, under real
`local[2]` multiprocessing, confirmed recoverable by the scheduler
alone. This is flagged explicitly rather than claiming this proves
executor-loss recovery in the distributed sense, it proves recovery from
lost *data*, which is the part of the mechanism that generalizes; true
multi-machine executor loss is Milestone 10 territory (remote worker
architecture readiness).

**Checkpointing reuses the shuffle block format instead of inventing a
second on-disk record format.** `storage/checkpoint.py` writes one file
per partition as back-to-back pickled Records, exactly `shuffle/
writer.py`'s block format minus the checksum and `ShuffleBlockMeta`
(a checkpoint is not registered with a `ShuffleManager` and is not
verified against a checksum on read, since, unlike a shuffle block, it
is not expected to be deleted out from under a running query). The
alternative, a columnar/Parquet-backed checkpoint format, was set aside
for Milestone 7, which introduces columnar execution generally; adding
it only for checkpoints now would mean solving that problem twice.

**`DataFrame.checkpoint()` returns a new `DataFrame`, it does not mutate
the one it was called on.** This matches how every other `DataFrame`
method (`filter`, `select`, `join`, ...) behaves, immutable, plan-
building, never touching `self`, rather than making `checkpoint()` a
special exception that mutates a DataFrame's own plan in place. The
returned `DataFrame`'s plan is a bare `Scan` over a fresh
`CheckpointDataSource`; nothing about the original `DataFrame` (still
usable, still describing the original, uncheckpointed plan) changes.

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

Per the build spec's milestone breakdown: performance optimization/
skew/spilling (Milestone 9) and remote-worker architecture readiness
(Milestone 10). Each has a numbered section in the build spec and lands
in the milestone assigned to it, see `README.md`'s status section for
the current cut line. Within what Milestone 5 does cover: `Join` only supports `how="inner"` with
common-named `on=` columns (no left/right/full outer, no semi/anti, no
differently-named join keys); there is no sort-merge join, only hash
join (broadcast or shuffled); broadcast-vs-shuffle join selection is an
explicit hint, never automatic; `order_by()`'s range partitioning is
equal-width over the observed min/max, not equal-row-count from a real
sample, and only exists for numeric sort keys (a string key still sorts
correctly, just through a single, non-parallel shuffle partition).
Within what Milestone 6 does cover (see Key Milestone-6 design
decisions, below, and `docs/execution-model.md`'s "Lineage-based
recomputation" and "Checkpointing" sections for the full picture):
lineage-based recomputation is stage-granular, not per-source-task, a
lost target partition's data is recovered by recomputing its entire
upstream stage, not just the specific tasks that wrote to it; a stage is
recomputed at most once per query, a source that is genuinely,
permanently unreadable still fails the query; and nothing manages
checkpoint directory lifetime automatically, `DataFrame.checkpoint()`
never deletes an old checkpoint, that is left to the caller. Within what
Milestone 7 does cover (see Key Milestone-7 design decisions, above, and
`docs/columnar-storage.md`): execution stays row-at-a-time everywhere
except the Parquet read path itself, there is no vectorized Filter/
Project operating on Arrow batches; predicate pushdown only fires along
a Filter-directly-on-Scan chain, never past a Project (a rename/computed
boundary); pushdown only covers comparisons, `And`/`Or`/`Not`, and
`IsNull`/`IsNotNull`, arithmetic inside a predicate (`(a + b) > 5`) is
never pushed; only `bool`/`int`/`float`/`str`/`null` types round-trip
through Parquet, matching `core/types.py`'s existing closed set, no
date/timestamp/decimal/nested/list/struct support; row-group-to-
partition assignment is contiguous chunking, not size- or row-count-
aware balancing, so a Parquet file with very unevenly sized row groups
can still produce skewed partitions; and `write.parquet()` always writes
one file per partition with no target-file-size control or coalescing.
Within what Milestone 8 does cover (see Key Milestone-8 design
decisions, above, `docs/sql.md`, and `docs/execution-model.md`'s
"Metrics and profiling"): SQL supports one `SELECT` statement's worth of
grammar (`FROM`/one inner `JOIN`/`WHERE`/`GROUP BY`/`HAVING`/`ORDER BY`,
comparisons, boolean connectives, arithmetic, five aggregate functions),
no subqueries, `UNION`, window functions, `LIMIT`, CTEs, or UDFs, and a
`JOIN ... ON` clause must compare a same-named column on both sides,
exactly `Join`'s own restriction; `peak_memory_bytes` is a task's RSS at
completion, not a continuously sampled true peak; `QueryMetrics` is
never available before an action runs (by design, see "Metrics are a
plain scheduler attribute" above); and the benchmark scripts in
`benchmarks/` are single-trial, uncontrolled measurements on one
development machine, not a reproducible, isolated benchmark suite (see
`docs/benchmarks.md`'s own stated caveat). Nothing spills an in-progress
aggregate's hash table or sort buffer to disk under memory pressure
(Milestone 9), and shuffle output is never compressed
(`ExecutionConfig.shuffle_compression` exists but is unread).
