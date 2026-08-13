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
DAG Builder               <- not yet implemented (Milestone 3)
   |
Stage Planner              <- not yet implemented (Milestone 3)
   |
Scheduler                  <- not yet implemented (Milestone 3)
   |
Task Execution              <- not yet implemented (Milestone 3)
   |
Storage / Shuffle
```

**Milestone 2 status**: everything above "DAG Builder" is implemented.
`DataFrame` actions (`collect`/`show`/`count`/`explain`) now run, in order:
`logical/analyzer.py` (validates the plan), `optimizer/optimizer.py`
(rewrites it), `physical/planner.py` (translates it to a physical plan),
`physical/operators.py` (executes it). There is still no DAG, stage
planner, scheduler, or task abstraction; all of the above runs in one
process, one node at a time, exactly like Milestone 1 did. What changed is
*what* runs (an optimized physical plan, not the raw logical plan) and
*through how many stages* it passes before rows come out, not the fact
that it is still a single-process tree walk.

`minispark/execution/executor.py`, Milestone 1's naive, single-process,
tree-walking interpreter of the *logical* plan, is no longer on the
`DataFrame` action path. It is retained as the correctness oracle:
`tests/unit/test_physical_plan.py` and `tests/unit/test_optimizer.py`
assert that physical execution and post-optimization execution produce the
same rows the naive executor would produce on the equivalent unoptimized
plan. See `minispark/execution/executor.py`'s module docstring.

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

- **`minispark/logical/`**: `Scan`, `Filter`, `Project` plan nodes, an
  `explain()` pretty-printer (`plan.py`), and (Milestone 2) `analyzer.py`.
  Aggregate/Join/Sort/Limit/Union/Repartition/Distinct nodes are
  intentionally not stubbed out empty here; they are added in the
  milestone that gives each one real behavior (e.g. `Aggregate` alongside
  shuffle-based group-by in Milestone 4) so that "the node exists" always
  means "the node does something." `analyzer.py`'s `analyze()` validates
  every `Column` reference in a plan against its child schema before
  anything executes, raising `AnalysisException` with the offending column
  name and the available columns; it does not do type inference beyond
  "does this column exist" (see Key design decisions below).

- **`minispark/optimizer/`** (Milestone 2): `rules.py` (five rules:
  `ConstantFolding`, `FilterSimplification`, `PredicatePushdown`,
  `ProjectionPruning`, `RedundantProjectionElimination`), `optimizer.py`
  (`Optimizer`, which runs the rule list to a text-comparison fixed point),
  `statistics.py` (`compute_statistics()`, an exact full-scan statistics
  computation that nothing consumes yet). Depends on `logical/` only, never
  on `physical/` or `execution/`: an optimizer rule rewrites plan shape, it
  never touches a Dataset or a Partition.

- **`minispark/physical/`** (Milestone 2): `plan.py` (`ScanExec`,
  `FilterExec`, `ProjectExec`, one physical node per logical node type),
  `planner.py` (`plan_physical()`, a 1:1 structural translation today),
  `operators.py` (`execute()`, which walks a `PhysicalPlan` into a
  `Dataset`, and currently duplicates `execution/executor.py`'s logic
  exactly, because there is only one execution strategy per node so far).
  This package is the seam Milestone 4/5 use to make a real choice
  (HashAggregate vs SortAggregate, HashJoin vs BroadcastJoin) instead of a
  1:1 copy; it exists now, ahead of that being interesting, so the seam
  does not have to be retrofitted later.

- **`minispark/storage/`** — `DataSource` (abstract), `MemoryDataSource`,
  `CSVDataSource`. Depends only on `core/`. A `Scan` logical node holds an
  already-`.read()` `Dataset`, not a `DataSource` reference — so the
  logical-plan layer never imports the storage layer's I/O code, only the
  data model it produces.

- **`minispark/execution/`** — `executor.py`'s `NaiveExecutor`-equivalent
  (a module-level `execute()` function, not a class, there is no state to
  hold yet). As of Milestone 2 this is no longer called by `DataFrame`
  (see the Milestone-2-status note above); it is kept as the correctness
  oracle physical execution is tested against. This package's contents are
  still expected to change shape substantially in Milestone 3.

- **`minispark/api/`** — `DataFrame` (lazy; `filter`/`select` build plan
  nodes, `collect`/`show`/`count`/`explain` are the only things that
  trigger analysis/optimization/execution), `MiniSparkSession` (+
  builder), `functions.py` (`col()`, `lit()`). `explain(optimized=False)`
  (the default) prints the raw logical plan, matching Milestone 1's
  behavior exactly. `explain(optimized=True)` prints three sections in
  order: "Analyzed Logical Plan" (post-`analyze()`, pre-rewrite),
  "Optimized Logical Plan" (post-`Optimizer.optimize()`), "Physical Plan"
  (post-`plan_physical()`), so a user can see what each stage changed.

- **`minispark/config/`** — `Config`/`EngineConfig`/`ExecutionConfig`/
  `MemoryConfig`/`OptimizerConfig` dataclasses matching the shape in the
  build spec, and structured logging setup (`log.py`). As of Milestone 2,
  `OptimizerConfig.predicate_pushdown` and `OptimizerConfig.
  projection_pruning` are read by `optimizer/optimizer.py`'s
  `default_rules()` to decide whether to include those two rules; the rest
  of the config (`execution`, `memory`, and `max_task_retries`) is still
  unread, waiting on the scheduler/memory-manager that will consume it.

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
The textbook example (`Project -> Join -> Filter` becomes `Project -> Join
(one side wrapped in Filter)`) needs a Join node, which does not exist
until Milestone 4/5. Pushing a Filter below a Project, when every column
the filter needs is present on the Project's child, is the meaningful
instance of the same idea available with today's node set: rows get
dropped before the (comparatively cheap) projection work runs, instead of
after.

**Projection pruning narrows columns in the plan, not bytes read from
disk.** `ProjectionPruning` inserts a Column-only `Project` directly above
`Scan` when the Scan's schema is wider than what is referenced anywhere
above it. This shrinks the `Record` dicts flowing through the rest of the
tree. `CSVDataSource.read()` still parses every column of every row
regardless: true source-level pruning (skip parsing unrequested CSV
columns, or a Parquet reader that only opens requested column chunks)
needs the storage layer to accept a "requested columns" hint, which does
not exist yet.

**Statistics are exact but unused.** `optimizer/statistics.py`'s
`compute_statistics()` does one full scan and returns exact row counts,
null counts, min/max, and distinct counts (`distinct_count` costs memory
proportional to cardinality: it is a Python `set`, not an approximate
sketch like HyperLogLog). No rule and no rewrite currently reads a
`TableStatistics` value; there is no decision to make with them until
Milestone 5 needs to choose a join strategy by relation size. Built now,
ahead of that dependency, so the Dataset-scanning code and its tests exist
before anything's correctness depends on them.

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

Per the build spec's milestone breakdown: DAG/stage/task/scheduler/worker,
shuffle, joins, aggregations, fault tolerance (retry/lineage/checkpointing),
columnar execution, SQL, and benchmarking. Each has a numbered section in
the build spec and lands in the milestone assigned to it, see `README.md`'s
status section for the current cut line. As of Milestone 2, the analyzer
and optimizer exist but only validate/rewrite Scan/Filter/Project plans;
they have no rules for Aggregate/Join/Sort because those nodes do not
exist yet.
