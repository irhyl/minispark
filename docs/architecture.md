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
Query Optimizer          <- not yet implemented (Milestone 2)
   |
Physical Plan            <- not yet implemented (Milestone 2)
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

**Milestone 1 status**: everything above "Query Optimizer" is implemented.
Below that line, there is no optimizer, physical plan, DAG, stage planner,
scheduler, or task abstraction yet. In their place, `DataFrame` actions
(`collect`/`show`/`count`) call `minispark.execution.executor.execute()`
directly against the *logical* plan — a naive, single-process,
tree-walking interpreter. This is a deliberate, temporary simplification:
it is the smallest thing that makes `filter().select().collect()` produce
correct results, and it gives the real optimizer/scheduler (once built) a
correctness oracle to check against. See
`minispark/execution/executor.py`'s module docstring for the exact
contract this naive executor is expected to be replaced under.

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

- **`minispark/logical/`** — `Scan`, `Filter`, `Project` plan nodes, plus
  an `explain()` pretty-printer (`plan.py`). Aggregate/Join/Sort/Limit/
  Union/Repartition/Distinct nodes are intentionally not stubbed out empty
  here; they are added in the milestone that gives each one real behavior
  (e.g. `Aggregate` alongside shuffle-based group-by in Milestone 4) so
  that "the node exists" always means "the node does something."

- **`minispark/storage/`** — `DataSource` (abstract), `MemoryDataSource`,
  `CSVDataSource`. Depends only on `core/`. A `Scan` logical node holds an
  already-`.read()` `Dataset`, not a `DataSource` reference — so the
  logical-plan layer never imports the storage layer's I/O code, only the
  data model it produces.

- **`minispark/execution/`** — `executor.py`'s `NaiveExecutor`-equivalent
  (a module-level `execute()` function, not a class — there is no state to
  hold yet). See the Milestone-1-status note above; this package's contents
  are expected to change shape substantially in Milestone 3.

- **`minispark/api/`** — `DataFrame` (lazy; `filter`/`select` build plan
  nodes, `collect`/`show`/`count`/`explain` are the only things that
  execute anything), `MiniSparkSession` (+ builder), `functions.py`
  (`col()`, `lit()`).

- **`minispark/config/`** — `Config`/`EngineConfig`/`ExecutionConfig`/
  `MemoryConfig`/`OptimizerConfig` dataclasses matching the shape in the
  build spec, and structured logging setup (`log.py`). Reality check:
  only `engine.master` is read anywhere in Milestone 1 (for display), since
  there is no scheduler/optimizer/memory-manager yet to consume the rest.
  They exist now so each subsystem introduced later has a config home from
  day one instead of a hardcoded constant that later needs to be threaded
  through.

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

**No analyzer yet.** `df.select("does_not_exist")` does not fail until
`collect()`/`show()`/`count()` actually evaluates a `Column` against a row
and gets a `KeyError`. The build spec calls for an analyzer that validates
column references before execution (Milestone 2); until then, errors
surface late and as a Python `KeyError` rather than a MiniSpark-specific
"column not found during analysis" error.

**Expression equality is overloaded.** `Expression.__eq__` returns an
`Equal` expression node (so `col("x") == 1` builds a tree) rather than a
bool, following the same DSL idiom used by PySpark and SQLAlchemy. This
forfeits the default identity-based `__hash__`, so `Expression.__hash__`
falls back to `id()` explicitly — expression trees are not intended to be
used as dict/set keys for value equality.

## What's deliberately not here yet

Per the build spec's milestone breakdown: analyzer, query optimizer (predicate
pushdown, projection pruning, constant folding, filter simplification),
physical plan, DAG/stage/task/scheduler/worker, shuffle, joins, real
aggregations, fault tolerance (retry/lineage/checkpointing), columnar
execution, SQL, and benchmarking. Each has a numbered section in the build
spec and lands in the milestone assigned to it — see `README.md`'s status
section for the current cut line.
