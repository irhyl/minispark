# MiniSpark

A first-principles, single-machine distributed data-processing engine, built to
understand, not to replace, systems like Apache Spark.

MiniSpark is a research/educational project. It is not production-ready, not a
Spark replacement, and no performance claim in this repository is made without
an accompanying, actually-run benchmark (see `docs/benchmarks.md`; every
number there was measured on a real machine, not invented, including results
that do not flatter the design).

## Status

Implemented so far:

- **Data model** (`minispark/core/`): `DataType`, `Schema`, `Field`, `Record`,
  `Partition`, `Dataset`. Partitions are lazy (row data is pulled on demand
  via a factory function), so `partition -> operator -> partition` doesn't
  require the whole dataset in memory.
- **Expressions** (`minispark/expressions/`): a real expression tree
  (`Column`, `Literal`, comparison/arithmetic/boolean operators, `IsNull` /
  `IsNotNull` / `Not`, `Alias`) built via operator overloading, e.g.
  `col("age") > 18`. Every expression exposes `children` for generic
  tree-walking (used by the analyzer and the optimizer).
- **Logical plan** (`minispark/logical/`): `Scan`, `Filter`, `Project`,
  `Aggregate`, `Join`, `Sort`, plus an `explain()` pretty-printer.
- **Analyzer** (`minispark/logical/analyzer.py`): validates every `Column`
  reference against its child schema (including inside `group_by`/`agg`,
  `join`, and `order_by`), rejects a `group_by`/`order_by` on anything but
  a plain column, a non-aggregate expression in `agg()`, an unsupported
  join type or missing/colliding join columns, and duplicate output
  names, raising `AnalysisException` with a clear message instead of a
  late `KeyError`.
- **Query optimizer** (`minispark/optimizer/`): a rule-based optimizer
  (constant folding, filter simplification, predicate pushdown -
  including through a `Join`, projection pruning, redundant projection
  elimination; every rule handles every logical node type) that runs to a
  fixed point, plus `statistics.py` (exact, full-scan table/column
  statistics, used directly by `Sort`'s physical planning, not yet by any
  optimizer rule). These rules compute plan *shape* only (never touch
  data); `physical/planner.py`'s scan-pushdown pass is what turns their
  output into an actual, smaller read (see below).
- **Physical plan** (`minispark/physical/`): translates an optimized
  logical plan into physical operators. `Scan`/`Filter`/`Project`
  translate 1:1; `Aggregate` becomes a partial (map-side) aggregate, a
  shuffle exchange, and a final (reduce-side) aggregate; `Join` becomes a
  `HashJoinExec` fed by either two shuffle exchanges (shuffle hash join,
  the default) or one broadcast exchange (`broadcast=True`); `Sort`
  becomes a local sort, a range-partitioned exchange, and a final sort.
  Before any of that translation, a scan-pushdown pass re-reads a `Scan`
  with real column/predicate hints wherever a Project/Filter chain and
  the underlying source both allow it (real for Parquet: column pruning
  and row-group-level predicate pushdown; real column pruning only for
  CSV/Memory/Checkpoint), the second, deliberate exception (after
  `Sort`'s range boundaries) to "building a plan never touches data."
- **Shuffle** (`minispark/shuffle/`): `HashPartitioner` (stable across
  processes, does not use Python's randomized builtin `hash()`, used by
  `group_by`/`join`) and `RangePartitioner` (used by `order_by()`), a
  disk-backed, checksummed block writer/reader (pickled records, not
  JSON, to preserve exact types like an `Avg` aggregate's `(sum, count)`
  partial state), and a driver-side `ShuffleManager` tracking which
  blocks exist for which stage/partition, including broadcast reads.
- **Storage** (`minispark/storage/`): an in-memory `DataSource`, a CSV
  `DataSource` with schema inference and partitioned, streaming reads, a
  checkpoint `DataSource` (reads back a `Dataset` durably materialized to
  local disk by `DataFrame.checkpoint()`), and a Parquet `DataSource`
  (real, pyarrow-backed columnar storage: genuine column pruning and
  row-group-level predicate pushdown, partitioned at row-group
  granularity, plus a writer producing one `.parquet` file per
  partition). `pyarrow` is an optional extra (`pip install
  minispark[columnar]`); nothing outside `storage/parquet.py` imports it,
  and even that module's callers (`session.read.parquet()`, `df.write.
  parquet()`) import it lazily, so using every other source never
  requires it installed.
- **Lazy DataFrame API** (`minispark/api/`): `filter()` / `select()` /
  `group_by()` / `join()` / `order_by()` (alias `sort()`) build plan nodes
  only; `collect()` / `show()` / `count()` / `explain()` are the only
  things that trigger analysis, optimization, and execution.
  `explain(optimized=True)` shows the analyzed, optimized, and physical
  plans, plus every stage the query splits into. `checkpoint()` also
  triggers execution (eagerly, like `collect()`), then returns a new
  DataFrame whose plan is a fresh scan over the durably-materialized
  result, cutting the original plan's lineage at that point. `write`
  returns a `DataFrameWriter` (`df.write.parquet(path)`), also eager.
  `last_run_metrics` exposes the most recently collected `QueryMetrics`
  after an action runs (`None` before that): per-stage task counts,
  wall-clock time, rows, bytes, shuffle bytes, retries, and, if `psutil`
  is installed, CPU time and peak memory.
- **SQL** (`minispark/sql/`): `session.sql("SELECT ...")` parses SQL text
  (hand-written tokenizer + recursive-descent parser, no grammar
  library) directly into the same logical plan nodes the DataFrame API
  builds, no separate execution path. Supports one `SELECT` statement's
  worth of grammar: `FROM` a registered temp view
  (`session.create_or_replace_temp_view(name, df)`), one inner `JOIN
  ... ON` (same-named column on both sides only, matching `Join`'s own
  scope), `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, comparisons, boolean
  connectives, arithmetic, and `COUNT`/`SUM`/`AVG`/`MIN`/`MAX`. No
  subqueries, `UNION`, window functions, `LIMIT`, CTEs, or UDFs: SQL is a
  translator into existing capability, not a way to add new capability
  without also extending the DataFrame API and logical plan.
- **DAG, stages, and tasks** (`minispark/execution/dag.py`,
  `stages.py`, `tasks.py`): classifies every physical node's dependency
  as narrow or wide (a shuffle `Exchange` is the one wide node), splits a
  physical plan into stages at wide-dependency boundaries (one stage for
  a shuffle-free plan, two or more for `group_by`/`join`/`order_by`,
  splitting each side of a `Join` independently), and represents one
  partition's worth of work as a `Task` (whose `shuffle_blocks` is keyed
  by source stage, since a join task reads from two upstream stages).
- **Local scheduler** (`minispark/execution/scheduler.py`,
  `worker.py`): `LocalScheduler` runs a plan's stages in order, turning
  each into tasks that run either sequentially (`local[1]`) or across a
  real `ProcessPoolExecutor` (`local[N>1]`, actual OS processes, not
  threads), retries individual failed tasks up to
  `engine.max_task_retries`, moves data between stages through a real
  disk-backed shuffle (including broadcast reads), and merges the last
  stage's results into a `Dataset`. `DataFrame` actions run through this
  scheduler. When a task fails because a prior stage's shuffle blocks are
  missing or corrupted (not an ordinary failure), the scheduler
  recomputes that upstream stage from scratch and retries, lineage-based
  recovery, rather than retrying a read that could never succeed on its
  own; bounded to at most one recompute per stage per query.
- **Naive executor** (`minispark/execution/executor.py`) and
  `physical/operators.py`'s whole-Dataset `execute()`: earlier
  milestones' execution paths, kept only as correctness oracles other
  tests check the real path against.
- **Spilling and memory-aware execution** (`minispark/physical/spill.py`,
  `docs/spilling.md`): `SortExec` (external merge sort) and
  `HashAggregateExec` (grace-hash partitioned spilling) both spill to
  local disk once their in-memory buffer crosses `MemoryConfig.
  spill_threshold_bytes`, instead of growing it without bound; both are
  byte-for-byte identical to their pre-Milestone-9 behavior when the
  threshold is never crossed (the default). `CSVDataSource` now records
  a byte offset per partition so each one seeks straight to its own row
  range instead of re-parsing every row before it. `benchmarks/skew.py`
  and `benchmarks/spilling.py` measure, respectively, data skew's effect
  on a reduce stage and spilling's real wall-clock cost (see
  `docs/benchmarks.md`).

Not implemented yet (by design, not oversight): left/right/full outer or
semi/anti joins, differently-named join keys, sort-merge join,
cost-based join strategy selection, and remote-worker architecture
readiness (real network communication and cloud deployment). Every one
of those has a numbered section in the build spec this project follows
and lands in its own milestone (or,
for the join-scope items, is an explicit, documented simplification of
Milestone 5's own scope, see `logical/nodes.py`'s `Join` docstring).
Within what Milestone 6 does
cover: lineage-based recomputation is stage-granular (a lost partition is
recovered by recomputing its entire upstream stage, not just the
specific tasks that produced it) and capped at one recompute per stage
per query; there is no automatic checkpoint directory lifetime
management, `checkpoint()` never deletes an old checkpoint. Within what
Milestone 7 does cover (see `docs/columnar-storage.md`): execution stays
row-at-a-time everywhere except the Parquet read path itself, there is no
vectorized Filter/Project; predicate pushdown only fires along a
Filter-directly-on-Scan chain (never past a Project) and only covers
comparisons/`And`/`Or`/`Not`/null-checks, no arithmetic; only
bool/int/float/str/null round-trip through Parquet, no date/timestamp/
decimal/nested types; row-group-to-partition assignment is contiguous
chunking, not size-aware balancing; and `write.parquet()` has no
target-file-size control or small-file coalescing. Within what
Milestone 8 does cover (see `docs/sql.md` and `docs/benchmarks.md`): SQL
has no subqueries/`UNION`/window functions/`LIMIT`/CTEs/UDFs, and a
`JOIN ... ON` clause must compare a same-named column on both sides;
`peak_memory_bytes` is a task's RSS at completion, not a continuously
sampled true peak; and the benchmark scripts in `benchmarks/` are
single-trial, uncontrolled measurements on one development machine, not
a reproducible, isolated benchmark suite. For spilling and CSV
byte-offset seeking (see `docs/spilling.md`): grace-hash aggregate
spilling bounds memory but not time per bucket, so one hash bucket
holding a
disproportionate share of distinct keys (skew) is measured
(`benchmarks/skew.py`) but not mitigated; a spill resets the whole
in-memory table, not just the excess, which is why spilling measurably
costs more for `group_by` than for `order_by` (`docs/benchmarks.md`);
and CSV byte-offset seeking does not correctly handle a quoted field
containing a literal embedded newline (misparses or raises
`ValueError`, see `storage/csv.py`).

## Quick start

```bash
pip install -e ".[dev]"
pytest
python examples/basic_dataframe.py
python examples/aggregations.py
python examples/joins.py
python examples/checkpointing.py
python examples/sql.py

# Parquet support needs the optional columnar extra:
pip install -e ".[columnar]"
python examples/parquet.py

# Benchmarks (see docs/benchmarks.md for recorded numbers and caveats):
python -m benchmarks.scaling
python -m benchmarks.join_strategy
python -m benchmarks.csv_vs_parquet   # needs the columnar extra above
python -m benchmarks.skew
python -m benchmarks.spilling
```

```python
from minispark.api.session import MiniSparkSession
from minispark.api.functions import col, count, avg

session = MiniSparkSession.builder.master("local[4]").app_name("demo").get_or_create()

df = session.read.csv("data/users.csv")

result = (
    df.filter(col("age") >= 18)
    .group_by("country")
    .agg(count("*").alias("adults"), avg("age").alias("avg_age"))
)
result.explain(optimized=True)
result.show()
```

## Architecture

See `docs/architecture.md` for the layered design and why each layer
exists, `docs/query-planning.md` for how a DataFrame call (or a SQL
query, see below) gets from a logical plan to a physical plan (analyzer,
optimizer, physical plan, including the scan-pushdown pass),
`docs/execution-model.md` for how that physical plan actually runs (DAG,
stages, tasks, the local scheduler, what `local[N]` really does,
lineage-based recomputation and checkpointing, and how query metrics are
collected), `docs/shuffle.md` for exactly what happens at a shuffle
boundary (partial aggregation, hash partitioning, the on-disk block
format shared by shuffle blocks and checkpoints), `docs/columnar-storage.md`
for Parquet reading/writing, real column pruning, and real predicate
pushdown, `docs/sql.md` for exactly what SQL is and is not supported and
why, `docs/spilling.md` for how `order_by`/`group_by` spill to local
disk under memory pressure and the CSV byte-offset read optimization,
and `docs/benchmarks.md` for real, actually-measured numbers (and their
caveats) on `local[N]` scaling, CSV vs. Parquet, broadcast vs. shuffle
joins, data skew, and spilling.

## Development

```bash
make test     # pytest
make lint     # ruff check
make format   # ruff format
```
