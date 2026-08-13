# MiniSpark

A first-principles, single-machine distributed data-processing engine, built to
understand, not to replace, systems like Apache Spark.

MiniSpark is a research/educational project. It is not production-ready, not a
Spark replacement, and no performance claim in this repository is made without
an accompanying, reproducible benchmark (see `docs/benchmarks.md`, added once
there is something real to measure).

## Status: Milestone 5

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
  optimizer rule).
- **Physical plan** (`minispark/physical/`): translates an optimized
  logical plan into physical operators. `Scan`/`Filter`/`Project`
  translate 1:1; `Aggregate` becomes a partial (map-side) aggregate, a
  shuffle exchange, and a final (reduce-side) aggregate; `Join` becomes a
  `HashJoinExec` fed by either two shuffle exchanges (shuffle hash join,
  the default) or one broadcast exchange (`broadcast=True`); `Sort`
  becomes a local sort, a range-partitioned exchange, and a final sort.
- **Shuffle** (`minispark/shuffle/`): `HashPartitioner` (stable across
  processes, does not use Python's randomized builtin `hash()`, used by
  `group_by`/`join`) and `RangePartitioner` (used by `order_by()`), a
  disk-backed, checksummed block writer/reader (pickled records, not
  JSON, to preserve exact types like an `Avg` aggregate's `(sum, count)`
  partial state), and a driver-side `ShuffleManager` tracking which
  blocks exist for which stage/partition, including broadcast reads.
- **Storage** (`minispark/storage/`): an in-memory `DataSource` and a CSV
  `DataSource` with schema inference and partitioned, streaming reads.
- **Lazy DataFrame API** (`minispark/api/`): `filter()` / `select()` /
  `group_by()` / `join()` / `order_by()` (alias `sort()`) build plan nodes
  only; `collect()` / `show()` / `count()` / `explain()` are the only
  things that trigger analysis, optimization, and execution.
  `explain(optimized=True)` shows the analyzed, optimized, and physical
  plans, plus every stage the query splits into.
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
  scheduler.
- **Naive executor** (`minispark/execution/executor.py`) and
  `physical/operators.py`'s whole-Dataset `execute()`: earlier
  milestones' execution paths, kept only as correctness oracles other
  tests check the real path against.

Not implemented yet (by design, not oversight): left/right/full outer or
semi/anti joins, differently-named join keys, sort-merge join,
cost-based join strategy selection, lineage-based fault recovery,
checkpointing, columnar execution, SQL, and benchmarking. Every one of
those has a numbered section in the build spec this project follows and
lands in its own milestone (or, for the join-scope items, is an explicit,
documented simplification of Milestone 5's own scope, see
`logical/nodes.py`'s `Join` docstring).

## Quick start

```bash
pip install -e ".[dev]"
pytest
python examples/basic_dataframe.py
python examples/aggregations.py
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
exists, `docs/query-planning.md` for how a DataFrame call gets from a
logical plan to a physical plan (analyzer, optimizer, physical plan),
`docs/execution-model.md` for how that physical plan actually runs (DAG,
stages, tasks, the local scheduler, and what `local[N]` really does), and
`docs/shuffle.md` for exactly what happens at a shuffle boundary
(partial aggregation, hash partitioning, the on-disk block format).

## Development

```bash
make test     # pytest
make lint     # ruff check
make format   # ruff format
```
