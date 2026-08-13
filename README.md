# MiniSpark

A first-principles, single-machine distributed data-processing engine, built to
understand, not to replace, systems like Apache Spark.

MiniSpark is a research/educational project. It is not production-ready, not a
Spark replacement, and no performance claim in this repository is made without
an accompanying, reproducible benchmark (see `docs/benchmarks.md`, added once
there is something real to measure).

## Status: Milestone 2

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
- **Logical plan** (`minispark/logical/`): `Scan`, `Filter`, `Project`, plus
  an `explain()` pretty-printer.
- **Analyzer** (`minispark/logical/analyzer.py`): validates every `Column`
  reference against its child schema and rejects duplicate `select()`
  output names before anything executes, raising `AnalysisException` with
  a clear message instead of a late `KeyError`.
- **Query optimizer** (`minispark/optimizer/`): a rule-based optimizer
  (constant folding, filter simplification, predicate pushdown, projection
  pruning, redundant projection elimination) that runs to a fixed point,
  plus `statistics.py` (exact, full-scan table/column statistics; not
  consumed by any rule yet).
- **Physical plan** (`minispark/physical/`): translates an optimized
  logical plan into physical operators (`ScanExec`/`FilterExec`/
  `ProjectExec`) and executes them. Currently a 1:1 translation, since each
  logical node has exactly one execution strategy so far.
- **Storage** (`minispark/storage/`): an in-memory `DataSource` and a CSV
  `DataSource` with schema inference and partitioned, streaming reads.
- **Lazy DataFrame API** (`minispark/api/`): `filter()` / `select()` build
  plan nodes only; `collect()` / `show()` / `count()` / `explain()` are the
  only things that trigger analysis, optimization, and execution.
  `explain(optimized=True)` shows the analyzed, optimized, and physical
  plans.
- **Naive executor** (`minispark/execution/executor.py`): Milestone 1's
  single-process, tree-walking interpreter of the logical plan. No longer
  on the `DataFrame` action path; retained as the correctness oracle
  physical execution is tested against.

Not implemented yet (by design, not oversight): DAG/stage/task scheduling,
shuffle, joins, aggregations, fault tolerance, checkpointing, columnar
execution, SQL, and benchmarking. Every one of those has a numbered section
in the build spec this project follows and lands in its own milestone.

## Quick start

```bash
pip install -e ".[dev]"
pytest
python examples/basic_dataframe.py
```

```python
from minispark.api.session import MiniSparkSession
from minispark.api.functions import col

session = MiniSparkSession.builder.master("local[4]").app_name("demo").get_or_create()

df = session.read.csv("data/users.csv")

result = df.filter(col("age") > 18).select("name", "age")
result.explain(optimized=True)
result.show()
```

## Architecture

See `docs/architecture.md` for the layered design and why each layer exists,
and `docs/query-planning.md` for how a DataFrame call currently gets from a
logical plan to rows (analyzer, optimizer, physical plan). `docs/execution-
model.md` (DAG/stage/task/worker/scheduler) is added once Milestone 3 gives
it something real to describe.

## Development

```bash
make test     # pytest
make lint     # ruff check
make format   # ruff format
```
