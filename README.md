# MiniSpark

A first-principles, single-machine distributed data-processing engine, built to
understand — not to replace — systems like Apache Spark.

MiniSpark is a research/educational project. It is not production-ready, not a
Spark replacement, and no performance claim in this repository is made without
an accompanying, reproducible benchmark (see `docs/benchmarks.md`, added once
there is something real to measure).

## Status: Milestone 1

Implemented so far:

- **Data model** (`minispark/core/`): `DataType`, `Schema`, `Field`, `Record`,
  `Partition`, `Dataset`. Partitions are lazy (row data is pulled on demand
  via a factory function), so `partition -> operator -> partition` doesn't
  require the whole dataset in memory.
- **Expressions** (`minispark/expressions/`): a real expression tree
  (`Column`, `Literal`, comparison/arithmetic/boolean operators, `IsNull` /
  `IsNotNull` / `Not`) built via operator overloading, e.g. `col("age") > 18`.
- **Logical plan** (`minispark/logical/`): `Scan`, `Filter`, `Project`, plus
  an `explain()` pretty-printer.
- **Storage** (`minispark/storage/`): an in-memory `DataSource` and a CSV
  `DataSource` with schema inference and partitioned, streaming reads.
- **Lazy DataFrame API** (`minispark/api/`): `filter()` / `select()` build
  plan nodes only; `collect()` / `show()` / `count()` / `explain()` are the
  only things that execute anything.
- **Naive executor** (`minispark/execution/executor.py`): a temporary,
  single-process, tree-walking interpreter of the logical plan. There is no
  optimizer, physical plan, DAG, or scheduler yet — those are Milestones 2
  and 3. This executor exists to give later milestones a correctness
  baseline to check their (much more complex) execution paths against.

Not implemented yet (by design, not oversight): the analyzer, query
optimizer, physical plan, DAG/stage/task scheduling, shuffle, joins,
aggregations, fault tolerance, checkpointing, columnar execution, SQL, and
benchmarking. Every one of those has a numbered section in the build spec
this project follows and lands in its own milestone.

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
result.explain()
result.show()
```

## Architecture

See `docs/architecture.md` for the layered design and why each layer exists,
and `docs/execution-model.md` for how a query currently gets from a
DataFrame call to rows.

## Development

```bash
make test     # pytest
make lint     # ruff check
make format   # ruff format
```
