# MiniSpark

A single-machine data processing engine built from scratch: a DataFrame API, SQL, a query planner and optimizer, a multi-process execution engine, disk-backed shuffle, Parquet support, fault tolerance, and memory-aware spilling. Built to understand how engines like Apache Spark work internally, not to replace one.

[![CI](https://github.com/irhyl/minispark/actions/workflows/ci.yml/badge.svg)](https://github.com/irhyl/minispark/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A query goes through the same stages a distributed engine uses: build a logical plan (or parse one from SQL), analyze it, optimize it, turn it into a physical plan, split it into stages at shuffle boundaries, and run it as tasks across worker processes. `local[N]` means N real OS processes (`concurrent.futures.ProcessPoolExecutor`), not threads and not a simulation.

Everything above is implemented, not stubbed out, and tested against independently computed references rather than just checked for not crashing. There is no cluster, no cloud, and no GPU involved; this runs on one machine.

Not production-ready: no cost-based optimizer, no vectorized execution outside the Parquet read path, no dynamic resource allocation. Performance claims in this repo are backed by benchmarks that were actually run and reported as measured, including results that don't flatter the design (see [Benchmarks](#benchmarks)).

## Architecture

```
User API (DataFrame / SQL)
        |
Expression System            (Column, Literal, operators, Alias)
        |
Logical Plan                 (Scan, Filter, Project, Aggregate, Join, Sort)
        |
Analyzer                     (resolves columns, raises AnalysisException early)
        |
Query Optimizer               (predicate pushdown, projection pruning, constant folding, ...)
        |
Physical Plan                 (partial/final aggregates, hash/broadcast join, sort, scan pushdown)
        |
DAG Builder / Stage Planner   (splits at shuffle boundaries)
        |
Local Scheduler               (local[1] sequential, local[N>1] real ProcessPoolExecutor)
        |
Task Execution                (per-partition operators, spilling under memory pressure)
        |
Storage / Shuffle             (CSV, Parquet, checkpoints, checksummed shuffle blocks on disk)
```

The scheduler doesn't know SQL syntax, the parser doesn't know how tasks are executed, the optimizer never touches data, and the storage layer doesn't depend on the DataFrame API. Full package-by-package breakdown in `docs/architecture.md`.

## Features

- **Data model**: lazy partitions (a factory function, not materialized rows), so a query never has to hold a whole dataset in memory.
- **Expressions**: a small expression tree built with operator overloading, e.g. `col("age") > 18`.
- **Logical plan and analyzer**: `Scan`/`Filter`/`Project`/`Aggregate`/`Join`/`Sort`, with column validation that raises `AnalysisException` before execution starts instead of a late `KeyError`.
- **Optimizer**: constant folding, filter simplification, predicate pushdown (including through a join), projection pruning, redundant projection elimination, run to a fixed point.
- **Physical planning and shuffle**: an aggregate becomes partial aggregate + shuffle + final aggregate; a join becomes a shuffle hash join or a broadcast join; a sort becomes local sort + range partition + final sort. Shuffle blocks are checksummed files on local disk.
- **Scheduler**: physical plans split into stages at shuffle boundaries, one task per partition per stage. `local[1]` runs sequentially; `local[N>1]` runs across a real process pool.
- **Fault tolerance**: failed tasks retry individually; if an upstream stage's shuffle output goes missing or is corrupted, that whole stage recomputes and the query still finishes. Verified by a test that deletes a shuffle block file mid-query under real multiprocessing.
- **Checkpointing**: `df.checkpoint()` materializes a DataFrame to disk and returns a new one whose lineage stops there.
- **Parquet**: column pruning and row-group-level predicate pushdown that actually reduce what gets read, not just an optimizer pass with nothing behind it. Optional dependency (`pyarrow`); nothing else in the engine requires it.
- **SQL**: a hand-written parser that compiles into the same logical plan the DataFrame API builds, so there's no separate execution path. Supports one `SELECT` statement's worth of grammar: `FROM`, one `JOIN ... ON`, `WHERE`, `GROUP BY`, `HAVING`, `ORDER BY`, and five aggregate functions.
- **Spilling**: sort and hash-aggregate execution both spill to local disk past a configurable memory threshold instead of growing without bound, with the same result whether or not spilling triggers.
- **Metrics**: per-stage task counts, timing, rows, bytes, and retries after any action runs, plus CPU time and peak memory if `psutil` is installed.
- **Distributed readiness**: a written analysis (`docs/distributed-readiness.md`) of what would and wouldn't need to change for workers to run on separate machines. No networking code exists; `local[N]` is still the only supported mode.

## Install

```bash
pip install -e ".[dev]"

# Parquet support:
pip install -e ".[columnar]"

# CPU/memory metrics:
pip install -e ".[monitoring]"
```

Requires Python 3.12+. The core engine has no third-party runtime dependencies; `pyarrow`, `psutil`, `pandas`, and `duckdb` are optional extras, only needed for the specific feature that uses them.

## Quickstart

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
result.explain(optimized=True)  # analyzed -> optimized -> physical plan -> stages
result.show()
```

Same query in SQL, compiling into the identical plan:

```python
session.create_or_replace_temp_view("users", df)
session.sql("""
    SELECT country, COUNT(*) AS adults, AVG(age) AS avg_age
    FROM users
    WHERE age >= 18
    GROUP BY country
""").show()
```

More examples in `examples/`: `basic_dataframe.py`, `aggregations.py`, `joins.py`, `checkpointing.py`, `sql.py`, `parquet.py`.

## Testing

```bash
make test        # pytest: unit + integration, including real multiprocessing
make lint         # ruff check
make format       # ruff format
make typecheck    # mypy minispark
```

Operators are tested against independently computed references (a manual calculation or plain Python), not just against MiniSpark's own output, covering nulls, empty datasets, duplicates, and skewed keys. Fault tolerance and real multiprocessing have dedicated integration tests rather than mocks: `tests/integration/test_scheduler_multiprocessing.py`, `test_lineage_recovery_e2e.py`.

## Benchmarks

Numbers in `docs/benchmarks.md` come from actually running the scripts in `benchmarks/` on one development machine, single trial, uncontrolled.

```bash
python -m benchmarks.scaling          # local[1] vs local[N] on a shuffle-heavy query
python -m benchmarks.join_strategy    # broadcast join vs shuffle hash join
python -m benchmarks.csv_vs_parquet   # column pruning + predicate pushdown (needs [columnar])
python -m benchmarks.skew             # data skew's effect on a reduce stage
python -m benchmarks.spilling         # the wall-clock cost of spilling to disk
```

| Comparison | Result |
|---|---|
| Parquet vs. CSV (filtered, narrow projection) | Parquet ~4.4x faster |
| Broadcast join vs. shuffle hash join | Broadcast ~1.9x faster |
| `local[1]` vs. `local[16]` (small/medium data) | `local[1]` was faster, reported as measured |
| Spilling vs. staying in memory | 1.8x-3.2x slower, the cost of bounding memory |

## Documentation

| Doc | Covers |
|---|---|
| `docs/architecture.md` | Package-by-package layering and the reasoning behind each design choice |
| `docs/query-planning.md` | Logical plan, analyzer, optimizer, physical plan |
| `docs/execution-model.md` | DAG, stages, tasks, the local scheduler, lineage recovery, checkpointing, metrics |
| `docs/shuffle.md` | Partitioning, the on-disk block format, joins, sort |
| `docs/columnar-storage.md` | Parquet reading/writing, column pruning, predicate pushdown |
| `docs/sql.md` | What SQL is and isn't supported, and why |
| `docs/spilling.md` | External merge sort, grace-hash aggregate spilling, CSV byte-offset seeking |
| `docs/distributed-readiness.md` | What would and wouldn't need to change for multi-machine execution |
| `docs/benchmarks.md` | Every benchmark number and its caveats |

## License

MIT. See [LICENSE](LICENSE).
