# MiniSpark Build Spec

This is the governing spec for the project: the architecture, the technology
constraints, the milestone breakdown, and the rules that every milestone must
follow. It is the source of truth for what gets built, in what order, and
why. Update `README.md` and `docs/architecture.md` as each milestone lands;
this file itself only changes if the plan changes.

## Objective

A first-principles, single-machine distributed data-processing engine, built
to understand, not replace, systems like Apache Spark. Not a tutorial
project or a wrapper around an existing engine.

Covers: query/dataframe APIs, logical plans, physical plans, query
optimization, DAG construction, partitioned data, task scheduling, worker
execution, shuffle, joins, aggregations, serialization, fault tolerance,
checkpointing, columnar execution, local parallelism, observability,
benchmarking.

Runs on one laptop initially (about 16 GB RAM, consumer CPU, local SSD, no
GPU, no cloud). The same architecture should extend to multiple machines
later without a rewrite. No fake distributed functionality for appearances;
every component must have a clear reason to exist and must be testable.

## Layering

```
User API
DataFrame / Dataset API
Expression System
Logical Plan
Query Optimizer
Physical Plan
DAG Builder
Stage Planner
Scheduler
Task Execution
Storage / Shuffle
```

Concerns stay separate: the scheduler does not know SQL syntax, the parser
does not know how workers execute tasks, the optimizer does not directly
execute data, the storage layer does not depend on the DataFrame API.

## Technology

Python 3.12+. Allowed: standard library, pytest, pyarrow, pandas (only for
interop/testing), numpy, duckdb (only for validation/comparison, not as the
engine), psutil, a lightweight parser if needed for SQL. Not allowed inside
the engine: Spark, Dask, Ray, Polars, Modin, or any distributed dataframe
framework as the execution engine; those may only be external reference or
benchmark systems. Future distributed communication should be designed
behind interfaces (gRPC/HTTP/multiprocessing/sockets) without introducing
that complexity during the local phase.

## Milestones

1. Dataset, Partition, DataFrame, Scan, Filter, Select, Collect.
2. Logical Plan, Analyzer, Explain, Optimizer, Physical Plan.
3. DAG, Stage, Task, Local Scheduler, multiple processes.
4. Shuffle, GroupBy, Aggregation.
5. Hash Join, Broadcast Join, Sort.
6. Task Retry, Lineage, Checkpointing.
7. Columnar Execution, Parquet, Predicate Pushdown, Projection Pruning.
8. SQL, Explain, Metrics, Profiling, Benchmarks.
9. Performance optimization, skew experiments, memory-aware execution,
   spilling.
10. Architecture readiness for remote workers, network communication, cloud
    deployment (not implemented until the local engine is stable).

Do not skip milestones. Do not implement a later milestone's subsystem early
just because it would be convenient.

## Non-negotiables

- If an implementation is simplified, document the simplification.
- If an optimization is incomplete, say so.
- If a benchmark is limited by laptop hardware, say so, and write
  "NOT RUN (hardware limitation)" rather than inventing a number.
- Never fabricate benchmark values.
- Never claim distributed execution when the system is only multiprocessing.
- Never claim fault tolerance when tasks are simply restarted without
  preserving correctness.
- Never claim scalability without measurements.
- Never claim production readiness.
- Document every deviation from the sketched package structure.

## Definition of done (local phase)

```python
session = MiniSparkSession.builder \
    .master("local[4]") \
    .app_name("analytics") \
    .get_or_create()

df = session.read.parquet("data/events")

result = (
    df
    .filter(col("age") >= 18)
    .select("user_id", "country", "revenue")
    .group_by("country")
    .agg(
        count("*").alias("users"),
        sum("revenue").alias("revenue")
    )
    .order_by(desc("revenue"))
)

result.explain()
result.show()
result.write.parquet("output/result")
```

This must construct a lazy logical plan, analyze it, optimize it, generate a
physical plan, build a DAG, divide work into stages, create tasks, execute
partitions, shuffle, aggregate correctly, retry injected failures, collect
metrics, produce deterministic results, and write the result.

## Development methodology

Build incrementally, one subsystem at a time. After each major phase: run
tests, run a small example, inspect the architecture, document it, benchmark
it, commit the work. Every major operator needs correctness tests against a
reference (pandas, DuckDB, or a manual calculation), covering nulls, empty
datasets, duplicates, skewed keys, large values, negative values, missing
values, malformed input, and schema mismatches.
