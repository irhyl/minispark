# Execution Model

How a physical plan becomes running tasks, as of Milestone 4. See
`docs/query-planning.md` for everything upstream of this (logical plan,
analyzer, optimizer, physical plan) and `docs/shuffle.md` for what
specifically happens at a shuffle boundary; this document covers the
DAG/Stage/Task/Worker/Scheduler machinery in general.

```
Physical Plan
        |
DAG                  (execution/dag.py: narrow/wide dependency classification)
        |
Stages                (execution/stages.py: split at wide-dependency boundaries)
        |
Tasks                  (execution/tasks.py: one Task per partition per stage)
        |
LocalScheduler           (execution/scheduler.py: runs stages in order, retries failures)
        |
Worker                    (execution/worker.py: execute_task, runs one Task)
        |
Dataset (rows, merged back from the last stage's task results)
```

## Dataset and Partition

`Dataset` (`core/dataset.py`) is an ordered collection of `Partition`s
sharing one `Schema`. A `Partition` (`core/partition.py`) does not hold
materialized rows; it holds a zero-argument `records_fn` that produces an
iterator on demand. As of Milestone 3, `records_fn` must be picklable, not
just callable: `MemoryDataSource`, `CSVDataSource`, and
`Dataset.repartition()` all build it with `functools.partial(iter, rows)`
or `functools.partial(some_module_level_function, ...)` rather than a
lambda or a closure, because a `local[N]` scheduler with `N > 1` sends
whole `Dataset` objects (inside a `Task`'s `PhysicalPlan`) across a real
process boundary with the standard library `pickle` module, and neither
lambdas nor closures over enclosing-function variables are picklable no
matter what they capture. See `storage/memory.py`'s `_make_records_fn`
docstring for the specific fix and why it preserves `Partition`'s public
`records_fn: Callable[[], Iterator[Record]]` contract exactly.

## Narrow and wide dependencies, and how a wide one becomes two stages

A **narrow** dependency means a child partition depends on exactly one
parent partition: `map`, `filter`, `project`, and (within one partition)
`group by`'s own grouping logic. A **wide** dependency means a child
partition depends on data from every parent partition: that is exactly
what a shuffle boundary is. Producing a wide-dependency operator's output
needs every upstream partition to finish and be written out before any
downstream task can start, which is why it forces a stage boundary.

`ScanExec`, `FilterExec`, `ProjectExec`, and `HashAggregateExec` are all
narrow: even `HashAggregateExec` only groups rows *within* the one
partition it is given (see docs/shuffle.md), it does not itself move data
between partitions. `ExchangeExec` is the one wide node, the marker the
physical planner (`physical/planner.py`) leaves at a shuffle boundary
when it translates `group_by(...).agg(...)` into a partial aggregate, an
exchange, and a final aggregate. `execution/dag.py`'s `dependency_kind()`
classifies every node that exists; `execution/stages.py`'s
`build_stages()` walks the plan and, at each `ExchangeExec`, rewrites it
into a `ShuffleWriteExec` (closing the upstream stage) and a
`ShuffleReadExec` (opening the downstream stage). A plan with no
`ExchangeExec` still produces exactly one stage, unchanged from
Milestone 3; a plan with one produces two; a plan with more than one
(e.g. two chained `group_by().agg()` calls) produces more than two, the
splitting is not special-cased to "at most one shuffle."

## Task, TaskContext, TaskResult, TaskMetrics

A `Task` (`execution/tasks.py`) is one partition of one stage: it carries
`task_id`, `stage_id`, `partition_id`, and the stage's whole `PhysicalPlan`
(shared across every task in that stage). `TaskContext` carries per-attempt
identity (`attempt_number`) into a task's execution for logging; nothing
reads from it yet beyond that (no accumulators, no broadcast-variable
access: not needed by anything that exists).

`TaskMetrics` fields and what they actually are:

| Field | Status |
|---|---|
| `execution_time_seconds` | exact (`time.perf_counter()`) |
| `input_records` | exact: from Partition metadata for a Scan leaf (CSV/memory sources always populate it), or from shuffle block record counts for a `ShuffleReadExec` leaf (already known from the write side, no extra I/O); `None` if neither source knows |
| `output_records` | exact for a normal task; `0` for a shuffle-write task (its output is blocks, not rows, see `shuffle_bytes`) |
| `output_bytes` | rough heuristic (`sys.getsizeof` summed over row values), not an on-disk byte count, same caveat as `optimizer/statistics.py`; `0` for a shuffle-write task |
| `input_bytes` | not implemented (would need byte-offset tracking in the storage layer) |
| `cpu_time_seconds`, `peak_memory_bytes` | not implemented (would need the `psutil` optional dependency, not added yet) |
| `shuffle_bytes` | `0` for a task with no shuffle input or output; for a shuffle-write task, total bytes written across all target partitions; for a shuffle-read task, total bytes read for its one target partition |

`TaskResult` carries `state` (a `TaskState`: `PENDING`/`RUNNING`/`SUCCESS`/
`FAILED`/`RETRYING`/`CANCELLED`), the task's materialized `rows`, its
`metrics`, and `error` as a plain string when failed, not an exception
object (an exception instance is not guaranteed picklable/reconstructable
across a process boundary; a message string always is).

## Worker

`execution/worker.py`'s `execute_task(task, attempt_number)` is a plain
module-level function, not a method on a stateful `Worker` object,
specifically so it stays importable and picklable: that is exactly what
`ProcessPoolExecutor` needs to run it in a separate process. For a normal
task it calls `physical/operators.py`'s `execute_partition()` to run the
stage's physical operators for exactly one partition, materializes the
result into a plain `list[Record]`, and returns a `TaskResult`. For a
task whose plan is rooted at `ShuffleWriteExec` (its stage ends at a
shuffle boundary) it takes a different path instead: compute the child's
rows, hash-partition them by the exchange's partition expressions, and
write them to shuffle storage via `shuffle/writer.py`, returning the
resulting block metadata instead of rows (see `docs/shuffle.md`). Either
way, an exception raised while executing becomes a `FAILED` `TaskResult`
instead of propagating: letting it propagate out of a worker process
would be indistinguishable from the process itself crashing, and the
scheduler needs to tell those two cases apart.

## LocalScheduler

`execution/scheduler.py`'s `LocalScheduler.run_plan()` runs a list of
`Stage`s *in order*: a stage that reads shuffle input cannot start until
the stage that wrote it has fully finished (that is what a wide
dependency means), so stages are never pipelined or run concurrently with
each other, only the tasks within one stage are. For each stage,
`run_plan()` turns it into one `Task` per partition, runs them, and
either registers the resulting shuffle block metadata into a
`ShuffleManager` (a stage ending in `ShuffleWriteExec`) or merges the
resulting rows into a `Dataset` (the last stage). `run_stage(stage)`
remains as a thin wrapper around `run_plan([stage])` for the common
single-stage case. `local[N]` controls *how* each stage's tasks run:

- `N == 1`: tasks run sequentially in this process, calling `execute_task`
  directly. No multiprocessing overhead, but the exact same `Task` ->
  `TaskResult` path as `N > 1`.
- `N > 1`: tasks run across a real `concurrent.futures.ProcessPoolExecutor`
  with `N` worker processes, real OS processes, not threads. The build
  spec is explicit that threads are the wrong tool for CPU-bound work here
  because of the GIL.

Retries happen in the scheduler's own process, not inside a worker:
because `execute_task` already turns a failure into a `FAILED`
`TaskResult` rather than raising, "should this be retried" is always a
plain decision made by inspecting a `TaskResult`, regardless of whether it
came back from a direct call or from a pool worker. Failed tasks are
retried individually (a failure on one partition does not force every
other partition to retry) up to `engine.max_task_retries` (`Config`); a
task that is still failing after that raises `TaskExecutionError` in the
scheduler's process, naming the task and the last error. There is no
lineage-based recomputation of a *lost* partition yet (Milestone 6); this
retry only re-runs a task that reported failure while the process that ran
it stayed alive.

`_run_task` is an injectable constructor argument specifically so tests
can exercise scheduling/retry/state-tracking logic with a fast, synchronous
stub instead of paying real subprocess cost for every test (see
`tests/unit/test_scheduler.py`); genuine multiprocessing still gets its
own dedicated tests that assert on things a stub cannot fake, like
observing a worker's OS process id
(`tests/integration/test_scheduler_multiprocessing.py`).

## What this is not, yet

No DAG scheduler in the Spark sense (stage retries, speculative execution,
locality-aware placement), no lineage-based fault recovery, no
checkpointing, no dynamic resource allocation, no join (so no
broadcast-join alternative to a shuffle). `local[N]` is real
multiprocessing on one machine; nothing here talks to another machine, and
nothing claims to.
