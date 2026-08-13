# Execution Model

How a physical plan becomes running tasks, as of Milestone 3. See
`docs/query-planning.md` for everything upstream of this (logical plan,
analyzer, optimizer, physical plan); this document picks up where that one
leaves off.

```
Physical Plan
        |
DAG                  (execution/dag.py: narrow/wide dependency classification)
        |
Stages                (execution/stages.py: split at wide-dependency boundaries)
        |
Tasks                  (execution/tasks.py: one Task per partition per stage)
        |
LocalScheduler           (execution/scheduler.py: runs tasks, retries failures)
        |
Worker                    (execution/worker.py: execute_task, runs one Task)
        |
Dataset (rows, merged back from every task's result)
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

## Narrow and wide dependencies, and why there is only one stage today

A **narrow** dependency means a child partition depends on exactly one
parent partition: `map`, `filter`, `project`. A **wide** dependency means
a child partition depends on data from every parent partition: `group by`,
`join`, `sort`. Producing a wide-dependency operator's output needs a
shuffle, moving data between partitions, which is why it forces a stage
boundary: every upstream partition must finish and be written out before
any downstream task can start.

`ScanExec`, `FilterExec`, and `ProjectExec` (the only physical nodes that
exist as of Milestone 3) are all narrow. `execution/dag.py`'s
`dependency_kind()` classifies every node that exists, and
`execution/stages.py`'s `build_stages()` genuinely checks for a wide
dependency (raising `NotImplementedError` if it ever finds one, since
splitting at a wide boundary is not implemented yet) rather than
hardcoding "one stage." The result is honest: every plan today produces
exactly one `Stage` holding the whole plan, because there is no shuffle
boundary to cut at yet. Milestone 4's `Aggregate` node is expected to be
the first wide dependency; at that point `build_stages()` needs an actual
splitting algorithm, not just a check that one is not needed yet.

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
| `input_records` | exact when Partition metadata knows the row count (CSV/memory sources always populate it), otherwise `None` |
| `output_records` | exact |
| `output_bytes` | rough heuristic (`sys.getsizeof` summed over row values), not an on-disk byte count, same caveat as `optimizer/statistics.py` |
| `input_bytes` | not implemented (would need byte-offset tracking in the storage layer) |
| `cpu_time_seconds`, `peak_memory_bytes` | not implemented (would need the `psutil` optional dependency, not added yet) |
| `shuffle_bytes` | always `0`: there is no shuffle to read from or write to until Milestone 4 |

`TaskResult` carries `state` (a `TaskState`: `PENDING`/`RUNNING`/`SUCCESS`/
`FAILED`/`RETRYING`/`CANCELLED`), the task's materialized `rows`, its
`metrics`, and `error` as a plain string when failed, not an exception
object (an exception instance is not guaranteed picklable/reconstructable
across a process boundary; a message string always is).

## Worker

`execution/worker.py`'s `execute_task(task, attempt_number)` is a plain
module-level function, not a method on a stateful `Worker` object,
specifically so it stays importable and picklable: that is exactly what
`ProcessPoolExecutor` needs to run it in a separate process. It calls
`physical/operators.py`'s `execute_partition()` (new in Milestone 3,
alongside the existing whole-Dataset `execute()`) to run the stage's
physical operators for exactly one partition, materializes the result
into a plain `list[Record]`, and returns a `TaskResult`. An exception
raised while executing becomes a `FAILED` `TaskResult` instead of
propagating: letting it propagate out of a worker process would be
indistinguishable from the process itself crashing, and the scheduler
needs to tell those two cases apart.

## LocalScheduler

`execution/scheduler.py`'s `LocalScheduler` turns a `Stage` into one
`Task` per partition, runs them, and merges the results back into a
`Dataset`. `local[N]` controls *how*:

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
locality-aware placement), no shuffle, no lineage-based fault recovery, no
checkpointing, no dynamic resource allocation. `local[N]` is real
multiprocessing on one machine; nothing here talks to another machine, and
nothing claims to.
