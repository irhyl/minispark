# Execution Model

How a physical plan becomes running tasks. See
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
iterator on demand. `records_fn` must be picklable, not
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

`ScanExec`, `FilterExec`, `ProjectExec`, `HashAggregateExec`,
`HashJoinExec`, and `SortExec` are all narrow: even `HashJoinExec` only
builds and probes a hash table *within* the one partition (pair) it is
given, and even `SortExec` only sorts the rows *within* the one partition
it is given (see docs/shuffle.md); none of them move data between
partitions on their own. `ExchangeExec` is the one wide node, the marker
the physical planner (`physical/planner.py`) leaves at a shuffle
boundary: `group_by(...).agg(...)`'s partial-aggregate/exchange/final-
aggregate shape, `join(...)`'s shuffled-or-broadcast side(s), and
`order_by(...)`'s local-sort/range-exchange/final-sort shape all produce
one or more of them. `execution/dag.py`'s `dependency_kind()` classifies
every node that exists; `execution/stages.py`'s `build_stages()` walks
the plan and, at each `ExchangeExec`, rewrites it into a
`ShuffleWriteExec` (closing the upstream stage) and a `ShuffleReadExec`
(opening the downstream stage). A `HashJoinExec` has two children, so
`build_stages()` splits each side independently; either, both, or
neither may close off its own upstream stage(s) (a shuffle hash join
closes both, a broadcast join only the broadcast side). A plan with no
`ExchangeExec` still produces exactly one stage; a plan with one
produces two (a `group_by`, or a broadcast
join); a plan with more (a shuffle hash join produces three: two writes,
one join; `order_by` produces two: one write, one final sort; a query
combining several of these produces more still), the splitting is not
special-cased to "at most one shuffle."

## Task, TaskContext, TaskResult, TaskMetrics

A `Task` (`execution/tasks.py`) is one partition of one stage: it carries
`task_id`, `stage_id`, `partition_id`, and the stage's whole `PhysicalPlan`
(shared across every task in that stage). `shuffle_blocks` is keyed by
source `stage_id` (`dict[int, list[ShuffleBlockMeta]]`), not a single
flat list: a `HashJoinExec`-rooted stage reads from two prior stages, one
per side, and each `ShuffleReadExec` leaf needs to find only its own
stage's blocks (see docs/shuffle.md's "Reading from more than one prior
stage"). `TaskContext` carries per-attempt identity (`attempt_number`)
into a task's execution for logging; nothing reads from it yet beyond
that (no accumulators, no broadcast-variable access: not needed by
anything that exists).

`TaskMetrics` fields and what they actually are:

| Field | Status |
|---|---|
| `execution_time_seconds` | exact (`time.perf_counter()`) |
| `input_records` | exact: from Partition metadata for a Scan leaf (CSV/memory sources always populate it), or from shuffle block record counts for a `ShuffleReadExec` leaf (already known from the write side, no extra I/O); `None` if neither source knows |
| `output_records` | exact for a normal task; `0` for a shuffle-write task (its output is blocks, not rows, see `shuffle_bytes`) |
| `output_bytes` | rough heuristic (`sys.getsizeof` summed over row values), not an on-disk byte count, same caveat as `optimizer/statistics.py`; `0` for a shuffle-write task |
| `input_bytes` | not implemented (would need byte-offset tracking in the storage layer) |
| `cpu_time_seconds`, `peak_memory_bytes` | Filled in by `execution/worker.py` via the optional `psutil` dependency, `None` if it is not installed; `peak_memory_bytes` is this worker process's RSS at task completion, not a true continuously-sampled peak (would need a concurrently running background thread, not implemented) |
| `shuffle_bytes` | `0` for a task with no shuffle input or output; for a shuffle-write task, total bytes written across all target partitions; for a shuffle-read task, total bytes read for its one target partition |

`TaskResult` carries `state` (a `TaskState`: `PENDING`/`RUNNING`/`SUCCESS`/
`FAILED`/`RETRYING`/`CANCELLED`), the task's materialized `rows`, its
`metrics`, and `error` as a plain string when failed, not an exception
object (an exception instance is not guaranteed picklable/reconstructable
across a process boundary; a message string always is). A failed result
also carries `missing_shuffle_stage_id: int | None`, set only when the
failure was a `MissingShuffleDataError` (see "Lineage-based
recomputation" below), which is how `LocalScheduler` tells a task whose
input needs to be regenerated apart from one that just needs an ordinary
retry.

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
  with `N` worker processes, real OS processes, not threads: threads are
  the wrong tool for CPU-bound work in Python because of the GIL.

Retries happen in the scheduler's own process, not inside a worker:
because `execute_task` already turns a failure into a `FAILED`
`TaskResult` rather than raising, "should this be retried" is always a
plain decision made by inspecting a `TaskResult`, regardless of whether it
came back from a direct call or from a pool worker. Failed tasks are
retried individually (a failure on one partition does not force every
other partition to retry) up to `engine.max_task_retries` (`Config`); a
task that is still failing after that raises `TaskExecutionError` in the
scheduler's process, naming the task and the last error. This retry only
re-runs a task that reported failure while the process that ran it
stayed alive; recomputing a partition whose *upstream* data has gone
missing entirely is a separate mechanism, lineage-based recomputation,
covered below.

`_run_task` is an injectable constructor argument specifically so tests
can exercise scheduling/retry/state-tracking logic with a fast, synchronous
stub instead of paying real subprocess cost for every test (see
`tests/unit/test_scheduler.py`); genuine multiprocessing still gets its
own dedicated tests that assert on things a stub cannot fake, like
observing a worker's OS process id
(`tests/integration/test_scheduler_multiprocessing.py`).

## Lineage-based recomputation

Ordinary task retry (above) re-runs the *same* task in place, using
the exact same shuffle blocks it was given the first time: correct for an
ordinary, possibly-transient failure, but useless if the failure is that
those blocks are no longer readable, retrying the same read just fails
the same way forever. `physical/operators.py`'s
`_execute_shuffle_read_partition` catches exactly that case (a block
file gone, `FileNotFoundError`, or corrupted, `ShuffleChecksumError`,
see `docs/shuffle.md`) and re-raises it as `MissingShuffleDataError`
(`shuffle/reader.py`), naming the upstream `stage_id` and target
partition it came from. `execution/worker.py`'s `execute_task` catches
that specific exception type separately from every other, and records
the stage_id on the returned `TaskResult.missing_shuffle_stage_id`.

`LocalScheduler._try_recover_missing_shuffle` is what actually
recomputes: it looks up the stage that produced the missing blocks (every
stage a `run_plan()` call is given is kept in a `stage_id -> Stage`
dict), re-runs every one of that stage's tasks from scratch exactly like
its first run, re-registers whatever fresh blocks that produces into the
`ShuffleManager` (overwriting the stale entry, see `docs/shuffle.md`),
and rebuilds the originally-failing task with freshly resolved
`shuffle_blocks` before handing it back to the retry loop. If that
upstream stage itself reads shuffle input that also turns out to be
missing, the same mechanism fires again first, so recovery can walk back
through more than one stage, bounded only by how many stages the plan
has. A `run_plan()` call recomputes any one stage at most once: a `set`
of already-recomputed stage_ids is threaded through the whole run, and a
stage whose data goes missing a second time is treated as an ordinary
failure (retried up to `max_retries`, then `TaskExecutionError`) instead
of being recomputed forever. A successful recovery does not consume any
of the failing task's own retry budget, since the task itself did not do
anything wrong, its input was gone.

**What this recovers, and what it does not.** This regenerates data that
was computed successfully once and then became unreadable (the scenario
proven in `tests/integration/test_lineage_recovery_e2e.py`, which deletes
a real shuffle block file mid-query under real `local[2]` multiprocessing
and confirms the query still produces the correct result). It cannot
recover from a source that was never readable in the first place (a
missing CSV file still fails the query, correctly), and it does not track
exactly which *source task* wrote a lost block, a lost target partition's
data is recovered by recomputing its *entire* upstream stage, not just
the specific source tasks that happened to contribute to that partition.
This is the same coarse-grained, stage-level recomputation granularity
Spark's shuffle fetch-failure recovery falls back to without fine-grained
map-output tracking; a real map-output tracker (recording exactly which
source task wrote which target partition, and recomputing only those
tasks) is more precise but is not needed to demonstrate the core
mechanism here. There is also no distinct "worker lost its local disk"
failure domain to simulate: every shuffle block for a query already lives
under one shared scratch directory on the one local machine (see
`docs/shuffle.md`), not on a per-worker-process local disk the way a real
multi-machine cluster's executors would have, so the realistic failure
this design simulates is a lost or corrupted block *file*, not a lost
*machine*.

## Checkpointing

`DataFrame.checkpoint()` (`api/dataframe.py`) runs the current plan now
(exactly like `collect()`) and writes the result to a durable, on-disk
checkpoint directory
(`storage/checkpoint.py`), then returns a *new* `DataFrame` whose logical
plan is a single, fresh `Scan` over that checkpoint. Everything that
built the checkpointed data, every `Filter`/`Project`/`Join`/`Aggregate`/
`Sort` in the original plan, is gone from the new plan. That is what
"truncates lineage" means in practice: if a query built on the
checkpointed `DataFrame` is later recomputed (lineage-based recovery,
above, or simply run again), it re-reads the checkpoint directory, not
the original computation, however expensive that was.
`tests/integration/test_checkpoint_e2e.py` proves this directly, not just
by inspecting plan shape: the original source is instrumented to count
how many times it is read, and that count does not increase after
`checkpoint()`, even when the checkpointed `DataFrame` is further
transformed and collected. Unlike the shuffle scratch directory (removed
at the end of every query, see `docs/shuffle.md`), nothing removes a
checkpoint directory automatically: a checkpoint is meant to outlive the
query that wrote it, so there is no safe point to delete it from without
being told to. Managing checkpoint lifetime is left to the caller; there
is no checkpoint garbage collector.

## Metrics and profiling

Per-task `TaskMetrics` exists on every `TaskResult`, but on its own it is
not aggregated across a stage or a whole query, and nothing about a task
running exposes it to a caller directly. `execution/metrics.py` adds
`StageMetrics` (one per stage
*run*, summed/maxed from that stage's tasks' `TaskMetrics`) and
`QueryMetrics` (every `StageMetrics` from one `run_plan()` call, plus
total wall-clock time). `LocalScheduler.run_plan()` builds one
`QueryMetrics` per call and stores it on `self.last_metrics`, a plain
attribute rather than a change to `run_plan()`'s existing `Dataset`-only
return type (every caller already depends on that signature).
`api/dataframe.py`'s `DataFrame._collect_dataset()` (the shared seam
behind `collect()`/`count()`/`checkpoint()`/`write.parquet()`) reads
`scheduler.last_metrics` right after calling `run_plan()` and exposes it
as `DataFrame.last_run_metrics`, `None` until an action has actually
run.

A lineage-recomputed stage (see above) gets its *own* `StageMetrics`
entry in `QueryMetrics.stages`, marked `recomputed=True`, not merged
into the stage's original entry: both runs did real, separately
measurable work, and folding them together would understate the total
cost a fault actually imposed.

Deliberately **not** threaded into `DataFrame.explain()`: `explain()`
never executes anything, and
conflating "show me the plan" with "run the query and show me what
happened" would be an unwanted change to an already-stable, highly
visible method's contract. Metrics describe what *happened*, only ever
available after an action has actually run, mirroring how real Spark's
execution metrics come from the Spark UI/status tracker after a job
runs, not from `df.explain()`.

## Spilling

`SortExec` and `HashAggregateExec` both accumulate an in-memory buffer (a
sort buffer, a hash-aggregate table) whose size would otherwise be
unbounded, correct only for a partition small enough to fit in memory.
Both carry a `spill_threshold_bytes` (sourced from `MemoryConfig.
spill_threshold_bytes`, threaded through `physical/planner.py`, see
`docs/architecture.md`'s Key spilling and CSV byte-offset design
decisions): crossing it moves part of the buffer to a local-disk spill
file (`physical/spill.py`) instead of growing further. See
`docs/spilling.md` for the full design, including a real correctness bug
(sort-tie ordering, fixed by tagging every record with a sequence number)
found by testing, and `docs/benchmarks.md`'s
"Spilling: what does it cost?" for the measured slowdown spilling
actually costs on this machine (1.83x for a spilling sort, 3.16x for a
spilling grace-hash aggregate). This is entirely inside one `Task`'s own
`execute_partition()` call, not a scheduler- or stage-level concern:
`LocalScheduler`/`execution/stages.py` above are unaware a task spilled
at all, and no new `TaskMetrics` field currently reports it (spilling is
observable today only via wall-clock time and, indirectly, via
`docs/benchmarks.md`'s dedicated benchmark, not via `QueryMetrics`).

## What this is not, yet

No DAG scheduler in the Spark sense (stage retries, speculative execution,
locality-aware placement), no fine-grained (per-source-task) map-output
tracking for lineage recovery, only coarse-grained (per-stage) recompute
(see "Lineage-based recomputation" above), no automatic checkpoint
lifetime management, no dynamic resource allocation, no cost-based join
strategy selection (broadcast is an explicit hint, see
`logical/nodes.py`'s `Join` docstring), no sort-merge join (only hash
join, broadcast or shuffled), no true continuous-sampling peak-memory
profiling (see the `TaskMetrics` table above). `local[N]` is real
multiprocessing on one machine; nothing here talks to another machine,
and nothing claims to.
