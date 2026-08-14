"""Worker: executes one Task and reports a TaskResult.

`execute_task` is a plain module-level function, not a method on a
`Worker` class holding state, specifically so it stays an importable,
picklable callable: that is exactly what `ProcessPoolExecutor` needs to
run it in a separate process (see execution/scheduler.py). Nothing here
assumes the worker is in the same process as the caller; the same
function runs identically whether the scheduler calls it directly
(`local[1]`) or through a process pool (`local[N>1]`). This is the seam
the build spec asks for: "design the worker API so it can later become a
remote process."

A task's exception becomes a FAILED TaskResult, not a crash: letting an
exception propagate out of a worker process would be indistinguishable
from the process itself dying, and the scheduler needs to tell those two
cases apart (a caught exception can be retried in-place; a dead worker
process cannot). The exception is reported as a formatted string, not
re-raised as an exception object: exception instances are not guaranteed
picklable/reconstructable across a process boundary (they may hold
unpicklable state, e.g. a file handle), a message string always is.

A task whose plan is rooted at `ShuffleWriteExec` (its stage ends at a
shuffle boundary, see execution/stages.py) takes a different path: its
output is shuffle blocks written to disk, not rows returned to the
driver. `_execute_shuffle_write_task` handles that case; every other task
goes through the normal `execute_partition` -> rows path.

A `MissingShuffleDataError` (shuffle/reader.py, raised when a task tries
to read a prior stage's shuffle blocks and the block file is gone or its
checksum no longer matches) is caught separately from every other
exception, and reports which upstream stage_id was affected on the
returned `TaskResult`. That is what lets execution/scheduler.py tell a
task that needs its missing input recomputed apart from one that just
needs to be retried in place (Milestone 6's lineage-based recovery).

Profiling (Milestone 8): `cpu_time_seconds`/`peak_memory_bytes` on the
returned `TaskMetrics` are filled in here via `psutil`, if it is
installed; both stay `None` otherwise, exactly the state they were left
in since Milestone 3. This is a different optional-dependency pattern
than `storage/parquet.py`'s (pyarrow): that module is only imported when
a caller actually uses Parquet, so its import can be deferred to inside
those specific methods. `execute_task` runs unconditionally for *every*
task regardless of what the query does, so there is no method boundary
to defer behind; the import is instead attempted once at module load
time and the module-level name is `None` on failure, with every use
guarded by `if _psutil is not None`. `peak_memory_bytes` is this
process's RSS at task completion, not a true peak (`psutil` does expose
`memory_info().rss` continuously, but sampling it *during* execution
would need a background thread polling concurrently with the task, not
implemented); documented here rather than silently overstating what is
measured.
"""

from __future__ import annotations

import functools
import sys
import time
from collections.abc import Callable
from typing import Any

from minispark.core.record import Record
from minispark.execution.tasks import Task, TaskContext, TaskMetrics, TaskResult, TaskState
from minispark.expressions.base import Expression
from minispark.physical.operators import execute_partition
from minispark.physical.plan import (
    PhysicalPlan,
    ScanExec,
    ShuffleReadExec,
    ShuffleWriteExec,
    leaves,
)
from minispark.shuffle.partitioner import HashPartitioner, Partitioner, RangePartitioner
from minispark.shuffle.reader import MissingShuffleDataError
from minispark.shuffle.writer import write_shuffle_partition

try:
    import psutil as _psutil
except ImportError:
    _psutil = None


def _process_cpu_seconds() -> float | None:
    if _psutil is None:
        return None
    times = _psutil.Process().cpu_times()
    return times.user + times.system


def _process_peak_memory_bytes() -> int | None:
    if _psutil is None:
        return None
    return _psutil.Process().memory_info().rss


def _cpu_delta(cpu_start: float | None) -> float | None:
    if cpu_start is None:
        return None
    current = _process_cpu_seconds()
    return None if current is None else current - cpu_start


def execute_task(task: Task, attempt_number: int = 0) -> TaskResult:
    _context = TaskContext(
        task_id=task.task_id,
        stage_id=task.stage_id,
        partition_id=task.partition_id,
        attempt_number=attempt_number,
    )
    start = time.perf_counter()
    cpu_start = _process_cpu_seconds()
    try:
        if isinstance(task.plan, ShuffleWriteExec):
            result = _execute_shuffle_write_task(task)
        else:
            result = _execute_normal_task(task)
    except MissingShuffleDataError as exc:
        elapsed = time.perf_counter() - start
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.FAILED,
            metrics=TaskMetrics(
                execution_time_seconds=elapsed,
                cpu_time_seconds=_cpu_delta(cpu_start),
                peak_memory_bytes=_process_peak_memory_bytes(),
            ),
            error=str(exc),
            missing_shuffle_stage_id=exc.stage_id,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.FAILED,
            metrics=TaskMetrics(
                execution_time_seconds=elapsed,
                cpu_time_seconds=_cpu_delta(cpu_start),
                peak_memory_bytes=_process_peak_memory_bytes(),
            ),
            error=f"{type(exc).__name__}: {exc}",
        )
    result.metrics.execution_time_seconds = time.perf_counter() - start
    result.metrics.cpu_time_seconds = _cpu_delta(cpu_start)
    result.metrics.peak_memory_bytes = _process_peak_memory_bytes()
    return result


def _execute_normal_task(task: Task) -> TaskResult:
    partition = execute_partition(task.plan, task.partition_id, task.shuffle_blocks)
    rows = partition.to_list()
    metrics = TaskMetrics(
        input_records=_input_record_count(task),
        output_records=len(rows),
        output_bytes=_estimate_bytes(rows),
        shuffle_bytes=_shuffle_read_bytes(task),
    )
    return TaskResult(task_id=task.task_id, state=TaskState.SUCCESS, rows=rows, metrics=metrics)


def _execute_shuffle_write_task(task: Task) -> TaskResult:
    write_exec = task.plan
    assert isinstance(write_exec, ShuffleWriteExec)
    if task.shuffle_root_dir is None:
        raise ValueError(
            f"ShuffleWriteExec task {task.task_id} has no shuffle_root_dir "
            "(see execution/scheduler.py, which must set it on every task "
            "in a shuffle-write stage)"
        )
    parent = execute_partition(write_exec.child, task.partition_id, task.shuffle_blocks)
    rows = list(parent)
    partition_exprs = write_exec.partition_exprs
    partitioner: Partitioner
    key_fn: Callable[[Record], Any]

    if write_exec.range_boundaries is not None:
        # A range exchange (order_by(), see physical/planner.py) always
        # has exactly one partition expression (the primary sort key);
        # RangePartitioner.partition_for() compares a single scalar
        # against the boundary list, not a tuple.
        (sort_key_expr,) = partition_exprs
        key_fn = functools.partial(_scalar_key, sort_key_expr)
        partitioner = RangePartitioner(write_exec.num_partitions, write_exec.range_boundaries)
    else:
        key_fn = functools.partial(_tuple_key, partition_exprs)
        partitioner = HashPartitioner(write_exec.num_partitions)

    blocks = write_shuffle_partition(
        root_dir=task.shuffle_root_dir,
        stage_id=task.stage_id,
        source_task_id=task.task_id,
        records=rows,
        key_fn=key_fn,
        partitioner=partitioner,
    )
    metrics = TaskMetrics(
        input_records=_input_record_count(task),
        output_records=0,
        output_bytes=0,
        shuffle_bytes=sum(b.byte_length for b in blocks),
    )
    return TaskResult(
        task_id=task.task_id,
        state=TaskState.SUCCESS,
        rows=[],
        metrics=metrics,
        shuffle_blocks=blocks,
    )


def _input_record_count(task: Task) -> int | None:
    """The total row count across every leaf this task's plan reads from,
    from cheap, already-known information rather than by scanning:
    Partition metadata for a ScanExec leaf (populated by
    CSVDataSource/MemoryDataSource), or shuffle block metadata's record
    counts for a ShuffleReadExec leaf (already known from the write side,
    see shuffle/writer.py). A stage's plan can have more than one leaf
    (a HashJoinExec-rooted stage has two, one per side); this sums across
    all of them. Returns None only if no leaf's count is known.
    """
    counts = [_leaf_record_count(leaf, task) for leaf in leaves(task.plan)]
    known = [c for c in counts if c is not None]
    return sum(known) if known else None


def _leaf_record_count(leaf: PhysicalPlan, task: Task) -> int | None:
    if isinstance(leaf, ScanExec):
        return leaf.dataset.partition(task.partition_id).row_count()
    if isinstance(leaf, ShuffleReadExec) and task.shuffle_blocks is not None:
        blocks = task.shuffle_blocks.get(leaf.from_stage_id)
        if blocks is not None:
            return sum(b.record_count for b in blocks)
    return None


def _shuffle_read_bytes(task: Task) -> int:
    if task.shuffle_blocks is None:
        return 0
    total = 0
    for leaf in leaves(task.plan):
        if isinstance(leaf, ShuffleReadExec):
            blocks = task.shuffle_blocks.get(leaf.from_stage_id)
            if blocks is not None:
                total += sum(b.byte_length for b in blocks)
    return total


def _tuple_key(exprs: list[Expression], record: Record) -> tuple:
    return tuple(expr.evaluate(record) for expr in exprs)


def _scalar_key(expr: Expression, record: Record) -> object:
    return expr.evaluate(record)


def _estimate_bytes(rows: list[dict]) -> int:
    """Rough, Python-object-overhead-inclusive estimate, not an on-disk byte
    count. Same heuristic and same caveat as optimizer/statistics.py's
    `compute_statistics`."""
    return sum(sys.getsizeof(v) for row in rows for v in row.values())
