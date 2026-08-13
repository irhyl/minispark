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
"""

from __future__ import annotations

import sys
import time

from minispark.execution.tasks import Task, TaskContext, TaskMetrics, TaskResult, TaskState
from minispark.physical.operators import execute_partition
from minispark.physical.plan import PhysicalPlan, ScanExec, ShuffleReadExec, ShuffleWriteExec
from minispark.shuffle.partitioner import HashPartitioner
from minispark.shuffle.writer import write_shuffle_partition


def execute_task(task: Task, attempt_number: int = 0) -> TaskResult:
    _context = TaskContext(
        task_id=task.task_id,
        stage_id=task.stage_id,
        partition_id=task.partition_id,
        attempt_number=attempt_number,
    )
    start = time.perf_counter()
    try:
        if isinstance(task.plan, ShuffleWriteExec):
            result = _execute_shuffle_write_task(task)
        else:
            result = _execute_normal_task(task)
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.FAILED,
            metrics=TaskMetrics(execution_time_seconds=elapsed),
            error=f"{type(exc).__name__}: {exc}",
        )
    result.metrics.execution_time_seconds = time.perf_counter() - start
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

    def key_fn(record: dict) -> tuple:
        return tuple(expr.evaluate(record) for expr in partition_exprs)

    blocks = write_shuffle_partition(
        root_dir=task.shuffle_root_dir,
        stage_id=task.stage_id,
        source_task_id=task.task_id,
        records=rows,
        key_fn=key_fn,
        partitioner=HashPartitioner(write_exec.num_partitions),
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
    """The source partition's row count, read from cheap, already-known
    information rather than by scanning: Partition metadata for a Scan
    leaf (populated by CSVDataSource/MemoryDataSource), or the shuffle
    block metadata's record counts for a ShuffleReadExec leaf (already
    known from the write side, see shuffle/writer.py). Returns None if
    neither source knows.
    """
    leaf = _leaf_node(task.plan)
    if isinstance(leaf, ScanExec):
        return leaf.dataset.partition(task.partition_id).row_count()
    if isinstance(leaf, ShuffleReadExec) and task.shuffle_blocks is not None:
        return sum(b.record_count for b in task.shuffle_blocks)
    return None


def _shuffle_read_bytes(task: Task) -> int:
    if isinstance(_leaf_node(task.plan), ShuffleReadExec) and task.shuffle_blocks is not None:
        return sum(b.byte_length for b in task.shuffle_blocks)
    return 0


def _leaf_node(plan: PhysicalPlan) -> PhysicalPlan:
    if not plan.children:
        return plan
    return _leaf_node(plan.children[0])


def _estimate_bytes(rows: list[dict]) -> int:
    """Rough, Python-object-overhead-inclusive estimate, not an on-disk byte
    count. Same heuristic and same caveat as optimizer/statistics.py's
    `compute_statistics`."""
    return sum(sys.getsizeof(v) for row in rows for v in row.values())
