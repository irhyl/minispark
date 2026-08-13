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
"""

from __future__ import annotations

import sys
import time

from minispark.execution.tasks import Task, TaskContext, TaskMetrics, TaskResult, TaskState
from minispark.physical.operators import execute_partition
from minispark.physical.plan import PhysicalPlan, ScanExec


def execute_task(task: Task, attempt_number: int = 0) -> TaskResult:
    _context = TaskContext(
        task_id=task.task_id,
        stage_id=task.stage_id,
        partition_id=task.partition_id,
        attempt_number=attempt_number,
    )
    start = time.perf_counter()
    try:
        partition = execute_partition(task.plan, task.partition_id)
        rows = partition.to_list()
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.FAILED,
            metrics=TaskMetrics(execution_time_seconds=elapsed),
            error=f"{type(exc).__name__}: {exc}",
        )
    elapsed = time.perf_counter() - start
    metrics = TaskMetrics(
        execution_time_seconds=elapsed,
        input_records=_input_record_count(task.plan, task.partition_id),
        output_records=len(rows),
        output_bytes=_estimate_bytes(rows),
    )
    return TaskResult(task_id=task.task_id, state=TaskState.SUCCESS, rows=rows, metrics=metrics)


def _input_record_count(plan: PhysicalPlan, partition_id: int) -> int | None:
    """The source partition's row count, from Partition metadata if known.

    Read from metadata (already populated by CSVDataSource/MemoryDataSource)
    rather than by scanning: the task is about to read this partition's
    rows anyway, scanning it twice just to count would double CSV I/O for
    no benefit. Returns None if the metadata does not know (metadata is
    optional; see core/partition.py).
    """
    scan = _leaf_scan(plan)
    return scan.dataset.partition(partition_id).row_count()


def _leaf_scan(plan: PhysicalPlan) -> ScanExec:
    if isinstance(plan, ScanExec):
        return plan
    return _leaf_scan(plan.children[0])


def _estimate_bytes(rows: list[dict]) -> int:
    """Rough, Python-object-overhead-inclusive estimate, not an on-disk byte
    count. Same heuristic and same caveat as optimizer/statistics.py's
    `compute_statistics`."""
    return sum(sys.getsizeof(v) for row in rows for v in row.values())
