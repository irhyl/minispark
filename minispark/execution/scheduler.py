"""LocalScheduler: turns a Stage into Tasks, runs them, retries failures,
and merges results back into one Dataset.

`local[N]` selects how: `N == 1` runs tasks sequentially in this process
(no multiprocessing overhead, still goes through the exact same Task ->
TaskResult path as `N > 1`); `N > 1` runs tasks across a real
`ProcessPoolExecutor`, actual OS processes, not threads (the build spec is
explicit that threads are the wrong tool here because of the GIL). This is
also why every Task and its PhysicalPlan had to become genuinely
picklable (see storage/memory.py, storage/csv.py, physical/plan.py):
`ProcessPoolExecutor` sends both across a real process boundary with
`pickle`, there is no shortcut around that requirement.

Retries happen in this process, not inside a worker: `execute_task`
(execution/worker.py) already converts an exception into a FAILED
TaskResult instead of raising, so "should this be retried" is always a
plain decision this scheduler makes by inspecting a TaskResult, whether
that result came back from a local call or from a pool worker.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

from minispark.config.log import get_logger
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Schema
from minispark.execution.stages import Stage
from minispark.execution.tasks import Task, TaskResult, TaskState
from minispark.execution.worker import execute_task

logger = get_logger("scheduler")

RunTaskFn = Callable[[Task, int], TaskResult]


class TaskExecutionError(Exception):
    """A task exhausted its retries. No lineage-based recomputation exists
    yet (Milestone 6); Milestone 3's failure handling stops at retry."""


class LocalScheduler:
    def __init__(
        self,
        num_workers: int = 1,
        max_retries: int = 3,
        run_task: RunTaskFn | None = None,
    ):
        self.num_workers = max(1, num_workers)
        self.max_retries = max_retries
        # Overridable for tests that want to exercise scheduling/retry logic
        # without paying real subprocess cost, or that want to inject
        # deterministic failures (fault injection). Must stay a plain,
        # importable, module-level callable when num_workers > 1: it is
        # sent to worker processes exactly like `execute_task` is.
        self._run_task = run_task or execute_task

    def run_stage(self, stage: Stage) -> Dataset:
        tasks = [
            Task(task_id=i, stage_id=stage.stage_id, partition_id=i, plan=stage.plan)
            for i in range(stage.num_partitions)
        ]
        logger.info("StageStarted stage_id=%s tasks=%s", stage.stage_id, len(tasks))
        results = self._run_to_completion(tasks)
        logger.info("StageCompleted stage_id=%s", stage.stage_id)
        return _results_to_dataset(stage.plan.schema, results)

    def _run_to_completion(self, tasks: list[Task]) -> list[TaskResult]:
        pending: dict[int, tuple[Task, int]] = {t.task_id: (t, 0) for t in tasks}
        done: dict[int, TaskResult] = {}
        while pending:
            batch = list(pending.values())
            batch_results = self._run_batch(batch)
            pending = {}
            for (task, attempt), result in zip(batch, batch_results, strict=True):
                if result.state is TaskState.FAILED and attempt < self.max_retries:
                    logger.info(
                        "TaskRetrying task_id=%s attempt=%s error=%s",
                        task.task_id,
                        attempt + 1,
                        result.error,
                    )
                    pending[task.task_id] = (task, attempt + 1)
                else:
                    if result.state is TaskState.FAILED:
                        logger.info(
                            "TaskFailed task_id=%s attempts=%s error=%s",
                            task.task_id,
                            attempt + 1,
                            result.error,
                        )
                    done[task.task_id] = result
        results = [done[t.task_id] for t in tasks]
        failed = [r for r in results if r.state is TaskState.FAILED]
        if failed:
            first = failed[0]
            raise TaskExecutionError(
                f"{len(failed)} of {len(results)} task(s) failed after "
                f"{self.max_retries} retries; first error "
                f"(task_id={first.task_id}): {first.error}"
            )
        return results

    def _run_batch(self, batch: list[tuple[Task, int]]) -> list[TaskResult]:
        if self.num_workers == 1 or len(batch) <= 1:
            return [self._run_task(task, attempt) for task, attempt in batch]
        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            return list(
                pool.map(self._run_task, [t for t, _ in batch], [a for _, a in batch])
            )


def _results_to_dataset(schema: Schema, results: list[TaskResult]) -> Dataset:
    partitions = [
        Partition(
            partition_id=i,
            schema=schema,
            records_fn=functools.partial(iter, result.rows),
            metadata=PartitionMetadata(row_count=result.metrics.output_records),
        )
        for i, result in enumerate(results)
    ]
    return Dataset(schema, partitions)
