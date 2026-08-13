"""LocalScheduler: turns Stages into Tasks, runs them, retries failures,
shuffles data between stages, and merges the last stage's results into
one Dataset.

`local[N]` selects how tasks run: `N == 1` runs them sequentially in this
process (no multiprocessing overhead, still goes through the exact same
Task -> TaskResult path as `N > 1`); `N > 1` runs them across a real
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

`run_plan()` runs a list of Stages *in order*: a stage that reads shuffle
input cannot start until the stage that wrote it has fully finished (that
is what a wide dependency means, see execution/dag.py), so stages are not
pipelined or run concurrently with each other, only the tasks within one
stage are. A stage whose plan is rooted at `ShuffleWriteExec` does not
produce this query's final rows; its tasks' shuffle block metadata is
registered into a `ShuffleManager` instead, so the next stage's tasks know
what to read.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor

from minispark.config.log import get_logger
from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Schema
from minispark.execution.stages import Stage
from minispark.execution.tasks import Task, TaskResult, TaskState
from minispark.execution.worker import execute_task
from minispark.physical.plan import ShuffleReadExec, ShuffleWriteExec, leaves
from minispark.shuffle.manager import ShuffleManager
from minispark.shuffle.writer import ShuffleBlockMeta

logger = get_logger("scheduler")

RunTaskFn = Callable[[Task, int], TaskResult]


class TaskExecutionError(Exception):
    """A task exhausted its retries. No lineage-based recomputation exists
    yet (Milestone 6); failure handling stops at retry."""


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
        """Run a single stage. A thin wrapper around `run_plan([stage])`,
        kept because a plan with no shuffle boundary (Scan/Filter/Project
        only) is still exactly one stage, and callers/tests with a single
        Stage in hand should not have to build a one-element list."""
        return self.run_plan([stage])

    def run_plan(self, stages: list[Stage]) -> Dataset:
        shuffle_manager = ShuffleManager()
        task_ids = itertools.count()
        try:
            final_dataset: Dataset | None = None
            for stage in stages:
                results = self._run_stage_tasks(stage, shuffle_manager, task_ids)
                if isinstance(stage.plan, ShuffleWriteExec):
                    blocks: list[ShuffleBlockMeta] = [
                        b for r in results for b in r.shuffle_blocks
                    ]
                    shuffle_manager.register_blocks(stage.stage_id, blocks)
                    logger.info(
                        "ShuffleCompleted stage_id=%s blocks=%s", stage.stage_id, len(blocks)
                    )
                else:
                    final_dataset = _results_to_dataset(stage.plan.schema, results)
            if final_dataset is None:
                raise AssertionError("run_plan() received no stages")
            return final_dataset
        finally:
            shuffle_manager.cleanup()

    def _run_stage_tasks(
        self, stage: Stage, shuffle_manager: ShuffleManager, task_ids: itertools.count
    ) -> list[TaskResult]:
        # A stage's plan can read from more than one prior stage (a
        # HashJoinExec-rooted stage has one ShuffleReadExec leaf per
        # side). Each is resolved independently: a normal read fetches
        # this task's own partition_id; a broadcast read (see
        # physical/plan.py's ShuffleReadExec.is_broadcast) always fetches
        # target partition 0, the same blocks for every task in this
        # stage, regardless of that task's own partition_id.
        read_leaves = [leaf for leaf in leaves(stage.plan) if isinstance(leaf, ShuffleReadExec)]
        tasks = [
            Task(
                task_id=next(task_ids),
                stage_id=stage.stage_id,
                partition_id=pid,
                plan=stage.plan,
                shuffle_root_dir=shuffle_manager.root_dir,
                shuffle_blocks=(
                    {
                        leaf.from_stage_id: shuffle_manager.blocks_for(
                            leaf.from_stage_id, 0 if leaf.is_broadcast else pid
                        )
                        for leaf in read_leaves
                    }
                    if read_leaves
                    else None
                ),
            )
            for pid in range(stage.num_partitions)
        ]
        logger.info("StageStarted stage_id=%s tasks=%s", stage.stage_id, len(tasks))
        results = self._run_to_completion(tasks)
        logger.info("StageCompleted stage_id=%s", stage.stage_id)
        return results

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
