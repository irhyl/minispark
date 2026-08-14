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

Lineage-based recomputation (Milestone 6): a task can fail because the
shuffle blocks it needs to read are simply gone or corrupted (a
`MissingShuffleDataError`, see shuffle/reader.py and execution/worker.py),
not because of an ordinary, possibly-transient error. Retrying that same
task in place can never succeed in that case; the data has to be
regenerated first. `_try_recover_missing_shuffle` handles this by looking
up the stage that produced the missing blocks (every stage this run built
is kept in `stages_by_id`) and re-running that stage's tasks from
scratch, exactly the same way it ran the first time, re-registering
whatever fresh blocks it produces into the `ShuffleManager`. If *that*
stage itself reads shuffle input that also turns out to be missing, the
same mechanism fires again for it first, so recovery walks back through
however many stages it takes, bounded only by the number of stages in the
plan (each stage is recomputed at most once per `run_plan()` call, see
`recomputed_stages` below; a stage whose data goes missing a second time
is treated as an ordinary failure and follows the normal retry-then-raise
path instead of recomputing forever). This only recovers data that was
computed successfully once and then lost; it does not invent data that
was never produced (a permanently unreadable source still fails the
query, correctly).
"""

from __future__ import annotations

import dataclasses
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
from minispark.physical.plan import PhysicalPlan, ShuffleReadExec, ShuffleWriteExec, leaves
from minispark.shuffle.manager import ShuffleManager
from minispark.shuffle.writer import ShuffleBlockMeta

logger = get_logger("scheduler")

RunTaskFn = Callable[[Task, int], TaskResult]


class TaskExecutionError(Exception):
    """A task exhausted its retries. Raised either for an ordinary failure
    (see `LocalScheduler.max_retries`), or for a missing-shuffle-data
    failure whose stage had already been recomputed once this run and
    still could not produce the needed blocks (see
    `_try_recover_missing_shuffle`): lineage-based recomputation is not
    unlimited retried, it is one extra attempt to regenerate lost data,
    not a way to paper over a source that is genuinely gone."""


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
        stages_by_id = {stage.stage_id: stage for stage in stages}
        # Which stage_ids have already been recomputed once because a
        # downstream task found their shuffle blocks missing. Bounds
        # lineage-based recomputation to at most one extra full run per
        # stage per query, so a stage whose data keeps going missing (a
        # genuinely broken source, not a one-off loss) cannot make the
        # scheduler recompute it forever; see _try_recover_missing_shuffle.
        recomputed_stages: set[int] = set()
        try:
            final_dataset: Dataset | None = None
            for stage in stages:
                results = self._run_stage(
                    stage, shuffle_manager, task_ids, stages_by_id, recomputed_stages
                )
                if not isinstance(stage.plan, ShuffleWriteExec):
                    final_dataset = _results_to_dataset(stage.plan.schema, results)
            if final_dataset is None:
                raise AssertionError("run_plan() received no stages")
            return final_dataset
        finally:
            shuffle_manager.cleanup()

    def _run_stage(
        self,
        stage: Stage,
        shuffle_manager: ShuffleManager,
        task_ids: itertools.count,
        stages_by_id: dict[int, Stage],
        recomputed_stages: set[int],
    ) -> list[TaskResult]:
        """Run every task in `stage` and, if it is a shuffle-write stage,
        register the blocks its tasks produced. Used both for a stage's
        normal, once-per-query run (from `run_plan`) and for a lineage
        recomputation re-run of an upstream stage whose blocks were found
        missing (from `_try_recover_missing_shuffle`); both cases need
        exactly the same "run tasks, then register whatever it wrote"
        behavior, so this is the one place that does it.
        """
        results = self._run_stage_tasks(
            stage, shuffle_manager, task_ids, stages_by_id, recomputed_stages
        )
        if isinstance(stage.plan, ShuffleWriteExec):
            blocks: list[ShuffleBlockMeta] = [b for r in results for b in r.shuffle_blocks]
            shuffle_manager.register_blocks(stage.stage_id, blocks)
            logger.info("ShuffleCompleted stage_id=%s blocks=%s", stage.stage_id, len(blocks))
        return results

    def _run_stage_tasks(
        self,
        stage: Stage,
        shuffle_manager: ShuffleManager,
        task_ids: itertools.count,
        stages_by_id: dict[int, Stage],
        recomputed_stages: set[int],
    ) -> list[TaskResult]:
        tasks = [
            Task(
                task_id=next(task_ids),
                stage_id=stage.stage_id,
                partition_id=pid,
                plan=stage.plan,
                shuffle_root_dir=shuffle_manager.root_dir,
                shuffle_blocks=_resolve_shuffle_blocks(stage.plan, pid, shuffle_manager),
            )
            for pid in range(stage.num_partitions)
        ]
        logger.info("StageStarted stage_id=%s tasks=%s", stage.stage_id, len(tasks))
        results = self._run_to_completion(
            tasks, shuffle_manager, task_ids, stages_by_id, recomputed_stages
        )
        logger.info("StageCompleted stage_id=%s", stage.stage_id)
        return results

    def _run_to_completion(
        self,
        tasks: list[Task],
        shuffle_manager: ShuffleManager,
        task_ids: itertools.count,
        stages_by_id: dict[int, Stage],
        recomputed_stages: set[int],
    ) -> list[TaskResult]:
        pending: dict[int, tuple[Task, int]] = {t.task_id: (t, 0) for t in tasks}
        done: dict[int, TaskResult] = {}
        while pending:
            batch = list(pending.values())
            batch_results = self._run_batch(batch)
            pending = {}
            for (task, attempt), result in zip(batch, batch_results, strict=True):
                if (
                    result.state is TaskState.FAILED
                    and result.missing_shuffle_stage_id is not None
                ):
                    recovered = self._try_recover_missing_shuffle(
                        task, result, shuffle_manager, task_ids, stages_by_id, recomputed_stages
                    )
                    if recovered is not None:
                        # Not attempt + 1: the task itself did not fail on
                        # its own merits, its input was regenerated, so
                        # this does not spend any of its ordinary retry
                        # budget.
                        pending[task.task_id] = (recovered, attempt)
                        continue
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

    def _try_recover_missing_shuffle(
        self,
        task: Task,
        result: TaskResult,
        shuffle_manager: ShuffleManager,
        task_ids: itertools.count,
        stages_by_id: dict[int, Stage],
        recomputed_stages: set[int],
    ) -> Task | None:
        """If `result` failed because an upstream stage's shuffle blocks
        were missing, and that stage has not already been recomputed once
        this run, recompute it and return `task` with fresh shuffle_blocks
        so the caller can retry it. Returns None (no recovery attempted,
        caller falls back to ordinary retry-then-raise) when the stage is
        unknown or has already been recomputed once: recomputation is a
        one-shot repair per stage, not a second retry loop layered on top
        of the first.
        """
        missing_stage_id = result.missing_shuffle_stage_id
        assert missing_stage_id is not None
        if missing_stage_id in recomputed_stages:
            return None
        missing_stage = stages_by_id.get(missing_stage_id)
        if missing_stage is None:
            return None
        logger.info(
            "LineageRecompute stage_id=%s triggered_by_task=%s reason=%s",
            missing_stage_id,
            task.task_id,
            result.error,
        )
        recomputed_stages.add(missing_stage_id)
        self._run_stage(missing_stage, shuffle_manager, task_ids, stages_by_id, recomputed_stages)
        fresh_blocks = _resolve_shuffle_blocks(task.plan, task.partition_id, shuffle_manager)
        return dataclasses.replace(task, shuffle_blocks=fresh_blocks)

    def _run_batch(self, batch: list[tuple[Task, int]]) -> list[TaskResult]:
        if self.num_workers == 1 or len(batch) <= 1:
            return [self._run_task(task, attempt) for task, attempt in batch]
        with ProcessPoolExecutor(max_workers=self.num_workers) as pool:
            return list(
                pool.map(self._run_task, [t for t, _ in batch], [a for _, a in batch])
            )


def _resolve_shuffle_blocks(
    plan: PhysicalPlan, partition_id: int, shuffle_manager: ShuffleManager
) -> dict[int, list[ShuffleBlockMeta]] | None:
    """Every `ShuffleReadExec` leaf in `plan`, resolved to the blocks
    `shuffle_manager` currently has registered for it (a normal read
    fetches `partition_id`; a broadcast read, see physical/plan.py's
    `ShuffleReadExec.is_broadcast`, always fetches target partition 0,
    the same blocks regardless of `partition_id`). Used both to build a
    stage's tasks the first time and to refresh one task's shuffle_blocks
    after lineage-based recomputation has re-registered fresh blocks for
    a stage it depends on (`LocalScheduler._try_recover_missing_shuffle`);
    both need exactly the same leaf-by-leaf resolution.
    """
    read_leaves = [leaf for leaf in leaves(plan) if isinstance(leaf, ShuffleReadExec)]
    if not read_leaves:
        return None
    return {
        leaf.from_stage_id: shuffle_manager.blocks_for(
            leaf.from_stage_id, 0 if leaf.is_broadcast else partition_id
        )
        for leaf in read_leaves
    }


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
