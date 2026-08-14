"""Unit tests for LocalScheduler's lineage-based recomputation: when a
task's TaskResult reports a missing_shuffle_stage_id, the scheduler must
re-run the upstream stage that produced the missing blocks and retry the
failing task with fresh blocks, rather than simply retrying it in place
(which could never succeed, since the data it needs would still be gone).

Uses an injected stub run_task, exactly like tests/unit/test_scheduler.py,
so this stays fast and deterministic; genuine multiprocessing with real
shuffle files actually going missing gets its own dedicated test:
tests/integration/test_lineage_recovery_e2e.py.
"""

from __future__ import annotations

import pytest

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT
from minispark.execution.scheduler import LocalScheduler, TaskExecutionError
from minispark.execution.stages import Stage
from minispark.execution.tasks import Task, TaskMetrics, TaskResult, TaskState
from minispark.expressions.column import Column
from minispark.physical.plan import ScanExec, ShuffleReadExec, ShuffleWriteExec
from minispark.shuffle.writer import ShuffleBlockMeta


def make_two_stage_plan(upstream_partitions: int = 2) -> list[Stage]:
    schema = Schema([Field("x", INT)])
    partitions = [
        Partition(i, schema, lambda i=i: iter([{"x": i}]), PartitionMetadata(row_count=1))
        for i in range(upstream_partitions)
    ]
    dataset = Dataset(schema, partitions)
    write_plan = ShuffleWriteExec(ScanExec(dataset, "test"), 1, [Column("x")])
    stage0 = Stage(stage_id=0, plan=write_plan, num_partitions=upstream_partitions)
    read_plan = ShuffleReadExec(from_stage_id=0, schema=schema)
    stage1 = Stage(stage_id=1, plan=read_plan, num_partitions=1)
    return [stage0, stage1]


def _fake_block(task_id: int) -> ShuffleBlockMeta:
    return ShuffleBlockMeta(
        stage_id=0,
        source_task_id=task_id,
        target_partition=0,
        path=f"fake_block_{task_id}.pkl",
        record_count=1,
        byte_length=10,
        checksum="deadbeef",
    )


def test_recomputes_missing_upstream_stage_and_then_succeeds():
    calls = {"stage0": 0, "stage1": 0}

    def stub(task: Task, attempt: int) -> TaskResult:
        if task.stage_id == 0:
            calls["stage0"] += 1
            return TaskResult(
                task_id=task.task_id,
                state=TaskState.SUCCESS,
                metrics=TaskMetrics(),
                shuffle_blocks=[_fake_block(task.task_id)],
            )
        calls["stage1"] += 1
        if calls["stage1"] == 1:
            return TaskResult(
                task_id=task.task_id,
                state=TaskState.FAILED,
                error="MissingShuffleDataError: gone",
                missing_shuffle_stage_id=0,
            )
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.SUCCESS,
            rows=[{"x": 1}],
            metrics=TaskMetrics(output_records=1),
        )

    # max_retries=0: an ordinary failure would raise immediately with no
    # retries left, so the recompute-and-retry only working because of
    # missing_shuffle_stage_id (not because of leftover retry budget) is
    # exactly what this proves.
    scheduler = LocalScheduler(num_workers=1, max_retries=0, run_task=stub)
    dataset = scheduler.run_plan(make_two_stage_plan(upstream_partitions=2))

    assert list(dataset.iter_records()) == [{"x": 1}]
    # Stage 0 ran twice (its normal run, plus one full recompute): 2
    # partitions each time.
    assert calls["stage0"] == 4
    # Stage 1's one task failed once, then succeeded once, after recovery.
    assert calls["stage1"] == 2


def test_a_stage_is_recomputed_at_most_once_per_run():
    calls = {"stage0": 0, "stage1": 0}

    def always_missing(task: Task, attempt: int) -> TaskResult:
        if task.stage_id == 0:
            calls["stage0"] += 1
            return TaskResult(
                task_id=task.task_id,
                state=TaskState.SUCCESS,
                metrics=TaskMetrics(),
                shuffle_blocks=[_fake_block(task.task_id)],
            )
        calls["stage1"] += 1
        # Stage 1 reports the same missing stage every single time, as if
        # recomputing stage 0 never actually fixes anything (e.g. a
        # permanently broken source). Recovery must not loop forever.
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.FAILED,
            error="MissingShuffleDataError: still gone",
            missing_shuffle_stage_id=0,
        )

    scheduler = LocalScheduler(num_workers=1, max_retries=0, run_task=always_missing)
    with pytest.raises(TaskExecutionError):
        scheduler.run_plan(make_two_stage_plan(upstream_partitions=1))

    # Exactly one recompute of stage 0 (its normal run, plus one retry),
    # not an unbounded number.
    assert calls["stage0"] == 2
    # Stage 1's task was tried once, recovered from once, and tried again
    # once more before the bound kicked in and it was left to fail.
    assert calls["stage1"] == 2
