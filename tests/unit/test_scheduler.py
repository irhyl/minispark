"""Unit tests for LocalScheduler: success, retry-then-succeed, and
retry-exhausted paths, using an injected stub `run_task` so these stay
fast and deterministic (no real subprocess spawn: num_workers=1 never
touches ProcessPoolExecutor). Real multiprocessing gets its own dedicated
test: tests/integration/test_scheduler_multiprocessing.py.
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
from minispark.physical.plan import ScanExec


def make_stage(num_partitions: int = 3) -> Stage:
    schema = Schema([Field("x", INT)])
    partitions = [
        Partition(i, schema, lambda i=i: iter([{"x": i}]), PartitionMetadata(row_count=1))
        for i in range(num_partitions)
    ]
    dataset = Dataset(schema, partitions)
    plan = ScanExec(dataset, "test")
    return Stage(stage_id=0, plan=plan, num_partitions=num_partitions)


def test_run_stage_success_merges_all_partitions():
    def stub(task: Task, attempt: int) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.SUCCESS,
            rows=[{"x": task.partition_id}],
            metrics=TaskMetrics(output_records=1),
        )

    scheduler = LocalScheduler(num_workers=1, run_task=stub)
    dataset = scheduler.run_stage(make_stage(3))
    assert sorted(r["x"] for r in dataset.iter_records()) == [0, 1, 2]


def test_task_retries_on_failure_then_succeeds():
    attempts_seen: list[int] = []

    def stub(task: Task, attempt: int) -> TaskResult:
        attempts_seen.append(attempt)
        if attempt < 2:
            return TaskResult(task_id=task.task_id, state=TaskState.FAILED, error="boom")
        return TaskResult(task_id=task.task_id, state=TaskState.SUCCESS, rows=[{"x": 0}])

    scheduler = LocalScheduler(num_workers=1, max_retries=3, run_task=stub)
    dataset = scheduler.run_stage(make_stage(1))
    assert list(dataset.iter_records()) == [{"x": 0}]
    assert attempts_seen == [0, 1, 2]


def test_task_failure_exhausting_retries_raises_with_the_last_error():
    def always_fails(task: Task, attempt: int) -> TaskResult:
        return TaskResult(task_id=task.task_id, state=TaskState.FAILED, error="permanent failure")

    scheduler = LocalScheduler(num_workers=1, max_retries=2, run_task=always_fails)
    with pytest.raises(TaskExecutionError, match="permanent failure"):
        scheduler.run_stage(make_stage(1))


def test_only_the_failing_partition_is_retried():
    """A failure on one partition must not cause every partition to retry."""
    attempts_by_partition: dict[int, int] = {}

    def stub(task: Task, attempt: int) -> TaskResult:
        pid = task.partition_id
        attempts_by_partition[pid] = attempts_by_partition.get(pid, 0) + 1
        if pid == 1 and attempt == 0:
            return TaskResult(task_id=task.task_id, state=TaskState.FAILED, error="flaky")
        return TaskResult(task_id=task.task_id, state=TaskState.SUCCESS, rows=[{"x": pid}])

    scheduler = LocalScheduler(num_workers=1, max_retries=1, run_task=stub)
    dataset = scheduler.run_stage(make_stage(3))
    assert sorted(r["x"] for r in dataset.iter_records()) == [0, 1, 2]
    assert attempts_by_partition == {0: 1, 1: 2, 2: 1}


def test_last_metrics_is_none_before_any_run():
    scheduler = LocalScheduler(num_workers=1)
    assert scheduler.last_metrics is None


def test_last_metrics_records_one_stage_with_correct_task_count():
    def stub(task: Task, attempt: int) -> TaskResult:
        return TaskResult(
            task_id=task.task_id,
            state=TaskState.SUCCESS,
            rows=[{"x": task.partition_id}],
            metrics=TaskMetrics(output_records=1, output_bytes=10),
        )

    scheduler = LocalScheduler(num_workers=1, run_task=stub)
    scheduler.run_stage(make_stage(3))

    metrics = scheduler.last_metrics
    assert metrics is not None
    assert len(metrics.stages) == 1
    stage = metrics.stages[0]
    assert stage.num_tasks == 3
    assert stage.total_output_records == 3
    assert stage.total_output_bytes == 30
    assert stage.retried_tasks == 0
    assert stage.recomputed is False
    assert metrics.total_wall_clock_seconds >= 0


def test_last_metrics_counts_retried_tasks_once_each():
    def stub(task: Task, attempt: int) -> TaskResult:
        if task.partition_id == 1 and attempt == 0:
            return TaskResult(task_id=task.task_id, state=TaskState.FAILED, error="flaky")
        return TaskResult(
            task_id=task.task_id, state=TaskState.SUCCESS, rows=[{"x": task.partition_id}]
        )

    scheduler = LocalScheduler(num_workers=1, max_retries=1, run_task=stub)
    scheduler.run_stage(make_stage(3))

    stage = scheduler.last_metrics.stages[0]
    assert stage.retried_tasks == 1
