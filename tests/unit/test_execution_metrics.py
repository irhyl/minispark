"""Unit tests for execution/metrics.py's aggregation: StageMetrics.
from_task_metrics() and QueryMetrics.summary(). Pure functions, no
scheduler/worker involved; execution/scheduler.py's own wiring of these
into a real run is covered separately (tests/unit/test_scheduler.py,
tests/integration/test_metrics_e2e.py).
"""

from __future__ import annotations

import pytest

from minispark.execution.metrics import QueryMetrics, StageMetrics
from minispark.execution.tasks import TaskMetrics


def test_sums_exact_fields_across_tasks():
    metrics = [
        TaskMetrics(
            execution_time_seconds=1.0, output_records=3, output_bytes=100, shuffle_bytes=10
        ),
        TaskMetrics(
            execution_time_seconds=2.0, output_records=5, output_bytes=200, shuffle_bytes=20
        ),
    ]
    stage = StageMetrics.from_task_metrics(
        stage_id=0, metrics=metrics, wall_clock_seconds=1.5, retried_tasks=0, recomputed=False
    )
    assert stage.num_tasks == 2
    assert stage.total_execution_time_seconds == 3.0
    assert stage.total_output_records == 8
    assert stage.total_output_bytes == 300
    assert stage.total_shuffle_bytes == 30
    assert stage.wall_clock_seconds == 1.5


def test_input_records_none_when_every_task_unknown():
    metrics = [TaskMetrics(input_records=None), TaskMetrics(input_records=None)]
    stage = StageMetrics.from_task_metrics(0, metrics, 0.1, 0, False)
    assert stage.total_input_records is None


def test_input_records_sums_known_values_even_when_some_are_unknown():
    metrics = [
        TaskMetrics(input_records=5),
        TaskMetrics(input_records=None),
        TaskMetrics(input_records=3),
    ]
    stage = StageMetrics.from_task_metrics(0, metrics, 0.1, 0, False)
    assert stage.total_input_records == 8


def test_cpu_time_and_peak_memory_none_without_psutil_data():
    metrics = [TaskMetrics(), TaskMetrics()]
    stage = StageMetrics.from_task_metrics(0, metrics, 0.1, 0, False)
    assert stage.total_cpu_time_seconds is None
    assert stage.max_peak_memory_bytes is None


def test_cpu_time_sums_and_peak_memory_takes_the_max():
    metrics = [
        TaskMetrics(cpu_time_seconds=0.1, peak_memory_bytes=1000),
        TaskMetrics(cpu_time_seconds=0.2, peak_memory_bytes=3000),
    ]
    stage = StageMetrics.from_task_metrics(0, metrics, 0.1, 0, False)
    assert stage.total_cpu_time_seconds == pytest.approx(0.3)
    assert stage.max_peak_memory_bytes == 3000


def test_retried_tasks_and_recomputed_flag_pass_through():
    stage = StageMetrics.from_task_metrics(
        0, [TaskMetrics()], 0.1, retried_tasks=2, recomputed=True
    )
    assert stage.retried_tasks == 2
    assert stage.recomputed is True


def test_query_metrics_summary_includes_every_stage():
    stages = [
        StageMetrics.from_task_metrics(0, [TaskMetrics(output_records=3)], 0.1, 0, False),
        StageMetrics.from_task_metrics(1, [TaskMetrics(output_records=5)], 0.2, 1, True),
    ]
    query = QueryMetrics(stages=stages, total_wall_clock_seconds=0.5)
    text = query.summary()
    assert "0.500s wall clock" in text
    assert "2 stage run(s)" in text
    assert "Stage 0:" in text
    assert "Stage 1 (lineage recompute):" in text
    assert "retries=1" in text


def test_query_metrics_default_construction():
    query = QueryMetrics()
    assert query.stages == []
    assert query.total_wall_clock_seconds == 0.0
    assert "0 stage run(s)" in query.summary()
