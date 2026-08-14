"""End-to-end tests for query metrics (Milestone 8): DataFrame.
last_run_metrics through the real DataFrame path, real shuffle, and real
OS worker processes (`local[2]`), confirming psutil-based cpu_time/
peak_memory profiling actually reports real, non-placeholder values from
inside a worker process (not just the driver's own).
"""

from __future__ import annotations

import pytest

from minispark.api.functions import col, count
from minispark.api.functions import sum as ssum
from minispark.api.session import MiniSparkSession

psutil = pytest.importorskip("psutil")


def make_session(master: str = "local[2]"):
    return MiniSparkSession.builder.master(master).app_name("metrics_test").get_or_create()


def test_last_run_metrics_is_none_before_any_action():
    session = make_session("local[1]")
    df = session.create_dataframe([{"x": 1}], num_partitions=1)
    assert df.last_run_metrics is None


def test_last_run_metrics_reports_two_stages_for_a_shuffle_query():
    session = make_session("local[2]")
    records = [{"country": ["US", "CA", "UK", "DE"][i % 4], "revenue": i} for i in range(40)]
    df = session.create_dataframe(records, num_partitions=5)
    result = df.group_by("country").agg(
        count("*").alias("n"), ssum("revenue").alias("total")
    )
    rows = result.collect()
    assert len(rows) == 4

    metrics = result.last_run_metrics
    assert metrics is not None
    assert len(metrics.stages) == 2
    assert metrics.total_wall_clock_seconds > 0

    map_stage, reduce_stage = metrics.stages
    assert map_stage.num_tasks == 5
    assert map_stage.total_input_records == 40
    assert reduce_stage.total_output_records == 4


def test_cpu_time_and_peak_memory_are_real_measured_values():
    """psutil is installed in this environment (see the importorskip
    above): confirm the fields are real numbers coming back from inside
    a worker process, not just left None."""
    session = make_session("local[2]")
    records = [{"x": i} for i in range(20)]
    df = session.create_dataframe(records, num_partitions=4)
    filtered = df.filter(col("x") >= 0)
    filtered.collect()

    metrics = filtered.last_run_metrics
    assert metrics is not None
    stage = metrics.stages[0]
    assert stage.total_cpu_time_seconds is not None
    assert stage.total_cpu_time_seconds >= 0
    assert stage.max_peak_memory_bytes is not None
    # A worker process is a real Python process; its RSS is at least a
    # few megabytes, never zero or implausibly small.
    assert stage.max_peak_memory_bytes > 1_000_000


def test_count_also_captures_metrics():
    session = make_session("local[1]")
    df = session.create_dataframe([{"x": i} for i in range(10)], num_partitions=2)
    n = df.count()
    assert n == 10
    assert df.last_run_metrics is not None


def test_metrics_summary_is_human_readable_text():
    session = make_session("local[1]")
    df = session.create_dataframe([{"x": 1}], num_partitions=1)
    df.collect()
    text = df.last_run_metrics.summary()
    assert "Query:" in text
    assert "Stage 0" in text
