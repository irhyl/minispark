"""End-to-end tests for DataFrame.order_by() through the real DataFrame
path: analyzer, optimizer, physical planner, stage splitting, and
LocalScheduler, including real disk-backed shuffle and (for the
`local[3]` test) real OS-process parallelism.
"""

from __future__ import annotations

import random

import pytest

from minispark.api.session import MiniSparkSession
from minispark.logical.analyzer import AnalysisException


def make_session(master: str = "local[1]"):
    return MiniSparkSession.builder.master(master).app_name("sort_test").get_or_create()


def test_order_by_ascending_local1():
    session = make_session("local[1]")
    records = [{"age": 30}, {"age": 5}, {"age": 60}, {"age": 15}, {"age": 45}]
    df = session.create_dataframe(records, num_partitions=2)
    rows = df.order_by("age").collect()
    assert [r["age"] for r in rows] == [5, 15, 30, 45, 60]


def test_order_by_descending_local1():
    session = make_session("local[1]")
    records = [{"age": 30}, {"age": 5}, {"age": 60}, {"age": 15}, {"age": 45}]
    df = session.create_dataframe(records, num_partitions=2)
    rows = df.order_by("age", ascending=False).collect()
    assert [r["age"] for r in rows] == [60, 45, 30, 15, 5]


def test_sort_is_an_alias_for_order_by():
    session = make_session("local[1]")
    records = [{"age": 30}, {"age": 5}]
    df = session.create_dataframe(records, num_partitions=1)
    rows = df.sort("age").collect()
    assert [r["age"] for r in rows] == [5, 30]


def test_order_by_real_multiprocessing_matches_reference():
    session = make_session("local[3]")
    random.seed(7)
    records = [{"id": i, "age": random.randint(0, 200)} for i in range(80)]
    df = session.create_dataframe(records, num_partitions=6)

    rows = df.order_by("age").collect()
    ages = [r["age"] for r in rows]
    assert len(rows) == 80
    assert ages == sorted(r["age"] for r in records)

    rows_desc = df.order_by("age", ascending=False).collect()
    assert [r["age"] for r in rows_desc] == sorted((r["age"] for r in records), reverse=True)


def test_multi_key_sort_mixed_direction_real_multiprocessing():
    session = make_session("local[3]")
    random.seed(11)
    records = [
        {"id": i, "age": random.randint(0, 5), "score": random.randint(0, 1000)}
        for i in range(60)
    ]
    df = session.create_dataframe(records, num_partitions=4)
    rows = df.order_by("age", "score", ascending=[True, False]).collect()

    expected = sorted(records, key=lambda r: (r["age"], -r["score"]))
    assert [(r["age"], r["score"]) for r in rows] == [
        (r["age"], r["score"]) for r in expected
    ]


def test_sort_on_string_column_still_produces_a_correct_global_order():
    """No multi-bucket range partitioning exists for non-numeric keys
    (see physical/planner.py); this proves the single-partition fallback
    is still correct, just not parallel across the shuffle."""
    session = make_session("local[2]")
    names = ["carol", "alice", "erin", "bob", "dave"]
    records = [{"name": n} for n in names]
    df = session.create_dataframe(records, num_partitions=2)
    rows = df.order_by("name").collect()
    assert [r["name"] for r in rows] == sorted(names)


def test_explain_optimized_shows_local_and_final_sort_stages(capsys):
    session = make_session("local[1]")
    df = session.create_dataframe([{"age": 1}, {"age": 2}], num_partitions=1)
    df.order_by("age").explain(optimized=True)
    out = capsys.readouterr().out
    assert "Sort[age ASC]" in out
    assert "SortExec[age ASC]" in out
    assert "Exchange[" in out
    assert "Stage 0 " in out
    assert "Stage 1 " in out


def test_order_by_unknown_column_raises_analysis_exception():
    session = make_session("local[1]")
    df = session.create_dataframe([{"age": 1}], num_partitions=1)
    with pytest.raises(AnalysisException, match="does_not_exist"):
        df.order_by("does_not_exist").collect()


def test_order_by_requires_at_least_one_column():
    session = make_session("local[1]")
    df = session.create_dataframe([{"age": 1}], num_partitions=1)
    with pytest.raises(ValueError, match="order_by\\(\\) requires"):
        df.order_by()
