"""End-to-end tests for DataFrame.join() through the real DataFrame path:
analyzer, optimizer, physical planner, stage splitting, and
LocalScheduler, including real disk-backed shuffle and (for the
`local[3]` test) real OS-process parallelism. Correctness is checked
against a plain Python join computed independently of MiniSpark.
"""

from __future__ import annotations

import pytest

from minispark.api.functions import col
from minispark.api.session import MiniSparkSession
from minispark.logical.analyzer import AnalysisException


def make_session(master: str = "local[1]"):
    return MiniSparkSession.builder.master(master).app_name("join_test").get_or_create()


USERS = [
    {"id": 1, "name": "alice"},
    {"id": 2, "name": "bob"},
    {"id": 3, "name": "carol"},
]
ORDERS = [
    {"id": 1, "amount": 10},
    {"id": 1, "amount": 20},
    {"id": 2, "amount": 5},
    {"id": 4, "amount": 99},
]


def _reference_inner_join(left, right, on):
    return [
        {**left_row, **{k: v for k, v in right_row.items() if k != on}}
        for left_row in left
        for right_row in right
        if left_row[on] == right_row[on]
    ]


def _by_id_and_amount(row):
    return (row["id"], row["amount"])


def test_shuffle_hash_join_local1():
    session = make_session("local[1]")
    users_df = session.create_dataframe(USERS, num_partitions=2)
    orders_df = session.create_dataframe(ORDERS, num_partitions=2)
    result = users_df.join(orders_df, on="id")
    rows = sorted(result.collect(), key=_by_id_and_amount)
    expected = sorted(_reference_inner_join(USERS, ORDERS, "id"), key=_by_id_and_amount)
    assert rows == expected


def test_broadcast_join_local1_matches_shuffle_join():
    session = make_session("local[1]")
    users_df = session.create_dataframe(USERS, num_partitions=2)
    orders_df = session.create_dataframe(ORDERS, num_partitions=2)
    result = users_df.join(orders_df, on="id", broadcast=True)
    rows = sorted(result.collect(), key=_by_id_and_amount)
    expected = sorted(_reference_inner_join(USERS, ORDERS, "id"), key=_by_id_and_amount)
    assert rows == expected


def test_join_real_multiprocessing_matches_reference():
    session = make_session("local[3]")
    users = [{"id": i, "name": f"user{i}"} for i in range(20)]
    orders = [{"id": i % 20, "amount": i} for i in range(50)]
    users_df = session.create_dataframe(users, num_partitions=4)
    orders_df = session.create_dataframe(orders, num_partitions=5)

    result = users_df.join(orders_df, on="id")
    rows = sorted(result.collect(), key=_by_id_and_amount)
    expected = sorted(_reference_inner_join(users, orders, "id"), key=_by_id_and_amount)
    assert rows == expected


def test_join_with_filter_pushed_into_one_side():
    session = make_session("local[1]")
    users_df = session.create_dataframe(USERS, num_partitions=2)
    orders_df = session.create_dataframe(ORDERS, num_partitions=2)
    result = users_df.join(orders_df, on="id").filter(col("name") == "alice")
    rows = sorted(result.collect(), key=lambda r: r["amount"])
    assert rows == [
        {"id": 1, "name": "alice", "amount": 10},
        {"id": 1, "name": "alice", "amount": 20},
    ]


def test_explain_optimized_shows_join_stages(capsys):
    session = make_session("local[1]")
    users_df = session.create_dataframe(USERS, num_partitions=1)
    orders_df = session.create_dataframe(ORDERS, num_partitions=1)
    users_df.join(orders_df, on="id").explain(optimized=True)
    out = capsys.readouterr().out
    assert "Join[inner, on=(id)]" in out
    assert "HashJoinExec[inner, on=(id)]" in out
    assert "ShuffleWriteExec[hash(id)" in out
    assert "ShuffleReadExec[stage 0]" in out
    assert "ShuffleReadExec[stage 1]" in out
    assert "Stage 0 " in out
    assert "Stage 1 " in out
    assert "Stage 2 " in out


def test_join_unknown_on_column_raises_analysis_exception():
    session = make_session("local[1]")
    users_df = session.create_dataframe(USERS, num_partitions=1)
    orders_df = session.create_dataframe(ORDERS, num_partitions=1)
    with pytest.raises(AnalysisException, match="not found"):
        users_df.join(orders_df, on="does_not_exist").collect()


def test_join_requires_at_least_one_on_column():
    session = make_session("local[1]")
    users_df = session.create_dataframe(USERS, num_partitions=1)
    orders_df = session.create_dataframe(ORDERS, num_partitions=1)
    with pytest.raises(ValueError, match="join\\(\\) requires"):
        users_df.join(orders_df, on=[])
