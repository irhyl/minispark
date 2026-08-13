from minispark.api.functions import col
from minispark.core.types import FLOAT, INT, STRING
from minispark.expressions.aggregate import Avg, Count, Max, Min, Sum


def _run(agg, records):
    state = agg.initialize()
    for record in records:
        state = agg.update(state, record)
    return agg.finalize(state)


def test_count_star_counts_all_rows():
    agg = Count(None)
    result = _run(agg, [{"a": 1}, {"a": None}, {"a": 3}])
    assert result == 3


def test_count_column_ignores_nulls():
    agg = Count(col("a"))
    result = _run(agg, [{"a": 1}, {"a": None}, {"a": 3}])
    assert result == 2


def test_sum_ignores_nulls():
    agg = Sum(col("a"))
    result = _run(agg, [{"a": 1}, {"a": None}, {"a": 3}])
    assert result == 4


def test_sum_of_all_nulls_is_none():
    agg = Sum(col("a"))
    result = _run(agg, [{"a": None}, {"a": None}])
    assert result is None


def test_avg_ignores_nulls():
    agg = Avg(col("a"))
    result = _run(agg, [{"a": 10}, {"a": None}, {"a": 20}])
    assert result == 15.0


def test_avg_of_no_rows_is_none():
    agg = Avg(col("a"))
    result = _run(agg, [])
    assert result is None


def test_min_and_max():
    records = [{"a": 5}, {"a": 1}, {"a": None}, {"a": 9}]
    assert _run(Min(col("a")), records) == 1
    assert _run(Max(col("a")), records) == 9


def test_merge_combines_two_partial_states():
    agg = Sum(col("a"))
    state_a = _run_to_state(agg, [{"a": 1}, {"a": 2}])
    state_b = _run_to_state(agg, [{"a": 10}])
    merged = agg.merge(state_a, state_b)
    assert agg.finalize(merged) == 13


def test_merge_min_max():
    agg_min = Min(col("a"))
    state_a = _run_to_state(agg_min, [{"a": 5}])
    state_b = _run_to_state(agg_min, [{"a": 2}])
    assert agg_min.finalize(agg_min.merge(state_a, state_b)) == 2

    agg_max = Max(col("a"))
    state_a = _run_to_state(agg_max, [{"a": 5}])
    state_b = _run_to_state(agg_max, [{"a": 2}])
    assert agg_max.finalize(agg_max.merge(state_a, state_b)) == 5


def test_merge_avg_combines_sum_and_count_correctly():
    agg = Avg(col("a"))
    state_a = _run_to_state(agg, [{"a": 10}, {"a": 20}])  # sum=30, count=2
    state_b = _run_to_state(agg, [{"a": 40}])  # sum=40, count=1
    merged = agg.merge(state_a, state_b)
    assert agg.finalize(merged) == 70 / 3


def test_result_types():
    assert Count(None).result_type(STRING) == INT
    assert Sum(col("a")).result_type(INT) == INT
    assert Avg(col("a")).result_type(INT) == FLOAT
    assert Min(col("a")).result_type(INT) == INT
    assert Max(col("a")).result_type(INT) == INT


def _run_to_state(agg, records):
    state = agg.initialize()
    for record in records:
        state = agg.update(state, record)
    return state
