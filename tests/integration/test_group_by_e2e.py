"""End-to-end tests for group_by().agg() through the real DataFrame path:
analyzer, optimizer, physical planner, stage splitting, and
LocalScheduler, including a real disk-backed shuffle and (for the
`local[3]` test) real OS-process parallelism. Correctness is checked
against a plain Python group-by computed independently of MiniSpark, per
the build spec's "MiniSpark result == reference result" testing rule.
"""

from __future__ import annotations

import collections

import pytest

from minispark.api.functions import avg, col, count
from minispark.api.functions import max as mmax
from minispark.api.functions import min as mmin
from minispark.api.functions import sum as ssum
from minispark.api.session import MiniSparkSession
from minispark.logical.analyzer import AnalysisException


def make_session(master: str = "local[1]"):
    return MiniSparkSession.builder.master(master).app_name("group_by_test").get_or_create()


def _reference_group_by(records, key_fn):
    groups: dict = collections.defaultdict(list)
    for r in records:
        groups[key_fn(r)].append(r)
    return groups


def test_group_by_count_sum_avg_local1():
    session = make_session("local[1]")
    records = [
        {"country": "US", "revenue": 10},
        {"country": "CA", "revenue": 3},
        {"country": "US", "revenue": 20},
        {"country": "US", "revenue": 5},
        {"country": "UK", "revenue": 7},
        {"country": "CA", "revenue": 2},
    ]
    df = session.create_dataframe(records, num_partitions=3)
    result = df.group_by("country").agg(
        count("*").alias("n"), ssum("revenue").alias("total"), avg("revenue").alias("avg_rev")
    )
    rows = {r["country"]: r for r in result.collect()}

    reference = _reference_group_by(records, key_fn=lambda r: r["country"])
    for country, group_rows in reference.items():
        assert rows[country]["n"] == len(group_rows)
        assert rows[country]["total"] == sum(r["revenue"] for r in group_rows)
        assert rows[country]["avg_rev"] == pytest.approx(
            sum(r["revenue"] for r in group_rows) / len(group_rows)
        )


def test_group_by_min_max():
    session = make_session("local[1]")
    records = [
        {"country": "US", "revenue": 10},
        {"country": "US", "revenue": 55},
        {"country": "US", "revenue": 1},
        {"country": "CA", "revenue": 8},
    ]
    df = session.create_dataframe(records, num_partitions=2)
    result = df.group_by("country").agg(mmin("revenue").alias("lo"), mmax("revenue").alias("hi"))
    rows = {r["country"]: r for r in result.collect()}
    assert rows["US"] == {"country": "US", "lo": 1, "hi": 55}
    assert rows["CA"] == {"country": "CA", "lo": 8, "hi": 8}


def test_group_by_with_filter_before_it():
    session = make_session("local[1]")
    records = [
        {"country": "US", "age": 30, "revenue": 10},
        {"country": "US", "age": 15, "revenue": 999},  # filtered out
        {"country": "CA", "age": 40, "revenue": 3},
    ]
    df = session.create_dataframe(records, num_partitions=2)
    result = (
        df.filter(col("age") >= 18)
        .group_by("country")
        .agg(ssum("revenue").alias("total"))
    )
    rows = {r["country"]: r["total"] for r in result.collect()}
    assert rows == {"US": 10, "CA": 3}


def test_group_by_real_multiprocessing_matches_reference():
    session = make_session("local[3]")
    countries = ["US", "CA", "UK", "DE"]
    records = [
        {"country": countries[i % len(countries)], "revenue": i}
        for i in range(40)
    ]
    df = session.create_dataframe(records, num_partitions=5)
    result = df.group_by("country").agg(
        count("*").alias("n"), ssum("revenue").alias("total")
    )
    rows = {r["country"]: r for r in result.collect()}

    reference = _reference_group_by(records, key_fn=lambda r: r["country"])
    assert set(rows.keys()) == set(reference.keys())
    for country, group_rows in reference.items():
        assert rows[country]["n"] == len(group_rows)
        assert rows[country]["total"] == sum(r["revenue"] for r in group_rows)


def test_group_by_on_csv_source(tmp_path):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "country,revenue\nUS,10\nCA,3\nUS,20\nUK,7\nCA,2\n", encoding="utf-8"
    )
    session = make_session("local[2]")
    df = session.read.csv(str(csv_path), num_partitions=2)
    result = df.group_by("country").agg(ssum("revenue").alias("total"))
    rows = {r["country"]: r["total"] for r in result.collect()}
    assert rows == {"US": 30, "CA": 5, "UK": 7}


def test_explain_optimized_shows_both_shuffle_stages(capsys):
    session = make_session("local[1]")
    df = session.create_dataframe([{"country": "US", "revenue": 1}], num_partitions=1)
    df.group_by("country").agg(count("*").alias("n")).explain(optimized=True)
    out = capsys.readouterr().out
    assert "Aggregate[groupBy=(country)" in out
    assert "HashAggregateExec[partial]" in out
    assert "HashAggregateExec[final]" in out
    assert "Exchange[hash(country)" in out
    assert "ShuffleWriteExec[hash(country)" in out
    assert "ShuffleReadExec[stage 0]" in out
    assert "Stage 0 " in out
    assert "Stage 1 " in out


def test_group_by_unknown_column_raises_analysis_exception():
    session = make_session("local[1]")
    df = session.create_dataframe([{"country": "US", "revenue": 1}], num_partitions=1)
    with pytest.raises(AnalysisException, match="does_not_exist"):
        df.group_by("does_not_exist").agg(count("*").alias("n")).collect()


def test_agg_requires_at_least_one_aggregate():
    session = make_session("local[1]")
    df = session.create_dataframe([{"country": "US", "revenue": 1}], num_partitions=1)
    with pytest.raises(ValueError, match="agg\\(\\) requires"):
        df.group_by("country").agg()


def test_group_by_requires_at_least_one_column():
    session = make_session("local[1]")
    df = session.create_dataframe([{"country": "US", "revenue": 1}], num_partitions=1)
    with pytest.raises(ValueError, match="group_by\\(\\) requires"):
        df.group_by()
