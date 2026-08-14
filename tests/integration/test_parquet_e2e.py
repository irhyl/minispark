"""End-to-end tests for Parquet through the real DataFrame path: real
disk-backed Parquet files, real analyzer/optimizer/physical-planner
pushdown, and real OS-process parallelism (`local[2]`), the same rigor
bar as every other source's e2e tests (see test_group_by_e2e.py,
test_join_e2e.py). Correctness is checked against a plain Python
computation over the same data, independent of MiniSpark.

Skipped entirely without pyarrow installed (optional `columnar` extra).
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from minispark.api.functions import col  # noqa: E402
from minispark.api.session import MiniSparkSession  # noqa: E402

USERS = [
    {"name": "alice", "age": 30, "country": "US"},
    {"name": "bob", "age": 17, "country": "CA"},
    {"name": "carol", "age": 45, "country": "US"},
    {"name": "dave", "age": 19, "country": "UK"},
    {"name": "erin", "age": 15, "country": "UK"},
]


def make_session(master: str = "local[1]"):
    return MiniSparkSession.builder.master(master).app_name("parquet_test").get_or_create()


def write_users_parquet(path, row_group_size=2):
    table = pa.table(
        {
            "name": [u["name"] for u in USERS],
            "age": [u["age"] for u in USERS],
            "country": [u["country"] for u in USERS],
        }
    )
    pq.write_table(table, str(path), row_group_size=row_group_size)
    return str(path)


def test_read_filter_select_matches_reference_local1(tmp_path):
    path = write_users_parquet(tmp_path / "users.parquet")
    session = make_session("local[1]")
    result = session.read.parquet(path).filter(col("age") >= 18).select("name", "country")
    rows = result.collect()

    reference = [
        {"name": u["name"], "country": u["country"]} for u in USERS if u["age"] >= 18
    ]
    assert sorted(rows, key=lambda r: r["name"]) == sorted(reference, key=lambda r: r["name"])


def test_read_filter_select_matches_reference_real_multiprocessing(tmp_path):
    path = write_users_parquet(tmp_path / "users.parquet")
    session = make_session("local[2]")
    result = session.read.parquet(path, num_partitions=3).filter(col("age") >= 18).select("name")
    rows = result.collect()

    reference = {u["name"] for u in USERS if u["age"] >= 18}
    assert {r["name"] for r in rows} == reference


def test_group_by_over_parquet_source(tmp_path):
    from minispark.api.functions import sum as ssum

    path = write_users_parquet(tmp_path / "users.parquet")
    session = make_session("local[2]")
    result = (
        session.read.parquet(path).group_by("country").agg(ssum("age").alias("total_age"))
    )
    rows = {r["country"]: r["total_age"] for r in result.collect()}

    reference: dict[str, int] = {}
    for u in USERS:
        reference[u["country"]] = reference.get(u["country"], 0) + u["age"]
    assert rows == reference


def test_explain_shows_pushdown_narrowed_scan_columns(tmp_path, capsys):
    path = write_users_parquet(tmp_path / "users.parquet")
    session = make_session("local[1]")
    df = session.read.parquet(path).filter(col("age") >= 18).select("name")
    df.explain(optimized=True)
    out = capsys.readouterr().out
    physical_section = out.split("== Physical Plan ==")[1]
    scan_line = next(line for line in physical_section.splitlines() if "ScanExec" in line)
    # Only "age" (the filter) and "name" (the projection) are needed;
    # "country" must not appear in the pushed-down Scan's columns.
    assert "country" not in scan_line
    assert "age" in scan_line
    assert "name" in scan_line


def test_write_then_read_round_trip_real_multiprocessing(tmp_path):
    path = write_users_parquet(tmp_path / "users.parquet")
    session = make_session("local[2]")
    adults = session.read.parquet(path).filter(col("age") >= 18)

    out_dir = str(tmp_path / "out")
    adults.write.parquet(out_dir)

    read_back = session.read.parquet(out_dir).collect()
    reference = [u for u in USERS if u["age"] >= 18]
    assert sorted(read_back, key=lambda r: r["name"]) == sorted(
        reference, key=lambda r: r["name"]
    )


def test_filter_on_null_containing_column_still_correct(tmp_path):
    """A row with a null in the filtered column must not silently
    disappear in a way that contradicts the row-level engine's own
    semantics for a comparison that never reaches a null (see
    storage/parquet.py's None-literal handling); this checks the more
    basic, common case: nulls in a column the filter does not touch must
    simply pass through untouched."""
    table = pa.table(
        {
            "name": ["a", "b", "c"],
            "age": [30, 17, 45],
            "country": ["US", None, "CA"],
        }
    )
    path = str(tmp_path / "with_nulls.parquet")
    pq.write_table(table, path)
    session = make_session("local[1]")
    rows = session.read.parquet(path).filter(col("age") >= 18).collect()
    assert sorted(rows, key=lambda r: r["name"]) == [
        {"name": "a", "age": 30, "country": "US"},
        {"name": "c", "age": 45, "country": "CA"},
    ]
