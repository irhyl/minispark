"""Unit tests for CSVDataSource.read(columns=...): Milestone 7's real (if
partial) projection pruning, as opposed to Milestone 2's plan-shape-only
version. See storage/csv.py's module docstring for exactly what "real"
means here (skips the per-value parse work for unrequested columns, not
the underlying csv.reader tokenizing of every field on every line).
"""

from __future__ import annotations

from minispark.storage.csv import CSVDataSource


def write_csv(path, rows_text):
    path.write_text(rows_text, encoding="utf-8")
    return str(path)


def test_columns_narrows_schema_and_rows(tmp_path):
    csv_path = write_csv(
        tmp_path / "users.csv",
        "name,age,country\nalice,30,US\nbob,17,CA\n",
    )
    dataset = CSVDataSource(csv_path, num_partitions=1).read(columns=["name", "country"])
    assert dataset.schema.field_names() == ["name", "country"]
    rows = list(dataset.iter_records())
    assert rows == [
        {"name": "alice", "country": "US"},
        {"name": "bob", "country": "CA"},
    ]


def test_columns_none_returns_every_column(tmp_path):
    csv_path = write_csv(tmp_path / "users.csv", "name,age\nalice,30\n")
    dataset = CSVDataSource(csv_path, num_partitions=1).read()
    assert dataset.schema.field_names() == ["name", "age"]


def test_pruned_columns_work_across_multiple_partitions(tmp_path):
    csv_path = write_csv(
        tmp_path / "users.csv",
        "name,age,country\nalice,30,US\nbob,17,CA\ncarol,45,US\ndave,19,UK\n",
    )
    dataset = CSVDataSource(csv_path, num_partitions=2).read(columns=["age"])
    assert dataset.num_partitions() == 2
    rows = list(dataset.iter_records())
    assert all(set(r.keys()) == {"age"} for r in rows)
    assert sorted(r["age"] for r in rows) == [17, 19, 30, 45]


def test_filter_argument_is_accepted_but_does_not_change_output(tmp_path):
    """CSV has no statistics to skip rows against; a filter hint is
    accepted for interface uniformity with every DataSource but ignored,
    the row-level FilterExec above it remains the sole source of
    correctness (see storage/csv.py's module docstring)."""
    csv_path = write_csv(tmp_path / "users.csv", "name,age\nalice,30\nbob,17\n")
    from minispark.expressions.binary import GreaterThan
    from minispark.expressions.column import Column
    from minispark.expressions.literal import Literal

    dataset = CSVDataSource(csv_path, num_partitions=1).read(
        filter=GreaterThan(Column("age"), Literal(18))
    )
    rows = list(dataset.iter_records())
    assert len(rows) == 2  # filter was not applied by the source
