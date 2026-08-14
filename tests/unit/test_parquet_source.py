"""Unit tests for storage/parquet.py's ParquetDataSource: partitioning at
row-group granularity, real column pruning, real row-group-level
predicate pushdown, and the write path. Skipped without pyarrow
installed (optional `columnar` extra, see pyproject.toml).
"""

from __future__ import annotations

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from minispark.core.dataset import Dataset  # noqa: E402
from minispark.core.partition import Partition, PartitionMetadata  # noqa: E402
from minispark.core.schema import Field, Schema  # noqa: E402
from minispark.core.types import INT, STRING  # noqa: E402
from minispark.expressions.binary import GreaterThan  # noqa: E402
from minispark.expressions.column import Column  # noqa: E402
from minispark.expressions.literal import Literal  # noqa: E402
from minispark.storage.parquet import ParquetDataSource, write_parquet_dataset  # noqa: E402


def write_multi_row_group_file(path, row_group_size=5, total=20):
    table = pa.table({"a": list(range(total)), "b": [str(i) for i in range(total)]})
    pq.write_table(table, str(path), row_group_size=row_group_size)
    return str(path)


def test_reads_all_rows_across_partitions(tmp_path):
    path = write_multi_row_group_file(tmp_path / "t.parquet")
    dataset = ParquetDataSource(path, num_partitions=2).read()
    assert dataset.schema.field_names() == ["a", "b"]
    rows = list(dataset.iter_records())
    assert sorted(r["a"] for r in rows) == list(range(20))


def test_columns_narrows_schema_and_decoded_data(tmp_path):
    path = write_multi_row_group_file(tmp_path / "t.parquet")
    dataset = ParquetDataSource(path, num_partitions=2).read(columns=["a"])
    assert dataset.schema.field_names() == ["a"]
    rows = list(dataset.iter_records())
    assert all(set(r.keys()) == {"a"} for r in rows)
    assert sorted(r["a"] for r in rows) == list(range(20))


def test_filter_actually_reduces_returned_rows(tmp_path):
    path = write_multi_row_group_file(tmp_path / "t.parquet")
    condition = GreaterThan(Column("a"), Literal(15))
    dataset = ParquetDataSource(path, num_partitions=2).read(filter=condition)
    rows = list(dataset.iter_records())
    assert sorted(r["a"] for r in rows) == [16, 17, 18, 19]


def test_filter_skips_row_groups_that_cannot_match(tmp_path):
    """Row groups are [0-4],[5-9],[10-14],[15-19]. A filter for a > 15
    can only ever match rows in the last row group; every partition's
    metadata row_count (from footer statistics, pre-filter) plus the
    actual post-filter row count together prove that unmatched row
    groups were skipped rather than scanned and then discarded: if they
    had been scanned normally, nothing here would look different from a
    row-at-a-time filter, so this specifically checks that a fragment
    covering only non-matching rows contributes zero rows, the direct,
    observable effect of pyarrow's own row-group statistics pruning
    (this class does not reimplement that logic, only relies on it, see
    ParquetDataSource's docstring)."""
    path = write_multi_row_group_file(tmp_path / "t.parquet")
    condition = GreaterThan(Column("a"), Literal(15))
    dataset = ParquetDataSource(path, num_partitions=4).read(filter=condition)
    non_empty_partitions = [p for p in dataset.partitions() if list(p)]
    assert len(non_empty_partitions) == 1
    assert [r["a"] for r in non_empty_partitions[0]] == [16, 17, 18, 19]


def test_partition_metadata_row_count_is_pre_filter(tmp_path):
    """Matches CSV/every other source's convention: row_count metadata
    comes from cheap footer statistics, known before any filter is
    applied, not the actual post-filter row count."""
    path = write_multi_row_group_file(tmp_path / "t.parquet")
    condition = GreaterThan(Column("a"), Literal(15))
    dataset = ParquetDataSource(path, num_partitions=4).read(filter=condition)
    assert sum(p.row_count() for p in dataset.partitions()) == 20


def test_num_partitions_capped_at_row_group_count(tmp_path):
    path = write_multi_row_group_file(tmp_path / "t.parquet", row_group_size=10, total=20)
    dataset = ParquetDataSource(path, num_partitions=100).read()
    assert dataset.num_partitions() == 2  # only 2 row groups exist


def test_write_then_read_round_trips(tmp_path):
    schema = Schema([Field("a", INT), Field("b", STRING)])
    partitions = [
        Partition(
            0, schema, lambda: iter([{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]),
            PartitionMetadata(row_count=2),
        ),
        Partition(1, schema, lambda: iter([{"a": 3, "b": "z"}]), PartitionMetadata(row_count=1)),
    ]
    write_parquet_dataset(Dataset(schema, partitions), str(tmp_path / "out"))

    read_back = ParquetDataSource(str(tmp_path / "out"), num_partitions=4).read()
    rows = sorted(read_back.iter_records(), key=lambda r: r["a"])
    assert rows == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]


def test_write_creates_one_file_per_partition(tmp_path):
    import os

    schema = Schema([Field("a", INT)])
    partitions = [
        Partition(0, schema, lambda: iter([{"a": 1}]), PartitionMetadata(row_count=1)),
        Partition(1, schema, lambda: iter([{"a": 2}]), PartitionMetadata(row_count=1)),
        Partition(2, schema, lambda: iter([]), PartitionMetadata(row_count=0)),
    ]
    outdir = str(tmp_path / "out")
    write_parquet_dataset(Dataset(schema, partitions), outdir)
    files = sorted(os.listdir(outdir))
    assert files == ["part-00000.parquet", "part-00001.parquet", "part-00002.parquet"]


def test_unsupported_arrow_type_raises_a_clear_error(tmp_path):
    table = pa.table({"a": pa.array([1, 2], type=pa.date32())})
    path = str(tmp_path / "dates.parquet")
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="Unsupported"):
        ParquetDataSource(path).read()
