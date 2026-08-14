"""Unit tests for MemoryDataSource.read(columns=...): real column
pruning (smaller Record dicts flow through the rest of the engine), even
though there is no I/O to save since the data is already resident in
memory. See storage/memory.py's module docstring.
"""

from __future__ import annotations

from minispark.storage.memory import MemoryDataSource


def test_columns_narrows_schema_and_rows():
    records = [{"id": 1, "name": "a", "age": 30}, {"id": 2, "name": "b", "age": 17}]
    dataset = MemoryDataSource(records, num_partitions=2).read(columns=["id", "age"])
    assert dataset.schema.field_names() == ["id", "age"]
    rows = list(dataset.iter_records())
    assert rows == [{"id": 1, "age": 30}, {"id": 2, "age": 17}]


def test_columns_none_returns_every_column():
    records = [{"id": 1, "name": "a"}]
    dataset = MemoryDataSource(records, num_partitions=1).read()
    assert dataset.schema.field_names() == ["id", "name"]


def test_filter_argument_is_accepted_but_does_not_change_output():
    from minispark.expressions.binary import GreaterThan
    from minispark.expressions.column import Column
    from minispark.expressions.literal import Literal

    records = [{"age": 30}, {"age": 17}]
    dataset = MemoryDataSource(records, num_partitions=1).read(
        filter=GreaterThan(Column("age"), Literal(18))
    )
    assert len(list(dataset.iter_records())) == 2
