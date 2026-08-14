"""Unit tests for storage/checkpoint.py: writing a Dataset to a checkpoint
directory and reading it back must round-trip rows, schema, and partition
count exactly, since this is what DataFrame.checkpoint() (api/dataframe.py)
relies on to make a plain Scan stand in for whatever plan produced the
checkpointed data.
"""

from __future__ import annotations

import pytest

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.storage.checkpoint import CheckpointDataSource, write_checkpoint

SCHEMA = Schema([Field("country", STRING), Field("revenue", INT)])


def make_dataset(rows_per_partition: list[list[dict]]) -> Dataset:
    partitions = [
        Partition(i, SCHEMA, lambda rows=rows: iter(rows), PartitionMetadata(row_count=len(rows)))
        for i, rows in enumerate(rows_per_partition)
    ]
    return Dataset(SCHEMA, partitions)


def test_round_trips_rows_across_partitions(tmp_path):
    dataset = make_dataset(
        [
            [{"country": "US", "revenue": 1}, {"country": "US", "revenue": 2}],
            [{"country": "CA", "revenue": 3}],
        ]
    )
    write_checkpoint(dataset, str(tmp_path))
    checkpointed = CheckpointDataSource(str(tmp_path)).read()

    assert checkpointed.num_partitions() == 2
    assert list(checkpointed.iter_records()) == [
        {"country": "US", "revenue": 1},
        {"country": "US", "revenue": 2},
        {"country": "CA", "revenue": 3},
    ]


def test_preserves_schema(tmp_path):
    dataset = make_dataset([[{"country": "US", "revenue": 1}]])
    write_checkpoint(dataset, str(tmp_path))
    checkpointed = CheckpointDataSource(str(tmp_path)).read()
    assert checkpointed.schema == SCHEMA


def test_partition_row_counts_are_exact(tmp_path):
    dataset = make_dataset([[{"country": "US", "revenue": 1}] * 3, []])
    write_checkpoint(dataset, str(tmp_path))
    checkpointed = CheckpointDataSource(str(tmp_path)).read()
    assert checkpointed.partition(0).row_count() == 3
    assert checkpointed.partition(1).row_count() == 0


def test_records_fn_is_repeatable_not_a_one_shot_iterator(tmp_path):
    """Partition.__iter__ must be callable more than once (core/partition.py's
    re-computability contract): reading a checkpointed partition twice
    must produce the same rows both times, not an empty second read."""
    dataset = make_dataset([[{"country": "US", "revenue": 1}]])
    write_checkpoint(dataset, str(tmp_path))
    checkpointed = CheckpointDataSource(str(tmp_path)).read()
    partition = checkpointed.partition(0)
    assert list(partition) == list(partition)


def test_missing_meta_file_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Not a checkpoint directory"):
        CheckpointDataSource(str(tmp_path))


def test_write_checkpoint_overwrites_an_existing_checkpoint(tmp_path):
    write_checkpoint(make_dataset([[{"country": "US", "revenue": 1}]]), str(tmp_path))
    write_checkpoint(
        make_dataset([[{"country": "CA", "revenue": 2}], [{"country": "UK", "revenue": 3}]]),
        str(tmp_path),
    )
    checkpointed = CheckpointDataSource(str(tmp_path)).read()
    assert checkpointed.num_partitions() == 2
    assert sorted(r["country"] for r in checkpointed.iter_records()) == ["CA", "UK"]
