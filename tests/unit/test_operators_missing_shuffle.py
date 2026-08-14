"""Unit tests for how physical/operators.py's _execute_shuffle_read_partition
translates a missing or corrupted shuffle block into a MissingShuffleDataError
that names the stage and target partition it came from, distinct from an
ordinary exception, which is what execution/worker.py and
execution/scheduler.py rely on to trigger lineage-based recomputation
(Milestone 6) instead of a plain in-place task retry.
"""

from __future__ import annotations

import os

import pytest

from minispark.core.schema import Field, Schema
from minispark.core.types import INT, STRING
from minispark.physical.operators import execute_partition
from minispark.physical.plan import ShuffleReadExec
from minispark.shuffle.partitioner import HashPartitioner
from minispark.shuffle.reader import MissingShuffleDataError
from minispark.shuffle.writer import write_shuffle_partition


def key_fn(record):
    return (record["country"],)


SCHEMA = Schema([Field("country", STRING), Field("revenue", INT)])


def test_missing_block_file_raises_missing_shuffle_data_error(tmp_path):
    blocks = write_shuffle_partition(
        root_dir=str(tmp_path),
        stage_id=3,
        source_task_id=0,
        records=[{"country": "US", "revenue": 1}],
        key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    for block in blocks:
        os.remove(block.path)

    plan = ShuffleReadExec(from_stage_id=3, schema=SCHEMA)
    with pytest.raises(MissingShuffleDataError) as exc_info:
        execute_partition(plan, partition_id=0, shuffle_blocks={3: blocks})

    assert exc_info.value.stage_id == 3
    assert exc_info.value.target_partition == 0


def test_corrupted_block_raises_missing_shuffle_data_error(tmp_path):
    blocks = write_shuffle_partition(
        root_dir=str(tmp_path),
        stage_id=5,
        source_task_id=0,
        records=[{"country": "US", "revenue": 1}],
        key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    (block,) = blocks
    with open(block.path, "ab") as f:
        f.write(b"corruption")

    plan = ShuffleReadExec(from_stage_id=5, schema=SCHEMA)
    with pytest.raises(MissingShuffleDataError) as exc_info:
        execute_partition(plan, partition_id=0, shuffle_blocks={5: blocks})

    assert exc_info.value.stage_id == 5


def test_healthy_blocks_do_not_raise(tmp_path):
    blocks = write_shuffle_partition(
        root_dir=str(tmp_path),
        stage_id=1,
        source_task_id=0,
        records=[{"country": "US", "revenue": 1}],
        key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    plan = ShuffleReadExec(from_stage_id=1, schema=SCHEMA)
    partition = execute_partition(plan, partition_id=0, shuffle_blocks={1: blocks})
    assert partition.to_list() == [{"country": "US", "revenue": 1}]
