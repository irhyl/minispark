import pytest

from minispark.shuffle.partitioner import HashPartitioner
from minispark.shuffle.reader import ShuffleChecksumError, read_shuffle_blocks
from minispark.shuffle.writer import write_shuffle_partition


def key_fn(record):
    return (record["country"],)


def test_write_then_read_round_trips_all_records(tmp_path):
    records = [
        {"country": "US", "revenue": 10},
        {"country": "CA", "revenue": 3},
        {"country": "US", "revenue": 20},
    ]
    partitioner = HashPartitioner(4)
    blocks = write_shuffle_partition(
        root_dir=str(tmp_path),
        stage_id=0,
        source_task_id=0,
        records=records,
        key_fn=key_fn,
        partitioner=partitioner,
    )
    read_back = list(read_shuffle_blocks(blocks))
    assert sorted(read_back, key=lambda r: (r["country"], r["revenue"])) == sorted(
        records, key=lambda r: (r["country"], r["revenue"])
    )


def test_partitioning_groups_same_key_records_together(tmp_path):
    """Every "US" row must land in the same target partition. With only a
    few buckets, a different key (here "CA") may legitimately hash into
    that same bucket too (a real collision, not a bug), so this checks
    "both US rows are together," not "US has a block all to itself."
    """
    records = [
        {"country": "US", "revenue": 10},
        {"country": "CA", "revenue": 3},
        {"country": "US", "revenue": 20},
    ]
    partitioner = HashPartitioner(4)
    blocks = write_shuffle_partition(
        root_dir=str(tmp_path),
        stage_id=0,
        source_task_id=0,
        records=records,
        key_fn=key_fn,
        partitioner=partitioner,
    )
    us_target = partitioner.partition_for(("US",))
    us_block = next(b for b in blocks if b.target_partition == us_target)
    us_rows = [r for r in read_shuffle_blocks([us_block]) if r["country"] == "US"]
    assert len(us_rows) == 2
    assert {r["revenue"] for r in us_rows} == {10, 20}


def test_no_block_written_for_a_partition_with_no_records(tmp_path):
    partitioner = HashPartitioner(100)  # far more buckets than records
    blocks = write_shuffle_partition(
        root_dir=str(tmp_path),
        stage_id=0,
        source_task_id=0,
        records=[{"country": "US", "revenue": 1}],
        key_fn=key_fn,
        partitioner=partitioner,
    )
    assert len(blocks) == 1  # only the one target partition that got a row


def test_block_metadata_has_exact_record_count_and_positive_byte_length(tmp_path):
    blocks = write_shuffle_partition(
        root_dir=str(tmp_path),
        stage_id=0,
        source_task_id=0,
        records=[{"country": "US", "revenue": 1}, {"country": "US", "revenue": 2}],
        key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    (block,) = blocks
    assert block.record_count == 2
    assert block.byte_length > 0
    assert len(block.checksum) == 32  # hex md5 digest


def test_corrupted_block_fails_checksum_verification(tmp_path):
    blocks = write_shuffle_partition(
        root_dir=str(tmp_path),
        stage_id=0,
        source_task_id=0,
        records=[{"country": "US", "revenue": 1}],
        key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    (block,) = blocks
    with open(block.path, "ab") as f:
        f.write(b"corruption")
    with pytest.raises(ShuffleChecksumError):
        list(read_shuffle_blocks([block]))


def test_reading_with_verify_checksum_false_ignores_corruption(tmp_path):
    blocks = write_shuffle_partition(
        root_dir=str(tmp_path),
        stage_id=0,
        source_task_id=0,
        records=[{"country": "US", "revenue": 1}],
        key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    (block,) = blocks
    rows = list(read_shuffle_blocks([block], verify_checksum=False))
    assert rows == [{"country": "US", "revenue": 1}]


def test_reading_multiple_blocks_for_one_partition_concatenates_them(tmp_path):
    blocks_a = write_shuffle_partition(
        str(tmp_path), 0, source_task_id=0,
        records=[{"country": "US", "revenue": 1}], key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    blocks_b = write_shuffle_partition(
        str(tmp_path), 0, source_task_id=1,
        records=[{"country": "US", "revenue": 2}], key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    rows = list(read_shuffle_blocks(blocks_a + blocks_b))
    assert sorted(r["revenue"] for r in rows) == [1, 2]


def test_preserves_tuple_types_that_json_would_have_turned_into_lists(tmp_path):
    """Avg's partial state is a (sum, count) tuple (expressions/aggregate.py).
    This is the whole reason blocks are pickled records, not JSON lines."""
    records = [{"country": "US", "__agg_state_0": (30, 2)}]
    blocks = write_shuffle_partition(
        str(tmp_path), 0, source_task_id=0, records=records,
        key_fn=key_fn, partitioner=HashPartitioner(1),
    )
    (row,) = list(read_shuffle_blocks(blocks))
    assert isinstance(row["__agg_state_0"], tuple)
    assert row["__agg_state_0"] == (30, 2)
