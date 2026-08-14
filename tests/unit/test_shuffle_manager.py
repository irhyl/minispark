import os

from minispark.shuffle.manager import ShuffleManager
from minispark.shuffle.partitioner import HashPartitioner
from minispark.shuffle.writer import write_shuffle_partition


def key_fn(record):
    return (record["country"],)


def test_manager_creates_a_scratch_directory():
    mgr = ShuffleManager()
    assert os.path.isdir(mgr.root_dir)
    mgr.cleanup()
    assert not os.path.exists(mgr.root_dir)


def test_blocks_for_returns_only_the_requested_partition():
    mgr = ShuffleManager()
    blocks = write_shuffle_partition(
        mgr.root_dir, stage_id=0, source_task_id=0,
        records=[{"country": "US", "revenue": 1}, {"country": "CA", "revenue": 2}],
        key_fn=key_fn, partitioner=HashPartitioner(4),
    )
    mgr.register_blocks(0, blocks)

    us_target = HashPartitioner(4).partition_for(("US",))
    us_blocks = mgr.blocks_for(0, us_target)
    assert all(b.target_partition == us_target for b in us_blocks)
    assert len(us_blocks) >= 1

    other_stage_blocks = mgr.blocks_for(1, us_target)
    assert other_stage_blocks == []

    mgr.cleanup()


def test_register_blocks_covers_every_source_task_in_one_call():
    """execution/scheduler.py always calls register_blocks() exactly once
    per stage, with every task's blocks already combined into one list
    (see LocalScheduler._run_stage); this is what that call looks like."""
    mgr = ShuffleManager()
    blocks_a = write_shuffle_partition(
        mgr.root_dir, 0, source_task_id=0,
        records=[{"country": "US", "revenue": 1}], key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    blocks_b = write_shuffle_partition(
        mgr.root_dir, 0, source_task_id=1,
        records=[{"country": "US", "revenue": 2}], key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    mgr.register_blocks(0, blocks_a + blocks_b)
    assert len(mgr.blocks_for(0, 0)) == 2
    mgr.cleanup()


def test_register_blocks_overwrites_a_stage_registered_twice():
    """A second register_blocks() call for the same stage_id must replace
    the first, not accumulate alongside it: this is what lets lineage-
    based recomputation (execution/scheduler.py's
    _try_recover_missing_shuffle) re-register a stage's fresh blocks after
    the original ones were found missing or corrupted, without the stale
    metadata sticking around and letting a later reader pick it again."""
    mgr = ShuffleManager()
    stale = write_shuffle_partition(
        mgr.root_dir, 0, source_task_id=0,
        records=[{"country": "US", "revenue": 1}], key_fn=key_fn,
        partitioner=HashPartitioner(1),
    )
    fresh = write_shuffle_partition(
        mgr.root_dir, 0, source_task_id=1,
        records=[{"country": "US", "revenue": 2}, {"country": "US", "revenue": 3}],
        key_fn=key_fn, partitioner=HashPartitioner(1),
    )
    mgr.register_blocks(0, stale)
    mgr.register_blocks(0, fresh)
    blocks = mgr.blocks_for(0, 0)
    assert len(blocks) == 1
    assert blocks[0].source_task_id == 1
    mgr.cleanup()


def test_metrics_for_sums_records_and_bytes_across_blocks():
    mgr = ShuffleManager()
    blocks = write_shuffle_partition(
        mgr.root_dir, 0, source_task_id=0,
        records=[{"country": "US", "revenue": 1}, {"country": "US", "revenue": 2}],
        key_fn=key_fn, partitioner=HashPartitioner(1),
    )
    mgr.register_blocks(0, blocks)
    metrics = mgr.metrics_for(0, 0)
    assert metrics.block_count == 1
    assert metrics.record_count == 2
    assert metrics.byte_length > 0
    mgr.cleanup()


def test_cleanup_is_safe_to_call_twice():
    mgr = ShuffleManager()
    mgr.cleanup()
    mgr.cleanup()  # must not raise
