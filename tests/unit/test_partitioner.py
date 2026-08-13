import subprocess
import sys

import pytest

from minispark.shuffle.partitioner import HashPartitioner, RangePartitioner


def test_hash_partitioner_stays_in_range():
    p = HashPartitioner(4)
    for key in [("US",), ("CA",), ("UK",), (1, "x"), (None,)]:
        target = p.partition_for(key)
        assert 0 <= target < 4


def test_hash_partitioner_same_key_same_partition():
    p = HashPartitioner(8)
    assert p.partition_for(("US",)) == p.partition_for(("US",))


def test_hash_partitioner_different_keys_can_differ():
    p = HashPartitioner(8)
    targets = {p.partition_for((k,)) for k in ["US", "CA", "UK", "DE", "FR", "JP"]}
    # Not a strict requirement of hashing, but with 8 buckets and 6 distinct
    # keys, seeing every key land in the same bucket would indicate a
    # broken (constant) hash function, not bad luck.
    assert len(targets) > 1


def test_hash_partitioner_rejects_zero_partitions():
    with pytest.raises(ValueError):
        HashPartitioner(0)


def test_hash_partitioner_is_stable_across_processes():
    """The whole point of avoiding the builtin hash() (see partitioner.py's
    _stable_hash) is that two different worker processes must agree on a
    key's target partition. Spawn a real second process and check its
    answer matches this one's."""
    p = HashPartitioner(6)
    here = p.partition_for(("US", 1))
    script = (
        "from minispark.shuffle.partitioner import HashPartitioner;"
        "print(HashPartitioner(6).partition_for(('US', 1)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=True
    )
    assert int(out.stdout.strip()) == here


def test_range_partitioner_assigns_contiguous_ranges():
    p = RangePartitioner(3, boundaries=[10, 20])
    assert p.partition_for(5) == 0
    assert p.partition_for(10) == 1
    assert p.partition_for(15) == 1
    assert p.partition_for(20) == 2
    assert p.partition_for(100) == 2


def test_range_partitioner_rejects_wrong_boundary_count():
    with pytest.raises(ValueError):
        RangePartitioner(3, boundaries=[10])
