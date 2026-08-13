"""Partitioners: decide which target partition a shuffle key belongs to.

`HashPartitioner` is what group-by aggregation actually uses (see
physical/planner.py): it needs no information about the data ahead of
time, just a target partition count, and it guarantees every row sharing
a key lands in the same target partition, which is the property a
reduce-side aggregate depends on.

`RangePartitioner` exists because the build spec explicitly asks for it
("Implement HashPartitioner, RangePartitioner initially") but has no
consumer yet: it is the right partitioner for a total-order sort
(assigning contiguous key ranges to contiguous target partitions so a
per-partition local sort plus partition ordering gives a globally sorted
result), which is Milestone 5's Sort, not Milestone 4's GroupBy. It is
implemented and unit-tested standalone now so Milestone 5 does not have
to build it under time pressure alongside Sort itself.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from bisect import bisect_right
from collections.abc import Sequence
from typing import Any


class Partitioner(ABC):
    def __init__(self, num_partitions: int):
        if num_partitions < 1:
            raise ValueError(f"num_partitions must be >= 1, got {num_partitions}")
        self.num_partitions = num_partitions

    @abstractmethod
    def partition_for(self, key: Any) -> int:
        """Which target partition (in `range(self.num_partitions)`) `key` belongs to."""


class HashPartitioner(Partitioner):
    """`hash(key) % num_partitions`. `key` must be hashable, which is why
    Aggregate's group_by keys are built as tuples of plain scalar values
    (str/int/float/bool/None), never lists or dicts.
    """

    def partition_for(self, key: Any) -> int:
        # Python's hash() is randomized per-process for strings by default
        # (PYTHONHASHSEED), which would be disastrous here: two different
        # worker processes hashing the *same* key must agree on its target
        # partition, or rows for one group would scatter across multiple
        # reduce partitions. `hash()` is intentionally not used; a stable,
        # process-independent hash is computed instead.
        return _stable_hash(key) % self.num_partitions


class RangePartitioner(Partitioner):
    """Assigns contiguous key ranges to contiguous target partitions, given
    pre-computed sorted boundary keys (e.g. sampled from the data).

    `boundaries` must have `num_partitions - 1` entries, sorted ascending;
    `boundaries[i]` is the smallest key that belongs to partition `i + 1`
    rather than partition `i`. No physical operator builds a
    RangePartitioner yet (see module docstring); computing `boundaries`
    from a real dataset (e.g. by sampling) is Milestone 5's job, alongside
    Sort.
    """

    def __init__(self, num_partitions: int, boundaries: Sequence[Any]):
        super().__init__(num_partitions)
        if len(boundaries) != num_partitions - 1:
            raise ValueError(
                f"RangePartitioner needs exactly {num_partitions - 1} boundaries "
                f"for {num_partitions} partitions, got {len(boundaries)}"
            )
        self.boundaries = list(boundaries)

    def partition_for(self, key: Any) -> int:
        return bisect_right(self.boundaries, key)


def _stable_hash(key: Any) -> int:
    """A hash of `key` that is the same across processes and runs.

    `key` is always a tuple of plain scalars (see HashPartitioner's
    docstring), so `repr()` round-trips it into a deterministic string,
    and hashing *that string's bytes* with a fixed, non-randomized
    algorithm (`hashlib`, not the builtin `hash()`) sidesteps
    PYTHONHASHSEED entirely.
    """
    digest = hashlib.md5(repr(key).encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:8], byteorder="big")
