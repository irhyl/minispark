"""Partition: an independently processable slice of a Dataset.

A Partition does not hold materialized rows. It holds a zero-argument
factory that *produces* an iterator over Records on demand. This gives us
two things for free, ahead of when we actually need them:

  * streaming semantics — `partition -> operator -> partition` never
    requires the whole partition (let alone the whole dataset) to be
    resident in memory at once, per the resource-safety requirement.
  * re-computability — calling the factory again re-derives the same
    rows from the same source. That is the seed of lineage-based fault
    tolerance (Milestone 6): a lost partition can be recomputed by
    re-invoking its factory chain instead of being replicated.

The cost: some sources (e.g. a network stream) cannot honor "call the
factory twice." Milestone 1 only has file/in-memory sources, so this
tradeoff is free for now; it is revisited when lineage is implemented.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from minispark.core.record import Record
from minispark.core.schema import Schema


@dataclass
class PartitionMetadata:
    row_count: int | None = None
    estimated_size_bytes: int | None = None
    location: str | None = None


class Partition:
    def __init__(
        self,
        partition_id: int,
        schema: Schema,
        records_fn: Callable[[], Iterator[Record]],
        metadata: PartitionMetadata | None = None,
    ):
        self.partition_id = partition_id
        self.schema = schema
        self._records_fn = records_fn
        self.metadata = metadata or PartitionMetadata()

    def __iter__(self) -> Iterator[Record]:
        return iter(self._records_fn())

    def row_count(self) -> int | None:
        """Known row count, or None if it would require a full scan to know."""
        return self.metadata.row_count

    def estimated_size_bytes(self) -> int | None:
        return self.metadata.estimated_size_bytes

    def to_list(self) -> list[Record]:
        """Materialize this partition's records into a list. Use sparingly."""
        return list(self)

    def __repr__(self) -> str:
        return (
            f"Partition(id={self.partition_id}, rows={self.metadata.row_count}, "
            f"location={self.metadata.location!r})"
        )
