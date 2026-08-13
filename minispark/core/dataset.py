"""Dataset: an ordered collection of Partitions sharing one Schema.

Dataset is the runtime data structure that flows through the naive
executor in Milestone 1, and will later flow through physical operators
(Milestone 2+) as the thing tasks (Milestone 3) actually operate on.
It is deliberately NOT the same object as the lazy `DataFrame` in
`minispark.api.dataframe` — DataFrame wraps a *logical plan* and builds
one of these only when an action runs.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator

from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.core.schema import Schema


class Dataset:
    def __init__(self, schema: Schema, partitions: list[Partition]):
        self.schema = schema
        self._partitions = list(partitions)

    def num_partitions(self) -> int:
        return len(self._partitions)

    def partition(self, i: int) -> Partition:
        return self._partitions[i]

    def partitions(self) -> list[Partition]:
        return list(self._partitions)

    def repartition(self, n: int) -> Dataset:
        """Redistribute rows into `n` partitions, round-robin.

        Milestone-1 limitation: this materializes all rows in memory to
        redistribute them, because round-robin assignment needs to see
        every row before it knows the new partition boundaries. A
        streaming-friendly version (e.g. consistent hashing over a row
        index, or accepting a target partition *size* instead of count)
        is possible but not implemented yet — flagged here rather than
        silently pretending this scales to unbounded data.
        """
        if n < 1:
            raise ValueError(f"repartition target must be >= 1, got {n}")
        buckets: list[list[Record]] = [[] for _ in range(n)]
        for i, record in enumerate(self.iter_records()):
            buckets[i % n].append(record)

        # functools.partial(iter, rows), not `lambda: iter(rows)`: a lambda
        # is not picklable by the standard library `pickle` module (used by
        # `multiprocessing`) regardless of what it closes over. See
        # storage/memory.py's `_make_records_fn` for the same fix.
        new_partitions = [
            Partition(
                partition_id=i,
                schema=self.schema,
                records_fn=functools.partial(iter, rows),
                metadata=PartitionMetadata(row_count=len(rows)),
            )
            for i, rows in enumerate(buckets)
        ]
        return Dataset(self.schema, new_partitions)

    def iter_records(self) -> Iterator[Record]:
        """Iterate every record across every partition, in order."""
        for p in self._partitions:
            yield from p

    def row_count(self) -> int:
        """Exact row count. Uses partition metadata where known, otherwise scans."""
        total = 0
        for p in self._partitions:
            rc = p.row_count()
            total += rc if rc is not None else sum(1 for _ in p)
        return total

    def __repr__(self) -> str:
        return f"Dataset(schema={self.schema}, partitions={len(self._partitions)})"
