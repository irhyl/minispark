"""In-memory DataSource: wraps a Python list[dict] as a Dataset.

Used for unit tests and small examples where reading a real file would be
noise. Chunks rows contiguously into `num_partitions` partitions; unlike
CSVDataSource, records are already in memory so there is no streaming
benefit to be had from a lazier partition factory here.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.core.schema import Field, Schema
from minispark.core.types import infer_type
from minispark.storage.datasource import DataSource


def infer_schema(records: list[Record]) -> Schema:
    if not records:
        raise ValueError("Cannot infer schema from an empty list of records")
    first = records[0]
    fields = [Field(k, infer_type(v), nullable=v is None) for k, v in first.items()]
    return Schema(fields)


class MemoryDataSource(DataSource):
    def __init__(
        self,
        records: list[Record],
        schema: Schema | None = None,
        num_partitions: int = 4,
    ):
        self._records = records
        self._schema = schema or infer_schema(records)
        self._num_partitions = max(1, min(num_partitions, len(records) or 1))

    @property
    def name(self) -> str:
        return "memory"

    def read(self) -> Dataset:
        n = self._num_partitions
        total = len(self._records)
        chunk_size = -(-total // n) if n else total  # ceil division
        partitions = []
        for i in range(n):
            start, end = i * chunk_size, min((i + 1) * chunk_size, total)
            if start >= end and total > 0:
                continue
            rows = self._records[start:end]
            partitions.append(
                Partition(
                    partition_id=i,
                    schema=self._schema,
                    records_fn=_make_records_fn(rows),
                    metadata=PartitionMetadata(row_count=len(rows)),
                )
            )
        if not partitions:
            partitions = [
                Partition(0, self._schema, _make_records_fn([]), PartitionMetadata(row_count=0))
            ]
        return Dataset(self._schema, partitions)


def _make_records_fn(rows: list[Record]) -> Callable[[], Iterator[Record]]:
    """`functools.partial(iter, rows)` rather than `lambda: iter(rows)`.

    A lambda is not picklable by the standard library `pickle` module
    (used by `multiprocessing`) no matter what it closes over. A
    `functools.partial` wrapping a picklable callable (`iter`, a builtin)
    and picklable arguments (`rows`, a list of plain dicts) is picklable.
    This is what lets a Dataset built by MemoryDataSource be sent whole to
    a worker process (Milestone 3) without changing Partition's public
    `records_fn: Callable[[], Iterator[Record]]` contract at all.
    """
    return functools.partial(iter, rows)
