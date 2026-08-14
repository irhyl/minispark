"""Checkpoint DataSource: reads back a Dataset materialized to durable
local disk by `write_checkpoint()`.

Unlike a shuffle block (shuffle/writer.py), a checkpoint is not scratch
space cleaned up at the end of one query: it is meant to outlive the
query that created it, so a later query (or a later run of this one) can
read it back without recomputing whatever produced it. This is what
`DataFrame.checkpoint()` (api/dataframe.py) uses to cut a logical plan's
lineage at a chosen point: everything above the checkpoint becomes a
plain `Scan` over this DataSource, and everything below it never needs to
run again for that DataFrame to be recomputed (e.g. by lineage-based
recovery, see execution/scheduler.py).

Reuses the same on-disk record format as a shuffle block (back-to-back
`pickle.dump` calls to one file per partition, read back with repeated
`pickle.load` until EOF, see shuffle/writer.py's docstring for why
pickle, not newline-delimited JSON) rather than inventing a second
format for what is structurally the same problem: durably persist a
sequence of Records and read them back exactly.
"""

from __future__ import annotations

import functools
import pickle
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.core.schema import Schema
from minispark.storage.datasource import DataSource

_META_FILENAME = "_meta.pkl"


@dataclass(frozen=True)
class CheckpointPartitionMeta:
    partition_id: int
    path: str
    row_count: int


def _partition_path(directory: Path, partition_id: int) -> Path:
    return directory / f"partition_{partition_id}.pkl"


def _read_checkpoint_partition(path: str) -> Iterator[Record]:
    """Module-level, not a nested closure, so `CheckpointDataSource.read()`
    can bind it with `functools.partial` into a picklable `records_fn`
    (see storage/memory.py's `_make_records_fn` for the same constraint
    applied to the in-memory source: a lambda or a closure over an
    enclosing method's variables is never picklable, no matter what it
    captures, and `records_fn` must survive being sent to a worker
    process when `local[N]` has `N > 1`)."""
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def write_checkpoint(dataset: Dataset, directory: str) -> None:
    """Write every partition of `dataset` to `directory`, one file per
    partition, plus a `_meta.pkl` recording the schema and each
    partition's path and row count. Overwrites anything already at
    `directory`: a checkpoint directory is meant to hold exactly one
    dataset at a time, not accumulate across calls.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    parts: list[CheckpointPartitionMeta] = []
    for partition in dataset.partitions():
        path = _partition_path(root, partition.partition_id)
        row_count = 0
        with path.open("wb") as f:
            for record in partition:
                pickle.dump(record, f)
                row_count += 1
        parts.append(CheckpointPartitionMeta(partition.partition_id, str(path), row_count))

    with (root / _META_FILENAME).open("wb") as f:
        pickle.dump((dataset.schema, parts), f)


class CheckpointDataSource(DataSource):
    def __init__(self, directory: str):
        self._directory = Path(directory)
        meta_path = self._directory / _META_FILENAME
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Not a checkpoint directory (missing {_META_FILENAME}): {directory}"
            )
        with meta_path.open("rb") as f:
            self._schema, self._parts = pickle.load(f)

    @property
    def name(self) -> str:
        return f"checkpoint:{self._directory}"

    def read(self) -> Dataset:
        schema: Schema = self._schema
        parts: list[CheckpointPartitionMeta] = self._parts
        partitions = [
            Partition(
                partition_id=p.partition_id,
                schema=schema,
                records_fn=functools.partial(_read_checkpoint_partition, p.path),
                metadata=PartitionMetadata(location=p.path, row_count=p.row_count),
            )
            for p in parts
        ]
        return Dataset(schema, partitions)
