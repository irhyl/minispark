"""ShuffleWriter: partitions records with a Partitioner and writes them to
local disk, one block file per (stage, source task, target partition).

Streams: a record is pickled and appended to its target partition's file
as soon as it is seen, one record at a time, never buffering a complete
target-partition bucket in memory before writing. This is what "do not
hold the entire shuffle dataset in RAM" (build spec) means in practice:
the only per-target-partition state kept in memory while writing is one
open file handle and a running checksum, not the records themselves.

Blocks are pickled Records (a sequence of back-to-back `pickle.dump`
calls to one file, read back with repeated `pickle.load` until EOF), not
newline-delimited JSON. JSON would silently turn a Python tuple (e.g. an
Avg aggregate's `(sum, count)` partial state, see expressions/aggregate.py)
into a list on the way back out, and cannot represent NaN by default.
Pickle preserves exact Python types across the round trip, and every
worker process reading these files is already a MiniSpark process willing
to unpickle MiniSpark data (see execution/scheduler.py's picklability
notes for Task/PhysicalPlan, the same trust boundary already applies).
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from minispark.core.record import Record
from minispark.shuffle.partitioner import Partitioner


@dataclass(frozen=True)
class ShuffleBlockMeta:
    stage_id: int
    source_task_id: int
    target_partition: int
    path: str
    record_count: int
    byte_length: int
    checksum: str  # hex MD5 digest of the block file's bytes


def shuffle_partition_dir(root_dir: str, stage_id: int, target_partition: int) -> Path:
    return Path(root_dir) / f"stage_{stage_id}" / f"partition_{target_partition}"


class _BlockWriter:
    """Per-target-partition write state: one open file, a running checksum,
    and running counters. Not exported; an implementation detail of
    `write_shuffle_partition`."""

    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("wb")
        self._hasher = hashlib.md5(usedforsecurity=False)
        self.record_count = 0
        self.byte_length = 0

    def write(self, record: Record) -> None:
        chunk = pickle.dumps(record)
        self._file.write(chunk)
        self._hasher.update(chunk)
        self.record_count += 1
        self.byte_length += len(chunk)

    def close(self) -> str:
        self._file.close()
        return self._hasher.hexdigest()


def write_shuffle_partition(
    root_dir: str,
    stage_id: int,
    source_task_id: int,
    records: Iterable[Record],
    key_fn: Callable[[Record], Any],
    partitioner: Partitioner,
) -> list[ShuffleBlockMeta]:
    """Partition `records` by `partitioner.partition_for(key_fn(record))`
    and write one block file per target partition that received at least
    one record (a target partition no source task ever wrote to for
    simply has no block; readers treat that as zero rows, not an error).
    """
    writers: dict[int, _BlockWriter] = {}
    try:
        for record in records:
            target = partitioner.partition_for(key_fn(record))
            writer = writers.get(target)
            if writer is None:
                directory = shuffle_partition_dir(root_dir, stage_id, target)
                directory.mkdir(parents=True, exist_ok=True)
                writer = _BlockWriter(directory / f"block_{source_task_id}.pkl")
                writers[target] = writer
            writer.write(record)
    finally:
        checksums = {target: writer.close() for target, writer in writers.items()}

    return [
        ShuffleBlockMeta(
            stage_id=stage_id,
            source_task_id=source_task_id,
            target_partition=target,
            path=str(writer.path),
            record_count=writer.record_count,
            byte_length=writer.byte_length,
            checksum=checksums[target],
        )
        for target, writer in writers.items()
    ]
