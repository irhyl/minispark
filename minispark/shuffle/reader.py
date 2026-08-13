"""ShuffleReader: reads back the blocks written for one target partition.

Reads one block at a time: a reduce task never needs a whole target
partition (which may span many blocks, one per source task that produced
at least one row for it) resident in memory as a single structure. Within
one block, checksum verification reads that block's raw bytes into memory
once (to hash exactly the bytes that were written, not a re-serialization
of the deserialized objects, which pickle does not guarantee is
byte-identical), then deserializes records from that in-memory buffer.
This bounds memory by one block's size, not by the whole target
partition's or dataset's size; it is not a fully streaming reader for a
single very large block. A production system would checksum fixed-size
framed chunks instead of whole blocks to stream a multi-GB block; that is
unneeded complexity for what Milestone 4 needs to demonstrate.
"""

from __future__ import annotations

import hashlib
import io
import pickle
from collections.abc import Iterator

from minispark.core.record import Record
from minispark.shuffle.writer import ShuffleBlockMeta


class ShuffleChecksumError(Exception):
    """Raised when a block's bytes on disk do not match its recorded checksum."""


def read_shuffle_blocks(
    blocks: list[ShuffleBlockMeta], verify_checksum: bool = True
) -> Iterator[Record]:
    for block in blocks:
        yield from _read_block(block, verify_checksum)


def _read_block(block: ShuffleBlockMeta, verify_checksum: bool) -> Iterator[Record]:
    with open(block.path, "rb") as f:
        raw = f.read()

    if verify_checksum:
        digest = hashlib.md5(raw, usedforsecurity=False).hexdigest()
        if digest != block.checksum:
            raise ShuffleChecksumError(
                f"Checksum mismatch reading {block.path}: expected "
                f"{block.checksum}, got {digest}"
            )

    buffer = io.BytesIO(raw)
    while True:
        try:
            yield pickle.load(buffer)
        except EOFError:
            break
