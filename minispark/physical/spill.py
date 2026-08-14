"""Spill file I/O for Milestone 9's memory-aware `SortExec`/
`HashAggregateExec` (`physical/operators.py`).

A spill file is ephemeral, same-process, same-machine scratch: written
and read back within the exact same `Task`, never crossing a process
boundary. That is what lets this be simpler than `shuffle/writer.py`'s
block format: no checksum, no `ShuffleBlockMeta`, no driver-side
tracking, just back-to-back `pickle.dump`/`pickle.load`, the same
stream-of-pickled-objects shape shuffle blocks and checkpoint files both
already use (see `shuffle/writer.py`'s and `storage/checkpoint.py`'s
docstrings for why pickle over newline-delimited JSON: exact Python type
round-tripping, e.g. an `Avg` aggregate's `(sum, count)` tuple state).
`items` is untyped (`Any`) deliberately: a sort spill file holds `Record`
dicts, an aggregate spill file holds `(key, state)` tuples, and both
shapes round-trip through pickle identically without this module needing
to know the difference.
"""

from __future__ import annotations

import pickle
import shutil
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def make_spill_dir(prefix: str) -> str:
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_spill_dir(directory: str | None) -> None:
    if directory is not None:
        shutil.rmtree(directory, ignore_errors=True)


def write_spill_file(directory: str, name: str, items: Iterable[Any]) -> str:
    path = Path(directory) / name
    with path.open("wb") as f:
        for item in items:
            pickle.dump(item, f)
    return str(path)


def read_spill_file(path: str) -> Iterator[Any]:
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
