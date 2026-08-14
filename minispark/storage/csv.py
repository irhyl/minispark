"""CSV DataSource.

Reading proceeds in three passes over the file, all at `read()` time:

  1. Count total rows, via `f.readline()` (not `csv.reader`, see below),
     so row ranges can be assigned to partitions.
  2. Infer a schema from a sample of rows (`_infer_schema_from_sample`,
     unchanged since Milestone 1: a separate, capped-length pass, not
     related to partitioning).
  3. Milestone 9: walk the file a second time with `f.readline()`,
     recording the byte offset (`f.tell()`) immediately before each
     partition's first row (`_locate_partition_offsets`). This is metadata
     only (no row is retained in memory afterward), and the recorded list
     is one offset per partition, not per row, so it stays small
     regardless of file size.

At iteration time (i.e. when a partition's factory is actually called),
`_read_csv_range` seeks straight to its partition's recorded offset and
reads exactly its own row range, instead of re-parsing every row before
it. Before Milestone 9, that prefix was re-parsed by every partition on
every read (partition i's factory ran `csv.reader` from the top of the
file and threw away the first `i * chunk_size` parsed rows via
`itertools.islice`), meaning the file's data section was effectively
parsed `num_partitions` times per query. The byte-offset seek removes
that: each row is now parsed exactly once, by whichever partition owns
it, at the cost of the two extra `readline()`-only (no CSV tokenizing)
full-file passes above, which is a real net win whenever `num_partitions`
extra passes would have been.

`csv.reader(f)` cannot be used for the offset-recording pass or for
seeking back into the middle of the file to read from an offset: iterating
`csv.reader(f)` calls `next()` on `f` internally in a way that disables
`f.tell()` for the rest of that file object's lifetime (raises `OSError:
telling position disabled by next() call` on any subsequent `f.tell()`).
`f.readline()` does not have this problem and round-trips correctly with
`f.seek()`. This is why `_read_csv_range` parses each line individually
via `next(csv.reader([line]))` after seeking, rather than opening one
`csv.reader` over the file and skipping ahead. The accepted cost: a
quoted CSV field containing a literal embedded newline is, under
`readline()`, split across two calls, something `csv.reader(f)` iterated
from the top handles correctly; depending on how the split lands this
either silently misparses the row or, when the split leaves a line with
fewer fields than the header, raises `ValueError` out of `_coerce_row`'s
`zip(..., strict=True)` (see `tests/unit/test_csv_byte_offset.py`'s
`test_embedded_newline_in_quoted_field_is_a_known_limitation`, which
pins down the latter). This codebase's CSV reader was never a full RFC
4180 implementation (see `_try_parse`'s type inference and the lack of
custom delimiters/quoting options); this is one more, now-documented,
gap in that same spirit.

`read(columns=...)` (Milestone 7) is real, if partial, projection
pruning: `csv.reader` still tokenizes every field on every line (there is
no way to skip that for a row-oriented text format without an index), but
`_coerce_row` only runs `_try_parse` (the actual per-value conversion
work) for requested columns, and every Record built downstream is
already the narrow, pruned width. `filter` is accepted for interface
uniformity with every DataSource but is not honored: CSV has no
statistics to skip rows or row ranges against, unlike Parquet's row-group
metadata (see storage/parquet.py), so a pushed filter would only mean
"evaluate it earlier," not "read less," and is not worth the added
complexity here.
"""

from __future__ import annotations

import csv
import functools
import itertools
from collections.abc import Iterator
from pathlib import Path

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.core.schema import Field, Schema
from minispark.core.types import STRING, DataType, infer_type
from minispark.expressions.base import Expression
from minispark.storage.datasource import DataSource

_SCHEMA_SAMPLE_ROWS = 1000


def _try_parse(raw: str) -> object:
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _infer_schema_from_sample(path: Path, header: list[str]) -> Schema:
    column_types: dict[str, DataType] = {}
    saw_null: dict[str, bool] = {name: False for name in header}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in itertools.islice(reader, _SCHEMA_SAMPLE_ROWS):
            for name, raw in zip(header, row, strict=True):
                value = _try_parse(raw)
                if value is None:
                    saw_null[name] = True
                    continue
                inferred = infer_type(value)
                current = column_types.get(name)
                if current is None:
                    column_types[name] = inferred
                elif current != inferred:
                    # Mixed types in the sample (e.g. "1" then "abc") widen to
                    # string rather than raising: schema inference is a
                    # best-effort convenience in Milestone 1, not a contract
                    # enforced by an analyzer yet.
                    column_types[name] = STRING

    fields = [
        Field(name, column_types.get(name, STRING), nullable=saw_null.get(name, True))
        for name in header
    ]
    return Schema(fields)


def _coerce_row(header: list[str], row: list[str], wanted: frozenset[str] | None) -> Record:
    if wanted is None:
        return {name: _try_parse(raw) for name, raw in zip(header, row, strict=True)}
    return {
        name: _try_parse(raw)
        for name, raw in zip(header, row, strict=True)
        if name in wanted
    }


def _parse_csv_line(line: str) -> list[str]:
    """Tokenize one already-read line the same way `csv.reader` would.
    See the module docstring for why this reads one line at a time
    (`f.readline()`) instead of iterating a `csv.reader(f)` from the
    file's current seek position, and the accepted embedded-newline
    limitation that follows from it."""
    return next(csv.reader([line]))


def _count_rows(path: Path) -> int:
    """Count data rows (excluding the header) via `f.readline()`, not
    `csv.reader`: counting does not need each row's fields, just the
    line count, so this avoids tokenizing every field on every line
    (unlike the pre-Milestone-9 `sum(1 for _ in reader)`, which iterated
    a `csv.reader` and so parsed each row's fields only to discard them).
    """
    with path.open(newline="", encoding="utf-8") as f:
        f.readline()  # header
        count = 0
        while f.readline():
            count += 1
        return count


def _locate_partition_offsets(path: Path, start_rows: list[int]) -> dict[int, int]:
    """For each data-row index in `start_rows` (0-based, header excluded),
    find the byte offset (`f.tell()`, opaque but seek-safe, see module
    docstring) of the position immediately before that row. One
    `readline()`-only pass over the file, not one seek-search per
    partition: `start_rows` is sorted and walked alongside the file in a
    single forward scan.
    """
    wanted = sorted(set(start_rows))
    offsets: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        f.readline()  # header
        next_wanted_idx = 0
        row_idx = 0
        pos = f.tell()
        while next_wanted_idx < len(wanted):
            line = f.readline()
            if not line:
                break
            if row_idx == wanted[next_wanted_idx]:
                offsets[row_idx] = pos
                next_wanted_idx += 1
            row_idx += 1
            pos = f.tell()
        # A requested row index at or past EOF (only possible for a
        # partition assigned zero rows, e.g. an empty/header-only file:
        # `CSVDataSource.read()` still builds one 0-row partition rather
        # than none) has nothing to seek to; `end - start == 0` there, so
        # `_read_csv_range` never actually reads using this offset, and
        # the current (EOF) position is a safe, harmless placeholder.
        for row_idx in wanted[next_wanted_idx:]:
            offsets[row_idx] = pos
    return offsets


def _read_csv_range(
    path: Path, header: list[str], start: int, end: int, offset: int, columns: list[str] | None
) -> Iterator[Record]:
    """Stream rows `[start, end)` of `path`, re-opening the file and
    seeking straight to `offset` (the byte position immediately before
    row `start`, computed once in `CSVDataSource.read()` by
    `_locate_partition_offsets`) rather than re-parsing every row before
    it. See the module docstring for the full before/after picture.

    Module-level, not a nested closure, so `CSVDataSource._make_records_fn`
    can bind it with `functools.partial` into a picklable `records_fn`. A
    closure over `path`/`header`/`start`/`end`/`offset` captured from an
    enclosing method is not picklable by the standard library `pickle`
    module (used by `multiprocessing`); a `functools.partial` wrapping a
    module-level function and picklable arguments is. See storage/
    memory.py's `_make_records_fn` for the same fix on the in-memory
    source.
    """
    wanted = frozenset(columns) if columns is not None else None
    with path.open(newline="", encoding="utf-8") as f:
        f.seek(offset)
        for _ in range(end - start):
            line = f.readline()
            if not line:
                break
            row = _parse_csv_line(line)
            yield _coerce_row(header, row, wanted)


class CSVDataSource(DataSource):
    def __init__(self, path: str, schema: Schema | None = None, num_partitions: int = 4):
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")
        self._explicit_schema = schema
        self._num_partitions = num_partitions

    @property
    def name(self) -> str:
        return f"csv:{self._path}"

    def read(self, columns: list[str] | None = None, filter: Expression | None = None) -> Dataset:
        with self._path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader([f.readline()]))
        row_count = _count_rows(self._path)

        full_schema = self._explicit_schema or _infer_schema_from_sample(self._path, header)
        schema = full_schema.select(columns) if columns is not None else full_schema

        n = max(1, min(self._num_partitions, row_count or 1))
        chunk_size = -(-row_count // n) if row_count else 0
        ranges = []
        for i in range(n):
            start, end = i * chunk_size, min((i + 1) * chunk_size, row_count)
            if start >= end and row_count > 0:
                continue
            ranges.append((i, start, end))

        offsets = _locate_partition_offsets(self._path, [start for _, start, _ in ranges])
        partitions = [
            Partition(
                partition_id=i,
                schema=schema,
                records_fn=self._make_records_fn(header, start, end, offsets[start], columns),
                metadata=PartitionMetadata(location=str(self._path), row_count=end - start),
            )
            for i, start, end in ranges
        ]
        if not partitions:
            partitions = [
                Partition(0, schema, functools.partial(iter, []), PartitionMetadata(row_count=0))
            ]
        return Dataset(schema, partitions)

    def _make_records_fn(
        self, header: list[str], start: int, end: int, offset: int, columns: list[str] | None
    ):
        return functools.partial(_read_csv_range, self._path, header, start, end, offset, columns)


def read_csv(path: str, schema: Schema | None = None, num_partitions: int = 4) -> Dataset:
    return CSVDataSource(path, schema=schema, num_partitions=num_partitions).read()
