"""CSV DataSource.

Reading proceeds in two passes over the file:

  1. At `read()` time: scan the file once to (a) infer a schema from a
     sample of rows, and (b) count total rows so row ranges can be
     assigned to partitions. This is metadata only — no row is retained
     in memory afterward.
  2. At iteration time (i.e. when a partition's factory is actually
     called): re-open the file and stream just that partition's row
     range with `itertools.islice`, coercing each row as it is read.

This means a CSV file much larger than RAM can still be scanned, at the
cost of re-reading (not re-parsing the whole file, just seeking past
earlier rows) once per partition. A production system would instead
record byte offsets per partition to seek directly; that optimization is
skipped here as unnecessary complexity for Milestone 1's goals.

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


def _read_csv_range(
    path: Path, header: list[str], start: int, end: int, columns: list[str] | None
) -> Iterator[Record]:
    """Stream rows `[start, end)` of `path`, re-opening the file.

    Module-level, not a nested closure, so `CSVDataSource._make_records_fn`
    can bind it with `functools.partial` into a picklable `records_fn`. A
    closure over `path`/`header`/`start`/`end` captured from an enclosing
    method is not picklable by the standard library `pickle` module (used
    by `multiprocessing`); a `functools.partial` wrapping a module-level
    function and picklable arguments is. See storage/memory.py's
    `_make_records_fn` for the same fix on the in-memory source.
    """
    wanted = frozenset(columns) if columns is not None else None
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        window = itertools.islice(reader, start, end)
        for row in window:
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
            reader = csv.reader(f)
            header = next(reader)
            row_count = sum(1 for _ in reader)

        full_schema = self._explicit_schema or _infer_schema_from_sample(self._path, header)
        schema = full_schema.select(columns) if columns is not None else full_schema

        n = max(1, min(self._num_partitions, row_count or 1))
        chunk_size = -(-row_count // n) if row_count else 0
        partitions = []
        for i in range(n):
            start, end = i * chunk_size, min((i + 1) * chunk_size, row_count)
            if start >= end and row_count > 0:
                continue
            partitions.append(
                Partition(
                    partition_id=i,
                    schema=schema,
                    records_fn=self._make_records_fn(header, start, end, columns),
                    metadata=PartitionMetadata(location=str(self._path), row_count=end - start),
                )
            )
        if not partitions:
            partitions = [
                Partition(0, schema, functools.partial(iter, []), PartitionMetadata(row_count=0))
            ]
        return Dataset(schema, partitions)

    def _make_records_fn(self, header: list[str], start: int, end: int, columns: list[str] | None):
        return functools.partial(_read_csv_range, self._path, header, start, end, columns)


def read_csv(path: str, schema: Schema | None = None, num_partitions: int = 4) -> Dataset:
    return CSVDataSource(path, schema=schema, num_partitions=num_partitions).read()
