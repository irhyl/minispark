"""Parquet DataSource: real columnar storage, feeding the same row-based
engine every other DataSource feeds.

Milestone 7's "columnar execution" is scoped deliberately: pyarrow reads
Parquet with genuine column pruning (an unrequested column's data pages
are never decoded) and genuine predicate pushdown (a row group whose own
min/max statistics prove it cannot match the pushed filter is skipped
without reading its data), so bytes actually not read shrink for real.
Everything downstream of this file, every physical operator, the shuffle,
the checkpoint format, stays exactly the row-at-a-time Python engine that
exists today: a fragment's matching rows are decoded into plain `Record`
dicts (`RecordBatch.to_pylist()`) at the partition boundary, and nothing
past that point knows or cares that the source was columnar. A fully
vectorized execution path (Filter/Project operating on Arrow batches
without ever materializing Records) is explicit future work, not
attempted here; see docs/architecture.md's Key Milestone-7 design
decisions for why this split was chosen.

This is the only module in the package that imports pyarrow: `columnar`
is an optional extra (see pyproject.toml), and nothing here is imported
by anything that runs without it (`api/session.py`'s `.parquet()` and
`api/writer.py`'s `.parquet()` both import this module lazily, inside the
method body, specifically so `import minispark.api.session` alone never
requires pyarrow to be installed).
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as pa_dataset
import pyarrow.parquet as pq

from minispark.core.dataset import Dataset
from minispark.core.partition import Partition, PartitionMetadata
from minispark.core.record import Record
from minispark.core.schema import Field, Schema
from minispark.core.types import BOOL, FLOAT, INT, NULL, STRING, DataType
from minispark.expressions.base import Expression
from minispark.expressions.binary import (
    And,
    Equal,
    GreaterEqual,
    GreaterThan,
    LessEqual,
    LessThan,
    NotEqual,
    Or,
)
from minispark.expressions.column import Column
from minispark.expressions.literal import Literal
from minispark.expressions.predicates import IsNotNull, IsNull, Not
from minispark.storage.datasource import DataSource

_COMPARISONS = (GreaterThan, GreaterEqual, LessThan, LessEqual, Equal, NotEqual)

# Sentinel distinguishing "this expression translates to nothing" from a
# legitimate scalar value of None (e.g. Literal(None)): using plain None
# for both would make `col("x") == None` indistinguishable from "could
# not translate", so a value-level None must be able to survive here.
_UNSUPPORTED = object()


def _arrow_type_to_datatype(t: pa.DataType) -> DataType:
    if pa.types.is_boolean(t):
        return BOOL
    if pa.types.is_integer(t):
        return INT
    if pa.types.is_floating(t):
        return FLOAT
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return STRING
    if pa.types.is_null(t):
        return NULL
    raise ValueError(
        f"Unsupported Parquet/Arrow type for MiniSpark: {t!r}. Only bool, "
        "integer, floating-point, string, and null are supported; "
        "date/timestamp/decimal/nested types are not implemented."
    )


def _datatype_to_arrow_type(t: DataType) -> pa.DataType:
    if t == BOOL:
        return pa.bool_()
    if t == INT:
        return pa.int64()
    if t == FLOAT:
        return pa.float64()
    if t == STRING:
        return pa.string()
    if t == NULL:
        return pa.null()
    raise ValueError(f"Unsupported MiniSpark type for Parquet: {t!r}")


def _arrow_schema_to_minispark(arrow_schema: pa.Schema) -> Schema:
    return Schema(
        [Field(f.name, _arrow_type_to_datatype(f.type), nullable=f.nullable) for f in arrow_schema]
    )


def _minispark_schema_to_arrow(schema: Schema) -> pa.Schema:
    return pa.schema(
        [
            pa.field(f.name, _datatype_to_arrow_type(f.data_type), nullable=f.nullable)
            for f in schema
        ]
    )


def _translate_operand(expr: Expression) -> Any:
    if isinstance(expr, Column):
        return pa_dataset.field(expr.name)
    if isinstance(expr, Literal):
        # A None literal is deliberately excluded, not translated to a
        # pyarrow null scalar: pyarrow's comparison operators use SQL's
        # three-valued NULL logic (`x == null` never matches, even when x
        # is itself null), but MiniSpark's row engine evaluates `==`/`!=`
        # as plain Python equality, where `None == None` is True. Pushing
        # a None-literal comparison as pyarrow would evaluate it could
        # wrongly exclude rows the row-level Filter (which always stays
        # in the plan) would have kept, an over-exclusion, not just a
        # missed optimization; safer to never push it at all.
        if expr.value is None:
            return _UNSUPPORTED
        return expr.value
    return _UNSUPPORTED


def translate_predicate(expr: Expression) -> pa_dataset.Expression | None:
    """Best-effort translation of a MiniSpark `Expression` into a pyarrow
    dataset filter. Returns `None` when no part of `expr` could be
    translated; a caller pushing a partial result down as an optimization
    (e.g. one side of an `And`) must never treat `None` as "filters
    everything out", only as "nothing to push here."

    Only ever narrows what is pushed relative to what `expr` actually
    means, never widens it: an `And` may push just one translatable side
    (a safe superset, the untranslated side still gets checked by the
    row-level Filter that always stays in the physical plan regardless of
    what was pushed), but an `Or` only pushes if *both* sides translate,
    since pushing only one side of an "or" could wrongly exclude rows the
    untranslated side would have kept. Arithmetic (Add/Subtract/...) is
    never translated: pushing `(a + b) > 5` is possible in principle via
    pyarrow.compute but is not implemented, out of scope for what this
    milestone needs to demonstrate.
    """
    if isinstance(expr, And):
        left = translate_predicate(expr.left)
        right = translate_predicate(expr.right)
        if left is not None and right is not None:
            return left & right
        return left if left is not None else right
    if isinstance(expr, Or):
        left = translate_predicate(expr.left)
        right = translate_predicate(expr.right)
        if left is not None and right is not None:
            return left | right
        return None
    if isinstance(expr, Not):
        inner = translate_predicate(expr.child)
        return ~inner if inner is not None else None
    if isinstance(expr, IsNull):
        operand = _translate_operand(expr.child)
        return operand.is_null() if isinstance(operand, pa_dataset.Expression) else None
    if isinstance(expr, IsNotNull):
        operand = _translate_operand(expr.child)
        return operand.is_valid() if isinstance(operand, pa_dataset.Expression) else None
    if isinstance(expr, _COMPARISONS):
        left = _translate_operand(expr.left)
        right = _translate_operand(expr.right)
        if not isinstance(left, pa_dataset.Expression) or right is _UNSUPPORTED:
            return None
        return expr.op(left, right)
    return None


def _read_fragments(
    fragments: list[pa_dataset.Fragment],
    columns: list[str] | None,
    pa_filter: pa_dataset.Expression | None,
) -> Iterator[Record]:
    """Module-level, not a closure, so it stays picklable for a
    `ProcessPoolExecutor` worker (same constraint as every other
    DataSource's records_fn, see storage/csv.py's `_read_csv_range`
    docstring). `fragments` and `pa_filter` are genuine pyarrow objects,
    not a path-plus-offsets reconstruction: `ParquetFileFragment` and
    `pyarrow.dataset.Expression` are both directly picklable (verified
    against pyarrow 20 with a real ProcessPoolExecutor round trip before
    relying on it here), so there is no need to re-derive them from
    scratch inside the worker process the way a plain file path would.
    """
    for fragment in fragments:
        table = fragment.to_table(columns=columns, filter=pa_filter)
        yield from table.to_pylist()


class ParquetDataSource(DataSource):
    """Reads a single Parquet file or a directory of them (whatever
    `pyarrow.dataset.dataset()` accepts) as a Dataset, partitioned at
    row-group granularity: each of `num_partitions` buckets gets a
    disjoint subset of the dataset's row groups (found via `pyarrow.
    dataset`'s fragment API, split with `split_by_row_group()`), and a
    partition's `records_fn` reads only its own assigned row groups, with
    `columns`/`filter` applied by pyarrow itself, not by this class.

    Row-group-level predicate pushdown (skipping a row group whose
    min/max statistics prove it cannot match) and column pruning
    (decoding only requested columns) are pyarrow's own, already-tested
    behavior; this class does not re-implement or re-verify that logic,
    it only translates MiniSpark's `Expression` tree into the
    `pyarrow.dataset.Expression` pyarrow's scanner understands (see
    `translate_predicate`) and assigns row groups to partitions.
    """

    def __init__(self, path: str, num_partitions: int = 4):
        self._path = path
        self._num_partitions = max(1, num_partitions)

    @property
    def name(self) -> str:
        return f"parquet:{self._path}"

    def read(self, columns: list[str] | None = None, filter: Expression | None = None) -> Dataset:
        dataset = pa_dataset.dataset(self._path, format="parquet")
        full_schema = _arrow_schema_to_minispark(dataset.schema)
        schema = full_schema.select(columns) if columns is not None else full_schema

        pa_filter = translate_predicate(filter) if filter is not None else None
        row_groups = [
            rg for frag in dataset.get_fragments() for rg in frag.split_by_row_group()
        ]

        n = max(1, min(self._num_partitions, len(row_groups) or 1))
        chunk_size = -(-len(row_groups) // n) if row_groups else 0
        partitions = []
        for i in range(n):
            start, end = i * chunk_size, min((i + 1) * chunk_size, len(row_groups))
            if start >= end and row_groups:
                continue
            group = row_groups[start:end]
            row_count = sum(rg.num_rows for frag in group for rg in frag.row_groups)
            partitions.append(
                Partition(
                    partition_id=i,
                    schema=schema,
                    records_fn=functools.partial(_read_fragments, group, columns, pa_filter),
                    metadata=PartitionMetadata(location=self._path, row_count=row_count),
                )
            )
        if not partitions:
            partitions = [
                Partition(
                    0, schema, functools.partial(_read_fragments, [], columns, pa_filter),
                    PartitionMetadata(row_count=0),
                )
            ]
        return Dataset(schema, partitions)


def write_parquet_dataset(dataset: Dataset, directory: str) -> None:
    """Write every partition of `dataset` to `directory` as its own
    `.parquet` file (`part-00000.parquet`, `part-00001.parquet`, ...),
    mirroring how a real distributed write produces one file per
    partition rather than one giant file requiring a coordinating merge.
    Each partition is materialized into one `pyarrow.Table` before
    writing (`pyarrow.Table.from_pylist`); Parquet's format is not
    row-at-a-time appendable the way a shuffle block or a checkpoint file
    is, so unlike those, this is not a streaming writer.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrow_schema = _minispark_schema_to_arrow(dataset.schema)
    for partition in dataset.partitions():
        rows = list(partition)
        table = pa.Table.from_pylist(rows, schema=arrow_schema)
        pq.write_table(table, str(root / f"part-{partition.partition_id:05d}.parquet"))
