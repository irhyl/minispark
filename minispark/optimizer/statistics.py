"""Table and column statistics.

Milestone 2 scope: statistics exist as infrastructure, computed on demand
by a single full scan of a Dataset. Nothing in the optimizer or physical
planner consults them yet, because there is no decision to make with them
until Milestone 5 needs to choose between join strategies based on size.
Building this now (rather than when Milestone 5 needs it) means the
Dataset-scanning code and the statistics data shape exist and are tested
before anything depends on them being correct under load.

What is exact versus estimated, precisely:

  * row_count, null_count, min_value, max_value, distinct_count: exact.
    Computed by iterating every record once. distinct_count keeps every
    distinct value seen in a Python set, so its memory cost is
    proportional to cardinality, not row count. A production system would
    use an approximate structure (e.g. HyperLogLog) for high-cardinality
    columns; that is not implemented here.
  * estimated_size_bytes: a rough heuristic (`sys.getsizeof` summed over
    each row's values), not an on-disk byte count. It captures relative
    order of magnitude, not a number you should build a memory budget on.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from minispark.core.dataset import Dataset


@dataclass
class ColumnStatistics:
    null_count: int = 0
    distinct_count: int = 0
    min_value: Any = None
    max_value: Any = None


@dataclass
class TableStatistics:
    row_count: int = 0
    estimated_size_bytes: int = 0
    columns: dict[str, ColumnStatistics] = field(default_factory=dict)


def compute_statistics(dataset: Dataset, columns: list[str] | None = None) -> TableStatistics:
    """Scan `dataset` once and compute exact row/column statistics.

    `columns` restricts which columns get per-column statistics (all
    schema columns by default); row_count and estimated_size_bytes are
    always computed over the whole row regardless.
    """
    target_columns = columns if columns is not None else dataset.schema.field_names()
    distinct_values: dict[str, set[Any]] = {c: set() for c in target_columns}
    null_counts: dict[str, int] = dict.fromkeys(target_columns, 0)
    min_values: dict[str, Any] = dict.fromkeys(target_columns)
    max_values: dict[str, Any] = dict.fromkeys(target_columns)

    row_count = 0
    estimated_size_bytes = 0
    for record in dataset.iter_records():
        row_count += 1
        estimated_size_bytes += sum(sys.getsizeof(v) for v in record.values())
        for name in target_columns:
            value = record.get(name)
            if value is None:
                null_counts[name] += 1
                continue
            distinct_values[name].add(value)
            if min_values[name] is None or value < min_values[name]:
                min_values[name] = value
            if max_values[name] is None or value > max_values[name]:
                max_values[name] = value

    columns_stats = {
        name: ColumnStatistics(
            null_count=null_counts[name],
            distinct_count=len(distinct_values[name]),
            min_value=min_values[name],
            max_value=max_values[name],
        )
        for name in target_columns
    }
    return TableStatistics(
        row_count=row_count,
        estimated_size_bytes=estimated_size_bytes,
        columns=columns_stats,
    )
