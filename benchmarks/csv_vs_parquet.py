"""Benchmark: does Parquet's real column pruning and predicate pushdown
(Milestone 7) actually read less than CSV for the same filtered,
narrow-projected query?

Writes the same synthetic data as both a CSV file and a Parquet file
(multi-row-group, `id` monotonically increasing so each row group covers
a distinct, disjoint `id` range and a filter on `id` can actually skip
whole row groups, not just individual rows within one), then runs the
identical `filter(...).select(...)` query against each, reporting
wall-clock time and `DataFrame.last_run_metrics`'s total_input_records
for the Scan-side stage as a direct, quantitative measure of "how much
data actually got read," not just a timing number that could be
explained by something else. See docs/benchmarks.md for the recorded
numbers and this project's benchmark-honesty caveat (single trial, this
machine, not a controlled rig).

Run with (from the repository root):

    python -m benchmarks.csv_vs_parquet

Requires the optional `columnar` extra (`pip install -e ".[columnar]"`).
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks._common import machine_info, timed
from minispark.api.functions import col
from minispark.api.session import MiniSparkSession

ROWS = 200_000
NUM_COLUMNS = 10  # id, value, and 8 padding columns never referenced by the query
FILTER_THRESHOLD = ROWS - ROWS // 20  # last 5% of ids only: most row groups fully excluded


def make_rows(n: int) -> dict[str, list]:
    columns: dict[str, list] = {"id": list(range(n)), "value": [i % 1000 for i in range(n)]}
    for i in range(NUM_COLUMNS - 2):
        columns[f"pad_{i}"] = [f"padding-value-{j}-{i}" for j in range(n)]
    return columns


def write_csv(path: Path, columns: dict[str, list]) -> None:
    names = list(columns.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        for row in zip(*columns.values(), strict=True):
            writer.writerow(row)


def write_parquet(path: Path, columns: dict[str, list]) -> None:
    table = pa.table(columns)
    pq.write_table(table, str(path), row_group_size=5_000)


def main() -> None:
    print(machine_info())
    print(f"rows={ROWS}, columns={NUM_COLUMNS} (query only needs 'id' and 'value')")
    print(f"filter: id >= {FILTER_THRESHOLD} (last 5% of rows)")
    print()

    tmpdir = Path(tempfile.mkdtemp(prefix="minispark-bench-"))
    columns = make_rows(ROWS)
    csv_path = tmpdir / "data.csv"
    parquet_path = tmpdir / "data.parquet"
    write_csv(csv_path, columns)
    write_parquet(parquet_path, columns)

    session = (
        MiniSparkSession.builder.master("local[1]").app_name("bench_csv_parquet").get_or_create()
    )

    csv_df = (
        session.read.csv(str(csv_path), num_partitions=4)
        .filter(col("id") >= FILTER_THRESHOLD)
        .select("id", "value")
    )
    with timed() as t_csv:
        csv_rows = csv_df.collect()
    csv_input_records = csv_df.last_run_metrics.stages[0].total_input_records

    parquet_df = (
        session.read.parquet(str(parquet_path), num_partitions=4)
        .filter(col("id") >= FILTER_THRESHOLD)
        .select("id", "value")
    )
    with timed() as t_parquet:
        parquet_rows = parquet_df.collect()
    parquet_input_records = parquet_df.last_run_metrics.stages[0].total_input_records

    assert len(csv_rows) == len(parquet_rows)

    header = (
        f"{'source':>10} | {'wall time':>10} | "
        f"{'input_records (pre-filter, from metadata)':>42}"
    )
    print(header)
    print("-" * len(header))
    print(f"{'csv':>10} | {t_csv['seconds']:>9.3f}s | {csv_input_records:>42}")
    print(f"{'parquet':>10} | {t_parquet['seconds']:>9.3f}s | {parquet_input_records:>42}")
    print()
    print(f"matching rows returned by both: {len(csv_rows)}")
    print(
        "Note: input_records is identical for both sources here, by design, not a "
        "bug: it is pre-filter row-group metadata (row groups *assigned* to a "
        "partition), not rows actually decoded, so it does not capture row-group "
        "skipping (see execution/metrics.py's own documented imprecision). The real "
        "effect of Parquet's column pruning and row-group-level predicate pushdown "
        "shows up only in wall time above, not in this particular metric."
    )


if __name__ == "__main__":
    main()
