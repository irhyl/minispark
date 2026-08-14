"""Milestone 7 end-to-end example: session.read.parquet() / df.write.parquet(),
with real column pruning and predicate pushdown.

Writes a small Parquet file, reads it back with a filter + a narrow
select(), and prints the physical plan so the pushed-down Scan (fewer
columns than the source file has) is visible directly in `explain(
optimized=True)`'s "Physical Plan" section. Then writes the filtered
result back out and reads it back, round-tripping through Parquet twice.
Run with:

    python examples/parquet.py

Requires the optional `columnar` extra: `pip install -e ".[columnar]"`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from minispark.api.functions import col
from minispark.api.session import MiniSparkSession


def write_sample_parquet(path: Path) -> None:
    table = pa.table(
        {
            "name": ["alice", "bob", "carol", "dave", "erin"],
            "age": [30, 17, 45, 19, 15],
            "country": ["US", "US", "CA", "UK", "UK"],
        }
    )
    pq.write_table(table, str(path), row_group_size=2)


def main() -> None:
    tmpdir = Path(tempfile.mkdtemp(prefix="minispark-parquet-example-"))
    source_path = tmpdir / "users.parquet"
    write_sample_parquet(source_path)

    session = (
        MiniSparkSession.builder.master("local[2]").app_name("parquet_example")
        .get_or_create()
    )

    adults = session.read.parquet(str(source_path)).filter(col("age") >= 18).select("name")

    print("Physical plan (note the pushed-down Scan reads only name/age, not country):")
    adults.explain(optimized=True)
    print()

    print("Result:")
    adults.show()

    out_path = tmpdir / "adults"
    adults.write.parquet(str(out_path))
    print(f"\nWrote result to {out_path}, reading it back:")
    session.read.parquet(str(out_path)).show()


if __name__ == "__main__":
    main()
