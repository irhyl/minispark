"""Unit tests for Milestone 9's CSV byte-offset seeking: `CSVDataSource.
read()` records one byte offset per partition (`_locate_partition_offsets`)
so `_read_csv_range` can seek straight to its own row range instead of
re-parsing every row before it. See storage/csv.py's module docstring for
the full design and the accepted embedded-newline limitation this test
file also documents (`test_embedded_newline_in_quoted_field_...`).
"""

from __future__ import annotations

import random

import pytest

from minispark.storage.csv import CSVDataSource, _count_rows, _locate_partition_offsets


def write_csv(path, rows_text):
    path.write_text(rows_text, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("num_partitions", [1, 2, 3, 5, 20, 600])
def test_byte_offset_seeking_matches_full_scan_across_partition_counts(tmp_path, num_partitions):
    random.seed(3)
    rows = [(i, random.randint(0, 1000), f"name-{i}") for i in range(137)]
    lines = ["id,age,name"] + [f"{i},{a},{n}" for i, a, n in rows]
    csv_path = write_csv(tmp_path / "big.csv", "\n".join(lines) + "\n")

    dataset = CSVDataSource(csv_path, num_partitions=num_partitions).read()
    got = sorted(dataset.iter_records(), key=lambda r: r["id"])
    expected = [{"id": i, "age": a, "name": n} for i, a, n in rows]
    assert got == expected
    assert dataset.num_partitions() <= num_partitions


def test_file_without_trailing_newline(tmp_path):
    csv_path = write_csv(tmp_path / "no_trailing.csv", "a,b\n1,2\n3,4\n5,6")
    dataset = CSVDataSource(csv_path, num_partitions=3).read()
    got = sorted(dataset.iter_records(), key=lambda r: r["a"])
    assert got == [{"a": 1, "b": 2}, {"a": 3, "b": 4}, {"a": 5, "b": 6}]


def test_quoted_fields_with_commas_still_parse_correctly_with_offset_seeking(tmp_path):
    csv_path = write_csv(
        tmp_path / "quoted.csv",
        'id,label\n1,"hello, world"\n2,"foo, bar, baz"\n3,plain\n',
    )
    dataset = CSVDataSource(csv_path, num_partitions=2).read()
    got = sorted(dataset.iter_records(), key=lambda r: r["id"])
    assert got == [
        {"id": 1, "label": "hello, world"},
        {"id": 2, "label": "foo, bar, baz"},
        {"id": 3, "label": "plain"},
    ]


def test_single_row_file(tmp_path):
    csv_path = write_csv(tmp_path / "single.csv", "x\n42\n")
    dataset = CSVDataSource(csv_path, num_partitions=4).read()
    assert list(dataset.iter_records()) == [{"x": 42}]


def test_header_only_file_produces_no_rows_and_does_not_raise(tmp_path):
    """Regression test: an earlier version of _locate_partition_offsets
    raised KeyError for a header-only file, because the single (empty)
    partition's requested start-row offset was never found (the
    readline() loop hit EOF before reaching it). Fixed by filling in any
    unfound offsets with the end-of-file position, which is harmless
    since a 0-row partition never actually reads using it."""
    csv_path = write_csv(tmp_path / "empty.csv", "x,y\n")
    dataset = CSVDataSource(csv_path, num_partitions=4).read()
    assert list(dataset.iter_records()) == []


def test_partition_is_reiterable_with_offset_seeking(tmp_path):
    csv_path = write_csv(tmp_path / "small.csv", "n\n1\n2\n3\n")
    dataset = CSVDataSource(csv_path, num_partitions=1).read()
    partition = dataset.partition(0)
    assert list(partition) == [{"n": 1}, {"n": 2}, {"n": 3}]
    assert list(partition) == [{"n": 1}, {"n": 2}, {"n": 3}]


def test_columns_projection_still_works_with_offset_seeking(tmp_path):
    csv_path = write_csv(
        tmp_path / "users.csv",
        "name,age,country\nalice,30,US\nbob,17,CA\ncarol,45,US\ndave,19,UK\n",
    )
    dataset = CSVDataSource(csv_path, num_partitions=3).read(columns=["age"])
    rows = list(dataset.iter_records())
    assert all(set(r.keys()) == {"age"} for r in rows)
    assert sorted(r["age"] for r in rows) == [17, 19, 30, 45]


def test_embedded_newline_in_quoted_field_is_a_known_limitation(tmp_path):
    """Documents, rather than hides, the accepted gap from reading one
    line at a time via `f.readline()` (needed so `f.tell()`/`f.seek()`
    stay usable, see storage/csv.py's module docstring): a quoted field
    containing a literal embedded newline is split across two
    `readline()` calls, unlike a single `csv.reader(f)` iterated from the
    top of the file, which does handle this correctly. Depending on how
    the split lands, this either misparses the row or, as here (the
    second half of the split line has fewer fields than the header),
    raises ValueError from `_coerce_row`'s `zip(..., strict=True)`. A
    real, new limitation introduced by switching to readline()-based
    parsing, traded for real byte-offset seeking; asserted here as a
    known, deterministic failure mode, not silently wrong data.
    """
    csv_path = write_csv(
        tmp_path / "embedded_newline.csv", 'id,note\n1,"line one\nline two"\n2,ok\n'
    )
    dataset = CSVDataSource(csv_path, num_partitions=1).read()
    with pytest.raises(ValueError, match="zip"):
        list(dataset.iter_records())


def test_count_rows_excludes_header(tmp_path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("a\n1\n2\n3\n4\n", encoding="utf-8")
    assert _count_rows(csv_path) == 4


def test_count_rows_on_header_only_file_is_zero(tmp_path):
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("a\n", encoding="utf-8")
    assert _count_rows(csv_path) == 0


def test_locate_partition_offsets_lets_each_start_row_seek_directly(tmp_path):
    csv_path = tmp_path / "rows.csv"
    # newline="" on write, matching how CSVDataSource always reads (see
    # module docstring): otherwise Path.write_text()'s default universal-
    # newline translation would write "\r\n" line endings on Windows, and
    # this test's exact-line assertions below compare raw readline()
    # output, not csv-module-parsed fields (which strip "\r" regardless).
    csv_path.write_text("a\n10\n20\n30\n40\n50\n", encoding="utf-8", newline="")
    offsets = _locate_partition_offsets(csv_path, [0, 2, 4])
    assert set(offsets.keys()) == {0, 2, 4}
    with csv_path.open(newline="", encoding="utf-8") as f:
        f.seek(offsets[2])
        assert f.readline() == "30\n"
        f.seek(offsets[4])
        assert f.readline() == "50\n"
