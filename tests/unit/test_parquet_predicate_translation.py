"""Unit tests for storage/parquet.py's translate_predicate(): the
best-effort MiniSpark Expression -> pyarrow.dataset.Expression translator
that makes real predicate pushdown to Parquet possible.

Skipped entirely (via importorskip) if pyarrow is not installed: pyarrow
is an optional extra (see pyproject.toml's `columnar` extra), and the
core test suite must still pass without it, matching every other
optional-dependency test file's convention in this repo.
"""

from __future__ import annotations

import pytest

pa_dataset = pytest.importorskip("pyarrow.dataset")

from minispark.expressions.binary import (  # noqa: E402
    Add,
    And,
    Equal,
    GreaterEqual,
    GreaterThan,
    LessEqual,
    LessThan,
    NotEqual,
    Or,
)
from minispark.expressions.column import Column  # noqa: E402
from minispark.expressions.literal import Literal  # noqa: E402
from minispark.expressions.predicates import IsNotNull, IsNull, Not  # noqa: E402
from minispark.storage.parquet import translate_predicate  # noqa: E402


def _matches(table, filt):
    import pyarrow as pa

    ds = pa_dataset.dataset(pa.table(table))
    return ds.to_table(filter=filt).to_pylist()


COMPARISONS = [
    (GreaterThan, [1, 2, 3], [3]),
    (GreaterEqual, [1, 2, 3], [2, 3]),
    (LessThan, [1, 2, 3], [1]),
    (LessEqual, [1, 2, 3], [1, 2]),
    (Equal, [1, 2, 3], [2]),
    (NotEqual, [1, 2, 3], [1, 3]),
]


@pytest.mark.parametrize("expr_cls,values,expected", COMPARISONS)
def test_comparisons_translate_and_filter_correctly(expr_cls, values, expected):
    condition = expr_cls(Column("a"), Literal(2))
    filt = translate_predicate(condition)
    assert filt is not None
    rows = _matches({"a": values}, filt)
    assert sorted(r["a"] for r in rows) == expected


def test_and_pushes_both_sides_when_both_translate():
    condition = And(GreaterThan(Column("a"), Literal(1)), LessThan(Column("a"), Literal(3)))
    filt = translate_predicate(condition)
    rows = _matches({"a": [1, 2, 3]}, filt)
    assert [r["a"] for r in rows] == [2]


def test_and_pushes_only_the_translatable_side():
    """A safe superset: the untranslatable side (arithmetic) is simply
    not included, never causing rows to be wrongly excluded."""
    condition = And(
        GreaterThan(Column("a"), Literal(1)),
        GreaterThan(Add(Column("a"), Column("a")), Literal(100)),
    )
    filt = translate_predicate(condition)
    rows = _matches({"a": [1, 2, 3]}, filt)
    assert sorted(r["a"] for r in rows) == [2, 3]  # only a > 1 was actually pushed


def test_or_does_not_push_when_one_side_is_untranslatable():
    """Pushing only one side of an OR could wrongly exclude rows the
    untranslated side would have kept, so it must not push at all."""
    condition = Or(
        GreaterThan(Column("a"), Literal(1)),
        GreaterThan(Add(Column("a"), Column("a")), Literal(100)),
    )
    assert translate_predicate(condition) is None


def test_or_pushes_when_both_sides_translate():
    condition = Or(Equal(Column("a"), Literal(1)), Equal(Column("a"), Literal(3)))
    filt = translate_predicate(condition)
    rows = _matches({"a": [1, 2, 3]}, filt)
    assert sorted(r["a"] for r in rows) == [1, 3]


def test_not_translates():
    condition = Not(GreaterThan(Column("a"), Literal(2)))
    filt = translate_predicate(condition)
    rows = _matches({"a": [1, 2, 3]}, filt)
    assert sorted(r["a"] for r in rows) == [1, 2]


def test_is_null_and_is_not_null_translate():
    is_null_filt = translate_predicate(IsNull(Column("a")))
    is_not_null_filt = translate_predicate(IsNotNull(Column("a")))
    rows_null = _matches({"a": [1, None, 3]}, is_null_filt)
    rows_not_null = _matches({"a": [1, None, 3]}, is_not_null_filt)
    assert rows_null == [{"a": None}]
    assert sorted(r["a"] for r in rows_not_null) == [1, 3]


def test_arithmetic_does_not_translate():
    condition = GreaterThan(Add(Column("a"), Column("b")), Literal(5))
    assert translate_predicate(condition) is None


def test_none_literal_comparison_does_not_translate():
    """pyarrow's `== null` uses SQL three-valued logic (never matches,
    even for a null row), but MiniSpark's row engine evaluates `==` as
    plain Python equality (None == None is True). Pushing this down
    would wrongly exclude rows the row-level Filter would have kept, so
    it must not be translated at all, not even partially."""
    assert translate_predicate(Equal(Column("a"), Literal(None))) is None
    assert translate_predicate(NotEqual(Column("a"), Literal(None))) is None


def test_column_to_column_comparison_translates():
    """Not just Column-vs-Literal: both operands may be Columns, since
    pyarrow's comparison operators accept another field expression on
    the right just as readily as a scalar."""
    filt = translate_predicate(GreaterThan(Column("a"), Column("b")))
    rows = _matches({"a": [1, 5, 3], "b": [2, 2, 2]}, filt)
    assert sorted(r["a"] for r in rows) == [3, 5]


def test_literal_on_the_left_does_not_translate():
    """The left operand must be a Column for the translation to have a
    field to push a comparison against; `5 > col("a")` (Literal first)
    is not rewritten to the equivalent `col("a") < 5`, a known, narrow
    gap rather than something worth a rewrite pass for."""
    assert translate_predicate(GreaterThan(Literal(5), Column("a"))) is None
