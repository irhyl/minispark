"""Unit tests for sql/tokenizer.py."""

from __future__ import annotations

import pytest

from minispark.sql.tokenizer import SqlLexError, TokenType, tokenize


def _types(text: str) -> list[TokenType]:
    return [t.type for t in tokenize(text)]


def _values(text: str) -> list[str]:
    return [t.value for t in tokenize(text)]


def test_keywords_are_case_insensitive_and_uppercased():
    tokens = tokenize("select * from Users")
    assert tokens[0].value == "SELECT"
    assert tokens[0].type is TokenType.KEYWORD
    assert tokens[2].value == "FROM"
    assert tokens[2].type is TokenType.KEYWORD


def test_identifiers_preserve_original_case():
    tokens = tokenize("SELECT Name FROM Users")
    identifiers = [t.value for t in tokens if t.type is TokenType.IDENTIFIER]
    assert identifiers == ["Name", "Users"]


def test_numbers_integer_and_float():
    tokens = [t for t in tokenize("SELECT 1, 2.5") if t.type is TokenType.NUMBER]
    assert [t.value for t in tokens] == ["1", "2.5"]


def test_string_literal_with_escaped_quote():
    tokens = tokenize("SELECT 'it''s' FROM t")
    strings = [t.value for t in tokens if t.type is TokenType.STRING]
    assert strings == ["it's"]


def test_unterminated_string_raises():
    with pytest.raises(SqlLexError):
        tokenize("SELECT 'oops FROM t")


def test_multi_char_operators_recognized():
    values = _values("a <= b >= c != d <> e")
    ops = [v for v in values if v in ("<=", ">=", "!=", "<>")]
    assert ops == ["<=", ">=", "!=", "<>"]


def test_single_char_operators_not_confused_with_multi_char():
    values = _values("a < b > c = d")
    ops = [v for v in values if v in ("<", ">", "=")]
    assert ops == ["<", ">", "="]


def test_line_comment_is_skipped():
    tokens = tokenize("SELECT 1 -- this is a comment\nFROM t")
    values = [t.value for t in tokens if t.type is not TokenType.EOF]
    assert values == ["SELECT", "1", "FROM", "t"]


def test_punctuation_tokens():
    tokens = [t for t in tokenize("f(a, b.c)") if t.type is TokenType.PUNCTUATION]
    assert [t.value for t in tokens] == ["(", ",", ".", ")"]


def test_unexpected_character_raises():
    with pytest.raises(SqlLexError):
        tokenize("SELECT # FROM t")


def test_ends_with_eof_token():
    tokens = tokenize("SELECT 1")
    assert tokens[-1].type is TokenType.EOF


def test_qualified_name_dot_is_punctuation_not_part_of_number():
    # A leading-dot number like ".5" should not be confused with "t.col".
    tokens = tokenize("t.col")
    assert _values("t.col") == ["t", ".", "col", ""]
    assert tokens[0].type is TokenType.IDENTIFIER
    assert tokens[1].type is TokenType.PUNCTUATION
