"""SQL tokenizer: turns a SQL string into a flat list of Tokens.

Hand-written, not a generated lexer: the build spec explicitly allows "a
lightweight parser if needed for SQL," and the grammar this project
supports (see sql/parser.py) is small enough that a generated lexer
would be more machinery than the problem needs, the same reasoning
`optimizer/rules.py` gives for not having a generic visitor abstraction
over six logical node types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

KEYWORDS = {
    "SELECT", "FROM", "WHERE", "GROUP", "BY", "HAVING", "ORDER", "ASC",
    "DESC", "JOIN", "ON", "AS", "AND", "OR", "NOT", "NULL", "IS", "TRUE",
    "FALSE",
}


class TokenType(Enum):
    KEYWORD = auto()
    IDENTIFIER = auto()
    NUMBER = auto()
    STRING = auto()
    OPERATOR = auto()
    PUNCTUATION = auto()
    EOF = auto()


@dataclass(frozen=True)
class Token:
    type: TokenType
    value: str
    position: int


class SqlSyntaxError(Exception):
    """Raised for a malformed SQL string, both by the tokenizer (an
    unterminated string literal, an unrecognized character) and by the
    parser (see sql/parser.py's SqlParseError, a subclass): a caller can
    catch just this one type to mean "the SQL text itself was bad,"
    without needing to know which stage of parsing found the problem.
    """


class SqlLexError(SqlSyntaxError):
    pass


_MULTI_CHAR_OPERATORS = ("<=", ">=", "!=", "<>")
_SINGLE_CHAR_OPERATORS = "=<>+-*/"
_PUNCTUATION = "(),."


def tokenize(text: str) -> list[Token]:
    tokens: list[Token] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch.isspace():
            i += 1
            continue
        if ch == "-" and i + 1 < n and text[i + 1] == "-":
            # Line comment: skip to end of line.
            newline = text.find("\n", i)
            i = n if newline == -1 else newline + 1
            continue
        if ch == "'":
            value, i = _read_string(text, i)
            tokens.append(Token(TokenType.STRING, value, i))
            continue
        if ch.isdigit() or (ch == "." and i + 1 < n and text[i + 1].isdigit()):
            value, i = _read_number(text, i)
            tokens.append(Token(TokenType.NUMBER, value, i))
            continue
        if ch.isalpha() or ch == "_":
            value, i = _read_identifier(text, i)
            if value.upper() in KEYWORDS:
                tokens.append(Token(TokenType.KEYWORD, value.upper(), i))
            else:
                tokens.append(Token(TokenType.IDENTIFIER, value, i))
            continue
        multi = text[i : i + 2]
        if multi in _MULTI_CHAR_OPERATORS:
            tokens.append(Token(TokenType.OPERATOR, multi, i))
            i += 2
            continue
        if ch in _SINGLE_CHAR_OPERATORS:
            tokens.append(Token(TokenType.OPERATOR, ch, i))
            i += 1
            continue
        if ch in _PUNCTUATION:
            tokens.append(Token(TokenType.PUNCTUATION, ch, i))
            i += 1
            continue
        raise SqlLexError(f"Unexpected character {ch!r} at position {i} in: {text!r}")
    tokens.append(Token(TokenType.EOF, "", n))
    return tokens


def _read_string(text: str, start: int) -> tuple[str, int]:
    i = start + 1
    n = len(text)
    chars: list[str] = []
    while True:
        if i >= n:
            raise SqlLexError(f"Unterminated string literal starting at position {start}")
        ch = text[i]
        if ch == "'":
            if i + 1 < n and text[i + 1] == "'":
                chars.append("'")
                i += 2
                continue
            return "".join(chars), i + 1
        chars.append(ch)
        i += 1


def _read_number(text: str, start: int) -> tuple[str, int]:
    i = start
    n = len(text)
    saw_dot = False
    while i < n and (text[i].isdigit() or (text[i] == "." and not saw_dot)):
        if text[i] == ".":
            saw_dot = True
        i += 1
    return text[start:i], i


def _read_identifier(text: str, start: int) -> tuple[str, int]:
    i = start
    n = len(text)
    while i < n and (text[i].isalnum() or text[i] == "_"):
        i += 1
    return text[start:i], i
