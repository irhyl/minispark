"""MiniSpark data types.

Milestone 1 uses a small, closed set of scalar types sufficient for CSV /
in-memory data. This is intentionally row-oriented Python objects (int,
float, str, bool, None) at this stage — Milestone 7 introduces a columnar
(Arrow-backed) representation. DataType exists now so Schema/Field have a
real type system to validate against once the analyzer (Milestone 2) lands,
rather than bolting types on after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass


class DataType:
    """Base class for MiniSpark scalar data types."""

    name: str = "unknown"

    def __repr__(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DataType) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)


@dataclass(frozen=True, eq=False)
class IntType(DataType):
    name: str = "int"


@dataclass(frozen=True, eq=False)
class FloatType(DataType):
    name: str = "float"


@dataclass(frozen=True, eq=False)
class StringType(DataType):
    name: str = "string"


@dataclass(frozen=True, eq=False)
class BoolType(DataType):
    name: str = "bool"


@dataclass(frozen=True, eq=False)
class NullType(DataType):
    name: str = "null"


# Singletons: types are stateless, so reuse one instance per type rather
# than constructing new objects everywhere a schema is built.
INT = IntType()
FLOAT = FloatType()
STRING = StringType()
BOOL = BoolType()
NULL = NullType()


_PYTHON_TYPE_TO_DATATYPE: dict[type, DataType] = {
    bool: BOOL,  # must precede int: bool is a subclass of int in Python
    int: INT,
    float: FLOAT,
    str: STRING,
    type(None): NULL,
}


def infer_type(value: object) -> DataType:
    """Infer a DataType from a Python value. Falls back to STRING."""
    for py_type, data_type in _PYTHON_TYPE_TO_DATATYPE.items():
        if isinstance(value, py_type):
            return data_type
    return STRING
