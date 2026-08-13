"""Field and Schema: the structural description of a Dataset's rows."""

from __future__ import annotations

from dataclasses import dataclass

from minispark.core.types import DataType


@dataclass(frozen=True)
class Field:
    name: str
    data_type: DataType
    nullable: bool = True

    def __repr__(self) -> str:
        suffix = "" if self.nullable else " NOT NULL"
        return f"{self.name}: {self.data_type}{suffix}"


class Schema:
    """An ordered collection of Fields, with fast name lookup."""

    def __init__(self, fields: list[Field]):
        self.fields = list(fields)
        self._by_name = {f.name: f for f in self.fields}
        if len(self._by_name) != len(self.fields):
            raise ValueError(f"Schema has duplicate field names: {[f.name for f in self.fields]}")

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]

    def has_field(self, name: str) -> bool:
        return name in self._by_name

    def get_field(self, name: str) -> Field:
        try:
            return self._by_name[name]
        except KeyError:
            raise KeyError(
                f"Column '{name}' not found in schema. Available columns: {self.field_names()}"
            ) from None

    def select(self, names: list[str]) -> Schema:
        """A new Schema containing only the given field names, in that order."""
        return Schema([self.get_field(n) for n in names])

    def __len__(self) -> int:
        return len(self.fields)

    def __iter__(self):
        return iter(self.fields)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Schema) and self.fields == other.fields

    def __repr__(self) -> str:
        body = ", ".join(repr(f) for f in self.fields)
        return f"Schema({body})"
