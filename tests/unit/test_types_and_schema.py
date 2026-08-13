import pytest

from minispark.core.schema import Field, Schema
from minispark.core.types import BOOL, FLOAT, INT, STRING, infer_type


def test_infer_type():
    assert infer_type(1) == INT
    assert infer_type(1.5) == FLOAT
    assert infer_type("x") == STRING
    assert infer_type(True) == BOOL


def test_schema_lookup():
    schema = Schema([Field("id", INT, nullable=False), Field("name", STRING)])
    assert schema.field_names() == ["id", "name"]
    assert schema.has_field("id")
    assert not schema.has_field("missing")
    assert schema.get_field("id").data_type == INT


def test_schema_get_field_missing_raises():
    schema = Schema([Field("id", INT)])
    with pytest.raises(KeyError):
        schema.get_field("nope")


def test_schema_select_reorders():
    schema = Schema([Field("id", INT), Field("name", STRING), Field("age", INT)])
    sub = schema.select(["age", "id"])
    assert sub.field_names() == ["age", "id"]


def test_schema_rejects_duplicate_field_names():
    with pytest.raises(ValueError):
        Schema([Field("id", INT), Field("id", STRING)])
