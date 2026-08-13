from minispark.storage.memory import MemoryDataSource


def test_infers_schema_and_partitions_rows():
    records = [{"id": i, "name": f"user{i}"} for i in range(10)]
    source = MemoryDataSource(records, num_partitions=4)
    dataset = source.read()

    assert dataset.schema.field_names() == ["id", "name"]
    assert dataset.num_partitions() <= 4
    assert dataset.row_count() == 10
    assert sorted(r["id"] for r in dataset.iter_records()) == list(range(10))


def test_explicit_schema_is_respected():
    from minispark.core.schema import Field, Schema
    from minispark.core.types import INT, STRING

    schema = Schema([Field("id", INT), Field("name", STRING)])
    records = [{"id": 1, "name": "a"}]
    source = MemoryDataSource(records, schema=schema, num_partitions=2)
    dataset = source.read()
    assert dataset.schema == schema
