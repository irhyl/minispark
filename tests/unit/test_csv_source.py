import pytest

from minispark.storage.csv import CSVDataSource, read_csv


def write_csv(path, rows_text):
    path.write_text(rows_text, encoding="utf-8")
    return str(path)


def test_schema_inference_and_partitioned_read(tmp_path):
    csv_path = write_csv(
        tmp_path / "users.csv",
        "name,age,country\n"
        "alice,30,US\n"
        "bob,17,CA\n"
        "carol,45,US\n"
        "dave,,UK\n",
    )
    dataset = read_csv(csv_path, num_partitions=2)

    assert dataset.schema.field_names() == ["name", "age", "country"]
    assert dataset.schema.get_field("age").data_type.name == "int"
    assert dataset.num_partitions() == 2
    assert dataset.row_count() == 4

    rows = list(dataset.iter_records())
    ages = {r["name"]: r["age"] for r in rows}
    assert ages["alice"] == 30
    assert ages["dave"] is None  # empty field parses to None


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        CSVDataSource(str(tmp_path / "does_not_exist.csv"))


def test_partition_is_reiterable(tmp_path):
    csv_path = write_csv(tmp_path / "small.csv", "n\n1\n2\n3\n")
    dataset = read_csv(csv_path, num_partitions=1)
    partition = dataset.partition(0)
    assert list(partition) == [{"n": 1}, {"n": 2}, {"n": 3}]
    # streamed from disk each time -> re-iterating must work, not just
    # not-crash
    assert list(partition) == [{"n": 1}, {"n": 2}, {"n": 3}]
