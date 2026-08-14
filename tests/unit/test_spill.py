import os

from minispark.physical.spill import (
    cleanup_spill_dir,
    make_spill_dir,
    read_spill_file,
    write_spill_file,
)


def test_write_then_read_round_trips_arbitrary_picklable_items():
    directory = make_spill_dir("test-spill-")
    try:
        items = [{"a": 1}, ("tuple", 2), None, [1, 2, 3], "text"]
        path = write_spill_file(directory, "run_0.pkl", items)
        assert list(read_spill_file(path)) == items
    finally:
        cleanup_spill_dir(directory)


def test_write_spill_file_empty_iterable_reads_back_empty():
    directory = make_spill_dir("test-spill-")
    try:
        path = write_spill_file(directory, "run_0.pkl", [])
        assert list(read_spill_file(path)) == []
    finally:
        cleanup_spill_dir(directory)


def test_cleanup_spill_dir_removes_the_directory_and_its_contents():
    directory = make_spill_dir("test-spill-")
    write_spill_file(directory, "run_0.pkl", [1, 2, 3])
    assert os.path.isdir(directory)
    cleanup_spill_dir(directory)
    assert not os.path.exists(directory)


def test_cleanup_spill_dir_tolerates_none():
    cleanup_spill_dir(None)
