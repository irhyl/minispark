"""Core data model: DataType, Schema, Field, Record, Partition, Dataset.

This package is a deliberate deviation from the structure sketched in the
build prompt (which only listed a top-level `storage/` package). The data
model — what a row, a partition, and a dataset *are* — is a concern shared
by the DataFrame API, the logical/physical planners, and the execution
engine. Putting it under `storage/` would wrongly suggest it belongs to the
I/O layer. `core/` has no dependency on `storage/`, `logical/`, or
`api/`; those depend on it.
"""
