"""Data sources: turn external data (files, in-memory rows) into a Dataset.

This layer depends on `minispark.core` only. It has no knowledge of the
DataFrame API or logical plans — `Scan` nodes hold a Dataset produced by a
DataSource, not a DataSource itself, keeping the dependency one-directional.
"""
