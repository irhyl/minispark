"""Record: the row-oriented unit of data in Milestone 1.

A Record is a plain `dict[str, Any]` mapping column name to value. This is
the simplest possible representation and is what lets the naive executor
(minispark/execution/executor.py) stay a straightforward tree-walking
interpreter. It is deliberately not columnar yet: Milestone 7 adds a
columnar (Arrow-backed) Partition implementation for the physical
execution path, at which point Record remains useful as the row-oriented
interchange format (e.g. for collect()/show()).
"""

from __future__ import annotations

from typing import Any

Record = dict[str, Any]
