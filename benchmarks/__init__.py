"""Benchmark scripts (Milestone 8), run as `python -m benchmarks.<name>`
from the repository root, not `python benchmarks/<name>.py` directly:
each script imports shared helpers from `benchmarks._common`, which only
resolves when `benchmarks` itself is importable as a package (i.e. the
repository root, not this directory, is on `sys.path`), exactly what
`-m` gives and a bare script invocation does not.

None of these are a controlled, isolated benchmark environment; see
`_common.py`'s module docstring and docs/benchmarks.md for the honesty
caveat repeated at every result these scripts produce.
"""

from __future__ import annotations
