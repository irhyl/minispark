"""Shared helpers for the scripts in this directory.

None of these benchmarks claim to be a controlled, isolated measurement
environment: they run on whatever machine (and however busy it happens
to be at the time) `python benchmarks/<script>.py` is invoked on, a
single trial each, not an averaged/warmed-up/isolated series. Numbers
here show the *qualitative* pattern each milestone's design predicts
(e.g. "more workers finishes a shuffle-heavy query faster," "Parquet
reads fewer bytes than CSV for the same filtered query"), not an
absolute, reproducible performance claim; see docs/benchmarks.md for the
actual recorded numbers and this same caveat repeated there.
"""

from __future__ import annotations

import platform
import sys
import time
from contextlib import contextmanager

import psutil


def machine_info() -> str:
    return (
        f"Python {sys.version.split()[0]}, {platform.platform()}, "
        f"{psutil.cpu_count(logical=True)} logical CPUs, "
        f"{psutil.virtual_memory().total / 1e9:.1f} GB RAM"
    )


@contextmanager
def timed():
    start = time.perf_counter()
    result = {}
    yield result
    result["seconds"] = time.perf_counter() - start
