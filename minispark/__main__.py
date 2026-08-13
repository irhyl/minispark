"""Minimal CLI entry point.

Only `--version` is implemented. The build prompt's full command set
(`run`, `explain`, `benchmark`, `profile`, `workers`) is added alongside
the milestones that give each command something real to do — a `run`
command before there is a scheduler would just be a facade over the naive
executor pretending to be more than it is.
"""

from __future__ import annotations

import argparse
import sys

from minispark import __version__


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="minispark")
    parser.add_argument("--version", action="store_true", help="print the MiniSpark version")
    args = parser.parse_args(argv)

    if args.version:
        print(f"minispark {__version__}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
