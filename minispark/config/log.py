"""Structured logging setup.

Format matches the build prompt's example (`TIMESTAMP LEVEL Event key=val`).
Only a handful of events exist to log in Milestone 1 (no query/stage/task
lifecycle yet — that arrives with the scheduler in Milestone 3), so this
module is just the shared formatter/level configuration those later events
will use, applied now to session startup.
"""

from __future__ import annotations

import logging

_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_configured = False


def configure_logging(level: int = logging.INFO) -> None:
    global _configured
    if _configured:
        logging.getLogger("minispark").setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
    logger = logging.getLogger("minispark")
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"minispark.{name}")
