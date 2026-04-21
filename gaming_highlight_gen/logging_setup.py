"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def setup_logging(level: str, fmt: str) -> None:
    """Configure structlog for application-wide structured logging.

    Args:
        level: Log level string (e.g. ``"INFO"``, ``"DEBUG"``).
        fmt: Renderer mode – ``"json"`` for :class:`structlog.processors.JSONRenderer`
            (production) or ``"console"`` for
            :class:`structlog.dev.ConsoleRenderer` with colours (development).
            Every log record includes ``timestamp``, ``level``, ``event``, and
            ``logger`` (module name).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[Any] = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.ExceptionRenderer(),
    ]

    renderer: Any
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Always update the root logger level, even if basicConfig already ran.
    root_logger = logging.getLogger()
    for handler in root_logger.handlers or [logging.StreamHandler(sys.stdout)]:
        handler.setFormatter(logging.Formatter("%(message)s"))
    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)
