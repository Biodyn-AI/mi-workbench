"""Structured logging configuration for MI-Workbench."""
from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import datetime
from typing import Optional


_current_run_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "run_id", default=None
)
_current_iteration: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "iteration", default=None
)


class StructuredFormatter(logging.Formatter):
    """JSON log formatter with correlation IDs."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in ("run_id", "iteration", "role", "adapter", "workspace_id", "duration_s"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry)


class RunContextFilter(logging.Filter):
    """Injects run context (run_id, iteration) from contextvars."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "run_id") or getattr(record, "run_id", None) is None:
            record.run_id = _current_run_id.get(None)  # type: ignore[attr-defined]
        if not hasattr(record, "iteration") or getattr(record, "iteration", None) is None:
            record.iteration = _current_iteration.get(None)  # type: ignore[attr-defined]
        return True


def set_run_context(run_id: str, iteration: Optional[int] = None) -> None:
    """Set the current run context for log correlation."""
    _current_run_id.set(run_id)
    _current_iteration.set(iteration)


def clear_run_context() -> None:
    """Clear the current run context."""
    _current_run_id.set(None)
    _current_iteration.set(None)


def setup_logging(level: str = "INFO", json_format: bool = True) -> None:
    """Configure logging for the application."""
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )

    handler.addFilter(RunContextFilter())
    root.handlers = [handler]
