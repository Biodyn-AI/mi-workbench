"""Application configuration for MI-Workbench backend."""
from __future__ import annotations

import os
from pathlib import Path


WORKBENCH_DIR = Path(__file__).resolve().parent.parent  # mi-workbench root


class AppConfig:
    """Centralised configuration loaded from environment variables."""

    def __init__(self) -> None:
        self.db_path: str = os.environ.get(
            "MIW_DB_PATH",
            str(WORKBENCH_DIR / "data" / "mi_workbench.db"),
        )
        self.host: str = os.environ.get("MIW_HOST", "127.0.0.1")
        self.port: int = int(os.environ.get("MIW_PORT", "8000"))
        self.workspace_base_path: str = os.environ.get(
            "MIW_WORKSPACE_BASE",
            str(WORKBENCH_DIR / "workspaces"),
        )
        self.prompts_dir: str = os.environ.get(
            "MIW_PROMPTS_DIR",
            str(WORKBENCH_DIR / "prompts"),
        )
        self.loops_dir: str = os.environ.get(
            "MIW_LOOPS_DIR",
            str(WORKBENCH_DIR / "loops"),
        )
        self.log_level: str = os.environ.get("MIW_LOG_LEVEL", "INFO")
        # SQLite lock handling (see backend/database.py). ``db_busy_timeout``
        # is the per-statement busy timeout (seconds) set on every connection;
        # write transactions that still hit "database is locked/busy" are
        # retried up to ``db_write_attempts`` times with exponential backoff
        # and jitter (``db_retry_base_delay`` doubling, each sleep capped at
        # ``db_retry_max_delay``); no new attempt starts once
        # ``db_retry_deadline`` seconds have passed since the first one.
        self.db_busy_timeout: float = _env_float("MIW_DB_BUSY_TIMEOUT", 30.0)
        self.db_write_attempts: int = max(1, int(_env_float("MIW_DB_WRITE_ATTEMPTS", 8)))
        self.db_retry_base_delay: float = _env_float("MIW_DB_RETRY_BASE_DELAY", 0.05)
        self.db_retry_max_delay: float = _env_float("MIW_DB_RETRY_MAX_DELAY", 2.0)
        self.db_retry_deadline: float = _env_float("MIW_DB_RETRY_DEADLINE", 300.0)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return float(default)
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {raw!r}")
    return value


config = AppConfig()
