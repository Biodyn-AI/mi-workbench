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


config = AppConfig()
