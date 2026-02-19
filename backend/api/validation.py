"""Input validation helpers for API endpoints."""
import re
from fastapi import HTTPException

# Valid ID pattern: alphanumeric + hyphens + underscores, 1-64 chars
_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Max task length
MAX_TASK_LENGTH = 50_000  # 50KB
MAX_CONFIG_KEYS = 50


def validate_id(value: str, field_name: str = "id") -> str:
    """Validate that an ID is safe (no injection, reasonable length)."""
    if not _ID_PATTERN.match(value):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid {field_name}: must be 1-64 alphanumeric/hyphen/underscore characters",
        )
    return value


def validate_task(task: str) -> str:
    """Validate task string length."""
    if len(task) > MAX_TASK_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Task too long: {len(task)} chars (max {MAX_TASK_LENGTH})",
        )
    return task


def validate_config(config: dict) -> dict:
    """Validate config overrides are reasonable."""
    if len(config) > MAX_CONFIG_KEYS:
        raise HTTPException(
            status_code=422,
            detail=f"Too many config keys: {len(config)} (max {MAX_CONFIG_KEYS})",
        )
    return config


def validate_workspace_name(name: str) -> str:
    """Validate workspace name is non-empty and reasonable length."""
    if not name or not name.strip():
        raise HTTPException(
            status_code=422,
            detail="Workspace name must not be empty",
        )
    if len(name) > 200:
        raise HTTPException(
            status_code=422,
            detail=f"Workspace name too long: {len(name)} chars (max 200)",
        )
    return name


def validate_workspace_path(path: str) -> str:
    """Validate workspace path is non-empty and reasonable length."""
    if not path or not path.strip():
        raise HTTPException(
            status_code=422,
            detail="Workspace path must not be empty",
        )
    if len(path) > 500:
        raise HTTPException(
            status_code=422,
            detail=f"Workspace path too long: {len(path)} chars (max 500)",
        )
    return path
