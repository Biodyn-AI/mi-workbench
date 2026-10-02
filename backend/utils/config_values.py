"""Strict coercion of run-config values that may arrive as strings.

Run configs come from JSON (API), ``key=value`` pairs (CLI, where a value that
is not valid JSON stays a string, e.g. ``False``) and loop YAML. ``bool("false")``
is True, so boolean keys must never be coerced with ``bool()``.
"""
from __future__ import annotations

from typing import Any

_TRUE = ("1", "true", "yes", "on")
_FALSE = ("0", "false", "no", "off")


def parse_bool(value: Any, name: str = "value") -> bool:
    """Booleans, 0/1 numbers and the strings true/false/1/0/yes/no/on/off.

    Raises ``ValueError`` for any other string or type, so a typo fails
    loudly instead of silently meaning True.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in _TRUE:
            return True
        if v in _FALSE:
            return False
    raise ValueError(f"{name} must be a boolean (true/false), got {value!r}")
