"""Artifact content validation utilities."""
from __future__ import annotations

import re


def validate_mech_md(content: str) -> list[str]:
    """Validate a mechanistic insight markdown artifact."""
    issues: list[str] = []
    if not content.strip():
        issues.append("Content is empty")
        return issues
    if "# " not in content:
        issues.append("Missing top-level heading")
    required_sections = ["hypothesis", "evidence", "conclusion"]
    lower = content.lower()
    for section in required_sections:
        if section not in lower:
            issues.append(f"Missing expected section: {section}")
    return issues


def validate_eval_md(content: str) -> list[str]:
    """Validate an evaluation markdown artifact."""
    issues: list[str] = []
    if not content.strip():
        issues.append("Content is empty")
        return issues
    if "# " not in content:
        issues.append("Missing top-level heading")
    required_sections = ["metric", "result"]
    lower = content.lower()
    for section in required_sections:
        if section not in lower:
            issues.append(f"Missing expected section: {section}")
    return issues


_VALIDATORS = {
    "MECH": validate_mech_md,
    "EVAL": validate_eval_md,
}


def validate_artifact_structure(artifact_type: str, content: str) -> tuple[bool, list[str]]:
    """Validate artifact content based on its type. Returns (valid, issues)."""
    validator = _VALIDATORS.get(artifact_type.upper())
    if validator is None:
        return True, []
    issues = validator(content)
    return len(issues) == 0, issues
