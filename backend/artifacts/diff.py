"""Diff computation and storage utilities."""
from __future__ import annotations

import difflib
from pathlib import Path


def compute_diff(old_content: str, new_content: str) -> str:
    """Return a unified diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile="old",
        tofile="new",
    )
    return "".join(diff)


def save_diff_patch(run_dir: str | Path, iteration: int, diff_text: str) -> Path:
    """Save a diff patch to the iteration directory."""
    run_dir = Path(run_dir)
    iter_dir = run_dir / f"iter_{iteration:04d}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    patch_path = iter_dir / "changes.patch"
    patch_path.write_text(diff_text)
    return patch_path
