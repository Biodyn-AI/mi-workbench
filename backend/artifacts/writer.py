"""Artifact writer utilities."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.models import ArtifactInfo, RunMeta


def ensure_run_dirs(workspace_path: str, run_id: str) -> tuple[Path, Path]:
    """Create and return (run_dir, latest_iteration_dir)."""
    run_dir = Path(workspace_path) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, run_dir


def _iteration_dir(run_dir: Path, iteration_number: int) -> Path:
    d = run_dir / f"iter_{iteration_number:04d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_artifact(
    run_dir: str | Path,
    iteration_number: int,
    artifact_name: str,
    content: str,
) -> Path:
    """Write an artifact file and return its path."""
    run_dir = Path(run_dir)
    dest_dir = _iteration_dir(run_dir, iteration_number)
    artifact_path = dest_dir / artifact_name
    artifact_path.write_text(content)
    return artifact_path


def write_run_meta(run_dir: str | Path, meta: RunMeta) -> Path:
    """Write run metadata JSON to run_dir/run_meta.json."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "run_meta.json"
    meta_path.write_text(meta.model_dump_json(indent=2))
    return meta_path


def get_latest_artifacts(workspace_path: str) -> list[ArtifactInfo]:
    """Scan workspace for the latest artifacts across all runs."""
    ws = Path(workspace_path)
    runs_dir = ws / "runs"
    if not runs_dir.exists():
        return []
    artifacts: list[ArtifactInfo] = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        run_id = run_dir.name
        # Find latest iteration dir
        iter_dirs = sorted(
            [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("iter_")],
            reverse=True,
        )
        for iter_dir in iter_dirs:
            try:
                iter_num = int(iter_dir.name.split("_")[1])
            except (IndexError, ValueError):
                continue
            for f in iter_dir.iterdir():
                if f.is_file():
                    stat = f.stat()
                    atype = _infer_type(f.name)
                    artifacts.append(ArtifactInfo(
                        name=f.name,
                        path=str(f),
                        artifact_type=atype,
                        iteration=iter_num,
                        run_id=run_id,
                        size_bytes=stat.st_size,
                        modified_at=datetime.fromtimestamp(stat.st_mtime),
                    ))
    return artifacts


def _infer_type(filename: str) -> str:
    lower = filename.lower()
    if "mech" in lower:
        return "MECH"
    if "eval" in lower:
        return "EVAL"
    if "xp" in lower or "experiment" in lower:
        return "XP"
    if "method" in lower:
        return "METHOD"
    if "knowledge" in lower:
        return "KNOWLEDGE"
    if "review" in lower:
        return "REVIEW"
    return "OTHER"
