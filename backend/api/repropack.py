"""Reproducibility Package API routes."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from backend.database import get_run, get_workspace
from backend.repropack.generator import ReproPackGenerator

router = APIRouter(prefix="/repropack", tags=["repropack"])

_generator = ReproPackGenerator()


def _resolve_run_dir(workspace_path: str, run_id: str) -> Path:
    """Resolve the run directory inside the workspace."""
    run_dir = Path(workspace_path) / "runs" / run_id
    if not run_dir.is_dir():
        # Fallback: try direct path
        run_dir = Path(workspace_path) / run_id
    return run_dir


_MAX_FILE_MB = Query(None, ge=0, description=(
    "Per-file size cap (MB) for iteration files and persisted execution outputs; "
    "larger files are skipped and listed in the package README (0 = no cap)"))


@router.post("/{run_id}")
async def generate_repropack(run_id: str,
                             max_file_mb: Optional[float] = _MAX_FILE_MB) -> StreamingResponse:
    """Generate and download a reproducibility package zip for a run."""
    run_state = await get_run(run_id)
    if run_state is None:
        raise HTTPException(status_code=404, detail="Run not found")

    workspace = await get_workspace(run_state.workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    run_dir = _resolve_run_dir(workspace.path, run_id)

    zip_bytes = _generator.generate(
        run_dir=str(run_dir),
        workspace_path=workspace.path,
        run_state=run_state,
        max_file_mb=max_file_mb,
    )

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="repropack-{run_id}.zip"',
            "Content-Length": str(len(zip_bytes)),
        },
    )


@router.get("/{run_id}/preview")
async def preview_repropack(run_id: str, max_file_mb: Optional[float] = _MAX_FILE_MB) -> dict:
    """Preview what would be included in the repro pack (file list + sizes)."""
    run_state = await get_run(run_id)
    if run_state is None:
        raise HTTPException(status_code=404, detail="Run not found")

    workspace = await get_workspace(run_state.workspace_id)
    if workspace is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    run_dir = _resolve_run_dir(workspace.path, run_id)

    files = _generator.preview(
        run_dir=str(run_dir),
        workspace_path=workspace.path,
        run_state=run_state,
        max_file_mb=max_file_mb,
    )

    return {
        "run_id": run_id,
        "files": files,
        "total_files": len(files),
    }
