"""Artifact browsing and retrieval API routes."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from backend.database import get_run, get_workspace

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


@router.get("/browse", response_model=list[dict])
async def browse_artifacts(
    workspace_id: str = Query(...),
    path: str = Query(""),
) -> list[dict]:
    ws = await get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    base = Path(ws.path)
    target = base / path if path else base
    if not target.exists():
        raise HTTPException(status_code=404, detail="Path not found")
    if not str(target.resolve()).startswith(str(base.resolve())):
        raise HTTPException(status_code=403, detail="Access denied: path traversal")
    if target.is_file():
        return [{"name": target.name, "type": "file", "size": target.stat().st_size}]
    entries = []
    for child in sorted(target.iterdir()):
        entry = {
            "name": child.name,
            "type": "directory" if child.is_dir() else "file",
        }
        if child.is_file():
            entry["size"] = child.stat().st_size
        entries.append(entry)
    return entries


@router.get("/{run_id}/{iteration_id}/{filename}")
async def get_artifact(
    run_id: str,
    iteration_id: str,
    filename: str,
) -> PlainTextResponse:
    run = await get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    ws = await get_workspace(run.workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    artifact_path = Path(ws.path) / "runs" / run_id / iteration_id / filename
    if not artifact_path.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    if not str(artifact_path.resolve()).startswith(str(Path(ws.path).resolve())):
        raise HTTPException(status_code=403, detail="Access denied: path traversal")
    content = artifact_path.read_text(errors="replace")
    return PlainTextResponse(content)
