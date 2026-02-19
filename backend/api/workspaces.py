"""Workspace management API routes."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from backend.api.validation import validate_id, validate_workspace_name, validate_workspace_path
from backend.config import config
from backend.database import (
    create_workspace,
    delete_workspace,
    get_workspace,
    list_workspaces,
)
from backend.models import WorkspaceConfig, WorkspaceCreate, WorkspaceSummary

router = APIRouter(prefix="/workspaces", tags=["workspaces"])


@router.post("", response_model=WorkspaceConfig, status_code=201)
async def create_workspace_endpoint(body: WorkspaceCreate) -> WorkspaceConfig:
    validate_workspace_name(body.name)
    validate_workspace_path(body.path)
    ws = WorkspaceConfig(
        name=body.name,
        path=body.path,
        providers_enabled=body.providers_enabled,
        default_provider=body.default_provider,
        git_mode=body.git_mode,
    )
    # Scaffold workspace directories
    ws_path = Path(body.path)
    if not ws_path.is_absolute():
        ws_path = Path(config.workspace_base_path) / body.path
    ws_path.mkdir(parents=True, exist_ok=True)
    (ws_path / "runs").mkdir(exist_ok=True)
    (ws_path / "artifacts").mkdir(exist_ok=True)
    (ws_path / "knowledge").mkdir(exist_ok=True)
    return await create_workspace(ws)


@router.get("", response_model=list[WorkspaceSummary])
async def list_workspaces_endpoint(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[WorkspaceSummary]:
    workspaces = await list_workspaces(limit=limit, offset=offset)
    return [
        WorkspaceSummary(
            id=ws.id,
            name=ws.name,
            path=ws.path,
            default_provider=ws.default_provider,
            git_mode=ws.git_mode,
            created_at=ws.created_at,
        )
        for ws in workspaces
    ]


@router.get("/{workspace_id}", response_model=WorkspaceConfig)
async def get_workspace_endpoint(workspace_id: str) -> WorkspaceConfig:
    validate_id(workspace_id, "workspace_id")
    ws = await get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws


@router.delete("/{workspace_id}", status_code=204)
async def delete_workspace_endpoint(workspace_id: str) -> None:
    validate_id(workspace_id, "workspace_id")
    deleted = await delete_workspace(workspace_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Workspace not found")
