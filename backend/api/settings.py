"""Workspace settings API routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

from backend.database import get_workspace
from backend.models import GitMode, ProviderName, WorkspaceConfig

router = APIRouter(prefix="/settings", tags=["settings"])


class WorkspaceSettings(BaseModel):
    """Subset of WorkspaceConfig that is user-editable."""
    default_provider: Optional[ProviderName] = None
    default_model: Optional[str] = None
    git_mode: Optional[GitMode] = None
    providers_enabled: Optional[list[ProviderName]] = None
    budget_max_tokens_per_run: Optional[int] = None
    budget_max_cost_per_run: Optional[float] = None
    budget_max_iterations: Optional[int] = None
    safety_deny_destructive_commands: Optional[bool] = None


@router.get("/{workspace_id}", response_model=WorkspaceConfig)
async def get_settings(workspace_id: str) -> WorkspaceConfig:
    ws = await get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws


@router.put("/{workspace_id}", response_model=WorkspaceConfig)
async def update_settings(workspace_id: str, body: WorkspaceSettings) -> WorkspaceConfig:
    ws = await get_workspace(workspace_id)
    if ws is None:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Apply changes in-memory
    updates = body.model_dump(exclude_none=True)
    if not updates:
        return ws

    # Re-persist the whole workspace via database
    import json
    import aiosqlite
    from backend.config import config as app_config

    set_parts = []
    values = []
    for key, val in updates.items():
        if key == "providers_enabled":
            set_parts.append("providers_enabled = ?")
            values.append(json.dumps([p.value for p in val]))
        elif key == "default_provider":
            set_parts.append("default_provider = ?")
            values.append(val.value)
        elif key == "git_mode":
            set_parts.append("git_mode = ?")
            values.append(val.value)
        elif key == "safety_deny_destructive_commands":
            set_parts.append("safety_deny_destructive_commands = ?")
            values.append(int(val))
        else:
            set_parts.append(f"{key} = ?")
            values.append(val)

    values.append(workspace_id)
    sql = f"UPDATE workspaces SET {', '.join(set_parts)} WHERE id = ?"
    async with aiosqlite.connect(app_config.db_path) as db:
        db.row_factory = aiosqlite.Row
        await db.execute(sql, values)
        await db.commit()

    return await get_workspace(workspace_id)
