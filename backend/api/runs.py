"""Run management API routes."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.api.validation import validate_config, validate_id, validate_task
from backend.database import (
    create_iteration,
    create_run,
    get_iterations_for_run,
    get_run,
    list_runs,
    update_run,
)
from backend.models import (
    IterationResult,
    RunCreate,
    RunState,
    RunStatus,
    RunSummary,
)
from backend.orchestrator.presets import get_preset
from backend.orchestrator.recovery import can_resume_run
from backend.orchestrator.runner import (
    MAX_CONCURRENT_RUNS,
    cancel_run,
    execute_run,
    get_active_run_ids,
    is_run_active,
    _active_tasks,
)

router = APIRouter(prefix="/runs", tags=["runs"])

# Config keys that cannot be overridden via API
DENIED_CONFIG_KEYS = {"follow_up_depth", "follow_up_proposals", "parent_run_id"}


def sanitize_config(config: dict) -> dict:
    """Remove internal-only config keys."""
    return {k: v for k, v in config.items() if k not in DENIED_CONFIG_KEYS}


@router.post("", response_model=RunState, status_code=201)
async def create_run_endpoint(
    body: RunCreate,
) -> RunState:
    # Input validation
    validate_id(body.workspace_id, "workspace_id")
    validate_task(body.task)
    validate_config(body.config_overrides)
    if body.max_iterations < 1 or body.max_iterations > 1000:
        raise HTTPException(
            status_code=422,
            detail=f"max_iterations must be between 1 and 1000, got {body.max_iterations}",
        )

    # Concurrent run limit
    if len(get_active_run_ids()) >= MAX_CONCURRENT_RUNS:
        raise HTTPException(
            status_code=429,
            detail=f"Maximum concurrent runs ({MAX_CONCURRENT_RUNS}) exceeded",
        )

    # The loop must resolve now (Python preset, loops/<name>.yaml or custom:<path>)
    try:
        get_preset(body.loop_preset)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Sanitize config overrides
    safe_config = sanitize_config(body.config_overrides)
    if body.reasoning_effort and not safe_config.get("reasoning_effort"):
        safe_config["reasoning_effort"] = body.reasoning_effort

    run = RunState(
        workspace_id=body.workspace_id,
        loop_preset=body.loop_preset,
        task=body.task,
        provider=body.provider,
        model=body.model,
        max_iterations=body.max_iterations,
        config=safe_config,
    )
    await create_run(run)
    # Dispatch execution as an asyncio task
    task = asyncio.create_task(execute_run(run.run_id))
    _active_tasks[run.run_id] = task
    return run


@router.get("", response_model=list[RunSummary])
async def list_runs_endpoint(
    workspace_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[RunSummary]:
    runs = await list_runs(workspace_id, limit=limit, offset=offset)
    return [
        RunSummary(
            run_id=r.run_id,
            workspace_id=r.workspace_id,
            loop_preset=r.loop_preset,
            task=r.task,
            provider=r.provider,
            status=r.status,
            current_iteration=r.current_iteration,
            total_tokens=r.total_tokens,
            total_cost=r.total_cost,
            created_at=r.created_at,
            started_at=r.started_at,
            stopped_at=r.stopped_at,
        )
        for r in runs
    ]


@router.get("/{run_id}", response_model=RunState)
async def get_run_endpoint(run_id: str) -> RunState:
    validate_id(run_id, "run_id")
    run = await get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.post("/{run_id}/stop", response_model=RunState)
async def stop_run_endpoint(run_id: str) -> RunState:
    validate_id(run_id, "run_id")
    run = await get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status not in (RunStatus.RUNNING, RunStatus.PENDING, RunStatus.PAUSED):
        raise HTTPException(status_code=400, detail=f"Cannot stop run with status {run.status.value}")
    # Signal the runner to cancel
    await cancel_run(run_id)
    # Also update DB directly in case the engine hasn't caught the signal yet
    updated = await update_run(
        run_id,
        status=RunStatus.STOPPED,
        stopped_at=datetime.utcnow(),
    )
    return updated


@router.post("/{run_id}/resume", response_model=RunState)
async def resume_run_endpoint(
    run_id: str,
) -> RunState:
    validate_id(run_id, "run_id")
    run = await get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.status not in (RunStatus.STOPPED, RunStatus.PAUSED):
        raise HTTPException(status_code=400, detail=f"Cannot resume run with status {run.status.value}")
    if is_run_active(run_id):
        # A stop request only signals the engine; its in-flight step (up to
        # adapter_timeout) still has to finish. Starting a second engine now
        # would run the same iteration twice.
        raise HTTPException(
            status_code=409,
            detail="Run is still stopping; retry when the in-flight step has finished",
        )
    updated = await update_run(run_id, status=RunStatus.RUNNING)
    task = asyncio.create_task(execute_run(run_id))
    _active_tasks[run_id] = task
    return updated


@router.get("/{run_id}/can-resume")
async def can_resume_endpoint(run_id: str) -> dict:
    """Check if a run can be safely resumed (has iterations, workspace exists)."""
    validate_id(run_id, "run_id")
    resumable, reason = await can_resume_run(run_id)
    return {"can_resume": resumable, "reason": reason}


@router.post("/{run_id}/execute", response_model=RunState)
async def execute_run_endpoint(run_id: str) -> RunState:
    """Trigger execution for a pending (or stopped/paused) run."""
    validate_id(run_id, "run_id")
    run = await get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    if is_run_active(run_id):
        raise HTTPException(status_code=409, detail="Run is already executing")
    if run.status not in (RunStatus.PENDING, RunStatus.STOPPED, RunStatus.PAUSED):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot execute run with status {run.status.value}",
        )
    task = asyncio.create_task(execute_run(run.run_id))
    _active_tasks[run.run_id] = task
    return run


@router.get("/{run_id}/iterations/{iteration_number}", response_model=IterationResult)
async def get_iteration_endpoint(run_id: str, iteration_number: int) -> IterationResult:
    validate_id(run_id, "run_id")
    iterations = await get_iterations_for_run(run_id)
    for it in iterations:
        if it.iteration_number == iteration_number:
            return it
    raise HTTPException(status_code=404, detail="Iteration not found")
