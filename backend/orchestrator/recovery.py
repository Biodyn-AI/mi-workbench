"""Crash recovery - detect and handle interrupted runs on startup."""
from __future__ import annotations

import logging
from pathlib import Path

from backend.database import get_run, list_runs, update_run
from backend.models import RunStatus

logger = logging.getLogger(__name__)


async def recover_interrupted_runs() -> list[str]:
    """Scan DB for runs stuck in RUNNING/PENDING status and handle them.

    Called during server startup (in lifespan).

    For RUNNING runs:
    - If the run has iterations, mark as PAUSED (resumable)
    - If no iterations yet, mark as FAILED with error "interrupted by server restart"

    For PENDING runs:
    - With persisted iterations (a run interrupted before its RUNNING status
      was written, e.g. by an older version): mark as PAUSED (resumable)
    - Otherwise leave as PENDING (they might be legitimately queued)

    Interrupted runs get ``stop_reason="interrupted"``.

    Returns list of recovered run IDs.
    """
    recovered: list[str] = []
    all_runs = await list_runs(limit=1_000_000)  # every run, not only the newest page

    for run in all_runs:
        if run.status == RunStatus.PENDING:
            full_run = await get_run(run.run_id)
            if full_run is not None and (full_run.iterations or full_run.current_iteration > 0):
                await update_run(run.run_id, status=RunStatus.PAUSED, stop_reason="interrupted")
                logger.info("Recovered run %s: PENDING with progress -> PAUSED", run.run_id)
                recovered.append(run.run_id)
            continue
        if run.status == RunStatus.RUNNING:
            # Reload with iterations
            full_run = await get_run(run.run_id)
            if full_run is None:
                continue

            if full_run.iterations:
                # Has progress -- mark as paused so it can be resumed
                await update_run(run.run_id, status=RunStatus.PAUSED, stop_reason="interrupted")
                logger.info(
                    "Recovered run %s: RUNNING -> PAUSED (%d iterations completed)",
                    run.run_id,
                    len(full_run.iterations),
                )
            else:
                # No progress -- mark as failed
                await update_run(
                    run.run_id,
                    status=RunStatus.FAILED,
                    error="interrupted by server restart",
                )
                logger.info("Recovered run %s: RUNNING -> FAILED (no iterations)", run.run_id)
            recovered.append(run.run_id)

    return recovered


async def can_resume_run(run_id: str) -> tuple[bool, str]:
    """Check if a run can be resumed.

    Returns (can_resume, reason).
    A run can resume if:
    - Status is PAUSED or STOPPED
    - Has at least one completed iteration (to have context)
    - Workspace still exists
    """
    run = await get_run(run_id)
    if run is None:
        return False, "Run not found"

    if run.status not in (RunStatus.PAUSED, RunStatus.STOPPED):
        return False, f"Run status is {run.status.value}, must be paused or stopped"

    if not run.iterations:
        return False, "Run has no completed iterations to resume from"

    # Check workspace exists
    from backend.database import get_workspace
    workspace = await get_workspace(run.workspace_id)
    if workspace is None:
        return False, f"Workspace {run.workspace_id} not found"
    if not Path(workspace.path).exists():
        return False, f"Workspace path does not exist: {workspace.path}"

    return True, "ok"
