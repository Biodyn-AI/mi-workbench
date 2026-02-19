"""Background run executor - bridges API to orchestrator engine."""
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from backend.adapters.registry import get_adapter
from backend.artifacts.parser import ArtifactParser
from backend.logging_config import set_run_context, clear_run_context
from backend.artifacts.writer import ensure_run_dirs, write_artifact, write_run_meta
from backend.database import create_iteration, create_run, get_run, get_workspace, update_run
from backend.models import FollowUpPolicy, FollowUpProposal, ProviderName, RunMeta, RunState, RunStatus
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.followups import FollowUpExecutor
from backend.orchestrator.presets import get_preset

logger = logging.getLogger(__name__)

# Track active runs for cancellation
_active_runs: dict[str, asyncio.Event] = {}
_active_tasks: dict[str, asyncio.Task] = {}

# Maximum number of concurrent runs
MAX_CONCURRENT_RUNS = 20

# WebSocket broadcast callback, set by main.py at startup
_ws_broadcast: Optional[Callable[[str, dict], Coroutine]] = None


def set_ws_broadcast(fn: Callable[[str, dict], Coroutine]) -> None:
    """Set the WebSocket broadcast function (called from main.py)."""
    global _ws_broadcast
    _ws_broadcast = fn


async def execute_run(run_id: str) -> None:
    """Execute a run in background. Called as asyncio task."""
    set_run_context(run_id)

    # Safety check: enforce concurrent run limit
    if len(_active_runs) >= MAX_CONCURRENT_RUNS:
        logger.error("Run %s rejected: maximum concurrent runs (%d) exceeded", run_id, MAX_CONCURRENT_RUNS)
        await update_run(run_id, status=RunStatus.FAILED, error="Maximum concurrent runs exceeded")
        clear_run_context()
        return

    run = await get_run(run_id)
    if run is None:
        logger.error("Run %s not found in DB", run_id)
        clear_run_context()
        return

    # Resolve workspace path for artifact writing
    workspace = await get_workspace(run.workspace_id)
    workspace_path = workspace.path if workspace else ""

    # Get adapter
    adapter = get_adapter(run.provider)

    # Get loop definition from preset
    try:
        loop_def = get_preset(run.loop_preset)
    except ValueError:
        await update_run(run_id, status=RunStatus.FAILED, error=f"Unknown preset: {run.loop_preset}")
        return

    # Override max_iterations from run config
    loop_def.max_iterations = run.max_iterations

    # Create cancel event
    cancel_event = asyncio.Event()
    _active_runs[run_id] = cancel_event

    # Ensure run directories exist
    if workspace_path:
        run_dir, _ = ensure_run_dirs(workspace_path, run_id)
    else:
        run_dir = None

    # Event callback: update DB, write artifacts, broadcast via WebSocket
    async def _on_event_async(event_type: str, data: dict[str, Any]) -> None:
        if _ws_broadcast:
            try:
                await _ws_broadcast(run_id, {"type": event_type, **data})
            except Exception:
                logger.debug("WS broadcast failed for %s", event_type)

    def on_event(event_type: str, data: dict[str, Any]) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_on_event_async(event_type, data))
        except RuntimeError:
            pass

    # Artifact writer callback for the engine
    def artifact_writer(iteration_num: int, artifact_name: str, content: str) -> None:
        if run_dir:
            write_artifact(run_dir, iteration_num, artifact_name, content)

    max_retries = run.config.get("max_retries", 3)

    engine = LoopEngine(
        adapter=adapter,
        on_event=on_event,
        artifact_writer=artifact_writer if run_dir else None,
        max_retries=max_retries,
    )

    # Detect resume: if the run was paused or already has completed iterations
    resume_from = 0
    if run.status == RunStatus.PAUSED or (run.current_iteration > 0 and run.status != RunStatus.PENDING):
        resume_from = run.current_iteration
        # Set status to RUNNING before entering engine (engine expects PAUSED or RUNNING)
        if run.status == RunStatus.PAUSED:
            pass  # engine.run_loop handles PAUSED->RUNNING transition
        if run.started_at is None:
            run.started_at = datetime.utcnow()

    try:
        final_state = await engine.run_loop(
            run_state=run,
            loop_def=loop_def,
            cancel_event=cancel_event,
            workspace_path=workspace_path,
            resume_from=resume_from,
        )

        # Persist each iteration to DB
        for it in final_state.iterations:
            try:
                await create_iteration(it, run_id)
            except Exception:
                logger.debug("Iteration %s may already exist", it.iteration_id)

            # Write parsed artifacts to disk
            if run_dir and it.output_summary:
                try:
                    parser = ArtifactParser()
                    parsed = parser.parse_output(it.output_summary, role=it.role)
                    if parsed:
                        for art_name, art_content in parsed.items():
                            write_artifact(run_dir, it.iteration_number, art_name, art_content)
                    else:
                        artifact_name = f"{it.role}_output.md"
                        write_artifact(run_dir, it.iteration_number, artifact_name, it.output_summary)
                except Exception:
                    logger.debug("Failed to write artifact for iter %d", it.iteration_number)

        # Update run in DB with final state
        await update_run(
            run_id,
            status=final_state.status,
            current_iteration=final_state.current_iteration,
            total_tokens=final_state.total_tokens,
            total_cost=final_state.total_cost,
            started_at=final_state.started_at,
            stopped_at=final_state.stopped_at,
            error=final_state.error,
        )

        # Write run_meta.json
        if run_dir:
            meta = RunMeta(
                run_id=run_id,
                workspace_id=final_state.workspace_id,
                provider=final_state.provider,
                model=final_state.model,
                loop_preset=final_state.loop_preset,
                task=final_state.task,
                started_at=final_state.started_at,
                completed_at=final_state.stopped_at,
                total_iterations=final_state.current_iteration,
                total_tokens=final_state.total_tokens,
                total_cost=final_state.total_cost,
                status=final_state.status,
            )
            write_run_meta(run_dir, meta)

        # Check for follow-up proposals and auto-dispatch if policy allows
        raw_proposals = final_state.config.get("follow_up_proposals", [])
        policy_str = final_state.config.get("follow_up_policy", "ask_approval")
        if raw_proposals:
            # Enforce follow-up depth limit
            current_depth = final_state.config.get("follow_up_depth", 0)
            max_depth = final_state.config.get("max_follow_up_depth", 3)
            if current_depth >= max_depth:
                logger.info(
                    "Follow-up depth limit reached (%d/%d) for run %s, skipping follow-ups",
                    current_depth, max_depth, run_id,
                )
            else:
                try:
                    policy = FollowUpPolicy(policy_str)
                    proposals = [FollowUpProposal(**p) for p in raw_proposals]
                    fu_executor = FollowUpExecutor()
                    selected = fu_executor.select_followups(proposals, policy)
                    for proposal in selected:
                        new_run_create = fu_executor.create_run_from_proposal(
                            proposal,
                            parent_workspace_id=final_state.workspace_id,
                            parent_run_id=run_id,
                            provider=final_state.provider,
                        )
                        # Propagate depth tracking to child run
                        new_config = dict(new_run_create.config_overrides)
                        new_config["follow_up_depth"] = current_depth + 1
                        new_config["max_follow_up_depth"] = max_depth
                        new_run = RunState(
                            workspace_id=new_run_create.workspace_id,
                            loop_preset=new_run_create.loop_preset,
                            task=new_run_create.task,
                            provider=new_run_create.provider,
                            model=final_state.model,
                            max_iterations=new_run_create.max_iterations,
                            config=new_config,
                        )
                        await create_run(new_run)
                        follow_up_task = asyncio.create_task(execute_run(new_run.run_id))
                        _active_tasks[new_run.run_id] = follow_up_task
                        logger.info(
                            "Auto-dispatched follow-up run %s from parent %s (depth %d/%d): %s",
                            new_run.run_id, run_id, current_depth + 1, max_depth, proposal.title,
                        )
                        on_event("follow_up_dispatched", {
                            "run_id": run_id,
                            "follow_up_run_id": new_run.run_id,
                            "title": proposal.title,
                        })
                except Exception:
                    logger.debug("Follow-up dispatch failed for run %s", run_id)

    except Exception as exc:
        logger.exception("Run %s failed with exception", run_id)
        await update_run(
            run_id,
            status=RunStatus.FAILED,
            error=str(exc),
            stopped_at=datetime.utcnow(),
        )
    finally:
        _active_runs.pop(run_id, None)
        _active_tasks.pop(run_id, None)
        clear_run_context()


async def cancel_run(run_id: str) -> bool:
    """Cancel an active run. Returns True if cancellation was signalled."""
    event = _active_runs.get(run_id)
    if event is None:
        return False
    event.set()
    return True


def get_active_run_ids() -> list[str]:
    """Return list of currently executing run IDs."""
    return list(_active_runs.keys())


def is_run_active(run_id: str) -> bool:
    """Check if a run is currently executing."""
    return run_id in _active_runs
