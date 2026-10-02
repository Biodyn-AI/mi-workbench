"""Background run executor - bridges API to orchestrator engine.

``execute_run`` loads the run, resolves the loop (``get_preset``: Python
presets, ``loops/<name>.yaml`` or ``custom:<path>``), drives the
:class:`LoopEngine` and persists the outcome:

* every finished iteration is written to the ``iterations`` table as soon as
  it ends (incremental persistence; the run row's iteration counter and
  token / cost totals are updated in the same transaction). Database writes
  are retried on lock contention (``backend.database``); an iteration whose
  write still fails is queued and written with the next successful write or
  at run end (``persistence_warnings`` in run_meta.json), never dropped;
* artifacts are written by the engine during the loop (full step outputs plus
  parsed artifacts); nothing is re-derived from the 500-character
  ``output_summary`` afterwards;
* the run row is set to RUNNING before the loop starts (so startup
  recovery can find an interrupted run and mark it PAUSED); the final
  status, ``stop_reason`` and totals go to the ``runs`` table and to
  ``runs/<run_id>/run_meta.json`` (with per-iteration provider, model,
  reasoning effort, CLI version, token split, cost, duration and agent-tool
  provenance, plus the run's effective tool settings and effective stopping
  policy: ``config.revision_budget`` and ``config.effective_stopping``);
* a run is resumed (prompt memory, convergence history and plan position
  restored) whenever it already has persisted progress, whatever its status;
* the workspace's per-run budgets are the engine's lowest-priority defaults
  for ``budget_max_tokens`` / ``budget_max_cost`` (run config and loop
  ``stop_conditions`` win); the run's own ``max_iterations`` always applies.

The whole body runs inside ``try/finally`` so the active-run registries are
always cleaned up, also on early returns and cancellation.
"""
import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Coroutine, Optional

from backend.adapters.registry import get_adapter
from backend.logging_config import set_run_context, clear_run_context
from backend.artifacts.writer import ensure_run_dirs, write_artifact, write_run_meta
from backend.database import (
    create_run,
    get_run,
    get_workspace,
    record_iterations_progress,
    update_run,
    update_run_if_status,
)
from backend.models import (
    FollowUpPolicy,
    FollowUpProposal,
    IterationResult,
    LoopDefinition,
    RunMeta,
    RunState,
    RunStatus,
)
from backend.orchestrator.engine import LoopEngine
from backend.orchestrator.followups import FollowUpExecutor
from backend.orchestrator.presets import get_preset
from backend.telemetry.collector import TelemetryAdapter

logger = logging.getLogger(__name__)

# Track active runs for cancellation
_active_runs: dict[str, asyncio.Event] = {}
_active_tasks: dict[str, asyncio.Task] = {}

# Maximum number of concurrent runs
MAX_CONCURRENT_RUNS = 20

# WebSocket broadcast callback, set by main.py at startup
_ws_broadcast: Optional[Callable[[str, dict], Coroutine]] = None

# Run-config keys copied into run_meta.json (secrets never live in run config,
# but keep the provenance block focused on settings that affect results).
_META_CONFIG_PREFIXES = (
    "consensus_", "convergence_", "code_execution_", "budget_", "reasoning_effort",
    "reviewer_allow_tools", "executor_allow_tools", "adapter_timeout", "max_retries",
    "grade_at_least", "flags", "adapter_options", "revision_budget",
)


def set_ws_broadcast(fn: Callable[[str, dict], Coroutine]) -> None:
    """Set the WebSocket broadcast function (called from main.py)."""
    global _ws_broadcast
    _ws_broadcast = fn


def iteration_meta(it: IterationResult) -> dict[str, Any]:
    """Per-iteration provenance entry for run_meta.json."""
    return {
        "iteration": it.iteration_number,
        "role": it.role,
        "status": it.status.value,
        "provider": it.provider,
        "model": it.model,
        "reasoning_effort": it.reasoning_effort,
        "cli_version": it.cli_version,
        "input_tokens": it.input_tokens,
        "output_tokens": it.output_tokens,
        "cached_input_tokens": it.cached_input_tokens,
        "token_usage": it.token_usage,
        "cost_estimate": it.cost_estimate,
        "duration_seconds": it.duration_seconds,
        "grade": it.grade,
        "critical_count": it.critical_count,
        "high_count": it.high_count,
        "code_execution": it.code_execution,
        "agent_tools": it.agent_tools,
        "consensus_grade_valid": ((it.consensus_report or {}).get("grade_valid")
                                  if it.consensus_report is not None else None),
        "artifacts": list(it.artifacts_produced),
        "error": it.error,
    }


def build_run_meta(
    state: RunState,
    loop_def: Optional[LoopDefinition] = None,
    convergence: Optional[dict[str, Any]] = None,
    tool_settings: Optional[dict[str, Any]] = None,
    persistence_warnings: Optional[list[dict[str, Any]]] = None,
    stopping_settings: Optional[dict[str, Any]] = None,
) -> RunMeta:
    """Assemble the run_meta.json model for a (finished) run.

    ``tool_settings`` (``LoopEngine.tool_settings()``) records the EFFECTIVE
    executor / reviewer agent-tool settings, including defaults.
    ``stopping_settings`` (``LoopEngine.stopping_settings()``) records the
    EFFECTIVE stopping policy: ``config.revision_budget`` is then the budget
    the run used (also when it came from the loop or the built-in default;
    ``None`` = disabled) and ``config.effective_stopping`` holds the budget,
    the convergence switch, their sources, the adaptive rule and the executor
    submissions / revisions made. Without it, ``config`` has only the run's
    own keys.
    ``persistence_warnings`` lists iteration writes that had to be deferred
    (database lock contention); every such iteration was persisted later."""
    its = sorted(state.iterations, key=lambda i: i.iteration_number)

    def uniq(values) -> list[str]:
        out: list[str] = []
        for v in values:
            for part in (v or "").split(","):
                part = part.strip()
                if part and part not in out:
                    out.append(part)
        return out

    config = {k: v for k, v in (state.config or {}).items()
              if any(k.startswith(p) for p in _META_CONFIG_PREFIXES)}
    if tool_settings:
        config["effective_tool_settings"] = dict(tool_settings)
    if stopping_settings:
        config["effective_stopping"] = dict(stopping_settings)
        if "revision_budget" in stopping_settings:
            config["revision_budget"] = stopping_settings["revision_budget"]
    return RunMeta(
        run_id=state.run_id,
        workspace_id=state.workspace_id,
        provider=state.provider,
        model=state.model,
        loop_preset=state.loop_preset,
        task=state.task,
        started_at=state.started_at,
        completed_at=state.stopped_at,
        total_iterations=state.current_iteration,
        total_tokens=state.total_tokens,
        total_cost=state.total_cost,
        status=state.status,
        stop_reason=state.stop_reason,
        error=state.error,
        reasoning_effort=str((state.config or {}).get("reasoning_effort", "") or ""),
        total_input_tokens=state.total_input_tokens,
        total_output_tokens=state.total_output_tokens,
        total_cached_input_tokens=state.total_cached_input_tokens,
        cli_versions=uniq(it.cli_version for it in its),
        models_used=uniq(it.model for it in its),
        loop_source=(loop_def.source if loop_def is not None else ""),
        stop_conditions=(list(loop_def.stop_conditions) if loop_def is not None else []),
        config=config,
        convergence=convergence or {},
        iterations=[iteration_meta(it) for it in its],
        persistence_warnings=list(persistence_warnings or []),
    )


async def _mark_failed(run_id: str, error: str, reason: str) -> None:
    try:
        await update_run(
            run_id,
            status=RunStatus.FAILED,
            error=error,
            stop_reason=f"failed:{reason}",
            stopped_at=datetime.utcnow(),
        )
    except Exception:
        logger.exception("Could not mark run %s as failed", run_id)


async def execute_run(run_id: str) -> None:
    """Execute a run in background. Called as asyncio task."""
    set_run_context(run_id)
    cancel_event: Optional[asyncio.Event] = None
    try:
        # Safety check: enforce concurrent run limit
        if len(_active_runs) >= MAX_CONCURRENT_RUNS:
            logger.error("Run %s rejected: maximum concurrent runs (%d) exceeded",
                         run_id, MAX_CONCURRENT_RUNS)
            await _mark_failed(run_id, "Maximum concurrent runs exceeded", "concurrency_limit")
            return

        if run_id in _active_runs:
            # Another engine is still driving this run (e.g. a stop request
            # whose in-flight step has not finished): never run two at once.
            logger.error("Run %s is already executing; not starting a second engine", run_id)
            return

        run = await get_run(run_id)
        if run is None:
            logger.error("Run %s not found in DB", run_id)
            return

        # Create the cancel event first so that a stop request arriving while
        # the run is being set up is honoured.
        cancel_event = asyncio.Event()
        _active_runs[run_id] = cancel_event

        # Resolve workspace path for artifact writing
        workspace = await get_workspace(run.workspace_id)
        workspace_path = workspace.path if workspace else ""
        workspace_defaults: dict[str, Any] = {}
        if workspace is not None:
            workspace_defaults = {
                "budget_max_tokens": workspace.budget_max_tokens_per_run,
                "budget_max_cost": workspace.budget_max_cost_per_run,
            }

        # Get loop definition (Python preset, loops/<name>.yaml or custom:<path>)
        try:
            loop_def = get_preset(run.loop_preset)
        except ValueError as exc:
            await _mark_failed(run_id, str(exc), "unknown_preset")
            return

        # Get adapter; every call (each attempt, each lens, the adjudicator) is
        # recorded by the telemetry proxy (in-memory collector + DB table).
        adapter = TelemetryAdapter(get_adapter(run.provider), run_id=run_id,
                                   iteration_getter=lambda: run.current_iteration)

        # Override max_iterations from run config
        loop_def.max_iterations = run.max_iterations

        # Ensure run directories exist
        if workspace_path:
            run_dir, _ = ensure_run_dirs(workspace_path, run_id)
        else:
            run_dir = None

        # Event callback: broadcast via WebSocket
        async def _on_event_async(event_type: str, data: dict[str, Any]) -> None:
            if _ws_broadcast:
                try:
                    await _ws_broadcast(run_id, {"type": event_type, **data})
                except Exception:
                    logger.debug("WS broadcast failed for %s", event_type)

        def on_event(event_type: str, data: dict[str, Any]) -> None:
            if _ws_broadcast is None:
                return
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_on_event_async(event_type, data))
            except RuntimeError:
                pass

        # Artifact writer callback for the engine
        def artifact_writer(iteration_num: int, artifact_name: str, content: str) -> None:
            if run_dir:
                write_artifact(run_dir, iteration_num, artifact_name, content)

        # Incremental iteration persistence. Each finished iteration is written
        # together with the run totals in one transaction (retried on lock
        # contention by backend.database). If that write still fails, the
        # iteration (and its telemetry rows) are queued and written with the
        # next successful progress write or at run end; the deferral is logged,
        # emitted as a ``persistence_warning`` event, noted in the iteration's
        # ``logs`` and recorded in run_meta.json (``persistence_warnings``).
        persisted: set[str] = {it.iteration_id for it in run.iterations}
        deferred: list[IterationResult] = []
        deferred_telemetry: list = []
        persistence_warnings: list[dict[str, Any]] = []

        def _totals(state: RunState) -> dict[str, Any]:
            return dict(
                current_iteration=state.current_iteration,
                total_tokens=state.total_tokens,
                total_cost=state.total_cost,
                total_input_tokens=state.total_input_tokens,
                total_output_tokens=state.total_output_tokens,
                total_cached_input_tokens=state.total_cached_input_tokens,
            )

        async def on_iteration(it: IterationResult, state: RunState) -> None:
            if it.iteration_id in persisted:
                return
            batch = [d for d in deferred if d.iteration_id not in persisted]
            if all(d.iteration_id != it.iteration_id for d in batch):
                batch.append(it)
            telemetry = deferred_telemetry + adapter.drain_pending()
            try:
                await record_iterations_progress(batch, run_id, telemetry=telemetry,
                                                 **_totals(state))
            except asyncio.CancelledError:
                deferred[:] = batch
                deferred_telemetry[:] = telemetry
                raise
            except Exception as exc:  # noqa: BLE001 - never lose the iteration
                deferred[:] = batch
                deferred_telemetry[:] = telemetry
                warning = {
                    "kind": "iteration_persist_deferred",
                    "iteration": it.iteration_number,
                    "iteration_id": it.iteration_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "deferred_iterations": [d.iteration_number for d in batch],
                    "at": datetime.utcnow().isoformat(),
                }
                persistence_warnings.append(warning)
                it.logs.append(
                    f"[persistence] progress write deferred at {warning['at']}: "
                    f"{warning['error']}")
                logger.warning(
                    "Run %s: could not persist iteration %d (%s); queued %d iteration(s) "
                    "for the next write", run_id, it.iteration_number, warning["error"],
                    len(batch))
                on_event("persistence_warning", {"run_id": run_id, **warning})
                return
            persisted.update(d.iteration_id for d in batch)
            if len(batch) > 1:
                logger.info("Run %s: persisted %d deferred iteration(s) with iteration %d",
                            run_id, len(batch) - 1, it.iteration_number)
            deferred.clear()
            deferred_telemetry.clear()

        max_retries = run.config.get("max_retries", 3)

        engine = LoopEngine(
            adapter=adapter,
            on_event=on_event,
            artifact_writer=artifact_writer if run_dir else None,
            max_retries=max_retries,
            on_iteration=on_iteration,
        )

        # Detect resume from persisted progress (also for a run that was
        # interrupted while still PENDING / RUNNING in the DB).
        resume_from = 0
        if run.current_iteration > 0 and run.iterations:
            resume_from = run.current_iteration
            if run.started_at is None:
                run.started_at = datetime.utcnow()
        elif run.status == RunStatus.PAUSED:
            resume_from = run.current_iteration
        run.stop_reason = None

        # Persist RUNNING before the loop starts, so an interrupted run is
        # visible to startup recovery (the in-memory status stays as loaded
        # for the engine's start/resume transition). Compare-and-set: if /stop
        # changed the status since the run was loaded, honour the stop.
        if run.status in (RunStatus.PENDING, RunStatus.PAUSED):
            marked = await update_run_if_status(
                run_id, run.status, status=RunStatus.RUNNING,
                started_at=run.started_at or datetime.utcnow())
            if not marked:
                cancel_event.set()

        final_state = await engine.run_loop(
            run_state=run,
            loop_def=loop_def,
            cancel_event=cancel_event,
            workspace_path=workspace_path,
            resume_from=resume_from,
            workspace_defaults=workspace_defaults,
        )

        # Persist every iteration the incremental hook did not store (deferred
        # writes, or iterations recorded without the hook), together with the
        # telemetry of calls not tied to a persisted iteration, in one retried
        # transaction. Iterations already stored are skipped (INSERT OR IGNORE).
        missing = [it for it in final_state.iterations if it.iteration_id not in persisted]
        for d in deferred:
            if d.iteration_id not in persisted and all(
                    m.iteration_id != d.iteration_id for m in missing):
                missing.append(d)
        leftover = deferred_telemetry + adapter.drain_pending()
        if missing or leftover:
            await record_iterations_progress(
                sorted(missing, key=lambda i: i.iteration_number), run_id,
                telemetry=leftover)
            persisted.update(it.iteration_id for it in missing)
            if deferred:
                logger.info("Run %s: persisted %d deferred iteration(s) at run end",
                            run_id, len(deferred))
            deferred.clear()
            deferred_telemetry.clear()

        # Update run in DB with final state
        await update_run(
            run_id,
            status=final_state.status,
            current_iteration=final_state.current_iteration,
            total_tokens=final_state.total_tokens,
            total_cost=final_state.total_cost,
            total_input_tokens=final_state.total_input_tokens,
            total_output_tokens=final_state.total_output_tokens,
            total_cached_input_tokens=final_state.total_cached_input_tokens,
            started_at=final_state.started_at,
            stopped_at=final_state.stopped_at,
            error=final_state.error,
            stop_reason=final_state.stop_reason,
        )

        # Write run_meta.json
        if run_dir:
            try:
                meta = build_run_meta(final_state, loop_def,
                                      engine._convergence.get_metrics(),
                                      tool_settings=engine.tool_settings(),
                                      persistence_warnings=persistence_warnings,
                                      stopping_settings=engine.stopping_settings())
                write_run_meta(run_dir, meta)
            except Exception:
                logger.exception("Failed to write run_meta.json for %s", run_id)

        # Check for follow-up proposals and auto-dispatch if policy allows
        raw_proposals = final_state.config.get("follow_up_proposals", [])
        policy_str = final_state.config.get(
            "follow_up_policy", (loop_def.config or {}).get("follow_up_policy", "ask_approval"))
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
                        if "reasoning_effort" in final_state.config:
                            new_config.setdefault("reasoning_effort",
                                                  final_state.config["reasoning_effort"])
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

    except asyncio.CancelledError:
        # Task cancelled (e.g. server shutdown): leave the DB row as is (it is
        # RUNNING once the loop has started); startup recovery marks RUNNING
        # (and PENDING) runs with persisted progress as PAUSED.
        raise
    except Exception as exc:
        logger.exception("Run %s failed with exception", run_id)
        await _mark_failed(run_id, str(exc), "exception")
    finally:
        if cancel_event is not None and _active_runs.get(run_id) is cancel_event:
            _active_runs.pop(run_id, None)
        task = _active_tasks.get(run_id)
        current = asyncio.current_task()
        if task is None or task is current or task.done():
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
