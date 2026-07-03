"""Main loop execution engine — runs iterations via adapter calls."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Optional

from backend.adapters.base import BaseAdapter
from backend.logging_config import set_run_context
from backend.artifacts.parser import ArtifactParser
from backend.models import (
    AdapterRunRequest,
    BudgetStatus,
    FollowUpPolicy,
    IterationResult,
    IterationStatus,
    LoopDefinition,
    PromptBundle,
    RunState,
    RunStatus,
    WorkspaceContext,
)
from backend.orchestrator.code_executor import CodeExecutor
from backend.orchestrator.consensus import ConsensusReviewer
from backend.orchestrator.context import WorkspaceContextReader
from backend.orchestrator.convergence import ConvergenceDetector
from backend.orchestrator.dsl import ExecutionPlan, ExecutionStep, compile_loop
from backend.orchestrator.feedback import FeedbackFormatter
from backend.orchestrator.followups import FollowUpExecutor
from backend.orchestrator.retry import run_with_retry
from backend.orchestrator.state_machine import RunStateMachine
from backend.registry.prompts import PromptRegistry
from backend.utils.output_cleaner import strip_thinking_traces

logger = logging.getLogger(__name__)


EventCallback = Callable[[str, dict[str, Any]], Any]
ArtifactWriter = Callable[[int, str, str], Any]


class LoopEngine:
    """Async engine that drives the executor/reviewer/adversarial loop."""

    def __init__(
        self,
        adapter: BaseAdapter,
        on_event: Optional[EventCallback] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        artifact_writer: Optional[ArtifactWriter] = None,
        max_retries: int = 3,
    ):
        self.adapter = adapter
        self._on_event = on_event
        self._prompt_registry = prompt_registry or PromptRegistry()
        self._artifact_writer = artifact_writer
        self._max_retries = max_retries
        self._context_reader: Optional[WorkspaceContextReader] = None
        self._artifact_parser = ArtifactParser()
        self._feedback_formatter = FeedbackFormatter()
        self._followup_executor = FollowUpExecutor()
        self._convergence = ConvergenceDetector()
        # Set when a consensus gate is holding the loop open on unresolved CRITICAL.
        self._consensus_block = False

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self._on_event:
            self._on_event(event_type, data)

    def _check_budget(self, run_state: RunState) -> BudgetStatus:
        exceeded = False
        reason = None

        tokens_limit = run_state.config.get("budget_max_tokens", 500_000)
        cost_limit = run_state.config.get("budget_max_cost", 10.0)
        iter_limit = run_state.max_iterations

        if run_state.total_tokens >= tokens_limit:
            exceeded = True
            reason = f"Token budget exceeded: {run_state.total_tokens}/{tokens_limit}"
        elif run_state.total_cost >= cost_limit:
            exceeded = True
            reason = f"Cost budget exceeded: {run_state.total_cost:.4f}/{cost_limit:.2f}"
        elif run_state.current_iteration >= iter_limit:
            exceeded = True
            reason = f"Iteration limit reached: {run_state.current_iteration}/{iter_limit}"

        return BudgetStatus(
            tokens_used=run_state.total_tokens,
            tokens_limit=tokens_limit,
            cost_used=run_state.total_cost,
            cost_limit=cost_limit,
            iterations_used=run_state.current_iteration,
            iterations_limit=iter_limit,
            exceeded=exceeded,
            reason=reason,
        )

    def _should_run_step(
        self, step: ExecutionStep, iteration: int,
        run_state: Optional[RunState] = None,
    ) -> bool:
        """Evaluate whether a step should execute given its condition."""
        cond = step.condition
        if cond == "always":
            return True
        if cond.startswith("every_k:"):
            k = int(cond.split(":")[1])
            return k > 0 and iteration % k == 0
        if cond.startswith("on_flag:"):
            # Run when the named flag is truthy in run_state.config["flags"].
            flag = cond.split(":", 1)[1].strip()
            if run_state is None:
                return False
            flags = run_state.config.get("flags", {})
            return bool(flags.get(flag, False))
        return True

    def _build_prompt_bundle(
        self,
        step: ExecutionStep,
        run_state: RunState,
        previous_output: str,
    ) -> PromptBundle:
        """Build a prompt bundle for a step, incorporating previous feedback.

        Tries to load a template from the PromptRegistry using the step's
        prompt_ref (format: "role/name"). Falls back to inline prompts if
        the registry lookup fails.
        """
        task = run_state.task
        role = step.role
        variables = {
            "role": role,
            "iteration": str(run_state.current_iteration),
            "task": task,
        }

        # Try loading from PromptRegistry
        system_prompt = None
        if step.prompt_ref and self._prompt_registry:
            try:
                parts = step.prompt_ref.split("/", 1)
                if len(parts) == 2:
                    template = self._prompt_registry.load_prompt(parts[0], parts[1])
                    system_prompt = self._prompt_registry.resolve_template(template, variables)
            except Exception:
                logger.debug("Prompt registry lookup failed for %s, using inline", step.prompt_ref)

        # Fallback to inline prompt
        if system_prompt is None:
            system_prompt = (
                f"You are a {role} in an automated MI research pipeline.\n"
                f"Task: {task}\n"
                f"Iteration: {run_state.current_iteration}"
            )

        user_prompt = task
        if previous_output:
            iteration = run_state.current_iteration
            user_prompt = (
                f"=== REVISION REQUEST (Iteration {iteration}) ===\n"
                f"You produced artifacts in the previous iteration.\n"
                f"The reviewer has identified specific issues that MUST be addressed.\n\n"
                f"{previous_output}\n\n"
                f"TASK: Revise your artifacts to address ALL required fixes above.\n"
                f"For each fix, make the specific change in the referenced artifact and section.\n"
                f"Do NOT rewrite from scratch — make targeted revisions.\n\n"
                f"Original task: {task}"
            )

        # Inject workspace context if available
        if self._context_reader:
            if role in ("executor", "adversarial", "idea_generator"):
                context_section = self._context_reader.build_context_section()
                if context_section:
                    user_prompt = context_section + "\n\n" + user_prompt
            elif role == "reviewer":
                # For reviewers, inject the executor's latest output
                reviewer_ctx = self._context_reader.read_latest_artifacts()
                if reviewer_ctx:
                    ctx_parts = []
                    for name, content in sorted(reviewer_ctx.items()):
                        ctx_parts.append(f"### {name}\n{content}")
                    user_prompt = (
                        "=== Artifacts to Review ===\n\n"
                        + "\n\n".join(ctx_parts)
                        + "\n\n=== End Artifacts ===\n\n"
                        + user_prompt
                    )

        return PromptBundle(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            variables=variables,
        )

    def _resolve_role_system_prompt(
        self, role: str, prompt_ref: str, run_state: RunState
    ) -> str:
        """Resolve a role's system prompt from the registry, or fall back inline."""
        variables = {
            "role": role,
            "iteration": str(run_state.current_iteration),
            "task": run_state.task,
        }
        if prompt_ref and self._prompt_registry:
            try:
                parts = prompt_ref.split("/", 1)
                if len(parts) == 2:
                    template = self._prompt_registry.load_prompt(parts[0], parts[1])
                    return self._prompt_registry.resolve_template(template, variables)
            except Exception:
                logger.debug("Prompt lookup failed for %s, using inline", prompt_ref)
        return (
            f"You are the {role} in an automated MI research pipeline.\n"
            f"Task: {run_state.task}\n"
            f"Review the following artifacts and report critiques as "
            f"[SEVERITY] lines."
        )

    async def _run_consensus_step(
        self,
        step: ExecutionStep,
        run_state: RunState,
        iteration_num: int,
        artifacts_content: str,
        workspace_path: str,
    ) -> tuple[Any, str, list]:
        """Run a consensus panel concurrently and merge critiques.

        Returns (merged ReviewResult, formatted feedback string, raw results).
        """
        panel = [(pm.role, pm.prompt_ref) for pm in step.panel]
        reviewer = ConsensusReviewer(
            similarity_threshold=run_state.config.get("consensus_similarity_threshold", 0.5),
            consensus_threshold=run_state.config.get("consensus_threshold", 2),
            panel=panel or None,
            role_weights=run_state.config.get("consensus_role_weights") or None,
        )
        ctx = WorkspaceContext(
            workspace_path=workspace_path,
            run_dir=f"runs/{run_state.run_id}",
            iteration_dir=f"runs/{run_state.run_id}/iter_{iteration_num:03d}",
        )
        merged, raw_results = await reviewer.run_panel(
            artifacts_content,
            self.adapter,
            ctx,
            system_prompt_builder=lambda role, ref: self._resolve_role_system_prompt(
                role, ref, run_state
            ),
        )
        formatted = self._feedback_formatter.format_for_executor(merged)
        return merged, formatted, raw_results

    async def run_loop(
        self,
        run_state: RunState,
        loop_def: LoopDefinition,
        cancel_event: Optional[asyncio.Event] = None,
        workspace_path: str = "",
        resume_from: int = 0,
    ) -> RunState:
        """Execute the full loop until completion, budget exceeded, or cancellation."""
        sm = RunStateMachine(
            run_state,
            on_state_change=lambda old, new, st: self._emit(
                "state_change", {"old": old.value, "new": new.value, "run_id": st.run_id}
            ),
        )

        plan = compile_loop(loop_def)

        # Handle resume vs fresh start
        if resume_from > 0 and run_state.status in (RunStatus.PAUSED, RunStatus.RUNNING):
            # For resumed runs, transition PAUSED->RUNNING if needed
            if run_state.status == RunStatus.PAUSED:
                try:
                    sm.resume()
                except Exception as exc:
                    sm.fail(str(exc))
                    return run_state
            # Otherwise already RUNNING — no transition needed
        else:
            try:
                sm.start()
            except Exception as exc:
                sm.fail(str(exc))
                return run_state

        # Set up workspace context reader
        if workspace_path:
            self._context_reader = WorkspaceContextReader(workspace_path, run_state.run_id)
        else:
            self._context_reader = None

        self._emit("loop_started", {"run_id": run_state.run_id, "plan_steps": len(plan.steps)})

        previous_output = ""
        step_index = 0

        # Restore state when resuming from a previous iteration
        if resume_from > 0 and run_state.iterations:
            run_state.current_iteration = resume_from
            step_index = resume_from

            # Reconstruct previous_output from last iteration's output_summary
            last_iter = run_state.iterations[-1]
            if last_iter.output_summary:
                previous_output = last_iter.output_summary
            elif self._context_reader:
                # Fall back to reading artifacts from disk
                artifacts = self._context_reader.read_run_artifacts()
                if artifacts:
                    previous_output = "\n\n".join(artifacts.values())

            # Re-seed convergence detector with historical data
            for it in run_state.iterations:
                self._convergence.record_iteration(
                    iteration_num=it.iteration_number,
                    role=it.role,
                    output=it.output_summary or "",
                    review=None,
                )

        while True:
            # Check cancellation
            if cancel_event and cancel_event.is_set():
                sm.stop()
                self._emit("loop_cancelled", {"run_id": run_state.run_id})
                return run_state

            # Check budget
            budget = self._check_budget(run_state)
            if budget.exceeded:
                self._emit("budget_exceeded", {"run_id": run_state.run_id, "reason": budget.reason})
                sm.complete()
                return run_state

            # Get current step (cycle through plan)
            if plan.cycle_length == 0:
                sm.complete()
                return run_state

            step = plan.steps[step_index % plan.cycle_length]

            # Check if this conditional step should run
            if not self._should_run_step(step, run_state.current_iteration, run_state):
                step_index += 1
                continue

            # Advance iteration
            iteration_num = sm.next_iteration()
            set_run_context(run_state.run_id, iteration_num)
            self._emit("iteration_started", {
                "run_id": run_state.run_id,
                "iteration": iteration_num,
                "role": step.role,
                "node_id": step.node_id,
            })

            # Consensus step: fan out to the reviewer panel concurrently, merge,
            # and feed the ranked critiques back to the executor.
            if step.kind == "consensus" and step.panel:
                iter_result = IterationResult(
                    iteration_number=iteration_num,
                    role=step.role,
                    status=IterationStatus.RUNNING,
                    started_at=datetime.utcnow(),
                )
                try:
                    merged, formatted, raw_results = await self._run_consensus_step(
                        step, run_state, iteration_num, previous_output, workspace_path
                    )
                except Exception as exc:
                    iter_result.status = IterationStatus.FAILED
                    iter_result.error = str(exc)
                    iter_result.completed_at = datetime.utcnow()
                    run_state.iterations.append(iter_result)
                    sm.fail(f"Consensus error at iteration {iteration_num}: {exc}")
                    return run_state

                panel_tokens = sum(r.token_usage for r in raw_results)
                panel_cost = sum(r.cost_estimate for r in raw_results)
                run_state.total_tokens += panel_tokens
                run_state.total_cost += panel_cost

                report = ConsensusReviewer(
                    consensus_threshold=run_state.config.get("consensus_threshold", 2),
                    panel=[(pm.role, pm.prompt_ref) for pm in step.panel] or None,
                ).compute_consensus_report(merged)

                iter_result.status = IterationStatus.COMPLETED
                iter_result.completed_at = datetime.utcnow()
                iter_result.output_summary = formatted[:500]
                iter_result.token_usage = panel_tokens
                iter_result.cost_estimate = panel_cost
                iter_result.feedback = formatted
                if self._artifact_writer and formatted:
                    try:
                        self._artifact_writer(iteration_num, "EVAL.md", formatted)
                        iter_result.artifacts_produced = ["EVAL.md"]
                    except Exception:
                        logger.debug("Consensus artifact write failed for iter %d", iteration_num)
                run_state.iterations.append(iter_result)

                self._emit("consensus_merged", {
                    "run_id": run_state.run_id,
                    "iteration": iteration_num,
                    "panel_size": len(step.panel),
                    "report": report,
                })

                previous_output = formatted

                # Optional advancement gate (deadlock policy): while unresolved
                # CRITICAL critiques remain, the run is not allowed to converge
                # or complete; it keeps revising (bounded by max_iterations).
                gate = run_state.config.get("consensus_gate", False)
                self._consensus_block = bool(gate and report.get("unresolved_critical", 0) > 0)

                self._convergence.record_iteration(
                    iteration_num=iteration_num,
                    role=step.role,
                    output=formatted,
                    review=merged,
                )
                if run_state.config.get("convergence_enabled", True) and not self._consensus_block:
                    converged, reason = self._convergence.should_stop()
                    if converged:
                        self._emit("convergence_detected", {
                            "run_id": run_state.run_id,
                            "reason": reason,
                            "metrics": self._convergence.get_metrics(),
                        })
                        sm.complete()
                        return run_state

                step_index += 1
                continue

            # Build prompt and request
            prompt_bundle = self._build_prompt_bundle(step, run_state, previous_output)
            request = AdapterRunRequest(
                prompt_bundle=prompt_bundle,
                workspace_context=WorkspaceContext(
                    workspace_path=workspace_path,
                    run_dir=f"runs/{run_state.run_id}",
                    iteration_dir=f"runs/{run_state.run_id}/iter_{iteration_num:03d}",
                ),
                timeout_seconds=run_state.config.get("adapter_timeout", 900),
            )

            # Create iteration result
            iter_result = IterationResult(
                iteration_number=iteration_num,
                role=step.role,
                status=IterationStatus.RUNNING,
                started_at=datetime.utcnow(),
            )

            # Call adapter with retry
            def _on_retry(attempt: int, error: str, delay: float) -> None:
                self._emit("adapter_retry", {
                    "run_id": run_state.run_id,
                    "iteration": iteration_num,
                    "attempt": attempt,
                    "error": error,
                    "delay": delay,
                })

            try:
                adapter_result = await run_with_retry(
                    self.adapter,
                    request,
                    max_retries=self._max_retries,
                    on_retry=_on_retry,
                )
            except Exception as exc:
                iter_result.status = IterationStatus.FAILED
                iter_result.error = str(exc)
                iter_result.completed_at = datetime.utcnow()
                run_state.iterations.append(iter_result)
                sm.fail(f"Adapter error at iteration {iteration_num}: {exc}")
                return run_state

            # Clean thinking traces from adapter output before any downstream use
            if adapter_result.output:
                adapter_result.output = strip_thinking_traces(adapter_result.output)

            # Update iteration result
            iter_result.status = (
                IterationStatus.COMPLETED if adapter_result.success
                else IterationStatus.FAILED
            )
            iter_result.completed_at = datetime.utcnow()
            iter_result.output_summary = adapter_result.output[:500]
            iter_result.artifacts_produced = adapter_result.artifacts_written
            iter_result.token_usage = adapter_result.token_usage
            iter_result.cost_estimate = adapter_result.cost_estimate
            iter_result.error = adapter_result.error

            run_state.iterations.append(iter_result)
            run_state.total_tokens += adapter_result.token_usage
            run_state.total_cost += adapter_result.cost_estimate

            # Parse adapter output into discrete artifacts
            parsed_artifacts: dict[str, str] = {}
            if adapter_result.output:
                try:
                    parsed_artifacts = self._artifact_parser.parse_output(
                        adapter_result.output, role=step.role
                    )
                except Exception:
                    logger.debug("Artifact parsing failed for iter %d", iteration_num)

            # Write parsed artifacts via callback (or fall back to single blob)
            if self._artifact_writer and adapter_result.output:
                try:
                    if parsed_artifacts:
                        for art_name, art_content in parsed_artifacts.items():
                            self._artifact_writer(iteration_num, art_name, art_content)
                        iter_result.artifacts_produced = list(parsed_artifacts.keys())
                    else:
                        artifact_name = f"{step.role}_output.md"
                        self._artifact_writer(iteration_num, artifact_name, adapter_result.output)
                        iter_result.artifacts_produced = [artifact_name]
                except Exception:
                    logger.debug("Artifact writer failed for iter %d", iteration_num)

            self._emit("iteration_completed", {
                "run_id": run_state.run_id,
                "iteration": iteration_num,
                "role": step.role,
                "success": adapter_result.success,
                "tokens": adapter_result.token_usage,
            })

            if not adapter_result.success:
                sm.fail(f"Adapter failed at iteration {iteration_num}: {adapter_result.error}")
                return run_state

            # After idea_generator step, parse follow-up proposals
            if step.role == "idea_generator" and adapter_result.output:
                try:
                    proposals = self._followup_executor.parse_proposals(adapter_result.output)
                    if proposals:
                        run_state.config["follow_up_proposals"] = [
                            p.model_dump() for p in proposals
                        ]
                        self._emit("follow_up_proposals", {
                            "run_id": run_state.run_id,
                            "count": len(proposals),
                            "titles": [p.title for p in proposals],
                        })
                except Exception:
                    logger.debug("Follow-up parsing failed for iter %d", iteration_num)

            # Build role-aware previous_output for next step:
            # reviewer output -> structured feedback for executors
            # executor output -> raw output for reviewers
            review_result = None
            if step.role in ("reviewer", "adversarial_reviewer"):
                try:
                    review_result = self._feedback_formatter.parse_review(adapter_result.output)
                    formatted = self._feedback_formatter.format_for_executor(review_result)
                    iter_result.feedback = formatted
                    previous_output = formatted
                except Exception:
                    logger.debug("Feedback formatting failed for iter %d", iteration_num)
                    previous_output = adapter_result.output
            else:
                previous_output = adapter_result.output

            # Optionally execute analysis code emitted by the executor so that
            # reviewers critique executed results, not an untested plan.
            if (step.role == "executor"
                    and run_state.config.get("code_execution_enabled", False)
                    and adapter_result.output):
                try:
                    executor = CodeExecutor(
                        timeout_seconds=run_state.config.get("code_execution_timeout", 20)
                    )
                    exec_report = await executor.run_artifact(adapter_result.output)
                    if exec_report.executed:
                        report_text = exec_report.to_feedback()
                        previous_output = previous_output + "\n\n" + report_text
                        iter_result.code_execution = {
                            "executed": exec_report.executed,
                            "passed": exec_report.passed,
                            "failed": exec_report.failed,
                        }
                        if self._artifact_writer:
                            try:
                                self._artifact_writer(iteration_num, "RUN_LOG.md", report_text)
                            except Exception:
                                logger.debug("Run-log write failed for iter %d", iteration_num)
                        self._emit("code_executed", {
                            "run_id": run_state.run_id,
                            "iteration": iteration_num,
                            "executed": exec_report.executed,
                            "passed": exec_report.passed,
                            "failed": exec_report.failed,
                        })
                except Exception:
                    logger.debug("Code execution failed for iter %d", iteration_num)

            # Record for convergence detection
            self._convergence.record_iteration(
                iteration_num=iteration_num,
                role=step.role,
                output=adapter_result.output,
                review=review_result,
            )

            # Check convergence (if enabled and not held open by the consensus gate)
            if run_state.config.get("convergence_enabled", True) and not self._consensus_block:
                converged, reason = self._convergence.should_stop()
                if converged:
                    self._emit("convergence_detected", {
                        "run_id": run_state.run_id,
                        "reason": reason,
                        "metrics": self._convergence.get_metrics(),
                    })
                    sm.complete()
                    return run_state

            step_index += 1

        # Should not reach here, but just in case
        sm.complete()
        return run_state
