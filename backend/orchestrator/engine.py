"""Main loop execution engine — runs iterations via adapter calls.

One engine instance drives one run. Each loop iteration executes one compiled
step: a single adapter call (executor, reviewer lens, idea generator, ...) or a
consensus step (a reviewer panel run concurrently and merged).

Prompt assembly
---------------
* Executor steps (role ``executor``) receive the task on the first iteration
  and afterwards a ``=== REVISION REQUEST ===`` prompt with all feedback
  produced since their previous submission (formatted reviews, consensus
  feedback, idea-generator proposals), the code-execution report of their
  previous submission (when verified execution is enabled) and their previous
  submission itself.
* Every other step (reviewer lenses, consensus panels, idea generator,
  knowledge extractor, ...) receives a neutral framing: the task given to the
  executor (as context only), the iteration number and
  ``=== ARTIFACT UNDER REVIEW ===`` followed by the executor's latest FULL
  output plus its code-execution report. Non-executor steps never receive the
  executor's revision instructions.

Adapter calls carry the run's ``model``, the run-config ``reasoning_effort``
and run-config ``adapter_options``. Non-executor calls (single reviewer steps,
every consensus lens and the LLM adjudicator) run with ``allow_tools=False``
unless run config ``reviewer_allow_tools`` is true. Executor calls keep agent
tools (``executor_allow_tools``, default true) EXCEPT when verified code
execution is enabled: then the default is false, so every reported number has
to come from the orchestrator's sandboxed ``CodeExecutor`` and not from the
agent's own shell (setting ``executor_allow_tools=true`` explicitly together
with code execution is allowed but logged, emitted as an event and recorded
in run_meta). Tokens (input + output) and provider-reported cost, including
the LLM adjudicator's, are accumulated on the RunState so that
``budget_max_tokens`` / ``budget_max_cost`` take effect.

Stopping
--------
Checked before every step, in this order: cancellation (``cancelled``),
budgets (``budget_tokens``, ``budget_cost``, ``max_iterations``), ``on_flag``
stop conditions (``stop_condition:on_flag:<flag>``) and, before an executor
step that will run, the revision budget (``revision_budget``). Checked after
every step: ``grade_at_least`` after review / consensus steps
(``stop_condition:grade_at_least``) and, when ``convergence_enabled`` is true,
the configurable convergence rule (``converged:<signals>``, see
:mod:`backend.orchestrator.convergence`). Whichever condition is reached
first stops the run; when several hold at the same point the order above
decides the reason (e.g. a run whose ``max_iterations`` cap coincides with the
end of its revision budget stops with ``max_iterations``). Failures end the
run with ``failed:<reason>``. The reason is stored in ``RunState.stop_reason``.

Default stopping policy (calibrated, see :mod:`backend.orchestrator.convergence`):
run config ``revision_budget`` (default
:data:`~backend.orchestrator.convergence.DEFAULT_REVISION_BUDGET` = 4; an
integer >= 0, or ``None`` / ``null`` to disable) is the maximum number of
executor revisions after the initial executor submission. A
``seed_executor_output`` counts as the initial submission. When the executor
has produced its initial submission plus ``revision_budget`` revisions, the
run executes the remaining non-executor steps (so the final artifact is
reviewed) and stops before the next executor step with stop reason
``revision_budget``. Loops without an executor step are never stopped by it.
A run-config key that is present with the value ``None`` disables the budget
(unlike other keys, where ``None`` means "unset"). ``convergence_enabled``
defaults to false; the adaptive rule (``DEFAULT_STOPPING_RULE``) is evaluated
only when it is enabled. On resume the executor submissions are counted from
the stored completed executor iterations (plus the seed).

Consensus panels: a panel with no usable lens is retried once and then fails
the step (``INCOMPLETE`` is never a grade). A *partial* panel (some lenses
failed, timed out or returned unparseable output) is retried once for the
failed lenses only. If it is still partial, ``consensus_partial_policy``
decides: ``no_stop`` (default) keeps the iteration and its feedback but the
grade is not stop-eligible (no ``grade_at_least``, no convergence
observation, flagged ``grade_valid=False`` in the consensus report);
``count`` treats the partial grade as a full review; ``fail`` fails the step
(``failed:consensus_panel_partial``).

Conditional steps: ``every_k:N`` runs in every N-th cycle of the compiled
plan (cycles counted from 1; with ``executor -> reviewer -> adversarial
(every_k:3)`` the adversarial runs in cycles 3, 6, 9, ...). ``on_flag:F``
runs when ``flags[F]`` in run config is true (flags are static run-config
switches; nothing sets them at runtime).

Configuration precedence (highest first): run config (``RunState.config``)
> loop ``stop_conditions`` > loop ``config`` block > consensus-merger node
config (consensus keys only) > workspace defaults (``budget_max_tokens`` /
``budget_max_cost`` from the workspace's per-run budgets) > built-in
defaults.

Persistence
-----------
For every single step the FULL cleaned output is written as
``<role>_output.md`` in ``runs/<run_id>/iter_NNNN/`` in addition to the
artifacts parsed from it; consensus steps write ``EVAL.md`` /
``consensus_merger_output.md`` (merged feedback), one ``<lens>_output.md`` per
panel lens and ``CONSENSUS.json``; executor steps with verified execution
write ``RUN_LOG.md`` and ``CODE_EXECUTION.json``. Every non-executor step also
writes ``<role>_feedback.md``: the exact text queued for the next executor
prompt, so a resumed run rebuilds the same prompt. An optional
``on_iteration`` callback receives every finished :class:`IterationResult`
(the runner uses it to persist iterations incrementally).

Seeded start
------------
Run config ``seed_executor_output`` (text) is treated as the executor's
iteration-0 output: before the first step it becomes the latest executor
submission (the artifact under review of the first non-executor step and the
"previous submission (iteration 0)" of the first executor revision) and is
recorded as the first executor output of the output-similarity signal (it
does not count as an engine iteration). A fresh run then starts with the
plan step AFTER the first executor step, e.g. ``reviewer_consensus`` runs
panel, executor, panel, ... A resumed run re-applies the seed before the
stored iterations are restored. A non-string value, or a plan without an
executor step, fails the run with ``failed:invalid_config``; an empty or
whitespace-only string means no seed.

Resume
------
``run_loop(resume_from=N)`` replays the compiled plan against the stored
iterations to find the next plan step (gated steps make the step position
differ from the iteration count), restores the latest executor output, the
pending feedback (from ``<role>_feedback.md``), the convergence history
(partial-panel grades excluded under ``no_stop``), the executor-submission
count of the revision budget and the consensus gate.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional, Union

from backend.adapters.base import BaseAdapter
from backend.logging_config import set_run_context
from backend.artifacts.parser import ArtifactParser
from backend.models import (
    AdapterRunRequest,
    AdapterRunResult,
    BudgetStatus,
    IterationResult,
    IterationStatus,
    LoopDefinition,
    PromptBundle,
    ReviewResult,
    RunState,
    RunStatus,
    SeverityLevel,
    WorkspaceContext,
)
from backend.orchestrator.code_executor import CodeExecutor
from backend.orchestrator.consensus import ConsensusReviewer
from backend.orchestrator.context import MAX_ARTIFACT_CHARS, WorkspaceContextReader
from backend.orchestrator.convergence import (
    DEFAULT_CONVERGENCE_ENABLED,
    DEFAULT_REVISION_BUDGET,
    REVISION_BUDGET_STOP_REASON,
    ConvergenceDetector,
    StoppingRule,
    grade_at_least,
    is_executor_role,
    parse_revision_budget,
)
from backend.orchestrator.dsl import (
    ExecutionStep,
    compile_loop,
    merger_config_defaults,
    parse_stop_conditions,
)
from backend.orchestrator.feedback import FeedbackFormatter
from backend.orchestrator.followups import FollowUpExecutor
from backend.orchestrator.retry import run_with_retry
from backend.orchestrator.state_machine import RunStateMachine
from backend.registry.prompts import PromptRegistry
from backend.utils.output_cleaner import strip_thinking_traces

logger = logging.getLogger(__name__)


EventCallback = Callable[[str, dict[str, Any]], Any]
ArtifactWriter = Callable[[int, str, str], Any]
IterationCallback = Callable[[IterationResult, RunState], Union[Awaitable[None], None]]

#: Iteration directory name; identical to ``backend.artifacts.writer``.
ITERATION_DIR_FMT = "iter_{:04d}"

#: Single-step roles whose output is parsed as a review (grade + critiques).
REVIEW_ROLES = frozenset({
    "reviewer", "adversarial_reviewer", "adversarial", "bio_plausibility_checker",
})

#: Prompt appended to the executor system prompt when code execution is on.
EXECUTOR_CODE_PROMPT_REF = "executor/mi_executor_code"

_CANONICAL_ARTIFACTS = ("EVAL.md", "MECH.md", "METHOD.md", "XP.md")

_NO_CODE_REPORT = (
    "=== CODE EXECUTION REPORT ===\n"
    "Verified execution is enabled, but the executor output contains no fenced "
    "```python block, so nothing was executed. No number in this write-up is "
    "backed by executed code.\n"
    "=== END CODE EXECUTION REPORT ==="
)

_FALLBACK_CODE_ADDENDUM = (
    "VERIFIED CODE EXECUTION IS ENABLED. Put your complete analysis in ONE "
    "self-contained ```python block that reads only from the data paths given in "
    "the task, prints a compact results summary and writes results.json in the "
    "working directory. Report only numbers that the code computes."
)

# path -> ((path, mtime_ns, size), PromptTemplate); see LoopEngine._get_template
_TEMPLATE_CACHE: dict[str, tuple[tuple[str, int, int], Any]] = {}

_DEFAULT_TOKEN_BUDGET = 500_000
_DEFAULT_COST_BUDGET = 10.0
_CONSENSUS_ATTEMPTS = 2  # one retry when the panel failed or is partial

#: What to do with a consensus panel that is still partial after its retry.
PARTIAL_POLICIES = ("no_stop", "count", "fail")
DEFAULT_PARTIAL_POLICY = "no_stop"


def iteration_dir_name(iteration_number: int) -> str:
    """Directory name of an iteration (``iter_0007``)."""
    return ITERATION_DIR_FMT.format(int(iteration_number))


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _safe_name(role: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", role or "step").strip("._") or "step"


def _result_tokens(r: AdapterRunResult) -> int:
    """Tokens of one call: input + output when reported, else ``token_usage``."""
    io = int(getattr(r, "input_tokens", 0) or 0) + int(getattr(r, "output_tokens", 0) or 0)
    return io if io > 0 else int(getattr(r, "token_usage", 0) or 0)


def _join_unique(values) -> str:
    seen: list[str] = []
    for v in values:
        v = (v or "").strip()
        if v and v not in seen:
            seen.append(v)
    return ",".join(seen)


def _severity_counts(review: ReviewResult) -> tuple[int, int]:
    critical = sum(1 for c in review.critiques if c.severity == SeverityLevel.CRITICAL)
    high = sum(1 for c in review.critiques if c.severity == SeverityLevel.HIGH)
    return critical, high


class _RequestDefaultsAdapter:
    """Adapter proxy that fills unset request fields (model, effort, tools).

    Used for the LLM adjudicator of the consensus step, which builds its own
    requests.
    """

    def __init__(self, inner: Any, defaults: dict[str, Any]):
        self.inner = inner
        self.defaults = dict(defaults)
        self.name = getattr(inner, "name", "")

    async def run(self, request: AdapterRunRequest) -> AdapterRunResult:
        update = {}
        for key, value in self.defaults.items():
            if key == "allow_tools":
                update[key] = value
            elif not getattr(request, key, None):
                update[key] = value
        if update:
            request = request.model_copy(update=update)
        return await self.inner.run(request)


@dataclass
class _StepOutcome:
    failed: bool = False
    reason: str = ""
    message: str = ""
    review_grade: Optional[str] = None  # set for review / consensus steps
    # A partial consensus panel under the "no_stop" policy: the step's grade
    # must not trigger grade_at_least or convergence.
    partial: bool = False
    failed_lenses: Optional[list[str]] = None


def _as_number(name: str, value: Any) -> float:
    """Positive budget number; strings such as "2,000,000" are rejected."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive number, got {value!r}") from exc
    if not num > 0:
        raise ValueError(f"{name} must be a positive number, got {value!r}")
    return num


def _tool_counts(results: list[AdapterRunResult], allow_tools: bool) -> dict[str, Any]:
    """Aggregate agent-tool provenance over a step's adapter calls."""
    out: dict[str, Any] = {
        "tools_enabled": bool(allow_tools), "calls": len(results), "sandbox": "",
        "tool_calls": 0, "command_executions": 0, "web_searches": 0, "mcp_tool_calls": 0,
    }
    sandboxes: list[str] = []
    for r in results:
        so = getattr(r, "structured_output", None) or {}
        meta = so.get("adapter_meta") or {}
        if meta.get("sandbox"):
            sandboxes.append(str(meta["sandbox"]))
        counts = so.get("tool_counts") or {}
        for key in ("tool_calls", "command_executions", "web_searches", "mcp_tool_calls"):
            try:
                out[key] += int(counts.get(key, 0) or 0)
            except (TypeError, ValueError):
                pass
    out["sandbox"] = _join_unique(sandboxes)
    return out


class LoopEngine:
    """Async engine that drives the executor/reviewer/adversarial loop."""

    def __init__(
        self,
        adapter: BaseAdapter,
        on_event: Optional[EventCallback] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        artifact_writer: Optional[ArtifactWriter] = None,
        max_retries: int = 3,
        on_iteration: Optional[IterationCallback] = None,
    ):
        self.adapter = adapter
        self._on_event = on_event
        self._prompt_registry = prompt_registry or PromptRegistry()
        self._artifact_writer = artifact_writer
        self._max_retries = max_retries
        self._on_iteration = on_iteration
        self._context_reader: Optional[WorkspaceContextReader] = None
        self._artifact_parser = ArtifactParser()
        self._feedback_formatter = FeedbackFormatter()
        self._followup_executor = FollowUpExecutor()
        self._convergence = ConvergenceDetector()
        # Set when a consensus gate is holding the loop open on unresolved CRITICAL.
        self._consensus_block = False
        self._reset_run_memory()

    # ── per-run memory ───────────────────────────────────────────────

    def _reset_run_memory(self) -> None:
        self._loop_defaults: dict[str, Any] = {}
        self._stop_flags: list[str] = []
        self._stop_unknown: list[str] = []
        self._workspace_path = ""
        self._run_id = ""
        self._code_executor: Optional[CodeExecutor] = None
        # Latest executor submission (full cleaned output) and its execution report.
        self._last_executor_output = ""
        self._last_executor_iteration = 0
        self._last_exec_report = ""
        # Feedback produced since the latest executor submission: (role, text).
        self._pending_feedback: list[tuple[str, str]] = []
        # Lowest-priority config defaults supplied by the runner (workspace budgets).
        self._workspace_defaults: dict[str, Any] = {}
        # Effective agent-tool settings of this run (recorded in run_meta).
        self._tool_settings: dict[str, Any] = {}
        self._partial_policy = DEFAULT_PARTIAL_POLICY
        # Run config ``seed_executor_output``: the executor's iteration-0 output.
        self._seed_output = ""
        # Revision budget (None = disabled) and the executor submissions so far
        # (the seed counts as the initial submission).
        self._revision_budget: Optional[int] = None
        self._executor_submissions = 0
        # Effective stopping policy of this run (recorded in run_meta).
        self._stopping_settings: dict[str, Any] = {}

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        if self._on_event:
            self._on_event(event_type, data)

    # ── configuration ────────────────────────────────────────────────

    def _conf(self, run_state: RunState, key: str, default: Any = None) -> Any:
        """Effective config value: run config > loop stop conditions / config
        > workspace defaults > ``default``."""
        cfg = run_state.config or {}
        if cfg.get(key) is not None:
            return cfg[key]
        if self._loop_defaults.get(key) is not None:
            return self._loop_defaults[key]
        if self._workspace_defaults.get(key) is not None:
            return self._workspace_defaults[key]
        return default

    def tool_settings(self) -> dict[str, Any]:
        """Effective agent-tool settings of the current/last run (for run_meta)."""
        return dict(self._tool_settings)

    def stopping_settings(self) -> dict[str, Any]:
        """Effective stopping policy of the current/last run (for run_meta):
        revision budget and convergence switch with their sources
        (``config`` / ``loop`` / ``default``), the adaptive rule, and the
        executor submissions / revisions made so far. Empty when the run
        failed before its configuration was resolved."""
        if not self._stopping_settings:
            return {}
        out = dict(self._stopping_settings)
        out["executor_submissions"] = self._executor_submissions
        out["executor_revisions"] = max(0, self._executor_submissions - 1)
        return out

    def _config_source(self, run_state: RunState, key: str) -> str:
        """Where the effective value of ``key`` comes from (``_conf`` order)."""
        if (run_state.config or {}).get(key) is not None:
            return "config"
        if self._loop_defaults.get(key) is not None:
            return "loop"
        if self._workspace_defaults.get(key) is not None:
            return "workspace"
        return "default"

    def _resolve_revision_budget(self, run_state: RunState) -> tuple[Optional[int], str]:
        """Effective ``revision_budget`` and its source. Unlike ``_conf``, the
        key's PRESENCE counts: ``None`` in the run config (or
        ``revision_budget:none`` in the loop) disables the budget instead of
        falling through to the default. Raises ``ValueError``."""
        cfg = run_state.config or {}
        if "revision_budget" in cfg:
            return parse_revision_budget(cfg["revision_budget"]), "config"
        if "revision_budget" in self._loop_defaults:
            return parse_revision_budget(self._loop_defaults["revision_budget"]), "loop"
        return DEFAULT_REVISION_BUDGET, "default"

    def effective_config(
        self, run_state: RunState, extra_defaults: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Merged config dict (``extra_defaults`` < loop defaults < run config)."""
        merged: dict[str, Any] = dict(self._workspace_defaults)
        merged.update(extra_defaults or {})
        merged.update({k: v for k, v in self._loop_defaults.items() if v is not None})
        merged.update({k: v for k, v in (run_state.config or {}).items() if v is not None})
        return merged

    def _configure(self, run_state: RunState, loop_def: LoopDefinition) -> None:
        """Resolve loop defaults, the stopping rule and the code executor.

        Raises ``ValueError`` on invalid configuration.
        """
        stop = parse_stop_conditions(loop_def.stop_conditions)
        defaults: dict[str, Any] = dict(loop_def.config or {})
        defaults.update(stop.config)
        self._loop_defaults = defaults
        self._stop_flags = list(stop.flags)
        self._stop_unknown = list(stop.unknown)
        cfg = self.effective_config(run_state)
        self._convergence = ConvergenceDetector(rule=StoppingRule.from_config(cfg))
        thr = self._conf(run_state, "grade_at_least")
        if thr is not None and str(thr).strip():
            from backend.orchestrator.dsl import _to_grade
            _to_grade(thr)  # validate
        if _as_bool(self._conf(run_state, "code_execution_enabled", False)):
            self._code_executor = CodeExecutor.from_run_config(cfg)
        else:
            self._code_executor = None
        self._limits(run_state)  # validate budgets (raises ValueError)
        policy = str(self._conf(run_state, "consensus_partial_policy", DEFAULT_PARTIAL_POLICY)
                     ).strip().lower()
        if policy not in PARTIAL_POLICIES:
            raise ValueError(f"consensus_partial_policy must be one of {PARTIAL_POLICIES}, "
                             f"got {policy!r}")
        self._partial_policy = policy
        opts = self._conf(run_state, "adapter_options")
        if opts is not None and not isinstance(opts, dict):
            raise ValueError(f"adapter_options must be a mapping, got {opts!r}")
        seed = self._conf(run_state, "seed_executor_output")
        if seed is not None and not isinstance(seed, str):
            raise ValueError("seed_executor_output must be a string (the executor's "
                             f"iteration-0 output), got {type(seed).__name__}")
        self._seed_output = seed if (seed and seed.strip()) else ""
        # Default stopping policy: revision budget + convergence switch.
        budget, budget_source = self._resolve_revision_budget(run_state)
        self._revision_budget = budget
        self._stopping_settings = {
            "revision_budget": budget,
            "revision_budget_source": budget_source,
            "convergence_enabled": _as_bool(self._conf(
                run_state, "convergence_enabled", DEFAULT_CONVERGENCE_ENABLED)),
            "convergence_enabled_source": self._config_source(run_state, "convergence_enabled"),
            "stopping_rule": self._convergence.rule.to_dict(),
            "seeded": bool(self._seed_output),
        }
        # Effective agent-tool settings (recorded in run_meta).
        raw_exec = self._conf(run_state, "executor_allow_tools")
        code_on = self._code_executor is not None
        exec_tools = (not code_on) if raw_exec is None else _as_bool(raw_exec)
        self._tool_settings = {
            "executor_allow_tools": exec_tools,
            "executor_allow_tools_source": "default" if raw_exec is None else "config",
            "reviewer_allow_tools": _as_bool(self._conf(run_state, "reviewer_allow_tools", False)),
            "code_execution_enabled": code_on,
            "executor_tools_with_code_execution": bool(code_on and exec_tools),
        }
        if code_on and exec_tools:
            logger.warning(
                "Run %s: executor_allow_tools=true with verified code execution; the "
                "executor agent can run its own commands outside the CodeExecutor",
                run_state.run_id)

    def _request_kwargs(self, run_state: RunState, *, executor: bool) -> dict[str, Any]:
        kw: dict[str, Any] = {}
        model = (run_state.model or "").strip()
        if model:
            kw["model"] = model
        effort = str(self._conf(run_state, "reasoning_effort", "") or "").strip()
        if effort:
            kw["reasoning_effort"] = effort
        if executor:
            raw = self._conf(run_state, "executor_allow_tools")
            if raw is None:
                # Verified execution on => no agent tools by default, so every
                # number must come from the sandboxed CodeExecutor.
                kw["allow_tools"] = self._code_executor is None
            else:
                kw["allow_tools"] = _as_bool(raw)
        else:
            kw["allow_tools"] = _as_bool(self._conf(run_state, "reviewer_allow_tools", False))
        opts = self._conf(run_state, "adapter_options")
        if isinstance(opts, dict) and opts:
            kw["adapter_options"] = dict(opts)
        return kw

    # ── budget / conditions ──────────────────────────────────────────

    def _limits(self, run_state: RunState) -> tuple[float, float, int]:
        """(token budget, cost budget, iteration cap). Invalid budget values
        raise ``ValueError`` (the run fails with ``failed:invalid_config``)
        instead of silently falling back to the defaults."""
        tokens_limit = _as_number(
            "budget_max_tokens", self._conf(run_state, "budget_max_tokens", _DEFAULT_TOKEN_BUDGET))
        cost_limit = _as_number(
            "budget_max_cost", self._conf(run_state, "budget_max_cost", _DEFAULT_COST_BUDGET))
        iter_limit = int(run_state.max_iterations)
        cap = self._conf(run_state, "max_iterations")
        if cap is not None:
            try:
                iter_limit = min(iter_limit, int(cap))
            except (TypeError, ValueError):
                pass
        return tokens_limit, cost_limit, iter_limit

    def _evaluate_budget(self, run_state: RunState) -> tuple[BudgetStatus, Optional[str]]:
        tokens_limit, cost_limit, iter_limit = self._limits(run_state)
        kind: Optional[str] = None
        reason = None
        if run_state.total_tokens >= tokens_limit:
            kind = "budget_tokens"
            reason = f"Token budget exceeded: {run_state.total_tokens}/{tokens_limit:g}"
        elif run_state.total_cost >= cost_limit:
            kind = "budget_cost"
            reason = f"Cost budget exceeded: {run_state.total_cost:.4f}/{cost_limit:.2f}"
        elif run_state.current_iteration >= iter_limit:
            kind = "max_iterations"
            reason = f"Iteration limit reached: {run_state.current_iteration}/{iter_limit}"
        status = BudgetStatus(
            tokens_used=run_state.total_tokens,
            tokens_limit=int(tokens_limit),
            cost_used=run_state.total_cost,
            cost_limit=cost_limit,
            iterations_used=run_state.current_iteration,
            iterations_limit=iter_limit,
            exceeded=kind is not None,
            reason=reason,
        )
        return status, kind

    def _check_budget(self, run_state: RunState) -> BudgetStatus:
        return self._evaluate_budget(run_state)[0]

    def _flags(self, run_state: RunState) -> dict[str, Any]:
        flags = self._conf(run_state, "flags", {}) or {}
        return flags if isinstance(flags, dict) else {}

    def _triggered_stop_flag(self, run_state: RunState) -> Optional[str]:
        flags = self._flags(run_state)
        for flag in self._stop_flags:
            if _as_bool(flags.get(flag, False)):
                return flag
        return None

    def _should_run_step(
        self, step: ExecutionStep, iteration: int,
        run_state: Optional[RunState] = None,
        cycle: Optional[int] = None,
    ) -> bool:
        """Evaluate whether a step should execute given its condition.

        ``every_k:N`` runs in every N-th plan cycle when ``cycle`` (1-based)
        is given (the engine always passes it); without it the legacy
        iteration-count test ``iteration % N == 0`` is used. ``on_flag:F``
        runs when ``flags[F]`` is true (strings parsed like every other
        boolean config value, so ``"false"`` is false).
        """
        cond = step.condition
        if cond == "always":
            return True
        if cond.startswith("every_k:"):
            k = int(cond.split(":")[1])
            if k <= 0:
                return False
            return (cycle % k == 0) if cycle is not None else (iteration % k == 0)
        if cond.startswith("on_flag:"):
            # Run when the named flag is truthy in run_state.config["flags"].
            flag = cond.split(":", 1)[1].strip()
            if run_state is None:
                return False
            return _as_bool(self._flags(run_state).get(flag, False))
        return True

    # ── prompts ──────────────────────────────────────────────────────

    def _get_template(self, group: str, name: str):
        """Load a prompt template, cached by file path + mtime + size.

        YAML parsing costs ~20 ms per prompt; the cache keeps the engine's
        per-step overhead small while still picking up edited prompt files.
        """
        registry = self._prompt_registry
        try:
            path = Path(registry.prompts_dir) / group / f"{name}.yaml"
            st = path.stat()
            key = (str(path), st.st_mtime_ns, st.st_size)
        except Exception:
            return registry.load_prompt(group, name)
        cached = _TEMPLATE_CACHE.get(key[0])
        if cached is not None and cached[0] == key:
            return cached[1]
        template = registry.load_prompt(group, name)
        _TEMPLATE_CACHE[key[0]] = (key, template)
        return template

    def _load_template(self, prompt_ref: str, variables: dict[str, str]) -> Optional[str]:
        if not prompt_ref or not self._prompt_registry:
            return None
        try:
            parts = prompt_ref.split("/", 1)
            if len(parts) == 2:
                template = self._get_template(parts[0], parts[1])
                return self._prompt_registry.resolve_template(template, variables)
        except Exception:
            logger.debug("Prompt registry lookup failed for %s, using inline", prompt_ref)
        return None

    def _code_addendum(self, run_state: RunState) -> str:
        ex = self._code_executor
        ro = []
        backend = "subprocess"
        timeout = self._conf(run_state, "code_execution_timeout", 20)
        if ex is not None:
            ro = list(getattr(ex, "read_only_paths", []) or [])
            backend = getattr(ex, "backend", backend)
            timeout = getattr(ex, "timeout_seconds", timeout)
        variables = {
            "timeout_seconds": f"{float(timeout):g}",
            "backend": str(backend),
            "data_paths": ", ".join(str(p) for p in ro) if ro else "(see the task)",
        }
        text = self._load_template(EXECUTOR_CODE_PROMPT_REF, variables)
        return text if text else _FALLBACK_CODE_ADDENDUM

    def _resolve_system_prompt(
        self, role: str, prompt_ref: str, run_state: RunState, iteration: Optional[int] = None,
    ) -> str:
        """System prompt for a role: registry template, else an inline fallback.

        Executor prompts get the verified-execution addendum
        (``prompts/executor/mi_executor_code.yaml``) when code execution is on.
        """
        it = run_state.current_iteration if iteration is None else iteration
        variables = {"role": role, "iteration": str(it), "task": run_state.task}
        system_prompt = self._load_template(prompt_ref, variables)
        executor = is_executor_role(role)
        if system_prompt is None:
            if executor:
                system_prompt = (
                    f"You are the {role} in an automated MI research pipeline.\n"
                    f"Produce the research artifacts requested by the task."
                )
            else:
                system_prompt = (
                    f"You are the {role} in an automated MI research pipeline.\n"
                    f"Review the artifact under review and report critiques as "
                    f"[SEVERITY] lines."
                )
        if executor and self._code_executor is not None:
            system_prompt = system_prompt.rstrip() + "\n\n" + self._code_addendum(run_state)
        return system_prompt

    def _resolve_role_system_prompt(
        self, role: str, prompt_ref: str, run_state: RunState
    ) -> str:
        """Resolve a role's system prompt from the registry, or fall back inline."""
        return self._resolve_system_prompt(role, prompt_ref, run_state)

    def _context_section(self) -> str:
        """Latest canonical artifacts of this run (deterministic; disk)."""
        reader = self._context_reader
        if reader is None:
            return ""
        try:
            dirs = reader._iter_dirs()
        except Exception:
            return ""
        found: dict[str, tuple[int, str]] = {}
        for num, path in reversed(dirs):
            for name in _CANONICAL_ARTIFACTS:
                if name in found:
                    continue
                p = path / name
                if p.is_file():
                    try:
                        found[name] = (num, p.read_text())
                    except Exception:
                        logger.debug("Failed to read %s", p)
        if not found:
            return ""
        sections = ["=== Current Artifact State ===", ""]
        for name in sorted(found):
            num, content = found[name]
            if len(content) > MAX_ARTIFACT_CHARS:
                content = "...[truncated]\n" + content[-MAX_ARTIFACT_CHARS:]
            sections.append(f"### {name} (from iteration {num})")
            sections.append(content)
            sections.append("")
        sections.append("=== End Current State ===")
        return "\n".join(sections)

    def _artifact_under_review(self) -> str:
        if self._last_executor_output:
            text = self._last_executor_output.rstrip()
        else:
            text = "(No executor output is available yet.)"
        if self._last_exec_report:
            text += "\n\n" + self._last_exec_report.strip()
        return text

    def _build_review_user_prompt(
        self, role: Optional[str], run_state: RunState, iteration: int,
    ) -> str:
        """Neutral framing for every non-executor step (and consensus lenses)."""
        if role is None or role in REVIEW_ROLES:
            who = (f"You are the {role} of an automated MI review pipeline."
                   if role else "You are one lens of an automated MI review panel.")
            header = f"=== REVIEW REQUEST (Iteration {iteration}) ==="
            ask = ("Assess the artifact under review below. Do not carry out the task "
                   "yourself, do not revise the artifact and do not write any files; "
                   "respond only with your assessment in the output format required "
                   "by your instructions.")
        else:
            who = f"You are the {role} in an automated MI research pipeline."
            header = f"=== INPUT FOR {role.upper()} (Iteration {iteration}) ==="
            ask = ("Work from the artifact under review below. Do not carry out the "
                   "executor's task and do not revise the artifact; respond only in the "
                   "output format required by your instructions.")
        if self._last_exec_report:
            ask += (" The artifact is followed by the report of the verified execution "
                    "of its code; check every reported number against it.")
        return (
            f"{header}\n{who}\n{ask}\n\n"
            f"=== TASK GIVEN TO THE EXECUTOR (context only) ===\n"
            f"{run_state.task}\n\n"
            f"=== ARTIFACT UNDER REVIEW ===\n"
            f"{self._artifact_under_review()}\n"
            f"=== END ARTIFACT UNDER REVIEW ==="
        )

    def _build_executor_user_prompt(self, run_state: RunState, iteration: int) -> str:
        task = run_state.task
        feedback = list(self._pending_feedback)
        have_prior = bool(self._last_executor_output or feedback or self._last_exec_report)
        if not have_prior:
            ctx = self._context_section()
            return f"{ctx}\n\n{task}" if ctx else task

        parts: list[str] = []
        if feedback:
            parts.append(
                f"=== REVISION REQUEST (Iteration {iteration}) ===\n"
                f"You produced artifacts in a previous iteration.\n"
                f"The reviewer(s) identified specific issues that MUST be addressed."
            )
            for role, text in feedback:
                parts.append(f"--- Feedback from {role} ---\n{text.strip()}")
        else:
            parts.append(
                f"=== REVISION REQUEST (Iteration {iteration}) ===\n"
                f"You produced artifacts in a previous iteration. No reviewer feedback "
                f"was produced since then; check and refine your artifacts."
            )
        if self._last_exec_report:
            k = self._last_executor_iteration
            parts.append(
                f"--- Verified execution of the code in your previous submission "
                f"(iteration {k}) ---\n"
                f"Fix any error shown here before reporting numbers; report only "
                f"numbers that the executed code computes.\n"
                f"{self._last_exec_report.strip()}"
            )
        if self._last_executor_output:
            k = self._last_executor_iteration
            parts.append(
                f"=== YOUR PREVIOUS SUBMISSION (iteration {k}) ===\n"
                f"{self._last_executor_output.strip()}\n"
                f"=== END PREVIOUS SUBMISSION ==="
            )
        else:
            ctx = self._context_section()
            if ctx:
                parts.append(ctx)
        parts.append(
            "TASK: Revise your artifacts to address ALL required fixes above.\n"
            "For each fix, make the specific change in the referenced artifact and section.\n"
            "Do NOT rewrite from scratch — make targeted revisions.\n\n"
            f"Original task: {task}"
        )
        return "\n\n".join(parts)

    def _build_prompt_bundle(
        self,
        step: ExecutionStep,
        run_state: RunState,
        previous_output: Optional[str] = None,
        iteration: Optional[int] = None,
    ) -> PromptBundle:
        """Build the prompt bundle for a single step (see module docstring).

        ``previous_output`` is accepted for backward compatibility: when given
        and the engine has no feedback of its own, it is treated as feedback
        (executor) or as the artifact under review (other roles).
        """
        it = run_state.current_iteration if iteration is None else iteration
        role = step.role
        if previous_output:
            if is_executor_role(role) and not self._pending_feedback:
                self._pending_feedback = [("previous step", previous_output)]
            elif not is_executor_role(role) and not self._last_executor_output:
                self._last_executor_output = previous_output
        variables = {"role": role, "iteration": str(it), "task": run_state.task}
        system_prompt = self._resolve_system_prompt(role, step.prompt_ref, run_state, it)
        if is_executor_role(role):
            user_prompt = self._build_executor_user_prompt(run_state, it)
        else:
            user_prompt = self._build_review_user_prompt(role, run_state, it)
        return PromptBundle(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            variables=variables,
        )

    # ── persistence helpers ──────────────────────────────────────────

    def _workspace_context(self, run_state: RunState, iteration_num: int) -> WorkspaceContext:
        return WorkspaceContext(
            workspace_path=self._workspace_path,
            run_dir=f"runs/{run_state.run_id}",
            iteration_dir=f"runs/{run_state.run_id}/{iteration_dir_name(iteration_num)}",
        )

    def _write(self, iteration_num: int, name: str, content: str) -> bool:
        if not self._artifact_writer:
            return False
        try:
            self._artifact_writer(iteration_num, name, content)
            return True
        except Exception:
            logger.debug("Artifact write failed for iter %d (%s)", iteration_num, name)
            return False

    def _iteration_file(self, run_state: RunState, iteration_num: int, name: str) -> Optional[Path]:
        if not self._workspace_path:
            return None
        run_dir = Path(self._workspace_path) / "runs" / run_state.run_id
        for d in (iteration_dir_name(iteration_num), f"iter_{iteration_num:03d}"):
            p = run_dir / d / name
            if p.is_file():
                return p
        return None

    def _read_iteration_file(self, run_state: RunState, iteration_num: int, name: str) -> str:
        p = self._iteration_file(run_state, iteration_num, name)
        if p is None:
            return ""
        try:
            return p.read_text()
        except Exception:
            return ""

    async def _record_iteration(self, run_state: RunState, it: IterationResult) -> None:
        run_state.iterations.append(it)
        if self._on_iteration is None:
            return
        try:
            res = self._on_iteration(it, run_state)
            if inspect.isawaitable(res):
                await res
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("on_iteration callback failed for iteration %d",
                           it.iteration_number, exc_info=True)

    def _account(
        self,
        run_state: RunState,
        it: IterationResult,
        results: list[AdapterRunResult],
        *,
        extra_tokens: int = 0,
        extra_cost: float = 0.0,
    ) -> None:
        """Add the calls' usage to the iteration and the run totals."""
        tokens = sum(_result_tokens(r) for r in results) + int(extra_tokens)
        cost = sum(float(getattr(r, "cost_estimate", 0.0) or 0.0) for r in results) + float(extra_cost)
        inp = sum(int(getattr(r, "input_tokens", 0) or 0) for r in results)
        out = sum(int(getattr(r, "output_tokens", 0) or 0) for r in results)
        cached = sum(int(getattr(r, "cached_input_tokens", 0) or 0) for r in results)
        it.token_usage += tokens
        it.cost_estimate += cost
        it.input_tokens += inp
        it.output_tokens += out
        it.cached_input_tokens += cached
        run_state.total_tokens += tokens
        run_state.total_cost += cost
        run_state.total_input_tokens += inp
        run_state.total_output_tokens += out
        run_state.total_cached_input_tokens += cached

    def _describe_calls(
        self, run_state: RunState, it: IterationResult, results: list[AdapterRunResult],
        request_kwargs: dict[str, Any],
    ) -> None:
        """Provider / model / effort / CLI version of the step's calls.

        Values come from the adapter results; the request values are used
        only when no result exists at all (e.g. the adapter raised), so an
        effort the CLI did not apply (Gemini: "unsupported") is never
        replaced by the requested one."""
        adapter_name = getattr(self.adapter, "name", "") or ""
        it.provider = _join_unique([getattr(r, "provider", "") for r in results]) or adapter_name
        it.model = (_join_unique([getattr(r, "model", "") for r in results])
                    or request_kwargs.get("model", ""))
        efforts = _join_unique([getattr(r, "reasoning_effort", "") for r in results])
        it.reasoning_effort = efforts if results else request_kwargs.get("reasoning_effort", "")
        it.cli_version = _join_unique([getattr(r, "cli_version", "") for r in results])
        it.agent_tools = _tool_counts(results, bool(request_kwargs.get("allow_tools", True)))

    # ── terminal transitions ─────────────────────────────────────────

    def _finish_event(self, run_state: RunState) -> None:
        self._emit("loop_finished", {
            "run_id": run_state.run_id,
            "status": run_state.status.value,
            "stop_reason": run_state.stop_reason,
        })

    def _complete(self, sm: RunStateMachine, run_state: RunState, reason: str) -> RunState:
        run_state.stop_reason = reason
        sm.complete()
        self._finish_event(run_state)
        return run_state

    def _fail(self, sm: RunStateMachine, run_state: RunState, reason: str, message: str) -> RunState:
        run_state.stop_reason = f"failed:{reason}"
        try:
            sm.fail(message)
        except Exception:  # already terminal
            run_state.error = message
        self._finish_event(run_state)
        return run_state

    # ── resume ───────────────────────────────────────────────────────

    def _restore_from_history(self, run_state: RunState) -> None:
        """Rebuild prompt memory, re-seed the convergence detector and restore
        the consensus gate from the stored iterations."""
        last_consensus_report: Optional[dict[str, Any]] = None
        for it in sorted(run_state.iterations, key=lambda i: i.iteration_number):
            n = it.iteration_number
            self._convergence.note_iteration(n)
            if it.status != IterationStatus.COMPLETED:
                continue
            if is_executor_role(it.role):
                full = (self._read_iteration_file(run_state, n, f"{_safe_name(it.role)}_output.md")
                        or it.output_summary or "")
                self._convergence.record_executor_output(full)
                self._executor_submissions += 1
                self._last_executor_output = full
                self._last_executor_iteration = n
                self._last_exec_report = (
                    self._read_iteration_file(run_state, n, "RUN_LOG.md")
                    if it.code_execution else ""
                )
                self._pending_feedback = []
                continue
            report = it.consensus_report if isinstance(it.consensus_report, dict) else None
            if report is not None:
                last_consensus_report = report
            partial = bool(report and report.get("panel_partial"))
            if partial and self._partial_policy != "count":
                pass  # a partial panel's grade was never an observation
            elif it.grade is not None or it.critical_count is not None:
                self._convergence.record_review(it.grade, it.critical_count or 0, it.high_count or 0)
            elif it.role in REVIEW_ROLES:
                # Rows written before grades were stored: re-parse the full output.
                raw = (self._read_iteration_file(run_state, n, f"{_safe_name(it.role)}_output.md")
                       or it.output_summary or "")
                try:
                    review = self._feedback_formatter.parse_review(raw)
                except Exception:
                    review = None
                if review is not None and (review.overall_grade or review.critiques):
                    crit, high = _severity_counts(review)
                    self._convergence.record_review(review.overall_grade, crit, high)
            text = self._restored_feedback(run_state, it)
            if text:
                self._pending_feedback.append((it.role, text))
        # The consensus gate holds while the latest consensus step left
        # unresolved CRITICAL critiques (exactly as in an uninterrupted run).
        gate = _as_bool(self._conf(run_state, "consensus_gate", False))
        self._consensus_block = bool(
            gate and (last_consensus_report or {}).get("unresolved_critical", 0) > 0)

    def _restored_feedback(self, run_state: RunState, it: IterationResult) -> str:
        """The exact feedback text a non-executor step queued for the executor.

        Prefers ``<role>_feedback.md`` (written live); otherwise rebuilds it
        from the full ``<role>_output.md`` with the live rules, falling back
        to the stored feedback / 500-character summary only when no file is
        available."""
        n = it.iteration_number
        name = _safe_name(it.role)
        saved = self._read_iteration_file(run_state, n, f"{name}_feedback.md")
        if saved:
            return saved
        if it.consensus_report is not None or it.role == "consensus_merger":
            return it.feedback or it.output_summary or ""
        raw = self._read_iteration_file(run_state, n, f"{name}_output.md")
        if not raw:
            return it.feedback or it.output_summary or ""
        if it.role in REVIEW_ROLES:
            _review, _formatted, _ok, text = self._review_feedback(it.role, raw)
            return text
        return raw

    def _resume_step_index(self, plan, run_state: RunState, resume_from: int) -> int:
        """Plan position after the stored iterations ``<= resume_from``.

        Replays the compiled plan (with its every_k / on_flag gating) against
        the stored iteration roles, so a resumed run continues with the step
        an uninterrupted run would have run next. Falls back to matching on
        roles only, then to ``resume_from`` (the legacy behaviour), when the
        stored history does not fit the plan (e.g. a changed loop)."""
        its = sorted((it for it in run_state.iterations if it.iteration_number <= resume_from),
                     key=lambda i: i.iteration_number)
        length = plan.cycle_length
        if not its or length == 0:
            return resume_from
        limit = (len(its) + 1) * length + length

        def replay(check_gate: bool) -> Optional[int]:
            s, i = 0, 0
            while i < len(its) and s < limit:
                step = plan.steps[s % length]
                cycle = s // length + 1
                gate_ok = (not check_gate) or self._should_run_step(
                    step, its[i].iteration_number - 1, run_state, cycle)
                if gate_ok and step.role == its[i].role:
                    i += 1
                s += 1
            return s if i == len(its) else None

        pos = replay(True)
        if pos is None:
            pos = replay(False)
        if pos is None:
            logger.warning("Run %s: stored iterations do not match the loop plan; resuming "
                           "at plan step %d", run_state.run_id, resume_from)
            return resume_from
        return pos

    # ── main loop ────────────────────────────────────────────────────

    async def run_loop(
        self,
        run_state: RunState,
        loop_def: LoopDefinition,
        cancel_event: Optional[asyncio.Event] = None,
        workspace_path: str = "",
        resume_from: int = 0,
        workspace_defaults: Optional[dict[str, Any]] = None,
    ) -> RunState:
        """Execute the full loop until completion, budget exceeded, or cancellation.

        ``workspace_defaults`` are lowest-priority config values (the runner
        passes the workspace's per-run budgets as ``budget_max_tokens`` /
        ``budget_max_cost``)."""
        sm = RunStateMachine(
            run_state,
            on_state_change=lambda old, new, st: self._emit(
                "state_change", {"old": old.value, "new": new.value, "run_id": st.run_id}
            ),
        )
        self._reset_run_memory()
        self._workspace_defaults = {
            k: v for k, v in (workspace_defaults or {}).items() if v is not None
        }
        self._consensus_block = False
        self._workspace_path = workspace_path or ""
        self._run_id = run_state.run_id

        plan = compile_loop(loop_def)

        # Handle resume vs fresh start
        if resume_from > 0 and run_state.status in (RunStatus.PAUSED, RunStatus.RUNNING):
            # For resumed runs, transition PAUSED->RUNNING if needed
            if run_state.status == RunStatus.PAUSED:
                try:
                    sm.resume()
                except Exception as exc:
                    return self._fail(sm, run_state, "start", str(exc))
            # Otherwise already RUNNING — no transition needed
        else:
            try:
                sm.start()
            except Exception as exc:
                return self._fail(sm, run_state, "start", str(exc))

        try:
            self._configure(run_state, loop_def)
        except Exception as exc:
            return self._fail(sm, run_state, "invalid_config", f"Invalid run configuration: {exc}")

        # seed_executor_output: a fresh run starts after the first executor step.
        seed_start = 0
        if self._seed_output:
            seed_start = next((i + 1 for i, s in enumerate(plan.steps)
                               if is_executor_role(s.role)), -1)
            if seed_start < 0 and plan.steps:
                return self._fail(sm, run_state, "invalid_config",
                                  "Invalid run configuration: seed_executor_output needs "
                                  "a loop with an executor step")
            seed_start = max(seed_start, 0)

        # Set up workspace context reader
        if workspace_path:
            self._context_reader = WorkspaceContextReader(workspace_path, run_state.run_id)
        else:
            self._context_reader = None

        self._emit("loop_started", {
            "run_id": run_state.run_id,
            "plan_steps": len(plan.steps),
            "loop_source": loop_def.source,
            "stopping_rule": self._convergence.rule.to_dict(),
            "tool_settings": dict(self._tool_settings),
            "consensus_partial_policy": self._partial_policy,
            "seed_executor_output_chars": len(self._seed_output),
            "revision_budget": self._revision_budget,
            "revision_budget_source": self._stopping_settings.get("revision_budget_source"),
            "convergence_enabled": self._stopping_settings.get("convergence_enabled"),
        })
        if self._tool_settings.get("executor_tools_with_code_execution"):
            self._emit("executor_tools_with_code_execution", {
                "run_id": run_state.run_id,
                "warning": "executor_allow_tools=true together with verified code execution: "
                           "the executor agent can run commands outside the CodeExecutor",
            })
        for warning in getattr(plan, "warnings", []) or []:
            logger.warning("Loop plan: %s", warning)
            self._emit("loop_plan_warning", {"run_id": run_state.run_id, "warning": warning})
        if self._stop_unknown:
            logger.warning("Ignoring unknown stop conditions: %s", self._stop_unknown)
            self._emit("stop_conditions_ignored", {
                "run_id": run_state.run_id, "conditions": list(self._stop_unknown),
            })

        step_index = 0

        # Seed: iteration-0 executor output (re-applied before a resume restore).
        if self._seed_output:
            self._last_executor_output = self._seed_output
            self._last_executor_iteration = 0
            self._convergence.record_executor_output(self._seed_output)
            self._executor_submissions = 1   # the seed is the initial submission
            step_index = seed_start

        # Restore state when resuming from a previous iteration
        if resume_from > 0 and run_state.iterations:
            run_state.current_iteration = resume_from
            step_index = self._resume_step_index(plan, run_state, resume_from)
            self._restore_from_history(run_state)

        skipped_in_a_row = 0
        while True:
            # Check cancellation
            if cancel_event and cancel_event.is_set():
                run_state.stop_reason = "cancelled"
                sm.stop()
                self._emit("loop_cancelled", {"run_id": run_state.run_id})
                self._finish_event(run_state)
                return run_state

            # Check budget
            budget, kind = self._evaluate_budget(run_state)
            if budget.exceeded:
                self._emit("budget_exceeded", {
                    "run_id": run_state.run_id, "reason": budget.reason, "stop_reason": kind,
                })
                return self._complete(sm, run_state, kind or "max_iterations")

            flag = self._triggered_stop_flag(run_state)
            if flag:
                reason = f"stop_condition:on_flag:{flag}"
                self._emit("stop_condition_met", {"run_id": run_state.run_id, "stop_reason": reason})
                return self._complete(sm, run_state, reason)

            # Get current step (cycle through plan)
            if plan.cycle_length == 0:
                return self._complete(sm, run_state, "empty_plan")

            step = plan.steps[step_index % plan.cycle_length]
            cycle = step_index // plan.cycle_length + 1

            # Check if this conditional step should run
            if not self._should_run_step(step, run_state.current_iteration, run_state, cycle):
                step_index += 1
                skipped_in_a_row += 1
                if skipped_in_a_row >= plan.cycle_length:
                    # A full cycle without a runnable step: nothing can ever run.
                    return self._complete(sm, run_state, "no_runnable_steps")
                continue
            skipped_in_a_row = 0

            # Revision budget: the initial submission plus ``revision_budget``
            # revisions are done (and every step after the last one has run,
            # so the final artifact has been reviewed) -> stop before the next
            # executor step.
            if (self._revision_budget is not None and is_executor_role(step.role)
                    and self._executor_submissions >= 1 + self._revision_budget):
                reason = REVISION_BUDGET_STOP_REASON
                self._emit("stop_condition_met", {
                    "run_id": run_state.run_id, "stop_reason": reason,
                    "revision_budget": self._revision_budget,
                    "executor_submissions": self._executor_submissions,
                    "executor_revisions": self._executor_submissions - 1,
                })
                return self._complete(sm, run_state, reason)

            # Advance iteration
            iteration_num = sm.next_iteration()
            set_run_context(run_state.run_id, iteration_num)
            self._emit("iteration_started", {
                "run_id": run_state.run_id,
                "iteration": iteration_num,
                "role": step.role,
                "node_id": step.node_id,
            })

            if step.kind == "consensus" and step.panel:
                outcome = await self._run_consensus_iteration(step, run_state, iteration_num)
            else:
                outcome = await self._run_single_iteration(step, run_state, iteration_num)

            if outcome.failed:
                return self._fail(sm, run_state, outcome.reason, outcome.message)

            # A partial panel (no_stop policy) never stops the run.
            stop_eligible = not outcome.partial
            partial_info = {"panel_partial": bool(outcome.failed_lenses),
                            "failed_lenses": list(outcome.failed_lenses or [])}

            # grade_at_least stop condition (after review / consensus steps)
            if (stop_eligible and outcome.review_grade is not None
                    and not self._consensus_block):
                thr = self._conf(run_state, "grade_at_least")
                if thr and grade_at_least(outcome.review_grade, str(thr)):
                    reason = "stop_condition:grade_at_least"
                    self._emit("stop_condition_met", {
                        "run_id": run_state.run_id, "stop_reason": reason,
                        "grade": outcome.review_grade, "threshold": str(thr).upper(),
                        **partial_info,
                    })
                    return self._complete(sm, run_state, reason)

            # Check convergence (if enabled -- off by default -- and not held
            # open by the consensus gate)
            if (stop_eligible
                    and _as_bool(self._conf(run_state, "convergence_enabled",
                                            DEFAULT_CONVERGENCE_ENABLED))
                    and not self._consensus_block):
                decision = self._convergence.evaluate()
                if decision.converged:
                    self._emit("convergence_detected", {
                        "run_id": run_state.run_id,
                        "reason": decision.reason,
                        "stop_reason": decision.stop_reason,
                        "signals": list(decision.signals),
                        "metrics": self._convergence.get_metrics(),
                        **partial_info,
                    })
                    return self._complete(sm, run_state, decision.stop_reason)

            step_index += 1

    # ── single step ──────────────────────────────────────────────────

    async def _run_single_iteration(
        self, step: ExecutionStep, run_state: RunState, iteration_num: int,
    ) -> _StepOutcome:
        role = step.role
        executor = is_executor_role(role)
        prompt_bundle = self._build_prompt_bundle(step, run_state, iteration=iteration_num)
        req_kwargs = self._request_kwargs(run_state, executor=executor)
        try:
            timeout = int(float(self._conf(run_state, "adapter_timeout", 900)))
        except (TypeError, ValueError):
            timeout = 900
        request = AdapterRunRequest(
            prompt_bundle=prompt_bundle,
            workspace_context=self._workspace_context(run_state, iteration_num),
            timeout_seconds=timeout,
            **req_kwargs,
        )

        iter_result = IterationResult(
            iteration_number=iteration_num,
            role=role,
            status=IterationStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        t0 = time.monotonic()

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
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            iter_result.status = IterationStatus.FAILED
            iter_result.error = str(exc)
            iter_result.completed_at = datetime.utcnow()
            iter_result.duration_seconds = round(time.monotonic() - t0, 3)
            await self._record_iteration(run_state, iter_result)
            return _StepOutcome(True, "adapter_error",
                                f"Adapter error at iteration {iteration_num}: {exc}")
        if adapter_result is None:
            adapter_result = AdapterRunResult(success=False, error="adapter returned no result")

        # Clean thinking traces from adapter output before any downstream use
        output = strip_thinking_traces(adapter_result.output) if adapter_result.output else ""
        adapter_result.output = output

        iter_result.status = (
            IterationStatus.COMPLETED if adapter_result.success else IterationStatus.FAILED
        )
        iter_result.completed_at = datetime.utcnow()
        iter_result.output_summary = output[:500]
        iter_result.artifacts_produced = list(adapter_result.artifacts_written)
        iter_result.error = adapter_result.error
        self._account(run_state, iter_result, [adapter_result])
        self._describe_calls(run_state, iter_result, [adapter_result], req_kwargs)

        # Persist the parsed artifacts AND the full cleaned output.
        if output:
            parsed: dict[str, str] = {}
            try:
                parsed = self._artifact_parser.parse_output(output, role=role)
            except Exception:
                logger.debug("Artifact parsing failed for iter %d", iteration_num)
            if self._artifact_writer:
                produced: list[str] = []
                for art_name, art_content in parsed.items():
                    if self._write(iteration_num, art_name, art_content):
                        produced.append(art_name)
                full_name = f"{_safe_name(role)}_output.md"
                if self._write(iteration_num, full_name, output) and full_name not in produced:
                    produced.append(full_name)
                if produced:
                    iter_result.artifacts_produced = produced

        self._emit("iteration_completed", {
            "run_id": run_state.run_id,
            "iteration": iteration_num,
            "role": role,
            "success": adapter_result.success,
            "tokens": iter_result.token_usage,
        })

        if not adapter_result.success:
            iter_result.duration_seconds = round(time.monotonic() - t0, 3)
            await self._record_iteration(run_state, iter_result)
            return _StepOutcome(True, "adapter_failed",
                                f"Adapter failed at iteration {iteration_num}: {adapter_result.error}")

        outcome = _StepOutcome()
        if executor:
            self._executor_submissions += 1
            self._last_executor_output = output
            self._last_executor_iteration = iteration_num
            self._pending_feedback = []
            self._last_exec_report = ""
            if self._code_executor is not None:
                self._last_exec_report = await self._execute_code(
                    run_state, iter_result, iteration_num, output)
            self._convergence.record_executor_output(output, iteration_num)
        elif role in REVIEW_ROLES:
            review, formatted, parsed_ok, feedback_text = self._review_feedback(role, output)
            if parsed_ok:
                assert review is not None
                crit, high = _severity_counts(review)
                iter_result.grade = review.overall_grade or ""
                iter_result.critical_count = crit
                iter_result.high_count = high
                self._convergence.record_review(review.overall_grade, crit, high, iteration_num)
                outcome.review_grade = review.overall_grade or ""
            else:
                # An unparseable review is not an observation (never "0 critiques").
                self._convergence.note_iteration(iteration_num)
            # Never store the empty formatter shell of an unparsed review.
            iter_result.feedback = formatted if (parsed_ok and formatted) else None
            self._queue_feedback(iteration_num, role, feedback_text)
        else:
            self._convergence.note_iteration(iteration_num)
            self._queue_feedback(iteration_num, role, output)
            # After idea_generator step, parse follow-up proposals
            if role == "idea_generator" and output:
                try:
                    proposals = self._followup_executor.parse_proposals(output)
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

        iter_result.duration_seconds = round(time.monotonic() - t0, 3)
        await self._record_iteration(run_state, iter_result)
        return outcome

    def _review_feedback(
        self, role: str, output: str,
    ) -> tuple[Optional[ReviewResult], str, bool, str]:
        """Parse a single reviewer's output; return ``(review, formatted,
        parsed_ok, feedback_text)``. Shared by the live step and resume.

        ``parsed_ok`` is False for an output with no usable critique format
        (including a contract block that cannot count as a review); its
        feedback then carries the full reviewer text instead."""
        review: Optional[ReviewResult] = None
        try:
            review = self._feedback_formatter.parse_review(output)
            formatted = self._feedback_formatter.format_for_executor(review)
        except Exception:
            logger.debug("Feedback formatting failed for %s", role)
            formatted = ""
        parsed_ok = review is not None and (
            bool(review.critiques) or bool(review.overall_grade)
            or review.parse_method not in ("", "none")
        )
        if parsed_ok and formatted:
            text = formatted
        else:
            text = (
                (formatted + "\n\n" if (parsed_ok and formatted) else "")
                + f"(The {role} response could not be parsed into structured "
                  f"critiques; its full text follows.)\n" + output
            )
        return review, formatted, parsed_ok, text

    def _queue_feedback(self, iteration_num: int, role: str, text: str) -> None:
        """Queue feedback for the next executor prompt and persist the exact
        text (``<role>_feedback.md``) so a resumed run rebuilds the same prompt."""
        self._pending_feedback.append((role, text))
        if text:
            self._write(iteration_num, f"{_safe_name(role)}_feedback.md", text)

    async def _execute_code(
        self, run_state: RunState, iter_result: IterationResult, iteration_num: int, output: str,
    ) -> str:
        """Run the executor's fenced python code; return the report text."""
        assert self._code_executor is not None
        report = None
        try:
            report = await self._code_executor.run_artifact(output)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            iter_result.code_execution = {"executed": 0, "passed": 0, "failed": 0, "error": msg}
            text = (
                "=== CODE EXECUTION REPORT ===\n"
                f"Code execution could not be performed: {msg}. No number in this "
                "write-up is backed by executed code.\n"
                "=== END CODE EXECUTION REPORT ==="
            )
            self._write(iteration_num, "RUN_LOG.md", text)
            self._emit("code_executed", {
                "run_id": run_state.run_id, "iteration": iteration_num,
                "executed": 0, "passed": 0, "failed": 0, "error": msg,
            })
            return text

        summary = report.summary()
        iter_result.code_execution = summary
        text = report.to_feedback() if report.executed else _NO_CODE_REPORT
        written = []
        if self._write(iteration_num, "RUN_LOG.md", text):
            written.append("RUN_LOG.md")
        payload = {"iteration": iteration_num, **report.to_dict()}
        if self._write(iteration_num, "CODE_EXECUTION.json",
                       json.dumps(payload, indent=2, default=str)):
            written.append("CODE_EXECUTION.json")
        for name in written:
            if name not in iter_result.artifacts_produced:
                iter_result.artifacts_produced.append(name)
        self._emit("code_executed", {
            "run_id": run_state.run_id,
            "iteration": iteration_num,
            "executed": report.executed,
            "passed": report.passed,
            "failed": report.failed,
        })
        return text

    # ── consensus step ───────────────────────────────────────────────

    def _build_consensus_reviewer(
        self, step: ExecutionStep, run_state: RunState,
    ) -> tuple[ConsensusReviewer, dict[str, Any]]:
        panel = [(pm.role, pm.prompt_ref) for pm in step.panel]
        node_defaults = merger_config_defaults(step.config)
        if "consensus_similarity_threshold" in node_defaults:
            # A threshold is only meaningful for its own similarity method.
            node_method = str(node_defaults.get("consensus_similarity_method") or "jaccard").lower()
            chosen = self._conf(run_state, "consensus_similarity_method")
            if chosen is not None and str(chosen).lower() != node_method:
                node_defaults.pop("consensus_similarity_threshold")
        cfg = self.effective_config(run_state, extra_defaults=node_defaults)
        req_kwargs = self._request_kwargs(run_state, executor=False)
        reviewer = ConsensusReviewer.from_config(
            cfg,
            panel=panel or None,
            request_kwargs=req_kwargs,
            adjudicator=_RequestDefaultsAdapter(self.adapter, req_kwargs),
        )
        return reviewer, req_kwargs

    async def _run_consensus_step(
        self,
        step: ExecutionStep,
        run_state: RunState,
        iteration_num: int,
        artifacts_content: Optional[str] = None,
        workspace_path: Optional[str] = None,
    ) -> tuple[Any, str, list]:
        """Run one consensus panel attempt (no retry, no accounting).

        Returns (merged ReviewResult, formatted feedback string, raw results).
        ``artifacts_content`` defaults to the neutral review prompt.
        """
        reviewer, _ = self._build_consensus_reviewer(step, run_state)
        content = (artifacts_content if artifacts_content is not None
                   else self._build_review_user_prompt(None, run_state, iteration_num))
        merged, raw_results = await reviewer.run_panel(
            content,
            self.adapter,
            self._workspace_context(run_state, iteration_num),
            system_prompt_builder=lambda role, ref: self._resolve_system_prompt(
                role, ref, run_state, iteration_num
            ),
        )
        formatted = self._feedback_formatter.format_for_executor(merged)
        return merged, formatted, raw_results

    async def _run_consensus_iteration(
        self, step: ExecutionStep, run_state: RunState, iteration_num: int,
    ) -> _StepOutcome:
        """Fan out to the reviewer panel, merge, and feed the ranked critiques
        back to the executor.

        A panel with no usable lens is retried once (whole panel); a second
        failure fails the run (``INCOMPLETE`` is never a grade). A partial
        panel is retried once for its failed lenses only (usable lens results
        are kept); if it is still partial, ``consensus_partial_policy``
        applies (see the module docstring)."""
        iter_result = IterationResult(
            iteration_number=iteration_num,
            role=step.role,
            status=IterationStatus.RUNNING,
            started_at=datetime.utcnow(),
        )
        t0 = time.monotonic()
        try:
            reviewer, req_kwargs = self._build_consensus_reviewer(step, run_state)
        except Exception as exc:
            iter_result.status = IterationStatus.FAILED
            iter_result.error = f"Invalid consensus configuration: {exc}"
            iter_result.completed_at = datetime.utcnow()
            await self._record_iteration(run_state, iter_result)
            return _StepOutcome(True, "invalid_config",
                                f"Invalid consensus configuration at iteration {iteration_num}: {exc}")

        content = self._build_review_user_prompt(None, run_state, iteration_num)
        ctx = self._workspace_context(run_state, iteration_num)
        role_names = [pm.role for pm in step.panel]
        attempts: list[dict[str, Any]] = []
        all_results: list[AdapterRunResult] = []   # every fresh call (accounting)
        merged: Optional[ReviewResult] = None
        raw_results: list[AdapterRunResult] = []   # final, aligned with the panel
        report: dict[str, Any] = {}
        last_error = ""
        reuse: Optional[list[Optional[AdapterRunResult]]] = None

        for attempt in range(1, _CONSENSUS_ATTEMPTS + 1):
            try:
                merged, attempt_results = await reviewer.run_panel(
                    content,
                    self.adapter,
                    ctx,
                    system_prompt_builder=lambda role, ref: self._resolve_system_prompt(
                        role, ref, run_state, iteration_num
                    ),
                    reuse=reuse,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                attempts.append({"attempt": attempt, "error": last_error})
                merged = None
                logger.warning("Consensus attempt %d failed at iteration %d: %s",
                               attempt, iteration_num, last_error)
                continue

            attempt_results = list(attempt_results)
            fresh = [r for i, r in enumerate(attempt_results)
                     if reuse is None or i >= len(reuse) or reuse[i] is None]
            raw_results = attempt_results
            calls = list(fresh)
            adj_result = self._adjudicator_result(merged)
            if adj_result is not None:
                calls.append(adj_result)
            all_results.extend(calls)
            self._account(run_state, iter_result, calls)
            report = reviewer.compute_consensus_report(merged)
            lens_status = report.get("lens_status") or {}
            attempts.append({
                "attempt": attempt,
                "panel_failed": report.get("panel_failed", False),
                "panel_partial": report.get("panel_partial", False),
                "failed_lenses": report.get("failed_lenses", []),
                "lens_errors": report.get("lens_errors", {}),
                "lenses_called": [k for k, st in lens_status.items()
                                  if k not in set((merged.consensus_meta or {})
                                                  .get("reused_lenses", []))],
            })
            self._emit("consensus_merged", {
                "run_id": run_state.run_id,
                "iteration": iteration_num,
                "panel_size": len(step.panel),
                "attempt": attempt,
                "report": report,
            })
            panel_failed = bool(report.get("panel_failed", False))
            panel_partial = bool(report.get("panel_partial", False))
            if not panel_failed and not panel_partial:
                break
            will_retry = attempt < _CONSENSUS_ATTEMPTS
            if panel_failed:
                last_error = (
                    "no reviewer lens returned a usable review ("
                    + "; ".join(f"{k}: {v}" for k, v in (report.get("lens_errors") or {}).items())
                    + ")"
                )
                self._emit("consensus_panel_failed", {
                    "run_id": run_state.run_id,
                    "iteration": iteration_num,
                    "attempt": attempt,
                    "will_retry": will_retry,
                    "lens_errors": report.get("lens_errors", {}),
                })
                reuse = None  # retry the whole panel
            else:
                self._emit("consensus_panel_partial", {
                    "run_id": run_state.run_id,
                    "iteration": iteration_num,
                    "attempt": attempt,
                    "will_retry": will_retry,
                    "failed_lenses": report.get("failed_lenses", []),
                    "lens_errors": report.get("lens_errors", {}),
                })
                # Retry only the failed lenses; keep the usable results.
                keys = (merged.consensus_meta or {}).get("lenses") or []
                reuse = [
                    res if (i < len(keys) and lens_status.get(keys[i]) == "ok") else None
                    for i, res in enumerate(attempt_results)
                ]

        self._describe_calls(run_state, iter_result, all_results, req_kwargs)
        self._write_lens_outputs(iteration_num, role_names, raw_results)
        succeeded = merged is not None and not report.get("panel_failed", False)
        still_partial = succeeded and bool(report.get("panel_partial", False))

        if not succeeded or (still_partial and self._partial_policy == "fail"):
            iter_result.status = IterationStatus.FAILED
            if succeeded:
                last_error = (
                    "reviewer lens(es) " + ", ".join(report.get("failed_lenses", []))
                    + " failed after retry (consensus_partial_policy=fail)"
                )
            iter_result.error = (
                f"Consensus panel failed at iteration {iteration_num} after "
                f"{len(attempts)} attempt(s): {last_error}"
            )
            iter_result.completed_at = datetime.utcnow()
            iter_result.duration_seconds = round(time.monotonic() - t0, 3)
            iter_result.consensus_report = {**report, "attempts": attempts} if report else {"attempts": attempts}
            await self._record_iteration(run_state, iter_result)
            if succeeded:
                reason = "consensus_panel_partial"
            else:
                reason = "consensus_panel_failed" if report else "consensus_error"
            return _StepOutcome(True, reason, iter_result.error)

        assert merged is not None
        stop_eligible = not (still_partial and self._partial_policy == "no_stop")
        report = {**report, "grade_valid": not still_partial,
                  "partial_policy": self._partial_policy if still_partial else None,
                  "stop_eligible": stop_eligible}
        formatted = self._feedback_formatter.format_for_executor(merged)
        crit, high = _severity_counts(merged)
        iter_result.status = IterationStatus.COMPLETED
        iter_result.completed_at = datetime.utcnow()
        iter_result.output_summary = formatted[:500]
        iter_result.feedback = formatted
        iter_result.grade = merged.overall_grade
        iter_result.critical_count = crit
        iter_result.high_count = high
        iter_result.consensus_report = {**report, "attempts": attempts}

        produced: list[str] = []
        for name in ("EVAL.md", f"{_safe_name(step.role)}_output.md"):
            if formatted and self._write(iteration_num, name, formatted):
                produced.append(name)
        consensus_payload = {
            "iteration": iteration_num,
            "report": report,
            "attempts": attempts,
            "grade": merged.overall_grade,
            "critiques": [c.model_dump(mode="json") for c in merged.critiques],
            "consensus_meta": merged.consensus_meta,
        }
        if self._write(iteration_num, "CONSENSUS.json",
                       json.dumps(consensus_payload, indent=2, default=str)):
            produced.append("CONSENSUS.json")
        produced.extend(self._lens_file_names(role_names, raw_results))
        iter_result.artifacts_produced = produced if self._artifact_writer else []

        self._queue_feedback(iteration_num, step.role, formatted)

        # Optional advancement gate (deadlock policy): while unresolved
        # CRITICAL critiques remain, the run is not allowed to converge or
        # stop on a grade; it keeps revising (bounded by the budgets).
        gate = _as_bool(self._conf(run_state, "consensus_gate", False))
        self._consensus_block = bool(gate and report.get("unresolved_critical", 0) > 0)

        failed_lenses = list(report.get("failed_lenses", [])) if still_partial else None
        if stop_eligible:
            self._convergence.record_review(merged.overall_grade, crit, high, iteration_num)
        else:
            # A partial panel's grade is not a review observation.
            self._convergence.note_iteration(iteration_num)
        iter_result.duration_seconds = round(time.monotonic() - t0, 3)
        await self._record_iteration(run_state, iter_result)
        return _StepOutcome(review_grade=merged.overall_grade or "",
                            partial=not stop_eligible, failed_lenses=failed_lenses)

    @staticmethod
    def _adjudicator_result(merged: Optional[ReviewResult]) -> Optional[AdapterRunResult]:
        """The LLM adjudicator's call(s) as one AdapterRunResult (usage and
        provenance), so they go through the same accounting as lens calls."""
        if merged is None:
            return None
        adj = ((merged.consensus_meta or {}).get("merge_info") or {}).get("adjudicator") or {}
        if not adj or not adj.get("called"):
            return None
        return AdapterRunResult(
            success=bool(adj.get("success")),
            error=adj.get("error"),
            token_usage=int(adj.get("token_usage", 0) or 0),
            cost_estimate=float(adj.get("cost_estimate", 0.0) or 0.0),
            input_tokens=int(adj.get("input_tokens", 0) or 0),
            output_tokens=int(adj.get("output_tokens", 0) or 0),
            cached_input_tokens=int(adj.get("cached_input_tokens", 0) or 0),
            provider=str(adj.get("provider", "") or ""),
            model=str(adj.get("model", "") or ""),
            reasoning_effort=str(adj.get("reasoning_effort", "") or ""),
            cli_version=str(adj.get("cli_version", "") or ""),
        )

    def _lens_file_names(self, role_names: list[str], results: list[AdapterRunResult]) -> list[str]:
        if not self._artifact_writer:
            return []
        totals: dict[str, int] = {}
        for r in role_names:
            totals[r] = totals.get(r, 0) + 1
        seen: dict[str, int] = {}
        names: list[str] = []
        for i, res in enumerate(results):
            role = role_names[i] if i < len(role_names) else f"lens{i + 1}"
            seen[role] = seen.get(role, 0) + 1
            base = _safe_name(role)
            names.append(f"{base}_output.md" if totals.get(role, 1) == 1
                         else f"{base}_{seen[role]}_output.md")
        return names

    def _write_lens_outputs(
        self, iteration_num: int, role_names: list[str], results: list[AdapterRunResult],
    ) -> None:
        """Persist each lens's full cleaned output (or its error) for audit."""
        if not self._artifact_writer:
            return
        for name, res in zip(self._lens_file_names(role_names, results), results):
            out = strip_thinking_traces(res.output) if res.output else ""
            if not getattr(res, "success", False):
                out = f"[lens call failed: {res.error}]\n\n{out}".rstrip() + "\n"
            self._write(iteration_num, name, out)
