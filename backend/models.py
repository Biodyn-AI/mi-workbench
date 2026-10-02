"""
MI-Workbench shared data models.
Pydantic models for API, database, and internal communication.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Enums ──────────────────────────────────────────────────────────────

class RunStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"
    FAILED = "failed"


class IterationStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class GitMode(str, enum.Enum):
    NONE = "none"
    COMMIT_PER_RUN = "commit_per_run"
    COMMIT_PER_ITERATION = "commit_per_iteration"
    BRANCH_PER_RUN = "branch_per_run"


class FollowUpPolicy(str, enum.Enum):
    AUTO_RUN_TOP_1 = "auto_run_top_1"
    AUTO_RUN_TOP_N = "auto_run_top_n"
    ASK_APPROVAL = "ask_approval"


class ProviderName(str, enum.Enum):
    CLAUDE_CODE = "claude_code"
    CODEX_CLI = "codex_cli"
    GEMINI_CLI = "gemini_cli"
    MOCK = "mock"


class SeverityLevel(str, enum.Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# ── Workspace ──────────────────────────────────────────────────────────

class WorkspaceConfig(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str
    path: str
    providers_enabled: list[ProviderName] = [ProviderName.CLAUDE_CODE, ProviderName.MOCK]
    default_provider: ProviderName = ProviderName.MOCK
    default_model: str = ""
    git_mode: GitMode = GitMode.NONE
    budget_max_tokens_per_run: int = 500_000
    budget_max_cost_per_run: float = 10.0
    budget_max_iterations: int = 50
    safety_deny_destructive_commands: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class WorkspaceCreate(BaseModel):
    name: str
    path: str
    providers_enabled: list[ProviderName] = [ProviderName.CLAUDE_CODE, ProviderName.MOCK]
    default_provider: ProviderName = ProviderName.MOCK
    git_mode: GitMode = GitMode.NONE


class WorkspaceSummary(BaseModel):
    id: str
    name: str
    path: str
    default_provider: ProviderName
    git_mode: GitMode
    created_at: datetime


# ── Prompts ────────────────────────────────────────────────────────────

class PromptVariable(BaseModel):
    name: str
    description: str
    required: bool = True
    default: Optional[str] = None


class PromptOutputField(BaseModel):
    name: str
    description: str
    required: bool = True


class PromptMetadata(BaseModel):
    role: str
    name: str
    version: str = "1.0.0"
    description: str = ""
    author: str = "system"
    changelog: list[str] = []
    tags: list[str] = []


class PromptTemplate(BaseModel):
    metadata: PromptMetadata
    system_prompt: str
    developer_prompt: str = ""
    variables: list[PromptVariable] = []
    output_schema: list[PromptOutputField] = []
    required_artifacts: list[str] = []
    stop_conditions: list[str] = []


class PromptUpdate(BaseModel):
    system_prompt: Optional[str] = None
    developer_prompt: Optional[str] = None
    variables: Optional[list[PromptVariable]] = None
    output_schema: Optional[list[PromptOutputField]] = None
    changelog_entry: Optional[str] = None


# ── Runs & Iterations ─────────────────────────────────────────────────

class RunCreate(BaseModel):
    workspace_id: str
    loop_preset: str = "executor_reviewer"
    task: str = ""
    task_file: Optional[str] = None
    provider: ProviderName = ProviderName.MOCK
    model: str = ""
    max_iterations: int = 50
    config_overrides: dict[str, Any] = {}
    # Reasoning-effort setting forwarded to every adapter call ("" = CLI default).
    # Stored as run config key ``reasoning_effort`` (a value in config_overrides wins).
    reasoning_effort: str = ""


class IterationResult(BaseModel):
    iteration_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    iteration_number: int
    role: str
    status: IterationStatus = IterationStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    artifacts_produced: list[str] = []
    logs: list[str] = []
    token_usage: int = 0
    cost_estimate: float = 0.0
    output_summary: str = ""
    feedback: Optional[str] = None
    error: Optional[str] = None
    code_execution: Optional[dict[str, Any]] = None
    # Per-call accounting (consensus steps: summed tokens over all lens calls,
    # distinct provider/model/cli_version values joined with ",").
    provider: str = ""
    model: str = ""
    reasoning_effort: str = ""
    cli_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    duration_seconds: float = 0.0
    # Review steps (single reviewer lens or consensus): parsed grade and severity
    # counts, used to re-seed the convergence detector on resume. None = not a review.
    grade: Optional[str] = None
    critical_count: Optional[int] = None
    high_count: Optional[int] = None
    # Consensus steps: ConsensusReviewer.compute_consensus_report() plus attempts.
    consensus_report: Optional[dict[str, Any]] = None
    # Agent-side tool provenance of the step's adapter calls: whether tools were
    # enabled, the sandbox, and counts of tool / shell-command / web-search / MCP
    # items the CLI reported (codex JSONL). None = not recorded.
    agent_tools: Optional[dict[str, Any]] = None


class RunState(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    workspace_id: str
    loop_preset: str
    task: str
    provider: ProviderName
    model: str
    status: RunStatus = RunStatus.PENDING
    current_iteration: int = 0
    max_iterations: int = 50
    iterations: list[IterationResult] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    total_tokens: int = 0
    total_cost: float = 0.0
    config: dict[str, Any] = {}
    error: Optional[str] = None
    # Why the loop ended: "converged:<signal>[+<signal>]", "max_iterations",
    # "budget_tokens", "budget_cost", "revision_budget" (run config
    # ``revision_budget``: the initial executor submission plus that many
    # revisions were made and reviewed), "cancelled", "failed:<reason>",
    # "stop_condition:<name>" (None while running / never started).
    stop_reason: Optional[str] = None
    # Token split (total_tokens == total_input_tokens + total_output_tokens for
    # provider-reported usage; cached input is a subset of input).
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_input_tokens: int = 0


class RunSummary(BaseModel):
    run_id: str
    workspace_id: str
    loop_preset: str
    task: str
    provider: ProviderName
    status: RunStatus
    current_iteration: int
    total_tokens: int
    total_cost: float
    created_at: datetime
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None


# ── Loop DSL ───────────────────────────────────────────────────────────

class LoopNode(BaseModel):
    id: str
    role: str
    prompt_ref: str = ""  # e.g. "executor/mi_executor"
    config: dict[str, Any] = {}


class LoopEdge(BaseModel):
    source: str
    target: str
    condition: str = "always"  # "always", "every_k:3", "on_flag:high_risk", etc.


class LoopDefinition(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0.0"
    nodes: list[LoopNode] = []
    edges: list[LoopEdge] = []
    # Evaluated by the engine (see backend.orchestrator.dsl.parse_stop_conditions):
    # "max_iterations:N", "budget_max_tokens:N", "budget_max_cost:X",
    # "grade_at_least:B", "on_flag:<flag>", "<convergence_key>:<value>";
    # "user_stop" / "budget_exceeded" are always active. Run config overrides.
    stop_conditions: list[str] = ["user_stop", "budget_exceeded"]
    max_iterations: int = 50
    config: dict[str, Any] = {}
    # Where the definition came from ("python" preset, a YAML path, or "").
    source: str = ""


# ── Provider / Adapter ─────────────────────────────────────────────────

class PromptBundle(BaseModel):
    system_prompt: str
    developer_prompt: str = ""
    user_prompt: str
    variables: dict[str, str] = {}


class WorkspaceContext(BaseModel):
    workspace_path: str
    artifact_paths: list[str] = []
    run_dir: str = ""
    iteration_dir: str = ""


class AdapterRunRequest(BaseModel):
    prompt_bundle: PromptBundle
    workspace_context: WorkspaceContext
    files_to_read: list[str] = []
    files_to_write: list[str] = []
    output_schema: dict[str, Any] = {}
    timeout_seconds: int = 300
    # E4: model / reasoning-effort selection forwarded to the CLI ("" = CLI default;
    # an adapter-level default set at construction is used when these are empty).
    model: str = ""
    reasoning_effort: str = ""
    # False disables agent tools (reviewer calls; executor calls when verified
    # execution is on): claude `--tools ""`; codex `--sandbox read-only` plus
    # web search, shell / unified-exec, apps, plugins and other tool features
    # disabled and the user config.toml ignored; gemini `--approval-mode
    # default` plus a system-settings file excluding every built-in tool.
    allow_tools: bool = True
    # Provider-specific options from run config ``adapter_options`` (e.g.
    # ``codex_ignore_user_config``, ``claude_max_turns``); unknown keys are ignored.
    adapter_options: dict[str, Any] = {}


class AdapterRunResult(BaseModel):
    success: bool
    output: str = ""
    artifacts_written: list[str] = []
    structured_output: dict[str, Any] = {}
    token_usage: int = 0
    cost_estimate: float = 0.0
    duration_seconds: float = 0.0
    exit_code: int = 0
    error: Optional[str] = None
    raw_log: str = ""
    # E4 accounting. token_usage == input_tokens + output_tokens for CLI adapters;
    # input_tokens includes cached input, cached_input_tokens is that cached subset;
    # raw_usage keeps the provider-native usage payload.
    provider: str = ""
    model: str = ""  # model that answered if discoverable, else the requested one
    reasoning_effort: str = ""
    cli_version: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    raw_usage: dict[str, Any] = {}


# ── Artifacts ──────────────────────────────────────────────────────────

class ArtifactInfo(BaseModel):
    name: str
    path: str
    artifact_type: str  # MECH, EVAL, XP, METHOD, etc.
    iteration: Optional[int] = None
    run_id: Optional[str] = None
    size_bytes: int = 0
    modified_at: Optional[datetime] = None


class RunMeta(BaseModel):
    run_id: str
    workspace_id: str
    provider: ProviderName
    model: str
    loop_preset: str
    task: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_iterations: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    status: RunStatus = RunStatus.PENDING
    prompts_used: dict[str, str] = {}
    # Revision-2 provenance (all optional / additive).
    stop_reason: Optional[str] = None
    error: Optional[str] = None
    reasoning_effort: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_input_tokens: int = 0
    cli_versions: list[str] = []
    models_used: list[str] = []
    loop_source: str = ""
    stop_conditions: list[str] = []
    # Run-config settings that affect results, plus (runner) the effective
    # agent-tool settings (``effective_tool_settings``) and the effective
    # stopping policy (``revision_budget``, ``effective_stopping``).
    config: dict[str, Any] = {}
    convergence: dict[str, Any] = {}
    # One entry per iteration: number, role, status, provider, model,
    # reasoning_effort, cli_version, input/output/cached tokens, cost, duration,
    # grade, code-execution summary.
    iterations: list[dict[str, Any]] = []
    # Iteration writes deferred because the database stayed locked after its
    # retries (each deferred iteration was persisted later); empty normally.
    persistence_warnings: list[dict[str, Any]] = []


# ── Knowledge Extractor ───────────────────────────────────────────────

class ClaimNode(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    claim: str
    evidence_pointers: list[str] = []
    uncertainty: float = 0.5
    falsification_tests: list[str] = []
    source_artifact: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


class KnowledgeBaseUpdate(BaseModel):
    claims: list[ClaimNode] = []
    facts_jsonl_path: Optional[str] = None
    knowledge_md_path: Optional[str] = None
    summary: str = ""


# ── Subsystem ──────────────────────────────────────────────────────────

class SubsystemManifest(BaseModel):
    name: str
    version: str = "1.0.0"
    description: str = ""
    roles: list[str] = []
    artifact_types: list[str] = []
    loop_presets: list[str] = []
    validation_rules: list[str] = []
    ui_panels: list[str] = []


# ── Safety / Budget ───────────────────────────────────────────────────

class BudgetStatus(BaseModel):
    tokens_used: int = 0
    tokens_limit: int = 500_000
    cost_used: float = 0.0
    cost_limit: float = 10.0
    iterations_used: int = 0
    iterations_limit: int = 50
    exceeded: bool = False
    reason: Optional[str] = None


class SafetyGate(BaseModel):
    name: str
    triggered: bool = False
    message: str = ""
    timestamp: Optional[datetime] = None


# ── Review / Critique ─────────────────────────────────────────────────

class ReviewCritique(BaseModel):
    severity: SeverityLevel
    category: str
    description: str
    required_fix: Optional[str] = None
    suggested_experiment: Optional[str] = None
    artifact_ref: Optional[str] = None  # e.g., "PROTOCOL.md", "MECH.md"
    section_ref: Optional[str] = None   # e.g., "Section 2.1 Controls"
    # Consensus only: distinct reviewer lenses that raised this (merged) critique.
    raised_by: list[str] = []
    # Consensus only: the other critiques merged into this one (each a dict with
    # lens, severity, description, required_fix), so no member's text is lost
    # when the longest description represents the group.
    merged_members: list[dict[str, Any]] = []


class ReviewResult(BaseModel):
    overall_grade: str = ""  # A/B/C/D/F or pass/fail; "INCOMPLETE" if the panel failed
    critiques: list[ReviewCritique] = []
    reproducibility_gaps: list[str] = []
    suspected_confounders: list[str] = []
    claim_validity: dict[str, str] = {}
    # One-sentence verdict from the JSON critique contract (if provided).
    overall_assessment: str = ""
    # How critiques were extracted: "json" | "yaml" | "regex" | "none" ("" = not parsed).
    parse_method: str = ""
    # JSON decode detail ("strict" | "repaired" | "truncated" | "salvaged") or
    # "invalid_contract:<reason>" when a contract block could not count as a review.
    parse_detail: str = ""
    # Consensus only: per-lens status, parse methods, merge groups, similarity
    # settings, panel failure flags (see ConsensusReviewer.compute_consensus_report).
    consensus_meta: dict[str, Any] = {}


# ── Follow-up ─────────────────────────────────────────────────────────

class CompareRequest(BaseModel):
    run_ids: list[str]


class FollowUpProposal(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    title: str
    description: str
    expected_value: str = ""
    effort: str = ""  # low/medium/high
    dependencies: list[str] = []
    priority: int = 0
    task_definition: str = ""
