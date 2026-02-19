// ── Type Definitions ──────────────────────────────────────────────────

export type RunStatus =
  | "pending"
  | "running"
  | "paused"
  | "stopped"
  | "completed"
  | "failed";

export type IterationStatus = "pending" | "running" | "completed" | "failed";

export type GitMode =
  | "none"
  | "commit_per_run"
  | "commit_per_iteration"
  | "branch_per_run";

export type ProviderName = "claude_code" | "codex_cli" | "gemini_cli" | "mock";

export type SeverityLevel = "info" | "low" | "medium" | "high" | "critical";

export interface WorkspaceSummary {
  id: string;
  name: string;
  path: string;
  default_provider: ProviderName;
  git_mode: GitMode;
  created_at: string;
}

export interface WorkspaceConfig {
  id: string;
  name: string;
  path: string;
  providers_enabled: ProviderName[];
  default_provider: ProviderName;
  default_model: string;
  git_mode: GitMode;
  budget_max_tokens_per_run: number;
  budget_max_cost_per_run: number;
  budget_max_iterations: number;
  safety_deny_destructive_commands: boolean;
  created_at: string;
}

export interface WorkspaceCreate {
  name: string;
  path: string;
  providers_enabled?: ProviderName[];
  default_provider?: ProviderName;
  git_mode?: GitMode;
}

export interface RunSummary {
  run_id: string;
  workspace_id: string;
  loop_preset: string;
  task: string;
  provider: ProviderName;
  status: RunStatus;
  current_iteration: number;
  total_tokens: number;
  total_cost: number;
  created_at: string;
  started_at: string | null;
  stopped_at: string | null;
}

export interface IterationResult {
  iteration_id: string;
  iteration_number: number;
  role: string;
  status: IterationStatus;
  started_at: string | null;
  completed_at: string | null;
  artifacts_produced: string[];
  logs: string[];
  token_usage: number;
  cost_estimate: number;
  output_summary: string;
  feedback: string | null;
  error: string | null;
}

export interface RunState {
  run_id: string;
  workspace_id: string;
  loop_preset: string;
  task: string;
  provider: ProviderName;
  model: string;
  status: RunStatus;
  current_iteration: number;
  max_iterations: number;
  iterations: IterationResult[];
  created_at: string;
  started_at: string | null;
  stopped_at: string | null;
  total_tokens: number;
  total_cost: number;
  config: Record<string, unknown>;
  error: string | null;
}

export interface RunCreate {
  workspace_id: string;
  loop_preset?: string;
  task: string;
  task_file?: string;
  provider?: ProviderName;
  model?: string;
  max_iterations?: number;
  config_overrides?: Record<string, unknown>;
}

export interface PromptVariable {
  name: string;
  description: string;
  required: boolean;
  default: string | null;
}

export interface PromptOutputField {
  name: string;
  description: string;
  required: boolean;
}

export interface PromptMetadata {
  role: string;
  name: string;
  version: string;
  description: string;
  author: string;
  changelog: string[];
  tags: string[];
}

export interface PromptTemplate {
  metadata: PromptMetadata;
  system_prompt: string;
  developer_prompt: string;
  variables: PromptVariable[];
  output_schema: PromptOutputField[];
  required_artifacts: string[];
  stop_conditions: string[];
}

export interface PromptUpdate {
  system_prompt?: string;
  developer_prompt?: string;
  variables?: PromptVariable[];
  output_schema?: PromptOutputField[];
  changelog_entry?: string;
}

export interface LoopNode {
  id: string;
  role: string;
  prompt_ref: string;
  config: Record<string, unknown>;
}

export interface LoopEdge {
  source: string;
  target: string;
  condition: string;
}

export interface LoopDefinition {
  name: string;
  description: string;
  version: string;
  nodes: LoopNode[];
  edges: LoopEdge[];
  stop_conditions: string[];
  max_iterations: number;
  config: Record<string, unknown>;
}

export interface ProviderInfo {
  name: ProviderName;
  available: boolean;
  default_model: string;
  models: string[];
}

export interface ArtifactInfo {
  name: string;
  path: string;
  artifact_type: string;
  iteration: number | null;
  run_id: string | null;
  size_bytes: number;
  modified_at: string | null;
}

export interface SettingsData {
  workspaces: WorkspaceSummary[];
  active_workspace_id: string | null;
}

export interface SmokeTestResult {
  success: boolean;
  message: string;
  duration_ms: number;
}

export interface PromptValidationResult {
  valid: boolean;
  errors: string[];
  warnings: string[];
}

// ── API Client ───────────────────────────────────────────────────────

const BASE = "/api";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new ApiError(res.status, body || res.statusText);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

// ── Workspaces ───────────────────────────────────────────────────────

export async function listWorkspaces(): Promise<WorkspaceSummary[]> {
  return request("/workspaces");
}

export async function createWorkspace(
  data: WorkspaceCreate,
): Promise<WorkspaceConfig> {
  return request("/workspaces", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getWorkspace(id: string): Promise<WorkspaceConfig> {
  return request(`/workspaces/${id}`);
}

export async function deleteWorkspace(id: string): Promise<void> {
  return request(`/workspaces/${id}`, { method: "DELETE" });
}

// ── Runs ─────────────────────────────────────────────────────────────

export async function listRuns(
  workspaceId?: string,
  status?: RunStatus,
): Promise<RunSummary[]> {
  const params = new URLSearchParams();
  if (workspaceId) params.set("workspace_id", workspaceId);
  if (status) params.set("status", status);
  const qs = params.toString();
  return request(`/runs${qs ? `?${qs}` : ""}`);
}

export async function createRun(data: RunCreate): Promise<RunState> {
  return request("/runs", { method: "POST", body: JSON.stringify(data) });
}

export async function getRun(runId: string): Promise<RunState> {
  return request(`/runs/${runId}`);
}

export async function stopRun(runId: string): Promise<RunState> {
  return request(`/runs/${runId}/stop`, { method: "POST" });
}

export async function resumeRun(runId: string): Promise<RunState> {
  return request(`/runs/${runId}/resume`, { method: "POST" });
}

// ── Prompts ──────────────────────────────────────────────────────────

export async function listPrompts(): Promise<PromptTemplate[]> {
  return request("/prompts");
}

export async function getPrompt(role: string): Promise<PromptTemplate> {
  return request(`/prompts/${role}`);
}

export async function updatePrompt(
  role: string,
  data: PromptUpdate,
): Promise<PromptTemplate> {
  return request(`/prompts/${role}`, {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

export async function validatePrompt(
  role: string,
): Promise<PromptValidationResult> {
  return request(`/prompts/${role}/validate`, { method: "POST" });
}

// ── Loops ────────────────────────────────────────────────────────────

export async function listLoops(): Promise<LoopDefinition[]> {
  return request("/loops");
}

export async function getLoop(name: string): Promise<LoopDefinition> {
  return request(`/loops/${name}`);
}

export async function saveLoop(data: LoopDefinition): Promise<LoopDefinition> {
  return request("/loops", { method: "POST", body: JSON.stringify(data) });
}

// ── Providers ────────────────────────────────────────────────────────

export async function listProviders(): Promise<ProviderInfo[]> {
  return request("/providers");
}

export async function smoketestProvider(
  name: ProviderName,
): Promise<SmokeTestResult> {
  return request(`/providers/${name}/smoketest`, { method: "POST" });
}

// ── Artifacts ────────────────────────────────────────────────────────

export async function browseArtifacts(
  workspaceId: string,
  runId?: string,
  artifactType?: string,
): Promise<ArtifactInfo[]> {
  const params = new URLSearchParams({ workspace_id: workspaceId });
  if (runId) params.set("run_id", runId);
  if (artifactType) params.set("artifact_type", artifactType);
  return request(`/artifacts?${params}`);
}

export async function getArtifact(path: string): Promise<string> {
  const res = await fetch(`${BASE}/artifacts/content?path=${encodeURIComponent(path)}`);
  if (!res.ok) throw new ApiError(res.status, await res.text());
  return res.text();
}

// ── Settings ─────────────────────────────────────────────────────────

export async function getSettings(): Promise<SettingsData> {
  return request("/settings");
}

export async function updateSettings(
  data: Partial<WorkspaceConfig>,
): Promise<WorkspaceConfig> {
  return request("/settings", {
    method: "PUT",
    body: JSON.stringify(data),
  });
}

// ── Knowledge ──────────────────────────────────────────────────────

export interface KnowledgeClaim {
  claim_id: string;
  run_id: string | null;
  claim_text: string;
  evidence_pointers: string[];
  uncertainty: number;
  strength: number;
  falsification_tests: string[];
  source_artifact: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface KnowledgeFact {
  fact_id: string;
  claim_id: string | null;
  fact_text: string;
  source: string;
  confidence: number;
  created_at: string;
}

export interface KnowledgeLink {
  link_id: string;
  source_claim_id: string;
  target_claim_id: string;
  link_type: string;
  weight: number;
}

export interface KnowledgeSummary {
  total_claims: number;
  total_facts: number;
  total_links: number;
  avg_uncertainty: number;
  avg_strength: number;
}

export interface KnowledgeGraph {
  nodes: Array<{
    id: string;
    data: { label: string; uncertainty: number; strength: number };
    position: { x: number; y: number };
  }>;
  edges: Array<{
    id: string;
    source: string;
    target: string;
    label: string;
  }>;
}

export async function getKnowledgeClaims(runId?: string): Promise<KnowledgeClaim[]> {
  const params = runId ? `?run_id=${runId}` : "";
  return request(`/knowledge/claims${params}`);
}

export async function getKnowledgeClaim(claimId: string): Promise<KnowledgeClaim> {
  return request(`/knowledge/claims/${claimId}`);
}

export async function getKnowledgeFacts(claimId?: string): Promise<KnowledgeFact[]> {
  const params = claimId ? `?claim_id=${claimId}` : "";
  return request(`/knowledge/facts${params}`);
}

export async function getKnowledgeLinks(claimId?: string): Promise<KnowledgeLink[]> {
  const params = claimId ? `?claim_id=${claimId}` : "";
  return request(`/knowledge/links${params}`);
}

export async function getKnowledgeSummary(): Promise<KnowledgeSummary> {
  return request("/knowledge/summary");
}

export async function getKnowledgeGraph(): Promise<KnowledgeGraph> {
  return request("/knowledge/graph");
}

// ── Comparison ─────────────────────────────────────────────────────

export interface CompareResult {
  metrics?: {
    runs: Array<{
      run_id: string;
      iterations: number;
      tokens: number;
      cost: number;
      duration_s: number;
      status: string;
    }>;
    best_by_cost?: string;
    best_by_tokens?: string;
    best_by_iterations?: string;
  };
  outputs?: {
    artifact_comparisons: Record<string, {
      pairwise_similarity: number[][];
      run_ids: string[];
    }>;
  };
  quality?: {
    runs: Array<{
      run_id: string;
      final_grade: string;
      critical_count: number;
      high_count: number;
      total_critiques: number;
    }>;
    best_by_grade?: string;
  };
}

export async function compareRuns(runIds: string[], mode: "metrics" | "outputs" | "quality" | "summary"): Promise<CompareResult> {
  return request(`/comparison/${mode}`, {
    method: "POST",
    body: JSON.stringify({ run_ids: runIds }),
  });
}
