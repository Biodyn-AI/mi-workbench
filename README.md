# MI-Workbench

An autonomous multi-agent platform for mechanistic interpretability research on biological foundation models.

MI-Workbench orchestrates iterative research loops in which LLM agents execute analyses, review methodology, red-team for confounders, and refine findings without human intervention.

## Key Features

- **Graph-based workflow DSL**: multi-agent research loops defined in YAML with conditional edges (`always`, `every_k:N`, `on_flag:F`), reviewer panels, and evaluated stop conditions (`max_iterations`, token / cost budgets, `revision_budget`, `grade_at_least`, convergence overrides).
- **Consensus review**: independent reviewer lenses (rigour reviewer, adversarial reviewer, biological plausibility checker) run concurrently. Their critiques follow one machine-readable JSON contract and are grouped by a pluggable similarity method (`llm` by default: LLM adjudication, the best method on the planted-flaw benchmark, with a fallback to `tfidf` when the adjudication fails; or `jaccard`, `tfidf`, `embedding` with order-independent average-linkage clustering at calibrated thresholds). Severity is escalated when distinct lenses agree; merging is cross-lens only by default. A panel without a usable lens is retried once and then fails the run; it is never counted as a clean review. A partial panel (some lenses failed, timed out or returned output that cannot count as a review, e.g. a truncated or echoed-template JSON block) is retried once for the failed lenses; if it is still partial its grade is kept for the record but never stops the run (`consensus_partial_policy`).
- **Calibrated, configurable stopping**: by default a run stops after a fixed budget of four executor revisions, each reviewed (`revision_budget`), the policy selected by the stopping calibration on real executor–panel trajectories. An adaptive convergence detector (opt-in, `convergence_enabled`) has configurable signals (grade stability, zero CRITICAL/HIGH critiques, similarity of consecutive executor outputs), combination rule (any / all / k-of-n), window and gates. Every run records why it stopped (`stop_reason`).
- **Verified execution (optional)**: the executor's fenced Python code is run in a resource-limited subprocess, optionally confined by macOS `sandbox-exec` or Docker. The execution report goes to the reviewers and back to the executor.
- **Provider-agnostic adapters**: Claude Code, OpenAI Codex CLI, Google Gemini CLI, or a mock adapter, through a common `BaseAdapter` interface. Each call records the model, reasoning effort, CLI version and token usage.
- **Structured feedback**: reviewer critiques are mapped to specific artifacts and sections, so the executor makes targeted revisions instead of rewriting.
- **Knowledge extraction**: structured claims are mined from artifacts with uncertainty estimates and falsification tests, and assembled into a claim graph.
- **Reproducibility packaging**: artifacts, every iteration directory, persisted code-execution outputs, prompts, environment metadata and git state are bundled into portable ZIP archives.
- **Real-time monitoring**: run events stream to clients over WebSocket.

## Architecture

```
Interface Layer     CLI (Click)  |  REST API (FastAPI)  |  WebSocket
                         │
Orchestration Layer      Runner ── Loop Engine
                                    ├── State Machine
                                    ├── Convergence Detector (stopping rules)
                                    ├── Consensus Reviewer + similarity methods
                                    ├── Feedback Formatter
                                    ├── Code Executor (verified execution)
                                    └── Follow-Up Executor
                         │
Adapter Layer       Claude Code  |  Codex CLI  |  Gemini CLI  |  Mock
                         │
Persistence Layer   SQLite (aiosqlite)  |  Artifact Writer  |  Knowledge Graph
```

## Quick Start

### Install

```bash
pip install -r requirements.txt            # backend + CLI
pip install -r requirements-dev.txt        # + test tools
pip install -r requirements-analysis.txt   # + numpy/scipy/... for analysis scripts
```

The backend itself is pure Python. Optional dependencies (for example `sentence-transformers` for the `embedding` similarity method) are imported only when used.

### Start the server

```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

### Run via CLI

```bash
# List available providers
python cli/miw.py providers list

# Create a workspace
python cli/miw.py workspace add ./my-workspace

# Start a run with the mock adapter
python cli/miw.py run preset executor_reviewer \
  --workspace-id <workspace_id> \
  --task "Analyze attention patterns in Geneformer for GRN inference" \
  --provider mock --max-iterations 5

# A consensus loop on Codex with a fixed model / effort and run-config overrides
# (default stopping: 4 reviewed revisions; here the adaptive rule is enabled too
# and whichever fires first stops the run)
python cli/miw.py run preset reviewer_consensus \
  --workspace-id <workspace_id> \
  --task "Investigate whether attention heads encode causal regulation" \
  --provider codex_cli --model gpt-5.5 --effort medium --max-iterations 10 \
  -c consensus_similarity_method=tfidf -c convergence_enabled=true -c convergence_rule=all \
  -c budget_max_tokens=3000000

# A custom loop YAML (sent to the server as custom:<absolute path>)
python cli/miw.py run loop ./my_loop.yaml --workspace-id <workspace_id> --task "..." \
  --provider mock --max-iterations 8
```

`-c/--config KEY=VALUE` is repeatable; values are decoded as JSON when possible (`3`, `0.5`, `true`, `["a","b"]`) and kept as strings otherwise.

### Run via API

```bash
# Health check
curl http://127.0.0.1:8000/api/health

# Create workspace
curl -X POST http://127.0.0.1:8000/api/workspaces \
  -H "Content-Type: application/json" \
  -d '{"name": "my-project", "path": "/abs/path/workspaces/my-project"}'

# Start a run
curl -X POST http://127.0.0.1:8000/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "<id>",
    "loop_preset": "reviewer_consensus",
    "task": "Analyze attention patterns in Geneformer",
    "provider": "codex_cli",
    "model": "gpt-5.5",
    "reasoning_effort": "medium",
    "max_iterations": 10,
    "config_overrides": {"revision_budget": 3, "code_execution_enabled": false}
  }'
```

An unknown preset, or a `custom:` path that does not resolve to a valid loop file, is rejected with HTTP 422 when the run is created.

## Loop Presets

| Preset | Source | Description |
|--------|--------|-------------|
| `executor_reviewer` | Python (`presets.py`) | Executor → reviewer loop; the adversarial reviewer runs in every 3rd executor–reviewer cycle (`every_k:3`: cycles 3, 6, 9, …) |
| `reviewer_consensus` | Python (`presets.py`) | Executor → three-lens reviewer panel (rigour, adversarial, bio-plausibility) → consensus merger |
| `research_followups` | Python (`presets.py`) | Executor → idea generator (follow-up proposals) |
| `example_custom` | YAML (`loops/example_custom.yaml`) | Conditional escalation to the adversarial reviewer and knowledge extraction (`on_flag` edges), stop conditions |

`get_preset(name)` resolves, in order:

1. `custom:<path>`: a loop YAML file. The path may be absolute, relative to the working directory, or relative to a loops directory, and must be an existing `.yaml`/`.yml` file. `miw run loop` sends this form.
2. A built-in Python preset (the first three above).
3. `<name>.yaml` (or `.yml`) in `MIW_LOOPS_DIR`, then in the repository's bundled `loops/` directory.

`loops/` also contains YAML *variants* of the three Python presets. They run through `custom:<path>` (the names `executor_reviewer`, `reviewer_consensus` and `research_followups` resolve to the Python presets first) and differ from them: `loops/executor_reviewer.yaml` and `loops/reviewer_consensus.yaml` add `grade_at_least:A` and `max_iterations:50`, and the former names its escalation role `adversarial_reviewer`; `loops/research_followups.yaml` adds a reviewer and runs the idea generator only when the static flag `milestone_complete` is set (the Python preset has no reviewer and proposes follow-ups every cycle). All loops are validated before use: node references, condition syntax and stop-condition values.

How a loop graph becomes a plan: the walk follows `always` edges from the first node and the cycle wraps to the first node when a node has no `always` successor. A conditional edge to a node not yet in the cycle becomes a gated step right after its source; a conditional edge back to a node already in the cycle (e.g. `reviewer -> executor` on `on_flag:needs_revision`) is read as the loop-back, not as an extra step; edges leaving conditional-only nodes are not followed. Both cases are reported as `loop_plan_warning` events. `every_k:N` runs in every N-th cycle of the plan. `on_flag:F` reads `config["flags"][F]` (strings such as `"false"` parse as booleans). Flags are static run-config switches: nothing sets them at runtime, so an `on_flag` step either never runs or runs in every cycle.

### Stop conditions

`stop_conditions` in a loop YAML is evaluated by the engine. A list of `key:value` strings, one-key mappings, or a single mapping is accepted:

```yaml
stop_conditions:
  - "user_stop"              # always active (informational)
  - "budget_exceeded"        # always active (informational)
  - "max_iterations:40"      # extra cap; the run's own max_iterations also applies
  - "budget_max_tokens:2000000"
  - "budget_max_cost:5.0"
  - "revision_budget:2"      # at most 2 executor revisions (built-in default 4; "revision_budget:none" disables it)
  - "grade_at_least:B"       # stop after a review/consensus step graded B or better
  - "on_flag:knowledge_graph_stable"   # stop when run.config["flags"][flag] is truthy
  - "convergence_enabled:true"         # opt in to the adaptive rule (off by default)
  - "convergence_rule:all"   # any convergence_* key, convergence_enabled, consensus_gate
```

Each key maps onto the run-config key of the same name. Precedence, from highest to lowest: run config (`config_overrides`), then loop `stop_conditions`, then the loop `config` block, then consensus-merger node settings (consensus keys only), then the workspace's per-run budgets (`budget_max_tokens_per_run` / `budget_max_cost_per_run` as `budget_max_tokens` / `budget_max_cost`), then built-in defaults. The run's `max_iterations` field and a `max_iterations` value from the run config or the loop are both caps: the loop stops at the smaller one. For `revision_budget` the key's presence counts: `null` in the run config (or `revision_budget:none` in the loop) disables the budget instead of falling back to the default. Unknown entries are reported in a `stop_conditions_ignored` event and otherwise ignored. A malformed value fails the run with `failed:invalid_config`.

Grades compare as A+ > A > A- > B+ > B > … > F. `INCOMPLETE` (failed panel) never satisfies `grade_at_least` and never enters the convergence history. A partial panel's grade (computed from the usable lenses only, flagged `grade_valid: false` in the consensus report) does not either under the default `consensus_partial_policy: no_stop`; `count` treats it as a full review and `fail` fails the step. A single reviewer's grade is derived from the severities in its JSON critique contract (a grade field inside the block can only make it worse; prose outside the block is ignored); a JSON block that cannot count as a review gives no grade. The consensus gate (`consensus_gate: true`) blocks both convergence and `grade_at_least` while unresolved CRITICAL critiques remain, also after a resume. Budgets, iteration caps and the revision budget still apply.

Whichever condition is reached first stops the run. The engine checks, before every step: cancellation, the token / cost budgets and iteration caps, `on_flag` stop conditions and, before an executor step, the revision budget; after every step: `grade_at_least` (review / consensus steps) and, when enabled, the convergence rule. When several conditions hold at the same point, that order decides the stop reason: a run whose `max_iterations` cap coincides with the end of its revision budget (e.g. `reviewer_consensus` with `max_iterations: 10`) stops with `max_iterations`.

### Default stopping policy (calibrated)

By default a run gets a **fixed budget of four executor revisions** (`revision_budget: 4`) and the adaptive convergence rule is **off** (`convergence_enabled: false`):

- `revision_budget` is the maximum number of executor revisions after the initial executor submission. When the executor has produced its initial submission plus `revision_budget` revisions, the loop runs the remaining non-executor steps of that cycle, so the final artifact is reviewed, and then stops with stop reason `revision_budget`. On executor-first `reviewer_consensus` the default gives 5 executor steps and 5 panels (10 iterations); with `seed_executor_output` the seed counts as the initial submission (`reviewer_consensus`: P0 E1 P1 … E4 P4, 9 iterations). Gated steps of the last cycle still run when their condition holds (`executor_reviewer` with `revision_budget: 2`: E R E R E R A).
- `revision_budget: 0` reviews only the initial submission; `null` disables the budget (open-ended loops such as fixed-horizon experiments or long `research_followups` runs set it explicitly). Loops without an executor step ignore it. On resume, the executor submissions already made are counted from the stored iterations.
- The adaptive rule stays fully available: set `convergence_enabled: true` (run config or loop `stop_conditions`); it then uses `DEFAULT_STOPPING_RULE` (any of grade-stable / no-Critical-High / output-similar) unless the `convergence_*` keys change it, and stops the run if it fires before the budget runs out.
- `run_meta.json` records the budget the run used (`config.revision_budget`, also when it came from the loop or the default) and `config.effective_stopping` (budget and convergence switch with their sources, the adaptive rule, executor submissions and revisions made).

Calibration: the stopping-rule calibration (`experiments/stopping/`; results in `experiments/stopping/results/stopping_results.json` and `stopping_rules.csv`) replayed every candidate rule on 24 real executor–panel trajectories (two model configurations × 12 planted-flaw write-ups, horizon of five revisions, every state judged against the planted flaws). Pooled over both configurations, the pre-specified selection rule (lowest premature-stop rate among rules that stop before the horizon in at least 50 % of runs; ties broken by fewer revisions) selects the fixed budget of four revisions: premature-stop rate 2/24 = 0.083. The adaptive any-of-three rule, which coincided with grade stability alone, had 3/24 = 0.125 at about three revisions; rules requiring no Critical/High critiques or similar executor outputs never stopped within five revisions. Limits: the difference from the adaptive rule is one trajectory (2 vs 3 of 24), so the budget is a conservative default rather than a demonstrated improvement; it costs about one revision more per run (4 vs about 3), does not detect convergence, and was calibrated on revision-only trajectories (the executor could not run analyses) of one loop (`reviewer_consensus` with two Codex model configurations). `backend.orchestrator.convergence.replay_stopping_rule(..., revision_budget=k)` replays the budget (alone with `convergence=False`, or together with an adaptive rule) on recorded trajectories; with the calibration's seeded panel-first trajectories, `revision_budget=k` ends at executor state E_k, the state of the replay's `fixed_k` rule, after one extra panel that reviews it (`backend/tests/test_revision_budget.py`).

### Stop reasons

`RunState.stop_reason`, the `runs.stop_reason` column and `run_meta.json` record why a loop ended:

| Value | Meaning |
|-------|---------|
| `converged:<signal>[+<signal>…]` | the stopping rule fired; the signals that held (`grade_stable`, `no_critical_high`, `output_similar`) |
| `max_iterations` | iteration cap reached |
| `budget_tokens` / `budget_cost` | token / cost budget reached (input + output tokens and cost as reported by the provider) |
| `revision_budget` | the initial executor submission plus `revision_budget` revisions were made and the steps after the last one (its review) have run; the default policy |
| `stop_condition:grade_at_least` / `stop_condition:on_flag:<flag>` | a loop stop condition was met |
| `cancelled` | stopped by the user (`POST /api/runs/{id}/stop`, `miw stop`) |
| `failed:<reason>` | e.g. `adapter_failed`, `adapter_error`, `consensus_panel_failed`, `consensus_panel_partial` (policy `fail`), `invalid_config` (including a malformed budget such as `"2,000,000"`), `unknown_preset`, `concurrency_limit`, `exception` |
| `no_runnable_steps` / `empty_plan` | every step of the plan is gated off / the plan is empty |

### Resume and crash recovery

A run is marked `running` in the database when its loop starts. On server start, `running` runs (and `pending` runs that already have iterations) with persisted iterations become `paused` with `stop_reason: interrupted`; runs without iterations become `failed`. `POST /api/runs/{id}/resume` is refused with HTTP 409 while a stopped run's engine is still finishing its in-flight step. A resumed run replays the compiled plan against the stored iterations, so it continues with the step an uninterrupted run would run next (gated `every_k` / `on_flag` steps included), and restores the latest executor output, the exact pending feedback (`<role>_feedback.md`), the convergence history (partial-panel grades excluded under `no_stop`), the executor submissions counted by the revision budget and the consensus gate.

## Run Configuration Keys

Run configuration is `RunCreate.config_overrides` (API) or `-c KEY=VALUE` (CLI). `model` and `reasoning_effort` are separate request fields.

### Models, tools, budgets

| Key | Default | Description |
|-----|---------|-------------|
| `model` (request field) | `""` (CLI default) | Model forwarded to every adapter call |
| `reasoning_effort` | `""` (CLI default) | Effort forwarded to every adapter call (Claude `--effort`, Codex `-c model_reasoning_effort=`; Gemini has no effort flag, so it is recorded as unsupported) |
| `reviewer_allow_tools` | `false` | Allow agent tools for every non-executor call (reviewer lenses, consensus panels, LLM adjudicator, idea generator, …) |
| `executor_allow_tools` | `true`, but `false` when `code_execution_enabled` | Allow agent tools for executor calls (Claude tools only when the workspace is set). With verified execution the executor has no agent tools by default, so every number must come from the sandboxed code executor; setting it to `true` together with code execution is allowed but logged, emitted as `executor_tools_with_code_execution` and recorded in `run_meta.json` (`config.effective_tool_settings`) |
| `adapter_options` | `{}` | Provider options, e.g. `{"codex_ignore_user_config": false}`, `{"claude_max_turns": 30}` |
| `adapter_timeout` | `900` | Seconds per single-step adapter call |
| `max_retries` | `3` | Transient-error retries per adapter call (errors categorised `[rate_limit]`, `[timeout]`, `[network]`, `[overloaded]`; never `[auth]`, `[quota]`, `[max_turns]`, `[config]`) |
| `budget_max_tokens` | workspace budget, else `500000` | Token budget (input + output, summed over all calls, LLM adjudicator included). Set it explicitly for real Codex / Claude runs |
| `budget_max_cost` | workspace budget, else `10.0` | Cost budget in USD (provider-reported; Codex and Gemini report none) |
| `grade_at_least` | unset | See stop conditions |
| `flags` | `{}` | Static flags for `on_flag:` edges and stop conditions |
| `seed_executor_output` | unset | Text treated as the executor's iteration-0 output: a fresh run starts with the plan step after the first executor step (`reviewer_consensus`: panel → executor → panel …). The seed is the first artifact under review, the executor's "previous submission (iteration 0)" in its first revision, and the first executor output of the `output_similar` signal; it is not an engine iteration. Re-applied before a resume restores stored iterations. A non-string value, or a loop without an executor step, fails the run with `failed:invalid_config`; an empty string means no seed |

### Consensus (`consensus_*`)

| Key | Default | Description |
|-----|---------|-------------|
| `consensus_similarity_method` | `llm` | `llm` (LLM adjudication with the run's adapter, model and effort; one extra adjudicator call per step) \| `tfidf` \| `jaccard` \| `embedding` |
| `consensus_similarity_threshold` | method default | jaccard 0.1, tfidf 0.1 (calibrated, see *Calibration* below), embedding 0.7 (uncalibrated); unused for `llm` |
| `consensus_llm_fallback` | `tfidf` | Method used, at its default threshold, when the LLM adjudication fails (call error, unparseable answer, or an invalid partition after its retry) or no adjudicator is available: `tfidf` \| `jaccard` \| `embedding` \| `none` (merge nothing, the earlier behaviour) |
| `consensus_threshold` | `2` | Distinct lenses needed to escalate a merged critique by one severity level |
| `consensus_role_weights` | `{}` | Per-lens ranking weights |
| `consensus_embedding_model` | unset | Sentence-transformers model for `embedding` (never downloaded automatically) |
| `consensus_lens_timeout` | `adapter_timeout`, else 900 | Seconds per lens call |
| `consensus_max_retries` | `max_retries`, else 3 | Transient-error retries per lens |
| `consensus_retry_base_delay` | `1.0` | Backoff base (seconds) |
| `consensus_max_concurrency` | unset (all lenses) | Concurrent lens calls |
| `consensus_same_lens_merge` | `false` | Allow two critiques of one reviewer call in a merge group (applied per call, for every method including `llm`) |
| `consensus_lens_identity` | `role` | Identity for escalation / `raised_by` when a panel repeats a role: `role` (copies of one role are one lens) or `call` (every panel member is a distinct reviewer) |
| `consensus_unparsed_is_failure` | `true` | Output with no recognisable critique format (or a contract block that cannot count as a review) counts as a failed lens |
| `consensus_min_ok_lenses` | `1` | Fewer usable lenses means the panel failed: retried once, then the run fails |
| `consensus_partial_policy` | `no_stop` | A panel still partial after its retry: `no_stop` (feedback kept, grade never stops the run or enters convergence), `count` (grade counts as a full review), `fail` (fail the step) |
| `consensus_gate` | `false` | Keep revising while unresolved CRITICAL critiques remain |

Boolean keys accept true/false/1/0/yes/no/on/off; any other string is an error. `consensus_threshold` must be at least 2. The LLM adjudicator's answer must be a partition of the critique indices: a consistently 1-based answer is shifted and indices merely left out become singletons (both recorded as repairs); any other invalid answer is retried once. If the adjudication still fails (or the call fails, or the answer has no parseable `{"groups": ...}` block), the step falls back to `consensus_llm_fallback` (`tfidf` at 0.1) instead of merging nothing; the fallback is recorded in `CONSENSUS.json` (`consensus_meta.merge_info.fallback`: method, threshold, reason) and in the consensus report (`similarity_fallback`, `effective_similarity_method`), and the adjudicator's usage is still accounted. The mock adapter answers adjudication requests with a deterministic partition (exact-duplicate critiques grouped), so mock runs stay deterministic. Merged critiques keep the text and fix of every other member (`merged_members`), and the executor feedback lists each critique's `required_fix`, suggested experiment and the merged members' points.

A `consensus_merger` node in a loop YAML may set `similarity_method`, `similarity_threshold`, `consensus_threshold`, `role_weights` and `embedding_model` as defaults (`loops/reviewer_consensus.yaml` sets `similarity_method: llm`). A node threshold is ignored when the run selects a different similarity method; a node threshold without a node `similarity_method` is read as a `jaccard` threshold (the meaning it had in older loop files).

#### Calibration

The default method and thresholds come from the merge evaluation on the planted-flaw benchmark (`experiments/benchmark/merge_eval.py`; results in `experiments/benchmark/results/merge_eval.json`): 162 three-lens panels, 4,556 cross-lens critique pairs labelled duplicate / non-duplicate by the judge, thresholds chosen by leave-one-scenario-out cross-validation over 12 flawed scenarios. Held-out pairwise F1: `llm` 0.824, `tfidf` 0.814, `jaccard` 0.804 (`embedding` not evaluated). `llm` was the pre-specified winner (best held-out F1, ties broken by cost) and is the default; `tfidf`, the next best, is its fallback. The folds selected thresholds of 0.075–0.1 for both `jaccard` and `tfidf`; the defaults are 0.1, the upper end of that range. The earlier defaults (jaccard 0.5, tfidf 0.3) were not calibrated: at 0.5, `jaccard` recalled 1.2 % of the duplicate pairs (tfidf at 0.3: 61.7 %). They remain available as `backend.orchestrator.similarity.UNCALIBRATED_THRESHOLDS`, and analyses that were pre-specified with them pin them explicitly (the benchmark's `analyze.py` severity outcomes: jaccard 0.5; the stopping calibration and the orchestration experiments: jaccard 0.5).

### Stopping policy (`revision_budget`)

| Key | Default | Description |
|-----|---------|-------------|
| `revision_budget` | `4` | Maximum executor revisions after the initial submission (a `seed_executor_output` counts as the initial submission); the steps after the last revision still run, then the run stops with `revision_budget`. Integer ≥ 0, or `null` to disable (the key's presence counts). Also a loop stop condition (`revision_budget:N` / `revision_budget:none`). See *Default stopping policy (calibrated)* |

### Convergence (`convergence_*`)

| Key | Default | Description |
|-----|---------|-------------|
| `convergence_enabled` | `false` | Evaluate the adaptive stopping rule (off by default since the stopping calibration; the revision budget applies either way) |
| `convergence_window` | `3` | Consecutive observations per signal |
| `convergence_min_iterations` | `4` | No convergence before this many iterations |
| `convergence_similarity_threshold` | `0.9` | Word-Jaccard threshold for consecutive executor outputs |
| `convergence_signals` | all three | Subset of `grade_stable`, `no_critical_high`, `output_similar` |
| `convergence_rule` | `any` | `any` \| `all` \| `k_of_n` |
| `convergence_k` | `1` | Signals required for `k_of_n` |
| `convergence_require_no_critical` | `false` | Also require zero CRITICAL critiques in the latest review |
| `convergence_required_signals` | none | Signals that must ALL hold in addition (the combination rule applies to the others); e.g. no-Critical/High ∧ (grade-stable ∨ similar) = `convergence_required_signals: no_critical_high`, `convergence_rule: any` |

The adaptive defaults are the single constant `DEFAULT_STOPPING_RULE` in `backend/orchestrator/convergence.py`; the default policy is `DEFAULT_CONVERGENCE_ENABLED` (false) and `DEFAULT_REVISION_BUDGET` (4) in the same module. `replay_stopping_rule()` applies a `StoppingRule` offline to a recorded trajectory the way the engine does: failed and (by default) partial panels are skipped and `gate=True` mirrors the consensus gate (`revision_budget=k` adds the revision budget, `convergence=False` leaves the rule out). The output-similarity signal compares consecutive executor outputs only. Reviewer, consensus-feedback and idea-generator text is never compared.

### Verified execution (`code_execution_*`)

| Key | Default | Description |
|-----|---------|-------------|
| `code_execution_enabled` | `false` | Run the executor's fenced Python code after each executor step |
| `code_execution_backend` | `subprocess` | `subprocess` \| `sandbox_exec` (macOS) \| `docker` |
| `code_execution_python` | backend interpreter | Interpreter for the analysis code (point it at an environment with numpy etc.) |
| `code_execution_timeout` | `20` | Wall-clock seconds per execution |
| `code_execution_cpu_seconds` | 2 × timeout | CPU limit (0 disables) |
| `code_execution_memory_mb` | `4096` | Address-space limit (Linux) / container memory (docker); 0 disables |
| `code_execution_read_only_paths` | `[]` | Readable data directories (enforced by `sandbox_exec` and `docker`; `/`, the home directory or its ancestors, volume roots and shared temp roots are rejected) |
| `code_execution_allow_network` | `false` | Network access (enforceable only by `sandbox_exec` and `docker`) |
| `code_execution_max_file_mb` | `64` | File-size limit |
| `code_execution_max_processes` | unset (docker 128) | Process limit |
| `code_execution_max_open_files` | `256` | Open-file limit |
| `code_execution_output_limit_bytes` | `4000` | Bytes kept per stream (stdout: first and last halves; stderr: a short head and its tail) |
| `code_execution_block_mode` | `single` | `single` (all blocks as one script) \| `separate` |
| `code_execution_docker_image` | `python:3.11-slim` | Image for the docker backend (must exist locally; never pulled) |
| `code_execution_capture_text_bytes` | `4000` | Bytes of text shown per file the code wrote (`results.json` in the work directory is read first) |
| `code_execution_capture_total_bytes` | `16000` | Bytes of text shown over all written files |
| `code_execution_capture_max_files` | `50` | Written files listed from the work directory |
| `code_execution_persist_dir` | unset | Directory that receives the complete outputs of every execution unit: `<dir>/<timestamp>_<id>/` with `code.py`, the full `stdout.txt` / `stderr.txt` (not only the head and tail shown in the report) and `work/` (every regular file the code wrote; symlinks, hard links and special files are never followed or copied). `CODE_EXECUTION.json` records the unit as `persisted_to`; reproducibility packages include these outputs. With `sandbox_exec` or `docker` the executed code cannot write there (`subprocess` confines nothing) |
| `code_execution_persist_max_mb` | `256` | MB copied per execution unit (a file that does not fit is cut and noted) |

When execution is enabled, the executor system prompt gets the addendum `prompts/executor/mi_executor_code.yaml`. It asks for one self-contained fenced ```` ```python ```` block that reads only the data paths given in the task, prints a compact results summary and writes `results.json`, and it forbids reporting numbers that the code does not compute. The execution report covers the exit status, stdout, stderr (head and tail kept if long, so the final exception is visible) and the files written, including `results.json`. It is appended to the artifact the reviewers see and is returned to the next executor iteration so the executor can fix errors. An executor output without a Python block is reported to the reviewers as "nothing was executed".

#### Security statement

The full statement is in the docstring of `backend/orchestrator/code_executor.py`; this is a summary.

- **All backends** run `python -I -B -u` in a fresh temporary working directory with a minimal environment (no secrets from the backend environment), stdin from `/dev/null`, byte-capped output and a wall-clock timeout. `-I` only isolates the interpreter from `PYTHON*` variables and user site-packages. **It is not a security boundary.**
- **`subprocess`** (default) provides resource-limited isolation, **not a security boundary**. It applies CPU, file-size, open-file, core and optional process rlimits (if one cannot be applied the code is not run), runs the code in a new session and kills the whole process group on timeout. It does **not** restrict the network or the filesystem: the code runs as the backend user and can read and write anything that user can. A descendant that calls `setsid()` escapes the group kill but stays CPU-limited.
- **`sandbox_exec`** (macOS) adds OS-level confinement through a Seatbelt profile enforced by the kernel and inherited by all descendants. Network access is denied (unless `allow_network`), writes are allowed only in the per-execution temporary directory and a few device nodes (`/dev/null`, `/dev/zero`, `/dev/tty`, `/dev/dtracehelper`, `/dev/fd/<n>`), and reads only in system directories, `/opt/homebrew` and `/opt/local` (readable in full), interpreter paths, the temporary directory and `code_execution_read_only_paths` (the exact allow-list is in the `code_executor.py` docstring). File metadata (path existence) and the process list may remain visible. Apple marks `sandbox-exec` as deprecated.
- **`docker`** provides container isolation: `--network none`, a read-only root filesystem with only `/work` and a tmpfs `/tmp` writable, read-only data mounts, hard memory/CPU/PID caps, all capabilities dropped and a non-root user. The container shares the host kernel (Docker Desktop on macOS adds a VM boundary).
- **Memory limits are not enforced on macOS** by `subprocess` or `sandbox_exec`, because the kernel ignores address-space rlimits. Only `docker` enforces a hard memory cap.
- If a backend is unavailable, execution **fails closed** and is reported as not run. It never falls back to a weaker backend.

Separately, the agent CLIs can run their own tools when tools are allowed (executor steps by default, but not when verified execution is enabled). Reviewer calls run without tools by default. Each iteration records the agent-tool provenance of its calls (`agent_tools`: tools enabled, sandbox, and the tool / shell-command / web-search / MCP item counts the CLI reported).

## Adapters

All adapters shell out to CLI binaries. The prompt is sent on stdin. The CLI runs in its own session and process group; on timeout or cancellation the group gets SIGTERM and, after a short grace period, SIGKILL (descendants that called `setsid()` are not reached), and after a normal exit leftover group members are terminated too. Output is read incrementally, so an answer is returned even if a leftover descendant still holds the pipes. Each result records `provider`, `model`, `reasoning_effort` (the effort actually applied: `default` when Claude ignored an unknown value, `unsupported` for Gemini), `cli_version`, `input_tokens` (including cached input), `output_tokens` and `cached_input_tokens`.

| Provider | Command (flags only when set) |
|----------|-------------------------------|
| `claude_code` | `claude -p --output-format json --max-turns 1\|N [--model M] [--effort E] [--system-prompt SP]`, plus `--allowedTools Read,Write,Edit,Bash,Glob,Grep` when tools are allowed and a workspace is set (turn limit N = `max_turns`, default 30), or `--tools "" --strict-mcp-config` with `--max-turns 1` when tools are off. Efforts outside low/medium/high/xhigh/max fail with `[config]` |
| `codex_cli` | `codex exec --skip-git-repo-check --ephemeral --json --sandbox read-only\|workspace-write [-m M] [-c model_reasoning_effort=E] [tools-off args] [--cd WS] -`. With tools off: `--sandbox read-only -c web_search="disabled" --disable shell_tool --disable unified_exec --disable apps --disable plugins --disable multi_agent --disable browser_use --disable computer_use --disable image_generation --disable in_app_browser --ignore-user-config` (no shell, no web search, no apps/plugins/MCP servers, `config.toml` not loaded; auth still from `CODEX_HOME`). `$CODEX_HOME/AGENTS.md` is still injected unless `codex_home` points to a home without it; `codex_home` is passed to the CLI as `CODEX_HOME` |
| `gemini_cli` | `gemini --yolo\|--approval-mode default [-m M] -o json -p " "` (`--yolo` only when tools are allowed). With tools off, a system-settings file (`GEMINI_CLI_SYSTEM_SETTINGS_PATH`) also excludes every built-in tool, including `google_web_search` and the file-read tools, and allows no MCP server. `@` in the prompt is escaped to stop the CLI from expanding files |
| `mock` | In-process synthetic responses (deterministic grade sequence C → B → B+ → A-), no external calls |

Failed CLI calls return `success=False` with a `[category] detail` error (`rate_limit`, `quota`, `auth`, `overloaded`, `timeout`, `network`, `max_turns`, `config`, `empty_output`, `unknown`); the detail always contains the line the category came from. They are never treated as empty successful output.

Every adapter call of a run (each attempt, each lens, the LLM adjudicator) is also recorded by `backend.telemetry.collector.TelemetryAdapter` in the in-memory collector served at `/api/telemetry/*` and in the `telemetry` table.

## Run Outputs

Each iteration writes to `<workspace>/runs/<run_id>/iter_NNNN/`:

- `<role>_output.md`: the step's full cleaned output, plus the artifacts parsed from it (`MECH.md`, `EVAL.md`, `XP.md`, `METHOD.md`, `RUN_SUMMARY.md`, …);
- consensus steps: `EVAL.md` and `consensus_merger_output.md` (merged, ranked feedback), one `<lens>_output.md` per panel lens, and `CONSENSUS.json` (report, attempts, merged critiques, merge provenance);
- every non-executor step: `<role>_feedback.md`, the exact text queued for the next executor prompt (used to rebuild that prompt on resume);
- executor steps with verified execution: `RUN_LOG.md` (the report as shown to the reviewers) and `CODE_EXECUTION.json` (full report; `blocks[].persisted_to` names the complete outputs when `code_execution_persist_dir` is set).

`runs/<run_id>/run_meta.json` records the provider, model, reasoning effort, stop reason, token split, cost, CLI versions, models used, loop source and stop conditions, the run-config settings that affect results, the convergence metrics, the effective agent-tool settings (`config.effective_tool_settings`), the effective stopping policy (`config.revision_budget`, `config.effective_stopping`), and one entry per iteration (role, status, provider, model, effort, CLI version, input/output/cached tokens, cost, duration, grade, whether a consensus grade came from a complete panel, code-execution summary, agent-tool provenance). The same per-iteration fields are stored in the `iterations` table as each iteration finishes. Columns are added in place to existing databases.

Database writes are robust to lock contention (many concurrent runs, slow disks): every connection waits up to `MIW_DB_BUSY_TIMEOUT` (30 s) for a lock, and every write transaction starts with `BEGIN IMMEDIATE` and is retried with exponential backoff and jitter on "database is locked / busy" (up to `MIW_DB_WRITE_ATTEMPTS` = 8 attempts, at most `MIW_DB_RETRY_DEADLINE` = 300 s). An iteration whose progress write still fails is never dropped: it is queued and written with the next successful write or at run end, logged, emitted as a `persistence_warning` event, noted in the iteration's `logs` and listed in `run_meta.json` (`persistence_warnings`). Transient contention therefore never fails a run.

A reproducibility package (`POST /api/repropack/{run_id}`, `miw run repropack <run_id>`) contains the top-level run files (`artifacts/`), every iteration directory in full (`iterations/iter_NNNN/`: step outputs, parsed artifacts, `CONSENSUS.json`, `CODE_EXECUTION.json`, `RUN_LOG.md`, `<role>_feedback.md`, lens outputs), the persisted execution outputs of the run's units (`code_execution/iter_NNNN/<unit>/`; read only from inside the run's `code_execution_persist_dir`), metadata (run state, environment, git state, timestamps) and a README. Files above the per-file cap (`max_file_mb` query parameter / `--max-file-mb`; default `MIW_REPROPACK_MAX_FILE_MB` = 25 MB; 0 = no cap) are left out and listed in the package README.

## Agent Roles

| Role | Prompt | Function |
|------|--------|----------|
| Executor | `executor/mi_executor` (+ `executor/mi_executor_code` when verified execution is on) | MI analysis: attention extraction, ablation, activation patching |
| Reviewer | `reviewer/mi_reviewer` | Critique claim validity, reproducibility, confounders, controls |
| Reviewer (combined checklist) | `reviewer/mi_reviewer_combined` | Single reviewer covering all three lens checklists |
| Adversarial Reviewer | `adversarial_reviewer/adversarial_reviewer` | Red-team for p-hacking, overclaiming, causal confusion, leakage |
| Bio-Plausibility Checker | `biological_plausibility/bio_plausibility_checker` | Validate tissue specificity, pathway consistency, expression levels |
| Idea Generator | `idea_generator/follow_up_proposer` | Prioritized follow-up proposals with value/feasibility scores |
| Knowledge Extractor | `knowledge_extractor/knowledge_miner` | Mine claims, estimate uncertainty, generate falsification tests |
| Automator | `automator/automation_proposer` | Design reusable pipelines from validated protocols |
| Publication Manager | `publication_manager/publication_manager` | Venue selection, checklists, cover letters |
| Article Publisher | `article_publisher/comms_writer` | Format results for blogs, social media, internal summaries |

Only executor steps receive the revision framing ("=== REVISION REQUEST … revise your artifacts"), together with the feedback, their previous submission and their execution report. Every other step receives the task as context, the iteration number and the executor's latest full output under `=== ARTIFACT UNDER REVIEW ===`.

## Project Structure

```
mi-workbench/
├── backend/
│   ├── adapters/          # Provider implementations
│   ├── api/               # FastAPI route handlers
│   ├── artifacts/         # Artifact parsing, validation, writing
│   ├── knowledge/         # Claim extraction, graph, parser
│   ├── orchestrator/      # Engine, runner, DSL, presets, convergence, consensus,
│   │                      # similarity, feedback, code executor (+ sandbox/)
│   ├── repropack/         # Reproducibility package generation
│   ├── telemetry/         # Metrics collection
│   ├── utils/             # Output cleaning utilities
│   ├── tests/             # pytest suite
│   ├── database.py        # SQLite schema, additive migrations and CRUD
│   ├── models.py          # Pydantic data models
│   └── main.py            # FastAPI app entry point
├── cli/                   # Click-based CLI
├── docs/                  # Reproduction guide
├── experiments/           # Evaluation harnesses
├── frontend/              # React UI
├── loops/                 # YAML loop definitions
├── prompts/               # YAML prompt templates
└── requirements*.txt
```

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest -q                       # uses pytest.ini (testpaths = backend/tests)

# Selected areas
python -m pytest backend/tests/test_engine_prompts.py -v      # prompt assembly / request wiring
python -m pytest backend/tests/test_convergence_rules.py -v   # stopping rules, stop reasons
python -m pytest backend/tests/test_revision_budget.py -v     # default stopping policy (revision budget)
python -m pytest backend/tests/test_presets_runnable.py -v    # all presets end-to-end (mock)
python -m pytest backend/tests/test_e2e_pipeline.py -v        # end-to-end through the API
python -m pytest backend/tests/test_stress.py -v              # concurrency stress
```

See `docs/REPRODUCING.md` for environment setup and the experiment reproduction commands.

## Configuration

Environment variables (all prefixed with `MIW_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MIW_DB_PATH` | `data/mi_workbench.db` | SQLite database path |
| `MIW_HOST` | `127.0.0.1` | Server host |
| `MIW_PORT` | `8000` | Server port |
| `MIW_WORKSPACE_BASE` | `workspaces/` | Base directory for workspaces |
| `MIW_PROMPTS_DIR` | `prompts/` | Prompt templates directory |
| `MIW_LOOPS_DIR` | `loops/` | Loop definitions directory (searched before the bundled `loops/`) |
| `MIW_LOG_LEVEL` | `INFO` | Logging level |
| `MIW_DB_BUSY_TIMEOUT` | `30` | Seconds each SQLite connection waits for a lock |
| `MIW_DB_WRITE_ATTEMPTS` | `8` | Attempts per write transaction on lock contention |
| `MIW_DB_RETRY_BASE_DELAY` / `MIW_DB_RETRY_MAX_DELAY` | `0.05` / `2.0` | Backoff before a retry: base doubled per attempt, capped, with jitter (seconds) |
| `MIW_DB_RETRY_DEADLINE` | `300` | No new write attempt after this many seconds |
| `MIW_REPROPACK_MAX_FILE_MB` | `25` | Per-file size cap of reproducibility packages (0 = no cap) |

## License

MIT (see `LICENSE`). Citation metadata: `CITATION.cff`.
