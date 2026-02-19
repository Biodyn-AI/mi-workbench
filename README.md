# MI-Workbench

An autonomous multi-agent platform for mechanistic interpretability research on biological foundation models.

MI-Workbench orchestrates iterative research loops where LLM agents execute analyses, review methodology, red-team for confounders, and refine findings — all without human intervention.

## Key Features

- **Graph-based workflow DSL** — Define multi-agent research loops in YAML with conditional edges (`always`, `every_k:N`, `on_flag:F`), parallel branching, and configurable stop conditions
- **Consensus review** — Run 3 independent critic agents in parallel (standard reviewer, adversarial reviewer, biological plausibility checker), deduplicate findings via Jaccard similarity, and escalate severity on inter-reviewer agreement
- **Structured feedback** — Map reviewer critiques to specific artifacts and sections, enabling targeted revision instead of wholesale rewriting
- **Provider-agnostic adapters** — Swap between Claude Code, OpenAI Codex, Google Gemini, or a mock adapter through a common `BaseAdapter` interface
- **Knowledge extraction** — Mine structured claims from artifacts, estimate uncertainty, generate falsification tests, and build a claim graph
- **Reproducibility packaging** — Bundle artifacts, prompts, environment metadata, and git state into portable ZIP archives
- **Real-time monitoring** — WebSocket streaming of run progress to connected clients

## Architecture

```
Interface Layer     CLI (Click)  |  REST API (FastAPI)  |  WebSocket
                         │
Orchestration Layer      Loop Engine
                         ├── State Machine
                         ├── Convergence Detector
                         ├── Feedback Formatter
                         ├── Consensus Reviewer
                         └── Follow-Up Executor
                         │
Adapter Layer       Claude Code  |  Codex CLI  |  Gemini CLI  |  Mock
                         │
Persistence Layer   SQLite (aiosqlite)  |  Artifact Writer  |  Knowledge Graph
```

## Quick Start

### Install

```bash
pip install -r requirements.txt
```

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
  --provider mock \
  --max-iterations 5

# Start a run with Gemini
python cli/miw.py run preset executor_reviewer \
  --workspace-id <workspace_id> \
  --task "Investigate whether attention heads encode causal regulation" \
  --provider gemini_cli \
  --max-iterations 10
```

### Run via API

```bash
# Health check
curl http://127.0.0.1:8000/health

# Create workspace
curl -X POST http://127.0.0.1:8000/api/workspaces \
  -H "Content-Type: application/json" \
  -d '{"name": "my-project", "path": "./workspaces/my-project"}'

# Start a run
curl -X POST http://127.0.0.1:8000/api/runs \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": "<id>",
    "loop_preset": "executor_reviewer",
    "task": "Analyze attention patterns in Geneformer",
    "provider": "mock"
  }'
```

## Loop Presets

| Preset | Description |
|--------|-------------|
| `executor_reviewer` | Standard executor-reviewer loop with periodic adversarial review (every 3rd iteration) |
| `reviewer_consensus` | Three parallel reviewers (standard, adversarial, bio-plausibility) with consensus merger |
| `research_followups` | Executor-reviewer with automated follow-up proposal generation |
| `example_custom` | Advanced loop with conditional escalation, knowledge extraction, and graph stability detection |

Custom loops can be defined in YAML and placed in the `loops/` directory.

## Agent Roles

Nine specialized roles with domain-specific system prompts:

| Role | Prompt | Function |
|------|--------|----------|
| Executor | `executor/mi_executor` | MI analysis: attention extraction, ablation, activation patching |
| Reviewer | `reviewer/mi_reviewer` | Critique claim validity, reproducibility, confounders, controls |
| Adversarial Reviewer | `adversarial_reviewer/adversarial_reviewer` | Red-team for p-hacking, overclaiming, causal confusion, leakage |
| Bio-Plausibility Checker | `biological_plausibility/bio_plausibility_checker` | Validate tissue specificity, pathway consistency, expression levels |
| Idea Generator | `idea_generator/follow_up_proposer` | Prioritized follow-up proposals with value/feasibility scores |
| Knowledge Extractor | `knowledge_extractor/knowledge_miner` | Mine claims, estimate uncertainty, generate falsification tests |
| Automator | `automator/automation_proposer` | Design reusable pipelines from validated protocols |
| Publication Manager | `publication_manager/publication_manager` | Venue selection, checklists, cover letters |
| Article Publisher | `article_publisher/comms_writer` | Format results for blogs, social media, internal summaries |

## Artifact Types

Each run iteration produces structured markdown artifacts:

- `MECH.md` — Mechanistic analysis with evidence tables
- `EVAL.md` — Evaluation report with severity-graded critiques
- `XP.md` — Experiment plan with hypotheses and controls
- `METHOD.md` — Detailed methodology documentation
- `KNOWLEDGE.md` — Extracted claims and relationships
- `RUN_SUMMARY.md` — High-level summary of findings

## Project Structure

```
mi-workbench/
├── backend/
│   ├── adapters/          # Provider implementations
│   ├── api/               # FastAPI route handlers (25+ endpoints)
│   ├── artifacts/         # Artifact parsing, validation, writing
│   ├── knowledge/         # Claim extraction, graph, parser
│   ├── orchestrator/      # Engine, DSL, state machine, consensus, feedback
│   ├── repropack/         # Reproducibility package generation
│   ├── telemetry/         # Metrics collection
│   ├── utils/             # Output cleaning utilities
│   ├── tests/             # 26 test files, 360 tests
│   ├── database.py        # SQLite schema and CRUD
│   ├── models.py          # Pydantic data models
│   └── main.py            # FastAPI app entry point
├── cli/                   # Click-based CLI
├── frontend/              # React UI
├── loops/                 # YAML loop definitions
├── prompts/               # YAML prompt templates (9 roles)
├── paper/                 # Research paper (TMLR format)
└── requirements.txt
```

## Testing

```bash
# Run all tests
python -m pytest backend/tests/ -v

# Run with coverage
python -m pytest backend/tests/ --cov=backend --cov-report=term-missing

# Run specific test categories
python -m pytest backend/tests/test_e2e_pipeline.py -v    # End-to-end
python -m pytest backend/tests/test_consensus.py -v        # Consensus review
python -m pytest backend/tests/test_stress.py -v           # Concurrency stress
python -m pytest backend/tests/test_parser_fuzz.py -v      # Fuzzing
```

## Configuration

Environment variables (all prefixed with `MIW_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `MIW_DB_PATH` | `data/mi_workbench.db` | SQLite database path |
| `MIW_HOST` | `127.0.0.1` | Server host |
| `MIW_PORT` | `8000` | Server port |
| `MIW_WORKSPACE_BASE` | `workspaces/` | Base directory for workspaces |
| `MIW_PROMPTS_DIR` | `prompts/` | Prompt templates directory |
| `MIW_LOOPS_DIR` | `loops/` | Loop definitions directory |
| `MIW_LOG_LEVEL` | `INFO` | Logging level |

## License

MIT
