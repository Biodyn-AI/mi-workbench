#!/usr/bin/env python
"""Corroborate the wired consensus panel with a real LLM backend (Gemini CLI).

Runs one executor artifact through the three-reviewer panel via the same
ConsensusReviewer.run_panel path the engine uses, with the real Gemini adapter,
and reports per-reviewer critique counts, merged distinct critiques, and
agreement-based escalations. This is a small, bounded corroboration (3 calls)
of the mechanism characterised in bulk with the mock adapter.

Usage:
    python scripts/real_consensus_probe.py [out.json]
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.adapters.gemini_cli import GeminiCliAdapter  # noqa: E402
from backend.orchestrator.consensus import ConsensusReviewer  # noqa: E402
from backend.registry.prompts import PromptRegistry  # noqa: E402
from backend.models import WorkspaceContext  # noqa: E402
from backend.utils.output_cleaner import strip_thinking_traces  # noqa: E402

PANEL = [
    ("reviewer", "reviewer/mi_reviewer"),
    ("adversarial_reviewer", "adversarial_reviewer/adversarial_reviewer"),
    ("bio_plausibility_checker", "biological_plausibility/bio_plausibility_checker"),
]

ARTIFACT = """\
# MECH.md

## Attention-based gene regulatory network inference in Geneformer

We extract layer-15 attention from Geneformer (V2-316M) on 2,000 K562 cells and
correlate attention weights with gene co-expression and with TRRUST regulatory
edges. Attention recovers co-expression (Pearson r = 0.72, p < 0.001) more than
causal edges. GRN-recovery AUROC is 0.743 for attention versus 0.703 for a
plain co-expression baseline; the split-sample AUROC is 0.75. Under
co-expression residualization, attention retains 76% of its signal. We conclude
that attention heads encode co-expression structure rather than causal
regulation.
"""

FORMAT_HINT = (
    "\n\nReview the artifact below. List each concern on its own line, each "
    "prefixed with a bracketed severity: [CRITICAL], [HIGH], [MEDIUM], [LOW], "
    "or [INFO]. Be concise."
)


def _build_system_prompt(reg, role, ref):
    try:
        parts = ref.split("/", 1)
        tmpl = reg.load_prompt(parts[0], parts[1])
        sp = reg.resolve_template(tmpl, {"role": role, "iteration": "1",
                                         "task": "probe attention vs regulation"})
    except Exception:
        sp = f"You are the {role} in an MI research pipeline."
    return sp + FORMAT_HINT


async def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/plos/data/real_consensus.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reg = PromptRegistry()
    adapter = GeminiCliAdapter()
    if not adapter.is_available():
        print("gemini not available; skipping real corroboration.")
        return

    cr = ConsensusReviewer(panel=PANEL)
    merged, raw = await cr.run_panel(
        ARTIFACT, adapter, WorkspaceContext(workspace_path=""),
        system_prompt_builder=lambda role, ref: _build_system_prompt(reg, role, ref),
    )
    report = cr.compute_consensus_report(merged)

    per_reviewer = {}
    for (role, _), r in zip(PANEL, raw):
        parsed = cr._parse_review_output(strip_thinking_traces(r.output))
        per_reviewer[role] = {
            "raw_critiques": len(parsed),
            "success": r.success,
        }

    result = {
        "backend": "gemini_cli",
        "artifact_chars": len(ARTIFACT),
        "per_reviewer": per_reviewer,
        "merged_distinct": report["total_merged"],
        "multi_reviewer_agreements": report["multi_reviewer"],
        "grade": report["grade"],
        "severity_histogram": report["severity_histogram"],
        "merged_critiques": [
            {"severity": c.severity.value, "description": c.description}
            for c in merged.critiques
        ],
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "merged_critiques"},
                     indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
