"""
Knowledge extractor for mechanistic interpretability research artifacts.
Parses MECH.md and EVAL.md to extract structured claims with evidence pointers,
uncertainty estimates, and falsification tests.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.models import ClaimNode, KnowledgeBaseUpdate


class KnowledgeExtractor:
    """Extracts structured claims from MI research artifacts."""

    # Patterns that typically introduce claims in research artifacts
    CLAIM_INDICATORS = [
        r"(?:we (?:found|observed|show|demonstrate) that)\s+(.+?)(?:\.|$)",
        r"(?:results (?:show|indicate|suggest|confirm) that)\s+(.+?)(?:\.|$)",
        r"(?:this (?:indicates|suggests|confirms|demonstrates) that)\s+(.+?)(?:\.|$)",
        r"AUROC\s*=\s*[\d.]+",
        r"p\s*[<>=]\s*[\d.]+",
        r"(?:significantly|marginally)\s+(?:higher|lower|different|better|worse)",
    ]

    CLAIM_TYPE_KEYWORDS = {
        "empirical": ["AUROC", "AUPRC", "p =", "p <", "CI", "bootstrap", "effect size"],
        "methodological": ["we used", "correction", "BH-FDR", "permutation", "algorithm"],
        "interpretive": ["suggests", "indicates", "consistent with", "implies", "captures"],
        "comparative": ["outperforms", "exceeds", "versus", "compared to", "vs."],
    }

    def __init__(self) -> None:
        self._claim_counter = 0

    def _next_id(self) -> str:
        self._claim_counter += 1
        return f"claim_{self._claim_counter:04d}"

    def _classify_claim(self, text: str) -> str:
        """Classify a claim by type based on keyword matching."""
        scores: dict[str, int] = {t: 0 for t in self.CLAIM_TYPE_KEYWORDS}
        text_lower = text.lower()
        for claim_type, keywords in self.CLAIM_TYPE_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    scores[claim_type] += 1
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best if scores[best] > 0 else "empirical"

    def _estimate_uncertainty(self, text: str) -> float:
        """Estimate uncertainty (0=certain, 1=highly uncertain) from claim text."""
        uncertainty = 0.5
        text_lower = text.lower()
        # Strong evidence lowers uncertainty
        if re.search(r"p\s*<\s*0\.001", text):
            uncertainty -= 0.2
        elif re.search(r"p\s*<\s*0\.01", text):
            uncertainty -= 0.15
        elif re.search(r"p\s*<\s*0\.05", text):
            uncertainty -= 0.1
        # Confidence intervals lower uncertainty
        if "95% ci" in text_lower or "confidence interval" in text_lower:
            uncertainty -= 0.1
        # Hedging language increases uncertainty
        for hedge in ["suggests", "may", "possibly", "marginally", "trend"]:
            if hedge in text_lower:
                uncertainty += 0.1
        # Large sample sizes lower uncertainty
        if re.search(r"n\s*[>=]\s*1000", text):
            uncertainty -= 0.1
        return max(0.05, min(0.95, uncertainty))

    def _suggest_falsification(self, claim: str) -> list[str]:
        """Generate falsification test suggestions for a claim."""
        tests = []
        claim_lower = claim.lower()
        if "auroc" in claim_lower or "outperform" in claim_lower:
            tests.append("Repeat with a stronger baseline (e.g., variance or mutual information)")
        if "layer" in claim_lower:
            tests.append("Apply split-sample validation to test layer selection stability")
        if "attention" in claim_lower and "correlation" in claim_lower:
            tests.append("Residualize attention against correlation and test if residual AUROC > 0.5")
        if "significant" in claim_lower:
            tests.append("Increase multiple testing correction stringency (e.g., Bonferroni)")
        if not tests:
            tests.append("Replicate on an independent dataset")
            tests.append("Test with a permutation null model")
        return tests

    def extract_claims(
        self, mech_content: str, eval_content: str
    ) -> list[ClaimNode]:
        """Extract structured claims from MECH.md and EVAL.md content."""
        claims: list[ClaimNode] = []

        for source, content in [("MECH.md", mech_content), ("EVAL.md", eval_content)]:
            lines = content.split("\n")
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                if not line_stripped or line_stripped.startswith("#"):
                    continue
                for pattern in self.CLAIM_INDICATORS:
                    matches = re.finditer(pattern, line_stripped, re.IGNORECASE)
                    for match in matches:
                        claim_text = match.group(0).strip()
                        if len(claim_text) < 15:
                            # Use the full line for short matches (e.g., AUROC values)
                            claim_text = line_stripped
                        # Avoid duplicate claims
                        if any(c.claim == claim_text for c in claims):
                            continue
                        claim = ClaimNode(
                            id=self._next_id(),
                            claim=claim_text,
                            evidence_pointers=[f"{source}:L{i + 1}"],
                            uncertainty=self._estimate_uncertainty(claim_text),
                            falsification_tests=self._suggest_falsification(claim_text),
                            source_artifact=source,
                        )
                        claims.append(claim)
        return claims

    def build_knowledge_md(self, claims: list[ClaimNode]) -> str:
        """Build a human-readable KNOWLEDGE.md from extracted claims."""
        sections: dict[str, list[ClaimNode]] = {
            "empirical": [],
            "methodological": [],
            "interpretive": [],
            "comparative": [],
        }
        for c in claims:
            ctype = self._classify_claim(c.claim)
            sections[ctype].append(c)

        lines = [
            "# Knowledge Base",
            "",
            f"**Extracted:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Total claims:** {len(claims)}",
            "",
        ]

        # Summary by confidence
        high_conf = [c for c in claims if c.uncertainty < 0.3]
        med_conf = [c for c in claims if 0.3 <= c.uncertainty < 0.6]
        low_conf = [c for c in claims if c.uncertainty >= 0.6]
        lines.append(f"**High confidence ({len(high_conf)})** | "
                     f"**Medium ({len(med_conf)})** | "
                     f"**Low ({len(low_conf)})**")
        lines.append("")

        type_labels = {
            "empirical": "Empirical Findings",
            "interpretive": "Interpretive Claims",
            "comparative": "Comparative Results",
            "methodological": "Methodological Notes",
        }
        for ctype, label in type_labels.items():
            group = sections[ctype]
            if not group:
                continue
            lines.append(f"## {label}")
            lines.append("")
            for c in group:
                conf_label = "HIGH" if c.uncertainty < 0.3 else "MED" if c.uncertainty < 0.6 else "LOW"
                lines.append(f"- [{conf_label}] {c.claim}")
                lines.append(f"  - Evidence: {', '.join(c.evidence_pointers)}")
                if c.falsification_tests:
                    lines.append(f"  - Falsification: {c.falsification_tests[0]}")
            lines.append("")

        # Open questions
        weak = [c for c in claims if c.uncertainty >= 0.6]
        if weak:
            lines.append("## Open Questions")
            lines.append("")
            for c in weak:
                lines.append(f"- {c.claim} (uncertainty: {c.uncertainty:.2f})")
            lines.append("")

        return "\n".join(lines)

    def build_facts_jsonl(self, claims: list[ClaimNode]) -> str:
        """Build machine-readable JSONL from claims (one JSON object per line)."""
        lines = []
        for c in claims:
            record = {
                "id": c.id,
                "claim": c.claim,
                "type": self._classify_claim(c.claim),
                "uncertainty": round(c.uncertainty, 3),
                "evidence": c.evidence_pointers,
                "falsification": c.falsification_tests,
                "source": c.source_artifact,
                "timestamp": c.created_at.isoformat(),
            }
            lines.append(json.dumps(record))
        return "\n".join(lines)

    def update_knowledge_base(
        self, workspace_path: str, claims: list[ClaimNode]
    ) -> KnowledgeBaseUpdate:
        """Write knowledge artifacts to the workspace and return an update record."""
        ws = Path(workspace_path)
        artifacts_dir = ws / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        knowledge_md_path = artifacts_dir / "KNOWLEDGE.md"
        facts_jsonl_path = artifacts_dir / "claims.jsonl"

        # Build content
        knowledge_md = self.build_knowledge_md(claims)
        facts_jsonl = self.build_facts_jsonl(claims)

        # Write files
        knowledge_md_path.write_text(knowledge_md)
        facts_jsonl_path.write_text(facts_jsonl)

        return KnowledgeBaseUpdate(
            claims=claims,
            facts_jsonl_path=str(facts_jsonl_path),
            knowledge_md_path=str(knowledge_md_path),
            summary=f"Extracted {len(claims)} claims "
                    f"({sum(1 for c in claims if c.uncertainty < 0.3)} high confidence, "
                    f"{sum(1 for c in claims if c.uncertainty >= 0.6)} low confidence)",
        )
