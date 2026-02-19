"""Parse MECH.md and EVAL.md artifacts to extract structured claims with evidence."""
from __future__ import annotations

import re
from typing import Optional

from backend.models import ClaimNode


class MechParser:
    """Parse mechanistic interpretability artifacts for claims."""

    # Explicit claim markers
    EXPLICIT_PATTERNS = [
        re.compile(r"(?:Claim|Finding|Result)\s*\d*\s*:\s*(.+?)(?:\n|$)", re.IGNORECASE),
    ]

    # Implicit claim language
    IMPLICIT_PATTERNS = [
        re.compile(r"(?:[Ww]e (?:found|observed|show|demonstrate) that)\s+(.+?)(?:\.\s|$)"),
        re.compile(r"(?:[Rr]esults? (?:show|indicate|suggest|confirm) that)\s+(.+?)(?:\.\s|$)"),
        re.compile(r"(?:[Tt]his (?:indicates|suggests|confirms|demonstrates) that)\s+(.+?)(?:\.\s|$)"),
        re.compile(r"(?:[Ee]vidence suggests)\s+(.+?)(?:\.\s|$)"),
    ]

    # Evidence references
    AUROC_PATTERN = re.compile(r"AUROC\s*[=~]\s*([\d.]+)")
    PVALUE_PATTERN = re.compile(r"p\s*[<>=]\s*([\d.eE\-^]+)")
    CI_PATTERN = re.compile(r"(?:CI|confidence interval)\s*\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?", re.IGNORECASE)
    N_PATTERN = re.compile(r"n\s*=\s*([\d,]+)")

    # Hedging language (increases uncertainty)
    HEDGE_WORDS = ["suggests", "may", "possibly", "marginally", "trend", "appears to", "seems"]
    # Strong language (decreases uncertainty)
    STRONG_WORDS = ["significantly", "demonstrates", "confirms", "strongly"]

    def __init__(self) -> None:
        self._claim_counter = 0

    def _next_id(self) -> str:
        self._claim_counter += 1
        return f"claim_{self._claim_counter:04d}"

    def _get_current_section(self, lines: list[str], line_idx: int) -> str:
        """Walk backwards to find the nearest section header."""
        for i in range(line_idx, -1, -1):
            stripped = lines[i].strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return ""

    def _extract_evidence_pointers(self, text: str) -> list[str]:
        """Extract references to statistics, figures, data from text."""
        pointers = []
        for m in self.AUROC_PATTERN.finditer(text):
            pointers.append(f"AUROC={m.group(1)}")
        for m in self.PVALUE_PATTERN.finditer(text):
            pointers.append(f"p={m.group(1)}")
        for m in self.CI_PATTERN.finditer(text):
            pointers.append(f"CI=[{m.group(1)}, {m.group(2)}]")
        for m in self.N_PATTERN.finditer(text):
            pointers.append(f"n={m.group(1)}")
        # Figure/table references
        for m in re.finditer(r"(?:Figure|Fig\.|Table)\s*(\d+)", text, re.IGNORECASE):
            pointers.append(f"Figure {m.group(1)}")
        return pointers

    def _estimate_uncertainty(self, text: str) -> float:
        """Estimate uncertainty (0=certain, 1=highly uncertain) from claim text."""
        uncertainty = 0.5
        text_lower = text.lower()
        # Strong p-values lower uncertainty
        if re.search(r"p\s*<\s*0\.001", text):
            uncertainty -= 0.2
        elif re.search(r"p\s*<\s*0\.01", text):
            uncertainty -= 0.15
        elif re.search(r"p\s*<\s*0\.05", text):
            uncertainty -= 0.1
        # CIs lower uncertainty
        if self.CI_PATTERN.search(text):
            uncertainty -= 0.1
        # Hedging increases
        for hedge in self.HEDGE_WORDS:
            if hedge in text_lower:
                uncertainty += 0.1
        # Strong language decreases
        for strong in self.STRONG_WORDS:
            if strong in text_lower:
                uncertainty -= 0.05
        return max(0.05, min(0.95, round(uncertainty, 3)))

    def _extract_falsification(self, text: str, context_lines: list[str], line_idx: int) -> list[str]:
        """Extract falsification tests if present, looking at nearby lines too."""
        tests = []
        search_range = context_lines[max(0, line_idx - 2):min(len(context_lines), line_idx + 3)]
        for line in search_range:
            lower = line.lower()
            if "what would change" in lower or "falsif" in lower or "would be refuted" in lower:
                tests.append(line.strip().lstrip("- "))
        return tests

    def parse_mech_md(self, content: str) -> list[ClaimNode]:
        """Extract claims from MECH.md content."""
        claims: list[ClaimNode] = []
        lines = content.split("\n")
        seen_texts: set[str] = set()

        def _add_claim(text: str, source_line: int) -> None:
            text = text.strip().rstrip(".")
            if len(text) < 15 or text in seen_texts:
                return
            seen_texts.add(text)
            section = self._get_current_section(lines, source_line)
            evidence = self._extract_evidence_pointers(text)
            if section:
                evidence.append(f"Section: {section}")
            evidence.append(f"MECH.md:L{source_line + 1}")
            claims.append(ClaimNode(
                id=self._next_id(),
                claim=text,
                evidence_pointers=evidence,
                uncertainty=self._estimate_uncertainty(text),
                falsification_tests=self._extract_falsification(text, lines, source_line),
                source_artifact="MECH.md",
            ))

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # Explicit markers
            for pat in self.EXPLICIT_PATTERNS:
                m = pat.search(stripped)
                if m:
                    _add_claim(m.group(1), i)

            # Implicit patterns
            for pat in self.IMPLICIT_PATTERNS:
                m = pat.search(stripped)
                if m:
                    _add_claim(m.group(1), i)

            # Sentences with strong statistical language that aren't already captured
            if any(kw in stripped for kw in ["AUROC", "significantly", "p <", "p ="]):
                has_claim_lang = any(
                    kw in stripped.lower()
                    for kw in ["found", "show", "demonstrate", "suggest", "indicate", "evidence"]
                )
                if has_claim_lang:
                    _add_claim(stripped, i)

        return claims

    def parse_eval_md(self, content: str) -> list[dict]:
        """Extract statistical evidence from EVAL.md.

        Returns list of {test_name, statistic, p_value, ci, interpretation}.
        """
        evidence: list[dict] = []
        lines = content.split("\n")
        current_section = ""

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#"):
                current_section = stripped.lstrip("#").strip()
                continue
            if not stripped:
                continue

            entry: dict = {"test_name": "", "statistic": None, "p_value": None, "ci": None, "interpretation": ""}

            # Extract AUROC or other test statistics
            auroc_m = self.AUROC_PATTERN.search(stripped)
            if auroc_m:
                entry["statistic"] = f"AUROC={auroc_m.group(1)}"

            # Extract p-value
            p_m = self.PVALUE_PATTERN.search(stripped)
            if p_m:
                raw = p_m.group(1)
                try:
                    entry["p_value"] = float(raw.replace("^", "e"))
                except ValueError:
                    entry["p_value"] = raw

            # Extract CI
            ci_m = self.CI_PATTERN.search(stripped)
            if ci_m:
                entry["ci"] = [float(ci_m.group(1)), float(ci_m.group(2))]

            # Only keep lines with actual statistical content
            if entry["statistic"] is None and entry["p_value"] is None and entry["ci"] is None:
                continue

            entry["test_name"] = current_section or f"Test at line {i + 1}"
            # Build interpretation
            if entry["p_value"] is not None:
                try:
                    pv = float(entry["p_value"])
                    if pv < 0.001:
                        entry["interpretation"] = "Highly significant"
                    elif pv < 0.01:
                        entry["interpretation"] = "Significant"
                    elif pv < 0.05:
                        entry["interpretation"] = "Marginally significant"
                    else:
                        entry["interpretation"] = "Not significant"
                except (ValueError, TypeError):
                    entry["interpretation"] = "Significant (exact p unavailable)"
            elif entry["statistic"]:
                entry["interpretation"] = "Statistic reported"

            evidence.append(entry)

        return evidence

    def compute_evidence_strength(self, claim: ClaimNode, eval_evidence: list[dict]) -> float:
        """Score evidence strength for a claim (0-1).

        Factors:
        - Number of evidence pointers (more = stronger)
        - Statistical significance of referenced tests
        - Whether negative controls exist
        - Whether replication is mentioned
        - Whether alternative explanations are addressed
        """
        score = 0.0

        # Base score from evidence pointers count (up to 0.3)
        n_pointers = len(claim.evidence_pointers)
        score += min(0.3, n_pointers * 0.06)

        # Match claim to eval evidence and score significance (up to 0.3)
        linked = self.link_claims_to_evidence([claim], eval_evidence)
        linked_evidence = linked.get(claim.id, [])
        if linked_evidence:
            best_p = 1.0
            for ev in linked_evidence:
                if ev.get("p_value") is not None:
                    try:
                        pv = float(ev["p_value"])
                        best_p = min(best_p, pv)
                    except (ValueError, TypeError):
                        pass
            if best_p < 0.001:
                score += 0.3
            elif best_p < 0.01:
                score += 0.25
            elif best_p < 0.05:
                score += 0.2
            elif linked_evidence:
                score += 0.1  # Has linked evidence but not significant

        # Replication mentioned (up to 0.15)
        claim_lower = claim.claim.lower()
        for kw in ["replicate", "split-sample", "cross-validation", "independent dataset"]:
            if kw in claim_lower:
                score += 0.15
                break

        # Negative controls (up to 0.15)
        for kw in ["negative control", "null model", "permutation", "baseline"]:
            if kw in claim_lower:
                score += 0.15
                break

        # Alternative explanations addressed (up to 0.1)
        for kw in ["alternative", "confound", "residuali"]:
            if kw in claim_lower:
                score += 0.1
                break

        return min(1.0, round(score, 3))

    def link_claims_to_evidence(
        self, claims: list[ClaimNode], evidence: list[dict]
    ) -> dict[str, list[dict]]:
        """Create claim -> evidence links based on keyword matching.

        Returns {claim_id: [matching evidence entries]}.
        """
        links: dict[str, list[dict]] = {}

        for claim in claims:
            claim_keywords = set()
            claim_lower = claim.claim.lower()
            # Extract keywords from claim: significant words > 3 chars
            for word in re.findall(r"[a-zA-Z]{4,}", claim.claim):
                claim_keywords.add(word.lower())
            # Also extract numeric values as keywords
            for num in re.findall(r"\d+\.?\d*", claim.claim):
                claim_keywords.add(num)

            matched = []
            for ev in evidence:
                # Build searchable text from evidence entry
                ev_text = " ".join(str(v) for v in ev.values() if v is not None).lower()
                # Count keyword overlaps
                overlap = sum(1 for kw in claim_keywords if kw in ev_text)
                if overlap >= 2:
                    matched.append(ev)

            links[claim.id] = matched

        return links
