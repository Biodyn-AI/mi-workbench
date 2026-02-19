"""Tests for the Claim Graph parser and API."""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.knowledge.parser import MechParser
from backend.knowledge.claim_graph import ClaimGraph
from backend.models import ClaimNode

# ── Realistic test content ────────────────────────────────────────────

SAMPLE_MECH_MD = """\
# Attention–GRN Recovery Analysis

## Layer Selection
We found that attention weights in layer 15 of Geneformer V2-316M achieve AUROC=0.743 for TRRUST edge recovery (p=0.017, bootstrap CI [0.707, 0.770]).

## Co-expression Confound
Results show that this signal is largely attributable to co-expression (residualization removes ~76% of signal, p<0.001).

Claim 1: Attention captures co-expression structure rather than causal regulatory relationships.
Finding 2: The incremental value of attention over gene-level features is negligible (AUROC delta = -0.0004).

## Cross-context Validation
This demonstrates that attention-based GRN inference does not generalize to CRISPRa perturbation data (AUROC=0.547 vs correlation 0.653, p<0.000001, n=77).

Evidence suggests that trivial baselines such as expression variance (AUROC=0.881) significantly outperform attention-derived networks.

## Head Ablation
Result: Ablating heads identified by TRRUST recovery does not causally affect perturbation prediction (p=0.24).
"""

SAMPLE_EVAL_MD = """\
# Statistical Tests

## Layer 15 TRRUST Recovery
AUROC=0.743, p=0.017, bootstrap CI [0.707, 0.770], n=280

## Co-expression Residualization
Residualized AUROC=0.52, p<0.001, original AUROC=0.743

## Adamson CRISPRa Cross-context
Attention AUROC=0.547, correlation AUROC=0.653, p<0.000001, n=77

## Incremental Value
Gene-only AUROC=0.895, attention delta=-0.0004, CI [-0.001, 0.000]

## Trivial Baselines
Variance AUROC=0.881, mean AUROC=0.841, dropout AUROC=0.808

## Head Ablation
Baseline=0.7035, ablate_regulatory=0.7037, p=0.24
"""


# ── Parser unit tests ─────────────────────────────────────────────────

class TestParserExplicitClaims:
    def test_parse_mech_finds_explicit_claims(self):
        parser = MechParser()
        claims = parser.parse_mech_md(SAMPLE_MECH_MD)
        claim_texts = [c.claim for c in claims]
        # Should find "Claim 1:" and "Finding 2:" and "Result:" markers
        assert any("co-expression structure" in t for t in claim_texts), (
            f"Expected explicit claim about co-expression; got: {claim_texts}"
        )
        assert any("incremental value" in t.lower() for t in claim_texts), (
            f"Expected explicit finding about incremental value; got: {claim_texts}"
        )
        assert any("causally" in t.lower() or "ablat" in t.lower() for t in claim_texts), (
            f"Expected explicit result about head ablation; got: {claim_texts}"
        )

    def test_parse_mech_finds_implicit_claims(self):
        parser = MechParser()
        claims = parser.parse_mech_md(SAMPLE_MECH_MD)
        claim_texts = [c.claim for c in claims]
        # "We found that..." and "Results show that..." should be captured
        assert any("AUROC=0.743" in t or "layer 15" in t.lower() for t in claim_texts), (
            f"Expected implicit claim from 'we found that'; got: {claim_texts}"
        )
        assert any("residualization" in t.lower() or "co-expression" in t.lower() for t in claim_texts), (
            f"Expected implicit claim from 'results show that'; got: {claim_texts}"
        )

    def test_claims_have_evidence_pointers(self):
        parser = MechParser()
        claims = parser.parse_mech_md(SAMPLE_MECH_MD)
        # Every claim should have at least a source line reference
        for c in claims:
            assert len(c.evidence_pointers) > 0, f"Claim '{c.claim[:40]}' has no evidence pointers"
            assert any("MECH.md:L" in p for p in c.evidence_pointers), (
                f"Claim missing source line reference: {c.evidence_pointers}"
            )

    def test_claims_have_section_context(self):
        parser = MechParser()
        claims = parser.parse_mech_md(SAMPLE_MECH_MD)
        # At least some claims should reference their section
        sections_found = set()
        for c in claims:
            for p in c.evidence_pointers:
                if p.startswith("Section:"):
                    sections_found.add(p)
        assert len(sections_found) > 0, "No section context found in any claim"


class TestParserEval:
    def test_parse_eval_extracts_statistics(self):
        parser = MechParser()
        evidence = parser.parse_eval_md(SAMPLE_EVAL_MD)
        assert len(evidence) >= 4, f"Expected at least 4 evidence entries, got {len(evidence)}"
        # Check we extracted AUROC values
        stats = [e["statistic"] for e in evidence if e["statistic"]]
        assert any("0.743" in s for s in stats), f"Missing AUROC=0.743; got: {stats}"

    def test_parse_eval_extracts_p_values(self):
        parser = MechParser()
        evidence = parser.parse_eval_md(SAMPLE_EVAL_MD)
        p_values = [e["p_value"] for e in evidence if e["p_value"] is not None]
        assert len(p_values) >= 3, f"Expected at least 3 p-values, got {len(p_values)}"

    def test_parse_eval_extracts_confidence_intervals(self):
        parser = MechParser()
        evidence = parser.parse_eval_md(SAMPLE_EVAL_MD)
        cis = [e["ci"] for e in evidence if e["ci"] is not None]
        assert len(cis) >= 1, "Expected at least one confidence interval"
        # Check the values
        assert cis[0] == [0.707, 0.770] or cis[0] == [0.707, 0.77]

    def test_parse_eval_has_interpretations(self):
        parser = MechParser()
        evidence = parser.parse_eval_md(SAMPLE_EVAL_MD)
        interps = [e["interpretation"] for e in evidence if e["interpretation"]]
        assert any("significant" in i.lower() for i in interps), (
            f"Expected significant interpretation; got: {interps}"
        )


class TestEvidenceStrength:
    def test_evidence_strength_scoring(self):
        parser = MechParser()
        claims = parser.parse_mech_md(SAMPLE_MECH_MD)
        evidence = parser.parse_eval_md(SAMPLE_EVAL_MD)

        for claim in claims:
            strength = parser.compute_evidence_strength(claim, evidence)
            assert 0.0 <= strength <= 1.0, f"Strength out of range: {strength}"

    def test_claims_with_pvalues_score_higher(self):
        parser = MechParser()
        evidence = parser.parse_eval_md(SAMPLE_EVAL_MD)

        strong_claim = ClaimNode(
            id="test_strong",
            claim="We found significantly different AUROC=0.743, p<0.001, CI [0.707, 0.770], negative control baseline included",
            evidence_pointers=["AUROC=0.743", "p<0.001", "CI=[0.707, 0.770]", "MECH.md:L5"],
            uncertainty=0.2,
            source_artifact="MECH.md",
        )
        weak_claim = ClaimNode(
            id="test_weak",
            claim="This may suggest a trend",
            evidence_pointers=["MECH.md:L10"],
            uncertainty=0.7,
            source_artifact="MECH.md",
        )

        strong_score = parser.compute_evidence_strength(strong_claim, evidence)
        weak_score = parser.compute_evidence_strength(weak_claim, evidence)
        assert strong_score > weak_score, (
            f"Strong claim ({strong_score}) should score higher than weak ({weak_score})"
        )


class TestLinkClaimsToEvidence:
    def test_link_claims_to_evidence(self):
        parser = MechParser()
        claims = parser.parse_mech_md(SAMPLE_MECH_MD)
        evidence = parser.parse_eval_md(SAMPLE_EVAL_MD)
        links = parser.link_claims_to_evidence(claims, evidence)

        assert isinstance(links, dict)
        # At least some claims should link to evidence
        linked_count = sum(1 for evs in links.values() if len(evs) > 0)
        assert linked_count > 0, "No claims linked to evidence"


# ── ClaimGraph unit tests ─────────────────────────────────────────────

class TestClaimGraph:
    def _build_graph(self) -> ClaimGraph:
        parser = MechParser()
        claims = parser.parse_mech_md(SAMPLE_MECH_MD)
        evidence = parser.parse_eval_md(SAMPLE_EVAL_MD)

        graph = ClaimGraph()
        for claim in claims:
            graph.add_claim(claim)
            strength = parser.compute_evidence_strength(claim, evidence)
            if strength > 0:
                ev_id = f"ev_{claim.id}"
                graph.add_evidence_link(claim.id, ev_id, strength)
        return graph

    def test_claim_graph_nodes_and_edges(self):
        graph = self._build_graph()
        assert graph.node_count > 0, "Graph should have nodes"
        assert graph.edge_count > 0, "Graph should have edges"

    def test_graph_json_format(self):
        graph = self._build_graph()
        gj = graph.get_graph_json()

        assert "nodes" in gj
        assert "edges" in gj
        assert "metadata" in gj

        # React Flow format: nodes have id, type, data, position
        for node in gj["nodes"]:
            assert "id" in node
            assert "type" in node
            assert node["type"] in ("claim", "evidence")
            assert "data" in node
            assert "position" in node
            assert "x" in node["position"]
            assert "y" in node["position"]

        # Edges have id, source, target, label
        for edge in gj["edges"]:
            assert "id" in edge
            assert "source" in edge
            assert "target" in edge
            assert "label" in edge

        # Metadata
        assert "node_count" in gj["metadata"]
        assert "overall_confidence" in gj["metadata"]

    def test_weakest_claims_sorted(self):
        graph = self._build_graph()
        weak = graph.get_weakest_claims(limit=5)
        assert len(weak) > 0
        # Verify ascending order by evidence strength
        strengths = [graph.get_evidence_strength(c.id) for c in weak]
        assert strengths == sorted(strengths), f"Not sorted ascending: {strengths}"

    def test_strongest_claims_sorted(self):
        graph = self._build_graph()
        strong = graph.get_strongest_claims(limit=5)
        assert len(strong) > 0
        # Verify descending order by evidence strength
        strengths = [graph.get_evidence_strength(c.id) for c in strong]
        assert strengths == sorted(strengths, reverse=True), f"Not sorted descending: {strengths}"

    def test_overall_confidence_in_range(self):
        graph = self._build_graph()
        conf = graph.compute_overall_confidence()
        assert 0.0 <= conf <= 1.0, f"Overall confidence out of range: {conf}"

    def test_add_evidence_link(self):
        graph = ClaimGraph()
        claim = ClaimNode(id="c1", claim="Test claim", uncertainty=0.4)
        graph.add_claim(claim)
        graph.add_evidence_link("c1", "ev_1", 0.8)
        assert graph.get_evidence_strength("c1") == 0.8
        # Evidence pseudo-node should exist
        ev_node = graph.get_claim("ev_1")
        assert ev_node is not None
        assert ev_node.claim.startswith("[Evidence]")

    def test_evidence_link_invalid_claim_raises(self):
        graph = ClaimGraph()
        with pytest.raises(ValueError, match="not in graph"):
            graph.add_evidence_link("nonexistent", "ev_1", 0.5)


# ── API tests ─────────────────────────────────────────────────────────

# Import conftest fixtures for async client
from backend.tests.conftest import client  # noqa: F401


class TestClaimsAPI:
    @pytest.mark.asyncio
    async def test_api_parse_endpoint(self, client: AsyncClient):
        resp = await client.post(
            "/api/claims/parse",
            json={
                "mech_content": SAMPLE_MECH_MD,
                "eval_content": SAMPLE_EVAL_MD,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "claims" in data
        assert "eval_evidence" in data
        assert "graph" in data
        assert len(data["claims"]) > 0
        # Each claim should have required fields
        for c in data["claims"]:
            assert "id" in c
            assert "claim" in c
            assert "uncertainty" in c
            assert "evidence_strength" in c

    @pytest.mark.asyncio
    async def test_api_parse_mech_only(self, client: AsyncClient):
        resp = await client.post(
            "/api/claims/parse",
            json={"mech_content": SAMPLE_MECH_MD},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["claims"]) > 0
        assert data["eval_evidence"] == []

    @pytest.mark.asyncio
    async def test_api_get_nonexistent_run(self, client: AsyncClient):
        resp = await client.get("/api/claims/no_such_run")
        assert resp.status_code == 404
