"""Tests for the persistent knowledge base (DB + API)."""
from __future__ import annotations

import pytest
import pytest_asyncio
from httpx import AsyncClient

from backend import database as db
from backend.models import ProviderName, RunState
from backend.tests.conftest import client  # noqa: F401


async def _create_workspace_and_run(client: AsyncClient, ws_name: str, run_id: str) -> str:
    """Helper: create a workspace (in a temp dir, never the repo root) and a
    run, returning the run_id."""
    import tempfile
    ws_path = tempfile.mkdtemp(prefix=f"{ws_name}-")
    resp = await client.post("/api/workspaces", json={"name": ws_name, "path": ws_path})
    ws_id = resp.json()["id"]
    run = RunState(
        run_id=run_id,
        workspace_id=ws_id,
        loop_preset="executor_reviewer",
        task="test",
        provider=ProviderName.MOCK,
        model="",
    )
    await db.create_run(run)
    return run_id


# ── DB-level tests ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_create_and_get_claim(client: AsyncClient):
    await _create_workspace_and_run(client, "ws-kc1", "run-1")
    claim = await db.create_knowledge_claim(
        claim_id="kc-1",
        run_id="run-1",
        claim_text="Attention captures co-expression, not causal regulation.",
        evidence_pointers=["AUROC=0.743", "p=0.017"],
        uncertainty=0.3,
        strength=0.7,
        falsification_tests=["residualization"],
        source_artifact="MECH.md",
    )
    assert claim["claim_id"] == "kc-1"
    assert claim["claim_text"] == "Attention captures co-expression, not causal regulation."
    assert claim["uncertainty"] == 0.3
    assert claim["strength"] == 0.7
    assert claim["evidence_pointers"] == ["AUROC=0.743", "p=0.017"]
    assert claim["status"] == "active"


@pytest.mark.asyncio
async def test_list_claims_filtered_by_run(client: AsyncClient):
    await _create_workspace_and_run(client, "ws-kc-ra", "run-a")
    await _create_workspace_and_run(client, "ws-kc-rb", "run-b")
    await db.create_knowledge_claim(
        claim_id="kc-r1", run_id="run-a", claim_text="Claim for run A",
        evidence_pointers=[], uncertainty=0.5, strength=0.5,
        falsification_tests=[], source_artifact="",
    )
    await db.create_knowledge_claim(
        claim_id="kc-r2", run_id="run-b", claim_text="Claim for run B",
        evidence_pointers=[], uncertainty=0.5, strength=0.5,
        falsification_tests=[], source_artifact="",
    )
    claims_a = await db.get_knowledge_claims(run_id="run-a")
    claims_b = await db.get_knowledge_claims(run_id="run-b")
    assert all(c["run_id"] == "run-a" for c in claims_a)
    assert all(c["run_id"] == "run-b" for c in claims_b)
    assert len(claims_a) >= 1
    assert len(claims_b) >= 1


@pytest.mark.asyncio
async def test_update_claim_uncertainty(client: AsyncClient):
    await _create_workspace_and_run(client, "ws-kc-upd", "run-upd")
    await db.create_knowledge_claim(
        claim_id="kc-upd", run_id="run-upd", claim_text="Updatable claim",
        evidence_pointers=[], uncertainty=0.8, strength=0.2,
        falsification_tests=[], source_artifact="",
    )
    updated = await db.update_knowledge_claim("kc-upd", uncertainty=0.1, strength=0.9)
    assert updated is not None
    assert updated["uncertainty"] == 0.1
    assert updated["strength"] == 0.9


@pytest.mark.asyncio
async def test_create_fact_linked_to_claim(client: AsyncClient):
    await db.create_knowledge_claim(
        claim_id="kc-fact-parent", run_id=None, claim_text="Parent claim",
        evidence_pointers=[], uncertainty=0.5, strength=0.5,
        falsification_tests=[], source_artifact="",
    )
    fact = await db.create_knowledge_fact(
        fact_id="kf-1",
        claim_id="kc-fact-parent",
        fact_text="AUROC=0.743 for TRRUST edge recovery",
        source="round9_full_layer_analysis.py",
        confidence=0.9,
    )
    assert fact["fact_id"] == "kf-1"
    assert fact["claim_id"] == "kc-fact-parent"
    assert fact["confidence"] == 0.9


@pytest.mark.asyncio
async def test_get_facts_for_claim(client: AsyncClient):
    await db.create_knowledge_claim(
        claim_id="kc-facts", run_id=None, claim_text="Claim with facts",
        evidence_pointers=[], uncertainty=0.5, strength=0.5,
        falsification_tests=[], source_artifact="",
    )
    await db.create_knowledge_fact(
        fact_id="kf-a", claim_id="kc-facts", fact_text="Fact A",
        source="src_a", confidence=0.8,
    )
    await db.create_knowledge_fact(
        fact_id="kf-b", claim_id="kc-facts", fact_text="Fact B",
        source="src_b", confidence=0.6,
    )
    await db.create_knowledge_fact(
        fact_id="kf-other", claim_id=None, fact_text="Unlinked fact",
        source="", confidence=0.5,
    )
    facts = await db.get_knowledge_facts(claim_id="kc-facts")
    assert len(facts) == 2
    assert all(f["claim_id"] == "kc-facts" for f in facts)


@pytest.mark.asyncio
async def test_create_and_get_link(client: AsyncClient):
    await db.create_knowledge_claim(
        claim_id="kc-src", run_id=None, claim_text="Source claim",
        evidence_pointers=[], uncertainty=0.5, strength=0.5,
        falsification_tests=[], source_artifact="",
    )
    await db.create_knowledge_claim(
        claim_id="kc-tgt", run_id=None, claim_text="Target claim",
        evidence_pointers=[], uncertainty=0.5, strength=0.5,
        falsification_tests=[], source_artifact="",
    )
    link = await db.create_knowledge_link(
        link_id="kl-1",
        source_claim_id="kc-src",
        target_claim_id="kc-tgt",
        link_type="supports",
        weight=0.85,
    )
    assert link["link_id"] == "kl-1"
    assert link["link_type"] == "supports"
    assert link["weight"] == 0.85

    links = await db.get_knowledge_links(claim_id="kc-src")
    assert len(links) >= 1
    assert any(l["link_id"] == "kl-1" for l in links)


@pytest.mark.asyncio
async def test_knowledge_summary_counts(client: AsyncClient):
    await db.create_knowledge_claim(
        claim_id="kc-sum1", run_id=None, claim_text="Summary claim 1",
        evidence_pointers=[], uncertainty=0.4, strength=0.6,
        falsification_tests=[], source_artifact="",
    )
    await db.create_knowledge_claim(
        claim_id="kc-sum2", run_id=None, claim_text="Summary claim 2",
        evidence_pointers=[], uncertainty=0.6, strength=0.8,
        falsification_tests=[], source_artifact="",
    )
    await db.create_knowledge_fact(
        fact_id="kf-sum", claim_id="kc-sum1", fact_text="A fact",
        source="", confidence=0.5,
    )
    summary = await db.get_knowledge_summary()
    assert summary["total_claims"] >= 2
    assert summary["total_facts"] >= 1
    assert 0.0 <= summary["avg_uncertainty"] <= 1.0
    assert 0.0 <= summary["avg_strength"] <= 1.0


# ── Knowledge graph format test ──────────────────────────────────────

@pytest.mark.asyncio
async def test_knowledge_graph_format(client: AsyncClient):
    await db.create_knowledge_claim(
        claim_id="kc-g1", run_id=None, claim_text="Graph claim 1",
        evidence_pointers=[], uncertainty=0.3, strength=0.7,
        falsification_tests=[], source_artifact="",
    )
    await db.create_knowledge_claim(
        claim_id="kc-g2", run_id=None, claim_text="Graph claim 2",
        evidence_pointers=[], uncertainty=0.5, strength=0.5,
        falsification_tests=[], source_artifact="",
    )
    await db.create_knowledge_link(
        link_id="kl-g1", source_claim_id="kc-g1", target_claim_id="kc-g2",
        link_type="contradicts", weight=0.9,
    )
    resp = await client.get("/api/knowledge/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    # Verify node structure
    for node in data["nodes"]:
        assert "id" in node
        assert "data" in node
        assert "label" in node["data"]
        assert "uncertainty" in node["data"]
        assert "strength" in node["data"]
        assert "position" in node
        assert "x" in node["position"]
        assert "y" in node["position"]
    # Verify edge structure
    for edge in data["edges"]:
        assert "id" in edge
        assert "source" in edge
        assert "target" in edge
        assert "label" in edge


# ── API endpoint tests ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_api_claims_endpoint(client: AsyncClient):
    # Create a workspace and run so the FK is satisfied
    await _create_workspace_and_run(client, "ws-kc-api", "run-api")
    # Create via API
    resp = await client.post("/api/knowledge/claims", json={
        "claim_id": "kc-api1",
        "run_id": "run-api",
        "claim_text": "API-created claim",
        "evidence_pointers": ["pointer1"],
        "uncertainty": 0.4,
        "strength": 0.6,
    })
    assert resp.status_code == 201
    data = resp.json()
    assert data["claim_id"] == "kc-api1"
    assert data["claim_text"] == "API-created claim"

    # List via API
    resp = await client.get("/api/knowledge/claims", params={"run_id": "run-api"})
    assert resp.status_code == 200
    claims = resp.json()
    assert any(c["claim_id"] == "kc-api1" for c in claims)

    # Get single via API
    resp = await client.get("/api/knowledge/claims/kc-api1")
    assert resp.status_code == 200
    claim = resp.json()
    assert claim["claim_id"] == "kc-api1"
    assert "facts" in claim
    assert "links" in claim

    # Update via API
    resp = await client.patch("/api/knowledge/claims/kc-api1", json={
        "uncertainty": 0.1,
    })
    assert resp.status_code == 200
    assert resp.json()["uncertainty"] == 0.1


@pytest.mark.asyncio
async def test_api_graph_endpoint(client: AsyncClient):
    # Create two claims and a link
    await client.post("/api/knowledge/claims", json={
        "claim_id": "kc-ag1", "claim_text": "Graph node 1",
    })
    await client.post("/api/knowledge/claims", json={
        "claim_id": "kc-ag2", "claim_text": "Graph node 2",
    })
    await db.create_knowledge_link(
        link_id="kl-ag1", source_claim_id="kc-ag1", target_claim_id="kc-ag2",
        link_type="supports", weight=1.0,
    )
    resp = await client.get("/api/knowledge/graph")
    assert resp.status_code == 200
    data = resp.json()
    node_ids = {n["id"] for n in data["nodes"]}
    assert "kc-ag1" in node_ids
    assert "kc-ag2" in node_ids
    edge_ids = {e["id"] for e in data["edges"]}
    assert "kl-ag1" in edge_ids

    # Summary endpoint
    resp = await client.get("/api/knowledge/summary")
    assert resp.status_code == 200
    summary = resp.json()
    assert "total_claims" in summary
    assert summary["total_claims"] >= 2
