"""Claims API router for parsing artifacts and querying claim graphs."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from backend.knowledge.parser import MechParser
from backend.knowledge.claim_graph import ClaimGraph

router = APIRouter(prefix="/claims", tags=["claims"])

# In-memory store of claim graphs per run_id
_graphs: dict[str, ClaimGraph] = {}
_parser = MechParser()


class ParseRequest(BaseModel):
    mech_content: str
    eval_content: str = ""


class ClaimSummary(BaseModel):
    id: str
    claim: str
    uncertainty: float
    evidence_strength: float
    evidence_pointers: list[str]
    source_artifact: str


def _get_graph(run_id: str) -> ClaimGraph:
    if run_id not in _graphs:
        raise HTTPException(status_code=404, detail=f"No claims found for run '{run_id}'")
    return _graphs[run_id]


def _claim_to_summary(graph: ClaimGraph, claim) -> ClaimSummary:
    return ClaimSummary(
        id=claim.id,
        claim=claim.claim,
        uncertainty=round(claim.uncertainty, 3),
        evidence_strength=round(graph.get_evidence_strength(claim.id), 3),
        evidence_pointers=claim.evidence_pointers,
        source_artifact=claim.source_artifact,
    )


@router.get("/{run_id}", response_model=list[ClaimSummary])
async def list_claims(run_id: str) -> list[ClaimSummary]:
    """List extracted claims with evidence strength for a run."""
    graph = _get_graph(run_id)
    real_claims = [
        c for c in graph._nodes.values()
        if not c.claim.startswith("[Evidence]")
    ]
    return [_claim_to_summary(graph, c) for c in real_claims]


@router.get("/{run_id}/graph")
async def get_graph(run_id: str) -> dict:
    """Return graph JSON for visualization (React Flow compatible)."""
    graph = _get_graph(run_id)
    return graph.get_graph_json()


@router.get("/{run_id}/weakest", response_model=list[ClaimSummary])
async def get_weakest(run_id: str, limit: int = Query(default=10, ge=1, le=100)) -> list[ClaimSummary]:
    """Return weakest claims sorted by evidence strength ascending."""
    graph = _get_graph(run_id)
    weak = graph.get_weakest_claims(limit=limit)
    return [_claim_to_summary(graph, c) for c in weak]


@router.get("/{run_id}/strongest", response_model=list[ClaimSummary])
async def get_strongest(run_id: str, limit: int = Query(default=10, ge=1, le=100)) -> list[ClaimSummary]:
    """Return strongest claims sorted by evidence strength descending."""
    graph = _get_graph(run_id)
    strong = graph.get_strongest_claims(limit=limit)
    return [_claim_to_summary(graph, c) for c in strong]


@router.post("/parse", response_model=dict)
async def parse_content(req: ParseRequest) -> dict:
    """Parse provided MECH.md + EVAL.md content directly.

    Returns parsed claims, evidence, and a graph JSON.
    """
    parser = MechParser()
    claims = parser.parse_mech_md(req.mech_content)
    eval_evidence = parser.parse_eval_md(req.eval_content) if req.eval_content else []

    # Build graph
    graph = ClaimGraph()
    for claim in claims:
        graph.add_claim(claim)
        strength = parser.compute_evidence_strength(claim, eval_evidence)
        if strength > 0:
            ev_id = f"ev_{claim.id}"
            graph.add_evidence_link(claim.id, ev_id, strength)

    # Link claims to evidence
    links = parser.link_claims_to_evidence(claims, eval_evidence)

    return {
        "claims": [
            {
                "id": c.id,
                "claim": c.claim,
                "uncertainty": round(c.uncertainty, 3),
                "evidence_strength": round(parser.compute_evidence_strength(c, eval_evidence), 3),
                "evidence_pointers": c.evidence_pointers,
                "source_artifact": c.source_artifact,
            }
            for c in claims
        ],
        "eval_evidence": eval_evidence,
        "graph": graph.get_graph_json(),
        "claim_evidence_links": {
            cid: [ev.get("test_name", "") for ev in evs]
            for cid, evs in links.items()
        },
    }
