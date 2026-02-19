"""Knowledge base API router for persistent claims, facts, and links."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend import database as db
from backend.api.validation import validate_id

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


# ── Request/Response models ──────────────────────────────────────────

class ClaimCreate(BaseModel):
    claim_id: str
    run_id: Optional[str] = None
    claim_text: str
    evidence_pointers: list[str] = Field(default_factory=list)
    uncertainty: float = 0.5
    strength: float = 0.0
    falsification_tests: list[str] = Field(default_factory=list)
    source_artifact: str = ""


class ClaimUpdate(BaseModel):
    claim_text: Optional[str] = None
    evidence_pointers: Optional[list[str]] = None
    uncertainty: Optional[float] = None
    strength: Optional[float] = None
    falsification_tests: Optional[list[str]] = None
    source_artifact: Optional[str] = None
    status: Optional[str] = None


# ── Endpoints ────────────────────────────────────────────────────────

@router.get("/claims")
async def list_claims(
    run_id: Optional[str] = Query(default=None),
    status: str = Query(default="active"),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    if run_id is not None:
        validate_id(run_id, "run_id")
    return await db.get_knowledge_claims(run_id=run_id, status=status, limit=limit, offset=offset)


@router.get("/claims/{claim_id}")
async def get_claim(claim_id: str) -> dict:
    validate_id(claim_id, "claim_id")
    claims = await db.get_knowledge_claims()
    for c in claims:
        if c["claim_id"] == claim_id:
            facts = await db.get_knowledge_facts(claim_id=claim_id)
            links = await db.get_knowledge_links(claim_id=claim_id)
            return {**c, "facts": facts, "links": links}
    raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found")


@router.post("/claims", status_code=201)
async def create_claim(req: ClaimCreate) -> dict:
    validate_id(req.claim_id, "claim_id")
    if req.run_id is not None:
        validate_id(req.run_id, "run_id")
    return await db.create_knowledge_claim(
        claim_id=req.claim_id,
        run_id=req.run_id,
        claim_text=req.claim_text,
        evidence_pointers=req.evidence_pointers,
        uncertainty=req.uncertainty,
        strength=req.strength,
        falsification_tests=req.falsification_tests,
        source_artifact=req.source_artifact,
    )


@router.patch("/claims/{claim_id}")
async def update_claim(claim_id: str, req: ClaimUpdate) -> dict:
    validate_id(claim_id, "claim_id")
    fields = {k: v for k, v in req.model_dump().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="No fields to update")
    result = await db.update_knowledge_claim(claim_id, **fields)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Claim '{claim_id}' not found")
    return result


@router.get("/facts")
async def list_facts(
    claim_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> list[dict]:
    return await db.get_knowledge_facts(claim_id=claim_id, limit=limit, offset=offset)


@router.get("/links")
async def list_links(
    claim_id: Optional[str] = Query(default=None),
) -> list[dict]:
    return await db.get_knowledge_links(claim_id=claim_id)


@router.get("/summary")
async def knowledge_summary() -> dict:
    return await db.get_knowledge_summary()


@router.get("/graph")
async def knowledge_graph() -> dict:
    claims = await db.get_knowledge_claims()
    links = await db.get_knowledge_links()

    nodes = []
    for i, c in enumerate(claims):
        nodes.append({
            "id": c["claim_id"],
            "data": {
                "label": c["claim_text"],
                "uncertainty": c["uncertainty"],
                "strength": c["strength"],
            },
            "position": {"x": (i % 5) * 250, "y": (i // 5) * 150},
        })

    edges = []
    for link in links:
        edges.append({
            "id": link["link_id"],
            "source": link["source_claim_id"],
            "target": link["target_claim_id"],
            "label": link["link_type"],
        })

    return {"nodes": nodes, "edges": edges}
