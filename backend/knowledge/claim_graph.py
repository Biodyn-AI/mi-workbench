"""
In-memory claim graph for tracking relationships between extracted knowledge claims.
Supports adding claims, querying weak edges, and exporting to JSON.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from backend.models import ClaimNode


@dataclass
class GraphEdge:
    """An edge in the claim graph representing a relationship between two claims."""
    source_id: str
    target_id: str
    relation: str  # "supports", "contradicts", "depends_on", "generalizes", "refines"
    weight: float = 1.0
    notes: str = ""


class ClaimGraph:
    """In-memory directed graph of claims and their relationships."""

    def __init__(self) -> None:
        self._nodes: dict[str, ClaimNode] = {}
        self._edges: list[GraphEdge] = []
        self._adjacency: dict[str, list[str]] = {}  # source_id -> [target_ids]

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def add_claim(self, claim: ClaimNode) -> None:
        """Add a claim node to the graph."""
        self._nodes[claim.id] = claim
        if claim.id not in self._adjacency:
            self._adjacency[claim.id] = []

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        notes: str = "",
    ) -> None:
        """Add a directed edge between two claims."""
        if source_id not in self._nodes:
            raise ValueError(f"Source claim '{source_id}' not in graph")
        if target_id not in self._nodes:
            raise ValueError(f"Target claim '{target_id}' not in graph")
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
            notes=notes,
        )
        self._edges.append(edge)
        self._adjacency.setdefault(source_id, []).append(target_id)

    def get_claim(self, claim_id: str) -> Optional[ClaimNode]:
        """Retrieve a claim by ID."""
        return self._nodes.get(claim_id)

    def get_neighbors(self, claim_id: str) -> list[str]:
        """Get IDs of claims connected from this claim."""
        return self._adjacency.get(claim_id, [])

    def get_edges_for(self, claim_id: str) -> list[GraphEdge]:
        """Get all edges originating from a claim."""
        return [e for e in self._edges if e.source_id == claim_id]

    def get_edges_to(self, claim_id: str) -> list[GraphEdge]:
        """Get all edges targeting a claim."""
        return [e for e in self._edges if e.target_id == claim_id]

    def get_weak_edges(self, uncertainty_threshold: float = 0.6) -> list[ClaimNode]:
        """Return claims with uncertainty above the threshold (weak claims)."""
        return [
            c for c in self._nodes.values()
            if c.uncertainty >= uncertainty_threshold
        ]

    def get_contradictions(self) -> list[tuple[ClaimNode, ClaimNode, GraphEdge]]:
        """Return all pairs of claims linked by a 'contradicts' edge."""
        result = []
        for edge in self._edges:
            if edge.relation == "contradicts":
                src = self._nodes.get(edge.source_id)
                tgt = self._nodes.get(edge.target_id)
                if src and tgt:
                    result.append((src, tgt, edge))
        return result

    def get_unsupported_claims(self) -> list[ClaimNode]:
        """Return claims that have no incoming 'supports' edges and high uncertainty."""
        supported_ids = {e.target_id for e in self._edges if e.relation == "supports"}
        return [
            c for c in self._nodes.values()
            if c.id not in supported_ids and c.uncertainty >= 0.5
        ]

    def export_json(self) -> dict:
        """Export the full graph as a JSON-serializable dict."""
        nodes = []
        for c in self._nodes.values():
            nodes.append({
                "id": c.id,
                "claim": c.claim,
                "uncertainty": round(c.uncertainty, 3),
                "evidence_pointers": c.evidence_pointers,
                "falsification_tests": c.falsification_tests,
                "source_artifact": c.source_artifact,
                "created_at": c.created_at.isoformat(),
            })
        edges = []
        for e in self._edges:
            edges.append({
                "source": e.source_id,
                "target": e.target_id,
                "relation": e.relation,
                "weight": round(e.weight, 3),
                "notes": e.notes,
            })
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "weak_claims": len(self.get_weak_edges()),
                "contradictions": len(self.get_contradictions()),
            },
        }

    def add_evidence_link(self, claim_id: str, evidence_id: str, strength: float) -> None:
        """Add an evidence link to a claim (as a special edge to an evidence node).

        Creates a synthetic evidence node if needed, then adds a 'supported_by' edge.
        """
        if claim_id not in self._nodes:
            raise ValueError(f"Claim '{claim_id}' not in graph")
        # Create evidence pseudo-node if not present
        if evidence_id not in self._nodes:
            ev_node = ClaimNode(
                id=evidence_id,
                claim=f"[Evidence] {evidence_id}",
                uncertainty=0.0,
                source_artifact="EVAL.md",
            )
            self.add_claim(ev_node)
        edge = GraphEdge(
            source_id=evidence_id,
            target_id=claim_id,
            relation="supported_by",
            weight=strength,
        )
        self._edges.append(edge)
        self._adjacency.setdefault(evidence_id, []).append(claim_id)
        # Store strength on the claim node metadata via evidence_pointers
        claim = self._nodes[claim_id]
        if not hasattr(claim, "_evidence_strength"):
            self._evidence_strengths: dict[str, float] = getattr(self, "_evidence_strengths", {})
        self._evidence_strengths = getattr(self, "_evidence_strengths", {})
        self._evidence_strengths[claim_id] = max(
            self._evidence_strengths.get(claim_id, 0.0), strength
        )

    def get_evidence_strength(self, claim_id: str) -> float:
        """Get the evidence strength for a claim (0 if no evidence linked)."""
        strengths = getattr(self, "_evidence_strengths", {})
        return strengths.get(claim_id, 0.0)

    def get_graph_json(self) -> dict:
        """Return {nodes, edges, metadata} suitable for React Flow visualization."""
        nodes = []
        strengths = getattr(self, "_evidence_strengths", {})
        for c in self._nodes.values():
            nodes.append({
                "id": c.id,
                "type": "evidence" if c.claim.startswith("[Evidence]") else "claim",
                "data": {
                    "label": c.claim,
                    "uncertainty": round(c.uncertainty, 3),
                    "evidence_strength": round(strengths.get(c.id, 0.0), 3),
                    "evidence_pointers": c.evidence_pointers,
                    "falsification_tests": c.falsification_tests,
                    "source_artifact": c.source_artifact,
                },
                "position": {"x": 0, "y": 0},  # layout computed client-side
            })
        edges = []
        for i, e in enumerate(self._edges):
            edges.append({
                "id": f"e{i}",
                "source": e.source_id,
                "target": e.target_id,
                "label": e.relation,
                "data": {
                    "weight": round(e.weight, 3),
                    "notes": e.notes,
                },
            })
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "node_count": len([n for n in nodes if n["type"] == "claim"]),
                "evidence_count": len([n for n in nodes if n["type"] == "evidence"]),
                "edge_count": len(edges),
                "weak_claims": len(self.get_weak_edges()),
                "contradictions": len(self.get_contradictions()),
                "overall_confidence": round(self.compute_overall_confidence(), 3),
            },
        }

    def get_weakest_claims(self, limit: int = 10) -> list[ClaimNode]:
        """Return claims sorted by evidence strength ascending (weakest first).

        Only returns actual claim nodes (not evidence pseudo-nodes).
        """
        strengths = getattr(self, "_evidence_strengths", {})
        real_claims = [
            c for c in self._nodes.values()
            if not c.claim.startswith("[Evidence]")
        ]
        return sorted(real_claims, key=lambda c: strengths.get(c.id, 0.0))[:limit]

    def get_strongest_claims(self, limit: int = 10) -> list[ClaimNode]:
        """Return top claims sorted by evidence strength descending."""
        strengths = getattr(self, "_evidence_strengths", {})
        real_claims = [
            c for c in self._nodes.values()
            if not c.claim.startswith("[Evidence]")
        ]
        return sorted(real_claims, key=lambda c: strengths.get(c.id, 0.0), reverse=True)[:limit]

    def compute_overall_confidence(self) -> float:
        """Return aggregate confidence score across all claims (0-1).

        Averages (1 - uncertainty) weighted by evidence strength.
        Falls back to simple mean of (1 - uncertainty) if no evidence linked.
        """
        strengths = getattr(self, "_evidence_strengths", {})
        real_claims = [
            c for c in self._nodes.values()
            if not c.claim.startswith("[Evidence]")
        ]
        if not real_claims:
            return 0.0
        if not strengths:
            return sum(1.0 - c.uncertainty for c in real_claims) / len(real_claims)
        total_weight = 0.0
        weighted_sum = 0.0
        for c in real_claims:
            w = strengths.get(c.id, 0.1)  # small base weight for unlinked claims
            weighted_sum += w * (1.0 - c.uncertainty)
            total_weight += w
        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def merge(self, other: ClaimGraph) -> None:
        """Merge another graph into this one, deduplicating by claim ID."""
        for claim_id, claim in other._nodes.items():
            if claim_id not in self._nodes:
                self.add_claim(claim)
        for edge in other._edges:
            if edge.source_id in self._nodes and edge.target_id in self._nodes:
                # Check for duplicate edges
                existing = any(
                    e.source_id == edge.source_id
                    and e.target_id == edge.target_id
                    and e.relation == edge.relation
                    for e in self._edges
                )
                if not existing:
                    self._edges.append(edge)
                    self._adjacency.setdefault(edge.source_id, []).append(edge.target_id)
