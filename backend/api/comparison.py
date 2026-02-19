"""Comparison API routes for multi-run analysis."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.analysis.comparison import RunComparator
from backend.database import get_run
from backend.models import CompareRequest, RunState

router = APIRouter(prefix="/comparison", tags=["comparison"])

_comparator = RunComparator()


async def _load_runs(run_ids: list[str]) -> list[RunState]:
    """Load RunState objects for the given IDs, raising 404 on missing."""
    runs: list[RunState] = []
    for rid in run_ids:
        run = await get_run(rid)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run {rid} not found")
        runs.append(run)
    return runs


@router.post("/metrics")
async def compare_metrics(body: CompareRequest) -> dict:
    runs = await _load_runs(body.run_ids)
    return _comparator.compare_metrics(runs)


@router.post("/outputs")
async def compare_outputs(body: CompareRequest) -> dict:
    runs = await _load_runs(body.run_ids)
    # Derive artifact base paths from workspace config + run_id
    # For now, use a convention: <workspace_path>/runs/<run_id>
    artifact_base_paths = [f"runs/{r.run_id}" for r in runs]
    return _comparator.compare_outputs(runs, artifact_base_paths)


@router.post("/quality")
async def compare_quality(body: CompareRequest) -> dict:
    runs = await _load_runs(body.run_ids)
    return _comparator.compare_quality(runs)


@router.post("/summary")
async def compare_summary(body: CompareRequest) -> dict:
    runs = await _load_runs(body.run_ids)
    artifact_base_paths = [f"runs/{r.run_id}" for r in runs]
    return _comparator.generate_summary(runs, artifact_base_paths)
