"""Tests for multi-run comparison feature."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from httpx import AsyncClient

from backend.analysis.comparison import RunComparator
from backend.models import (
    IterationResult,
    IterationStatus,
    ProviderName,
    RunState,
    RunStatus,
)

pytestmark = pytest.mark.asyncio

# ── Fixtures ─────────────────────────────────────────────────────────


def _make_run(
    run_id: str,
    iterations: int = 5,
    tokens: int = 1000,
    cost: float = 0.5,
    status: RunStatus = RunStatus.COMPLETED,
    duration_s: float = 60.0,
    reviewer_feedback: str | None = None,
) -> RunState:
    started = datetime(2025, 1, 1, 12, 0, 0)
    stopped = started + timedelta(seconds=duration_s)
    iters = []
    for i in range(iterations):
        role = "reviewer" if i % 2 == 1 else "executor"
        iters.append(
            IterationResult(
                iteration_number=i + 1,
                role=role,
                status=IterationStatus.COMPLETED,
                feedback=None,
            )
        )
    # Attach reviewer feedback to the last reviewer iteration
    if reviewer_feedback:
        for it in reversed(iters):
            if it.role == "reviewer":
                it.feedback = reviewer_feedback
                break
    return RunState(
        run_id=run_id,
        workspace_id="ws1",
        loop_preset="executor_reviewer",
        task="test task",
        provider=ProviderName.MOCK,
        model="mock",
        status=status,
        current_iteration=iterations,
        total_tokens=tokens,
        total_cost=cost,
        started_at=started,
        stopped_at=stopped,
        iterations=iters,
    )


# ── Unit tests: RunComparator ────────────────────────────────────────

comparator = RunComparator()


async def test_compare_metrics_basic() -> None:
    run_a = _make_run("run_a", iterations=5, tokens=1000, cost=0.5)
    run_b = _make_run("run_b", iterations=10, tokens=2000, cost=1.0)
    result = comparator.compare_metrics([run_a, run_b])

    assert len(result["runs"]) == 2
    assert result["best_by_cost"] == "run_a"
    assert result["best_by_tokens"] == "run_a"
    assert result["best_by_iterations"] == "run_a"


async def test_compare_metrics_single_run() -> None:
    run_a = _make_run("run_a")
    result = comparator.compare_metrics([run_a])

    assert len(result["runs"]) == 1
    assert result["best_by_cost"] == "run_a"
    assert result["best_by_tokens"] == "run_a"
    assert result["best_by_iterations"] == "run_a"


async def test_jaccard_similarity_identical() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    assert comparator._jaccard_similarity(text, text) == 1.0


async def test_jaccard_similarity_different() -> None:
    text_a = "the quick brown fox"
    text_b = "the slow red cat"
    sim = comparator._jaccard_similarity(text_a, text_b)
    assert 0.0 < sim < 1.0


async def test_jaccard_similarity_empty() -> None:
    assert comparator._jaccard_similarity("", "") == 0.0


async def test_compare_quality_with_grades() -> None:
    run_a = _make_run(
        "run_a",
        iterations=3,
        reviewer_feedback="Overall grade: A\n[CRITICAL] Missing control\n[HIGH] Needs more data",
    )
    run_b = _make_run(
        "run_b",
        iterations=3,
        reviewer_feedback="Grade: C\n[HIGH] Weak method\n[MEDIUM] Minor issues",
    )
    result = comparator.compare_quality([run_a, run_b])

    assert len(result["runs"]) == 2
    qa = next(r for r in result["runs"] if r["run_id"] == "run_a")
    qb = next(r for r in result["runs"] if r["run_id"] == "run_b")
    assert qa["final_grade"] == "A"
    assert qa["critical_count"] == 1
    assert qa["high_count"] == 1
    assert qb["final_grade"] == "C"
    assert result["best_by_grade"] == "run_a"


async def test_compare_quality_no_feedback() -> None:
    run_a = _make_run("run_a", iterations=2, reviewer_feedback=None)
    result = comparator.compare_quality([run_a])

    assert len(result["runs"]) == 1
    assert result["runs"][0]["final_grade"] == ""
    assert result["runs"][0]["total_critiques"] == 0
    # No best_by_grade when no graded runs
    assert "best_by_grade" not in result


async def test_generate_summary_combines_all() -> None:
    run_a = _make_run("run_a", reviewer_feedback="Grade: B\n[HIGH] Issue 1")
    run_b = _make_run("run_b", reviewer_feedback="Grade: A")
    result = comparator.generate_summary([run_a, run_b], ["path_a", "path_b"])

    assert "metrics" in result
    assert "outputs" in result
    assert "quality" in result
    assert result["metrics"]["best_by_cost"] in ("run_a", "run_b")
    assert result["quality"]["best_by_grade"] == "run_b"


# ── API integration tests ────────────────────────────────────────────


async def _create_run_via_api(client: AsyncClient, ws_id: str, task: str) -> str:
    resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": task,
        "provider": "mock",
    })
    assert resp.status_code == 201
    return resp.json()["run_id"]


async def test_api_compare_metrics(client: AsyncClient) -> None:
    ws = await client.post("/api/workspaces", json={"name": "ws-cmp", "path": "ws-cmp"})
    ws_id = ws.json()["id"]
    rid1 = await _create_run_via_api(client, ws_id, "task 1")
    rid2 = await _create_run_via_api(client, ws_id, "task 2")

    resp = await client.post("/api/comparison/metrics", json={"run_ids": [rid1, rid2]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["runs"]) == 2
    assert "best_by_cost" in data
    assert "best_by_tokens" in data


async def test_api_compare_summary(client: AsyncClient) -> None:
    ws = await client.post("/api/workspaces", json={"name": "ws-cmp2", "path": "ws-cmp2"})
    ws_id = ws.json()["id"]
    rid1 = await _create_run_via_api(client, ws_id, "task a")
    rid2 = await _create_run_via_api(client, ws_id, "task b")

    resp = await client.post("/api/comparison/summary", json={"run_ids": [rid1, rid2]})
    assert resp.status_code == 200
    data = resp.json()
    assert "metrics" in data
    assert "outputs" in data
    assert "quality" in data
