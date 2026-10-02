"""Run config ``seed_executor_output``: an iteration-0 executor output.

A seeded run treats the given text as the executor's previous submission, so
the first step is the one after the executor (``reviewer_consensus``: panel,
executor, panel, ...). The seed is the artifact under review of the first
panel, the "previous submission (iteration 0)" of the first executor revision
and the first executor output of the output-similarity signal.
"""
from __future__ import annotations

from pathlib import Path

from backend.models import (
    LoopDefinition,
    LoopEdge,
    LoopNode,
    ProviderName,
    RunState,
    RunStatus,
)
from backend.orchestrator.presets import get_preset
from backend.tests.fixes_r2_helpers import LENSES, ScriptedReviewAdapter, json_block, run_engine

SEED = ("# Original write-up\n\nWe found 19 significant heads (p < 0.05) "
        "and conclude they are the regulatory circuit. SEED-MARKER-7f3a")
HIGH = json_block([{"severity": "high",
                    "description": "No multiple-testing correction across 144 heads."}])


def _lens_reply(role, request, n):
    return HIGH


def _roles(state):
    return [it.role for it in sorted(state.iterations, key=lambda i: i.iteration_number)]


def _requests(adapter, role):
    return [r for r in adapter.requests if r.prompt_bundle.variables.get("role") == role]


def _disk_writer(root: Path, run_id: str):
    def write(n, name, content):
        d = root / "runs" / run_id / f"iter_{n:04d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(content)
    return write


def test_seeded_consensus_loop_starts_with_the_panel():
    adapter = ScriptedReviewAdapter(_lens_reply, roles=LENSES)
    events: list = []
    final, eng = run_engine(adapter, get_preset("reviewer_consensus"), 5,
                            {"convergence_enabled": False, "seed_executor_output": SEED},
                            events=events)
    assert final.stop_reason == "max_iterations"
    assert _roles(final) == ["consensus_merger", "executor", "consensus_merger",
                             "executor", "consensus_merger"]
    started = [d for t, d in events if t == "loop_started"][0]
    assert started["seed_executor_output_chars"] == len(SEED)

    # First panel reviews the seed with the neutral review framing only.
    first_panel = adapter.requests[:3]
    assert {r.prompt_bundle.variables["role"] for r in first_panel} == set(LENSES)
    for r in first_panel:
        up = r.prompt_bundle.user_prompt
        assert "=== ARTIFACT UNDER REVIEW ===\n" + SEED in up
        assert "REVISION REQUEST" not in up
        assert "PREVIOUS SUBMISSION" not in up

    # First executor call is a revision of the seed with the panel's feedback.
    ex = _requests(adapter, "executor")
    assert len(ex) == 2
    up = ex[0].prompt_bundle.user_prompt
    assert "=== REVISION REQUEST (Iteration 2) ===" in up
    assert "--- Feedback from consensus_merger ---" in up
    assert "multiple-testing correction" in up
    assert "=== YOUR PREVIOUS SUBMISSION (iteration 0) ===\n" + SEED in up

    # The second panel reviews the executor's revision, not the seed.
    exec_out = final.iterations[1].output_summary
    lens_requests = [r for r in adapter.requests
                     if r.prompt_bundle.variables.get("role") in LENSES]
    second_panel = lens_requests[3:6]
    assert len(second_panel) == 3
    for r in second_panel:
        up = r.prompt_bundle.user_prompt
        assert "SEED-MARKER-7f3a" not in up
        assert exec_out[:200] in up
        assert "REVISION REQUEST" not in up

    # The seed is the first executor output of the similarity signal.
    metrics = eng._convergence.get_metrics()
    assert len(metrics["similarity_scores"]) == 2      # (seed, E1), (E1, E2)
    assert metrics["iteration_count"] == 5             # the seed is not an iteration
    assert len(metrics["grade_history"]) == 3


def test_seeded_single_reviewer_loop_and_unseeded_default():
    adapter = ScriptedReviewAdapter(_lens_reply)
    final, _ = run_engine(adapter, get_preset("executor_reviewer"), 3,
                          {"convergence_enabled": False, "seed_executor_output": SEED})
    assert _roles(final) == ["reviewer", "executor", "reviewer"]
    assert SEED in _requests(adapter, "reviewer")[0].prompt_bundle.user_prompt

    plain = ScriptedReviewAdapter(_lens_reply)
    final2, _ = run_engine(plain, get_preset("executor_reviewer"), 3,
                           {"convergence_enabled": False})
    assert _roles(final2) == ["executor", "reviewer", "executor"]
    # Whitespace-only seed = no seed.
    blank = ScriptedReviewAdapter(_lens_reply)
    final3, _ = run_engine(blank, get_preset("executor_reviewer"), 2,
                           {"convergence_enabled": False, "seed_executor_output": "  \n"})
    assert _roles(final3) == ["executor", "reviewer"]


def test_seed_counts_for_output_similarity_convergence():
    # The executor (mock) output differs from the seed, so with window 1 and a
    # threshold of 0 the similarity signal holds right after E1 (pair seed/E1).
    adapter = ScriptedReviewAdapter(_lens_reply, roles=LENSES)
    final, _ = run_engine(adapter, get_preset("reviewer_consensus"), 9,
                          {"seed_executor_output": SEED, "convergence_enabled": True,
                           "convergence_window": 1,
                           "convergence_min_iterations": 0,
                           "convergence_signals": ["output_similar"],
                           "convergence_similarity_threshold": 0.0})
    assert final.stop_reason == "converged:output_similar"
    assert _roles(final) == ["consensus_merger", "executor"]


def test_invalid_seed_and_plan_without_executor_fail_cleanly():
    for bad in (123, {"text": SEED}, ["a"]):
        final, _ = run_engine(ScriptedReviewAdapter(_lens_reply), get_preset("executor_reviewer"),
                              3, {"seed_executor_output": bad})
        assert final.stop_reason == "failed:invalid_config"
        assert final.iterations == []
    reviewer_only = LoopDefinition(
        name="rev", nodes=[LoopNode(id="reviewer", role="reviewer",
                                    prompt_ref="reviewer/mi_reviewer")],
        edges=[LoopEdge(source="reviewer", target="reviewer", condition="always")],
        max_iterations=3)
    final, _ = run_engine(ScriptedReviewAdapter(_lens_reply), reviewer_only, 3,
                          {"seed_executor_output": SEED})
    assert final.stop_reason == "failed:invalid_config"
    assert "executor step" in (final.error or "")


def test_resume_of_a_seeded_run_keeps_the_seed_and_the_plan_position(tmp_path):
    run_id = "seeded1"
    cfg = {"convergence_enabled": False, "seed_executor_output": SEED}
    a1 = ScriptedReviewAdapter(_lens_reply, roles=LENSES)
    first = RunState(run_id=run_id, workspace_id="w", loop_preset="x", task="t",
                     provider=ProviderName.MOCK, model="", max_iterations=1, config=dict(cfg))
    f1, _ = run_engine(a1, get_preset("reviewer_consensus"), run_state=first,
                       workspace_path=str(tmp_path), artifact_writer=_disk_writer(tmp_path, run_id))
    assert _roles(f1) == ["consensus_merger"]

    # Resume after P0 only: the next step is the executor revising the seed.
    state = RunState(run_id=run_id, workspace_id="w", loop_preset="x", task="t",
                     provider=ProviderName.MOCK, model="", max_iterations=4,
                     status=RunStatus.PAUSED, current_iteration=1,
                     iterations=[i.model_copy() for i in f1.iterations], config=dict(cfg))
    a2 = ScriptedReviewAdapter(_lens_reply, roles=LENSES)
    f2, eng = run_engine(a2, get_preset("reviewer_consensus"), run_state=state, resume_from=1,
                         workspace_path=str(tmp_path),
                         artifact_writer=_disk_writer(tmp_path, run_id))
    assert _roles(f2) == ["consensus_merger", "executor", "consensus_merger", "executor"]
    up = _requests(a2, "executor")[0].prompt_bundle.user_prompt
    assert "=== YOUR PREVIOUS SUBMISSION (iteration 0) ===\n" + SEED in up
    assert "--- Feedback from consensus_merger ---" in up
    assert "multiple-testing correction" in up
    # seed + two executor revisions -> two similarity pairs, as uninterrupted.
    assert len(eng._convergence.get_metrics()["similarity_scores"]) == 2
