"""E2 prompt assembly and E4 request wiring in the live engine.

* Only executor steps receive the "=== REVISION REQUEST ... revise your
  artifacts" framing; reviewer lenses, consensus panels, the idea generator,
  etc. receive the task (context only), the iteration number and the
  executor's latest FULL output under "=== ARTIFACT UNDER REVIEW ===".
* Every adapter request carries the run's model / reasoning effort;
  non-executor requests run without tools unless ``reviewer_allow_tools``.
"""
from __future__ import annotations

import asyncio

import pytest

from backend.adapters.mock import MockAdapter
from backend.models import (
    LoopDefinition,
    LoopEdge,
    LoopNode,
    ProviderName,
    RunState,
)
from backend.orchestrator.engine import REVIEW_ROLES, LoopEngine
from backend.orchestrator.presets import get_preset

TASK = "Probe attention for TF->target structure. Save to MECH.md when done."

FORBIDDEN_IN_REVIEW = (
    "REVISION REQUEST",
    "Revise your artifacts",
    "Continue with",
    "make targeted revisions",
)


class Recorder:
    """Adapter wrapper recording every request (delegates to MockAdapter)."""

    name = "mock"

    def __init__(self, inner=None):
        self.inner = inner or MockAdapter(min_delay=0, max_delay=0)
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        return await self.inner.run(request)

    async def smoke_test(self):
        return {"status": "ok"}

    def is_available(self):
        return True

    def by_role(self, role):
        return [r for r in self.requests if r.prompt_bundle.variables.get("role") == role]


@pytest.fixture(autouse=True)
def _reset_mock():
    MockAdapter.reset_iteration_count()
    MockAdapter.fixed_iteration = None
    yield
    MockAdapter.fixed_iteration = None


def _run(preset, max_iterations, config=None, model="", adapter=None, loop=None):
    rec = adapter or Recorder()

    async def go():
        eng = LoopEngine(adapter=rec)
        loop_def = loop or get_preset(preset, max_iterations=max_iterations)
        run = RunState(workspace_id="w", loop_preset=preset, task=TASK,
                       provider=ProviderName.MOCK, model=model,
                       max_iterations=max_iterations,
                       config={"convergence_enabled": False, **(config or {})})
        return await eng.run_loop(run, loop_def)

    final = asyncio.run(go())
    return final, rec


def test_first_executor_prompt_is_the_task_only():
    _, rec = _run("executor_reviewer", 2)
    first = rec.by_role("executor")[0].prompt_bundle.user_prompt
    assert first.strip() == TASK
    assert "REVISION REQUEST" not in first


def test_single_reviewer_gets_neutral_framing_with_full_executor_output():
    final, rec = _run("executor_reviewer", 2)
    reviewer_req = rec.by_role("reviewer")[0]
    prompt = reviewer_req.prompt_bundle.user_prompt
    assert "=== ARTIFACT UNDER REVIEW ===" in prompt
    assert "=== END ARTIFACT UNDER REVIEW ===" in prompt
    assert "Iteration 2" in prompt
    assert TASK in prompt  # task statement as context
    # The executor's FULL output (well over the 500-char summary) is included.
    full = final.iterations[0]
    assert len(full.output_summary) == 500
    assert "auroc_attention = 0." in prompt  # near the end of the executor output
    assert "## Analysis Code" in prompt
    for bad in FORBIDDEN_IN_REVIEW:
        assert bad not in prompt.split("=== ARTIFACT UNDER REVIEW ===")[0], bad
    assert "REVISION REQUEST" not in prompt
    assert "Revise your artifacts" not in prompt


def test_adversarial_and_idea_generator_get_neutral_framing():
    # every_k:3 on the pre-increment counter: the adversarial lens is iteration 7.
    _, rec = _run("executor_reviewer", 7, config={})
    adv = rec.by_role("adversarial")
    assert adv, "adversarial step never ran"
    for req in adv:
        p = req.prompt_bundle.user_prompt
        assert "=== ARTIFACT UNDER REVIEW ===" in p
        assert "REVISION REQUEST" not in p and "Revise your artifacts" not in p

    _, rec2 = _run("research_followups", 4)
    ideas = rec2.by_role("idea_generator")
    assert ideas
    for req in ideas:
        p = req.prompt_bundle.user_prompt
        assert "=== ARTIFACT UNDER REVIEW ===" in p
        assert "INPUT FOR IDEA_GENERATOR" in p
        assert "REVISION REQUEST" not in p and "Revise your artifacts" not in p


def test_consensus_lenses_get_neutral_framing():
    _, rec = _run("reviewer_consensus", 4)
    lens_roles = ("reviewer", "adversarial_reviewer", "bio_plausibility_checker")
    lens_reqs = [r for r in rec.requests if r.prompt_bundle.variables.get("role") in lens_roles]
    assert len(lens_reqs) == 6  # two consensus steps x three lenses
    for req in lens_reqs:
        p = req.prompt_bundle.user_prompt
        assert p.startswith("=== REVIEW REQUEST (Iteration ")
        assert "=== ARTIFACT UNDER REVIEW ===" in p
        assert TASK in p
        assert "auroc_attention = 0." in p
        for bad in ("REVISION REQUEST", "Revise your artifacts", "Continue with"):
            assert bad not in p


def test_executor_revision_prompt_carries_feedback_and_previous_submission():
    _, rec = _run("executor_reviewer", 3)
    second = rec.by_role("executor")[1].prompt_bundle.user_prompt
    assert second.startswith("=== REVISION REQUEST (Iteration 3) ===")
    assert "--- Feedback from reviewer ---" in second
    assert "=== REVIEWER FEEDBACK" in second
    assert "=== YOUR PREVIOUS SUBMISSION (iteration 1) ===" in second
    assert "Original task: " + TASK in second


def test_consensus_feedback_reaches_executor():
    _, rec = _run("reviewer_consensus", 3)
    second = rec.by_role("executor")[1].prompt_bundle.user_prompt
    assert "--- Feedback from consensus_merger ---" in second
    assert "=== REVIEWER FEEDBACK" in second


def test_model_and_effort_on_every_request_and_tools_off_for_reviewers():
    _, rec = _run("reviewer_consensus", 4, config={"reasoning_effort": "low"}, model="gpt-x")
    assert rec.requests
    for req in rec.requests:
        assert req.model == "gpt-x"
        assert req.reasoning_effort == "low"
        role = req.prompt_bundle.variables.get("role")
        if role == "executor":
            assert req.allow_tools is True
        else:
            assert req.allow_tools is False, role

    _, rec2 = _run("executor_reviewer", 2, config={"reasoning_effort": "high"}, model="m1")
    for req in rec2.requests:
        assert (req.model, req.reasoning_effort) == ("m1", "high")
    assert rec2.by_role("reviewer")[0].allow_tools is False
    assert rec2.by_role("executor")[0].allow_tools is True


def test_reviewer_allow_tools_and_executor_allow_tools_overrides():
    _, rec = _run("reviewer_consensus", 2,
                  config={"reviewer_allow_tools": True, "executor_allow_tools": False})
    roles = {r.prompt_bundle.variables.get("role"): r.allow_tools for r in rec.requests}
    assert roles["executor"] is False
    assert roles["reviewer"] is True


def test_no_model_means_cli_default():
    _, rec = _run("executor_reviewer", 2)
    assert all(r.model == "" and r.reasoning_effort == "" for r in rec.requests)


def test_code_execution_addendum_and_report_flow():
    final, rec = _run("executor_reviewer", 3, config={"code_execution_enabled": True})
    ex_reqs = rec.by_role("executor")
    # Addendum (prompts/executor/mi_executor_code.yaml) only when enabled.
    assert "VERIFIED CODE EXECUTION IS ENABLED" in ex_reqs[0].prompt_bundle.system_prompt
    assert "results.json" in ex_reqs[0].prompt_bundle.system_prompt
    assert "{{" not in ex_reqs[0].prompt_bundle.system_prompt
    # Reviewers see the executed results next to the artifact.
    rp = rec.by_role("reviewer")[0].prompt_bundle.user_prompt
    art = rp.split("=== ARTIFACT UNDER REVIEW ===", 1)[1]
    assert "=== CODE EXECUTION REPORT ===" in art
    assert "incremental_auroc 0.033" in art
    # The next executor iteration gets its own execution report back.
    second = ex_reqs[1].prompt_bundle.user_prompt
    assert "Verified execution of the code in your previous submission (iteration 1)" in second
    assert "incremental_auroc 0.033" in second
    # And the reviewers' system prompts never get the executor addendum.
    assert "VERIFIED CODE EXECUTION" not in rec.by_role("reviewer")[0].prompt_bundle.system_prompt

    _, rec_off = _run("executor_reviewer", 2)
    assert "VERIFIED CODE EXECUTION" not in rec_off.by_role("executor")[0].prompt_bundle.system_prompt
    assert "CODE EXECUTION REPORT" not in rec_off.by_role("reviewer")[0].prompt_bundle.user_prompt


def test_code_execution_without_code_block_is_reported_to_reviewers():
    class NoCode(Recorder):
        async def run(self, request):
            res = await super().run(request)
            if request.prompt_bundle.variables.get("role") == "executor":
                res.output = "# MECH.md\n\nAUROC = 0.91 (no code shown)\n"
            return res

    final, rec = _run("executor_reviewer", 2, config={"code_execution_enabled": True},
                      adapter=NoCode())
    rp = rec.by_role("reviewer")[0].prompt_bundle.user_prompt
    assert "contains no fenced" in rp
    ex_it = final.iterations[0]
    assert ex_it.code_execution is not None and ex_it.code_execution["executed"] == 0


def test_review_roles_cover_single_step_lenses():
    assert {"reviewer", "adversarial_reviewer", "adversarial",
            "bio_plausibility_checker"} <= set(REVIEW_ROLES)


def test_custom_reviewer_first_loop_has_placeholder_artifact():
    loop = LoopDefinition(
        name="rev_first",
        nodes=[LoopNode(id="r", role="reviewer", prompt_ref="reviewer/mi_reviewer"),
               LoopNode(id="e", role="executor", prompt_ref="executor/mi_executor")],
        edges=[LoopEdge(source="r", target="e"), LoopEdge(source="e", target="r")],
    )
    _, rec = _run("custom", 2, loop=loop)
    p = rec.by_role("reviewer")[0].prompt_bundle.user_prompt
    assert "(No executor output is available yet.)" in p
