"""Tests for the stopping-rule calibration harness (X3): mock trajectories
through the real engine, resume, judge validation / caching and rule replay.

Run:
    TMPDIR="/Volumes/Crucial X6/tmp_miw" PYTHONDONTWRITEBYTECODE=1 \
      python -m pytest experiments/stopping/test_stopping.py -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import judge_states as js  # noqa: E402
import replay as rp  # noqa: E402
import run_loops as rl  # noqa: E402
import stopping_common as sc  # noqa: E402
from backend.models import AdapterRunResult  # noqa: E402

ITEM = "A01-flawed"
_ITER = re.compile(r"iter_(\d+)")


def _items():
    return {ITEM: sc.flawed_items()[ITEM]}


def _json_block(crits):
    return "Assessment.\n```json\n" + json.dumps({"critiques": crits}) + "\n```"


def _crit(sev, text):
    return {"severity": sev, "description": text, "required_fix": "fix it"}


class ScriptedAdapter:
    """Executor: deterministic revised write-ups; lenses: JSON critiques whose
    severity drops with the iteration. ``fail`` maps (role, iteration, n-th
    call of that role at that iteration) -> error string."""

    name = "scripted"

    def __init__(self, fail=None, always_fail_roles=()):
        self.requests = []
        self.fail = dict(fail or {})
        self.always_fail_roles = set(always_fail_roles)
        self.calls: dict[tuple, int] = {}

    async def run(self, request):
        self.requests.append(request)
        role = request.prompt_bundle.variables.get("role", "")
        m = _ITER.search(request.workspace_context.iteration_dir or "")
        it = int(m.group(1)) if m else 0
        key = (role, it)
        self.calls[key] = self.calls.get(key, 0) + 1
        err = self.fail.get((role, it, self.calls[key]))
        if err or role in self.always_fail_roles:
            return AdapterRunResult(success=False, error=err or "[unknown] lens broken",
                                    exit_code=1, provider="scripted")
        if role == "executor":
            k = it // 2
            out = (f"# Revised write-up v{k}\n\nWe report 19 heads (uncorrected). "
                   f"Revision {k} withdraws the circuit claim. " + "stable text " * (20 + k))
        else:
            if it <= 3:
                out = _json_block([_crit("high", f"{role}: no multiple-testing correction"),
                                   _crit("medium", f"{role}: missing co-expression baseline")])
            else:
                out = _json_block([_crit("low", f"{role}: minor wording")])
        return AdapterRunResult(success=True, output=out, provider="scripted",
                                model=request.model, reasoning_effort=request.reasoning_effort,
                                cli_version="0.0.1", input_tokens=100, output_tokens=40,
                                token_usage=140)

    def by_role(self, role):
        return [r for r in self.requests if r.prompt_bundle.variables.get("role") == role]


def _runner(tmp_path, adapter, horizon=sc.HORIZON, **kw):
    return rl.LoopRunner(tmp_path / "data", provider="mock", horizon=horizon, backoff=0,
                         workspace_root=tmp_path / "ws", adapter_factory=lambda p, m, e: adapter,
                         items=_items(), **kw)


# ── run_loops ───────────────────────────────────────────────────────────


def test_full_horizon_trajectory_order_prompts_and_record(tmp_path):
    adapter = ScriptedAdapter()
    runner = _runner(tmp_path, adapter)
    out = asyncio.run(runner.run(["sol-medium"], [ITEM]))
    assert out == {"complete": [f"sol-medium/{ITEM}"]}
    udir = sc.unit_dir(tmp_path / "data", "sol-medium", ITEM)
    traj = sc.read_json(udir / "trajectory.json")
    assert traj["observed_order"] == ["P0", "E1", "P1", "E2", "P2", "E3", "P3", "E4", "P4",
                                      "E5", "P5"]
    assert traj["stop_reason"] == "max_iterations"
    assert traj["similarity_matches_engine"] is True
    assert len(traj["engine_similarity_scores"]) == 5
    seed = _items()[ITEM].artifact_text
    assert (udir / "states" / "E0.md").read_text() == seed
    assert traj["states"][0]["similarity_to_previous"] is None
    # cross-lens agreement escalates the merged HIGH critique to CRITICAL
    assert traj["panels"][0]["critical"] == 1 and traj["panels"][0]["high"] == 1
    assert traj["panels"][0]["lenses"]["reviewer"]["high"] == 1
    assert traj["panels"][3]["critical"] + traj["panels"][3]["high"] == 0
    assert all(p["panel_status"] == "complete" for p in traj["panels"])
    assert traj["similarity_methods"] == ["jaccard"]   # pinned X3 merge (run_loops.X3_MERGE)
    assert set(traj["panels"][0]["lenses"]) == set(sc.LENS_ROLES)

    # Prompts: lenses see the current state with the neutral framing only.
    lens_reqs = [r for r in adapter.requests if r.prompt_bundle.variables["role"] in sc.LENS_ROLES]
    assert len(lens_reqs) == 18 and len(adapter.by_role("executor")) == 5
    for r in lens_reqs:
        up = r.prompt_bundle.user_prompt
        it = int(_ITER.search(r.workspace_context.iteration_dir).group(1))
        state = (udir / "states" / f"E{(it - 1) // 2}.md").read_text()
        assert "=== ARTIFACT UNDER REVIEW ===\n" + state.rstrip() in up
        assert "REVISION REQUEST" not in up and "PREVIOUS SUBMISSION" not in up
        assert r.allow_tools is False
    for k, r in enumerate(adapter.by_role("executor"), start=1):
        up = r.prompt_bundle.user_prompt
        prev = (udir / "states" / f"E{k - 1}.md").read_text()
        assert f"=== REVISION REQUEST (Iteration {2 * k}) ===" in up
        assert f"=== YOUR PREVIOUS SUBMISSION (iteration {2 * (k - 1)}) ===\n{prev.strip()}" in up
        assert "--- Feedback from consensus_merger ---" in up
        assert sc.EXECUTOR_TASK in up
        assert r.allow_tools is False and r.model == "gpt-5.6-sol"
        assert r.reasoning_effort == "medium"

    # Call records: CLI cwd outside the repo and never the unit directory.
    calls = sorted((udir / "calls").glob("call_*.json"))
    assert len(calls) == 23
    for c in calls:
        rec = sc.read_json(c)
        cwd = Path(rec["request"]["cli_cwd"]).resolve()
        assert sc.REPO_ROOT.resolve() not in cwd.parents
        assert rec["request"]["engine_workspace_path"] == str(udir)
        assert rec["request"]["system_prompt_is_inline_fallback"] is False
        assert rec["result"]["success"] is True
    meta = sc.read_json(udir / "runs" / f"sol-medium__{ITEM}" / "run_meta.json")
    assert meta["stop_reason"] == "max_iterations" and len(meta["iterations"]) == 11
    unit = sc.read_json(udir / "unit.json")
    assert unit["signature"]["run_config"]["convergence_enabled"] is False
    assert unit["platform"]["consensus_merge_used"] == {
        "consensus_similarity_method": "jaccard", "consensus_similarity_threshold": 0.5}
    assert "consensus_similarity_method" not in unit["signature"]["run_config"]

    # A second invocation reuses the complete unit without any call.
    again = ScriptedAdapter()
    out2 = asyncio.run(_runner(tmp_path, again).run(["sol-medium"], [ITEM]))
    assert out2 == {"cached": [f"sol-medium/{ITEM}"]} and again.requests == []


def test_failed_step_is_archived_and_resumed(tmp_path):
    adapter = ScriptedAdapter(fail={("executor", 4, 1): "[empty_output] codex returned nothing"})
    out = asyncio.run(_runner(tmp_path, adapter).run(["sol-medium"], [ITEM]))
    assert out == {"complete": [f"sol-medium/{ITEM}"]}
    udir = sc.unit_dir(tmp_path / "data", "sol-medium", ITEM)
    traj = sc.read_json(udir / "trajectory.json")
    assert traj["observed_order"][3] == "E2"
    assert len(traj["failed_attempts"]) == 1
    assert traj["failed_attempts"][0]["label"] == "E2"
    assert traj["failed_attempts"][0]["run_started_at"] == "P0"
    ex = adapter.by_role("executor")
    assert len(ex) == 6   # E2 called twice
    retry = ex[2].prompt_bundle.user_prompt
    e1 = (udir / "states" / "E1.md").read_text().strip()
    assert "=== YOUR PREVIOUS SUBMISSION (iteration 2) ===\n" + e1 in retry
    assert "no multiple-testing correction" in retry   # P1 feedback restored from disk
    assert traj["similarity_matches_engine"] is True


def test_partial_panel_policy_fail_then_kept_on_last_attempt(tmp_path):
    adapter = ScriptedAdapter(always_fail_roles=("bio_plausibility_checker",))
    runner = _runner(tmp_path, adapter, horizon=1)
    out = asyncio.run(runner.run(["sol-medium"], [ITEM]))
    assert out == {"complete": [f"sol-medium/{ITEM}"]}
    udir = sc.unit_dir(tmp_path / "data", "sol-medium", ITEM)
    traj = sc.read_json(udir / "trajectory.json")
    assert traj["observed_order"] == ["P0", "E1", "P1"]
    for p in traj["panels"]:
        assert p["panel_partial"] is True and p["grade_valid"] is False
        assert p["failed_lenses"] == ["bio_plausibility_checker"]
    assert len(traj["failed_attempts"]) == 4   # two failed attempts per panel
    assert [a["partial_policy"] for a in traj["failed_attempts"]] == ["fail"] * 4
    # the failed panels' lens outputs were archived before the retry
    archived = sorted(p.name for p in (udir / "failed_attempts").glob("*/iter_*"))
    assert archived == ["iter_0001", "iter_0001", "iter_0003", "iter_0003"]
    rows = [json.loads(x) for x in (udir / "events.jsonl").read_text().splitlines()]
    starts = [r for r in rows if r["type"] == "loop_started"]
    assert len(starts) == 6    # P0 x3 (last alone, no_stop), then E1+P1, P1 retry, P1 alone


def test_auth_error_is_fatal(tmp_path):
    adapter = ScriptedAdapter(fail={("executor", 2, 1): "[auth] not logged in"})
    with pytest.raises(rl.FatalRunError):
        asyncio.run(_runner(tmp_path, adapter).run(["sol-medium"], [ITEM]))


def test_signature_mismatch_is_never_overwritten(tmp_path):
    udir = sc.unit_dir(tmp_path / "data", "sol-medium", ITEM)
    udir.mkdir(parents=True)
    sc.atomic_write_json(udir / "unit.json", {"signature": {"model": "other"}})
    adapter = ScriptedAdapter()
    out = asyncio.run(_runner(tmp_path, adapter).run(["sol-medium"], [ITEM]))
    assert out == {"mismatch": [f"sol-medium/{ITEM}"]}
    assert adapter.requests == []
    assert sc.read_json(udir / "unit.json") == {"signature": {"model": "other"}}


def test_workspace_root_inside_repo_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        rl.LoopRunner(tmp_path, provider="mock", workspace_root=sc.REPO_ROOT / "x",
                      items=_items())


def test_geometry_helpers():
    assert sc.expected_roles(2) == ["consensus_merger", "executor", "consensus_merger",
                                    "executor", "consensus_merger"]
    assert [sc.step_label(n) for n in (0, 1, 2, 3, 10, 11)] == ["E0", "P0", "E1", "P1", "E5",
                                                                 "P5"]
    assert [sc.state_index_at(n) for n in (1, 2, 3, 4, 11)] == [0, 1, 1, 2, 5]
    assert sc.new_numbers("AUROC 0.64 and 0.71 after 7.2 of 144; L9H4 2", "AUROC 0.64, 144") \
        == ["0.71", "7.2"]
    assert sc.pending_markers("[PENDING ANALYSIS] x [pending analysis]") == 2


# ── judge_states ────────────────────────────────────────────────────────


def _make_complete_unit(tmp_path, texts):
    udir = sc.unit_dir(tmp_path / "data", "sol-medium", ITEM)
    (udir / "states").mkdir(parents=True, exist_ok=True)
    states = []
    for k, t in enumerate(texts):
        (udir / "states" / f"E{k}.md").write_text(t)
        states.append({"k": k, "sha256": sc.sha256_text(t), "path": f"states/E{k}.md"})
    sc.atomic_write_json(udir / "trajectory.json", {"complete": True, "states": states})
    return udir


def _valid_answer(labels, errors=()):
    flaws = [{"flaw_id": f"A01-F{i + 1}", "label": lab, "evidence": "x", "rationale": "y"}
             for i, lab in enumerate(labels)]
    return json.dumps({"flaws": flaws, "new_errors": list(errors)})


class FakeJudge:
    name = "fake"

    def __init__(self, answers):
        self.answers = list(answers)
        self.requests = []

    async def run(self, request):
        self.requests.append(request)
        ans = self.answers.pop(0) if self.answers else _valid_answer(["unresolved"] * 4)
        if isinstance(ans, AdapterRunResult):
            return ans
        return AdapterRunResult(success=True, output=ans, model=request.model,
                                reasoning_effort=request.reasoning_effort, input_tokens=10,
                                output_tokens=5)


def _judge(tmp_path, fake, **kw):
    return js.StateJudge(tmp_path / "data", adapter_factory=lambda p, m, e: fake,
                         workspace_root=tmp_path / "ws", items=_items(), backoff=0, **kw)


def test_validate_and_extract_json():
    ids = [f"A01-F{i}" for i in range(1, 5)]
    good = json.loads(_valid_answer(["resolved", "unresolved", "resolved_by_fabrication",
                                     "unresolved"],
                                    [{"category": "fabrication", "quote": "q",
                                      "description": "new AUROC 0.58"}]))
    j, errs = js.validate_judgment(good, ids)
    assert errs == [] and js.judgment_counts(j)["resolved_by_fabrication"] == 1
    assert js.judgment_counts(j)["new_fabrication"] == 1
    bad = json.loads(_valid_answer(["resolved", "maybe", "unresolved", "unresolved"]))
    assert js.validate_judgment(bad, ids)[0] is None
    missing = {"flaws": good["flaws"][:3], "new_errors": []}
    assert "missing" in " ".join(js.validate_judgment(missing, ids)[1])
    dup = {"flaws": good["flaws"] + [good["flaws"][0]], "new_errors": []}
    assert js.validate_judgment(dup, ids)[0] is None
    badcat = dict(good, new_errors=[{"category": "typo", "description": "d"}])
    assert js.validate_judgment(badcat, ids)[0] is None
    wrapped = "Here you go:\n```json\n" + json.dumps(good) + "\n```\nDone."
    assert js.extract_json(wrapped)["flaws"][0]["flaw_id"] == "A01-F1"
    assert js.extract_json("prose " + json.dumps(good) + " trailing")["flaws"]
    assert js.extract_json("no json here") is None


def test_judge_records_cache_reask_and_supersede(tmp_path):
    seed = _items()[ITEM].artifact_text
    udir = _make_complete_unit(tmp_path, [seed, seed + "\nRevised."])
    fake = FakeJudge([_valid_answer(["unresolved"] * 4), "not json at all",
                      _valid_answer(["resolved", "unresolved", "unresolved", "unresolved"])])
    judge = _judge(tmp_path, fake, concurrency=1)
    tasks = judge.tasks(["sol-medium"], [ITEM], [0, 1])
    assert tasks == [("sol-medium", ITEM, 0), ("sol-medium", ITEM, 1)]
    counts = asyncio.run(judge.run(tasks))
    assert counts == {"valid": 2}
    r0 = sc.read_json(sc.judgment_path(tmp_path / "data", "gpt-5.6-sol", "sol-medium", ITEM, 0))
    r1 = sc.read_json(sc.judgment_path(tmp_path / "data", "gpt-5.6-sol", "sol-medium", ITEM, 1))
    assert r0["valid"] and r0["counts"]["unresolved"] == 4
    assert r1["valid"] and len(r1["attempts"]) == 2 and r1["attempts"][1]["reask"] is True
    assert r1["counts"]["resolved"] == 1
    # Prompts: ground truth + original + current; judge request is tools-off.
    up = fake.requests[0].prompt_bundle.user_prompt
    assert "A01-F4" in up and "=== CURRENT WRITE-UP" in up and "=== ORIGINAL WRITE-UP" in up
    assert fake.requests[0].allow_tools is False
    assert fake.requests[0].model == "gpt-5.6-sol" and fake.requests[0].reasoning_effort == "high"
    assert "INVALID" in fake.requests[2].prompt_bundle.user_prompt
    # Cached: no new call.
    fake2 = FakeJudge([])
    assert asyncio.run(_judge(tmp_path, fake2).run(tasks)) == {"cached": 2}
    assert fake2.requests == []
    # Changed input -> superseded and re-judged.
    (udir / "states" / "E1.md").write_text(seed + "\nRevised again.")
    fake3 = FakeJudge([_valid_answer(["resolved"] * 4)])
    assert asyncio.run(_judge(tmp_path, fake3).run(tasks)) == {"cached": 1, "valid": 1}
    sup = list((sc.judgment_path(tmp_path / "data", "gpt-5.6-sol", "sol-medium", ITEM, 1)
                .parent / "superseded").glob("E1.*.json"))
    assert len(sup) == 1


def test_judge_invalid_twice_and_transient_failures(tmp_path):
    seed = _items()[ITEM].artifact_text
    _make_complete_unit(tmp_path, [seed])
    fake = FakeJudge(["{}", "still bad"])
    counts = asyncio.run(_judge(tmp_path, fake).run([("sol-medium", ITEM, 0)]))
    assert counts == {"invalid": 1}
    rec = sc.read_json(sc.judgment_path(tmp_path / "data", "gpt-5.6-sol", "sol-medium", ITEM, 0))
    assert rec["valid"] is False and len(rec["attempts"]) == 2
    # Invalid records are retried on the next invocation; transient failures back off.
    fail = AdapterRunResult(success=False, error="[timeout] slow")
    fake2 = FakeJudge([fail, _valid_answer(["unresolved"] * 4)])
    assert asyncio.run(_judge(tmp_path, fake2).run([("sol-medium", ITEM, 0)])) == {"valid": 1}
    sc.judgment_path(tmp_path / "data", "gpt-5.6-sol", "sol-medium", ITEM, 0).unlink()
    fake3 = FakeJudge([AdapterRunResult(success=False, error="[auth] expired")])
    with pytest.raises(js.FatalJudgeError):
        asyncio.run(_judge(tmp_path, fake3).run([("sol-medium", ITEM, 0)]))
    assert len(fake3.requests) == 1   # never retried


# ── replay ──────────────────────────────────────────────────────────────


def _synthetic(tmp_path, grades, ch, texts, unresolved, fab=None, crit=None, item=ITEM):
    """Write a synthetic complete trajectory + judgments; return data_dir."""
    data = tmp_path / "data"
    udir = sc.unit_dir(data, "sol-medium", item)
    (udir / "states").mkdir(parents=True, exist_ok=True)
    states = []
    for k, t in enumerate(texts):
        (udir / "states" / f"E{k}.md").write_text(t)
        sim = None if k == 0 else rl._jaccard_similarity(texts[k - 1], t)
        states.append({"k": k, "label": f"E{k}", "path": f"states/E{k}.md",
                       "similarity_to_previous": sim, "n_new_numbers": 0, "chars": len(t)})
    crit = crit or [0] * len(grades)
    panels, steps = [], []
    for n in range(1, 2 * (len(texts) - 1) + 2):
        tok = {"input": 10, "output": 5, "cached_input": 0, "total": 100}
        if n % 2 == 1:
            j = (n - 1) // 2
            p = {"j": j, "label": f"P{j}", "iteration": n, "grade": grades[j],
                 "grade_score": rl.grade_score(grades[j]), "critical": crit[j],
                 "high": ch[j] - crit[j], "n_critiques": ch[j] + 1, "panel_partial": False,
                 "grade_valid": True, "unresolved_critical": crit[j]}
            panels.append(p)
            steps.append({"iteration": n, "role": "consensus_merger", "tokens": tok, **p})
        else:
            steps.append({"iteration": n, "role": "executor", "tokens": tok})
    sc.atomic_write_json(udir / "trajectory.json", {
        "complete": True, "item": item, "config": "sol-medium", "model": "m", "effort": "e",
        "states": states, "panels": panels, "steps": steps, "similarity_methods": ["jaccard"]})
    fab = fab or [0] * len(texts)
    for k in range(len(texts)):
        labels = (["unresolved"] * unresolved[k] + ["resolved_by_fabrication"] * fab[k])
        labels += ["resolved"] * (4 - len(labels))
        flaws = [{"flaw_id": f"A01-F{i + 1}", "label": lab, "evidence": "", "rationale": ""}
                 for i, lab in enumerate(labels)]
        sc.atomic_write_json(sc.judgment_path(data, "gpt-5.6-sol", "sol-medium", item, k), {
            "valid": True, "state_sha256": "x",
            "judgment": {"flaws": flaws, "new_errors": [
                {"category": "fabrication", "quote": "", "description": "d"}] * fab[k]}})
    return data


def _rule(res, name):
    return next(s for s in res["rules"] if s["name"] == name)


def test_replay_rules_on_a_synthetic_trajectory(tmp_path):
    words = " ".join(f"w{i}" for i in range(50))
    texts = ["orig " + words, "rev1 x y z " + words[:60], words + " final",
             words + " final", words + " final", words + " final"]
    data = _synthetic(tmp_path,
                      grades=["F", "D", "D", "D", "C", "C"],
                      ch=[9, 5, 4, 0, 0, 0],
                      texts=texts,
                      unresolved=[4, 3, 2, 1, 0, 0],
                      fab=[0, 0, 0, 1, 0, 0],
                      crit=[2, 1, 0, 0, 0, 0])
    trajs = rp.load_trajectories(data, ["sol-medium"])
    res = rp.analyze_config("sol-medium", trajs["sol-medium"], data, "gpt-5.6-sol", 5, 50, 1)
    # grade_stable: D at P1, P2, P3 -> fires at P3 (iteration 7) -> state E3.
    g = _rule(res, "grade_stable_only")["runs"][0]
    assert (g["iteration"], g["k_stop"], g["stopped_before_horizon"]) == (7, 3, True)
    assert g["unresolved"] == 1 and g["fabricated_flaws"] == 1
    # output_similar: pairs (E2,E3), (E3,E4), (E4,E5) identical; with (E1,E2) low the
    # window of 3 first holds at E5 (iteration 10) -> state E5, at the horizon.
    s = _rule(res, "output_similar_only")["runs"][0]
    assert (s["iteration"], s["k_stop"], s["stopped_before_horizon"]) == (10, 5, False)
    # default any-of-3 = earliest of its signals.
    d = _rule(res, "default_any_of_3")
    assert d["runs"][0]["iteration"] == 7
    assert d["premature_stop_rate"] == 1.0 and d["premature_stop_rate_strict"] == 1.0
    # no_critical_high: zero C/H at P3, P4, P5 -> fires at P5 (iteration 11).
    nch = _rule(res, "no_critical_high_only")["runs"][0]
    assert nch["iteration"] == 11 and not nch["stopped_before_horizon"]
    # required no-C/H AND (grade OR similar): first at P5 as well.
    assert _rule(res, "noCH_and_(grade_stable_or_similar)")["runs"][0]["iteration"] == 11
    # fixed k.
    f2 = _rule(res, "fixed_k2")
    assert f2["runs"][0]["k_stop"] == 2 and f2["premature_stop_rate"] == 1.0
    assert f2["mean_panels"] == 2 and f2["mean_tokens"] == 400
    f4 = _rule(res, "fixed_k4")
    assert f4["premature_stop_rate"] == 0.0 and f4["mean_unresolved_at_stop"] == 0
    # Selection: eligible = rules stopping before H in >= 50% runs (here 1 run).
    sel = res["selection"]
    assert sel["selected"] == "fixed_k4"
    assert res["selection_adaptive_only"]["selected"] in ("default_any_of_3",
                                                          "grade_stable_only")
    assert res["oracle_first_all_resolved"][0]["first_k_all_resolved"] == 4
    assert res["judge_sanity"]["E0_frac_flaws_unresolved"] == 1.0
    assert res["per_revision"][3]["mean_fabricated_flaws"] == 1


def test_replay_gate_blocks_while_critical_remain(tmp_path):
    texts = ["same text here"] * 6
    data = _synthetic(tmp_path, grades=["C"] * 6, ch=[1] * 6, texts=texts,
                      unresolved=[1] * 6, crit=[1, 1, 1, 1, 0, 0])
    trajs = rp.load_trajectories(data, ["sol-medium"])
    t = trajs["sol-medium"][ITEM]
    plain = rp.apply_rule(t, {"name": "x", "family": "primary", "description": "",
                              "config": {"convergence_signals": ["grade_stable"]}}, 5)
    gated = rp.apply_rule(t, {"name": "x", "family": "primary", "description": "",
                              "config": {"convergence_signals": ["grade_stable"]},
                              "gate": True}, 5)
    assert plain["iteration"] == 5      # grade C at P0, P1, P2
    assert gated["iteration"] > plain["iteration"]
    assert gated["iteration"] == 9      # first panel without CRITICAL is P4


def test_spearman_and_bootstrap():
    assert rp.spearman([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(0.8)
    assert rp.spearman([1, 2, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, rel=1e-6)
    assert rp.spearman([1, 1, 1], [1, 2, 3]) is None
    block = rp.correlation_block({"a": [(1, 1), (2, 2), (3, 3)], "b": [(1, 3), (2, 2), (3, 1)]},
                                 n_boot=200, seed=3)
    assert block["n_pairs"] == 6 and block["n_items"] == 2
    assert block["mean_within_item_spearman"] == pytest.approx(0.0)
    lo, hi = rp.wilson(3, 12)
    assert 0 < lo < 0.25 < hi < 1


def test_replay_matches_the_live_engine_for_a_seeded_run(tmp_path):
    """The replayed default rule stops where the live engine (convergence on,
    same seed, same deterministic adapter) stops."""
    from backend.models import ProviderName, RunState
    from backend.orchestrator.engine import LoopEngine
    from backend.orchestrator.presets import get_preset

    adapter = ScriptedAdapter()
    asyncio.run(_runner(tmp_path, adapter).run(["sol-medium"], [ITEM]))
    trajs = rp.load_trajectories(tmp_path / "data", ["sol-medium"])
    traj = trajs["sol-medium"][ITEM]
    seed = _items()[ITEM].artifact_text
    for spec in [s for s in rp.rule_specs() if s["family"] != "fixed"]:
        replayed = rp.apply_rule(traj, spec, sc.HORIZON)
        cfg = rl.run_config("medium", seed)
        cfg["convergence_enabled"] = True
        cfg.update(spec.get("config") or {})
        cfg["consensus_gate"] = bool(spec.get("gate"))
        cfg["revision_budget"] = None

        async def live():
            eng = LoopEngine(adapter=ScriptedAdapter(), max_retries=0)
            run = RunState(workspace_id="w", loop_preset="reviewer_consensus",
                           task=sc.EXECUTOR_TASK, provider=ProviderName.MOCK,
                           model="gpt-5.6-sol", max_iterations=11, config=cfg)
            return await eng.run_loop(run, get_preset("reviewer_consensus"))

        final = asyncio.run(live())
        if replayed["fired"]:
            assert final.stop_reason == replayed["stop_reason"], spec["name"]
            assert final.current_iteration == replayed["iteration"], spec["name"]
        else:
            assert final.stop_reason == "max_iterations", spec["name"]
