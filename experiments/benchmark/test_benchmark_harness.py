"""Tests for the planted-flaw benchmark harness (fake adapters; no real LLM calls).

Run:
    python -m pytest experiments/benchmark/test_benchmark_harness.py -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import freeze  # noqa: E402
import judge as judge_mod  # noqa: E402
import run_reviews  # noqa: E402
import analyze  # noqa: E402
import merge_eval  # noqa: E402
from backend.models import AdapterRunResult  # noqa: E402

PILOT_ITEMS = ["A01-flawed", "A05-clean"]

# Which planted flaws (by suffix) each call "finds" in the fake reviewer.
FAKE_FINDS = {
    "rig1": ["F1"], "rig2": ["F1", "F2"], "rig3": ["F2"],
    "cmb1": ["F1", "F3"], "cmb2": ["F3"], "cmb3": ["F1"],
    "adv": ["F2", "F3"], "bio": ["F4"],
}
EXPECTED_RECALL = {"C1": 0.25, "C2": 0.5, "C3": 0.5, "C4": 0.5, "C5": 1.0}

TOPIC = {
    "F1": "no multiple testing correction across the per head tests marker F1",
    "F2": "coexpression confounding not controlled by the degree null marker F2",
    "F3": "causal mechanism claim from correlational attention evidence marker F3",
    "F4": "attention is not symmetric because query and key projections differ marker F4",
}


def tag_of_request(request) -> str:
    """Recover the call tag from the fake request (role + a counter)."""
    return request.prompt_bundle.variables.get("role", "")


def fake_review_output(finds: list[str], extra_text: str = "") -> str:
    crit = [{"severity": "high", "category": "statistics", "description": TOPIC[f],
             "required_fix": f"fix {f}"} for f in finds]
    crit.append({"severity": "low", "category": "reproducibility",
                 "description": "Share code, seeds and software versions for the whole pipeline.",
                 "required_fix": "Release code."})
    body = json.dumps({"critiques": crit, "overall_assessment": "fake"}, indent=2)
    return f"{extra_text}Some prose first.\n\n```json\n{body}\n```\n"


class FakeReviewer:
    """Returns a JSON critique block; which flaws depends on the call (by system prompt)."""

    def __init__(self, fail_first: int = 0, error: str = "[unknown] boom", sleep: float = 0.0):
        self.calls = 0
        self.fail_first = fail_first
        self.error = error
        self.sleep = sleep
        self.requests = []
        self.counter = {}

    async def cli_version(self):
        return "fake-cli 1.0"

    async def run(self, request):
        self.calls += 1
        self.requests.append(request)
        if self.sleep:
            await asyncio.sleep(self.sleep)
        if self.calls <= self.fail_first:
            return AdapterRunResult(success=False, error=self.error, exit_code=1)
        sp = request.prompt_bundle.system_prompt
        role = request.prompt_bundle.variables["role"]
        # Distinguish rig/cmb repeats by a per-(role, artifact) counter.
        key = (role, request.prompt_bundle.user_prompt, request.model)
        self.counter[key] = self.counter.get(key, 0) + 1
        k = self.counter[key]
        if role == "reviewer":
            tag = f"rig{k}"
        elif role == "reviewer_combined":
            tag = f"cmb{k}"
        elif role == "adversarial_reviewer":
            tag = "adv"
        else:
            tag = "bio"
        assert "REQUIRED OUTPUT FORMAT" in sp
        assert request.allow_tools is False
        return AdapterRunResult(
            success=True, output=fake_review_output(FAKE_FINDS[tag]), model=request.model,
            cli_version="fake-cli 1.0", input_tokens=1000, output_tokens=200,
            cached_input_tokens=400, raw_usage={"reasoning_output_tokens": 50},
            token_usage=1200,
        )


class FakeJudge:
    """Labels critiques by the 'marker Fk' token; can emit one invalid answer first."""

    def __init__(self, invalid_first: bool = False, disagree: bool = False):
        self.calls = 0
        self.invalid_first = invalid_first
        self.disagree = disagree
        self.prompts = []

    async def cli_version(self):
        return "fake-cli 1.0"

    async def run(self, request):
        self.calls += 1
        up = request.prompt_bundle.user_prompt
        self.prompts.append(up)
        m = re.search(r"valid flaw ids: ([^)]*)\)", up)
        valid = [x.strip() for x in m.group(1).split(",")] if m else []
        crit_section = up.split("CRITIQUES (N = ", 1)[1]
        n = int(crit_section.split(")", 1)[0])
        labels = []
        for line in crit_section.splitlines():
            mm = re.match(r"\[(\d+)\] (.*)", line)
            if not mm:
                continue
            idx, text = int(mm.group(1)), mm.group(2)
            fm = re.search(r"marker (F\d)", text)
            fid = "none"
            if fm and valid:
                fid = f"{valid[0].split('-')[0]}-{fm.group(1)}"
                if self.disagree and fm.group(1) == "F3":
                    fid = "none"
            labels.append({"index": idx, "flaw_id": fid,
                           "none_type": None if fid != "none" else ("generic" if "Share code" in text
                                                                    else "substantive"),
                           "note": "fake"})
        assert len(labels) == n
        if self.invalid_first and self.calls == 1:
            labels = labels[:-1]  # missing index -> invalid
        out = "```json\n" + json.dumps({"labels": labels}) + "\n```"
        return AdapterRunResult(success=True, output=out, model=request.model,
                                input_tokens=5000, output_tokens=800, cached_input_tokens=0)


# ── fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def items():
    return common.load_items()


@pytest.fixture
def bench_copy(tmp_path):
    dst = tmp_path / "bench"
    dst.mkdir()
    for sub in ("artifacts", "ground_truth"):
        (dst / sub).mkdir()
        for p in common.visible_files(common.BENCH_DIR / sub):
            shutil.copy(p, dst / sub / p.name)
    shutil.copy(common.BENCH_DIR / "ANALYSIS_PLAN.md", dst / "ANALYSIS_PLAN.md")
    return dst


def run_fake_reviews(data_dir: Path, items, models=("gpt-5.6-sol", "gpt-5.6-luna"), adapter=None,
                     item_ids=PILOT_ITEMS, **kw):
    adapter = adapter or FakeReviewer()
    runner = run_reviews.Runner(
        data_dir=data_dir, effort="medium", adapter_factory=lambda m, e: adapter,
        items=items, workspace_root=data_dir / "ws", stdout=False, sleep=_no_sleep, **kw)
    keys = run_reviews.schedule(list(models), list(item_ids), 1, list(common.ALL_CALLS))
    stats = asyncio.run(runner.run(keys, concurrency=1))
    return runner, adapter, stats


async def _no_sleep(_delay):
    return None


# ── freeze ──────────────────────────────────────────────────────────────


def test_freeze_validates_real_benchmark():
    problems, summary = freeze.validate(common.BENCH_DIR)
    assert problems == []
    assert summary["items"] == 18 and summary["flawed_items"] == 12 and summary["clean_items"] == 6
    assert summary["flaws"] == 48
    assert set(summary["flaws_per_type"].values()) == {4}
    assert summary["t1_symmetry_instances"] == ["A01-F4", "A02-F3", "A07-F2"]


def test_freeze_writes_and_verifies_manifest(bench_copy, capsys):
    assert freeze.main(["--bench-dir", str(bench_copy)]) == 0
    manifest = common.read_json(bench_copy / common.MANIFEST_NAME)
    assert len(manifest["files"]) == 18 + 18 + 1
    assert manifest["counts"]["flaws_per_type"]["T1"] == 4
    common.verify_manifest(bench_copy)
    # unchanged -> verified, file untouched
    before = (bench_copy / common.MANIFEST_NAME).read_text()
    assert freeze.main(["--bench-dir", str(bench_copy)]) == 0
    assert (bench_copy / common.MANIFEST_NAME).read_text() == before
    # changed artifact -> verify fails, re-freeze aborts without --force
    p = bench_copy / "artifacts" / "A01-clean.md"
    p.write_text(p.read_text() + "\nextra\n")
    with pytest.raises(common.ManifestError):
        common.verify_manifest(bench_copy)
    assert freeze.main(["--bench-dir", str(bench_copy)]) == 1
    assert freeze.main(["--bench-dir", str(bench_copy), "--force"]) == 0
    common.verify_manifest(bench_copy)


def test_freeze_aborts_on_non_verbatim_quote(bench_copy):
    p = bench_copy / "ground_truth" / "A02-flawed.json"
    gt = json.loads(p.read_text())
    gt["flaws"][0]["quote"] = gt["flaws"][0]["quote"] + " (paraphrased)"
    p.write_text(json.dumps(gt))
    problems, _ = freeze.validate(bench_copy)
    assert any("verbatim" in x for x in problems)
    assert freeze.main(["--bench-dir", str(bench_copy)]) == 1
    assert not (bench_copy / common.MANIFEST_NAME).exists()


def test_freeze_aborts_on_type_count(bench_copy):
    p = bench_copy / "ground_truth" / "A03-flawed.json"
    gt = json.loads(p.read_text())
    gt["flaws"][0]["type"] = "S1"  # was S2 -> S1 x5, S2 x3
    p.write_text(json.dumps(gt))
    problems, _ = freeze.validate(bench_copy)
    assert any("type S1 occurs 5" in x for x in problems)
    assert any("type S2 occurs 3" in x for x in problems)


# ── prompts / parsing / merging ─────────────────────────────────────────


def test_system_prompts_match_engine_resolution():
    from backend.adapters.mock import MockAdapter
    from backend.orchestrator.engine import LoopEngine

    engine = LoopEngine(adapter=MockAdapter())
    run_state = types.SimpleNamespace(current_iteration=1, task=common.TASK_STATEMENT, config={})
    for tag, (role, ref) in common.CALL_SPECS.items():
        info = common.load_system_prompt(ref, role)
        engine_text = engine._resolve_role_system_prompt(role, ref, run_state)
        assert info.text == engine_text, tag
        assert "REQUIRED OUTPUT FORMAT" in info.text
        assert info.version and len(info.sha256) == 64


def test_user_prompt_is_task_statement_plus_artifact(items):
    up = common.build_user_prompt(items["A01-flawed"].artifact_text)
    assert up.startswith(common.TASK_STATEMENT)
    assert items["A01-flawed"].artifact_text.strip() in up
    assert "A01-F" not in up and "planted" not in up.lower()


def test_resolve_provider():
    assert common.resolve_provider("gpt-5.6-sol") == ("codex", "gpt-5.6-sol")
    assert common.resolve_provider("claude-opus-4-1")[0] == "claude"
    assert common.resolve_provider("gemini:gemini-3-pro") == ("gemini", "gemini-3-pro")
    with pytest.raises(ValueError):
        common.resolve_provider("mystery-model")


def test_make_adapter_disables_tools():
    ad = common.make_adapter("gpt-5.6-sol", "medium")
    req = common.make_request("sys", "user", model="gpt-5.6-sol", effort="medium",
                              timeout_seconds=10, workspace="/tmp/x", role="reviewer")
    cmd = ad.build_command(req)
    assert "read-only" in cmd and "-m" in cmd and "gpt-5.6-sol" in cmd
    assert "model_reasoning_effort=medium" in cmd
    assert cmd[cmd.index("--cd") + 1] == "/tmp/x"
    # Default (frozen pilot condition): the backend adapter itself adds
    # exactly web_search="disabled" and keeps the user config; no wrapper.
    assert cmd[0] == "codex" and cmd[1] == "exec"
    i = cmd.index('web_search="disabled"')
    assert cmd[i - 1] == "-c"
    assert "--ignore-user-config" not in cmd and "--disable" not in cmd
    assert cmd.count('web_search="disabled"') == 1
    plain = common.make_adapter("gpt-5.6-sol", "medium", codex_extra=[])
    pcmd = plain.build_command(req)
    assert pcmd[0] == "codex" and 'web_search="disabled"' not in pcmd
    # strict: the backend's full tools-off set, user config ignored
    strict = common.make_adapter("gpt-5.6-sol", "medium",
                                 codex_extra=common.codex_extra_args("disabled", "strict"))
    scmd = strict.build_command(req)
    assert "--ignore-user-config" in scmd and "shell_tool" in scmd
    assert common.codex_extra_args("disabled", "strict") != common.codex_extra_args()


def test_codex_wrapper_inserts_args(tmp_path):
    import subprocess

    fake = tmp_path / "fakecodex"
    fake.write_text("#!/bin/sh\nfor a in \"$@\"; do echo \"$a\"; done\n")
    fake.chmod(0o755)
    env = dict(os.environ, MIW_BENCH_CODEX_REAL=str(fake),
               MIW_BENCH_CODEX_EXTRA_ARGS=json.dumps(["-c", 'web_search="disabled"']))
    out = subprocess.run([sys.executable, str(common.CODEX_WRAPPER), "exec", "--json", "-"],
                         env=env, capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[:5] == ["exec", "-c", 'web_search="disabled"', "--json", "-"]
    out = subprocess.run([sys.executable, str(common.CODEX_WRAPPER), "--version"],
                         env=env, capture_output=True, text=True, check=True).stdout.split("\n")
    assert out[0] == "--version"


def test_parse_output_uses_platform_parser():
    p = common.parse_output(fake_review_output(["F1", "F2"]))
    assert p["method"] == "json" and p["n_critiques"] == 3
    assert p["critiques"][0]["severity"] == "high"
    assert p["stripping_changed_parse"] is False
    none = common.parse_output("I could not review this.")
    assert none["method"] == "none" and none["n_critiques"] == 0


def test_merge_calls_lens_identity():
    same = {"description": "Effect sizes are missing for the main comparison of heads",
            "severity": "medium", "category": "statistics"}
    crit = {"rig1": [same], "rig2": [dict(same)], "rig3": [dict(same)]}
    by_tag, idx = common.merge_calls(crit, ("rig1", "rig2", "rig3"), lens_identity="tag")
    assert len(by_tag.result.critiques) == 1
    assert by_tag.result.critiques[0].severity.value == "high"  # escalated: 3 distinct calls
    assert sorted(idx) == [("rig1", 0), ("rig2", 0), ("rig3", 0)]
    by_role, _ = common.merge_calls(crit, ("rig1", "rig2", "rig3"), lens_identity="role")
    assert by_role.result.critiques[0].severity.value == "medium"  # one role -> no escalation


def test_merge_defaults_pinned_to_freeze_time():
    """The pre-specified severity outcomes use the platform merge's default
    settings at freeze time (jaccard 0.5), whatever the platform default is now."""
    from backend.orchestrator import similarity
    assert similarity.DEFAULT_SIMILARITY_METHOD == "llm"     # platform moved on ...
    assert similarity.DEFAULT_THRESHOLDS["jaccard"] == 0.1
    assert common.DEFAULT_MERGE_METHOD == "jaccard"          # ... the harness did not
    assert common.DEFAULT_MERGE_THRESHOLD == 0.5
    args = analyze.build_parser().parse_args([])
    assert args.merge_method == "jaccard" and args.merge_threshold is None
    assert common.freeze_time_threshold(args.merge_method, args.merge_threshold) == 0.5
    assert common.freeze_time_threshold("tfidf") == 0.3
    assert common.freeze_time_threshold("tfidf", 0.075) == 0.075
    # merge_calls without a threshold merges at 0.5: this pair has Jaccard 3/7 = 0.43.
    a = {"description": "effect sizes are missing", "severity": "medium"}
    b = {"description": "effect sizes are not reported anywhere", "severity": "medium"}
    out, _ = common.merge_calls({"rig1": [a], "adv": [b]}, ("rig1", "adv"))
    assert out.threshold == 0.5 and len(out.result.critiques) == 2
    out01, _ = common.merge_calls({"rig1": [a], "adv": [b]}, ("rig1", "adv"), threshold=0.1)
    assert len(out01.result.critiques) == 1


# ── run_reviews ─────────────────────────────────────────────────────────


def test_schedule_is_balanced():
    keys = run_reviews.schedule(["m1", "m2"], ["A01-flawed", "A05-clean", "A03-clean"], 2, ["rig1", "adv"])
    assert len(keys) == 2 * 3 * 2 * 2
    first_half = keys[:12]
    assert {k.repeat for k in first_half} == {1}
    # Units are contiguous: model alternates every len(calls) keys.
    assert [k.model for k in keys[:4]] == ["m1", "m1", "m2", "m2"]
    assert keys == run_reviews.schedule(["m1", "m2"], ["A01-flawed", "A05-clean", "A03-clean"], 2,
                                        ["rig1", "adv"])


def test_run_reviews_records_and_resume(tmp_path, items):
    data = tmp_path / "data"
    runner, adapter, stats = run_fake_reviews(data, items)
    assert stats["succeeded"] == 32 and stats["failed"] == 0
    assert adapter.calls == 32
    rec = common.load_review(data, "gpt-5.6-sol", "A01-flawed", 1, "adv")
    assert rec["success"] is True
    assert rec["request"]["prompt_ref"] == "adversarial_reviewer/adversarial_reviewer"
    assert rec["request"]["system_prompt_version"] == "2.0.0"
    assert rec["request"]["allow_tools"] is False
    assert rec["request"]["model"] == "gpt-5.6-sol" and rec["request"]["reasoning_effort"] == "medium"
    assert rec["request"]["cli_version"] == "fake-cli 1.0"
    assert len(rec["request"]["user_prompt_sha256"]) == 64
    assert rec["tokens"] == {"input": 1000, "output": 200, "cached_input": 400, "reasoning_output": 50,
                             "raw_usage": {"reasoning_output_tokens": 50}}
    assert rec["parse"]["method"] == "json" and len(rec["critiques"]) == 3
    assert rec["started_utc"].endswith("Z") and rec["finished_utc"].endswith("Z")
    assert len(rec["attempts"]) == 1
    log = (data / "logs" / "run_reviews.log").read_text().splitlines()
    assert sum(1 for line in log if " OK " in line) == 32
    # Workspaces are removed.
    assert not any((data / "ws").iterdir())
    # Resume: nothing is re-run.
    runner2, adapter2, stats2 = run_fake_reviews(data, items)
    assert adapter2.calls == 0 and stats2["skipped_cached"] == 32


def test_run_reviews_retries_failures(tmp_path, items):
    data = tmp_path / "data"
    ad = FakeReviewer(fail_first=2)
    _, _, stats = run_fake_reviews(data, items, models=("gpt-5.6-sol",), adapter=ad,
                                   item_ids=["A01-flawed"], max_attempts=3)
    assert stats["failed"] == 0 and stats["succeeded"] == 8
    first = common.load_review(data, "gpt-5.6-sol", "A01-flawed", 1, run_reviews.schedule(
        ["gpt-5.6-sol"], ["A01-flawed"], 1, list(common.ALL_CALLS))[0].tag)
    assert [a["success"] for a in first["attempts"]] == [False, False, True]


def test_run_reviews_failed_then_resumed(tmp_path, items):
    data = tmp_path / "data"
    ad = FakeReviewer(fail_first=1000, error="[auth] not logged in")
    _, _, stats = run_fake_reviews(data, items, models=("gpt-5.6-sol",), adapter=ad,
                                   item_ids=["A05-clean"], max_attempts=3)
    assert stats["failed"] == 8
    assert ad.calls == 8  # auth errors are not retried
    rec = common.load_review(data, "gpt-5.6-sol", "A05-clean", 1, "bio")
    assert rec["success"] is False and rec["error"].startswith("[auth]") and rec["critiques"] == []
    # A later invocation retries failed cached calls and keeps the attempt history.
    _, ad2, stats2 = run_fake_reviews(data, items, models=("gpt-5.6-sol",), item_ids=["A05-clean"])
    assert stats2["succeeded"] == 8 and ad2.calls == 8
    rec = common.load_review(data, "gpt-5.6-sol", "A05-clean", 1, "bio")
    assert rec["success"] is True and [a["success"] for a in rec["attempts"]] == [False, True]


def test_run_reviews_hard_timeout(tmp_path, items):
    data = tmp_path / "data"
    ad = FakeReviewer(sleep=0.5)
    _, _, stats = run_fake_reviews(data, items, models=("gpt-5.6-sol",), adapter=ad,
                                   item_ids=["A05-clean"], max_attempts=1, timeout=0.05,
                                   hard_timeout_grace=0.05)
    assert stats["failed"] == 8
    rec = common.load_review(data, "gpt-5.6-sol", "A05-clean", 1, "rig1")
    assert rec["error"].startswith("[timeout]")


def test_run_reviews_cache_mismatch_not_overwritten(tmp_path, items):
    data = tmp_path / "data"
    run_fake_reviews(data, items, models=("gpt-5.6-sol",), item_ids=["A05-clean"])
    ad = FakeReviewer()
    runner = run_reviews.Runner(data_dir=data, effort="low", adapter_factory=lambda m, e: ad,
                                items=items, workspace_root=data / "ws", stdout=False, sleep=_no_sleep)
    keys = run_reviews.schedule(["gpt-5.6-sol"], ["A05-clean"], 1, list(common.ALL_CALLS))
    stats = asyncio.run(runner.run(keys, concurrency=2))
    assert stats["cache_mismatch"] == 8 and ad.calls == 0
    assert common.load_review(data, "gpt-5.6-sol", "A05-clean", 1, "rig1")["request"]["reasoning_effort"] == "medium"
    # Different codex extra args (web search left on) also count as a mismatch.
    rec = common.load_review(data, "gpt-5.6-sol", "A05-clean", 1, "rig1")
    assert rec["request"]["cli_extra_args"] == ["-c", 'web_search="disabled"']
    runner = run_reviews.Runner(data_dir=data, effort="medium", adapter_factory=lambda m, e: ad,
                                items=items, workspace_root=data / "ws", stdout=False, sleep=_no_sleep,
                                codex_extra=[])
    stats = asyncio.run(runner.run(keys, concurrency=2))
    assert stats["cache_mismatch"] == 8 and ad.calls == 0


# ── judge ───────────────────────────────────────────────────────────────


def test_validate_labels():
    ok, probs = judge_mod.validate_labels(
        {"labels": [{"index": 0, "flaw_id": "A01-F1", "none_type": None},
                    {"index": 1, "flaw_id": "none", "none_type": "generic"}]}, 2, ["A01-F1"])
    assert probs == [] and ok[0]["flaw_id"] == "A01-F1" and ok[1]["none_type"] == "generic"
    _, probs = judge_mod.validate_labels({"labels": [{"index": 0, "flaw_id": "A01-F9"}]}, 2, ["A01-F1"])
    assert any("invalid flaw_id" in p for p in probs) and any("missing" in p for p in probs)
    _, probs = judge_mod.validate_labels(
        {"labels": [{"index": 0, "flaw_id": "none", "none_type": "meh"},
                    {"index": 0, "flaw_id": "none", "none_type": "generic"}]}, 1, [])
    assert any("none_type" in p for p in probs)
    _, probs = judge_mod.validate_labels(
        {"labels": [{"index": 0, "flaw_id": "none", "none_type": "generic"},
                    {"index": 0, "flaw_id": "none", "none_type": "generic"}]}, 1, [])
    assert any("more than once" in p for p in probs)
    ok, probs = judge_mod.parse_judge_output("no json here", 1, [])
    assert ok is None and probs


def test_judge_end_to_end(tmp_path, items):
    data = tmp_path / "data"
    run_fake_reviews(data, items)
    fj = FakeJudge(invalid_first=True)
    j = judge_mod.Judge(judge_model="gpt-5.6-sol", effort="high", data_dir=data, items=items,
                        adapter_factory=lambda m, e: fj, workspace_root=data / "ws", stdout=False,
                        sleep=_no_sleep)
    units = judge_mod.discover_units(data)
    assert len(units) == 4
    stats = asyncio.run(j.run(units, concurrency=2))
    assert stats["judged"] == 4 and stats["invalid"] == 0
    rec = common.read_json(common.judgment_path(data, "gpt-5.6-sol", "gpt-5.6-sol", "A01-flawed", 1))
    assert rec["valid"] is True
    n = rec["request"]["n_critiques"]
    assert n == sum(len(v) for v in FAKE_FINDS.values()) + 8
    assert sorted(lab["index"] for lab in rec["labels"]) == list(range(n))
    assert {c["tag"] for c in rec["critiques"]} == set(common.ALL_CALLS)
    # Exactly one unit got the invalid first answer and was re-asked.
    reasked = [common.read_json(p)["reasked"] for p in (data / "judgments").rglob("r1.json")]
    assert sum(reasked) == 1
    # Prompts carry no call/lens identity and no severity.
    for up in fj.prompts:
        crit = up.split("CRITIQUES (N = ", 1)[1]
        assert not re.search(r"\b(rig\d|cmb\d|adv|bio|adversarial|reviewer_combined|severity|high)\b", crit)
    clean_prompt = [p for p in fj.prompts if "control analysis" in p]
    assert clean_prompt and "A05-F" not in clean_prompt[0].split("CRITIQUES", 1)[0]
    # Deterministic shuffle; matches stored seed.
    recs = common.unit_records(data, "gpt-5.6-sol", "A01-flawed", 1)
    sh1, seed = judge_mod.shuffle_critiques(judge_mod.collect_critiques(recs), "gpt-5.6-sol", "A01-flawed", 1)
    assert seed == rec["request"]["shuffle_seed"]
    assert [c["description"] for c in sh1] == [c["description"] for c in rec["critiques"]]
    # Cached second run makes no calls.
    calls_before = fj.calls
    stats2 = asyncio.run(j.run(units, concurrency=2))
    assert fj.calls == calls_before and stats2["skipped_cached"] == 4


# ── analyze ─────────────────────────────────────────────────────────────


def _judge_all(data, items, judge_model, **kw):
    fj = FakeJudge(**kw)
    j = judge_mod.Judge(judge_model=judge_model, effort="high", data_dir=data, items=items,
                        adapter_factory=lambda m, e: fj, workspace_root=data / "ws", stdout=False,
                        sleep=_no_sleep)
    asyncio.run(j.run(judge_mod.discover_units(data), concurrency=2))


def test_analyze_end_to_end(tmp_path, items):
    data = tmp_path / "data"
    run_fake_reviews(data, items, item_ids=["A01-flawed", "A01-clean", "A05-clean", "A05-flawed"])
    _judge_all(data, items, "gpt-5.6-sol")
    _judge_all(data, items, "gpt-5.5", disagree=True)
    out = tmp_path / "results"
    rc = analyze.main(["--judges", "gpt-5.6-sol,gpt-5.5", "--data-dir", str(data), "--out-dir", str(out),
                       "--bootstrap", "200", "--bootstrap-descriptive", "100"])
    assert rc == 0
    res = common.read_json(out / "benchmark_results.json")
    rec = res["primary"]["recall"]
    for c, v in EXPECTED_RECALL.items():
        assert rec["pooled"][c]["estimate"] == pytest.approx(v)
        assert rec["gpt-5.6-sol"][c]["n_values"] == 2 and rec["gpt-5.6-sol"][c]["n_clusters"] == 2
    con = res["primary"]["contrasts"]["pooled"]
    assert con["C5-C1"]["estimate"] == pytest.approx(0.75)
    assert con["C4-C2"]["estimate"] == pytest.approx(0.0)
    assert con["C4-C2"]["p_boot"] == pytest.approx(1.0)
    assert con["C5-C1"]["p_holm"] >= con["C5-C1"]["p_boot"]
    # Severity-aware: flaw critiques are HIGH -> same as recall here.
    assert res["primary"]["severity_recall"]["pooled"]["C5"]["estimate"] == pytest.approx(1.0)
    # Unique lens contributions in C5: F1 only rig1, F4 only bio, F2/F3 only adv.
    u = res["primary"]["unique_lens_contribution_C5"]["pooled"]
    assert u["only_rig1"]["count"] == 4 and u["only_bio"]["count"] == 4 and u["only_adv"]["count"] == 8
    # Symmetry instance A01-F4 is found only by bio -> C5 rate 1, C1 rate 0.
    sym = res["primary"]["t1_symmetry"]
    assert sym["flaw_ids"] == ["A01-F4", "A02-F3", "A07-F2"]
    assert sym["detection"]["pooled"]["C5"]["estimate"] == pytest.approx(1.0)
    assert sym["detection"]["pooled"]["C1"]["estimate"] == pytest.approx(0.0)
    # Fake reviewer is item-blind -> no discrimination (AUROC 0.5).
    assert res["primary"]["discrimination_auroc"]["pooled"]["C5"]["severity_score"]["estimate"] == pytest.approx(0.5)
    comp = res["primary"]["unmatched_composition"]["pooled"]["C1"]["clean"]
    assert comp["matched"]["count"] == 0 and comp["none_generic"]["count"] == 4
    # Judge 2 drops F3 -> C2 recall 0.25; kappa < 1.
    s2 = res["sensitivity"]["judge:gpt-5.5"]["recall"]["pooled"]["C2"]["estimate"]
    assert s2 == pytest.approx(0.25)
    inter = res["sensitivity"]["intersection"]["recall"]["pooled"]["C2"]["estimate"]
    assert inter == pytest.approx(0.25)
    ka = res["judge_agreement"]["critique_level_flaw_id"]["kappa"]
    assert ka is not None and 0 < ka < 1
    assert res["cost_by_condition"]["pooled"]["C5"]["mean_input"] == 3000
    assert res["call_stats"]["ALL|ALL"]["parse_methods"] == {"json": 64}
    for name in ("recall_by_condition", "contrasts", "recall_by_type", "discrimination",
                 "cost_by_condition", "per_unit_detections", "unmatched_composition"):
        assert (out / f"{name}.csv").exists()


def test_stat_helpers():
    assert analyze.auroc([2, 3], [1, 1]) == 1.0
    assert analyze.auroc([1], [1]) == 0.5
    adj = analyze.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    from collections import Counter
    assert analyze.cohen_kappa_from_counts(Counter({("x", "x"): 5, ("y", "y"): 5})) == pytest.approx(1.0)
    assert analyze.cohen_kappa_from_counts(Counter({("x", "y"): 5, ("y", "x"): 5})) == pytest.approx(-1.0)
    assert analyze.bootstrap_p_two_sided([0.1] * 99) == pytest.approx(0.02)


# ── merge_eval ──────────────────────────────────────────────────────────


def _instances():
    I = merge_eval.Instance
    a = "no multiple testing correction across the 144 per head tests"
    b = "the 144 per head tests have no multiple testing correction applied"
    c = "attention is not symmetric since query and key projections differ"
    d = "coexpression may explain the attention enrichment rather than regulation"
    return [
        I("m", "A01-flawed", "A01", 1, True, [a, b, c, d],
          ["reviewer", "adversarial_reviewer", "bio_plausibility_checker", "adversarial_reviewer"],
          ["A01-F1", "A01-F1", "A01-F4", "A01-F2"]),
        I("m", "A02-flawed", "A02", 1, True, [a, b, d],
          ["reviewer", "bio_plausibility_checker", "adversarial_reviewer"],
          ["A02-F1", "A02-F1", "none"]),
    ]


def test_merge_eval_metrics():
    insts = _instances()
    pairs = merge_eval.gt_pairs(insts[0])
    assert pairs[(0, 1)] is True and pairs[(0, 2)] is False and (1, 3) not in pairs  # same lens excluded
    res = merge_eval.evaluate_matrix_method(insts, "tfidf")
    assert res["held_out"]["tp"] >= 1 and len(res["folds"]) == 2
    # The reference threshold is the pre-calibration default, not the platform's.
    assert res["default_threshold"] == 0.3
    assert merge_eval.adjusted_rand_index([1, 1, 2, 2], [5, 5, 6, 6]) == pytest.approx(1.0)
    assert merge_eval.adjusted_rand_index([1, 2], [1, 2]) == 1.0
    esc = merge_eval.escalation_stats([(insts[0], [[0, 1], [2], [3]])])
    assert esc["flaws_raised_by_2plus_lenses"] == 1 and esc["escalation_recall"] == 1.0
    assert esc["multi_lens_groups"]["share_majority_flaw"] == 1.0


def test_merge_eval_llm_cached(tmp_path):
    insts = _instances()
    calls = {"n": 0}

    class FakeAdj:
        async def run(self, request):
            calls["n"] += 1
            n = request.prompt_bundle.user_prompt.count("\n[")
            groups = [[0, 1]] + [[i] for i in range(2, n)]
            return AdapterRunResult(success=True, output="```json\n" + json.dumps({"groups": groups}) + "\n```",
                                    input_tokens=100, output_tokens=10)

    adjs = asyncio.run(merge_eval.adjudicate_all(insts, adjudicator="gpt-5.6-sol", effort="medium",
                                                 data_dir=tmp_path, adapter_factory=lambda m, e: FakeAdj(),
                                                 workspace_root=tmp_path / "ws"))
    assert calls["n"] == 2
    res = merge_eval.evaluate_llm(insts, adjs)
    assert res["held_out"]["tp"] == 2 and res["held_out"]["fp"] == 0
    asyncio.run(merge_eval.adjudicate_all(insts, adjudicator="gpt-5.6-sol", effort="medium",
                                          data_dir=tmp_path, adapter_factory=lambda m, e: FakeAdj(),
                                          workspace_root=tmp_path / "ws"))
    assert calls["n"] == 2  # cached


def test_merge_eval_embedding_skips_cleanly(monkeypatch):
    monkeypatch.setattr(merge_eval.sim, "embedding_backend_available", lambda: False)
    enc, reason = merge_eval.embedding_encoder_or_reason("x")
    assert enc is None and "not installed" in reason
    assert merge_eval.choose_default({"embedding": {"skipped": True}})["method"] is None


def test_cli_capture_records_tool_items():
    import backend.adapters.codex_cli as cc

    common.install_cli_capture()
    common.install_cli_capture()  # idempotent
    assert getattr(cc.run_cli_process, "_miw_capture", False)
    lines = "\n".join([
        json.dumps({"type": "item.completed", "item": {"id": "1", "type": "reasoning", "text": "x"}}),
        json.dumps({"type": "item.completed", "item": {"id": "2", "type": "command_execution",
                                                        "command": "ls /", "aggregated_output": "a" * 5000}}),
        json.dumps({"type": "item.completed", "item": {"id": "3", "type": "agent_message", "text": "ok"}}),
    ])

    async def go():
        holder = common.begin_capture()
        res = await cc.run_cli_process(["cat"], lines, timeout=10)
        return holder, res

    holder, res = asyncio.run(go())
    assert res.returncode == 0 and holder == [res.stdout]
    items = common.extract_tool_items(holder)
    assert len(items) == 1 and items[0]["type"] == "command_execution" and items[0]["command"] == "ls /"
    assert "chars cut" in items[0]["aggregated_output"]


# ── revision-2 code-review regressions ─────────────────────────────────


class ProseReviewer(FakeReviewer):
    """Answers with prose that has no critique format (or an echoed template)."""

    def __init__(self, text="I reviewed it and it looks broadly fine.", **kw):
        super().__init__(**kw)
        self.text = text

    async def run(self, request):
        self.calls += 1
        return AdapterRunResult(success=True, output=self.text, model=request.model,
                                input_tokens=10, output_tokens=5)


def test_unparsed_output_is_a_failed_attempt_and_excluded(tmp_path, items):
    """tests-claims-2 / consensus-1: unusable output never counts as a clean review."""
    data = tmp_path / "data"
    ad = ProseReviewer()
    _, _, stats = run_fake_reviews(data, items, models=("gpt-5.6-sol",), adapter=ad,
                                   item_ids=["A05-clean"], max_attempts=2)
    assert stats["failed"] == 8 and stats["succeeded"] == 0
    assert ad.calls == 16  # retried
    rec = common.load_review(data, "gpt-5.6-sol", "A05-clean", 1, "rig1")
    assert rec["success"] is False and rec["error"].startswith("[unparsed]")
    assert rec["parse"]["method"] == "none" and rec["raw_output"]
    assert [a["success"] for a in rec["attempts"]] == [False, False]
    units, meta = analyze.load_units(data, [], items)
    assert units == [] and meta["incomplete_units"]
    # An echoed contract template is unparsed too.
    template = ('```json\n{"critiques": [{"severity": "critical|high|medium|low|info", '
                '"category": "<short>", "description": "<the problem>"}]}\n```')
    data2 = tmp_path / "data2"
    _, _, stats2 = run_fake_reviews(data2, items, models=("gpt-5.6-sol",),
                                    adapter=ProseReviewer(template), item_ids=["A05-clean"],
                                    max_attempts=1)
    assert stats2["failed"] == 8
    rec2 = common.load_review(data2, "gpt-5.6-sol", "A05-clean", 1, "adv")
    assert "all_items_unreadable" in rec2["error"]


def test_legacy_success_record_with_none_parse_is_excluded(tmp_path, items):
    data = tmp_path / "data"
    run_fake_reviews(data, items, models=("gpt-5.6-sol",), item_ids=["A05-clean"])
    p = common.review_path(data, "gpt-5.6-sol", "A05-clean", 1, "bio")
    rec = common.read_json(p)
    rec["parse"]["method"] = "none"
    rec["critiques"] = []
    common.atomic_write_json(p, rec)
    units, meta = analyze.load_units(data, [], items)
    assert units == []
    assert meta["incomplete_units"][0]["unparsed"] == ["bio"]


def test_cache_signature_includes_codex_agents_md(tmp_path, items, monkeypatch):
    """tests-claims-1: a clean CODEX_HOME never reuses records made with AGENTS.md."""
    home_a = tmp_path / "homeA"
    home_b = tmp_path / "homeB"
    home_a.mkdir()
    home_b.mkdir()
    (home_a / "AGENTS.md").write_text("You are a senior engineer.")
    data = tmp_path / "data"
    run_fake_reviews(data, items, models=("gpt-5.6-sol",), item_ids=["A05-clean"],
                     codex_home=str(home_a))
    rec = common.load_review(data, "gpt-5.6-sol", "A05-clean", 1, "rig1")
    assert rec["request"]["codex_agents_md_sha256"] == common.sha256_file(home_a / "AGENTS.md")
    _, ad, stats = run_fake_reviews(data, items, models=("gpt-5.6-sol",), item_ids=["A05-clean"],
                                    codex_home=str(home_b))
    assert stats["cache_mismatch"] == 8 and stats["skipped_cached"] == 0 and ad.calls == 0
    _, ad2, stats2 = run_fake_reviews(data, items, models=("gpt-5.6-sol",), item_ids=["A05-clean"],
                                      codex_home=str(home_a))
    assert stats2["skipped_cached"] == 8 and ad2.calls == 0
    # Records made before the field existed are compared on request.codex_env.
    legacy = dict(rec["request"])
    legacy.pop("codex_agents_md_sha256")
    legacy.pop("codex_agents_override_md_sha256")
    sig = {k: rec["request"][k] for k in ("codex_agents_md_sha256",
                                          "codex_agents_override_md_sha256")}
    assert common.signature_matches(legacy, sig)
    legacy["codex_env"] = {"AGENTS.md": "other"}
    assert not common.signature_matches(legacy, sig)


def test_analyze_excludes_mixed_condition_units(tmp_path, items):
    data = tmp_path / "data"
    run_fake_reviews(data, items, models=("gpt-5.6-sol",),
                     item_ids=["A05-clean", "A01-flawed", "A01-clean"])
    p = common.review_path(data, "gpt-5.6-sol", "A05-clean", 1, "rig2")
    rec = common.read_json(p)
    rec["request"]["codex_env"] = dict(rec["request"].get("codex_env") or {}, **{"AGENTS.md": "x"})
    common.atomic_write_json(p, rec)
    units, meta = analyze.load_units(data, [], items)
    assert {u.item for u in units} == {"A01-flawed", "A01-clean"}
    assert meta["excluded_mixed_condition_units"][0]["calls"] == ["rig2"]


def test_cost_summary_includes_retries(tmp_path, items):
    """tests-claims-10: failed attempts' wall time and tokens are reported."""
    data = tmp_path / "data"
    run_fake_reviews(data, items, models=("gpt-5.6-sol",), item_ids=["A05-clean"])
    p = common.review_path(data, "gpt-5.6-sol", "A05-clean", 1, "cmb2")
    rec = common.read_json(p)
    rec["attempts"].insert(0, {"success": False, "wall_seconds": 89.4, "error": "[empty_output] x",
                               "tokens": {"input": 7, "output": 3, "cached_input": 0}})
    common.atomic_write_json(p, rec)
    units, _ = analyze.load_units(data, [], items)
    cost = analyze.cost_summary(units)["gpt-5.6-sol"]["C4"]
    assert cost["mean_wall_sum_incl_retries"] == pytest.approx(cost["mean_wall_sum"] + 89.4)
    assert cost["mean_input_incl_retries"] == cost["mean_input"] + 7
    assert cost["mean_failed_attempts"] == 1


def test_merge_calls_same_lens_rule_is_per_call():
    """consensus-6/-7: two critiques of one call never merge; copies of a role
    across calls still do (role identity)."""
    a = {"description": "Effect sizes are missing for the main comparison of heads",
         "severity": "medium", "category": "s"}
    b = {"description": "Effect sizes are missing for the main comparison of layers",
         "severity": "medium", "category": "s"}
    one_call, _ = common.merge_calls({"rig1": [a, b]}, ("rig1",))
    assert len(one_call.result.critiques) == 2
    across, _ = common.merge_calls({"rig1": [a], "rig2": [dict(a)]}, ("rig1", "rig2"),
                                   lens_identity="role")
    assert len(across.result.critiques) == 1


def test_parse_output_reports_contract_health():
    two = fake_review_output(["F1"]) + "\n```json\n{\"critiques\": []}\n```\n"
    p = common.parse_output(two)
    assert p["n_contract_blocks"] == 2 and p["dropped_earlier"] == 2 and p["n_critiques"] == 0
    ok = common.parse_output(fake_review_output(["F1"]))
    assert ok["n_contract_blocks"] == 1 and ok["n_dropped"] == 0 and ok["invalid_reason"] == ""


def test_merge_eval_cache_signature_and_strict_regroup(tmp_path):
    """tests-claims-4 / consensus-5."""
    insts = _instances()
    calls = {"n": 0}
    answers = {"ans": None}

    class FakeAdj:
        async def run(self, request):
            calls["n"] += 1
            n = request.prompt_bundle.user_prompt.count("\n[")
            groups = answers["ans"](n)
            return AdapterRunResult(success=True,
                                    output="```json\n" + json.dumps({"groups": groups}) + "\n```",
                                    input_tokens=100, output_tokens=10)

    answers["ans"] = lambda n: [[1, 2]] + [[i] for i in range(3, n + 1)]  # 1-based
    kw = dict(adjudicator="gpt-5.6-sol", data_dir=tmp_path,
              adapter_factory=lambda m, e: FakeAdj(), workspace_root=tmp_path / "ws")
    adjs = asyncio.run(merge_eval.adjudicate_all(insts, effort="medium", **kw))
    assert calls["n"] == 2
    for rec in adjs.values():
        assert rec["success"] and rec["repaired"] and rec["groups"][0] == [0, 1]
    res = merge_eval.evaluate_llm(insts, adjs)
    assert len(res["repaired_instances"]) == 2 and res["efforts_used"] == {"medium": 2}
    # same effort -> cached; different effort -> new calls (not relabelled)
    asyncio.run(merge_eval.adjudicate_all(insts, effort="medium", **kw))
    assert calls["n"] == 2
    adjs_hi = asyncio.run(merge_eval.adjudicate_all(insts, effort="high", **kw))
    assert calls["n"] == 4
    assert merge_eval.evaluate_llm(insts, adjs_hi)["efforts_used"] == {"high": 2}
    # an invalid partition is retried once, then counted as a failure
    answers["ans"] = lambda n: [[0, 0], [n + 5]]
    adjs_bad = asyncio.run(merge_eval.adjudicate_all(insts, effort="low", **kw))
    assert calls["n"] == 8
    assert merge_eval.evaluate_llm(insts, adjs_bad)["failed_instances"]
