"""Tests for the X4 executed case study (run_case_study.py, trace_numbers.py).

Offline: the end-to-end tests drive the REAL runner / LoopEngine / consensus
panel / CodeExecutor (sandbox_exec on macOS) with a fake adapter, so no model
is called.

Run from automation/mi-workbench:
    TMPDIR="/Volumes/Crucial X6/tmp_miw" PYTHONDONTWRITEBYTECODE=1 \
    /Users/ihorkendiukhov/anaconda3/envs/mi_workbench/bin/python -m pytest \
        experiments/case_study/agent_runs/test_case_study_agent_runs.py -q -p no:cacheprovider
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import zipfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import run_case_study as rcs  # noqa: E402
import trace_numbers as tn  # noqa: E402

from backend.adapters.mock import MockAdapter  # noqa: E402
from backend.models import AdapterRunResult  # noqa: E402
from backend.orchestrator.sandbox import seatbelt as _sb  # noqa: E402

_SB_OK, _SB_WHY = _sb.sandbox_exec_available()
needs_sandbox = pytest.mark.skipif(not _SB_OK, reason=f"sandbox-exec unavailable: {_SB_WHY}")

KIT_FILES = ["genes.tsv", "cells.tsv", "tokenized_cells.npz", "attention_layer_mean.npy",
             "attention_heads_f32.npy", "attention_heads.npy", "copresence_counts.npy",
             "rank_distance_mean.npy", "coexpr_pearson.npy", "coexpr_spearman.npy",
             "trrust_edges.tsv", "dorothea_abc_edges.tsv", "attention_per_cell_log.tsv",
             "prepare_summary.json", "kit_manifest.json", "README.md", "_checkpoint/"]


# ── Task text ─────────────────────────────────────────────────────────────


def test_task_text_executed_contains_kit_question_files_and_rule():
    t = rcs.build_task_text(True)
    assert rcs.QUESTION in t
    assert str(rcs.KIT_DIR) in t
    for f in KIT_FILES:
        assert f in t, f
    assert "Only numbers produced by your executed code may be reported" in t
    for phrase in ("Effect sizes with uncertainty", "Multiple-testing control",
                   "Baselines and null models", "FROM\n  query gene i TO key gene j"):
        assert phrase.replace("\n  ", " ") in t.replace("\n  ", " "), phrase


def test_plan_only_text_is_executed_text_minus_execution_section():
    ex, po = rcs.build_task_text(True), rcs.build_task_text(False)
    assert ex.startswith(po) and len(ex) > len(po)
    removed = ex[len(po):]
    assert "COMPUTING ENVIRONMENT AND REPORTING RULE" in removed
    assert "executed" not in po.lower() and "sandbox" not in po.lower()


def test_frozen_task_text_copies_match_builder():
    for variant, flag in (("executed", True), ("plan_only", False)):
        f = HERE / f"task_text_{variant}.md"
        assert f.read_text(encoding="utf-8") == rcs.build_task_text(flag), (
            f"{f.name} is stale; regenerate with print-task --variant {variant}")


def test_task_text_contains_no_results():
    """No decimal result-like literal at all (only the float16 normal-range
    constant), and none of the author-side validation numbers."""
    for flag in (True, False):
        t = rcs.build_task_text(flag)
        decimals = {m.group(0) for m in re.finditer(r"(?<![\w.])\d*\.\d+(?:e-?\d+)?", t)}
        assert decimals <= {"6.1e-5"}, decimals
        for word in ("AUROC", "AUPRC", "auroc", "Spearman(", "supported", "not supported"):
            assert word not in t, word
    val = rcs.REPO / "experiments/case_study/results/independent_validation.json"
    if val.is_file():
        found = set()

        def walk(x):
            if isinstance(x, dict):
                for v in x.values():
                    walk(v)
            elif isinstance(x, list):
                for v in x:
                    walk(v)
            elif isinstance(x, float) and 0 < abs(x) < 1000 and x != int(x):
                found.add(f"{x:.3f}")
                found.add(f"{x:.2f}")

        walk(json.loads(val.read_text()))
        t = rcs.build_task_text(True)
        leaked = {s for s in found if re.search(rf"(?<![\w.]){re.escape(s)}(?![\d])", t)}
        assert not leaked, leaked


# ── Run configuration ─────────────────────────────────────────────────────


def test_run_config_executed_runs():
    from backend.orchestrator.code_executor import CodeExecutor
    for key in ("sol_A", "sol_B", "gpt55"):
        spec = rcs.RUN_SPECS[key]
        cfg = rcs.run_config(spec)
        assert cfg["code_execution_enabled"] is True
        assert cfg["code_execution_backend"] == "sandbox_exec"
        assert cfg["code_execution_python"] == rcs.ANALYSIS_PYTHON
        assert cfg["code_execution_read_only_paths"] == [str(rcs.KIT_DIR)]
        assert cfg["code_execution_timeout"] == 900 and cfg["code_execution_cpu_seconds"] == 1800
        assert cfg["convergence_enabled"] is False and cfg["max_iterations"] == 10
        assert cfg["reviewer_allow_tools"] is False
        assert "executor_allow_tools" not in cfg     # verified-execution default (off)
        assert "grade_at_least" not in cfg
        assert cfg["reasoning_effort"] == "medium"
        ex = CodeExecutor.from_run_config(cfg)
        assert ex.backend == "sandbox_exec" and ex.capture_text_bytes == rcs.CAPTURE_TEXT_BYTES
        assert cfg["code_execution_persist_dir"] == str(rcs.DATA_DIR / "exec_outputs" / key)
        assert ex.persist_dir == os.path.realpath(cfg["code_execution_persist_dir"])
    assert rcs.RUN_SPECS["gpt55"].model == "gpt-5.5"
    assert rcs.RUN_SPECS["sol_A"].model == rcs.RUN_SPECS["sol_B"].model == "gpt-5.6-sol"
    assert rcs.RUN_SPECS["sol_A"].run_id != rcs.RUN_SPECS["sol_B"].run_id


def test_run_config_control_has_no_execution_and_no_tools():
    cfg = rcs.run_config(rcs.RUN_SPECS["sol_planonly"])
    assert cfg["code_execution_enabled"] is False
    assert cfg["executor_allow_tools"] is False
    assert not any(k.startswith("code_execution_") and k != "code_execution_enabled" for k in cfg)
    assert rcs.RUN_SPECS["sol_planonly"].model == "gpt-5.6-sol"


# ── Fake adapter for the end-to-end tests ────────────────────────────────

EXEC_WRITEUP = """\
# MECH.md
Best layer AUROC 0.673 (results.json key auroc_best); n = 424 edges.
Layer 3 is best; 1322 genes; 95% CI computed with 2000 bootstrap resamples.
An unexecuted claim: AUPRC 0.811.

```python
import json, os
KIT = {kit!r}
rows = open(os.path.join(KIT, "genes.tsv")).read().strip().splitlines()
res = {{"n_genes": len(rows) - 1, "auroc_best": 0.67312 + 0.0 * len(rows), "n_edges": 424}}
print("auroc_best", res["auroc_best"])
print("n_edges", res["n_edges"])
json.dump(res, open("results.json", "w"))
```
"""


class FakeAdapter(MockAdapter):
    """Executor: a write-up with one python block; adjudicator: singleton
    groups; lenses: the mock reviewer outputs. ``fail_executor_calls``: the
    executor call numbers (1-based) that fail with a non-retryable error."""

    def __init__(self, kit: str, fail_executor_calls=(), **kw):
        super().__init__(min_delay=0.0, max_delay=0.01, **kw)
        self.kit = kit
        self.fail = set(fail_executor_calls)
        self.executor_calls = 0
        self.requests = []

    @property
    def name(self):
        return "codex_cli"

    async def run(self, request):
        self.requests.append(request)
        role = request.prompt_bundle.variables.get("role", "")
        if role == "executor":
            self.executor_calls += 1
            if self.executor_calls in self.fail:
                return AdapterRunResult(success=False, error="[auth] simulated failure",
                                        provider="codex_cli")
            return AdapterRunResult(success=True, output=EXEC_WRITEUP.format(kit=self.kit),
                                    provider="codex_cli", model=request.model,
                                    reasoning_effort=request.reasoning_effort,
                                    input_tokens=100, output_tokens=50, token_usage=150)
        if role == "consensus_adjudicator":
            m = re.search(r"N = (\d+) critiques", request.prompt_bundle.user_prompt)
            n = int(m.group(1)) if m else 0
            return AdapterRunResult(success=True, provider="codex_cli",
                                    output="```json\n" + json.dumps(
                                        {"groups": [[i] for i in range(n)]}) + "\n```")
        return await super().run(request)


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from backend.config import config
    kit = tmp_path / "kit"
    kit.mkdir()
    (kit / "genes.tsv").write_text("gene_index\tsymbol\n0\tA\n1\tB\n2\tC\n")
    data = tmp_path / "data"
    monkeypatch.setattr(config, "db_path", str(data / "test.db"))
    monkeypatch.setenv("MIW_DB_PATH", str(data / "test.db"))
    paths = rcs.Paths(data)
    data.mkdir()
    rcs.setup_backend(paths.db)
    return {"kit": kit, "paths": paths, "data": data, "tmp": tmp_path}


def _run(spec, env, adapter, **kw):
    kw.setdefault("repair_waits", (0,))
    return asyncio.run(rcs.run_one(spec, env["paths"], adapter_factory=lambda p: adapter,
                                   kit_dir=env["kit"], analysis_python=sys.executable, **kw))


def _index(run_dir):
    return [json.loads(line) for line in
            (run_dir / "x4_prompts" / "index.jsonl").read_text().splitlines()]


@needs_sandbox
def test_end_to_end_executed_run_and_trace(env):
    spec = rcs.RunSpec("t_exec", "gpt-5.6-sol", "medium", True, "test")
    adapter = FakeAdapter(str(env["kit"]))
    st = _run(spec, env, adapter)
    assert st["status"] == "completed" and st["stop_reason"] == "max_iterations"
    roles = [it["role"] for it in st["iterations"]]
    assert roles == ["executor", "consensus_merger"] * 5
    run_dir = env["paths"].run_dir(spec)
    for it in st["iterations"]:
        d = run_dir / f"iter_{it['n']:04d}"
        if it["role"] == "executor":
            ce = json.loads((d / "CODE_EXECUTION.json").read_text())
            assert ce["passed"] == 1 and ce["backend"] == "sandbox_exec", ce
            assert "n_edges 424" in ce["blocks"][0]["stdout"]
            unit = Path(ce["blocks"][0]["persisted_to"])
            assert unit.parent == Path(os.path.realpath(env["paths"].exec_outputs(spec)))
            assert json.loads((unit / "work" / "results.json").read_text())["n_edges"] == 424
            assert "n_edges 424" in (unit / "stdout.txt").read_text()
            assert (d / "RUN_LOG.md").is_file()
        else:
            assert (d / "CONSENSUS.json").is_file()
    idx = _index(run_dir)
    lens = [e for e in idx if e["role"] in ("reviewer", "adversarial_reviewer",
                                            "bio_plausibility_checker")]
    assert len(lens) == 15 and all(e["user_prompt_has_execution_report"] for e in lens)
    assert all(e["allow_tools"] is False for e in idx), {e["role"]: e["allow_tools"] for e in idx}
    assert all(e["model"] == "gpt-5.6-sol" and e["reasoning_effort"] == "medium" for e in idx)
    assert any(e["role"] == "consensus_adjudicator" for e in idx)
    # Executor iterations after the first get their previous execution report.
    ex = [e for e in idx if e["role"] == "executor"]
    second = (run_dir / "x4_prompts" / ex[1]["file"]).read_text()
    assert "Verified execution of the code in your previous submission" in second
    assert "VERIFIED CODE EXECUTION IS ENABLED" in second
    meta = json.loads((run_dir / "run_meta.json").read_text())
    ts = meta["config"]["effective_tool_settings"]
    assert ts["executor_allow_tools"] is False and ts["code_execution_enabled"] is True
    assert meta["stop_reason"] == "max_iterations"
    zpath = Path(st["repropack"])
    with zipfile.ZipFile(zpath) as zf:
        names = set(zf.namelist())
    assert "metadata/run_meta.json" in names
    assert "run_dir/iter_0001/CODE_EXECUTION.json" in names
    assert "run_dir/x4_task.md" in names and "case_study/run_case_study.py" in names
    assert any(n.startswith("exec_outputs/") and n.endswith("/work/results.json") for n in names)
    # Trace: 0.673 and 424 traceable (current), 0.811 not.
    out = env["tmp"] / "trace_out"
    assert tn.main(["--data-dir", str(env["data"]), "--out-dir", str(out)]) == 0
    summ = json.loads((out / "trace_summary.json").read_text())["runs"]["t_exec"]
    it1 = summ["iterations"][0]
    assert it1["execution"]["complete"] is True and it1["output_possibly_truncated"] is False
    assert it1["counts"]["result"]["traceable_current"] >= 2
    assert [u["text"] for u in it1["untraceable_results"]] == ["0.811"]
    later = summ["iterations"][1]
    assert later["counts"]["result"]["untraceable"] == 1
    assert summ["executor_iterations"] == 5


@needs_sandbox
def test_end_to_end_plan_only_control(env):
    spec = rcs.RunSpec("t_plan", "gpt-5.6-sol", "medium", False, "control")
    adapter = FakeAdapter(str(env["kit"]))
    st = _run(spec, env, adapter)
    assert st["status"] == "completed" and len(st["iterations"]) == 10
    run_dir = env["paths"].run_dir(spec)
    assert not list(run_dir.glob("iter_*/CODE_EXECUTION.json"))
    idx = _index(run_dir)
    assert all(e["allow_tools"] is False for e in idx)
    ex_prompt = (run_dir / "x4_prompts" / [e for e in idx if e["role"] == "executor"][0]["file"]
                 ).read_text()
    assert "VERIFIED CODE EXECUTION" not in ex_prompt
    assert "COMPUTING ENVIRONMENT AND REPORTING RULE" not in ex_prompt
    out = env["tmp"] / "trace_out"
    tn.main(["--data-dir", str(env["data"]), "--out-dir", str(out)])
    summ = json.loads((out / "trace_summary.json").read_text())["runs"]["t_plan"]
    tot = summ["totals"]["result"]
    assert tot["total"] > 0 and tot["traceable"] == 0 and tot["untraceable"] == tot["total"]
    assert "untraceable by definition" in summ["iterations"][0]["note"]


@needs_sandbox
def test_failed_run_is_repaired_and_keeps_fixed_horizon(env):
    spec = rcs.RunSpec("t_fail", "gpt-5.6-sol", "medium", True, "repair")
    adapter = FakeAdapter(str(env["kit"]), fail_executor_calls={2})
    st = _run(spec, env, adapter, build_pack=False)
    assert st["status"] == "completed", st
    assert [it["role"] for it in st["iterations"]] == ["executor", "consensus_merger"] * 5
    assert all(it["status"] == "completed" for it in st["iterations"])
    run_dir = env["paths"].run_dir(spec)
    repairs = [json.loads(x) for x in (run_dir / "x4_repairs.jsonl").read_text().splitlines()]
    assert repairs[-1]["previous_stop_reason"] == "failed:adapter_failed"
    assert repairs[-1]["resumed_from_iteration"] == 2 and repairs[-1]["dropped_iteration_rows"] == [3]


@needs_sandbox
def test_interrupted_run_resumes_in_new_invocation(env):
    spec = rcs.RunSpec("t_int", "gpt-5.6-sol", "medium", True, "resume")
    st = _run(spec, env, FakeAdapter(str(env["kit"]), fail_executor_calls={3}),
              max_repairs=0, build_pack=False)
    assert st["status"] == "failed" and st["current_iteration"] == 5
    # A new invocation (fresh adapter) repairs and finishes the same run.
    st = _run(spec, env, FakeAdapter(str(env["kit"])), build_pack=False)
    assert st["status"] == "completed" and len(st["iterations"]) == 10
    # A completed run is a cached unit: nothing is called again.
    adapter = FakeAdapter(str(env["kit"]))
    st = _run(spec, env, adapter, build_pack=False)
    assert st["status"] == "completed" and adapter.requests == []


def test_changed_task_or_config_refuses_resume(env, monkeypatch):
    spec = rcs.RunSpec("t_frozen", "gpt-5.6-sol", "medium", True, "frozen")
    rd = env["paths"].run_dir(spec)
    rd.mkdir(parents=True)
    (rd / "x4_run_spec.json").write_text(json.dumps({"task_sha256": "x", "config": {}}))
    with pytest.raises(RuntimeError, match="changed"):
        _run(spec, env, FakeAdapter(str(env["kit"])))


@needs_sandbox
def test_smoke_sandbox_with_real_kit_config(tmp_path):
    if not (rcs.KIT_DIR / "attention_heads_f32.npy").is_file() or \
            not os.path.isfile(rcs.ANALYSIS_PYTHON):
        pytest.skip("real kit / analysis interpreter not available")
    out = asyncio.run(rcs.smoke_sandbox(rcs.Paths(tmp_path)))
    assert out["summary"]["passed"] == 1, out["feedback"]
    res = json.loads(out["report"]["blocks"][0]["stdout"].strip().splitlines()[-1])
    assert res["n_genes"] == 1322 and res["heads_shape"] == [12, 12, 1322, 1322]
    assert all(v not in (False,) for v in res["denied"].values()), res["denied"]


# ── trace_numbers units ──────────────────────────────────────────────────


def _texts(s):
    return [n.text for n in tn.extract_numbers(s)]


def test_extract_numbers_rules():
    assert _texts("AUROC 0.673, p = 3.2e-05 and 1.5 × 10^-3") == ["0.673", "3.2e-05", "1.5 × 10^-3"]
    assert tn.extract_numbers("1.5 × 10^-3")[0].value == pytest.approx(1.5e-3)
    assert tn.extract_numbers("3.2×10⁻⁵")[0].value == pytest.approx(3.2e-5)
    assert _texts("range 0.16-0.32") == ["0.16", "0.32"]
    vals = [n.value for n in tn.extract_numbers("from −0.29 to -0.05")]
    assert vals == [-0.29, -0.05]
    assert _texts("L3 layer_3 V2-104M 10x auroc_l3 v1.26.4 python 3.11.5") == []
    n = tn.extract_numbers("60,606 genes and 64%")
    assert n[0].value == 60606 and n[1].percent and n[1].value == 64
    assert tn.extract_numbers("0.673")[0].decimals == 3


def test_split_writeup_removes_code_and_keeps_lines():
    out = "a 0.1\n```python\nx = 0.5\nprint(x)\n```\nb 0.2\n"
    w, code = tn.split_writeup(out)
    assert "0.5" not in w and w.count("\n") == out.count("\n")
    assert code == ["x = 0.5\nprint(x)\n"]
    assert _texts(w) == ["0.1", "0.2"]


def _cats(text, task=""):
    nums = tn.extract_numbers(text)
    return dict(zip([n.text for n in nums], tn.classify(text, nums, tn.task_literals(task))))


def test_classify_categories():
    task = "Gene set G (1322 genes); seed 20261001.\n1. Define the score.\n"
    c = _cats("1. Results\n### 2.1 Methods\nLayers 3-6 and head 7; Fix 4; (l=8, h=11).\n"
              "Using 10,000 permutations, 95% CI, co-presence >= 50 cells, top 100 pairs.\n"
              "AUROC 0.673 (q < 0.05), p = 0.0099, rho −0.29; 5 of 12 layers; 1322 genes; "
              "as shown (2023).\n", task)
    assert c["1"] == "list_marker" and c["2.1"] == "list_marker"
    assert c["3"] == "index" and c["6"] == "index" and c["7"] == "index" and c["4"] == "index"
    assert c["11"] == "index" and c["8"] == "index"
    assert c["10,000"] == "parameter" and c["95%"] == "parameter"
    assert c["50"] == "parameter" and c["100"] == "parameter"
    assert c["0.05"] == "threshold"
    assert c["0.673"] == "result" and c["0.0099"] == "result" and c["−0.29"] == "result"
    assert c["12"] == "small_integer" and c["5"] == "small_integer"
    assert c["1322"] == "task_given"
    assert c["2023"] == "year"


def test_matching_rules():
    idx = tn.ValueIndex([0.67312, 64.1, 0.29, 3.24e-05, 424.0, 1234.0, 0.6412])
    m = lambda s: tn.match_number(tn.extract_numbers(s)[0], idx)  # noqa: E731
    assert m("0.673") == "exact"
    assert m("0.674") is None                       # 0.67312 -> 0.673, not 0.674
    assert m("0.67") == "exact"
    assert m("-0.29") == "sign_mismatch"
    assert m("3.2e-05") == "exact" and m("3.3e-05") is None
    assert m("424") == "exact" and m("1,234") == "exact"
    assert m("64%").startswith("exact")             # 64.1 (as percent) or 0.6412
    assert tn.match_number(tn.extract_numbers("0.641")[0],
                           tn.ValueIndex([64.1])) == "exact+fraction_as_percent"
    assert tn.match_number(tn.extract_numbers("64%")[0],
                           tn.ValueIndex([0.6412])) == "exact+percent_as_fraction"
    assert tn.match_number(tn.extract_numbers("0.5")[0], tn.ValueIndex([])) is None


def _fake_run(tmp_path, iters, execution=True):
    rd = tmp_path / "data" / "workspaces" / "k" / "runs" / "x4_k"
    for n, (writeup, stdout, results) in iters.items():
        d = rd / f"iter_{n:04d}"
        d.mkdir(parents=True)
        (d / "executor_output.md").write_text(writeup)
        if execution:
            files = [{"path": "results.json", "size_bytes": 10,
                      "content": json.dumps(results), "truncated": False, "skipped": None}]
            (d / "CODE_EXECUTION.json").write_text(json.dumps({
                "executed": 1, "passed": 1, "failed": 0,
                "blocks": [{"stdout": stdout, "stdout_truncated": False, "files": files}]}))
    (rd / "x4_task.md").write_text("KIT has 1322 genes.")
    (rd / "x4_run_spec.json").write_text(json.dumps(
        {"spec": {"model": "m", "effort": "medium"},
         "config": {"code_execution_enabled": execution}}))
    return tmp_path / "data"


def test_trace_run_current_earlier_and_literal_in_code(tmp_path):
    data = _fake_run(tmp_path, {
        1: ("Result 0.6731 and 0.512.\n```python\nprint(0.512)\n```\n", "auroc 0.67312\n0.512\n",
            {"x": 1}),
        3: ("Still 0.673; new 0.7001; made up 0.9; 1322 genes.\n```python\nprint(1)\n```\n",
            "other 0.70009\n", {"y": 2}),
    })
    out = tmp_path / "o"
    tn.main(["--data-dir", str(data), "--out-dir", str(out)])
    s = json.loads((out / "trace_summary.json").read_text())["runs"]["k"]
    i1, i3 = s["iterations"]
    assert i1["counts"]["result"]["traceable_current"] == 2
    assert i1["counts"]["result"]["traceable_but_literal_in_code"] == 1   # 0.512 hard-coded
    assert i3["counts"]["result"]["traceable_earlier"] == 1               # 0.673 from iter 1
    assert i3["counts"]["result"]["traceable_current"] == 1               # 0.7001
    assert [u["text"] for u in i3["untraceable_results"]] == ["0.9"]
    assert i3["excluded"]["task_given"] == 1
    assert s["totals"]["result"]["total"] == 5
    rows = (out / "trace_per_iteration.csv").read_text().splitlines()
    assert len(rows) == 3


def test_trace_plan_only_all_untraceable(tmp_path):
    data = _fake_run(tmp_path, {1: ("AUROC 0.673 and 0.7.\n", "", {})}, execution=False)
    out = tmp_path / "o"
    tn.main(["--data-dir", str(data), "--out-dir", str(out)])
    s = json.loads((out / "trace_summary.json").read_text())["runs"]["k"]
    assert s["totals"]["result"] == {"total": 2, "traceable": 0, "traceable_current": 0,
                                     "traceable_earlier": 0, "untraceable": 2,
                                     "traceable_but_literal_in_code": 0}


def test_trace_uses_complete_persisted_outputs(tmp_path):
    data = _fake_run(tmp_path, {1: ("Values 0.123 and 0.987.\n", "0.123\n...[truncated]", {})})
    d = data / "workspaces" / "k" / "runs" / "x4_k" / "iter_0001"
    unit = tmp_path / "unit"
    (unit / "work").mkdir(parents=True)
    (unit / "stdout.txt").write_text("0.123\nlater line 0.98712\n")
    (unit / "work" / "results.json").write_text("{}")
    (unit / "work" / "arr.npy").write_bytes(b"\x93NUMPY\x00\x00 0.987")
    ce = json.loads((d / "CODE_EXECUTION.json").read_text())
    ce["blocks"][0]["persisted_to"] = str(unit)
    ce["blocks"][0]["stdout_truncated"] = True
    (d / "CODE_EXECUTION.json").write_text(json.dumps(ce))
    out = tmp_path / "o"
    tn.main(["--data-dir", str(data), "--out-dir", str(out)])
    it = json.loads((out / "trace_summary.json").read_text())["runs"]["k"]["iterations"][0]
    assert it["execution"]["complete"] is True and it["output_possibly_truncated"] is False
    assert it["counts"]["result"]["traceable_current"] == 2


def test_trace_generated_reports_channel(tmp_path):
    """Code that writes MECH.md: formatted numbers are computed, a literal typed
    into a string is hard-coded unless an output also contains it."""
    code = ("```python\nauc = 0.6 + 0.0734\n"
            "open('MECH.md','w').write(f'AUROC {auc:.3f}; recalled 0.912; alpha 0.05; '"
            "'n 500 (also 0.25 lit)')\nprint(0.25)\n```\n")
    data = _fake_run(tmp_path, {1: (code, "0.25\n", {"auc": 0.6734})})
    d = data / "workspaces" / "k" / "runs" / "x4_k" / "iter_0001"
    unit = tmp_path / "unit"
    (unit / "work").mkdir(parents=True)
    (unit / "stdout.txt").write_text("0.25\n")
    (unit / "work" / "results.json").write_text(json.dumps({"auc": 0.6734}))
    (unit / "work" / "MECH.md").write_text(
        "AUROC 0.673; recalled 0.912; alpha 0.05; n 500 (also 0.25 lit)")
    ce = json.loads((d / "CODE_EXECUTION.json").read_text())
    ce["blocks"][0]["persisted_to"] = str(unit)
    (d / "CODE_EXECUTION.json").write_text(json.dumps(ce))
    out = tmp_path / "o"
    tn.main(["--data-dir", str(data), "--out-dir", str(out)])
    s = json.loads((out / "trace_summary.json").read_text())["runs"]["k"]
    it = s["iterations"][0]
    assert it["counts"]["result"]["total"] == 0            # code-only response
    g = it["generated_reports"]
    assert g["files"] == ["MECH.md"]
    r = g["counts"]["result"]
    # 0.673 formatted from a variable (computed); 0.25 literal but printed too;
    # 0.912 and 500 typed into the string literal (hard-coded); 0.05 = threshold.
    assert (r["computed"], r["literal_matches_output"], r["hard_coded"]) == (1, 1, 2), r
    assert [h["text"] for h in g["hard_coded_results"]] == ["0.912", "500"]
    assert g["counts"]["threshold"]["total"] == 1
    assert it["combined_result"] == {"total": 4, "traceable": 2, "untraceable": 2,
                                     "traceable_fraction": 0.5}
    assert s["combined_result"]["untraceable"] == 2
