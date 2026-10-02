"""Orchestration experiments (backend/analysis/experiments.py and the scripts).

Covers:

- mock implementation verification of the merge semantics, including a
  negative control showing the checks fail when escalation is broken;
- runner-path overhead with the real SQLite database and artifact writing;
- the integrity checker, against real corruptions;
- the real-backend concurrency and probe scripts, with an offline fake adapter.

Fast by default. Opt-in:

- ``MIW_RUN_SLOW=1``: full-size overhead sweep (several minutes);
- ``MIW_RUN_REAL=1``: one real Codex panel through the probe script (costs
  tokens).
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

from backend.adapters import registry
from backend.adapters.mock import MockAdapter
from backend.analysis import experiments as ex
from backend.config import config
from backend.models import ProviderName

REPO = Path(__file__).resolve().parents[2]
ARTIFACT = REPO / "experiments" / "benchmark" / "artifacts" / "A01-clean.md"

slow = pytest.mark.skipif(os.environ.get("MIW_RUN_SLOW") != "1",
                          reason="slow; set MIW_RUN_SLOW=1")
real = pytest.mark.skipif(os.environ.get("MIW_RUN_REAL") != "1",
                          reason="real model calls; set MIW_RUN_REAL=1")


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    scripts_dir = str(path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(f"miw_test_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── mock implementation verification ─────────────────────────────────


def test_consensus_ablation_monotonic_coverage():
    res = ex.consensus_ablation(panel_sizes=(1, 2, 3), tier=1)
    distinct = [r["distinct_critiques"] for r in res["rows"]]
    assert distinct == sorted(distinct) and distinct[0] < distinct[-1]
    assert res["rows"][-1]["multi_reviewer"] >= 1


def test_role_overlap_is_reported_as_exact_string_overlap():
    res = ex.role_overlap(tier=1)
    assert "exact" in res["overlap_measure"]
    for p in res["pairwise"]:
        assert "jaccard" not in p
        assert set(p) == {"roles", "exact_string_set_overlap", "shared_exact"}
    assert [p["exact_string_set_overlap"] for p in res["pairwise"]] == [0.2, 0.0, 0.0]
    assert res["union_size"] < res["sum_individual"]


def test_merge_semantics_checks_pass_on_mock():
    res = ex.merge_semantics_checks(tier=1)
    assert res["all_passed"], res["problems"]
    assert res["n_shared_exact"] == 1
    assert res["n_merged"] == res["n_distinct_exact"] == 7
    assert res["problems"] == []


def test_merge_semantics_checks_detect_broken_escalation(monkeypatch):
    """Negative control: with escalation disabled the check must fail."""
    from backend.orchestrator import consensus
    monkeypatch.setattr(consensus, "_escalate_severity", lambda s: s)
    res = ex.merge_semantics_checks(tier=1)
    assert not res["checks"]["agreement_escalates_one_level"]
    assert not res["all_passed"]
    assert any("severity" in p for p in res["problems"])


def test_convergence_profile_is_one_deterministic_trajectory():
    res = ex.convergence_profile(max_iterations=20, seeds=(0, 1))
    assert res["n_runs"] == 1 and res["deterministic"] is True
    assert "single deterministic run" in res["note"]
    assert res["determinism_check"]["identical"] is True
    assert res["determinism_check"]["seeds"] == [0, 1]
    assert 1 <= res["iterations"] <= 20
    assert len(res["trajectory"]) == res["iterations"]
    assert res["stop_reason"]
    assert MockAdapter.fixed_iteration is None


def test_mock_verification_record():
    res = ex.mock_verification(max_iterations=20, seeds=(0,))
    assert res["schema"] == ex.MOCK_VERIFICATION_SCHEMA
    assert "not evidence" in res["purpose"]
    assert res["merge_semantics"]["all_passed"]
    assert res["convergence_profile"]["n_runs"] == 1
    json.dumps(res)  # serialisable


# ── helpers ──────────────────────────────────────────────────────────


def test_interval_union_and_percentile():
    assert ex.interval_union([]) == 0.0
    assert ex.interval_union([(0, 1), (0.5, 2), (3, 4)]) == pytest.approx(3.0)
    assert ex.interval_union([(0, 1), (0, 1), (0, 1)]) == pytest.approx(1.0)
    assert ex.percentile([], 95) == 0.0
    assert ex.percentile([3, 1, 2], 50) == 2
    assert ex.percentile(list(range(1, 21)), 95) == 19
    assert ex.percentile([5.0], 95) == 5.0


# ── control-plane overhead through the runner ────────────────────────


def _state():
    return (config.db_path, registry._ADAPTERS[ProviderName.MOCK], MockAdapter.fixed_iteration)


def test_control_plane_overhead_small(tmp_path):
    before = _state()
    res = ex.control_plane_overhead(concurrency_levels=(1, 2), iterations_per_run=2,
                                    adapter_delays=(0.0, 0.05), repeats=1,
                                    parent_dir=str(tmp_path))
    assert _state() == before                      # global state restored
    assert res["schema"] == ex.OVERHEAD_SCHEMA
    assert res["integrity_ok"] is True
    assert not Path(res["environment"]["workdir"]).exists()   # temp dir removed
    assert res["environment"]["sqlite_version"]
    for sw in res["sweeps"]:
        assert sw["warmup"]["integrity_ok"]
        assert [r["concurrency"] for r in sw["rows"]] == [1, 2]
        for row in sw["rows"]:
            assert row["runs"] == row["concurrency"]
            # reviewer_consensus, 2 iterations = 1 executor call + 1 panel of 3 lenses
            assert row["adapter_calls_per_run"] == [4]
            assert row["integrity_ok"]
            integ = row["integrity"][0]
            assert integ["integrity_check"] == "ok"
            assert integ["foreign_key_violations"] == 0
            assert integ["orphan_iteration_rows"] == 0 and integ["orphan_run_dirs"] == 0
            assert integ["n_problems"] == 0
            assert row["overhead_ms_per_iteration"]["p50"] > 0
            assert row["throughput_iters_per_s"]["mean"] > 0
    slow_sweep = res["sweeps"][1]
    assert slow_sweep["adapter_delay_s"] == 0.05
    # one adapter round of 50 ms per iteration (panel lenses run in parallel)
    adapter_ms = slow_sweep["rows"][0]["adapter_ms_per_iteration"]["p50"]
    # The lower bound checks that the injected delay is counted; the upper
    # bound is loose because the p50 is a single wall-clock value and the
    # machine may be heavily loaded during the suite.
    assert 45.0 <= adapter_ms < 500.0
    # the integrity check covers every run so far: warm-up + 1 + 2
    assert slow_sweep["rows"][-1]["integrity"][0]["runs_checked"] == 4


def test_check_integrity_detects_corruption(tmp_path):
    res = ex.control_plane_overhead(concurrency_levels=(2,), iterations_per_run=2,
                                    adapter_delays=(0.0,), repeats=1,
                                    parent_dir=str(tmp_path), keep_workdir=True)
    root = Path(res["environment"]["workdir"]) / "delay_0s"
    db_path, ws = root / "miw.db", root / "workspace"
    assert asyncio.run(ex.check_integrity(str(db_path), str(ws)))["ok"]

    con = sqlite3.connect(db_path)
    run_id = con.execute("SELECT run_id FROM runs LIMIT 1").fetchone()[0]
    con.execute("PRAGMA foreign_keys=OFF")
    con.execute("INSERT INTO iterations (iteration_id, run_id, iteration_number) "
                "VALUES ('orphan-it', 'no-such-run', 1)")
    con.commit()
    con.close()
    # remove one iteration directory and add a stray run directory
    # (ignore_errors: exFAT AppleDouble "._*" files vanish with their main file)
    import shutil
    gone = ws / "runs" / run_id / "iter_0002"
    for _ in range(3):
        shutil.rmtree(gone, ignore_errors=True)
    assert not gone.exists()
    (ws / "runs" / "stray-run").mkdir()

    bad = asyncio.run(ex.check_integrity(str(db_path), str(ws)))
    assert not bad["ok"]
    assert bad["orphan_iteration_rows"] == 1
    assert bad["orphan_run_dirs"] == 1
    assert any(run_id in p and "on disk" in p for p in bad["problems"])
    assert bad["foreign_key_violations"] >= 1


@slow
def test_control_plane_overhead_full_size(tmp_path):
    res = ex.control_plane_overhead(parent_dir=str(tmp_path))
    assert res["integrity_ok"]
    assert [r["concurrency"] for r in res["sweeps"][0]["rows"]] == [1, 2, 5, 10, 20]


# ── real-backend scripts with an offline fake adapter ────────────────


def test_real_concurrency_records_retries_and_resumes(tmp_path):
    rc = _load_script("real_concurrency")
    out, raw = tmp_path / "rc.json", tmp_path / "raw"
    fake = rc.FakeReviewAdapter(delay=0.01, transient_failures=1)
    kw = dict(out_path=out, raw_dir=raw, artifact=ARTIFACT, model="gpt-5.6-luna",
              effort="low", pause=0.0, retry_base_delay=0.01, ws_root=tmp_path / "ws")
    res = asyncio.run(rc.run_experiment([1, 2], adapter=fake, **kw))
    s1, s2 = res["summary_by_level"]
    assert s1["k"] == 1 and s1["lens_calls"] == 3
    assert s1["attempts"] == 4 and s1["retries"] == 1          # one 429, retried
    assert s1["rate_limit_errors"] == 1 and s1["failed_attempts"] == 1
    assert s1["failed_lens_calls_final"] == 0 and s1["panels_complete"] == 1
    assert s2["attempts"] == 6 and s2["retries"] == 0 and s2["panels_complete"] == 2
    assert s2["panel_wall_p50_vs_k1"] > 0
    assert s1["tokens"]["input_tokens"] == 3000                  # failed attempt: 0 tokens
    lines = [json.loads(x) for x in (raw / "real_concurrency_K1.jsonl").read_text().splitlines()]
    assert len(lines) == 4 and any(x["raw_output"] for x in lines)
    assert any(x["error_category"] == "rate_limit" and x["transient"] for x in lines)
    saved = json.loads(out.read_text())
    assert saved["config"]["model"] == "gpt-5.6-luna" and saved["config"]["allow_tools"] is False
    assert "_raw_output" not in json.dumps(saved["levels"])     # raw text only in JSONL
    # resume: same configuration -> both levels skipped, no new calls
    fake2 = rc.FakeReviewAdapter(delay=0.01)
    res2 = asyncio.run(rc.run_experiment([1, 2], adapter=fake2, **kw))
    assert [lv["summary"]["attempts"] for lv in res2["levels"]] == [4, 6]


def test_real_concurrency_reports_failed_lenses(tmp_path):
    rc = _load_script("real_concurrency")
    fake = rc.FakeReviewAdapter(delay=0.0, fail_roles=["bio_plausibility_checker"])
    res = asyncio.run(rc.run_experiment(
        [2], out_path=tmp_path / "rc.json", raw_dir=tmp_path / "raw", artifact=ARTIFACT,
        model="gpt-5.6-luna", effort="low", adapter=fake, pause=0.0,
        retry_base_delay=0.01, ws_root=tmp_path / "ws"))
    s = res["summary_by_level"][0]
    assert s["panels_partial"] == 2 and s["panels_complete"] == 0
    assert s["failed_lens_calls_final"] == 2
    assert s["error_categories"] == {"unknown": 2}
    assert s["retries"] == 0                                     # permanent error: no retry


def test_real_consensus_probe_with_fake_adapter(tmp_path):
    rc = _load_script("real_concurrency")
    probe = _load_script("real_consensus_probe")
    res = asyncio.run(probe.probe(ARTIFACT, "gpt-5.6-luna", "low", 60.0,
                                  adapter=rc.FakeReviewAdapter(delay=0.0)))
    assert res["schema"] == probe.SCHEMA
    assert res["artifact"].endswith("A01-clean.md")
    assert res["model_requested"] == "gpt-5.6-luna" and res["cli_version"] == "fake-0"
    assert [x["status"] for x in res["lenses"]] == ["ok", "ok", "ok"]
    assert all(x["raw_output"] and x["output_tokens"] == 200 for x in res["lenses"])
    assert res["tokens_total"]["input_tokens"] == 3000
    assert res["report"]["total_merged"] == len(res["merged_critiques"]) > 0


@real
def test_real_consensus_probe_codex(tmp_path):
    probe = _load_script("real_consensus_probe")
    rc_ = probe.main(["--out", str(tmp_path / "probe.json")])
    data = json.loads((tmp_path / "probe.json").read_text())
    assert data["cli_version"]
    assert rc_ == 0, [x["error"] for x in data["lenses"]]


# ── figures (data preparation; plotting needs matplotlib) ────────────


def test_make_figures_data_prep(tmp_path):
    mf = _load_script("make_figures")
    ov = ex.control_plane_overhead(concurrency_levels=(1, 2), iterations_per_run=2,
                                   adapter_delays=(0.0,), repeats=1, parent_dir=str(tmp_path))
    overhead = {"results": [dict(ov, storage={"label": "tmp", "fs_type": "x"})]}
    series = mf.overhead_series(overhead)
    assert series and series[0]["concurrency"] == [1, 2]
    assert len(series[0]["overhead_p50"]) == 2
