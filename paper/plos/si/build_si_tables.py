"""Build S1 Table (run register) and S2 Table (full results) for the PLOS ONE manuscript.

Run from the repository root with the analysis environment:
    /Users/ihorkendiukhov/anaconda3/bin/python paper/plos/si/build_si_tables.py
Every number is computed from the raw records under experiments_data/ and the result
files under experiments/; macOS ._* files are ignored.
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments_data"
OUT = ROOT / "paper" / "plos" / "si"


def files(pattern: str) -> list[str]:
    return sorted(f for f in glob.glob(str(pattern), recursive=True) if "/._" not in f and "/archive/" not in f)


def tok(d: dict | None) -> tuple[int, int, int]:
    d = d or {}
    return (int(d.get("input") or d.get("input_tokens") or 0),
            int(d.get("cached_input") or d.get("cached_input_tokens") or 0),
            int(d.get("output") or d.get("output_tokens") or 0))


def n_attempts(rec: dict) -> int:
    a = rec.get("attempts")
    if isinstance(a, list):
        return len(a)
    if isinstance(a, int):
        return a
    return 1


def row(experiment, model, effort, calls, attempts, tin, tcached, tout, wall_s, location, cli="codex-cli 0.145.0",
        tools="off", notes=""):
    return {
        "experiment": experiment, "provider": "OpenAI (Codex CLI)", "cli_version": cli, "model": model,
        "reasoning_effort": effort, "decoding": "provider default (temperature not exposed by the CLI)",
        "agent_tools": tools, "successful_calls": calls, "attempts_incl_retries": attempts,
        "input_tokens": tin, "cached_input_tokens": tcached, "output_tokens": tout,
        "wall_time_hours_sum_of_calls": round(wall_s / 3600, 2) if wall_s is not None else None,
        "data_location": location, "notes": notes,
    }


def cli_versions(recs: list[dict]) -> str:
    vs = set()
    for r in recs:
        for k in ("cli_version",):
            v = (r.get("request") or {}).get(k) or r.get(k)
            if v:
                vs.add(v)
        for a in r.get("attempts") or [] if isinstance(r.get("attempts"), list) else []:
            if isinstance(a, dict) and a.get("cli_version"):
                vs.add(a["cli_version"])
    return ", ".join(sorted(vs)) or "codex-cli 0.145.0"


def s1_table() -> pd.DataFrame:
    rows = []
    # Planted-flaw benchmark reviewers.
    for model_dir in sorted(Path(DATA / "benchmark_runs_v1" / "reviews").iterdir()):
        if model_dir.name.startswith("._") or not model_dir.is_dir():
            continue
        recs = [json.load(open(f)) for f in files(model_dir / "*" / "r*" / "*.json")]
        ok = [r for r in recs if r.get("success")]
        tin = sum(tok(r.get("tokens"))[0] for r in recs)
        tc = sum(tok(r.get("tokens"))[1] for r in recs)
        to = sum(tok(r.get("tokens"))[2] for r in recs)
        wall = sum(float(r.get("wall_seconds") or 0) for r in recs)
        effort = (recs[0].get("request") or {}).get("reasoning_effort") or (recs[0].get("request") or {}).get("effort")
        rows.append(row("Planted-flaw benchmark: reviewer calls", model_dir.name, effort, len(ok),
                        sum(n_attempts(r) for r in recs), tin, tc, to, wall,
                        f"experiments_data/benchmark_runs_v1/reviews/{model_dir.name}/", cli=cli_versions(recs)))
    # Benchmark judges.
    for jdir in sorted(Path(DATA / "benchmark_runs_v1" / "judgments").iterdir()):
        if jdir.name.startswith("._") or not jdir.is_dir():
            continue
        recs = [json.load(open(f)) for f in files(jdir / "*" / "*" / "r*.json")]
        tin = sum(tok(r.get("tokens"))[0] for r in recs)
        tc = sum(tok(r.get("tokens"))[1] for r in recs)
        to = sum(tok(r.get("tokens"))[2] for r in recs)
        wall = sum(float(r.get("wall_seconds") or 0) for r in recs)
        rows.append(row("Planted-flaw benchmark: judge", jdir.name, "high", sum(1 for r in recs if r.get("valid")),
                        sum(n_attempts(r) for r in recs), tin, tc, to, wall,
                        f"experiments_data/benchmark_runs_v1/judgments/{jdir.name}/"))
    # Merge adjudicator.
    me = json.load(open(ROOT / "experiments/benchmark/results/merge_eval.json"))
    llm = me["methods"]["llm"]
    t = llm.get("tokens_total") or {}
    adj = [json.load(open(f)) for f in files(DATA / "benchmark_runs_v1" / "merge_eval" / "llm" / "**" / "r*.json")]
    wall = sum(float(r.get("wall_seconds") or 0) for r in adj) if adj else None
    rows.append(row("Merge evaluation: LLM adjudicator", llm.get("adjudicator_model"), llm.get("adjudicator_effort"),
                    llm.get("n_instances"), len(adj) or llm.get("n_instances"), t.get("input", 0),
                    t.get("cached_input", 0), t.get("output", 0), wall,
                    "experiments_data/benchmark_runs_v1/merge_eval/llm/"))
    # Stopping-calibration loops and state judge.
    for cfg, model, effort in (("sol-medium", "gpt-5.6-sol", "medium"), ("luna-low", "gpt-5.6-luna", "low")):
        trajs = [json.load(open(f)) for f in files(DATA / "stopping_runs" / cfg / "*" / "trajectory.json")]
        tin = sum(tr["totals"]["input_tokens"] for tr in trajs)
        tc = sum(tr["totals"]["cached_input_tokens"] for tr in trajs)
        to = sum(tr["totals"]["output_tokens"] for tr in trajs)
        wall = sum(tr["totals"]["duration_seconds"] for tr in trajs)
        calls = sum(tr["totals"]["n_calls"] for tr in trajs)
        failed = sum(len(tr.get("failed_attempts") or []) for tr in trajs)
        rows.append(row("Stopping calibration: executor-panel loops", model, effort, calls, calls + failed, tin, tc,
                        to, wall, f"experiments_data/stopping_runs/{cfg}/",
                        notes=f"{len(trajs)} trajectories x 11 steps (executor + 3 lenses per panel; jaccard merge); failed step attempts {failed}"))
        recs = [json.load(open(f)) for f in files(DATA / "stopping_runs" / "judgments" / "*" / cfg / "*" / "E*.json")]
        atts = [a for r in recs for a in (r.get("attempts") or [])]
        rows.append(row("Stopping calibration: state judge", "gpt-5.6-sol", "high", sum(1 for r in recs if r.get("valid")),
                        len(atts), sum(int(a.get("input_tokens") or 0) for a in atts),
                        sum(int(a.get("cached_input_tokens") or 0) for a in atts),
                        sum(int(a.get("output_tokens") or 0) for a in atts),
                        sum(float(a.get("wall_seconds") or 0) for a in atts),
                        f"experiments_data/stopping_runs/judgments/gpt-5.6-sol/{cfg}/",
                        notes=f"judged trajectories of {model} ({effort})"))
    # Executed case study.
    for meta_f in files(DATA / "case_study_agent_runs" / "workspaces" / "*" / "runs" / "*" / "run_meta.json"):
        m = json.load(open(meta_f))
        key = Path(meta_f).parts[-4]
        iters = m.get("iterations") or []
        calls = sum(int((it.get("agent_tools") or {}).get("calls") or 1) for it in iters)
        tin = int(m.get("total_input_tokens") or sum(int(it.get("input_tokens") or 0) for it in iters))
        tc = int(m.get("total_cached_input_tokens") or sum(int(it.get("cached_input_tokens") or 0) for it in iters))
        to = int(m.get("total_output_tokens") or sum(int(it.get("output_tokens") or 0) for it in iters))
        wall = sum(float(it.get("duration_seconds") or 0) for it in iters)
        ex = "verified execution (sandbox_exec)" if key != "sol_planonly" else "execution disabled (control)"
        rows.append(row(f"Executed case study: run {key}", m.get("model"), m.get("reasoning_effort"), calls, calls,
                        tin, tc, to, wall, f"experiments_data/case_study_agent_runs/workspaces/{key}/",
                        tools="off (executor and reviewers)", notes=ex + "; LLM-adjudicated merging"))
    # Real-backend concurrency.
    rc = json.load(open(ROOT / "experiments/orchestration/real_concurrency.json"))
    summ = rc["summary_by_level"]
    calls = sum(int(x["lens_calls"]) - int(x["failed_lens_calls_final"]) for x in summ)
    attempts = sum(int(x["attempts"]) for x in summ)
    tin = sum(tok(x.get("tokens"))[0] for x in summ)
    tc = sum(tok(x.get("tokens"))[1] for x in summ)
    to = sum(tok(x.get("tokens"))[2] for x in summ)
    wall = sum(float(x["call_latency_s_all_attempts"]["mean"]) * int(x["call_latency_s_all_attempts"]["n"]) for x in summ)
    rows.append(row("Orchestration: real-backend concurrency", rc["config"].get("model"), rc["config"].get("effort"),
                    calls, attempts, tin, tc, to, wall, "experiments/orchestration/real_concurrency.json",
                    cli=rc["environment"].get("cli_version") or "codex-cli 0.145.0",
                    notes="1, 2, 4 and 8 concurrent three-lens panels (15 panels)"))
    # Pilot calls excluded.
    pilot = files(DATA / "benchmark_runs" / "reviews" / "*" / "*" / "r*" / "*.json")
    rows.append(row("Excluded: benchmark pilot (not analysed)", "gpt-5.6-sol, gpt-5.6-luna", "medium", len(pilot), None,
                    None, None, None, None, "experiments_data/benchmark_runs/",
                    notes="pilot calls used to estimate run time and cost before the benchmark was run; not part of any reported result"))
    return pd.DataFrame(rows)


def s2_table(writer: pd.ExcelWriter) -> None:
    res = ROOT / "experiments/benchmark/results"
    sheets = [
        ("recall_by_condition", res / "recall_by_condition.csv"),
        ("contrasts", res / "contrasts.csv"),
        ("severity_recall", res / "severity_recall_by_condition.csv"),
        ("recall_by_family", res / "recall_by_family.csv"),
        ("recall_by_type", res / "recall_by_type.csv"),
        ("discrimination", res / "discrimination.csv"),
        ("false_alarms_clean", res / "false_alarms_clean.csv"),
        ("unmatched_composition", res / "unmatched_composition.csv"),
        ("cost_by_condition", res / "cost_by_condition.csv"),
        ("call_stats", res / "call_stats.csv"),
        ("sens_tfidf_severity", res / "sensitivity_tfidf/severity_recall_by_condition.csv"),
        ("sens_tfidf_false_alarms", res / "sensitivity_tfidf/false_alarms_clean.csv"),
        ("sens_tfidf_discrimination", res / "sensitivity_tfidf/discrimination.csv"),
        ("stopping_rules", ROOT / "experiments/stopping/results/stopping_rules.csv"),
    ]
    readme = [{"sheet": "README", "description": "This workbook; one sheet per result table."}]
    for name, path in sheets:
        pd.read_csv(path).to_excel(writer, sheet_name=name[:31], index=False)
        readme.append({"sheet": name[:31], "description": f"from {path.relative_to(ROOT)}"})
    br = json.load(open(res / "benchmark_results.json"))
    pf = br["primary"]["per_flaw_detection"]
    rows = []
    for model, conds in pf.items():
        for cond, flaws in (conds.items() if isinstance(conds, dict) else []):
            if isinstance(flaws, dict):
                for fid, v in flaws.items():
                    rows.append({"model": model, "condition": cond, "flaw_id": fid,
                                 **({k: v[k] for k in v if not isinstance(v[k], (dict, list))} if isinstance(v, dict) else {"value": v})})
    if rows:
        pd.DataFrame(rows).to_excel(writer, sheet_name="per_flaw_detection", index=False)
        readme.append({"sheet": "per_flaw_detection", "description": "benchmark_results.json primary.per_flaw_detection (primary judge)"})
    ag = br["judge_agreement"]
    agr = [{"level": "critique: flaw id", **ag["critique_level_flaw_id"]},
           {"level": "critique: matched vs none", **ag["critique_level_matched_vs_none"]}]
    for c, v in ag["item_flaw_level"].items():
        agr.append({"level": f"item-flaw: {c}", **v})
    pd.DataFrame(agr).to_excel(writer, sheet_name="judge_agreement", index=False)
    readme.append({"sheet": "judge_agreement", "description": "benchmark_results.json judge_agreement (Cohen's kappa, bootstrap CI)"})
    me = json.load(open(res / "merge_eval.json"))
    mrows = []
    for m, v in me["methods"].items():
        if v.get("skipped"):
            mrows.append({"method": m, "status": "not evaluated", "reason": v.get("reason")})
            continue
        h = v.get("held_out") or {}
        e = v.get("escalation_held_out_flawed") or {}
        mrows.append({"method": m, "status": "evaluated", "held_out_precision": h.get("precision"),
                      "held_out_recall": h.get("recall"), "held_out_f1": h.get("f1"),
                      "ari_held_out": v.get("ari_mean_held_out"), "uncalibrated_threshold": v.get("default_threshold"),
                      "f1_at_uncalibrated": (v.get("at_default_threshold") or {}).get("f1"),
                      "all_data_best_threshold": v.get("all_data_best_threshold"),
                      "multi_lens_groups_share_flaw": (e.get("multi_lens_groups") or {}).get("share_any_flaw"),
                      "single_lens_groups_share_flaw": (e.get("single_lens_groups") or {}).get("share_any_flaw"),
                      "escalation_recall": e.get("escalation_recall")})
    pd.DataFrame(mrows).to_excel(writer, sheet_name="merge_eval", index=False)
    readme.append({"sheet": "merge_eval", "description": "merge_eval.json: held-out (leave-one-scenario-out) pairwise metrics and escalation validity"})
    sj = json.load(open(ROOT / "experiments/stopping/results/stopping_results.json"))
    prow = []
    for cfg, v in sj["configs"].items():
        for p in v["per_revision"]:
            prow.append({"config": cfg, **{k: p[k] for k in p if not isinstance(p[k], (list, dict))}})
    pd.DataFrame(prow).to_excel(writer, sheet_name="stopping_per_revision", index=False)
    readme.append({"sheet": "stopping_per_revision", "description": "stopping_results.json configs.*.per_revision"})
    pd.DataFrame(readme).to_excel(writer, sheet_name="README", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s1 = s1_table()
    s1.to_csv(OUT / "S1_Table.csv", index=False)
    with pd.ExcelWriter(OUT / "S1_Table.xlsx") as w:
        s1.to_excel(w, sheet_name="run_register", index=False)
    with pd.ExcelWriter(OUT / "S2_Table.xlsx") as w:
        s2_table(w)
    print(s1.to_string())


if __name__ == "__main__":
    main()
