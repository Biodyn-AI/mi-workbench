"""Offline replay of candidate stopping rules on the recorded trajectories (X3).

Inputs: the complete trajectories (``<data-dir>/<config>/<item>/trajectory.json``,
from ``run_loops.py``) and the judge labels of every executor state
(``<data-dir>/judgments/<judge-model>/<config>/<item>/E<k>.json``, from
``judge_states.py``). Every rule is replayed with the platform's own
``backend.orchestrator.convergence.replay_stopping_rule`` on the event
sequence the engine saw:

    E0 (seed, iteration 0)  P0 (1)  E1 (2)  P1 (3)  ...  E5 (10)  P5 (11)

(executor events carry the full executor text; panel events carry the merged
grade, Critical / High counts, panel-partial flag and unresolved-CRITICAL
count). Rules (ANALYSIS_PLAN.md, "Stopping-rule calibration"):

* the current platform default (``DEFAULT_STOPPING_RULE``: any of
  grade-stable, no-Critical/High, output-similar; window 3, min 4
  iterations, similarity >= 0.9);
* each signal alone; all three;
* no-Critical/High AND (grade-stable OR similar)
  (``convergence_required_signals=no_critical_high``, ``convergence_rule=any``);
* the latter with the consensus gate (``gate=True``);
* fixed k = 1..5 revisions (stop after E_k);
* sensitivity variants (window 2; similarity threshold 0.8 / 0.95), reported
  separately and never used for the selection.

The state at a stop is the latest executor state: a stop at panel ``P_k`` or
at revision ``E_k`` both leave ``E_k`` (``k = iteration // 2``). A rule that
never fires runs to the horizon (state ``E5`` after ``P5``). "Before the
horizon" means ``k_stop < 5``.

Outcomes per rule (judged trajectories): premature-stop rate (stopped before
the horizon while >= 1 planted flaw is labelled ``unresolved``; the strict
variant also counts ``resolved_by_fabrication`` as not resolved), mean
unresolved flaws at stop, fabrication at stop (flaws resolved by fabrication
and new fabrication errors), cycles (executor revisions), panel reviews and
tokens consumed. Selection (pre-registered): the rule with the lowest
premature-stop rate among rules that stop before the horizon in >= 50% of
runs, ties broken by fewer cycles. Also: per-revision trajectories of grade,
severity and unresolved flaws, and Spearman correlations between panel
signals at ``P_k`` and unresolved flaws in ``E_k`` (pooled over item x k
with an item-cluster bootstrap CI, and the mean within-item correlation).

Pure Python (platform or analysis environment).

Usage:
    python experiments/stopping/replay.py
    python experiments/stopping/replay.py --configs sol-medium --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stopping_common as sc  # noqa: E402
from backend.orchestrator.convergence import (  # noqa: E402
    DEFAULT_STOPPING_RULE,
    StoppingRule,
    replay_stopping_rule,
)

DEFAULT_JUDGE_MODEL = "gpt-5.6-sol"
DEFAULT_BOOTSTRAP = 2000
DEFAULT_SEED = 20261001
SELECTION_MIN_STOP_RATE = 0.5

# ── rules ───────────────────────────────────────────────────────────────

NOCH_REQ = {"convergence_required_signals": ["no_critical_high"], "convergence_rule": "any"}


def rule_specs(horizon: int = sc.HORIZON) -> list[dict[str, Any]]:
    """Candidate rules. ``config`` holds run-config ``convergence_*`` keys
    applied on top of ``DEFAULT_STOPPING_RULE`` via ``StoppingRule.from_config``."""
    specs: list[dict[str, Any]] = [
        {"name": "default_any_of_3", "family": "primary",
         "description": "current platform default (DEFAULT_STOPPING_RULE): any of grade_stable, "
                        "no_critical_high, output_similar", "config": {}},
        {"name": "grade_stable_only", "family": "primary", "description": "grade_stable alone",
         "config": {"convergence_signals": ["grade_stable"]}},
        {"name": "no_critical_high_only", "family": "primary",
         "description": "no_critical_high alone",
         "config": {"convergence_signals": ["no_critical_high"]}},
        {"name": "output_similar_only", "family": "primary", "description": "output_similar alone",
         "config": {"convergence_signals": ["output_similar"]}},
        {"name": "all_of_3", "family": "primary", "description": "all three signals",
         "config": {"convergence_rule": "all"}},
        {"name": "noCH_and_(grade_stable_or_similar)", "family": "primary",
         "description": "no_critical_high AND (grade_stable OR output_similar)",
         "config": dict(NOCH_REQ)},
        {"name": "noCH_and_(grade_stable_or_similar)+gate", "family": "primary",
         "description": "no_critical_high AND (grade_stable OR output_similar), with the "
                        "consensus gate (no stop while unresolved CRITICAL critiques remain)",
         "config": dict(NOCH_REQ), "gate": True},
    ]
    for k in range(1, horizon + 1):
        specs.append({"name": f"fixed_k{k}", "family": "fixed",
                      "description": f"fixed horizon: stop after {k} executor revision(s)",
                      "fixed_k": k})
    sens_base = [s for s in specs if s["family"] == "primary"]
    for s in sens_base:
        cfg = dict(s["config"])
        cfg["convergence_window"] = 2
        specs.append({**s, "name": s["name"] + "@window2", "family": "sensitivity",
                      "description": s["description"] + " (window 2)", "config": cfg})
    for thr in (0.8, 0.95):
        specs.append({"name": f"output_similar_only@sim{thr:g}", "family": "sensitivity",
                      "description": f"output_similar alone, similarity >= {thr:g}",
                      "config": {"convergence_signals": ["output_similar"],
                                 "convergence_similarity_threshold": thr}})
    return specs


def rule_from_spec(spec: dict[str, Any]) -> Optional[StoppingRule]:
    if "fixed_k" in spec:
        return None
    return StoppingRule.from_config(spec.get("config") or {}, base=DEFAULT_STOPPING_RULE)


# ── inputs ──────────────────────────────────────────────────────────────


def load_trajectories(data_dir: Path, configs: Sequence[str]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for config in configs:
        cdir = Path(data_dir) / config
        if not cdir.is_dir():
            continue
        for udir in sorted(p for p in cdir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            tp = udir / "trajectory.json"
            if not tp.exists():
                continue
            traj = sc.read_json(tp)
            if not traj.get("complete"):
                continue
            texts = []
            for st in traj["states"]:
                texts.append((udir / st["path"]).read_text(encoding="utf-8"))
            traj["_texts"] = texts
            traj["_dir"] = str(udir)
            out.setdefault(config, {})[traj["item"]] = traj
    return out


def load_state_judgments(data_dir: Path, judge_model: str, config: str, item: str,
                         horizon: int) -> list[Optional[dict]]:
    """Per state k: {'labels': {flaw: label}, 'counts': {...}} or None."""
    out: list[Optional[dict]] = []
    for k in range(horizon + 1):
        p = sc.judgment_path(data_dir, judge_model, config, item, k)
        rec = None
        if p.exists():
            try:
                rec = sc.read_json(p)
            except (OSError, json.JSONDecodeError):
                rec = None
        if not rec or not rec.get("valid") or not rec.get("judgment"):
            out.append(None)
            continue
        labels = {f["flaw_id"]: f["label"] for f in rec["judgment"]["flaws"]}
        errs = rec["judgment"].get("new_errors") or []
        unresolved = sum(1 for v in labels.values() if v == "unresolved")
        fab = sum(1 for v in labels.values() if v == "resolved_by_fabrication")
        out.append({
            "labels": labels,
            "unresolved": unresolved,
            "fabricated_flaws": fab,
            "resolved": sum(1 for v in labels.values() if v == "resolved"),
            "not_genuinely_resolved": unresolved + fab,
            "new_errors": len(errs),
            "new_fabrications": sum(1 for e in errs if e.get("category") == "fabrication"),
            "new_error_categories": [e.get("category") for e in errs],
            "state_sha256": rec.get("state_sha256"),
        })
    return out


def replay_events(traj: dict) -> list[dict[str, Any]]:
    """The event sequence the engine's convergence detector saw."""
    texts = traj["_texts"]
    events: list[dict[str, Any]] = [{"iteration": 0, "role": sc.EXECUTOR_ROLE,
                                     "output": texts[0]}]
    for st in traj["steps"]:
        n = int(st["iteration"])
        if st["role"] == sc.EXECUTOR_ROLE:
            events.append({"iteration": n, "role": sc.EXECUTOR_ROLE,
                           "output": texts[n // 2]})
        else:
            events.append({
                "iteration": n, "role": st["role"], "grade": st.get("grade"),
                "critical": int(st.get("critical") or 0), "high": int(st.get("high") or 0),
                "panel_partial": bool(st.get("panel_partial")),
                "grade_valid": st.get("grade_valid", True),
                "unresolved_critical": int(st.get("unresolved_critical") or 0),
            })
    return events


# ── applying a rule ─────────────────────────────────────────────────────


def apply_rule(traj: dict, spec: dict[str, Any], horizon: int) -> dict[str, Any]:
    last = sc.total_iterations(horizon)
    if "fixed_k" in spec:
        k = int(spec["fixed_k"])
        it = 2 * k
        fired, reason = True, f"fixed:{k}"
    else:
        res = replay_stopping_rule(replay_events(traj), rule_from_spec(spec),
                                   gate=bool(spec.get("gate")))
        fired = bool(res["stopped"])
        it = int(res["iteration"]) if fired else last
        reason = res["stop_reason"] if fired else "horizon"
    k_stop = sc.state_index_at(it)
    tokens = sum(int(s["tokens"]["total"]) for s in traj["steps"] if int(s["iteration"]) <= it)
    return {
        "item": traj["item"],
        "fired": fired,
        "stopped_before_horizon": bool(fired and k_stop < horizon),
        "iteration": it,
        "label": sc.step_label(it) if it > 0 else "E0",
        "k_stop": k_stop,
        "stop_reason": reason,
        "cycles": k_stop,
        "panels": (it + 1) // 2,
        "tokens": tokens,
    }


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return (sum(xs) / len(xs)) if xs else None


def wilson(x: int, n: int, z: float = 1.959963984540054) -> Optional[list[float]]:
    if n <= 0:
        return None
    p = x / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def summarize_rule(spec: dict[str, Any], runs: list[dict[str, Any]],
                   judg: dict[str, list[Optional[dict]]], horizon: int) -> dict[str, Any]:
    n = len(runs)
    early = [r for r in runs if r["stopped_before_horizon"]]
    enriched: list[dict[str, Any]] = []
    judged: list[dict[str, Any]] = []
    for r in runs:
        j = judg.get(r["item"]) or []
        state = j[r["k_stop"]] if r["k_stop"] < len(j) else None
        r = dict(r)
        r["judged"] = state is not None
        if state is not None:
            r.update({
                "unresolved": state["unresolved"],
                "not_genuinely_resolved": state["not_genuinely_resolved"],
                "fabricated_flaws": state["fabricated_flaws"],
                "new_fabrications": state["new_fabrications"],
                "new_errors": state["new_errors"],
            })
            judged.append(r)
        enriched.append(r)
    runs = enriched
    nj = len(judged)
    prem = [r for r in judged if r["stopped_before_horizon"] and r["unresolved"] >= 1]
    prem_s = [r for r in judged if r["stopped_before_horizon"] and r["not_genuinely_resolved"] >= 1]
    early_j = [r for r in judged if r["stopped_before_horizon"]]
    out = {
        "name": spec["name"],
        "family": spec["family"],
        "description": spec["description"],
        "gate": bool(spec.get("gate")),
        "rule": (rule_from_spec(spec).to_dict() if "fixed_k" not in spec
                 else {"fixed_k": spec["fixed_k"]}),
        "n_runs": n,
        "n_judged": nj,
        "n_stopped_before_horizon": len(early),
        "stop_before_horizon_rate": (len(early) / n) if n else None,
        "n_fired": sum(1 for r in runs if r["fired"]),
        "mean_cycles": _mean([r["cycles"] for r in runs]),
        "mean_panels": _mean([r["panels"] for r in runs]),
        "mean_tokens": _mean([r["tokens"] for r in runs]),
        "k_stop_distribution": {str(k): sum(1 for r in runs if r["k_stop"] == k)
                                for k in range(horizon + 1)},
        "n_premature": len(prem),
        "premature_stop_rate": (len(prem) / nj) if nj else None,
        "premature_stop_rate_ci95": wilson(len(prem), nj),
        "n_premature_strict": len(prem_s),
        "premature_stop_rate_strict": (len(prem_s) / nj) if nj else None,
        "premature_given_early_stop": (
            (sum(1 for r in early_j if r["unresolved"] >= 1) / len(early_j)) if early_j else None),
        "any_unresolved_at_stop_rate": (
            (sum(1 for r in judged if r["unresolved"] >= 1) / nj) if nj else None),
        "mean_unresolved_at_stop": _mean([r["unresolved"] for r in judged]),
        "mean_not_genuinely_resolved_at_stop": _mean(
            [r["not_genuinely_resolved"] for r in judged]),
        "fabricated_flaws_at_stop": sum(r["fabricated_flaws"] for r in judged),
        "mean_fabricated_flaws_at_stop": _mean([r["fabricated_flaws"] for r in judged]),
        "new_fabrications_at_stop": sum(r["new_fabrications"] for r in judged),
        "mean_new_fabrications_at_stop": _mean([r["new_fabrications"] for r in judged]),
        "runs_with_fabrication_at_stop": sum(
            1 for r in judged if r["fabricated_flaws"] > 0 or r["new_fabrications"] > 0),
        "mean_new_errors_at_stop": _mean([r["new_errors"] for r in judged]),
        "runs": runs,
    }
    return out


def select_rule(summaries: list[dict[str, Any]], families: Sequence[str]) -> dict[str, Any]:
    """Pre-registered selection among ``families``."""
    cands = [s for s in summaries if s["family"] in families]
    eligible = [s for s in cands
                if s["stop_before_horizon_rate"] is not None
                and s["stop_before_horizon_rate"] >= SELECTION_MIN_STOP_RATE
                and s["premature_stop_rate"] is not None]
    if not eligible:
        return {"families": list(families), "eligible": [], "selected": None, "ties": [],
                "note": "no rule stops before the horizon in >= 50% of judged runs"}
    key = lambda s: (round(s["premature_stop_rate"], 12), round(s["mean_cycles"], 12))  # noqa: E731
    eligible.sort(key=key)
    best = key(eligible[0])
    ties = [s["name"] for s in eligible if key(s) == best]
    return {
        "families": list(families),
        "criterion": "lowest premature_stop_rate among rules with stop_before_horizon_rate "
                     ">= 0.5; ties broken by fewer mean cycles",
        "eligible": [{"name": s["name"], "premature_stop_rate": s["premature_stop_rate"],
                      "stop_before_horizon_rate": s["stop_before_horizon_rate"],
                      "mean_cycles": s["mean_cycles"]} for s in eligible],
        "selected": eligible[0]["name"],
        "ties": ties,
    }


# ── statistics ──────────────────────────────────────────────────────────


def rankdata(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    n = len(x)
    if n < 3 or n != len(y):
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 0 or syy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 3:
        return None
    return pearson(rankdata(list(x)), rankdata(list(y)))


def _quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def correlation_block(pairs_by_item: dict[str, list[tuple[float, float]]],
                      n_boot: int, seed: int) -> dict[str, Any]:
    """Pooled Spearman over all (item, k) pairs with an item-cluster bootstrap
    95% CI, plus the mean within-item Spearman (items whose two series both
    vary)."""
    items = sorted(i for i, p in pairs_by_item.items() if p)
    pooled = [p for i in items for p in pairs_by_item[i]]
    rho = spearman([a for a, _ in pooled], [b for _, b in pooled]) if pooled else None
    boots: list[float] = []
    if rho is not None and len(items) >= 2 and n_boot > 0:
        rng = random.Random(seed)
        for _ in range(n_boot):
            sample = [p for _ in items for p in pairs_by_item[items[rng.randrange(len(items))]]]
            r = spearman([a for a, _ in sample], [b for _, b in sample])
            if r is not None:
                boots.append(r)
    boots.sort()
    within = {}
    for i in items:
        r = spearman([a for a, _ in pairs_by_item[i]], [b for _, b in pairs_by_item[i]])
        if r is not None:
            within[i] = r
    return {
        "n_pairs": len(pooled),
        "n_items": len(items),
        "spearman_pooled": rho,
        "ci95_item_bootstrap": ([_quantile(boots, 0.025), _quantile(boots, 0.975)]
                                if len(boots) >= 20 else None),
        "n_bootstrap_valid": len(boots),
        "mean_within_item_spearman": _mean(list(within.values())),
        "n_items_within_defined": len(within),
        "within_item_spearman": within,
    }


# ── per-config analysis ─────────────────────────────────────────────────


def analyze_config(config: str, trajs: dict[str, dict], data_dir: Path, judge_model: str,
                   horizon: int, n_boot: int, seed: int) -> dict[str, Any]:
    items = sorted(trajs)
    judg = {i: load_state_judgments(data_dir, judge_model, config, i, horizon) for i in items}
    fully_judged = [i for i in items if all(s is not None for s in judg[i])]
    specs = rule_specs(horizon)
    summaries = []
    for spec in specs:
        runs = [apply_rule(trajs[i], spec, horizon) for i in items]
        summaries.append(summarize_rule(spec, runs, judg, horizon))

    # per-revision trajectories
    per_k = []
    for k in range(horizon + 1):
        rows = []
        for i in items:
            t = trajs[i]
            p = t["panels"][k] if k < len(t["panels"]) else None
            st = t["states"][k]
            j = judg[i][k]
            rows.append((p, st, j))
        js = [j for _, _, j in rows if j is not None]
        per_k.append({
            "k": k,
            "n": len(rows),
            "n_judged": len(js),
            "mean_unresolved": _mean([j["unresolved"] for j in js]),
            "mean_not_genuinely_resolved": _mean([j["not_genuinely_resolved"] for j in js]),
            "mean_fabricated_flaws": _mean([j["fabricated_flaws"] for j in js]),
            "mean_new_fabrications": _mean([j["new_fabrications"] for j in js]),
            "mean_new_errors": _mean([j["new_errors"] for j in js]),
            "frac_all_resolved": (_mean([1.0 if j["unresolved"] == 0 else 0.0 for j in js])
                                  if js else None),
            "frac_all_genuinely_resolved": (_mean(
                [1.0 if j["not_genuinely_resolved"] == 0 else 0.0 for j in js]) if js else None),
            "mean_grade_score": _mean([p["grade_score"] for p, _, _ in rows if p]),
            "grades": [p["grade"] for p, _, _ in rows if p],
            "mean_critical_high": _mean([p["critical"] + p["high"] for p, _, _ in rows if p]),
            "mean_critical": _mean([p["critical"] for p, _, _ in rows if p]),
            "mean_merged_critiques": _mean([p["n_critiques"] for p, _, _ in rows if p]),
            "n_partial_panels": sum(1 for p, _, _ in rows if p and p["panel_partial"]),
            "mean_similarity_to_previous": _mean(
                [st["similarity_to_previous"] for _, st, _ in rows
                 if st["similarity_to_previous"] is not None]),
            "mean_new_numbers_vs_original": _mean([st["n_new_numbers"] for _, st, _ in rows]),
            "mean_chars": _mean([st["chars"] for _, st, _ in rows]),
        })

    # correlations: panel P_k signal vs unresolved flaws in E_k
    def pairs(metric: Callable[[dict, dict, int], Optional[float]],
              target: str) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        for i in items:
            t = trajs[i]
            for k in range(horizon + 1):
                j = judg[i][k]
                if j is None or k >= len(t["panels"]):
                    continue
                v = metric(t["panels"][k], t["states"][k], k)
                if v is None:
                    continue
                out.setdefault(i, []).append((float(v), float(j[target])))
        return out

    metrics = {
        "grade_score": lambda p, s, k: p["grade_score"],
        "critical_plus_high": lambda p, s, k: p["critical"] + p["high"],
        "critical": lambda p, s, k: p["critical"],
        "merged_critiques": lambda p, s, k: p["n_critiques"],
        "executor_similarity_to_previous": lambda p, s, k: s["similarity_to_previous"],
    }
    correlations = {}
    for target in ("unresolved", "not_genuinely_resolved"):
        for mname, fn in metrics.items():
            correlations[f"{mname}__vs__{target}"] = correlation_block(
                pairs(fn, target), n_boot, seed)

    # changes between consecutive states (P_k - P_{k-1} vs E_k - E_{k-1})
    delta_pairs: dict[str, dict[str, list[tuple[float, float]]]] = {
        "delta_grade_score": {}, "delta_critical_plus_high": {}}
    for i in items:
        t = trajs[i]
        for k in range(1, horizon + 1):
            j0, j1 = judg[i][k - 1], judg[i][k]
            if j0 is None or j1 is None:
                continue
            p0, p1 = t["panels"][k - 1], t["panels"][k]
            du = float(j1["unresolved"] - j0["unresolved"])
            if p0["grade_score"] is not None and p1["grade_score"] is not None:
                delta_pairs["delta_grade_score"].setdefault(i, []).append(
                    (p1["grade_score"] - p0["grade_score"], du))
            delta_pairs["delta_critical_plus_high"].setdefault(i, []).append(
                (float((p1["critical"] + p1["high"]) - (p0["critical"] + p0["high"])), du))
    for name, pb in delta_pairs.items():
        correlations[f"{name}__vs__delta_unresolved"] = correlation_block(pb, n_boot, seed)

    # oracle reference: first state with all planted flaws resolved
    oracle = []
    for i in items:
        js = judg[i]
        first = next((k for k, j in enumerate(js) if j is not None and j["unresolved"] == 0), None)
        first_s = next((k for k, j in enumerate(js)
                        if j is not None and j["not_genuinely_resolved"] == 0), None)
        oracle.append({"item": i, "first_k_all_resolved": first,
                       "first_k_all_genuinely_resolved": first_s})

    # flaw-type resolution profile
    by_type: dict[str, dict[str, Any]] = {}
    items_meta = sc.flawed_items()
    for i in items:
        flaws = {f.flaw_id: f for f in items_meta[i].flaws} if i in items_meta else {}
        for k in range(horizon + 1):
            j = judg[i][k]
            if j is None:
                continue
            for fid, lab in j["labels"].items():
                f = flaws.get(fid)
                ftype = f.type if f else "?"
                d = by_type.setdefault(ftype, {"family": f.family if f else "?",
                                               "per_k": {}})
                pk = d["per_k"].setdefault(str(k), {"resolved": 0, "unresolved": 0,
                                                    "resolved_by_fabrication": 0})
                pk[lab] = pk.get(lab, 0) + 1

    e0 = [judg[i][0] for i in items if judg[i][0] is not None]
    primary = [s for s in summaries if s["family"] in ("primary", "fixed")]
    return {
        "config": config,
        "model": trajs[items[0]]["model"] if items else None,
        "effort": trajs[items[0]]["effort"] if items else None,
        "items": items,
        "n_trajectories": len(items),
        "n_fully_judged": len(fully_judged),
        "judge_sanity": {
            "n_E0_judged": len(e0),
            "E0_frac_flaws_unresolved": (
                sum(j["unresolved"] for j in e0) / sum(len(j["labels"]) for j in e0)
                if e0 else None),
            "E0_new_errors": sum(j["new_errors"] for j in e0),
        },
        "heuristic_crosscheck": {
            "note": "numbers in E_k absent from E0 (deterministic) vs judge fabrication labels",
            "pairs": correlation_block(
                {i: [(float(trajs[i]["states"][k]["n_new_numbers"]),
                      float(judg[i][k]["new_fabrications"] + judg[i][k]["fabricated_flaws"]))
                     for k in range(horizon + 1) if judg[i][k] is not None] for i in items},
                n_boot, seed),
        },
        "rules": summaries,
        "selection": select_rule(primary, ("primary", "fixed")),
        "selection_adaptive_only": select_rule(primary, ("primary",)),
        "per_revision": per_k,
        "correlations": correlations,
        "oracle_first_all_resolved": oracle,
        "by_flaw_type": by_type,
        "similarity_methods": sorted({m for t in trajs.values()
                                      for m in t.get("similarity_methods") or []}),
        "panels_partial_total": sum(1 for t in trajs.values() for p in t["panels"]
                                    if p["panel_partial"]),
    }


def write_csv(results: dict, path: Path) -> None:
    cols = ["config", "rule", "family", "n_runs", "n_judged", "stop_before_horizon_rate",
            "premature_stop_rate", "premature_stop_rate_strict", "premature_given_early_stop",
            "mean_unresolved_at_stop", "mean_not_genuinely_resolved_at_stop",
            "fabricated_flaws_at_stop", "new_fabrications_at_stop", "mean_cycles",
            "mean_panels", "mean_tokens"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for config, res in results["configs"].items():
            for s in res["rules"]:
                w.writerow([config, s["name"], s["family"]] + [s.get(c) for c in cols[3:]])


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--configs", default=",".join(sc.CONFIGS))
    ap.add_argument("--data-dir", default=str(sc.DEFAULT_DATA_DIR))
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--horizon", type=int, default=sc.HORIZON)
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--out", default=str(sc.RESULTS_DIR / "stopping_results.json"))
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)
    configs = sc.parse_csv(args.configs)
    trajs = load_trajectories(data_dir, configs)
    results: dict[str, Any] = {
        "schema": sc.RESULTS_SCHEMA,
        "generated_at": sc.utc_now(),
        "data_dir": str(data_dir),
        "judge_model": args.judge_model,
        "horizon": args.horizon,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "default_stopping_rule": DEFAULT_STOPPING_RULE.to_dict(),
        "rule_specs": [{k: v for k, v in s.items()} for s in rule_specs(args.horizon)],
        "definitions": {
            "state_at_stop": "latest executor state E_k, k = stop iteration // 2; E5 if the "
                             "rule never fires",
            "stopped_before_horizon": "rule fired with k_stop < horizon",
            "premature_stop": "stopped before the horizon while >= 1 planted flaw is labelled "
                              "unresolved in E_k (strict: unresolved or resolved_by_fabrication)",
            "cycles": "executor revisions up to the stop (k_stop)",
            "panels": "panel reviews up to the stop iteration",
            "tokens": "input + output tokens of all engine steps up to the stop iteration",
        },
        "configs": {},
    }
    for config in configs:
        if config not in trajs:
            continue
        results["configs"][config] = analyze_config(
            config, trajs[config], data_dir, args.judge_model, args.horizon,
            args.bootstrap, args.seed)
    out = Path(args.out)
    sc.atomic_write_json(out, results)
    write_csv(results, out.with_name("stopping_rules.csv"))
    for config, res in results["configs"].items():
        sel = res["selection"]
        print(f"{config}: {res['n_trajectories']} trajectories, {res['n_fully_judged']} fully "
              f"judged; selected={sel.get('selected')} ties={sel.get('ties')}")
        for s in res["rules"]:
            if s["family"] == "sensitivity":
                continue
            pr = s["premature_stop_rate"]
            print(f"  {s['name']:<42} stop<H={s['stop_before_horizon_rate']:.2f} "
                  f"premature={'-' if pr is None else f'{pr:.2f}'} "
                  f"unresolved@stop={s['mean_unresolved_at_stop']} "
                  f"cycles={s['mean_cycles']:.2f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
