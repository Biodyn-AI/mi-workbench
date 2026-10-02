"""Compute every X1 outcome in ANALYSIS_PLAN.md from cached reviews + judgments.

Outputs ``experiments/benchmark/results/benchmark_results.json`` and CSV tables
next to it. Works on partial data: every table reports its n (units, items).

Primary analysis uses judge 1 (``--judges``, first entry); headline recall and
the planned contrasts are re-computed with every other judge and with the
intersection of judges 1 and 2 (a critique matches a flaw only if both judges
say so). Units without a valid judgment from the judge in question are
excluded from that judge's outcomes.

Pure standard library + the platform package (for the merge), so it runs in
the analysis env (``/Users/ihorkendiukhov/anaconda3/bin/python``) or the
platform env.

The severity outcomes use the platform merge with its default settings at
freeze time (jaccard, threshold 0.5), as pre-specified; these are pinned here
and do not follow later changes of the platform defaults (``--merge-method`` /
``--merge-threshold`` select another merge for sensitivity analyses).

Usage:
    python experiments/benchmark/analyze.py --judges gpt-5.6-sol[,<judge2>]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

CONDS = tuple(common.CONDITIONS)
HIGH_SEVERITIES = ("high", "critical")
DEFAULT_SEED = 20261001


# ── data model ──────────────────────────────────────────────────────────


@dataclass
class Unit:
    model: str
    item: str
    repeat: int
    records: dict[str, dict]
    # labels[view][(tag, j)] = (flaw_id | "none", none_type | None)
    labels: dict[str, dict[tuple[str, int], tuple[str, Optional[str]]]] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.model, self.item, self.repeat)

    def critiques(self, tag: str) -> list[dict]:
        return (self.records.get(tag) or {}).get("critiques", []) or []


def _usable(rec: Optional[dict]) -> bool:
    """A successful call whose output parsed into a critique format.

    Records cached before unparseable outputs were treated as failures can be
    ``success=True`` with ``parse.method == "none"``; they are not reviews."""
    if not rec or not rec.get("success"):
        return False
    return (rec.get("parse") or {}).get("method", "none") != "none"


def _record_condition(rec: dict) -> tuple:
    req = rec.get("request", {}) or {}
    env = req.get("codex_env") or {}
    return (req.get("reasoning_effort"), env.get("AGENTS.md"), env.get("AGENTS.override.md"),
            tuple(req.get("cli_extra_args") or []))


def load_units(data_dir: Path, judges: Sequence[str], items: dict[str, common.Item]) -> tuple[list[Unit], dict]:
    """Complete units (all 8 calls succeeded with a usable parse) with any
    available judgments.

    Units are excluded (and listed in ``incomplete_units`` with a reason) if a
    call is missing, failed, or "succeeded" with unparseable output, or if
    any of their records was made under a different condition (reasoning
    effort, Codex AGENTS.md hash, Codex tools-off arguments) than the
    model's majority condition, so a run never silently pools conditions."""
    by_unit: dict[tuple[str, str, int], dict[str, dict]] = defaultdict(dict)
    all_records = common.discover_reviews(data_dir)
    for rec in all_records:
        k = rec.get("key", {})
        if k.get("item") not in items:
            continue
        by_unit[(k["model"], k["item"], int(k["repeat"]))][k["tag"]] = rec
    # Majority condition per model (over all its records).
    cond_counts: dict[str, Counter] = defaultdict(Counter)
    for (model, _item, _rep), recs in by_unit.items():
        for rec in recs.values():
            cond_counts[model][_record_condition(rec)] += 1
    majority = {m: c.most_common(1)[0][0] for m, c in cond_counts.items() if c}
    units: list[Unit] = []
    incomplete = []
    mixed_conditions = []
    for key in sorted(by_unit, key=lambda u: (u[0], u[1], u[2])):
        recs = by_unit[key]
        complete = all(_usable(recs.get(t)) for t in common.ALL_CALLS)
        if not complete:
            entry = {"model": key[0], "item": key[1], "repeat": key[2],
                     "missing_or_failed": [t for t in common.ALL_CALLS
                                           if not (recs.get(t) or {}).get("success")]}
            unparsed = [t for t in common.ALL_CALLS
                        if (recs.get(t) or {}).get("success") and not _usable(recs.get(t))]
            if unparsed:
                entry["unparsed"] = unparsed
                entry["reason"] = "successful call(s) with unparseable output"
            incomplete.append(entry)
            continue
        odd = [t for t in common.ALL_CALLS
               if _record_condition(recs[t]) != majority.get(key[0])]
        if odd:
            mixed_conditions.append({"model": key[0], "item": key[1], "repeat": key[2],
                                     "calls": odd,
                                     "reason": "record condition (effort / AGENTS.md / "
                                               "tools-off args) differs from the model's majority"})
            continue
        u = Unit(model=key[0], item=key[1], repeat=key[2], records=recs)
        for jm in judges:
            p = common.judgment_path(data_dir, jm, key[0], key[1], key[2])
            if not p.exists():
                continue
            j = common.read_json(p)
            if not j.get("valid"):
                continue
            # The judgment must describe the current critiques.
            current = {
                t: common.sha256_text((recs.get(t) or {}).get("raw_output") or "")
                for t in common.ALL_CALLS
            }
            if j.get("request", {}).get("review_raw_output_sha256") not in (None, current):
                continue
            lab: dict[tuple[str, int], tuple[str, Optional[str]]] = {}
            labels = {lab_["index"]: lab_ for lab_ in j.get("labels", [])}
            for c in j.get("critiques", []):
                lb = labels.get(c["index"])
                if lb is None:
                    continue
                lab[(c["tag"], int(c["critique_index"]))] = (lb["flaw_id"], lb.get("none_type"))
            n_expected = sum(len(u.critiques(t)) for t in common.ALL_CALLS)
            if len(lab) != n_expected:
                continue
            u.labels[jm] = lab
        units.append(u)
    meta = {"n_review_records": len(all_records), "n_complete_units": len(units),
            "incomplete_units": incomplete,
            "excluded_mixed_condition_units": mixed_conditions,
            "conditions_by_model": {m: {"reasoning_effort": c[0], "agents_md_sha256": c[1],
                                        "agents_override_md_sha256": c[2],
                                        "cli_extra_args": list(c[3])}
                                    for m, c in majority.items()}}
    return units, meta


def add_intersection_view(units: list[Unit], j1: str, j2: str, name: str = "intersection") -> None:
    """A critique matches flaw f only if both judges assign f (else none)."""
    for u in units:
        if j1 in u.labels and j2 in u.labels:
            a, b = u.labels[j1], u.labels[j2]
            lab = {}
            for k, (fa, nta) in a.items():
                fb, ntb = b.get(k, ("none", None))
                if fa != "none" and fa == fb:
                    lab[k] = (fa, None)
                else:
                    lab[k] = ("none", nta if fa == "none" else (ntb if fb == "none" else None))
            u.labels[name] = lab


# ── statistics helpers (pure Python) ────────────────────────────────────


def mean(xs: Sequence[float]) -> Optional[float]:
    return (math.fsum(xs) / len(xs)) if xs else None


def percentile(sorted_xs: Sequence[float], q: float) -> Optional[float]:
    if not sorted_xs:
        return None
    pos = (len(sorted_xs) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return sorted_xs[lo]
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (pos - lo)


def cluster_bootstrap_mean(clusters: dict[str, list[float]], B: int, rng: random.Random) -> dict:
    """Mean over all values; CI by resampling clusters (items) with replacement."""
    keys = [k for k, v in clusters.items() if v]
    vals = [x for k in keys for x in clusters[k]]
    est = mean(vals)
    out = {"estimate": est, "ci_low": None, "ci_high": None,
           "n_values": len(vals), "n_clusters": len(keys)}
    if not keys:
        return out
    sums = [math.fsum(clusters[k]) for k in keys]
    cnts = [len(clusters[k]) for k in keys]
    m = len(keys)
    boots = []
    for _ in range(B):
        s = c = 0.0
        for _i in range(m):
            j = rng.randrange(m)
            s += sums[j]
            c += cnts[j]
        boots.append(s / c)
    boots.sort()
    out["ci_low"], out["ci_high"] = percentile(boots, 0.025), percentile(boots, 0.975)
    out["_boots"] = boots
    return out


def bootstrap_p_two_sided(boots: Sequence[float]) -> Optional[float]:
    """Two-sided p from the bootstrap distribution of a difference (+1 corrected)."""
    if not boots:
        return None
    B = len(boots)
    le = sum(1 for b in boots if b <= 0)
    ge = sum(1 for b in boots if b >= 0)
    return min(1.0, 2.0 * min((le + 1) / (B + 1), (ge + 1) / (B + 1)))


def holm(pvals: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    items = [(k, p) for k, p in pvals.items() if p is not None]
    items.sort(key=lambda t: t[1])
    m = len(items)
    adj: dict[str, Optional[float]] = {k: None for k in pvals}
    running = 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def auroc(pos: Sequence[float], neg: Sequence[float]) -> Optional[float]:
    """Mann-Whitney AUROC (ties count 1/2)."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = r
        i = j + 1
    r_pos = math.fsum(r for r, (_, lab) in zip(ranks, allv) if lab == 1)
    n1, n0 = len(pos), len(neg)
    return (r_pos - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def stratified_auroc_bootstrap(pos_clusters: dict[str, list[float]], neg_clusters: dict[str, list[float]],
                               B: int, rng: random.Random) -> dict:
    pos = [x for v in pos_clusters.values() for x in v]
    neg = [x for v in neg_clusters.values() for x in v]
    est = auroc(pos, neg)
    out = {"estimate": est, "ci_low": None, "ci_high": None, "n_pos": len(pos), "n_neg": len(neg),
           "n_pos_items": len(pos_clusters), "n_neg_items": len(neg_clusters)}
    if est is None:
        return out
    pk, nk = list(pos_clusters), list(neg_clusters)
    boots = []
    for _ in range(B):
        bp = [x for _i in range(len(pk)) for x in pos_clusters[pk[rng.randrange(len(pk))]]]
        bn = [x for _i in range(len(nk)) for x in neg_clusters[nk[rng.randrange(len(nk))]]]
        a = auroc(bp, bn)
        if a is not None:
            boots.append(a)
    boots.sort()
    out["ci_low"], out["ci_high"] = percentile(boots, 0.025), percentile(boots, 0.975)
    return out


def cohen_kappa_from_counts(counts: Counter) -> Optional[float]:
    n = sum(counts.values())
    if n == 0:
        return None
    po = sum(v for (a, b), v in counts.items() if a == b) / n
    ra, rb = Counter(), Counter()
    for (a, b), v in counts.items():
        ra[a] += v
        rb[b] += v
    pe = sum(ra[c] * rb.get(c, 0) for c in ra) / (n * n)
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else None
    return (po - pe) / (1.0 - pe)


def kappa_bootstrap(per_item_counts: dict[str, Counter], B: int, rng: random.Random) -> dict:
    total = Counter()
    for c in per_item_counts.values():
        total.update(c)
    est = cohen_kappa_from_counts(total)
    out = {"kappa": est, "ci_low": None, "ci_high": None, "n": sum(total.values()),
           "n_items": len(per_item_counts),
           "percent_agreement": (sum(v for (a, b), v in total.items() if a == b) / sum(total.values()))
           if total else None}
    keys = list(per_item_counts)
    if est is None or not keys:
        return out
    boots = []
    for _ in range(B):
        c = Counter()
        for _i in range(len(keys)):
            c.update(per_item_counts[keys[rng.randrange(len(keys))]])
        k = cohen_kappa_from_counts(c)
        if k is not None:
            boots.append(k)
    boots.sort()
    out["ci_low"], out["ci_high"] = percentile(boots, 0.025), percentile(boots, 0.975)
    return out


def strip_boots(d: Any) -> Any:
    if isinstance(d, dict):
        return {k: strip_boots(v) for k, v in d.items() if k != "_boots"}
    if isinstance(d, list):
        return [strip_boots(v) for v in d]
    return d


def rnd(x: Any, nd: int = 4) -> Any:
    if isinstance(x, float):
        return round(x, nd)
    if isinstance(x, dict):
        return {k: rnd(v, nd) for k, v in x.items()}
    if isinstance(x, list):
        return [rnd(v, nd) for v in x]
    return x


# ── per-unit computations ───────────────────────────────────────────────


class UnitView:
    """Detections / merges of one unit under one label view."""

    def __init__(self, unit: Unit, view: str, item: common.Item, merge_kwargs: dict):
        self.unit = unit
        self.item = item
        self.labels = unit.labels[view]
        self.merge_kwargs = merge_kwargs
        self._merge_cache: dict[str, Any] = {}

    def flaw_of(self, tag: str, j: int) -> str:
        return self.labels.get((tag, j), ("none", None))[0]

    def detected(self, calls: Iterable[str]) -> set[str]:
        out = set()
        for tag in calls:
            for j in range(len(self.unit.critiques(tag))):
                f = self.flaw_of(tag, j)
                if f != "none":
                    out.add(f)
        return out

    def merged(self, cond: str):
        if cond not in self._merge_cache:
            crit = {t: self.unit.critiques(t) for t in common.CONDITIONS[cond]}
            self._merge_cache[cond] = common.merge_calls(crit, common.CONDITIONS[cond], **self.merge_kwargs)
        return self._merge_cache[cond]

    def severity_detected(self, cond: str) -> set[str]:
        outcome, index = self.merged(cond)
        out = set()
        for crit, group in zip(outcome.result.critiques, outcome.groups):
            if crit.severity.value not in HIGH_SEVERITIES:
                continue
            for pi in group:
                tag, j = index[pi]
                f = self.flaw_of(tag, j)
                if f != "none":
                    out.add(f)
        return out


def merged_scores(unit: Unit, cond: str, merge_kwargs: dict) -> dict:
    crit = {t: unit.critiques(t) for t in common.CONDITIONS[cond]}
    outcome, _ = common.merge_calls(crit, common.CONDITIONS[cond], **merge_kwargs)
    sevs = [c.severity.value for c in outcome.result.critiques]
    return {
        "grade": outcome.result.overall_grade,
        "grade_ordinal": common.GRADE_ORDINAL.get(outcome.result.overall_grade),
        "severity_score": sum(common.SEVERITY_SCORE.get(s, 0) for s in sevs),
        "n_merged": len(sevs),
        "n_high_critical_merged": sum(1 for s in sevs if s in HIGH_SEVERITIES),
        "n_raw": sum(len(v) for v in crit.values()),
        "n_high_critical_raw": sum(1 for v in crit.values() for c in v
                                   if str(c.get("severity")) in HIGH_SEVERITIES),
        "n_multi_lens_groups": outcome.info.get("n_groups_multi_lens", 0),
    }


# ── analysis for one label view ─────────────────────────────────────────


def analyze_view(units: list[Unit], view: str, items: dict[str, common.Item], merge_kwargs: dict,
                 B_contrast: int, B_desc: int, seed: int, headline_only: bool = False) -> dict:
    rng = random.Random(seed)
    vunits = [u for u in units if view in u.labels]
    flawed_units = [u for u in vunits if items[u.item].is_flawed]
    clean_units = [u for u in vunits if not items[u.item].is_flawed]
    models = sorted({u.model for u in vunits})
    groups = {m: [u for u in flawed_units if u.model == m] for m in models}
    groups["pooled"] = list(flawed_units)

    views = {u.key: UnitView(u, view, items[u.item], merge_kwargs) for u in vunits}
    det: dict[tuple, dict[str, set[str]]] = {}
    sev: dict[tuple, dict[str, set[str]]] = {}
    for u in flawed_units:
        uv = views[u.key]
        det[u.key] = {c: uv.detected(common.CONDITIONS[c]) for c in CONDS}
        if not headline_only:
            sev[u.key] = {c: uv.severity_detected(c) for c in CONDS}

    def recall(u: Unit, c: str, table) -> float:
        fl = items[u.item].flaw_ids
        return len(table[u.key][c] & set(fl)) / len(fl)

    out: dict[str, Any] = {
        "n_units": len(vunits), "n_flawed_units": len(flawed_units), "n_clean_units": len(clean_units),
        "models": models,
    }

    # Primary: flaw recall per condition per model + pooled.
    rec_tab: dict[str, dict] = {}
    boots_store: dict[tuple[str, str], list[float]] = {}
    for g, us in groups.items():
        rec_tab[g] = {}
        for c in CONDS:
            clusters: dict[str, list[float]] = defaultdict(list)
            for u in us:
                clusters[u.item].append(recall(u, c, det))
            res = cluster_bootstrap_mean(dict(clusters), B_desc, rng)
            rec_tab[g][c] = res
    out["recall"] = rec_tab

    # Planned contrasts with paired, item-clustered bootstrap + Holm per model.
    contrasts: dict[str, dict] = {}
    for g, us in groups.items():
        cres = {}
        for a, b in common.PLANNED_CONTRASTS:
            clusters = defaultdict(list)
            for u in us:
                clusters[u.item].append(recall(u, a, det) - recall(u, b, det))
            res = cluster_bootstrap_mean(dict(clusters), B_contrast, rng)
            res["p_boot"] = bootstrap_p_two_sided(res.get("_boots", []))
            cres[f"{a}-{b}"] = res
        adj = holm({k: v["p_boot"] for k, v in cres.items()})
        for k in cres:
            cres[k]["p_holm"] = adj[k]
        contrasts[g] = cres
    out["contrasts"] = contrasts
    if headline_only:
        return strip_boots(out)

    # Severity-aware recall.
    out["severity_recall"] = {
        g: {c: cluster_bootstrap_mean(
            {i: v for i, v in _group_by_item(us, lambda u, c=c: recall(u, c, sev)).items()}, B_desc, rng)
            for c in CONDS}
        for g, us in groups.items()
    }

    # Recall by family and by type (detection rate over planted flaws).
    def by_attr(attr: str) -> dict:
        res: dict[str, dict] = {}
        for g, us in groups.items():
            res[g] = {}
            for c in CONDS:
                per: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
                for u in us:
                    for f in items[u.item].flaws:
                        per[getattr(f, attr)][u.item].append(1.0 if f.flaw_id in det[u.key][c] else 0.0)
                res[g][c] = {k: cluster_bootstrap_mean(dict(v), B_desc, rng) for k, v in sorted(per.items())}
        return res

    out["recall_by_family"] = by_attr("family")
    out["recall_by_type"] = by_attr("type")

    # Per-flaw detection rates (incl. the T1 attention-symmetry instances).
    per_flaw: dict[str, dict] = {}
    sym_ids = sorted({f.flaw_id for i in items.values() for f in i.flaws if f.is_symmetry_instance})
    for g, us in groups.items():
        per_flaw[g] = {}
        for c in CONDS:
            rates: dict[str, list[float]] = defaultdict(list)
            for u in us:
                for f in items[u.item].flaws:
                    rates[f.flaw_id].append(1.0 if f.flaw_id in det[u.key][c] else 0.0)
            per_flaw[g][c] = {k: {"rate": mean(v), "n": len(v)} for k, v in sorted(rates.items())}
    sym = {}
    for g, us in groups.items():
        sym[g] = {}
        for c in CONDS:
            clusters = defaultdict(list)
            for u in us:
                for f in items[u.item].flaws:
                    if f.flaw_id in sym_ids:
                        clusters[u.item].append(1.0 if f.flaw_id in det[u.key][c] else 0.0)
            sym[g][c] = cluster_bootstrap_mean(dict(clusters), B_desc, rng)
    out["per_flaw_detection"] = per_flaw
    out["t1_symmetry"] = {"flaw_ids": sym_ids, "detection": sym}

    # Unique contribution of each panel lens within C5.
    uniq: dict[str, dict] = {}
    for g, us in groups.items():
        counts = Counter()
        n_flaws = 0
        for u in us:
            uv = views[u.key]
            by_lens = {t: uv.detected([t]) for t in common.CONDITIONS["C5"]}
            for f in items[u.item].flaw_ids:
                n_flaws += 1
                found_by = [t for t, s in by_lens.items() if f in s]
                if len(found_by) == 1:
                    counts[f"only_{found_by[0]}"] += 1
                elif len(found_by) >= 2:
                    counts["two_or_more_lenses"] += 1
                else:
                    counts["none"] += 1
        uniq[g] = {"n_unit_flaws": n_flaws, "n_units": len(us),
                   **{k: {"count": counts.get(k, 0),
                          "per_unit": (counts.get(k, 0) / len(us)) if us else None,
                          "share_of_flaws": (counts.get(k, 0) / n_flaws) if n_flaws else None}
                      for k in ("only_rig1", "only_adv", "only_bio", "two_or_more_lenses", "none")}}
    out["unique_lens_contribution_C5"] = uniq

    # Discrimination (flawed vs clean) and false alarms on clean items.
    scores: dict[tuple, dict[str, dict]] = {
        u.key: {c: merged_scores(u, c, merge_kwargs) for c in CONDS} for u in vunits
    }
    disc: dict[str, dict] = {}
    paired: dict[str, dict] = {}
    alarms: dict[str, dict] = {}
    for g in models + ["pooled"]:
        us = [u for u in vunits if g == "pooled" or u.model == g]
        disc[g], paired[g], alarms[g] = {}, {}, {}
        for c in CONDS:
            disc[g][c] = {}
            for metric in ("grade_ordinal", "severity_score"):
                pos, neg = defaultdict(list), defaultdict(list)
                for u in us:
                    v = scores[u.key][c][metric]
                    if v is None:
                        continue
                    (pos if items[u.item].is_flawed else neg)[u.item].append(float(v))
                disc[g][c][metric] = stratified_auroc_bootstrap(dict(pos), dict(neg), B_desc, rng)
            # paired flawed-clean comparison on matched scenarios
            pr = {"severity_score": [], "grade_ordinal": []}
            idx = {(u.model, u.item, u.repeat): u for u in us}
            for u in us:
                if not items[u.item].is_flawed:
                    continue
                twin = idx.get((u.model, f"{items[u.item].scenario}-clean", u.repeat))
                if twin is None:
                    continue
                for metric in pr:
                    a, b = scores[u.key][c][metric], scores[twin.key][c][metric]
                    if a is not None and b is not None:
                        pr[metric].append(a - b)
            paired[g][c] = {
                m: {"n_pairs": len(v), "mean_diff_flawed_minus_clean": mean(v),
                    "frac_flawed_higher": (sum(1 for d in v if d > 0) / len(v)) if v else None,
                    "frac_tied": (sum(1 for d in v if d == 0) / len(v)) if v else None}
                for m, v in pr.items()
            }
            cu = [u for u in us if not items[u.item].is_flawed]
            alarms[g][c] = {
                "n_clean_units": len(cu),
                "mean_high_critical_merged": mean([scores[u.key][c]["n_high_critical_merged"] for u in cu]),
                "mean_high_critical_raw": mean([scores[u.key][c]["n_high_critical_raw"] for u in cu]),
                "frac_units_any_high_critical": (
                    sum(1 for u in cu if scores[u.key][c]["n_high_critical_merged"] > 0) / len(cu)) if cu else None,
                "grade_distribution": dict(Counter(scores[u.key][c]["grade"] for u in cu)),
            }
    out["discrimination_auroc"] = disc
    out["paired_flawed_vs_clean"] = paired
    out["false_alarms_clean"] = alarms

    # Composition of unmatched critiques (raw critiques of each condition's calls).
    comp: dict[str, dict] = {}
    for g in models + ["pooled"]:
        us = [u for u in vunits if g == "pooled" or u.model == g]
        comp[g] = {}
        for c in CONDS:
            for subset in ("flawed", "clean", "all"):
                cnt = Counter()
                for u in us:
                    if subset != "all" and (items[u.item].is_flawed != (subset == "flawed")):
                        continue
                    lab = u.labels[view]
                    for t in common.CONDITIONS[c]:
                        for j in range(len(u.critiques(t))):
                            f, nt = lab.get((t, j), ("none", None))
                            cnt["matched" if f != "none" else f"none_{nt}"] += 1
                total = sum(cnt.values())
                comp[g].setdefault(c, {})[subset] = {
                    "n_critiques": total,
                    **{k: {"count": cnt.get(k, 0), "share": (cnt.get(k, 0) / total) if total else None}
                       for k in ("matched", "none_substantive", "none_generic", "none_incorrect")},
                }
    out["unmatched_composition"] = comp

    # Per-unit table for CSV.
    rows = []
    for u in flawed_units:
        for c in CONDS:
            for f in items[u.item].flaws:
                rows.append({
                    "view": view, "model": u.model, "item": u.item, "repeat": u.repeat,
                    "condition": c, "flaw_id": f.flaw_id, "type": f.type, "family": f.family,
                    "detected": int(f.flaw_id in det[u.key][c]),
                    "severity_detected": int(f.flaw_id in sev[u.key][c]),
                })
    out["_per_unit_rows"] = rows
    out["_scores"] = [
        {"view": view, "model": u.model, "item": u.item, "repeat": u.repeat,
         "version": items[u.item].version, "condition": c, **scores[u.key][c]}
        for u in vunits for c in CONDS
    ]
    return strip_boots(out)


def _group_by_item(us: list[Unit], fn: Callable[[Unit], float]) -> dict[str, list[float]]:
    d: dict[str, list[float]] = defaultdict(list)
    for u in us:
        d[u.item].append(fn(u))
    return dict(d)


# ── judge-independent summaries ─────────────────────────────────────────


def cost_summary(units: list[Unit]) -> dict:
    out: dict[str, dict] = {}
    models = sorted({u.model for u in units})
    for g in models + ["pooled"]:
        us = [u for u in units if g == "pooled" or u.model == g]
        out[g] = {}
        for c in CONDS:
            per = defaultdict(list)
            for u in us:
                recs = [u.records[t] for t in common.CONDITIONS[c]]
                tok = [r.get("tokens", {}) for r in recs]
                per["input"].append(sum(t.get("input", 0) for t in tok))
                per["cached_input"].append(sum(t.get("cached_input", 0) for t in tok))
                per["uncached_input"].append(sum(t.get("input", 0) - t.get("cached_input", 0) for t in tok))
                per["output"].append(sum(t.get("output", 0) for t in tok))
                per["reasoning_output"].append(sum(t.get("reasoning_output", 0) for t in tok))
                per["wall_sum"].append(sum(r.get("wall_seconds", 0) for r in recs))
                per["wall_max"].append(max(r.get("wall_seconds", 0) for r in recs))
                per["attempts"].append(sum(len(r.get("attempts", [])) for r in recs))
                # Total cost including failed / retried attempts (the fields
                # above are the successful calls' own cost).
                att = [a for r in recs for a in (r.get("attempts") or [])]
                atok = [a.get("tokens") or {} for a in att]
                per["wall_sum_incl_retries"].append(sum(a.get("wall_seconds", 0) or 0 for a in att))
                per["input_incl_retries"].append(sum(t.get("input", 0) for t in atok))
                per["cached_input_incl_retries"].append(sum(t.get("cached_input", 0) for t in atok))
                per["output_incl_retries"].append(sum(t.get("output", 0) for t in atok))
                per["reasoning_output_incl_retries"].append(
                    sum(t.get("reasoning_output", 0) for t in atok))
                per["failed_attempts"].append(sum(1 for a in att if not a.get("success")))
            out[g][c] = {"calls": len(common.CONDITIONS[c]), "n_units": len(us),
                         **{f"mean_{k}": mean(v) for k, v in per.items()}}
    return out


COST_DEFINITIONS = {
    "mean_input / mean_output / mean_wall_sum ...": "cost of the successful call of each "
                                                   "record (last attempt)",
    "*_incl_retries": "total over every attempt of each record, failed / retried ones included",
    "mean_failed_attempts": "failed attempts per unit and condition",
}


def call_stats(records: list[dict]) -> dict:
    """Per model x call-type reliability, parsing and token statistics."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        k = r.get("key", {})
        groups[(k.get("model"), k.get("tag"))].append(r)
        groups[(k.get("model"), "ALL")].append(r)
        groups[("ALL", "ALL")].append(r)
    out = {}
    for (m, t), rs in sorted(groups.items()):
        ok = [r for r in rs if r.get("success")]
        pm = Counter((r.get("parse") or {}).get("method", "none") for r in ok)
        pd = Counter((r.get("parse") or {}).get("detail", "") for r in ok)
        unparsed_records = [r for r in rs if not r.get("success")
                            and (r.get("parse") or {}).get("method") == "none"]
        out[f"{m}|{t}"] = {
            "model": m, "tag": t, "n_records": len(rs), "n_success": len(ok),
            "success_rate": len(ok) / len(rs) if rs else None,
            "attempts_total": sum(len(r.get("attempts", [])) for r in rs),
            "parse_methods": dict(pm),
            "parse_details": dict(pd),
            "stripping_changed_parse": sum(1 for r in ok if (r.get("parse") or {}).get("stripping_changed_parse")),
            # Contract health: calls that broke the one-block rule, calls with
            # unreadable entries, critiques only in earlier blocks, and calls
            # whose output never parsed (failed records).
            "multi_block_calls": sum(1 for r in ok
                                     if ((r.get("parse") or {}).get("n_contract_blocks") or 0) > 1),
            "calls_with_dropped_items": sum(1 for r in ok
                                            if (r.get("parse") or {}).get("n_dropped")),
            "dropped_earlier_critiques": sum((r.get("parse") or {}).get("dropped_earlier") or 0
                                             for r in ok),
            "unparsed_failed_records": len(unparsed_records),
            "unparsed_attempts": sum(1 for r in rs for a in (r.get("attempts") or [])
                                     if str(a.get("error") or "").startswith("[unparsed]")),
            "mean_critiques": mean([len(r.get("critiques", [])) for r in ok]),
            "zero_critique_calls": sum(1 for r in ok if not r.get("critiques")),
            "tool_event_calls": sum(1 for r in ok if r.get("tool_events")),
            "mean_wall_seconds": mean([r.get("wall_seconds", 0) for r in ok]),
            "max_wall_seconds": max([r.get("wall_seconds", 0) for r in ok], default=None),
            "mean_input_tokens": mean([r["tokens"]["input"] for r in ok]),
            "mean_cached_input_tokens": mean([r["tokens"]["cached_input"] for r in ok]),
            "mean_output_tokens": mean([r["tokens"]["output"] for r in ok]),
            "mean_reasoning_output_tokens": mean([r["tokens"].get("reasoning_output", 0) for r in ok]),
            "severity_distribution": dict(Counter(c.get("severity") for r in ok for c in r.get("critiques", []))),
        }
    return out


def judge_agreement(units: list[Unit], j1: str, j2: str, items: dict[str, common.Item],
                    B: int, seed: int) -> dict:
    rng = random.Random(seed)
    both = [u for u in units if j1 in u.labels and j2 in u.labels]
    multi, binary = defaultdict(Counter), defaultdict(Counter)
    for u in both:
        a, b = u.labels[j1], u.labels[j2]
        for k in a:
            fa, fb = a[k][0], b.get(k, ("none", None))[0]
            multi[u.item][(fa, fb)] += 1
            binary[u.item][(fa != "none", fb != "none")] += 1
    res = {"judges": [j1, j2], "n_units": len(both),
           "critique_level_flaw_id": kappa_bootstrap(dict(multi), B, rng),
           "critique_level_matched_vs_none": kappa_bootstrap(dict(binary), B, rng),
           "item_flaw_level": {}}
    for c in list(CONDS) + ["all_conditions"]:
        per_item = defaultdict(Counter)
        for u in both:
            if not items[u.item].is_flawed:
                continue
            conds = CONDS if c == "all_conditions" else (c,)
            for cc in conds:
                da = {u.labels[j1][(t, j)][0] for t in common.CONDITIONS[cc]
                      for j in range(len(u.critiques(t)))}
                db = {u.labels[j2][(t, j)][0] for t in common.CONDITIONS[cc]
                      for j in range(len(u.critiques(t)))}
                for f in items[u.item].flaw_ids:
                    per_item[u.item][(f in da, f in db)] += 1
        res["item_flaw_level"][c] = kappa_bootstrap(dict(per_item), B, rng)
    return res


# ── output ──────────────────────────────────────────────────────────────


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: rnd(v) for k, v in r.items()})


def flat_tables(view_name: str, res: dict) -> dict[str, list[dict]]:
    t: dict[str, list[dict]] = defaultdict(list)
    for key, table in (("recall", "recall_by_condition"), ("severity_recall", "severity_recall_by_condition")):
        for g, conds in res.get(key, {}).items():
            for c, v in conds.items():
                t[table].append({"view": view_name, "model": g, "condition": c,
                                 "label": common.CONDITION_LABELS[c], "estimate": v["estimate"],
                                 "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                                 "n_units": v["n_values"], "n_items": v["n_clusters"]})
    for g, cs in res.get("contrasts", {}).items():
        for k, v in cs.items():
            t["contrasts"].append({"view": view_name, "model": g, "contrast": k,
                                   "mean_diff": v["estimate"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                                   "p_boot": v["p_boot"], "p_holm": v["p_holm"],
                                   "n_units": v["n_values"], "n_items": v["n_clusters"]})
    for attr, table in (("recall_by_family", "recall_by_family"), ("recall_by_type", "recall_by_type")):
        for g, conds in res.get(attr, {}).items():
            for c, per in conds.items():
                for k, v in per.items():
                    t[table].append({"view": view_name, "model": g, "condition": c, attr.split("_")[-1]: k,
                                     "estimate": v["estimate"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                                     "n": v["n_values"], "n_items": v["n_clusters"]})
    for g, conds in res.get("discrimination_auroc", {}).items():
        for c, per in conds.items():
            for metric, v in per.items():
                pr = res["paired_flawed_vs_clean"][g][c][metric]
                t["discrimination"].append({"view": view_name, "model": g, "condition": c, "score": metric,
                                            "auroc": v["estimate"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                                            "n_flawed": v["n_pos"], "n_clean": v["n_neg"],
                                            "paired_n": pr["n_pairs"],
                                            "paired_mean_diff": pr["mean_diff_flawed_minus_clean"],
                                            "paired_frac_flawed_higher": pr["frac_flawed_higher"]})
    for g, conds in res.get("false_alarms_clean", {}).items():
        for c, v in conds.items():
            t["false_alarms_clean"].append({"view": view_name, "model": g, "condition": c,
                                            **{k: v[k] for k in v if k != "grade_distribution"},
                                            "grades": json.dumps(v["grade_distribution"])})
    for g, conds in res.get("unmatched_composition", {}).items():
        for c, subsets in conds.items():
            for s, v in subsets.items():
                row = {"view": view_name, "model": g, "condition": c, "items": s, "n_critiques": v["n_critiques"]}
                for k in ("matched", "none_substantive", "none_generic", "none_incorrect"):
                    row[f"{k}_count"] = v[k]["count"]
                    row[f"{k}_share"] = v[k]["share"]
                t["unmatched_composition"].append(row)
    t["per_unit_detections"] = res.get("_per_unit_rows", [])
    t["per_unit_scores"] = res.get("_scores", [])
    return t


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judges", default="gpt-5.6-sol",
                    help="comma-separated judge models; the first is primary")
    ap.add_argument("--data-dir", default=str(common.DEFAULT_DATA_DIR))
    ap.add_argument("--out-dir", default=str(common.RESULTS_DIR))
    ap.add_argument("--bootstrap", type=int, default=10000, help="resamples for the planned contrasts")
    ap.add_argument("--bootstrap-descriptive", type=int, default=2000,
                    help="resamples for descriptive CIs and kappa")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--merge-method", default=common.DEFAULT_MERGE_METHOD,
                    help="merge for the severity outcomes (default: jaccard, the platform "
                         "merge's default at freeze time, as pre-specified in "
                         "ANALYSIS_PLAN.md; NOT the current platform default, llm)")
    ap.add_argument("--merge-threshold", type=float, default=None,
                    help="similarity threshold; default: the freeze-time default of the "
                         "merge method (jaccard 0.5; tfidf 0.3; embedding 0.7), NOT the "
                         "calibrated platform default 0.1")
    ap.add_argument("--lens-identity", choices=("tag", "role"), default="tag",
                    help="reviewer identity for escalation: each call (tag) or lens role")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    items = common.load_items()
    try:
        manifest = common.verify_manifest()
        manifest_status = {"verified": True, "created_utc": manifest.get("created_utc")}
    except common.ManifestError as exc:
        manifest_status = {"verified": False, "error": str(exc)}
    judges = common.parse_csv_arg(args.judges)
    units, load_meta = load_units(data_dir, judges, items)
    # Pinned: jaccard 0.5 unless set explicitly (see common.DEFAULT_MERGE_*).
    merge_threshold = common.freeze_time_threshold(args.merge_method, args.merge_threshold)
    merge_kwargs = {"method": args.merge_method, "threshold": merge_threshold,
                    "consensus_threshold": common.DEFAULT_CONSENSUS_THRESHOLD,
                    "lens_identity": args.lens_identity}
    records = common.discover_reviews(data_dir)

    results: dict[str, Any] = {
        "schema": "miw-benchmark-results/1",
        "generated_utc": common.utc_now(),
        "manifest": manifest_status,
        "settings": {"judges": judges, "primary_judge": judges[0] if judges else None,
                     "bootstrap_contrasts": args.bootstrap,
                     "bootstrap_descriptive": args.bootstrap_descriptive, "seed": args.seed,
                     "merge": merge_kwargs, "conditions": common.CONDITIONS,
                     "planned_contrasts": [f"{a}-{b}" for a, b in common.PLANNED_CONTRASTS],
                     "recall_definition": "fraction of the item's planted flaws identified by >=1 "
                                          "critique of the condition's calls (judge label)",
                     "severity_recall_definition": "planted flaw identified by a critique whose merged "
                                                   "group has severity high or critical"},
        "data": {**load_meta,
                 "n_units_by_model": dict(Counter(u.model for u in units)),
                 "n_units_judged": {jm: sum(1 for u in units if jm in u.labels) for jm in judges}},
        "call_stats": call_stats(records),
        "cost_by_condition": cost_summary(units),
        "cost_definitions": COST_DEFINITIONS,
    }
    tables: dict[str, list[dict]] = defaultdict(list)
    if judges:
        primary = judges[0]
        res = analyze_view(units, primary, items, merge_kwargs, args.bootstrap,
                           args.bootstrap_descriptive, args.seed)
        for k, v in flat_tables(f"judge:{primary}", res).items():
            tables[k].extend(v)
        results["primary"] = {k: v for k, v in res.items() if not k.startswith("_")}
        sens: dict[str, Any] = {}
        for jm in judges[1:]:
            r2 = analyze_view(units, jm, items, merge_kwargs, args.bootstrap,
                              args.bootstrap_descriptive, args.seed, headline_only=True)
            sens[f"judge:{jm}"] = r2
            for k, v in flat_tables(f"judge:{jm}", r2).items():
                tables[k].extend(v)
        if len(judges) >= 2:
            add_intersection_view(units, judges[0], judges[1])
            r3 = analyze_view(units, "intersection", items, merge_kwargs, args.bootstrap,
                              args.bootstrap_descriptive, args.seed, headline_only=True)
            sens["intersection"] = r3
            for k, v in flat_tables("intersection", r3).items():
                tables[k].extend(v)
            results["judge_agreement"] = judge_agreement(units, judges[0], judges[1], items,
                                                         args.bootstrap_descriptive, args.seed)
        else:
            results["judge_agreement"] = {"note": "only one judge available; kappa not computed"}
        # Escalation sensitivity: lens identity = role (as a platform panel with repeated roles).
        alt = dict(merge_kwargs, lens_identity="role" if args.lens_identity == "tag" else "tag")
        r4 = analyze_view(units, primary, items, alt, args.bootstrap, args.bootstrap_descriptive, args.seed)
        sens[f"lens_identity:{alt['lens_identity']}"] = {
            "severity_recall": r4["severity_recall"], "discrimination_auroc": r4["discrimination_auroc"],
            "false_alarms_clean": r4["false_alarms_clean"],
        }
        results["sensitivity"] = sens

    out_dir.mkdir(parents=True, exist_ok=True)
    common.atomic_write_json(out_dir / "benchmark_results.json", rnd(results, 6))
    for name, rows in tables.items():
        write_csv(out_dir / f"{name}.csv", rows)
    cost_rows = [{"model": g, "condition": c, **v} for g, cs in results["cost_by_condition"].items()
                 for c, v in cs.items()]
    write_csv(out_dir / "cost_by_condition.csv", cost_rows)
    write_csv(out_dir / "call_stats.csv", [
        {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in s.items()}
        for s in results["call_stats"].values()])
    print(f"units: {len(units)} complete ({results['data']['n_units_by_model']}); judged: "
          f"{results['data']['n_units_judged']}; results -> {out_dir / 'benchmark_results.json'}")
    if judges and "primary" in results:
        for g, conds in results["primary"]["recall"].items():
            line = ", ".join(f"{c}={v['estimate']:.3f}" if v["estimate"] is not None else f"{c}=NA"
                             for c, v in conds.items())
            print(f"  recall [{g}] {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
