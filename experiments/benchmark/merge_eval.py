"""Offline evaluation of critique-merging methods on benchmark panels (X2).

Ground truth (ANALYSIS_PLAN.md, "Merge evaluation"): within one panel instance
(the C5 calls rig1, adv, bio of one model x item x repeat), two critiques from
different lenses are duplicates if the judge mapped both to the same planted
flaw and non-duplicates if they map to different planted flaws; critiques
mapped to ``none`` are excluded from pair labels.

Methods: lexical Jaccard (legacy), normalised TF-IDF cosine, sentence-embedding
cosine (only if sentence-transformers and a locally cached model are
available; otherwise skipped and reported), and LLM adjudication
(``--adjudicator-model``; one cached call per panel instance, plus one retry
when its answer is not a valid partition). Grouping uses the platform's
order-independent average-linkage clustering
(``backend.orchestrator.similarity``) with the platform default
``same_lens_merge=False`` (two critiques of one lens never merge; the
reviewer prompts require one critique per distinct problem); LLM groups are
split the same way. The ground-truth pairs are cross-lens only, so the
selected threshold is validated for cross-lens grouping.

LLM adjudication records are reused only if their request signature matches
(adjudicator, effort, adjudicator system prompt hash, Codex tools-off
arguments and AGENTS.md hashes, parser version); records made before the
signature existed are reused only if their stored effort and arguments match.
Groups are always re-derived from the cached raw answer with the current
parser. Instances whose answer needed a repair (indices left out, or a 1-based
answer shifted) are counted (``repaired_instances``) and the LLM metrics are
also reported without them; invalid partitions are failures (no merging).

Metrics: pairwise precision / recall / F1 (pooled counts) and adjusted Rand
index over flaw-matched critiques; thresholds by leave-one-scenario-out CV
(maximise F1 on the other scenarios, evaluate on the held-out one); escalation
validity. Default = best held-out F1, ties broken by cost
(jaccard < tfidf < embedding < llm). Output: ``results/merge_eval.json``.

Usage:
    python experiments/benchmark/merge_eval.py --judge gpt-5.6-sol \
        [--adjudicator-model gpt-5.6-sol --adjudicator-effort medium]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import analyze  # noqa: E402
from backend.models import WorkspaceContext  # noqa: E402
from backend.orchestrator import similarity as sim  # noqa: E402

PANEL_CALLS = common.CONDITIONS["C5"]
COST_ORDER = ("jaccard", "tfidf", "embedding", "llm")
THRESHOLD_GRID = tuple(round(0.05 + 0.025 * i, 3) for i in range(37))  # 0.05 .. 0.95


@dataclass
class Instance:
    model: str
    item: str
    scenario: str
    repeat: int
    flawed: bool
    texts: list[str]
    lenses: list[str]
    labels: list[str]          # flaw id or "none"

    @property
    def key(self) -> str:
        return f"{self.model}/{self.item}/r{self.repeat}"


def build_instances(units, view: str, items: dict[str, common.Item]) -> list[Instance]:
    out = []
    for u in units:
        if view not in u.labels:
            continue
        texts, lenses, labels = [], [], []
        for tag in PANEL_CALLS:
            for j, c in enumerate(u.critiques(tag)):
                texts.append(c.get("description") or "")
                lenses.append(common.CALL_SPECS[tag][0])
                labels.append(u.labels[view].get((tag, j), ("none", None))[0])
        it = items[u.item]
        out.append(Instance(u.model, u.item, it.scenario, u.repeat, it.is_flawed, texts, lenses, labels))
    return out


def gt_pairs(inst: Instance) -> dict[tuple[int, int], bool]:
    """Labelled cross-lens pairs: True = duplicate (same flaw), False = different flaws."""
    out = {}
    n = len(inst.texts)
    for i in range(n):
        if inst.labels[i] == "none":
            continue
        for k in range(i + 1, n):
            if inst.labels[k] == "none" or inst.lenses[i] == inst.lenses[k]:
                continue
            out[(i, k)] = inst.labels[i] == inst.labels[k]
    return out


def group_index(groups: Sequence[Sequence[int]], n: int) -> list[int]:
    g = [-1] * n
    for gi, members in enumerate(groups):
        for m in members:
            g[m] = gi
    for i in range(n):
        if g[i] < 0:
            g[i] = len(groups) + i
    return g


def pair_counts(inst: Instance, groups: Sequence[Sequence[int]]) -> Counter:
    gi = group_index(groups, len(inst.texts))
    c = Counter()
    for (i, k), dup in gt_pairs(inst).items():
        pred = gi[i] == gi[k]
        if pred and dup:
            c["tp"] += 1
        elif pred and not dup:
            c["fp"] += 1
        elif dup:
            c["fn"] += 1
        else:
            c["tn"] += 1
    return c


def prf(c: Counter) -> dict:
    tp, fp, fn = c.get("tp", 0), c.get("fp", 0), c.get("fn", 0)
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    if p and r:
        f = 2 * p * r / (p + r)
    elif r is not None and tp == 0:
        f = 0.0  # positives exist but none recovered (precision undefined or 0)
    else:
        f = None
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn, "tn": c.get("tn", 0)}


def _comb2(x: int) -> float:
    return x * (x - 1) / 2.0


def adjusted_rand_index(a: Sequence[Any], b: Sequence[Any]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    cont = Counter(zip(a, b))
    sa, sb = Counter(a), Counter(b)
    index = sum(_comb2(v) for v in cont.values())
    sum_a = sum(_comb2(v) for v in sa.values())
    sum_b = sum(_comb2(v) for v in sb.values())
    expected = sum_a * sum_b / _comb2(n)
    max_index = (sum_a + sum_b) / 2.0
    if max_index == expected:
        return 1.0 if list(cont.values()) and all(
            sa[x] == v and sb[y] == v for (x, y), v in cont.items()) else 0.0
    return (index - expected) / (max_index - expected)


def ari_on_matched(inst: Instance, groups: Sequence[Sequence[int]]) -> Optional[float]:
    gi = group_index(groups, len(inst.texts))
    idx = [i for i, lab in enumerate(inst.labels) if lab != "none"]
    if len(idx) < 2:
        return None
    return adjusted_rand_index([inst.labels[i] for i in idx], [gi[i] for i in idx])


def escalation_stats(insts_groups: list[tuple[Instance, list[list[int]]]], consensus_threshold: int = 2) -> dict:
    multi = Counter()
    single = Counter()
    flaw_multi_lens = 0
    flaw_escalated = 0
    for inst, groups in insts_groups:
        for g in groups:
            lens_set = {inst.lenses[i] for i in g}
            labs = Counter(inst.labels[i] for i in g)
            top, top_n = labs.most_common(1)[0]
            any_flaw = any(lab != "none" for lab in labs)
            majority = top != "none" and top_n * 2 >= len(g)
            bucket = multi if len(lens_set) >= consensus_threshold else single
            bucket["groups"] += 1
            bucket["any_flaw"] += int(any_flaw)
            bucket["majority_flaw"] += int(majority)
        if not inst.flawed:
            continue
        by_flaw = defaultdict(set)
        for i, lab in enumerate(inst.labels):
            if lab != "none":
                by_flaw[lab].add(inst.lenses[i])
        for f, ls in by_flaw.items():
            if len(ls) < 2:
                continue
            flaw_multi_lens += 1
            if any(len({inst.lenses[i] for i in g if inst.labels[i] == f}) >= 2 for g in groups):
                flaw_escalated += 1

    def share(c: Counter, k: str) -> Optional[float]:
        return c[k] / c["groups"] if c["groups"] else None

    return {
        "multi_lens_groups": {"n": multi["groups"], "share_any_flaw": share(multi, "any_flaw"),
                              "share_majority_flaw": share(multi, "majority_flaw")},
        "single_lens_groups": {"n": single["groups"], "share_any_flaw": share(single, "any_flaw"),
                               "share_majority_flaw": share(single, "majority_flaw")},
        "flaws_raised_by_2plus_lenses": flaw_multi_lens,
        "of_which_escalated": flaw_escalated,
        "escalation_recall": (flaw_escalated / flaw_multi_lens) if flaw_multi_lens else None,
    }


# ── matrix methods ──────────────────────────────────────────────────────


SAME_LENS_MERGE = common.DEFAULT_SAME_LENS_MERGE  # platform default (False)


def matrix_groups(inst: Instance, matrix, tau: float) -> list[list[int]]:
    keys = [(t, inst.lenses[i]) for i, t in enumerate(inst.texts)]
    return sim.average_linkage_groups(matrix, tau, keys=keys, lenses=inst.lenses,
                                      same_lens_merge=SAME_LENS_MERGE)


def best_threshold(curve: dict[float, Counter], default: Optional[float]) -> float:
    def key(t):
        f = prf(curve[t])["f1"]
        return (f if f is not None else -1.0, -abs(t - (default if default is not None else 0.5)))
    return max(curve, key=key)


def evaluate_matrix_method(instances: list[Instance], method: str, *, encoder: Any = None,
                           embedding_model: Optional[str] = None,
                           grid: Sequence[float] = THRESHOLD_GRID) -> dict:
    # Reference ("default_threshold" in the output): the method's threshold
    # before this calibration (jaccard 0.5, tfidf 0.3, embedding 0.7), so a
    # re-run reproduces the evaluation independently of the platform defaults
    # it has since set (similarity.DEFAULT_THRESHOLDS).
    default = sim.UNCALIBRATED_THRESHOLDS[method]
    grid = sorted(set(grid) | ({default} if default is not None else set()))
    groups_at: dict[str, dict[float, list[list[int]]]] = {}
    counts_at: dict[str, dict[float, Counter]] = {}
    for inst in instances:
        if len(inst.texts) == 0:
            groups_at[inst.key] = {t: [] for t in grid}
            counts_at[inst.key] = {t: Counter() for t in grid}
            continue
        m = sim.similarity_matrix(inst.texts, method, embedding_model=embedding_model,
                                  embedding_encoder=encoder)
        groups_at[inst.key] = {t: matrix_groups(inst, m, t) for t in grid}
        counts_at[inst.key] = {t: pair_counts(inst, groups_at[inst.key][t]) for t in grid}

    def curve_for(subset: list[Instance]) -> dict[float, Counter]:
        cur = {t: Counter() for t in grid}
        for inst in subset:
            for t in grid:
                cur[t].update(counts_at[inst.key][t])
        return cur

    scenarios = sorted({i.scenario for i in instances if i.flawed})
    held = Counter()
    folds = []
    chosen: dict[str, float] = {}
    for s in scenarios:
        train = [i for i in instances if i.scenario != s]
        test = [i for i in instances if i.scenario == s]
        train_curve = curve_for(train)
        n_train_pairs = sum(train_curve[grid[0]].values()) if grid else 0
        # No labelled training pairs (e.g. a one-scenario pilot): method default.
        tau = best_threshold(train_curve, default) if n_train_pairs else (
            default if default is not None else 0.5)
        c = Counter()
        for inst in test:
            c.update(counts_at[inst.key][tau])
            chosen[inst.key] = tau
        held.update(c)
        folds.append({"held_out": s, "threshold": tau, "n_train_pairs": n_train_pairs,
                      "threshold_source": "train_f1" if n_train_pairs else "method_default", **prf(c)})
    for inst in instances:  # clean-only scenarios: use the all-data threshold
        chosen.setdefault(inst.key, None)
    all_curve = curve_for(instances)
    tau_all = best_threshold(all_curve, default) if instances else default
    for k, v in chosen.items():
        if v is None:
            chosen[k] = tau_all
    aris = [a for inst in instances if inst.flawed
            for a in [ari_on_matched(inst, groups_at[inst.key][chosen[inst.key]])] if a is not None]
    aris_default = [a for inst in instances if inst.flawed and default is not None
                    for a in [ari_on_matched(inst, groups_at[inst.key][default])] if a is not None]
    esc = [(inst, groups_at[inst.key][chosen[inst.key]]) for inst in instances]
    return {
        "method": method,
        "default_threshold": default,
        "held_out": prf(held),
        "folds": folds,
        "ari_mean_held_out": analyze.mean(aris),
        "ari_n_instances": len(aris),
        "all_data_best_threshold": tau_all,
        "all_data_best": prf(all_curve[tau_all]) if instances else None,
        "at_default_threshold": prf(all_curve[default]) if default is not None and instances else None,
        "ari_mean_at_default": analyze.mean(aris_default),
        "curve": [{"threshold": t, **prf(all_curve[t])} for t in grid],
        "escalation_held_out_flawed": escalation_stats([e for e in esc if e[0].flawed]),
        "escalation_held_out_all_items": escalation_stats(esc),
    }


# ── LLM adjudication ────────────────────────────────────────────────────


class _CapturingAdapter:
    """Adapter proxy that keeps the last AdapterRunResult (tokens, errors)."""

    def __init__(self, inner):
        self.inner = inner
        self.last = None

    async def run(self, request):
        self.last = await self.inner.run(request)
        return self.last


def adjudication_path(data_dir: Path, adjudicator: str, inst: Instance) -> Path:
    return (Path(data_dir) / "merge_eval" / "llm" / common.model_dirname(adjudicator)
            / common.model_dirname(inst.model) / inst.item / f"r{inst.repeat}.json")


#: Bump when the interpretation of a cached adjudicator answer changes.
ADJUDICATION_PARSER_VERSION = 2


def adjudication_signature(adjudicator: str, effort: str, cli_extra_args: Optional[list],
                           codex_home: Optional[str] = None) -> dict:
    provider, _ = common.resolve_provider(adjudicator)
    return {
        "adjudicator": adjudicator,
        "effort": effort,
        "adjudicator_system_prompt_sha256": common.sha256_text(sim.ADJUDICATOR_SYSTEM_PROMPT),
        "cli_extra_args": cli_extra_args,
        **common.codex_env_signature(provider, codex_home),
    }


def adjudication_cache_ok(rec: dict, signature: dict) -> bool:
    """A cached adjudication may be reused only under the same request."""
    req = rec.get("request")
    if isinstance(req, dict):
        return common.signature_matches(req, signature)
    # Records made before the signature existed: compare what they stored.
    return (rec.get("effort") == signature["effort"]
            and rec.get("cli_extra_args") == signature["cli_extra_args"]
            and rec.get("key", {}).get("adjudicator") == signature["adjudicator"])


def regroup(rec: dict, n: int) -> tuple[list[list[int]], list[str], bool, bool]:
    """Re-derive ``(groups, issues, success, repaired)`` from a cached raw
    answer with the current (strict) parser."""
    if not rec.get("called") or n < 2:
        return [[i] for i in range(n)], [], True, False
    raw = rec.get("raw_output") or ""
    if not rec.get("success") and not raw:
        return [[i] for i in range(n)], list(rec.get("issues") or []), False, False
    groups, issues, repaired, error = sim.LLMAdjudicator._interpret(raw, n)
    if groups is None:
        return [[i] for i in range(n)], issues + ([error] if error else []), False, False
    return groups, issues, True, repaired


async def adjudicate_all(instances: list[Instance], *, adjudicator: str, effort: str, data_dir: Path,
                         adapter_factory: Optional[Callable[[str, str], Any]] = None,
                         timeout: float = 900.0, concurrency: int = 2,
                         workspace_root: Optional[Path] = None) -> dict[str, dict]:
    factory = adapter_factory or (lambda m, e: common.make_adapter(m, e))  # web search disabled
    sem = asyncio.Semaphore(max(1, concurrency))
    out: dict[str, dict] = {}
    signature = adjudication_signature(
        adjudicator, effort,
        list(common.CODEX_NO_WEB_ARGS) if adapter_factory is None else None)

    async def one(inst: Instance):
        path = adjudication_path(data_dir, adjudicator, inst)
        texts_sha = common.sha256_text("\n".join(inst.texts))
        if path.exists():
            rec = common.read_json(path)
            if (rec.get("texts_sha256") == texts_sha and adjudication_cache_ok(rec, signature)):
                groups, issues, ok, repaired = regroup(rec, len(inst.texts))
                if ok:
                    out[inst.key] = {**rec, "groups": groups, "issues": issues,
                                     "success": True, "repaired": repaired,
                                     "regrouped_with_parser": ADJUDICATION_PARSER_VERSION}
                    return
        async with sem:
            proxy = _CapturingAdapter(factory(adjudicator, effort))
            ws = common.make_workspace(workspace_root)
            adj = sim.LLMAdjudicator(proxy, timeout_seconds=timeout,
                                     workspace_context=WorkspaceContext(workspace_path=str(ws)))
            started = common.utc_now()
            t0 = time.monotonic()
            captured = common.begin_capture()
            try:
                res = await adj.group(inst.texts)
            finally:
                common.remove_workspace(ws)
            last = proxy.last
            rec = {
                "schema": "miw-benchmark-adjudication/2",
                "key": {"adjudicator": adjudicator, "model": inst.model, "item": inst.item,
                        "repeat": inst.repeat},
                "request": {**signature, "parser_version": ADJUDICATION_PARSER_VERSION},
                "effort": effort,
                "cli_extra_args": list(common.CODEX_NO_WEB_ARGS) if adapter_factory is None else None,
                "texts_sha256": texts_sha,
                "n_texts": len(inst.texts),
                "called": res.called,
                "n_calls": res.n_calls,
                "success": res.success if res.called else True,
                "repaired": res.repaired,
                "error": res.error,
                "issues": res.issues,
                "groups": res.groups,
                "raw_output": res.raw_output,
                "tokens": common.usage_tokens(last) if last is not None else None,
                "tool_items": common.extract_tool_items(captured),
                "started_utc": started,
                "wall_seconds": round(time.monotonic() - t0, 3),
            }
            common.atomic_write_json(path, rec)
            out[inst.key] = rec

    await asyncio.gather(*(one(i) for i in instances))
    return out


def evaluate_llm(instances: list[Instance], adjudications: dict[str, dict]) -> dict:
    total = Counter()
    total_clean = Counter()   # excluding repaired instances
    aris = []
    esc = []
    failures = []
    repaired = []
    splits = 0
    tokens = Counter()
    efforts: Counter = Counter()
    prompt_hashes: Counter = Counter()
    for inst in instances:
        rec = adjudications.get(inst.key)
        if rec is None:
            continue
        if not rec.get("success"):
            failures.append(inst.key)
        is_repaired = bool(rec.get("repaired")) or (bool(rec.get("issues")) and rec.get("success"))
        if is_repaired:
            repaired.append(inst.key)
        groups = rec.get("groups") or [[i] for i in range(len(inst.texts))]
        if not SAME_LENS_MERGE:
            from backend.orchestrator.consensus import split_same_lens_groups
            groups, n_split = split_same_lens_groups(groups, inst.lenses)
            splits += n_split
        counts = pair_counts(inst, groups)
        total.update(counts)
        if not is_repaired:
            total_clean.update(counts)
        if inst.flawed:
            a = ari_on_matched(inst, groups)
            if a is not None:
                aris.append(a)
        esc.append((inst, groups))
        for k, v in (rec.get("tokens") or {}).items():
            if isinstance(v, int):
                tokens[k] += v
        efforts[str(rec.get("effort"))] += 1
        req = rec.get("request") or {}
        prompt_hashes[str(req.get("adjudicator_system_prompt_sha256", "unrecorded"))] += 1
    return {
        "method": "llm",
        "held_out": prf(total),  # no threshold to tune: every instance is held out
        "held_out_excluding_repaired": prf(total_clean),
        "ari_mean_held_out": analyze.mean(aris),
        "ari_n_instances": len(aris),
        "n_instances": sum(1 for i in instances if i.key in adjudications),
        "failed_instances": failures,
        "repaired_instances": repaired,
        "same_lens_splits": splits,
        "efforts_used": dict(efforts),
        "adjudicator_prompt_sha256_used": dict(prompt_hashes),
        "tokens_total": dict(tokens),
        "escalation_held_out_flawed": escalation_stats([e for e in esc if e[0].flawed]),
        "escalation_held_out_all_items": escalation_stats(esc),
    }


def embedding_encoder_or_reason(model_name: str) -> tuple[Any, Optional[str]]:
    if not sim.embedding_backend_available():
        return None, "sentence-transformers is not installed"
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # never download
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        enc = sim._load_sentence_transformer(model_name)
    except Exception as exc:  # noqa: BLE001
        return None, f"model {model_name!r} not available locally: {exc}"
    return enc, None


def choose_default(methods: dict[str, dict]) -> dict:
    scored = []
    for name in COST_ORDER:
        m = methods.get(name)
        if not m or m.get("skipped"):
            continue
        f = (m.get("held_out") or {}).get("f1")
        if f is None:
            continue
        scored.append((round(f, 3), -COST_ORDER.index(name), name))
    if not scored:
        return {"method": None, "reason": "no method has a held-out F1 (no labelled pairs yet)"}
    scored.sort(reverse=True)
    best = scored[0][2]
    m = methods[best]
    return {"method": best, "threshold": m.get("all_data_best_threshold"),
            "held_out_f1": m["held_out"]["f1"],
            "rule": "best held-out F1 (3 d.p.), ties broken by cost jaccard<tfidf<embedding<llm"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--judge", default="gpt-5.6-sol", help="judge whose labels define ground truth")
    ap.add_argument("--data-dir", default=str(common.DEFAULT_DATA_DIR))
    ap.add_argument("--out", default=str(common.RESULTS_DIR / "merge_eval.json"))
    ap.add_argument("--methods", default="jaccard,tfidf,embedding,llm")
    ap.add_argument("--adjudicator-model", default=None, help="model for LLM adjudication (llm method)")
    ap.add_argument("--adjudicator-effort", default="medium")
    ap.add_argument("--adjudicator-concurrency", type=int, default=2)
    ap.add_argument("--embedding-model", default=sim.DEFAULT_EMBEDDING_MODEL)
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    common.install_cli_capture()
    items = common.load_items()
    units, meta = analyze.load_units(data_dir, [args.judge], items)
    instances = build_instances(units, args.judge, items)
    n_pairs = Counter()
    for inst in instances:
        for dup in gt_pairs(inst).values():
            n_pairs["duplicate" if dup else "non_duplicate"] += 1
    methods_req = common.parse_csv_arg(args.methods)
    methods: dict[str, dict] = {}
    for m in ("jaccard", "tfidf"):
        if m in methods_req:
            methods[m] = evaluate_matrix_method(instances, m)
    if "embedding" in methods_req:
        enc, reason = embedding_encoder_or_reason(args.embedding_model)
        if enc is None:
            methods["embedding"] = {"method": "embedding", "skipped": True, "reason": reason}
        else:
            methods["embedding"] = evaluate_matrix_method(instances, "embedding", encoder=enc,
                                                          embedding_model=args.embedding_model)
    if "llm" in methods_req:
        if not args.adjudicator_model:
            methods["llm"] = {"method": "llm", "skipped": True, "reason": "no --adjudicator-model given"}
        else:
            adjs = asyncio.run(adjudicate_all(
                instances, adjudicator=args.adjudicator_model, effort=args.adjudicator_effort,
                data_dir=data_dir, concurrency=args.adjudicator_concurrency))
            methods["llm"] = evaluate_llm(instances, adjs)
            methods["llm"]["adjudicator_model"] = args.adjudicator_model
            # Effort(s) of the records actually used (cache hits keep theirs).
            used = methods["llm"].get("efforts_used") or {}
            methods["llm"]["adjudicator_effort"] = (
                next(iter(used)) if len(used) == 1 else args.adjudicator_effort)
            if len(used) > 1:
                print(f"  WARNING: LLM adjudications mix efforts {used}", file=sys.stderr)
    result = {
        "schema": "miw-benchmark-merge-eval/1",
        "generated_utc": common.utc_now(),
        "judge": args.judge,
        "panel_calls": list(PANEL_CALLS),
        "n_instances": len(instances),
        "n_instances_flawed": sum(1 for i in instances if i.flawed),
        "n_scenarios_flawed": len({i.scenario for i in instances if i.flawed}),
        "n_critiques": sum(len(i.texts) for i in instances),
        "n_labelled_pairs": dict(n_pairs),
        "threshold_grid": list(THRESHOLD_GRID),
        "same_lens_merge": SAME_LENS_MERGE,
        "methods": methods,
        "recommended_default": choose_default(methods),
        "notes": [
            "pair labels: cross-lens pairs of flaw-matched critiques only; ARI over flaw-matched critiques "
            "(includes same-lens pairs)",
            "held-out metrics pool counts over leave-one-scenario-out folds",
        ],
    }
    common.atomic_write_json(Path(args.out), analyze.rnd(result, 6))
    print(f"{len(instances)} panel instances, labelled pairs {dict(n_pairs)}; -> {args.out}")
    for name, m in methods.items():
        if m.get("skipped"):
            print(f"  {name}: skipped ({m['reason']})")
        else:
            ho = m.get("held_out") or {}
            print(f"  {name}: held-out P={ho.get('precision')} R={ho.get('recall')} F1={ho.get('f1')}")
    print(f"  recommended: {result['recommended_default']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
