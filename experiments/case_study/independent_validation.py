#!/usr/bin/env python
"""Independent validation of the Geneformer attention case study (MI-Workbench paper).

Uses ONLY the files of the case-study kit built by ``build_kit.py``. Question:
    Does Geneformer attention encode TF->target regulation beyond co-expression,
    expression-rank proximity and hub degree?

Analyses (keys in results/independent_validation.json):
  a_asymmetry          corr(A_ij, A_ji), relative asymmetry, >=2x fraction; per layer and per head
  b_directionality     paired A[TF,target] vs A[target,TF] on reference edges (Wilcoxon, BH over layers)
  c_edge_recovery      AUROC/AUPRC of attention vs baselines, TF-bootstrap CIs
  d_residualised       attention residualised on co-expression, expression, rank distance
  e_rank_proximity     Spearman(attention, rank distance) and Spearman(attention, co-expression)
  f_permutation_null   degree-preserving (curveball) rewiring null, 1,000 permutations
  g_multiple_testing   number of tests and correction per family
  verdicts             pre-specified decision rules applied to the four statements

Orientation: attention[l, i, j] = mean attention from query gene i to key gene j.
    "q=TF"      score A[TF, target]  (TF token as query attends to target token as key)
    "q=target"  score A[target, TF]  (target token as query attends to TF token as key)
    "sym"       (A[TF, target] + A[target, TF]) / 2   (primary, pre-specified)

Run:
    /Users/ihorkendiukhov/anaconda3/bin/python \
        automation/mi-workbench/experiments/case_study/independent_validation.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

SCRIPT_PATH = Path(__file__).resolve()
WORKBENCH = SCRIPT_PATH.parents[2]
DEFAULT_KIT = WORKBENCH / "experiments_data" / "case_study_kit"
DEFAULT_RESULTS = SCRIPT_PATH.parent / "results"

SEED = 20261001
MIN_COPRESENCE = 50
N_BOOT = 1000          # TF-level bootstrap for AUROC / AUPRC
N_BOOT_MEDIAN = 2000   # edge-level bootstrap for median log2 ratios
N_PERM = 1000          # degree-preserving permutations
CURVEBALL_BURNIN = 20000
CURVEBALL_THIN = 1000
VARIANTS = ("sym", "qTF", "qTarget")

# Pre-specified decision thresholds for the verdicts (fixed before looking at results)
SYM_RHO_OK, SYM_RELASYM_OK = 0.90, 0.10     # "approximately symmetric"
SYM_RHO_BAD, SYM_RELASYM_BAD = 0.70, 0.25   # "not symmetric"
RANK_RHO_LARGE, RANK_RHO_PARTIAL = 0.50, 0.30
ALPHA = 0.05


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def r4(x):
    if x is None:
        return None
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(f"{float(x):.7g}")
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, dict):
        return {k: r4(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [r4(v) for v in x]
    return x


def bh(p):
    p = np.asarray(p, dtype=float)
    return multipletests(p, method="fdr_bh")[1]


def holm(p):
    return multipletests(np.asarray(p, dtype=float), method="holm")[1]


def summarise(v) -> dict:
    v = np.asarray(v, dtype=float)
    return {"min": v.min(), "q25": np.quantile(v, 0.25), "median": np.median(v),
            "q75": np.quantile(v, 0.75), "max": v.max(), "mean": v.mean()}


def spearman_fast(x, y) -> float:
    return float(np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1])


# ------------------------------------------------------------------ ranking metrics
class RankPrep:
    """Sort a score vector once (descending) so that weighted AUROC/AUPRC for any
    per-item weight vector (TF bootstrap multiplicities) costs O(n)."""

    def __init__(self, score: np.ndarray):
        self.order = np.argsort(-score, kind="mergesort")
        s = score[self.order]
        new = np.r_[True, s[1:] != s[:-1]]
        self.gid = np.cumsum(new) - 1
        self.ng = int(self.gid[-1]) + 1

    def metrics(self, y: np.ndarray, w: np.ndarray | None = None) -> tuple[float, float]:
        if getattr(self, "_y", None) is not y:
            self._y, self._yo = y, y[self.order].astype(float)
        yo = self._yo
        wo = np.ones_like(yo) if w is None else w[self.order].astype(float)
        P = np.bincount(self.gid, wo * yo, self.ng)
        N = np.bincount(self.gid, wo * (1 - yo), self.ng)
        TP, FP = np.cumsum(P), np.cumsum(N)
        Wp, Wn = TP[-1], FP[-1]
        if Wp == 0 or Wn == 0:
            return np.nan, np.nan
        auc = float(np.sum(P * ((Wn - FP) + 0.5 * N)) / (Wp * Wn))
        denom = TP + FP
        prec = np.divide(TP, denom, out=np.zeros_like(TP), where=denom > 0)
        ap = float(np.sum(P / Wp * prec))
        return auc, ap


class TFDecomp:
    """Exact TF-bootstrap AUROC/AUPRC. With TF multiplicities m (B x rows):
        AUROC(m) = m' U m / ((m . npos)(m . nneg)),  U[r,s] = sum over positives p in row r of
                   #(negatives in row s scored below p) + 0.5 #(ties)
        AUPRC(m) = sum_p m[row p] TP_p / (TP_p + FP_p) / (m . npos),
                   TP_p = sum_s m_s #(positives in row s scored >= s_p), FP_p likewise for negatives.
    Identical to weighted AUROC / average precision (sklearn tie convention), verified in tests."""

    def __init__(self, score: np.ndarray, y: np.ndarray, rl: np.ndarray, n_rows: int):
        pos = np.where(y == 1)[0]
        sp, self.prow = score[pos], rl[pos]
        self.Kpos = np.zeros((pos.size, n_rows))
        self.Kneg = np.zeros((pos.size, n_rows))
        self.U = np.zeros((n_rows, n_rows))
        order = np.argsort(rl, kind="stable")
        bounds = np.searchsorted(rl[order], np.arange(n_rows + 1))
        for r in range(n_rows):
            idx = order[bounds[r]:bounds[r + 1]]
            sc, ys = score[idx], y[idx]
            neg, posr = np.sort(sc[ys == 0]), np.sort(sc[ys == 1])
            lo = np.searchsorted(neg, sp, "left")
            hi = np.searchsorted(neg, sp, "right")
            self.Kneg[:, r] = neg.size - lo
            self.Kpos[:, r] = posr.size - np.searchsorted(posr, sp, "left")
            np.add.at(self.U[:, r], self.prow, lo + 0.5 * (hi - lo))
        self.npos_row = np.bincount(self.prow, minlength=n_rows).astype(float)
        self.nneg_row = np.bincount(rl[y == 0], minlength=n_rows).astype(float)

    def metrics(self, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        M = np.atleast_2d(M).astype(float)
        Wp, Wn = M @ self.npos_row, M @ self.nneg_row
        auc = np.einsum("br,rs,bs->b", M, self.U, M) / (Wp * Wn)
        TP, FP = M @ self.Kpos.T, M @ self.Kneg.T
        wp = M[:, self.prow]
        prec = np.divide(TP, TP + FP, out=np.zeros_like(TP), where=(TP + FP) > 0)
        ap = (wp * prec).sum(axis=1) / Wp
        bad = (Wp == 0) | (Wn == 0)
        auc[bad], ap[bad] = np.nan, np.nan
        return auc, ap


def auc_from_ranks(ranks: np.ndarray, pos_idx: np.ndarray, n: int) -> np.ndarray:
    """AUROC for one or many positive sets (rows of pos_idx) given average ranks."""
    npos = pos_idx.shape[-1]
    nneg = n - npos
    s = ranks[pos_idx].sum(axis=-1)
    return (s - npos * (npos + 1) / 2.0) / (npos * nneg)


# ------------------------------------------------------------------------- load
def load_kit(kit: Path) -> dict:
    d = {}
    d["genes"] = pd.read_csv(kit / "genes.tsv", sep="\t")
    d["cells"] = pd.read_csv(kit / "cells.tsv", sep="\t")
    d["A"] = np.load(kit / "attention_layer_mean.npy")
    heads_path = kit / "attention_heads_f32.npy"
    d["heads_file"] = heads_path.name if heads_path.exists() else "attention_heads.npy"
    d["H"] = np.load(kit / d["heads_file"], mmap_mode="r")
    d["C"] = np.load(kit / "copresence_counts.npy")
    d["RD"] = np.load(kit / "rank_distance_mean.npy")
    d["P"] = np.load(kit / "coexpr_pearson.npy").astype(np.float64)
    d["S"] = np.load(kit / "coexpr_spearman.npy").astype(np.float64)
    d["trrust"] = pd.read_csv(kit / "trrust_edges.tsv", sep="\t")
    d["dorothea"] = pd.read_csv(kit / "dorothea_abc_edges.tsv", sep="\t")
    d["manifest"] = json.loads((kit / "kit_manifest.json").read_text())
    return d


# ----------------------------------------------------------------- (a) asymmetry
def asym_stats(a: np.ndarray, b: np.ndarray) -> dict:
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    rel = np.abs(a - b) / (a + b)
    ratio2 = np.where(lo > 0, hi / np.where(lo > 0, lo, 1) >= 2.0, hi > 0)
    return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": spearman_fast(a, b),
            "median_rel_asym": float(np.median(rel)), "mean_rel_asym": float(np.mean(rel)),
            "frac_ge_2x": float(np.mean(ratio2))}


def analysis_a(d: dict, iu: tuple) -> dict:
    A, H = d["A"], d["H"]
    L, nh = H.shape[0], H.shape[1]
    per_layer = []
    for l in range(L):
        s = asym_stats(A[l][iu], A[l].T[iu])
        s["layer"] = l
        per_layer.append(s)
    marg = []
    for l in range(L):
        add, I = marginal_parts(A[l])
        X = add + I
        off = ~np.eye(A.shape[1], dtype=bool)
        r2_add = 1 - np.nanvar(I[off]) / np.nanvar(X[off])
        marg.append({"layer": l, "r2_additive_row_col_on_log10_attention": float(r2_add),
                     "spearman_interaction_ij_vs_ji": spearman_fast(I[iu], I.T[iu]),
                     "pearson_interaction_ij_vs_ji": float(np.corrcoef(I[iu], I.T[iu])[0, 1]),
                     "spearman_row_effect_vs_col_effect_per_gene": spearman_fast(np.nanmean(X, 1), np.nanmean(X, 0))})
    per_head = {k: np.zeros((L, nh)) for k in ("pearson", "spearman", "median_rel_asym", "frac_ge_2x")}
    for l in range(L):
        Hl = np.asarray(H[l])
        for h in range(nh):
            s = asym_stats(Hl[h][iu], Hl[h].T[iu])
            for k in per_head:
                per_head[k][l, h] = s[k]
        log(f"  (a) heads layer {l} done")
    # co-presence-matched reference: self-consistency of the symmetric co-expression is 1
    # by construction, so asymmetry is judged against the pre-specified thresholds.
    return {
        "n_pairs": int(iu[0].size),
        "pair_filter": f"i<j, copresence >= {MIN_COPRESENCE}",
        "per_layer_head_mean": per_layer,
        "per_head_summary": {k: summarise(v.ravel()) for k, v in per_head.items()},
        "per_head": {k: v.tolist() for k, v in per_head.items()},
        "heads_file": d["heads_file"],
        "supplementary_marginal_decomposition_post_hoc": {
            "description": ("log10 attention (head-mean, diagonal excluded) = row effect (query) + column effect (key) "
                            "- grand mean + interaction. R2 = variance share of the additive part; symmetry of the "
                            "pair-specific interaction. Added after inspecting the main results."),
            "per_layer": marg},
    }


# ---------------------------------------------------------- candidate construction
def build_reference(d: dict, edges: pd.DataFrame, rows_mode: str) -> dict:
    """Candidates: ordered pairs (tf, g) with tf a reference TF in G, g != tf,
    copresence >= MIN_COPRESENCE. Positives: non-self reference edges among them."""
    nG = len(d["genes"])
    e = edges[~edges.is_self]
    all_tf = np.array(sorted(set(edges.tf_index)))           # TFs with >=1 edge in G (incl. self)
    if rows_mode == "all_tfs_in_G":
        # every reference TF (as per genes.tsv) present in G
        col = "is_tf" if "trrust" in d["_ref_name"] else "is_dorothea_tf"
        rows = np.where(d["genes"][col].values)[0]
    elif rows_mode == "tfs_with_edge":
        rows = np.array(sorted(set(e.tf_index)))
    else:
        raise ValueError(rows_mode)
    C = d["C"]
    rr, cc = np.meshgrid(rows, np.arange(nG), indexing="ij")
    rloc = np.repeat(np.arange(rows.size), nG).reshape(rows.size, nG)
    ok = (C[rr, cc] >= MIN_COPRESENCE) & (rr != cc)
    ri, ci, rl = rr[ok], cc[ok], rloc[ok]
    edge_set = set(zip(e.tf_index.values, e.target_index.values))
    y = np.array([(a, b) in edge_set for a, b in zip(ri, ci)], dtype=np.int8)
    n_edges_total = len(edge_set)
    row_set = set(rows.tolist())
    n_edges_rows = int(sum(1 for a, _ in edge_set if a in row_set))
    return {"rows": rows, "ri": ri, "ci": ci, "rl": rl, "y": y, "all_tf": all_tf,
            "n_edges_nonself_in_G": n_edges_total, "n_edges_on_rows": n_edges_rows,
            "n_pos": int(y.sum()), "n": int(y.size)}


def baseline_scores(d: dict, cand: dict, edges: pd.DataFrame) -> dict:
    g = d["genes"]
    ri, ci, y = cand["ri"], cand["ci"], cand["y"]
    e = edges[~edges.is_self]
    outdeg = np.bincount(e.tf_index.values, minlength=len(g)).astype(float)
    indeg = np.bincount(e.target_index.values, minlength=len(g)).astype(float)
    mexpr = g.mean_norm_expr.values
    rng = np.random.default_rng(SEED + 7)
    jitter = rng.random(y.size) * 1e-9          # break exact ties for degree scores deterministically
    return {
        "abs_pearson": np.abs(d["P"][ri, ci]),
        "abs_spearman": np.abs(d["S"][ri, ci]),
        "expr_product": mexpr[ri] * mexpr[ci],
        "neg_rank_distance": -d["RD"][ri, ci].astype(float),
        "degree_product_circular": outdeg[ri] * indeg[ci] + jitter,
        "target_indegree_circular": indeg[ci] + jitter,
        "degree_product_loo": (outdeg[ri] - y) * (indeg[ci] - y) + jitter,
        "target_indegree_loo": (indeg[ci] - y) + jitter,
    }


ATT_FLOOR = 1e-12   # guards log transforms; attention means are strictly positive in practice


def attention_scores(M: np.ndarray, ri, ci) -> dict:
    a = np.maximum(M[ri, ci].astype(np.float64), ATT_FLOOR)
    b = np.maximum(M[ci, ri].astype(np.float64), ATT_FLOOR)
    return {"sym": 0.5 * (a + b), "qTF": a, "qTarget": b}


def pair_scores(M: np.ndarray, ri, ci) -> dict:
    """Same three readings for an arbitrary (possibly negative, log-scale) pair matrix."""
    a, b = M[ri, ci].astype(np.float64), M[ci, ri].astype(np.float64)
    return {"sym": 0.5 * (a + b), "qTF": a, "qTarget": b}


def marginal_parts(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-way decomposition of log10 attention (diagonal excluded): additive part
    row_i + col_j - grand (query 'hubness' + key 'attractiveness') and pair-specific
    interaction residual."""
    X = np.log10(np.maximum(M.astype(np.float64), ATT_FLOOR))
    np.fill_diagonal(X, np.nan)
    r, c, g = np.nanmean(X, axis=1), np.nanmean(X, axis=0), np.nanmean(X)
    add = r[:, None] + c[None, :] - g
    return add, X - add


def boot_weights(n_rows: int, rl: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mult = rng.multinomial(n_rows, np.full(n_rows, 1.0 / n_rows), size=n_boot)   # B x rows
    return mult, rl


def eval_score(score, y, mult, rl, keep_boot=False) -> dict:
    prep = RankPrep(score)
    auc, ap = prep.metrics(y)
    dec = TFDecomp(score, y, rl, mult.shape[1])
    a1, p1 = dec.metrics(np.ones((1, mult.shape[1])))
    assert abs(a1[0] - auc) < 1e-9 and abs(p1[0] - ap) < 1e-9, (a1, auc, p1, ap)
    ba, bp = dec.metrics(mult)
    boots = np.column_stack([ba, bp])
    out = {"auroc": auc, "auprc": ap,
           "auroc_ci95": [np.nanquantile(boots[:, 0], 0.025), np.nanquantile(boots[:, 0], 0.975)],
           "auprc_ci95": [np.nanquantile(boots[:, 1], 0.025), np.nanquantile(boots[:, 1], 0.975)]}
    if keep_boot:
        out["_boot"] = boots
    return out


# ---------------------------------------------------------------- (f) curveball
def curveball_null(cand: dict, n_perm: int, seed: int) -> np.ndarray:
    """Degree-preserving rewiring of the positive edges inside the candidate set
    (bipartite TF x gene matrix, forbidden = non-candidate cells). Returns an array
    (n_perm, n_pos) of candidate indices that are positive in each permutation."""
    rng = np.random.default_rng(seed)
    rl, ci, y = cand["rl"], cand["ci"], cand["y"]
    n_rows = cand["rows"].size
    allowed = [set() for _ in range(n_rows)]
    index = {}
    for k, (r, c) in enumerate(zip(rl, ci)):
        allowed[r].add(int(c))
        index[(int(r), int(c))] = k
    sets = [set() for _ in range(n_rows)]
    for k in np.where(y == 1)[0]:
        sets[rl[k]].add(int(ci[k]))
    active = [r for r in range(n_rows) if sets[r]]

    act = np.array(active)
    buf = {"pairs": None, "i": 0}

    def trade():
        if buf["pairs"] is None or buf["i"] >= len(buf["pairs"]):
            buf["pairs"] = rng.integers(0, act.size, size=(100000, 2)); buf["i"] = 0
        ia, ib = buf["pairs"][buf["i"]]; buf["i"] += 1
        if ia == ib:
            return
        a, b = int(act[ia]), int(act[ib])
        Sa, Sb = sets[a], sets[b]
        a_only = [c for c in Sa - Sb if c in allowed[b]]
        b_only = [c for c in Sb - Sa if c in allowed[a]]
        if not a_only and not b_only:
            return
        pool = a_only + b_only
        rng.shuffle(pool)
        na = len(a_only)
        Sa.difference_update(a_only); Sb.difference_update(b_only)
        Sa.update(pool[:na]); Sb.update(pool[na:])

    for _ in range(CURVEBALL_BURNIN):
        trade()
    out = np.empty((n_perm, int(y.sum())), dtype=np.int64)
    deg0 = [len(s) for s in sets]
    for p in range(n_perm):
        for _ in range(CURVEBALL_THIN):
            trade()
        idx = [index[(r, c)] for r in active for c in sets[r]]
        out[p] = np.sort(idx)
    assert [len(s) for s in sets] == deg0
    # column degrees preserved
    col_obs = np.bincount(ci[y == 1], minlength=ci.max() + 1)
    col_perm = np.bincount(ci[out[-1]], minlength=ci.max() + 1)
    assert np.array_equal(col_obs, col_perm)
    return out


# -------------------------------------------------------------- (b) directionality
def analysis_b(d: dict, edges: pd.DataFrame, cand: dict, rng_seed: int) -> dict:
    A = d["A"]
    C = d["C"]
    e = edges[~edges.is_self]
    e = e[C[e.tf_index.values, e.target_index.values] >= MIN_COPRESENCE]
    tf, tg = e.tf_index.values, e.target_index.values
    edge_set = set(zip(tf, tg))
    # non-edge control: candidate TF->g pairs that are not reference edges in either direction
    ri, ci = cand["ri"], cand["ci"]
    ctrl = np.array([(a, b) not in edge_set and (b, a) not in edge_set for a, b in zip(ri, ci)])
    cr, cc = ri[ctrl], ci[ctrl]
    mrank = d["genes"].mean_rank_when_present.values
    rng = np.random.default_rng(rng_seed)
    per_layer, pw, pm = [], [], []
    for l in range(A.shape[0]):
        x = np.maximum(A[l][tf, tg].astype(np.float64), ATT_FLOOR)   # query=TF, key=target
        z = np.maximum(A[l][tg, tf].astype(np.float64), ATT_FLOOR)   # query=target, key=TF
        dlog = np.log2(x / z)
        w = stats.wilcoxon(x, z, zero_method="wilcox", alternative="two-sided")
        bi = rng.integers(0, dlog.size, size=(N_BOOT_MEDIAN, dlog.size))
        bmed = np.median(dlog[bi], axis=1)
        dctrl = np.log2(np.maximum(A[l][cr, cc].astype(np.float64), ATT_FLOOR)
                        / np.maximum(A[l][cc, cr].astype(np.float64), ATT_FLOOR))
        mw = stats.mannwhitneyu(dlog, dctrl, alternative="two-sided")
        rho_rank = spearman_fast(dlog, mrank[tf] - mrank[tg])
        per_layer.append({
            "layer": l, "n_edges": int(dlog.size),
            "median_log2_ratio_qTF_over_qTarget": float(np.median(dlog)),
            "median_log2_ratio_ci95": [float(np.quantile(bmed, 0.025)), float(np.quantile(bmed, 0.975))],
            "frac_edges_qTF_gt_qTarget": float(np.mean(dlog > 0)),
            "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
            "nonedge_control_n": int(dctrl.size),
            "nonedge_control_median_log2_ratio": float(np.median(dctrl)),
            "edges_vs_nonedge_mwu_p": float(mw.pvalue),
            "spearman_log2ratio_vs_mean_rank_difference_tf_minus_target": rho_rank,
        })
        pw.append(w.pvalue)
        pm.append(mw.pvalue)
    for rec, q1, q2 in zip(per_layer, bh(pw), bh(pm)):
        rec["wilcoxon_p_bh"] = float(q1)
        rec["edges_vs_nonedge_mwu_p_bh"] = float(q2)
    return {
        "orientation": ("log2 ratio = log2(A[TF,target] / A[target,TF]); A[i,j] = attention from query i "
                        "to key j. Positive: the TF token (query) attends more to the target token (key) "
                        "than the target attends to the TF. Negative: the target token (query) attends more to "
                        "the TF token (key), i.e. the 'target reads from its regulator' reading."),
        "edge_filter": f"non-self reference edges with copresence >= {MIN_COPRESENCE}",
        "per_layer": per_layer,
        "n_tests_wilcoxon": len(pw), "n_tests_mwu": len(pm), "correction": "BH across layers",
    }


# --------------------------------------------------------------- (d) residualise
def ols_resid(yv: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    Xc = np.column_stack([np.ones(len(yv)), X])
    beta, *_ = np.linalg.lstsq(Xc, yv, rcond=None)
    res = yv - Xc @ beta
    r2 = 1 - res.var() / yv.var()
    return res, float(r2), beta


def grouped_cv_logit(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> tuple[float, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = m.decision_function(sc.transform(X[te]))
    return float(roc_auc_score(y, oof)), float(average_precision_score(y, oof))


# -------------------------------------------------------------------- per reference
def run_reference(d: dict, name: str, edges: pd.DataFrame, rows_mode: str, do_heads: bool,
                  seed: int) -> dict:
    t0 = time.time()
    d["_ref_name"] = name
    cand = build_reference(d, edges, rows_mode)
    ri, ci, y, rl = cand["ri"], cand["ci"], cand["y"], cand["rl"]
    n, npos = cand["n"], cand["n_pos"]
    log(f"[{name}/{rows_mode}] candidates {n}, positives {npos}, TF rows {cand['rows'].size}")
    mult, _ = boot_weights(cand["rows"].size, rl, N_BOOT, seed)
    A = d["A"]
    L = A.shape[0]
    g = d["genes"]

    # ---- scores
    layer_scores = {v: [] for v in VARIANTS}
    for l in range(L):
        s = attention_scores(A[l], ri, ci)
        for v in VARIANTS:
            layer_scores[v].append(s[v])
    base = baseline_scores(d, cand, edges)

    # ---- (d) residualisation covariates
    r = d["P"][ri, ci]
    X = np.column_stack([np.abs(r), r, np.log(g.mean_norm_expr.values[ri]),
                         np.log(g.mean_norm_expr.values[ci]), d["RD"][ri, ci].astype(float)])
    cov_names = ["abs_pearson", "pearson", "log_mean_expr_tf", "log_mean_expr_gene", "mean_rank_distance"]
    resid_scores = {v: [] for v in VARIANTS}
    resid_log_scores = {v: [] for v in VARIANTS}
    r2 = {v: [] for v in VARIANTS}
    r2_log = {v: [] for v in VARIANTS}
    betas = {v: [] for v in VARIANTS}
    for v in VARIANTS:
        for l in range(L):
            res, rsq, beta = ols_resid(layer_scores[v][l], X)
            resid_scores[v].append(res); r2[v].append(rsq); betas[v].append(beta.tolist())
            res2, rsq2, _ = ols_resid(np.log10(layer_scores[v][l]), X)
            resid_log_scores[v].append(res2); r2_log[v].append(rsq2)

    # ---- (c) metrics with TF bootstrap
    boot_keep = {}
    per_layer = {v: [] for v in VARIANTS}
    for v in VARIANTS:
        for l in range(L):
            m = eval_score(layer_scores[v][l], y, mult, rl, keep_boot=True)
            boot_keep[(v, l)] = m.pop("_boot")
            m["layer"] = l
            per_layer[v].append(m)
    baselines = {}
    for k, s in base.items():
        m = eval_score(s, y, mult, rl, keep_boot=True)
        boot_keep[("base", k)] = m.pop("_boot")
        baselines[k] = m
    log(f"[{name}] layer/baseline bootstraps done ({time.time() - t0:.0f}s)")

    # paired bootstrap differences vs co-expression baselines
    diffs = {v: [] for v in VARIANTS}
    pdiff = []
    best_coexpr = max(["abs_pearson", "abs_spearman"], key=lambda k: baselines[k]["auroc"])
    for v in VARIANTS:
        for l in range(L):
            rec = {"layer": l}
            for bname in ("abs_pearson", "abs_spearman"):
                obs = per_layer[v][l]["auroc"] - baselines[bname]["auroc"]
                bd = boot_keep[(v, l)][:, 0] - boot_keep[("base", bname)][:, 0]
                p = min(1.0, 2 * min(np.mean(bd <= 0), np.mean(bd >= 0)) + 1.0 / N_BOOT)
                rec[f"delta_auroc_vs_{bname}"] = obs
                rec[f"delta_auroc_vs_{bname}_ci95"] = [np.quantile(bd, 0.025), np.quantile(bd, 0.975)]
                rec[f"delta_auroc_vs_{bname}_boot_p"] = p
            obs_ap = per_layer[v][l]["auprc"] - baselines[best_coexpr]["auprc"]
            bdap = boot_keep[(v, l)][:, 1] - boot_keep[("base", best_coexpr)][:, 1]
            rec[f"delta_auprc_vs_{best_coexpr}"] = obs_ap
            rec[f"delta_auprc_vs_{best_coexpr}_ci95"] = [np.quantile(bdap, 0.025), np.quantile(bdap, 0.975)]
            diffs[v].append(rec)
            pdiff.append(rec[f"delta_auroc_vs_{best_coexpr}_boot_p"])
    qd = bh(pdiff)
    k = 0
    for v in VARIANTS:
        for l in range(L):
            diffs[v][l][f"delta_auroc_vs_{best_coexpr}_boot_p_bh"] = float(qd[k]); k += 1

    # residual metrics
    resid = {v: [] for v in VARIANTS}
    for v in VARIANTS:
        for l in range(L):
            m = eval_score(resid_scores[v][l], y, mult, rl)
            mlog = RankPrep(resid_log_scores[v][l]).metrics(y)
            resid[v].append({"layer": l, "residual_auroc": m["auroc"], "residual_auroc_ci95": m["auroc_ci95"],
                             "residual_auprc": m["auprc"], "residual_auprc_ci95": m["auprc_ci95"],
                             "r2_covariates": r2[v][l], "ols_coefficients": dict(zip(["intercept"] + cov_names, betas[v][l])),
                             "log10_attention_residual_auroc": mlog[0], "log10_attention_residual_auprc": mlog[1],
                             "r2_covariates_log10_attention": r2_log[v][l]})
    log(f"[{name}] residual metrics done ({time.time() - t0:.0f}s)")

    # grouped-CV logistic: covariates vs covariates + log10 attention (sym), per layer
    groups = ri
    cov_only = grouped_cv_logit(X, y, groups, seed)
    logit = []
    for l in range(L):
        Xa = np.column_stack([X, np.log10(layer_scores["sym"][l])])
        auc_a, ap_a = grouped_cv_logit(Xa, y, groups, seed)
        logit.append({"layer": l, "auroc_cov_plus_attention": auc_a, "auprc_cov_plus_attention": ap_a,
                      "delta_auroc": auc_a - cov_only[0], "delta_auprc": ap_a - cov_only[1]})
    log(f"[{name}] grouped-CV logistic done ({time.time() - t0:.0f}s)")

    # ---- (f) degree-preserving permutation null
    perm = curveball_null(cand, N_PERM, seed + 11)
    log(f"[{name}] curveball null generated ({time.time() - t0:.0f}s)")
    pos_obs = np.where(y == 1)[0]

    # ---- per head point estimates + degree-null p (streamed, one head at a time)
    per_head = None
    if do_heads:
        H = d["H"]
        nh = H.shape[1]
        per_head = {f"{v}_{m}": np.zeros((L, nh)) for v in VARIANTS for m in ("auroc", "auprc")}
        hp = {v: np.zeros((L, nh)) for v in VARIANTS}
        hnull_mean = {v: np.zeros((L, nh)) for v in VARIANTS}
        hmax = np.full(N_PERM, -np.inf)
        for l in range(L):
            Hl = np.asarray(H[l])
            for h in range(nh):
                sc = attention_scores(Hl[h], ri, ci)
                for v in VARIANTS:
                    auc, apv = RankPrep(sc[v]).metrics(y)
                    per_head[f"{v}_auroc"][l, h] = auc
                    per_head[f"{v}_auprc"][l, h] = apv
                    rk = stats.rankdata(sc[v])
                    nul = auc_from_ranks(rk, perm, n)
                    hp[v][l, h] = (1 + np.sum(nul >= auc - 1e-12)) / (1 + N_PERM)
                    hnull_mean[v][l, h] = nul.mean()
                    hmax = np.maximum(hmax, nul)
        log(f"[{name}] per-head metrics done ({time.time() - t0:.0f}s)")

    def null_for(score=None, ranks=None):
        rk = stats.rankdata(score) if ranks is None else ranks
        obs = float(auc_from_ranks(rk, pos_obs, n))
        nul = auc_from_ranks(rk, perm, n)
        return obs, nul

    null = {"layers": {}, "residual_layers": {}, "baselines": {}}
    null_mats = {}
    for v in VARIANTS:
        recs, nulls = [], []
        for l in range(L):
            obs, nul = null_for(layer_scores[v][l])
            nulls.append(nul)
            recs.append({"layer": l, "auroc": obs, "null_mean": nul.mean(), "null_sd": nul.std(),
                         "null_q95": np.quantile(nul, 0.95),
                         "null_ci95": [np.quantile(nul, 0.025), np.quantile(nul, 0.975)],
                         "p_empirical": (1 + np.sum(nul >= obs)) / (1 + N_PERM),
                         "excess_over_null": obs - nul.mean()})
        q = bh([r["p_empirical"] for r in recs])
        for rrec, qq in zip(recs, q):
            rrec["p_bh_layers"] = float(qq)
        null["layers"][v] = recs
        null_mats[v] = np.vstack(nulls)                # L x perm
        rres = []
        for l in range(L):
            obs, nul = null_for(resid_scores[v][l])
            rres.append({"layer": l, "residual_auroc": obs, "null_mean": nul.mean(),
                         "null_ci95": [np.quantile(nul, 0.025), np.quantile(nul, 0.975)],
                         "p_empirical": (1 + np.sum(nul >= obs)) / (1 + N_PERM)})
        for rrec, qq in zip(rres, bh([r["p_empirical"] for r in rres])):
            rrec["p_bh_layers"] = float(qq)
        null["residual_layers"][v] = rres
    for kname, s in base.items():
        if kname.endswith("_loo"):
            null["baselines"][kname] = {"auroc": baselines[kname]["auroc"], "null_mean": None,
                                        "note": "score depends on the labels (leave-one-out), so a label-permutation null is not defined"}
            continue
        obs, nul = null_for(s)
        null["baselines"][kname] = {"auroc": obs, "null_mean": nul.mean(),
                                    "null_ci95": [np.quantile(nul, 0.025), np.quantile(nul, 0.975)],
                                    "p_empirical": (1 + np.sum(nul >= obs)) / (1 + N_PERM)}
    # post hoc: degree-matched comparison with the best co-expression baseline (same permutations)
    obs_co, nul_co = null_for(base[best_coexpr])
    dm, pdm = {v: [] for v in VARIANTS}, []
    for v in VARIANTS:
        for l in range(L):
            obs = null["layers"][v][l]["auroc"]
            nd = null_mats[v][l] - nul_co
            rec = {"layer": l, "delta_auroc_observed": obs - obs_co, "null_delta_mean": nd.mean(),
                   "null_delta_ci95": [np.quantile(nd, 0.025), np.quantile(nd, 0.975)],
                   "delta_minus_null_delta": (obs - obs_co) - nd.mean(),
                   "attention_excess_over_own_null": obs - null_mats[v][l].mean(),
                   "coexpression_excess_over_own_null": obs_co - nul_co.mean(),
                   "p_empirical": (1 + np.sum(nd >= (obs - obs_co))) / (1 + N_PERM)}
            dm[v].append(rec); pdm.append(rec["p_empirical"])
    qdm = bh(pdm)
    k = 0
    for v in VARIANTS:
        for l in range(L):
            dm[v][l]["p_bh_36"] = float(qdm[k]); k += 1
    null["degree_matched_delta_vs_coexpression_post_hoc"] = {
        "baseline": best_coexpr,
        "description": ("For every permutation, AUROC(attention) - AUROC(co-expression) on the same rewired edge set; "
                        "p = share of permutations whose difference is >= the observed difference. Tests whether "
                        "attention beats co-expression by more than their degree-structure advantage. Added after "
                        "inspecting the main results."),
        "per_variant": dm, "n_tests": len(pdm), "correction": "BH across 36 layer x variant tests"}

    # post hoc: marginal (row/column) vs pair-specific parts of attention
    mrec = {"additive": {v: [] for v in VARIANTS}, "interaction": {v: [] for v in VARIANTS},
            "interaction_residualised": {v: [] for v in VARIANTS}}
    pint, pir = [], []
    for l in range(L):
        add, I = marginal_parts(A[l])
        for part, M in (("additive", add), ("interaction", I)):
            sc = pair_scores(M, ri, ci)
            for v in VARIANTS:
                m = eval_score(sc[v], y, mult, rl)
                obs, nul = null_for(sc[v])
                m.update({"layer": l, "null_mean": nul.mean(),
                          "p_empirical": (1 + np.sum(nul >= obs)) / (1 + N_PERM)})
                mrec[part][v].append(m)
                if part == "interaction":
                    pint.append(m["p_empirical"])
                    res_i, r2_i, _ = ols_resid(sc[v], X)
                    m2 = eval_score(res_i, y, mult, rl)
                    obs2, nul2 = null_for(res_i)
                    m2.update({"layer": l, "r2_covariates": r2_i, "null_mean": nul2.mean(),
                               "null_ci95": [np.quantile(nul2, 0.025), np.quantile(nul2, 0.975)],
                               "p_empirical": (1 + np.sum(nul2 >= obs2)) / (1 + N_PERM)})
                    mrec["interaction_residualised"][v].append(m2)
                    pir.append(m2["p_empirical"])
    qint, qir = bh(pint), bh(pir)
    k = 0
    for l in range(L):
        for v in VARIANTS:
            mrec["interaction"][v][l]["p_bh_36"] = float(qint[k])
            mrec["interaction_residualised"][v][l]["p_bh_36"] = float(qir[k]); k += 1
    marginal_recovery = {
        "description": ("Edge recovery by the two parts of log10 head-mean attention: 'additive' = query row effect + key "
                        "column effect (gene-level hubness only), 'interaction' = pair-specific residual. TF-bootstrap "
                        "CIs and degree-preserving null as for the raw scores. 'interaction_residualised' = the pair-specific "
                        "part additionally residualised (OLS) on |r|, r, log mean expression of both genes and mean rank "
                        "distance: the combined test of signal beyond co-expression, rank proximity and hub degree. "
                        "Added after inspecting the main results."),
        **mrec, "n_tests_interaction": len(pint), "correction": "BH across 36 layer x variant tests (interaction)"}

    # best layer (primary = sym) + max-statistic correction for layer selection
    best_l = int(np.argmax([r["auroc"] for r in null["layers"]["sym"]]))
    obs_best = null["layers"]["sym"][best_l]["auroc"]
    max_null = null_mats["sym"].max(axis=0)
    null["best_layer"] = {
        "variant": "sym", "layer": best_l, "auroc": obs_best,
        "null_mean": float(null_mats["sym"][best_l].mean()),
        "null_ci95": [float(np.quantile(null_mats["sym"][best_l], 0.025)),
                      float(np.quantile(null_mats["sym"][best_l], 0.975))],
        "p_empirical_naive": float((1 + np.sum(null_mats["sym"][best_l] >= obs_best)) / (1 + N_PERM)),
        "p_empirical_max_over_layers": float((1 + np.sum(max_null >= obs_best)) / (1 + N_PERM)),
        "null_hist_values": null_mats["sym"][best_l].tolist(),
    }
    # best over all 36 layer x variant scores, with max-statistic over all 36
    all_obs = np.array([[null["layers"][v][l]["auroc"] for l in range(L)] for v in VARIANTS])
    vbest, lbest = np.unravel_index(np.argmax(all_obs), all_obs.shape)
    max36 = np.max(np.stack([null_mats[v] for v in VARIANTS]), axis=(0, 1))
    null["best_layer_any_variant"] = {
        "variant": VARIANTS[vbest], "layer": int(lbest), "auroc": float(all_obs[vbest, lbest]),
        "p_empirical_max_over_36": float((1 + np.sum(max36 >= all_obs[vbest, lbest])) / (1 + N_PERM)),
    }
    if do_heads:
        allp = np.concatenate([hp[v].ravel() for v in VARIANTS])
        q = bh(allp)
        qh = holm(allp)
        nh = d["H"].shape[1]
        size = L * nh
        per_head["p_empirical"] = {v: hp[v].tolist() for v in VARIANTS}
        per_head["p_bh_all_heads_variants"] = {v: q[i * size:(i + 1) * size].reshape(L, nh).tolist()
                                               for i, v in enumerate(VARIANTS)}
        per_head["p_holm_all_heads_variants"] = {v: qh[i * size:(i + 1) * size].reshape(L, nh).tolist()
                                                 for i, v in enumerate(VARIANTS)}
        per_head["null_mean_auroc"] = {v: hnull_mean[v].tolist() for v in VARIANTS}
        best_h = max(((v, l, h) for v in VARIANTS for l in range(L) for h in range(nh)),
                     key=lambda t: per_head[f"{t[0]}_auroc"][t[1], t[2]])
        bh_auc = per_head[f"{best_h[0]}_auroc"][best_h[1], best_h[2]]
        per_head["best_head"] = {"variant": best_h[0], "layer": best_h[1], "head": best_h[2],
                                 "auroc": bh_auc,
                                 "p_empirical_naive": hp[best_h[0]][best_h[1], best_h[2]],
                                 "p_empirical_max_over_432": float((1 + np.sum(hmax >= bh_auc)) / (1 + N_PERM))}
        per_head["n_heads_p_bh_lt_0.05"] = {v: int(np.sum(np.array(per_head["p_bh_all_heads_variants"][v]) < ALPHA))
                                           for v in VARIANTS}
        per_head["summary_auroc"] = {v: summarise(per_head[f"{v}_auroc"].ravel()) for v in VARIANTS}
        for key in [k for k in per_head if k.endswith("_auroc") or k.endswith("_auprc")]:
            if isinstance(per_head[key], np.ndarray):
                per_head[key] = per_head[key].tolist()
    log(f"[{name}] permutation null evaluated ({time.time() - t0:.0f}s)")

    # degree structure diagnostics: does attention track reference degree?
    col = "is_tf" if name == "trrust" else "is_dorothea_tf"
    e = edges[~edges.is_self]
    indeg = np.bincount(e.target_index.values, minlength=len(g))
    diag = []
    for l in range(L):
        Al = A[l].astype(np.float64)
        Cm = d["C"] >= MIN_COPRESENCE
        np.fill_diagonal(Cm, False)
        recv = np.nanmean(np.where(Cm, Al, np.nan), axis=0)     # mean attention received as key
        diag.append({"layer": l,
                     "spearman_key_attention_received_vs_ref_indegree": spearman_fast(recv, indeg),
                     "spearman_key_attention_received_vs_mean_rank": spearman_fast(recv, g.mean_rank_when_present.values)})

    return {
        "reference": name,
        "candidate_rows": rows_mode,
        "candidates": {"n_tf_rows": int(cand["rows"].size), "n_candidate_pairs": n, "n_positive": npos,
                       "base_rate": npos / n, "n_nonself_edges_in_G": cand["n_edges_nonself_in_G"],
                       "n_nonself_edges_on_rows": cand["n_edges_on_rows"],
                       "n_positive_tfs": int(len(set(ri[y == 1]))),
                       "definition": (f"ordered pairs (TF, gene): TF = reference TF in G, gene in G, gene != TF, "
                                      f"copresence >= {MIN_COPRESENCE}; positive = non-self reference edge")},
        "per_layer": per_layer,
        "baselines": baselines,
        "baseline_notes": {
            "degree_product_circular": "outdeg(TF) x indeg(target) from the reference restricted to G, including the scored edge: circular, upper-bound reference-bias baseline",
            "target_indegree_circular": "indeg(target) from the reference restricted to G (circular, upper bound)",
            "degree_product_loo": "same with the scored edge removed from both degrees (hub prior without self-leakage)",
            "target_indegree_loo": "indeg(target) with the scored edge removed",
            "neg_rank_distance": "minus mean |rank_TF - rank_gene| over co-present cells",
            "expr_product": "mean_norm_expr(TF) x mean_norm_expr(gene)",
        },
        "delta_vs_coexpression": diffs,
        "best_coexpression_baseline": best_coexpr,
        "n_tests_delta_bh": len(pdiff),
        "residualised": {"covariates": cov_names, "per_variant": resid},
        "grouped_cv_logistic": {"covariates_only_auroc": cov_only[0], "covariates_only_auprc": cov_only[1],
                                "score_added": "log10 sym attention", "folds": "GroupKFold(5) by TF",
                                "per_layer": logit},
        "per_head": per_head,
        "permutation_null": null,
        "marginal_vs_pair_specific_post_hoc": marginal_recovery,
        "degree_diagnostics": diag,
        "runtime_s": time.time() - t0,
        "_plot": {"null_mats_sym": null_mats["sym"]},
    }


# ----------------------------------------------------------- (e) rank proximity
def analysis_e(d: dict, iu: tuple) -> dict:
    A, H = d["A"], d["H"]
    L, nh = H.shape[0], H.shape[1]
    rd = d["RD"][iu].astype(np.float64)
    pr = d["P"][iu]
    rk_rd, rk_abs, rk_pr = stats.rankdata(rd), stats.rankdata(np.abs(pr)), stats.rankdata(pr)
    # directed: all ordered pairs i != j with copresence >= MIN
    per_layer = []
    for l in range(L):
        sym = 0.5 * (A[l][iu] + A[l].T[iu]).astype(np.float64)
        rs = stats.rankdata(sym)
        lg = np.log10(np.maximum(sym, ATT_FLOOR))
        slope, intercept, rr, _, _ = stats.linregress(rd, lg)
        per_layer.append({
            "layer": l,
            "spearman_sym_attention_vs_rank_distance": float(np.corrcoef(rs, rk_rd)[0, 1]),
            "spearman_sym_attention_vs_abs_pearson": float(np.corrcoef(rs, rk_abs)[0, 1]),
            "spearman_sym_attention_vs_pearson": float(np.corrcoef(rs, rk_pr)[0, 1]),
            "spearman_qi_attention_vs_rank_distance_upper": spearman_fast(A[l][iu], rd),
            "spearman_qj_attention_vs_rank_distance_lower": spearman_fast(A[l].T[iu], rd),
            "r2_log10_sym_attention_on_rank_distance_linear": float(rr ** 2),
        })
    head_rd = np.zeros((L, nh))
    head_abs = np.zeros((L, nh))
    for l in range(L):
        Hl = np.asarray(H[l])
        for h in range(nh):
            rs = stats.rankdata(0.5 * (Hl[h][iu] + Hl[h].T[iu]))
            head_rd[l, h] = np.corrcoef(rs, rk_rd)[0, 1]
            head_abs[l, h] = np.corrcoef(rs, rk_abs)[0, 1]
    return {
        "n_pairs": int(iu[0].size),
        "note": "Spearman over gene pairs; pairs are not independent, so no p-values are reported (descriptive).",
        "per_layer_head_mean": per_layer,
        "per_head_spearman_vs_rank_distance": head_rd.tolist(),
        "per_head_spearman_vs_abs_pearson": head_abs.tolist(),
        "per_head_summary": {"vs_rank_distance": summarise(head_rd.ravel()),
                             "vs_abs_pearson": summarise(head_abs.ravel())},
        "rank_distance_vs_abs_pearson_spearman": float(np.corrcoef(rk_rd, rk_abs)[0, 1]),
    }


# ---------------------------------------------------------------- verdicts
def verdicts(res: dict) -> dict:
    out = {}
    a = res["a_asymmetry"]["per_layer_head_mean"]
    rho = np.array([x["spearman"] for x in a]); rel = np.array([x["median_rel_asym"] for x in a])
    ok = (rho >= SYM_RHO_OK) & (rel <= SYM_RELASYM_OK)
    bad = (rho < SYM_RHO_BAD) | (rel > SYM_RELASYM_BAD)
    hs = res["a_asymmetry"]["per_head_summary"]
    if ok.all():
        v = "supported"
    elif bad.sum() > len(a) / 2:
        v = "not supported"
    else:
        v = "partially supported"
    out["attention is approximately symmetric"] = {
        "verdict": v,
        "rule": (f"layer head-mean: Spearman(A_ij,A_ji) >= {SYM_RHO_OK} and median relative asymmetry <= "
                 f"{SYM_RELASYM_OK} in all layers -> supported; Spearman < {SYM_RHO_BAD} or median relative asymmetry "
                 f"> {SYM_RELASYM_BAD} in a majority of layers -> not supported; otherwise partially"),
        "n_layers_meeting_symmetric_rule": int(ok.sum()), "n_layers_meeting_asymmetric_rule": int(bad.sum()),
        "layer_spearman_range": [rho.min(), rho.max()], "layer_median_rel_asym_range": [rel.min(), rel.max()],
        "per_head_spearman_median": hs["spearman"]["median"], "per_head_median_rel_asym_median": hs["median_rel_asym"]["median"],
    }
    mg = res["a_asymmetry"]["supplementary_marginal_decomposition_post_hoc"]["per_layer"]
    ri_ = [x["spearman_interaction_ij_vs_ji"] for x in mg]
    out["attention is approximately symmetric"]["post_hoc"] = (
        f"pair-specific interaction part of log10 attention (row and column effects removed): Spearman(I_ij, I_ji) "
        f"{min(ri_):.3f} to {max(ri_):.3f} across layers; additive row+column part explains "
        f"{min(x['r2_additive_row_col_on_log10_attention'] for x in mg):.2f} to "
        f"{max(x['r2_additive_row_col_on_log10_attention'] for x in mg):.2f} of log10 attention variance")

    c = res["c_edge_recovery"]["trrust"]
    bestco = c["best_coexpression_baseline"]
    sig_layers = []
    for v in VARIANTS:
        for rec in c["delta_vs_coexpression"][v]:
            if rec[f"delta_auroc_vs_{bestco}"] > 0 and rec[f"delta_auroc_vs_{bestco}_boot_p_bh"] < ALPHA:
                sig_layers.append((v, rec["layer"]))
    resid_ok = [(v, r["layer"]) for v in VARIANTS for r in c["residualised"]["per_variant"][v]
                if r["residual_auroc_ci95"][0] > 0.5]
    nullp = [(v, r["layer"]) for v in VARIANTS for r in c["permutation_null"]["layers"][v] if r["p_bh_layers"] < ALPHA]
    both = sorted(set(sig_layers) & set(resid_ok))
    if both and c["permutation_null"]["best_layer_any_variant"]["p_empirical_max_over_36"] < ALPHA:
        v = "supported"
    elif sig_layers or resid_ok:
        v = "partially supported"
    else:
        v = "not supported"
    out["attention recovers TRRUST edges above co-expression"] = {
        "verdict": v,
        "rule": (f"supported if some (variant, layer) has AUROC above the best co-expression baseline ({bestco}) with "
                 f"BH-adjusted TF-bootstrap p < {ALPHA} AND a residualised AUROC whose TF-bootstrap 95% CI excludes 0.5, "
                 f"AND the best layer x variant survives the degree-preserving null with max-statistic correction; "
                 f"partially if only one of the first two holds; otherwise not supported"),
        "best_coexpression_baseline_auroc": c["baselines"][bestco]["auroc"],
        "variant_layers_above_coexpression_bh": [f"{a}:L{b}" for a, b in sig_layers],
        "variant_layers_residual_ci_above_0.5": [f"{a}:L{b}" for a, b in resid_ok],
        "variant_layers_beating_degree_null_bh": [f"{a}:L{b}" for a, b in nullp],
        "best_layer_any_variant": c["permutation_null"]["best_layer_any_variant"],
    }
    dmq = c["permutation_null"]["degree_matched_delta_vs_coexpression_post_hoc"]["per_variant"]
    dm_sig = [f"{v}:L{r['layer'] + 1}" for v in VARIANTS for r in dmq[v]
              if r["delta_auroc_observed"] > 0 and r["p_bh_36"] < ALPHA]
    intr = c["marginal_vs_pair_specific_post_hoc"]["interaction"]
    int_sig = [f"{v}:L{r['layer'] + 1}" for v in VARIANTS for r in intr[v] if r["p_bh_36"] < ALPHA]
    rs_null = [f"{v}:L{r['layer'] + 1}" for v in VARIANTS for r in c["permutation_null"]["residual_layers"][v]
               if r["p_bh_layers"] < ALPHA]
    ir = c["marginal_vs_pair_specific_post_hoc"]["interaction_residualised"]
    ir_sig = [f"{v}:L{r['layer'] + 1}" for v in VARIANTS for r in ir[v] if r["p_bh_36"] < ALPHA]
    k = "attention recovers TRRUST edges above co-expression"
    out[k]["post_hoc_combined_beyond_coexpr_rank_degree_bh"] = ir_sig
    out[k]["post_hoc_degree_matched_variant_layers_bh"] = dm_sig
    out[k]["post_hoc_interaction_only_beats_degree_null_bh"] = int_sig
    out[k]["residual_beats_degree_null_bh"] = rs_null
    out[k]["post_hoc"] = (
        "degree-matched: attention exceeds co-expression by more than the degree-preserving null difference in "
        + (", ".join(dm_sig) if dm_sig else "no layer x variant") + " (BH over 36); pair-specific (row/column-removed) "
        "attention beats the degree null in " + (", ".join(int_sig) if int_sig else "no layer x variant")
        + " (BH over 36); covariate residual beats the degree null in " + (", ".join(rs_null) if rs_null else "no layer x variant")
        + " (BH over 12 per variant); pair-specific part residualised on co-expression, expression and rank distance "
        "beats the degree null in " + (", ".join(ir_sig) if ir_sig else "no layer x variant") + " (BH over 36)")

    e = res["e_rank_proximity"]["per_layer_head_mean"]
    rr = np.array([abs(x["spearman_sym_attention_vs_rank_distance"]) for x in e])
    med = float(np.median(rr))
    v = ("supported (largely)" if med >= RANK_RHO_LARGE else
         "partially supported" if med >= RANK_RHO_PARTIAL else "not supported")
    out["attention is explained by expression rank proximity"] = {
        "verdict": v,
        "rule": (f"median over layers of |Spearman(sym attention, mean rank distance)| >= {RANK_RHO_LARGE} -> supported "
                 f"(largely); >= {RANK_RHO_PARTIAL} -> partially; otherwise not supported"),
        "median_abs_spearman_over_layers": med,
        "range_spearman": [min(x["spearman_sym_attention_vs_rank_distance"] for x in e),
                           max(x["spearman_sym_attention_vs_rank_distance"] for x in e)],
    }
    phs = res["e_rank_proximity"]["per_head_summary"]["vs_rank_distance"]
    r2r = [x["r2_log10_sym_attention_on_rank_distance_linear"] for x in e]
    out["attention is explained by expression rank proximity"]["post_hoc"] = (
        f"closer ranks go with more attention in every layer (Spearman {min(x['spearman_sym_attention_vs_rank_distance'] for x in e):.2f} "
        f"to {max(x['spearman_sym_attention_vs_rank_distance'] for x in e):.2f}); linear R2 of log10 attention on rank distance "
        f"{min(r2r):.3f}-{max(r2r):.3f}; individual heads range {phs['min']:.2f} to {phs['max']:.2f}")

    nl = c["permutation_null"]["layers"]["sym"]
    bl = c["permutation_null"]["best_layer"]
    excess = [r["null_mean"] - 0.5 for r in nl]
    frac = [(r["null_mean"] - 0.5) / (r["auroc"] - 0.5) if r["auroc"] > 0.5 else None for r in nl]
    deg_c = c["baselines"]["degree_product_circular"]["auroc"]
    deg_loo = c["baselines"]["degree_product_loo"]["auroc"]
    null_above = [r["null_ci95"][0] > 0.5 for r in nl]
    if sum(null_above) > len(nl) / 2 and deg_c > max(r["auroc"] for r in nl):
        v = "supported"
    elif any(null_above) or deg_loo > max(r["auroc"] for r in nl):
        v = "partially supported"
    else:
        v = "not supported"
    out["reference-degree bias inflates recovery"] = {
        "verdict": v,
        "rule": ("supported if the degree-preserving null AUROC of sym attention has its 95% interval above 0.5 in a "
                 "majority of layers (part of attention's recovery is reproduced by degree structure alone) AND the "
                 "circular degree baseline beats every attention layer; partially if either holds or the LOO degree "
                 "baseline beats every layer; otherwise not supported"),
        "sym_null_mean_minus_0.5_per_layer": excess,
        "sym_fraction_of_excess_auroc_reproduced_by_null_per_layer": frac,
        "n_layers_null_ci_above_0.5": int(sum(null_above)),
        "degree_product_circular_auroc": deg_c, "degree_product_loo_auroc": deg_loo,
        "best_sym_layer": bl["layer"], "best_sym_auroc": bl["auroc"], "best_sym_null_mean": bl["null_mean"],
    }
    mr = c["marginal_vs_pair_specific_post_hoc"]
    lb = bl["layer"]
    out["reference-degree bias inflates recovery"]["post_hoc"] = (
        f"at the best sym layer (L{lb + 1}) the additive row+column part of log10 attention alone reaches AUROC "
        f"{mr['additive']['sym'][lb]['auroc']:.3f} (degree-null mean {mr['additive']['sym'][lb]['null_mean']:.3f}); the "
        f"pair-specific part reaches {mr['interaction']['sym'][lb]['auroc']:.3f} (degree-null mean "
        f"{mr['interaction']['sym'][lb]['null_mean']:.3f}); leave-one-out target in-degree alone reaches "
        f"{c['baselines']['target_indegree_loo']['auroc']:.3f}")
    return out


# ------------------------------------------------------------------ figure
def make_figure(res: dict, plot: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 7, "axes.titlesize": 7.5,
                         "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6,
                         "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.labelcolor": "#0b0b0b", "legend.frameon": False})
    col = {"sym": "#2a78d6", "qTF": "#eb6834", "qTarget": "#1baf7a"}
    mk = {"sym": "o", "qTF": "s", "qTarget": "^"}
    lab = {"sym": "symmetrised", "qTF": "query = TF", "qTarget": "query = target"}
    ink2, grid = "#52514e", "#e1e0d9"
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.9), constrained_layout=True)
    Ls = np.arange(len(res["a_asymmetry"]["per_layer_head_mean"]))

    def style(ax):
        ax.grid(axis="y", color=grid, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xticks(Ls)
        ax.set_xticklabels([str(i + 1) for i in Ls])

    # A: symmetry
    ax = axs[0, 0]
    ph = np.array(res["a_asymmetry"]["per_head"]["spearman"])
    for l in Ls:
        ax.scatter(np.full(ph.shape[1], l) + np.linspace(-0.25, 0.25, ph.shape[1]), ph[l], s=5,
                   color="#9ec5f4", lw=0, zorder=2)
    ax.plot(Ls, [x["spearman"] for x in res["a_asymmetry"]["per_layer_head_mean"]], color=col["sym"],
            marker="o", ms=3.5, lw=1.5, zorder=3, label="head-mean")
    ax.axhline(SYM_RHO_OK, color=ink2, lw=0.6, ls=(0, (2, 2)))
    ax.text(Ls[-1], SYM_RHO_OK, "0.9", va="bottom", ha="right", color=ink2, fontsize=6)
    ax.set_title("A  Symmetry: Spearman(A$_{ij}$, A$_{ji}$)", loc="left")
    ax.set_xlabel("layer"); ax.set_ylabel("Spearman rho")
    ax.scatter([], [], s=5, color="#9ec5f4", label="individual heads")
    ax.legend(loc="lower left", fontsize=6)
    style(ax)

    c = res["c_edge_recovery"]["trrust"]
    # B: AUROC per layer vs baselines
    ax = axs[0, 1]
    for v in VARIANTS:
        au = [r["auroc"] for r in c["per_layer"][v]]
        if v == "sym":
            lo = [r["auroc_ci95"][0] for r in c["per_layer"][v]]
            hi = [r["auroc_ci95"][1] for r in c["per_layer"][v]]
            ax.fill_between(Ls, lo, hi, color=col[v], alpha=0.15, lw=0)
        ax.plot(Ls, au, color=col[v], marker=mk[v], ms=3, lw=1.2, label=lab[v])
    nm = [r["null_mean"] for r in c["permutation_null"]["layers"]["sym"]]
    ax.plot(Ls, nm, color=ink2, lw=0.8, ls=(0, (1, 1.5)), label="degree-preserving null (sym)")
    bl = c["baselines"]
    for k, txt in [("abs_pearson", "|Pearson|"), ("degree_product_loo", "degree (LOO)"),
                   ("neg_rank_distance", "rank proximity")]:
        ax.axhline(bl[k]["auroc"], color="#898781", lw=0.6)
        ax.text(-0.4, bl[k]["auroc"], txt, color=ink2, fontsize=5.5, va="bottom", ha="left")
    ax.axhline(0.5, color="#c3c2b7", lw=0.6)
    ax.set_title("B  TRRUST edge recovery", loc="left")
    ax.set_xlabel("layer"); ax.set_ylabel("AUROC (band: TF-bootstrap 95% CI)")
    style(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncol=4, fontsize=6.5)

    # C: what survives covariates and gene-level (row/column) effects, sym only
    ax = axs[0, 2]
    rr = c["residualised"]["per_variant"]["sym"]
    rn = c["permutation_null"]["residual_layers"]["sym"]
    it = c["marginal_vs_pair_specific_post_hoc"]["interaction"]["sym"]
    ax.fill_between(Ls, [r["residual_auroc_ci95"][0] for r in rr], [r["residual_auroc_ci95"][1] for r in rr],
                    color=col["sym"], alpha=0.12, lw=0)
    ax.plot(Ls, [r["residual_auroc"] for r in rr], color=col["sym"], marker="o", ms=3, lw=1.2,
            label="covariate residual")
    ax.plot(Ls, [r["null_mean"] for r in rn], color=col["sym"], lw=0.8, ls=(0, (1, 1.5)))
    ax.plot(Ls, [r["auroc"] for r in it], color="#0d366b", marker="D", ms=2.8, lw=1.2, mfc="#fcfcfb",
            label="pair-specific (row/col removed)")
    ax.plot(Ls, [r["null_mean"] for r in it], color="#0d366b", lw=0.8, ls=(0, (1, 1.5)))
    irr = c["marginal_vs_pair_specific_post_hoc"]["interaction_residualised"]["sym"]
    ax.plot(Ls, [r["auroc"] for r in irr], color="#52514e", marker="s", ms=2.8, lw=1.0,
            label="pair-specific + covariate residual")
    ax.plot(Ls, [r["null_mean"] for r in irr], color="#52514e", lw=0.8, ls=(0, (1, 1.5)))
    ax.axhline(0.5, color="#c3c2b7", lw=0.6)
    ax.set_title("C  Symmetrised attention, adjusted", loc="left")
    ax.set_xlabel("layer"); ax.set_ylabel("AUROC (dotted: degree-null mean)")
    ax.legend(loc="lower left", fontsize=5.3)
    style(ax)

    # D: rank proximity + co-expression
    ax = axs[1, 0]
    e = res["e_rank_proximity"]["per_layer_head_mean"]
    ax.plot(Ls, [x["spearman_sym_attention_vs_rank_distance"] for x in e], color=col["sym"], marker="o", ms=3,
            lw=1.2, label="vs rank distance")
    ax.plot(Ls, [x["spearman_sym_attention_vs_abs_pearson"] for x in e], color=col["qTF"], marker="s", ms=3,
            lw=1.2, label="vs |Pearson|")
    ax.axhline(0, color="#c3c2b7", lw=0.6)
    ax.set_title("D  Attention vs rank distance, co-expr.", loc="left")
    ax.set_xlabel("layer"); ax.set_ylabel("Spearman rho")
    ax.legend(loc="best", fontsize=6)
    style(ax)

    # E: directionality
    ax = axs[1, 1]
    b = res["b_directionality"]["trrust"]["per_layer"]
    med = np.array([x["median_log2_ratio_qTF_over_qTarget"] for x in b])
    lo = np.array([x["median_log2_ratio_ci95"][0] for x in b]); hi = np.array([x["median_log2_ratio_ci95"][1] for x in b])
    ctrl = [x["nonedge_control_median_log2_ratio"] for x in b]
    ax.errorbar(Ls, med, yerr=[med - lo, hi - med], color=col["sym"], marker="o", ms=3, lw=1.0, capsize=1.5,
                label="TRRUST edges")
    ax.plot(Ls, ctrl, color="#898781", marker="x", ms=3, lw=0.8, label="non-edge TF pairs")
    ax.axhline(0, color="#c3c2b7", lw=0.6)
    ax.set_title("E  Direction on TRRUST edges", loc="left")
    ax.set_xlabel("layer"); ax.set_ylabel("median log$_2$ A[TF,tg] / A[tg,TF]")
    ax.legend(loc="best", fontsize=6)
    style(ax)

    # F: permutation null for best layer
    ax = axs[1, 2]
    bl = c["permutation_null"]["best_layer"]
    nulv = np.array(bl["null_hist_values"])
    ax.hist(nulv, bins=30, color="#9ec5f4", edgecolor="#fcfcfb", lw=0.4)
    ax.axvline(bl["auroc"], color=col["sym"], lw=1.5)
    right = bl["auroc"] >= np.quantile(nulv, 0.8)
    ax.annotate(f"observed\nL{bl['layer'] + 1} = {bl['auroc']:.3f}\np = {bl['p_empirical_naive']:.3f}\n"
                f"p(max over layers) = {bl['p_empirical_max_over_layers']:.3f}",
                xy=(bl["auroc"], 0.97), xycoords=("data", "axes fraction"),
                xytext=(-4 if right else 4, 0), textcoords="offset points",
                va="top", ha="right" if right else "left", fontsize=5.5, color="#0b0b0b")
    ax.set_title("F  Degree-preserving null", loc="left")
    ax.set_xlabel("AUROC under rewired TRRUST edges"); ax.set_ylabel("permutations")
    ax.grid(axis="y", color=grid, lw=0.5); ax.set_axisbelow(True)

    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"fig_case_validation.{ext}", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------- summary
def fmt(x, n=3):
    return "NA" if x is None else f"{x:.{n}f}"


def interpretation(res: dict) -> str:
    """Plain-language reading of the numbers (every number interpolated from the JSON)."""
    c = res["c_edge_recovery"]["trrust"]; V = res["verdicts"]
    bl = c["permutation_null"]["best_layer"]; lb = bl["layer"]
    bc = c["best_coexpression_baseline"]
    frac = V["reference-degree bias inflates recovery"]["sym_fraction_of_excess_auroc_reproduced_by_null_per_layer"]
    frac = [f for f in frac if f is not None]
    dm = c["permutation_null"]["degree_matched_delta_vs_coexpression_post_hoc"]["per_variant"]["sym"][lb]
    it = c["marginal_vs_pair_specific_post_hoc"]["interaction"]["sym"][lb]
    ad = c["marginal_vs_pair_specific_post_hoc"]["additive"]["sym"][lb]
    a = res["a_asymmetry"]["per_layer_head_mean"]
    e = res["e_rank_proximity"]["per_layer_head_mean"]
    irm = c["marginal_vs_pair_specific_post_hoc"]["interaction_residualised"]
    irb_v, irb_l = max(((v, l) for v in VARIANTS for l in range(len(irm[v]))), key=lambda t: irm[t[0]][t[1]]["auroc"])
    irb, irb_name = irm[irb_v][irb_l], f"{irb_v} L{irb_l + 1}"
    n_ir = sum(1 for v in VARIANTS for r in irm[v] if r["p_bh_36"] < ALPHA)
    cdo = res["c_edge_recovery"]["dorothea_abc"]
    irmd = cdo["marginal_vs_pair_specific_post_hoc"]["interaction_residualised"]
    n_ird = sum(1 for v in VARIANTS for r in irmd[v] if r["p_bh_36"] < ALPHA)
    ex_t = [r["auroc"] - r["null_mean"] for r in irm["sym"]]
    ex_d = [r["auroc"] - r["null_mean"] for r in irmd["sym"]]
    n_dm_d = sum(1 for v in VARIANTS for r in cdo["permutation_null"]["degree_matched_delta_vs_coexpression_post_hoc"]["per_variant"][v]
                 if r["p_bh_36"] < ALPHA and r["delta_auroc_observed"] > 0)
    return (
        "**Reading.** "
        f"Head-mean attention is not symmetric (layer Spearman(A_ij, A_ji) {min(x['spearman'] for x in a):.2f}-"
        f"{max(x['spearman'] for x in a):.2f}). Symmetrised attention ranks TRRUST edges above non-edges (best layer "
        f"L{lb + 1}: AUROC {bl['auroc']:.3f}; AUPRC {c['per_layer']['sym'][lb]['auprc']:.4f} vs base rate "
        f"{c['candidates']['base_rate']:.4f}) and above {bc} ({c['baselines'][bc]['auroc']:.3f}). "
        f"However, rewiring the TRRUST edges while keeping every TF out-degree and target in-degree already gives "
        f"attention an AUROC of {bl['null_mean']:.3f} at that layer: across layers the degree-preserving null reproduces "
        f"{min(frac):.0%}-{max(frac):.0%} of attention's AUROC excess over 0.5. At L{lb + 1} the gene-level (row + column) "
        f"part of attention alone reaches AUROC {ad['auroc']:.3f}, the pair-specific part {it['auroc']:.3f} "
        f"(degree-null mean {it['null_mean']:.3f}, BH p {it['p_bh_36']:.3g}). "
        f"Attention's advantage over {bc} at L{lb + 1} ({dm['delta_auroc_observed']:.3f}) compares with "
        f"{dm['null_delta_mean']:.3f} expected from degree structure alone (BH p {dm['p_bh_36']:.3g}). "
        f"After also residualising the pair-specific part on co-expression, expression and rank distance, the best layer x "
        f"variant reaches AUROC {irb['auroc']:.3f} ({irb_name}; degree-null mean {irb['null_mean']:.3f}, BH p "
        f"{irb['p_bh_36']:.3g}); {n_ir}/36 layer x variant tests pass BH < 0.05. With DoRothEA A-C "
        f"({cdo['candidates']['n_positive']} edges vs {c['candidates']['n_positive']}) the same combined test passes BH < 0.05 in "
        f"{n_ird}/36; the sym excess over the degree null is {min(ex_d):.3f} to {max(ex_d):.3f} AUROC across layers "
        f"(TRRUST: {min(ex_t):.3f} to {max(ex_t):.3f}), and the degree-matched advantage over co-expression passes in "
        f"{n_dm_d}/36. "
        f"Leave-one-out target in-degree from TRRUST itself reaches AUROC {c['baselines']['target_indegree_loo']['auroc']:.3f}. "
        f"Rank proximity is only weakly related to attention (Spearman with rank distance "
        f"{min(x['spearman_sym_attention_vs_rank_distance'] for x in e):.2f} to "
        f"{max(x['spearman_sym_attention_vs_rank_distance'] for x in e):.2f}). "
        "Direct evidence: the AUROC, null and bootstrap numbers above. Inference: "
        + ("most of the apparent edge recovery reflects which genes are TRRUST hubs (well-studied TFs and frequently "
           "annotated targets) and which genes attract or emit attention, rather than pair-specific TF->target structure"
           + ("; a small pair-specific component (a few hundredths of AUROC) remains after removing co-expression, rank "
              "distance and degree structure, detectable with the larger DoRothEA A-C edge set but not with TRRUST, which is "
              "consistent with a power difference rather than a reference-specific effect (DoRothEA B-C levels partly derive "
              "from expression-based inference, so non-linear co-expression could contribute). "
              if (n_ird > 0 and n_ir == 0) else ". ")
           if float(np.median(frac)) >= 0.5 else
           "degree structure accounts for less than half of the apparent edge recovery in the median layer. ")
        + "Hypothesis (not tested here): hub-level agreement arises because broadly expressed, well-studied genes are "
        "both over-annotated in TRRUST and preferentially attended.")


def write_summary(res: dict, path: Path) -> None:
    a = res["a_asymmetry"]; b = res["b_directionality"]["trrust"]; c = res["c_edge_recovery"]["trrust"]
    cd = res["c_edge_recovery"]["dorothea_abc"]; e = res["e_rank_proximity"]; V = res["verdicts"]
    m = res["meta"]
    L = len(a["per_layer_head_mean"])
    lines = []
    lines.append("# Case study: does Geneformer V2-104M attention encode TF->target regulation?\n")
    lines.append(f"Generated by `{SCRIPT_PATH.name}` from the kit at `{m['kit']}` "
                 f"(kit manifest md5 `{m['kit_manifest_md5']}`). All numbers below are read from "
                 f"`results/independent_validation.json`; nothing here is hand-entered.\n")
    lines.append(f"Data: {m['n_cells']} Tabula Sapiens immune cells ({m['assay']}), {m['n_genes']} genes "
                 f"(G), 12 layers x 12 heads. Pair filter: co-presence >= {MIN_COPRESENCE} cells. "
                 f"A[i,j] = attention from query gene i to key gene j, averaged over co-present cells.\n")
    lines.append("## Verdicts on the four statements (pre-specified rules in the JSON)\n")
    lines.append("| statement | pre-specified verdict | post-hoc qualification |\n|---|---|---|")
    for k, v in V.items():
        lines.append(f"| {k} | **{v['verdict']}** | {v.get('post_hoc', '')} |")
    lines.append("")
    lines.append(interpretation(res))
    lines.append("")

    # a
    pl = a["per_layer_head_mean"]
    lines.append("## (a) Symmetry\n")
    lines.append("| layer | Pearson(A_ij,A_ji) | Spearman | median rel. asym. | frac >= 2x |\n|---|---|---|---|---|")
    for x in pl:
        lines.append(f"| {x['layer'] + 1} | {fmt(x['pearson'])} | {fmt(x['spearman'])} | {fmt(x['median_rel_asym'])} | {fmt(x['frac_ge_2x'])} |")
    hs = a["per_head_summary"]
    lines.append(f"\nAcross all 144 heads: Spearman median {fmt(hs['spearman']['median'])} "
                 f"(range {fmt(hs['spearman']['min'])} to {fmt(hs['spearman']['max'])}); median relative asymmetry "
                 f"median {fmt(hs['median_rel_asym']['median'])} (range {fmt(hs['median_rel_asym']['min'])} to "
                 f"{fmt(hs['median_rel_asym']['max'])}); fraction of pairs with >= 2x asymmetry median "
                 f"{fmt(hs['frac_ge_2x']['median'])}. Pairs: {a['n_pairs']}.\n")
    va = V["attention is approximately symmetric"]
    lines.append(f"Verdict: {va['verdict']} ({va['n_layers_meeting_symmetric_rule']}/{L} layers meet the symmetric "
                 f"rule, {va['n_layers_meeting_asymmetric_rule']}/{L} meet the asymmetric rule).\n")

    # b
    lines.append("## (b) Directionality on TRRUST edges\n")
    lines.append(b["orientation"] + "\n")
    lines.append("| layer | n | median log2 ratio [95% CI] | frac > 0 | Wilcoxon p (BH) | non-edge median | edges vs non-edges MWU p (BH) |\n|---|---|---|---|---|---|---|")
    for x in b["per_layer"]:
        lines.append(f"| {x['layer'] + 1} | {x['n_edges']} | {fmt(x['median_log2_ratio_qTF_over_qTarget'])} "
                     f"[{fmt(x['median_log2_ratio_ci95'][0])}, {fmt(x['median_log2_ratio_ci95'][1])}] | "
                     f"{fmt(x['frac_edges_qTF_gt_qTarget'])} | {x['wilcoxon_p']:.2g} ({x['wilcoxon_p_bh']:.2g}) | "
                     f"{fmt(x['nonedge_control_median_log2_ratio'])} | {x['edges_vs_nonedge_mwu_p']:.2g} ({x['edges_vs_nonedge_mwu_p_bh']:.2g}) |")
    lines.append("")

    # c
    cand = c["candidates"]
    lines.append("## (c) TRRUST edge recovery\n")
    lines.append(f"Candidates: {cand['n_candidate_pairs']} ordered (TF, gene) pairs over {cand['n_tf_rows']} TRRUST TFs in G; "
                 f"{cand['n_positive']} positives from {cand['n_positive_tfs']} TFs (base rate {cand['base_rate']:.4f}). "
                 f"95% CIs: TF-level bootstrap ({N_BOOT} resamples of TFs).\n")
    lines.append("| layer | AUROC sym [CI] | AUROC q=TF | AUROC q=target | AUPRC sym | delta AUROC sym vs "
                 f"{c['best_coexpression_baseline']} [CI] (BH p) | degree-null mean | null p (BH) |\n|---|---|---|---|---|---|---|---|")
    for l in range(L):
        s = c["per_layer"]["sym"][l]; dd = c["delta_vs_coexpression"]["sym"][l]
        nl = c["permutation_null"]["layers"]["sym"][l]
        bc = c["best_coexpression_baseline"]
        lines.append(f"| {l + 1} | {fmt(s['auroc'])} [{fmt(s['auroc_ci95'][0])}, {fmt(s['auroc_ci95'][1])}] | "
                     f"{fmt(c['per_layer']['qTF'][l]['auroc'])} | {fmt(c['per_layer']['qTarget'][l]['auroc'])} | "
                     f"{fmt(s['auprc'], 4)} | {fmt(dd[f'delta_auroc_vs_{bc}'])} [{fmt(dd[f'delta_auroc_vs_{bc}_ci95'][0])}, "
                     f"{fmt(dd[f'delta_auroc_vs_{bc}_ci95'][1])}] ({dd[f'delta_auroc_vs_{bc}_boot_p_bh']:.2g}) | "
                     f"{fmt(nl['null_mean'])} | {nl['p_empirical']:.3g} ({nl['p_bh_layers']:.3g}) |")
    lines.append("\nBaselines (AUROC [CI], AUPRC; degree-preserving-null mean):\n")
    lines.append("| baseline | AUROC [95% CI] | AUPRC | null mean |\n|---|---|---|---|")
    for k, v in c["baselines"].items():
        lines.append(f"| {k} | {fmt(v['auroc'])} [{fmt(v['auroc_ci95'][0])}, {fmt(v['auroc_ci95'][1])}] | {fmt(v['auprc'], 4)} | "
                     f"{fmt(c['permutation_null']['baselines'][k]['null_mean'])} |")
    lines.append("\n`*_circular` degree baselines use the scored edge itself (upper-bound reference-bias baseline); "
                 "`*_loo` remove it.\n")
    if c["per_head"] is not None:
        ph = c["per_head"]
        lines.append(f"Per head (144 heads x 3 variants = 432 tests): AUROC sym median {fmt(ph['summary_auroc']['sym']['median'])} "
                     f"(range {fmt(ph['summary_auroc']['sym']['min'])}-{fmt(ph['summary_auroc']['sym']['max'])}); "
                     f"heads with BH-adjusted degree-null p < 0.05: sym {ph['n_heads_p_bh_lt_0.05']['sym']}, "
                     f"q=TF {ph['n_heads_p_bh_lt_0.05']['qTF']}, q=target {ph['n_heads_p_bh_lt_0.05']['qTarget']}. "
                     f"Best head: {ph['best_head']['variant']} L{ph['best_head']['layer'] + 1}H{ph['best_head']['head'] + 1} "
                     f"AUROC {fmt(ph['best_head']['auroc'])}, naive p {ph['best_head']['p_empirical_naive']:.3g}, "
                     f"max-statistic p over 432 {ph['best_head']['p_empirical_max_over_432']:.3g}.\n")
    lines.append(f"DoRothEA A-C replication ({cd['candidates']['n_positive']} positives, base rate "
                 f"{cd['candidates']['base_rate']:.4f}): AUROC sym per layer "
                 + ", ".join(fmt(r["auroc"]) for r in cd["per_layer"]["sym"])
                 + f"; |Pearson| {fmt(cd['baselines']['abs_pearson']['auroc'])}; degree LOO "
                 f"{fmt(cd['baselines']['degree_product_loo']['auroc'])}; degree-null mean (sym) "
                 + ", ".join(fmt(r["null_mean"]) for r in cd["permutation_null"]["layers"]["sym"]) + ". "
                 + f"Best sym layer L{cd['permutation_null']['best_layer']['layer'] + 1}: AUROC {fmt(cd['permutation_null']['best_layer']['auroc'])}, "
                 f"degree-null p {cd['permutation_null']['best_layer']['p_empirical_max_over_layers']:.3g} (max over layers). "
                 f"Degree-matched advantage over {cd['best_coexpression_baseline']} passes BH < 0.05 in "
                 f"{sum(1 for v in VARIANTS for r in cd['permutation_null']['degree_matched_delta_vs_coexpression_post_hoc']['per_variant'][v] if r['p_bh_36'] < ALPHA and r['delta_auroc_observed'] > 0)}/36 "
                 f"layer x variant tests; pair-specific + covariate-residualised attention beats the degree null in "
                 f"{sum(1 for v in VARIANTS for r in cd['marginal_vs_pair_specific_post_hoc']['interaction_residualised'][v] if r['p_bh_36'] < ALPHA)}/36.\n")

    # d
    lines.append("## (d) Residualised attention\n")
    lines.append("OLS of attention on |r|, r, log mean expression of both genes and mean rank distance over the candidate pairs; "
                 "AUROC of the residual.\n")
    lines.append("| layer | R2 sym | residual AUROC sym [CI] | q=TF | q=target | log10-attention residual AUROC sym | grouped-CV logistic delta AUROC (cov + attention - cov) |\n|---|---|---|---|---|---|---|")
    g = c["grouped_cv_logistic"]
    for l in range(L):
        rs = c["residualised"]["per_variant"]["sym"][l]
        lines.append(f"| {l + 1} | {fmt(rs['r2_covariates'])} | {fmt(rs['residual_auroc'])} [{fmt(rs['residual_auroc_ci95'][0])}, "
                     f"{fmt(rs['residual_auroc_ci95'][1])}] | {fmt(c['residualised']['per_variant']['qTF'][l]['residual_auroc'])} | "
                     f"{fmt(c['residualised']['per_variant']['qTarget'][l]['residual_auroc'])} | "
                     f"{fmt(rs['log10_attention_residual_auroc'])} | {fmt(g['per_layer'][l]['delta_auroc'])} |")
    lines.append(f"\nCovariates-only grouped-CV logistic AUROC {fmt(g['covariates_only_auroc'])}, AUPRC {fmt(g['covariates_only_auprc'], 4)}.\n")

    # e
    lines.append("## (e) Rank proximity and co-expression\n")
    lines.append("| layer | Spearman(sym attention, rank distance) | Spearman(sym attention, abs Pearson) | Spearman(sym attention, Pearson) |\n|---|---|---|---|")
    for x in e["per_layer_head_mean"]:
        lines.append(f"| {x['layer'] + 1} | {fmt(x['spearman_sym_attention_vs_rank_distance'])} | "
                     f"{fmt(x['spearman_sym_attention_vs_abs_pearson'])} | {fmt(x['spearman_sym_attention_vs_pearson'])} |")
    lines.append(f"\nPer head, Spearman with rank distance: median {fmt(e['per_head_summary']['vs_rank_distance']['median'])} "
                 f"(range {fmt(e['per_head_summary']['vs_rank_distance']['min'])} to {fmt(e['per_head_summary']['vs_rank_distance']['max'])}). "
                 f"Rank distance vs |Pearson| Spearman: {fmt(e['rank_distance_vs_abs_pearson_spearman'])}.\n")

    # f
    bl = c["permutation_null"]["best_layer"]; ba = c["permutation_null"]["best_layer_any_variant"]
    lines.append("## (f) Degree-preserving permutation null\n")
    lines.append(f"Curveball rewiring of the {cand['n_positive']} TRRUST positives inside the candidate set, preserving every TF "
                 f"out-degree and target in-degree ({N_PERM} permutations, burn-in {CURVEBALL_BURNIN} trades, "
                 f"{CURVEBALL_THIN} trades between samples). Best sym layer: L{bl['layer'] + 1}, AUROC {fmt(bl['auroc'])}, "
                 f"null mean {fmt(bl['null_mean'])} [{fmt(bl['null_ci95'][0])}, {fmt(bl['null_ci95'][1])}], empirical p "
                 f"{bl['p_empirical_naive']:.3g} (naive), {bl['p_empirical_max_over_layers']:.3g} (max over 12 layers). "
                 f"Best of 36 layer x variant scores: {ba['variant']} L{ba['layer'] + 1}, AUROC {fmt(ba['auroc'])}, "
                 f"max-statistic p over 36 = {ba['p_empirical_max_over_36']:.3g}.\n")
    vd = V["reference-degree bias inflates recovery"]
    lines.append(f"Degree diagnostics: circular degree-product AUROC {fmt(vd['degree_product_circular_auroc'])}, LOO "
                 f"{fmt(vd['degree_product_loo_auroc'])}; layers whose degree-null 95% interval lies above 0.5: "
                 f"{vd['n_layers_null_ci_above_0.5']}/{L}.\n")

    # post hoc sections
    dmq = c["permutation_null"]["degree_matched_delta_vs_coexpression_post_hoc"]
    mr = c["marginal_vs_pair_specific_post_hoc"]
    mg = a["supplementary_marginal_decomposition_post_hoc"]["per_layer"]
    lines.append("## Post-hoc diagnostics (added after inspecting (a)-(f))\n")
    lines.append(dmq["description"] + " " + mr["description"] + "\n")
    lines.append(f"| layer | sym delta AUROC vs {dmq['baseline']} | null delta mean | p (BH 36) | additive-part AUROC sym (null mean) | pair-specific AUROC sym [CI] (null mean) | p (BH 36) | pair-specific + covariate-residualised AUROC sym (null mean, BH p) | R2 additive | Spearman(I_ij, I_ji) |\n|---|---|---|---|---|---|---|---|---|---|")
    for l in range(L):
        x = dmq["per_variant"]["sym"][l]; ad = mr["additive"]["sym"][l]; it = mr["interaction"]["sym"][l]
        lines.append(f"| {l + 1} | {fmt(x['delta_auroc_observed'])} | {fmt(x['null_delta_mean'])} | {x['p_empirical']:.3g} ({x['p_bh_36']:.3g}) | "
                     f"{fmt(ad['auroc'])} ({fmt(ad['null_mean'])}) | {fmt(it['auroc'])} [{fmt(it['auroc_ci95'][0])}, {fmt(it['auroc_ci95'][1])}] "
                     f"({fmt(it['null_mean'])}) | {it['p_empirical']:.3g} ({it['p_bh_36']:.3g}) | "
                     f"{fmt(mr['interaction_residualised']['sym'][l]['auroc'])} ({fmt(mr['interaction_residualised']['sym'][l]['null_mean'])}, "
                     f"{mr['interaction_residualised']['sym'][l]['p_bh_36']:.3g}) | "
                     f"{fmt(mg[l]['r2_additive_row_col_on_log10_attention'])} | {fmt(mg[l]['spearman_interaction_ij_vs_ji'])} |")
    lines.append("")

    # g
    lines.append("## (g) Multiple testing\n")
    for k, v in res["g_multiple_testing"].items():
        lines.append(f"- {k}: {v['n_tests']} tests, {v['correction']}")
    lines.append("")
    lines.append("## Caveats\n")
    for cv in res["caveats"]:
        lines.append(f"- {cv}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------- main
def main() -> None:
    global MIN_COPRESENCE, N_PERM, N_BOOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    ap.add_argument("--out", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--no-heads", action="store_true")
    ap.add_argument("--min-copresence", type=int, default=MIN_COPRESENCE, help="(smoke tests only)")
    ap.add_argument("--n-perm", type=int, default=N_PERM, help="(smoke tests only)")
    ap.add_argument("--n-boot", type=int, default=N_BOOT, help="(smoke tests only)")
    args = ap.parse_args()
    MIN_COPRESENCE, N_PERM, N_BOOT = args.min_copresence, args.n_perm, args.n_boot
    t0 = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    d = load_kit(args.kit)
    nG = len(d["genes"])
    C = d["C"]
    iu_all = np.triu_indices(nG, 1)
    keep = C[iu_all] >= MIN_COPRESENCE
    iu = (iu_all[0][keep], iu_all[1][keep])
    log(f"kit loaded: |G|={nG}, pairs i<j with C>={MIN_COPRESENCE}: {iu[0].size} of {iu_all[0].size}")

    res = {"meta": {
        "kit": str(args.kit), "kit_manifest_md5": md5_file(args.kit / "kit_manifest.json"),
        "script": str(SCRIPT_PATH), "script_md5": md5_file(SCRIPT_PATH),
        "n_cells": int(len(d["cells"])), "n_genes": nG, "assay": d["manifest"]["design"]["assay_filter"],
        "n_layers": int(d["A"].shape[0]), "n_heads": int(d["H"].shape[1]),
        "heads_file": d["heads_file"], "min_copresence": MIN_COPRESENCE,
        "n_pairs_upper_total": int(iu_all[0].size), "n_pairs_upper_copresence_ok": int(iu[0].size),
        "seed": SEED, "n_boot_tf": N_BOOT, "n_boot_median": N_BOOT_MEDIAN, "n_perm": N_PERM,
        "orientation": "attention[l,i,j] = mean attention from query gene i to key gene j",
        "primary_score": "sym = (A[TF,g] + A[g,TF]) / 2 (pre-specified); qTF = A[TF,g]; qTarget = A[g,TF]",
    }}
    log("(a) asymmetry"); res["a_asymmetry"] = analysis_a(d, iu)
    log("(e) rank proximity"); res["e_rank_proximity"] = analysis_e(d, iu)
    plot = {}
    res["c_edge_recovery"] = {}
    res["b_directionality"] = {}
    for name, edges, heads in [("trrust", d["trrust"], not args.no_heads), ("dorothea_abc", d["dorothea"], False)]:
        r = run_reference(d, name, edges, "all_tfs_in_G", heads, SEED)
        plot[name] = r.pop("_plot")
        res["c_edge_recovery"][name] = r
        cand = build_reference(d, edges, "all_tfs_in_G")
        res["b_directionality"][name] = analysis_b(d, edges, cand, SEED + 3)
    # sensitivity: rows restricted to TFs with >= 1 TRRUST target in G (no heads)
    d["_ref_name"] = "trrust"
    sens = run_reference(d, "trrust", d["trrust"], "tfs_with_edge", False, SEED)
    sens.pop("_plot")
    res["c_edge_recovery"]["trrust_rows_tfs_with_edge"] = {
        "candidates": sens["candidates"],
        "per_layer": sens["per_layer"], "baselines": sens["baselines"],
        "residualised_sym": sens["residualised"]["per_variant"]["sym"],
        "permutation_null_sym": sens["permutation_null"]["layers"]["sym"],
        "best_layer": {k: v for k, v in sens["permutation_null"]["best_layer"].items() if k != "null_hist_values"},
        "best_layer_any_variant": sens["permutation_null"]["best_layer_any_variant"],
    }
    # alias per spec naming
    res["d_residualised"] = {k: {"covariates": v["residualised"]["covariates"],
                                 "per_variant": v["residualised"]["per_variant"],
                                 "grouped_cv_logistic": v["grouped_cv_logistic"]}
                             for k, v in res["c_edge_recovery"].items() if "residualised" in v}
    res["f_permutation_null"] = {k: {"best_layer": v["permutation_null"]["best_layer"],
                                     "best_layer_any_variant": v["permutation_null"]["best_layer_any_variant"],
                                     "method": (f"curveball rewiring within candidate set; {N_PERM} permutations; "
                                                f"burn-in {CURVEBALL_BURNIN}, thinning {CURVEBALL_THIN} trades; "
                                                "p = (1 + #null >= obs) / (1 + n_perm)")}
                                 for k, v in res["c_edge_recovery"].items() if "permutation_null" in v}
    L = d["A"].shape[0]
    res["g_multiple_testing"] = {
        "b_wilcoxon_per_layer_trrust": {"n_tests": L, "correction": "BH across layers"},
        "b_edges_vs_nonedges_mwu_trrust": {"n_tests": L, "correction": "BH across layers"},
        "b_wilcoxon_per_layer_dorothea": {"n_tests": L, "correction": "BH across layers"},
        "c_layer_auroc_vs_degree_null_trrust": {"n_tests": 3 * L, "correction": "BH across layers within each variant (12 each); max-statistic over 12 layers for the best sym layer and over 36 for the best layer x variant"},
        "c_head_auroc_vs_degree_null_trrust": {"n_tests": 3 * L * int(d["H"].shape[1]), "correction": "BH and Holm across all 432 head x variant tests; max-statistic for the best head"},
        "c_delta_auroc_vs_best_coexpression_trrust": {"n_tests": 3 * L, "correction": "BH across 36 layer x variant tests"},
        "d_residual_auroc_vs_degree_null_trrust": {"n_tests": 3 * L, "correction": "BH across layers within each variant"},
        "same_families_dorothea_abc": {"n_tests": 3 * L * 3 + L, "correction": "as for TRRUST (no per-head tests)"},
        "a_and_e": {"n_tests": 0, "correction": "descriptive only (pairs are not independent)"},
        "post_hoc_degree_matched_delta_vs_coexpression_trrust": {"n_tests": 3 * L, "correction": "BH across 36"},
        "post_hoc_pair_specific_vs_degree_null_trrust": {"n_tests": 3 * L, "correction": "BH across 36"},
        "post_hoc_pair_specific_residualised_vs_degree_null_trrust": {"n_tests": 3 * L, "correction": "BH across 36"},
    }
    res["verdicts"] = verdicts(res)
    res["caveats"] = [
        "TRRUST is literature-curated and biased toward well-studied TFs/targets; recovery metrics inherit that bias, which is why the degree baselines and degree-preserving null are reported.",
        "Attention means are computed only over cells where both genes are present (co-presence >= 50 cells); present/absent structure itself is not scored.",
        "Pairs share genes, so pair-level tests (Wilcoxon, Mann-Whitney) are anti-conservative; TF-level bootstrap and permutation p-values are the primary inference.",
        "One tissue compartment (immune), one assay (10x 3' v3), 1,000 cells, one model (Geneformer V2-104M); per-head values use the float32 kit file.",
        "OLS residualisation is linear in the covariates; non-linear dependence on rank distance or expression could remain in the residual.",
    ]
    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    out_json = args.out / "independent_validation.json"
    out_json.write_text(json.dumps(r4(res), indent=1), encoding="utf-8")
    log(f"wrote {out_json}")
    make_figure(json.loads(out_json.read_text()), plot, args.out)
    write_summary(json.loads(out_json.read_text()), args.out / "SUMMARY.md")
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
