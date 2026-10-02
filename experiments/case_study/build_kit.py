#!/usr/bin/env python
"""Build the Geneformer V2-104M attention case-study kit (MI-Workbench paper).

Question the kit supports:
    Does Geneformer attention encode TF->target regulation beyond co-expression,
    expression-rank proximity and hub degree?

This is author-side code. It performs real computation on real data and writes
every downstream input for ``independent_validation.py`` into ``--out``:

    genes.tsv, cells.tsv, tokenized_cells.npz,
    attention_layer_mean.npy, attention_heads.npy, copresence_counts.npy,
    rank_distance_mean.npy, coexpr_pearson.npy, coexpr_spearman.npy,
    trrust_edges.tsv, dorothea_abc_edges.tsv, kit_manifest.json, README.md

Stages (each skipped when its outputs already exist, so the script is resumable):
    1. prepare   - stratified cell sample, Geneformer V2 rank-value tokenisation,
                   gene set G, co-expression matrices, reference edges.
    2. attention - per-cell forward pass (layer by layer, so only one layer's
                   attention tensor is alive at a time), accumulating per-head and
                   head-averaged attention over co-present gene pairs.  Checkpoints
                   every ``--ckpt-every`` cells and resumes from the last checkpoint.
    3. finalize  - means, sanity checks, manifest and README.

Run (see README.md in this folder):
    export TMPDIR="/Volumes/Crucial X6/tmp_miw"; export OMP_NUM_THREADS=4
    /Users/ihorkendiukhov/anaconda3/envs/subproject02-evalbias-rev/bin/python \
        experiments/case_study/build_kit.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import platform
import subprocess
import sys
import time
from importlib import metadata
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata

# --------------------------------------------------------------------------- paths
SCRIPT_PATH = Path(__file__).resolve()
WORKBENCH = SCRIPT_PATH.parents[2]
BIODYN = WORKBENCH.parents[1]
DEFAULT_OUT = WORKBENCH / "experiments_data" / "case_study_kit"
H5AD = BIODYN / "single_cell_mechinterp/data/raw/tabula_sapiens_immune_subset_20000.h5ad"
TRRUST = BIODYN / "single_cell_mechinterp/external/networks/trrust_human.tsv"
DOROTHEA = BIODYN / "single_cell_mechinterp/external/networks/dorothea_human.tsv"
GF_HUB = Path.home() / ".cache/huggingface/hub/models--ctheodoris--Geneformer/snapshots"
MODEL_SNAPSHOT = "fcd26c45fc30fba1989e586bdc46bc366dda8655"
MODEL_SUBDIR = "Geneformer-V2-104M"
DICT_FILES = {
    "token_dictionary": "geneformer/token_dictionary_gc104M.pkl",
    "gene_median_dictionary": "geneformer/gene_median_dictionary_gc104M.pkl",
    "gene_name_id_dict": "geneformer/gene_name_id_dict_gc104M.pkl",
}

# Cap the MPS caching allocator (default high watermark 1.7x the recommended working set
# let the cache grow to ~36 GB on a 32 GB machine and pushed the system into swap); the
# per-cell torch.mps.empty_cache() below releases cached blocks of varying sequence length.
MPS_HIGH_WATERMARK, MPS_LOW_WATERMARK = "0.40", "0.30"   # low must not exceed high
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", MPS_HIGH_WATERMARK)
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", MPS_LOW_WATERMARK)

# ------------------------------------------------------------------ design constants
SEED = 20261001
N_CELLS = 1000
TARGET_SUM = 10_000          # Geneformer transcriptome tokenizer: counts / n_counts * 1e4
MODEL_INPUT_SIZE = 4096      # V2: truncate genes to 4094, then prepend <cls>, append <eos>
N_TOP_GENES = 1200
TF_MIN_PRESENCE = 0.30
MAX_GENES = 1500
DOROTHEA_LEVELS = ("A", "B", "C")
DEFAULT_ASSAY = "10x 3' v3"  # single UMI chemistry: avoids assay-driven co-expression


# ------------------------------------------------------------------------ helpers
def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def md5_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def md5_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def atomic_save_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def resolve_dict_files() -> dict:
    """gc104M dictionaries are spread over several HF snapshots; pick the first copy
    (sorted by snapshot hash) and record the md5 of every copy for provenance."""
    out = {}
    for key, rel in DICT_FILES.items():
        hits = sorted(GF_HUB.glob(f"*/{rel}"))
        if not hits:
            raise FileNotFoundError(f"{rel} not found in any snapshot under {GF_HUB}")
        md5s = {h.parts[-3]: md5_file(h) for h in hits}
        if len(set(md5s.values())) != 1:
            raise RuntimeError(f"{rel}: copies differ across snapshots: {md5s}")
        out[key] = {
            "path": str(hits[0]),
            "snapshot": hits[0].parts[-3],
            "md5": md5s[hits[0].parts[-3]],
            "copies_md5": md5s,
        }
    return out


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def pkg_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in ["numpy", "scipy", "pandas", "h5py", "torch", "transformers", "safetensors",
              "anndata", "scanpy", "scikit-learn"]:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def git_head(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def read_categorical(grp: h5py.Group, name: str) -> np.ndarray:
    obj = grp[name]
    if isinstance(obj, h5py.Group) and "categories" in obj:
        cats = obj["categories"][:].astype(str)
        codes = obj["codes"][:]
        out = np.where(codes >= 0, cats[np.clip(codes, 0, None)], "NA")
        return out.astype(object)
    arr = obj[:]
    if arr.dtype.kind in ("S", "O"):
        arr = arr.astype(str)
    return arr


def read_csr_rows(grp: h5py.Group, rows: np.ndarray, n_cols: int) -> sparse.csr_matrix:
    indptr = grp["indptr"][:]
    data, indices, ptr = [], [], [0]
    for r in rows:
        s, e = int(indptr[r]), int(indptr[r + 1])
        data.append(grp["data"][s:e])
        indices.append(grp["indices"][s:e])
        ptr.append(ptr[-1] + (e - s))
    return sparse.csr_matrix(
        (np.concatenate(data), np.concatenate(indices), np.asarray(ptr)),
        shape=(len(rows), n_cols),
    )


# --------------------------------------------------------------------- stage 1
def stratified_sample(cell_types: np.ndarray, eligible: np.ndarray, n: int,
                      rng: np.random.Generator) -> tuple[np.ndarray, int, dict]:
    """Water-filling cap: largest per-type cap c with sum(min(n_t, c)) <= n; the
    remaining slots go (one each) to randomly chosen types that still have cells."""
    idx = np.where(eligible)[0]
    types = np.array(sorted(set(cell_types[idx])))
    counts = {t: int((cell_types[idx] == t).sum()) for t in types}
    if sum(counts.values()) < n:
        raise ValueError("not enough eligible cells")
    cap = 0
    while sum(min(v, cap + 1) for v in counts.values()) <= n:
        cap += 1
    take = {t: min(v, cap) for t, v in counts.items()}
    remainder = n - sum(take.values())
    open_types = [t for t in types if counts[t] > cap]
    if remainder > 0:
        extra = rng.choice(len(open_types), size=remainder, replace=False)
        for e in extra:
            take[open_types[e]] += 1
    chosen = []
    for t in types:
        pool = idx[cell_types[idx] == t]
        chosen.append(rng.choice(pool, size=take[t], replace=False))
    chosen = np.sort(np.concatenate(chosen))
    assert chosen.size == n
    return chosen, cap, {t: {"available": counts[t], "sampled": take[t]} for t in types}


def stage_prepare(args, out: Path, dicts: dict) -> dict:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    token_dict = load_pickle(dicts["token_dictionary"]["path"])
    median_dict = load_pickle(dicts["gene_median_dictionary"]["path"])
    name_dict = load_pickle(dicts["gene_name_id_dict"]["path"])
    for tok in ("<cls>", "<eos>", "<pad>", "<mask>"):
        assert tok in token_dict, f"{tok} missing from V2 token dictionary"
    cls_id, eos_id = int(token_dict["<cls>"]), int(token_dict["<eos>"])

    f = h5py.File(H5AD, "r")
    obs = f["obs"]
    obs_names = obs[obs.attrs["_index"]][:].astype(str)
    cell_type = read_categorical(obs, "cell_type")
    donor = read_categorical(obs, "donor_id")
    assay = read_categorical(obs, "assay")
    tissue = read_categorical(obs, "tissue")
    obs_total_counts = obs["total_counts"][:]
    var_ids = f["raw/var"][f["raw/var"].attrs["_index"]][:].astype(str)
    var_sym = read_categorical(f["raw/var"], "feature_name").astype(str)
    assert (var_ids == f["var"][f["var"].attrs["_index"]][:].astype(str)).all()
    n_obs, n_var = len(obs_names), len(var_ids)
    log(f"h5ad: {n_obs} cells x {n_var} genes; assay filter = {args.assay!r}")

    eligible = np.ones(n_obs, bool) if args.assay == "all" else (assay == args.assay)
    cells, cap, strata = stratified_sample(cell_type, eligible, args.n_cells, rng)
    log(f"sampled {cells.size} cells, per-type cap {cap}, {len(strata)} types")

    # raw integer counts (raw/X; adata.X is log-normalised in this file)
    Xraw = read_csr_rows(f["raw/X"], cells, n_var)
    f.close()
    Xd = Xraw.data
    assert np.all(np.abs(Xd - np.round(Xd)) < 1e-6), "raw/X is not integer counts"
    n_counts = np.asarray(Xraw.sum(axis=1)).ravel().astype(np.float64)
    assert (n_counts > 0).all()

    # tokenizable genes: var Ensembl IDs present in the V2 token dictionary (no
    # duplicates / version suffixes in this file, verified below)
    assert len(set(var_ids)) == len(var_ids)
    special = {"<pad>", "<mask>", "<cls>", "<eos>"}
    tok_cols = np.array([j for j, g in enumerate(var_ids) if g in token_dict and g not in special])
    tok_ens = var_ids[tok_cols]
    tok_ids = np.array([int(token_dict[g]) for g in tok_ens], dtype=np.int64)
    medians = np.array([float(median_dict[g]) for g in tok_ens], dtype=np.float64)
    assert (medians > 0).all()
    ens_to_sym_h5 = dict(zip(var_ids, var_sym))
    id_to_name = {}
    for k, v in name_dict.items():
        id_to_name.setdefault(v, k)
    tok_sym = np.array([ens_to_sym_h5.get(g) or id_to_name.get(g, g) for g in tok_ens], dtype=object)
    log(f"tokenizable genes in h5ad: {tok_cols.size} of {len(token_dict) - 4} dictionary genes")

    Xt = Xraw[:, tok_cols].toarray().astype(np.float64)            # cells x tokenizable
    cp10k = Xt / n_counts[:, None] * TARGET_SUM                     # normalised expression
    scaled = cp10k / medians[None, :]                               # median-scaled
    max_genes = MODEL_INPUT_SIZE - 2

    n_tok = tok_cols.size
    present = np.zeros((cells.size, n_tok), dtype=bool)
    rankpos = np.zeros((cells.size, n_tok), dtype=np.int32)         # 1-based, 0 = absent
    seqs, seq_lens, n_nonzero, truncated = [], [], [], []
    for c in range(cells.size):
        v = scaled[c]
        nz = np.nonzero(v)[0]
        order = nz[np.argsort(-v[nz], kind="stable")]
        n_nonzero.append(order.size)
        truncated.append(order.size > max_genes)
        order = order[:max_genes]
        present[c, order] = True
        rankpos[c, order] = np.arange(1, order.size + 1, dtype=np.int32)
        seq = np.concatenate([[cls_id], tok_ids[order], [eos_id]]).astype(np.int32)
        seqs.append(seq)
        seq_lens.append(seq.size)
    seq_lens = np.array(seq_lens)
    log(f"tokenised: seq len median {int(np.median(seq_lens))}, max {seq_lens.max()}, "
        f"truncated cells {int(np.sum(truncated))}")

    # ------------------------------------------------------------- gene set G
    freq = present.mean(axis=0)
    mean_cp10k = cp10k.mean(axis=0)
    order_all = np.lexsort((tok_ens, -mean_cp10k, -freq))          # freq desc, expr desc, id asc
    top = list(order_all[:args.n_top_genes])
    top_set = set(top)

    trrust = pd.read_csv(TRRUST, sep="\t", header=None, names=["tf", "target", "mode", "pmid"])
    doro = pd.read_csv(DOROTHEA, sep="\t")
    sym_to_ens_h5 = {}
    for g, s in zip(var_ids, var_sym):
        sym_to_ens_h5.setdefault(s, set()).add(g)

    map_stats = {"gene_name_id_dict": 0, "h5ad_feature_name": 0, "unmapped": 0}

    def sym2ens(s: str):
        if s in name_dict:
            map_stats["gene_name_id_dict"] += 1
            return name_dict[s]
        hits = sym_to_ens_h5.get(s)
        if hits and len(hits) == 1:
            map_stats["h5ad_feature_name"] += 1
            return next(iter(hits))
        map_stats["unmapped"] += 1
        return None

    all_syms = sorted(set(trrust.tf) | set(trrust.target) | set(doro.source) | set(doro.target))
    sym_map = {s: sym2ens(s) for s in all_syms}
    ens_to_tokcol = {g: i for i, g in enumerate(tok_ens)}
    trrust_tf_ens = {sym_map[s] for s in set(trrust.tf) if sym_map[s]}
    dorothea_tf_ens = {sym_map[s] for s in set(doro.source) if sym_map[s]}
    tf_cands = [ens_to_tokcol[g] for g in trrust_tf_ens
                if g in ens_to_tokcol and freq[ens_to_tokcol[g]] >= args.tf_min_presence]
    tf_extra = [i for i in tf_cands if i not in top_set]
    tf_extra = sorted(tf_extra, key=lambda i: (-freq[i], -mean_cp10k[i], tok_ens[i]))
    n_slots = args.max_genes - len(top)
    tf_added = tf_extra[:n_slots]
    tf_dropped = tf_extra[n_slots:]
    G_cols = np.array(top + tf_added)
    G_cols = G_cols[np.lexsort((tok_ens[G_cols], -mean_cp10k[G_cols], -freq[G_cols]))]
    nG = G_cols.size
    log(f"gene set G: {len(top)} top-frequency + {len(tf_added)} TRRUST TFs (>= "
        f"{args.tf_min_presence:.0%} presence) = {nG}; {len(tf_dropped)} TFs dropped by cap")

    presG = present[:, G_cols]
    rankG = rankpos[:, G_cols]
    logG = np.log1p(cp10k[:, G_cols])
    with np.errstate(invalid="ignore"):
        mean_rank = np.where(presG.any(0), (rankG * presG).sum(0) / np.maximum(presG.sum(0), 1), np.nan)

    genes = pd.DataFrame({
        "gene_index": np.arange(nG),
        "ensembl_id": tok_ens[G_cols],
        "symbol": tok_sym[G_cols],
        "token_id": tok_ids[G_cols],
        "is_tf": [g in trrust_tf_ens for g in tok_ens[G_cols]],
        "is_dorothea_tf": [g in dorothea_tf_ens for g in tok_ens[G_cols]],
        "selection": ["trrust_tf_ge30pct" if c in set(tf_added) else "top_frequency" for c in G_cols],
        "n_cells_present": presG.sum(0),
        "presence_frac": presG.mean(0),
        "mean_norm_expr": cp10k[:, G_cols].mean(0),
        "mean_log1p_norm_expr": logG.mean(0),
        "mean_median_scaled_expr": scaled[:, G_cols].mean(0),
        "mean_rank_when_present": mean_rank,
        "geneformer_median": medians[G_cols],
    })

    # ------------------------------------------------------------- co-expression
    pear = np.corrcoef(logG, rowvar=False)
    rk = np.apply_along_axis(rankdata, 0, logG)
    spear = np.corrcoef(rk, rowvar=False)
    assert pear.shape == (nG, nG) and not np.isnan(pear).any() and not np.isnan(spear).any()
    atomic_save_npy(out / "coexpr_pearson.npy", pear.astype(np.float32))
    atomic_save_npy(out / "coexpr_spearman.npy", spear.astype(np.float32))

    # ------------------------------------------------------------- reference edges
    ens_to_G = {g: i for i, g in enumerate(tok_ens[G_cols])}
    tr = trrust.copy()
    tr["tf_ensembl"] = tr.tf.map(sym_map)
    tr["target_ensembl"] = tr.target.map(sym_map)
    tr = tr[tr.tf_ensembl.isin(ens_to_G) & tr.target_ensembl.isin(ens_to_G)]
    tr_edges = (tr.groupby(["tf", "target", "tf_ensembl", "target_ensembl"], as_index=False)
                  .agg(mode=("mode", lambda s: ";".join(sorted(set(s)))),
                       pmids=("pmid", lambda s: ";".join(sorted({str(x) for x in s}))),
                       n_records=("mode", "size")))
    tr_edges.insert(4, "tf_index", tr_edges.tf_ensembl.map(ens_to_G).astype(int))
    tr_edges.insert(5, "target_index", tr_edges.target_ensembl.map(ens_to_G).astype(int))
    tr_edges["is_self"] = tr_edges.tf_index == tr_edges.target_index
    tr_edges = tr_edges.sort_values(["tf_index", "target_index"]).reset_index(drop=True)

    do = doro[doro.confidence.isin(DOROTHEA_LEVELS)].copy()
    do["source_ensembl"] = do.source.map(sym_map)
    do["target_ensembl"] = do.target.map(sym_map)
    do = do[do.source_ensembl.isin(ens_to_G) & do.target_ensembl.isin(ens_to_G)]
    do = (do.sort_values("confidence")
            .drop_duplicates(["source_ensembl", "target_ensembl"], keep="first"))
    do_edges = pd.DataFrame({
        "tf": do.source.values, "target": do.target.values,
        "tf_ensembl": do.source_ensembl.values, "target_ensembl": do.target_ensembl.values,
        "tf_index": do.source_ensembl.map(ens_to_G).astype(int).values,
        "target_index": do.target_ensembl.map(ens_to_G).astype(int).values,
        "confidence": do.confidence.values,
    })
    do_edges["is_self"] = do_edges.tf_index == do_edges.target_index
    do_edges = do_edges.sort_values(["tf_index", "target_index"]).reset_index(drop=True)

    outdeg = tr_edges[~tr_edges.is_self].groupby("tf_index").size()
    indeg = tr_edges[~tr_edges.is_self].groupby("target_index").size()
    genes["trrust_out_degree_in_G"] = genes.gene_index.map(outdeg).fillna(0).astype(int)
    genes["trrust_in_degree_in_G"] = genes.gene_index.map(indeg).fillna(0).astype(int)
    genes.to_csv(out / "genes.tsv", sep="\t", index=False, float_format="%.6g")
    tr_edges.to_csv(out / "trrust_edges.tsv", sep="\t", index=False)
    do_edges.to_csv(out / "dorothea_abc_edges.tsv", sep="\t", index=False)
    log(f"TRRUST edges in G: {len(tr_edges)} ({int(tr_edges.is_self.sum())} self); "
        f"DoRothEA A-C edges in G: {len(do_edges)}")

    # ------------------------------------------------------------- cells + tokens
    cells_df = pd.DataFrame({
        "cell_index": np.arange(cells.size),
        "obs_name": obs_names[cells],
        "h5ad_row": cells,
        "cell_type": cell_type[cells],
        "donor_id": donor[cells],
        "assay": assay[cells],
        "tissue": tissue[cells],
        "n_counts_raw_allgenes": n_counts.astype(np.int64),
        "obs_total_counts": obs_total_counts[cells],
        "n_tokenizable_nonzero": n_nonzero,
        "seq_len_with_special": seq_lens,
        "truncated": truncated,
        "n_G_present": presG.sum(1),
    })
    cells_df.to_csv(out / "cells.tsv", sep="\t", index=False)
    offsets = np.concatenate([[0], np.cumsum(seq_lens)]).astype(np.int64)
    # per-cell positions (token index in the sequence) of every gene in G present
    g_pos_rows, g_pos_idx, g_pos_off = [], [], [0]
    for c in range(cells.size):
        gi = np.nonzero(presG[c])[0]
        g_pos_idx.append(gi.astype(np.int32))
        g_pos_rows.append(rankG[c, gi].astype(np.int32))          # token position == rank
        g_pos_off.append(g_pos_off[-1] + gi.size)
    np.savez_compressed(
        out / "tokenized_cells.npz",
        input_ids=np.concatenate(seqs), offsets=offsets,
        g_index=np.concatenate(g_pos_idx), g_position=np.concatenate(g_pos_rows),
        g_offsets=np.asarray(g_pos_off, dtype=np.int64),
    )

    prep = {
        "assay_filter": args.assay,
        "n_eligible_cells": int(eligible.sum()),
        "per_type_cap": int(cap),
        "strata": strata,
        "sampled_obs_names_md5": md5_text("\n".join(obs_names[cells])),
        "n_tokenizable_genes_in_h5ad": int(n_tok),
        "cls_id": cls_id, "eos_id": eos_id,
        "seq_len": {"min": int(seq_lens.min()), "median": float(np.median(seq_lens)),
                    "max": int(seq_lens.max())},
        "n_truncated_cells": int(np.sum(truncated)),
        "gene_set": {
            "n_top_frequency": len(top), "n_tf_added": len(tf_added),
            "n_tf_dropped_by_cap": len(tf_dropped), "n_genes": int(nG),
            "n_is_tf": int(genes.is_tf.sum()),
            "min_presence_top_frequency": float(freq[top].min()),
        },
        "symbol_mapping": map_stats,
        "n_trrust_edges_in_G": int(len(tr_edges)),
        "n_trrust_self_edges_in_G": int(tr_edges.is_self.sum()),
        "n_dorothea_abc_edges_in_G": int(len(do_edges)),
        "n_counts_matches_obs_total_counts_frac": float(np.mean(np.isclose(n_counts, obs_total_counts[cells]))),
        "runtime_s": round(time.time() - t0, 1),
    }
    atomic_write_text(out / "prepare_summary.json", json.dumps(prep, indent=2, default=str))
    log(f"prepare done in {prep['runtime_s']} s")
    return prep


# --------------------------------------------------------------------- stage 2
def load_tokens(out: Path):
    z = np.load(out / "tokenized_cells.npz")
    return {k: z[k] for k in z.files}


def cell_view(tok: dict, c: int):
    ids = tok["input_ids"][tok["offsets"][c]:tok["offsets"][c + 1]]
    s, e = tok["g_offsets"][c], tok["g_offsets"][c + 1]
    return ids, tok["g_index"][s:e], tok["g_position"][s:e]


def pick_device(pref: str):
    import torch
    if pref == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if pref == "mps":
        log("MPS unavailable -> CPU")
    return torch.device("cpu")


def load_model(device):
    import torch
    from transformers import BertForMaskedLM
    model_dir = GF_HUB / MODEL_SNAPSHOT / MODEL_SUBDIR
    model = BertForMaskedLM.from_pretrained(str(model_dir), attn_implementation="eager",
                                            output_attentions=True)
    model.eval().to(device)
    torch.set_grad_enabled(False)
    return model


def manual_layer_attentions(bert, ids_t):
    """Yields (layer_index, attention_probs[heads, L, L]) running the encoder layer by
    layer, so that only one layer's attention tensor is held in memory."""
    h = bert.embeddings(input_ids=ids_t)
    for l, layer in enumerate(bert.encoder.layer):
        outs = layer(h, attention_mask=None, output_attentions=True)
        h = outs[0]
        yield l, outs[1][0]


def check_manual_equivalence(model, device, ids: np.ndarray, n_tok: int = 1024) -> float:
    import torch
    ids_c = np.concatenate([ids[: n_tok - 1], ids[-1:]])        # crop, keep <eos>
    ids_t = torch.tensor(ids_c, dtype=torch.long, device=device)[None]
    full = model(input_ids=ids_t, attention_mask=torch.ones_like(ids_t), output_attentions=True)
    diffs = []
    for l, A in manual_layer_attentions(model.bert, ids_t):
        diffs.append(float((A - full.attentions[l][0]).abs().max()))
    del full
    return max(diffs)


def stage_attention(args, out: Path) -> dict:
    import torch
    t0 = time.time()
    ckpt = out / "_checkpoint"
    ckpt.mkdir(exist_ok=True)
    genes = pd.read_csv(out / "genes.tsv", sep="\t")
    nG = len(genes)
    tok = load_tokens(out)
    n_cells = len(tok["offsets"]) - 1
    device = pick_device(args.device)
    model = load_model(device)
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    log(f"attention stage: {n_cells} cells, |G|={nG}, device={device}")

    state_file = ckpt / "state.json"
    if state_file.exists():
        state = json.loads(state_file.read_text())
        heads_sum = torch.from_numpy(np.load(ckpt / "heads_sum.npy")).to(device)
        layer_sum = np.load(ckpt / "layer_sum.npy")
        counts = np.load(ckpt / "counts.npy")
        rank_sum = np.load(ckpt / "rank_sum.npy")
        per_cell = state["per_cell"]
        state.setdefault("resumed_at", []).append({"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                   "n_done": state["n_done"]})
        log(f"resuming from checkpoint at cell {state['n_done']}")
    else:
        ids0, _, _ = cell_view(tok, 0)
        eq = check_manual_equivalence(model, device, ids0)
        log(f"manual layer loop vs full forward: max |dA| = {eq:.3e}")
        if eq > 1e-4:
            raise RuntimeError("manual layer loop does not reproduce model attentions")
        state = {"n_done": 0, "manual_vs_full_max_abs_diff": eq, "device": str(device),
                 "elapsed_s": 0.0}
        heads_sum = torch.zeros((n_layers, n_heads, nG, nG), dtype=torch.float32, device=device)
        layer_sum = np.zeros((n_layers, nG, nG), dtype=np.float64)
        counts = np.zeros((nG, nG), dtype=np.int64)
        rank_sum = np.zeros((nG, nG), dtype=np.int64)
        per_cell = []
    layer_chunk = torch.zeros((n_layers, nG, nG), dtype=torch.float32, device=device)
    elapsed_before = state.get("elapsed_s", 0.0)
    t_run = time.time()

    def save_checkpoint(n_done: int):
        nonlocal layer_chunk
        layer_sum[...] += layer_chunk.cpu().numpy().astype(np.float64)
        layer_chunk.zero_()
        atomic_save_npy(ckpt / "heads_sum.npy", heads_sum.cpu().numpy())
        atomic_save_npy(ckpt / "layer_sum.npy", layer_sum)
        atomic_save_npy(ckpt / "counts.npy", counts)
        atomic_save_npy(ckpt / "rank_sum.npy", rank_sum)
        state.update({"n_done": n_done, "per_cell": per_cell,
                      "elapsed_s": elapsed_before + (time.time() - t_run)})
        atomic_write_text(state_file, json.dumps(state))

    start = state["n_done"]
    stop = n_cells if args.max_cells is None else min(n_cells, args.max_cells)
    for c in range(start, stop):
        tc = time.time()
        ids, gi, gpos = cell_view(tok, c)
        k = gi.size
        ids_t = torch.tensor(ids.astype(np.int64), device=device)[None]
        pos_t = torch.tensor(gpos.astype(np.int64), device=device)
        gi64 = gi.astype(np.int64)
        lin = (gi64[:, None] * nG + gi64[None, :]).ravel()
        lin_t = torch.tensor(lin, device=device)
        max_dev = 0.0
        for l, A in manual_layer_attentions(model.bert, ids_t):
            dev = float((A.sum(-1) - 1.0).abs().max())
            max_dev = max(max_dev, dev)
            sub = A.index_select(1, pos_t).index_select(2, pos_t)      # heads x k x k
            heads_sum[l].view(n_heads, -1).index_add_(1, lin_t, sub.reshape(n_heads, -1))
            layer_chunk[l].view(-1).index_add_(0, lin_t, sub.mean(0).reshape(-1))
            del A, sub
        ix = np.ix_(gi64, gi64)
        counts[ix] += 1
        r = gpos.astype(np.int64)
        rank_sum[ix] += np.abs(r[:, None] - r[None, :])
        del ids_t, pos_t, lin_t
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        per_cell.append({"cell_index": c, "seq_len": int(ids.size), "n_G_present": int(k),
                         "max_rowsum_dev": max_dev, "sec": round(time.time() - tc, 3)})
        if max_dev > 1e-3:
            raise RuntimeError(f"cell {c}: attention rows do not sum to 1 (dev {max_dev})")
        done = c + 1
        if done % 25 == 0 and done % args.ckpt_every != 0:
            rate = (time.time() - t_run) / max(done - start, 1)
            mem = torch.mps.driver_allocated_memory() / 2**30 if device.type == "mps" else float("nan")
            log(f"progress {done}/{n_cells}  {rate:.2f} s/cell  mps driver mem {mem:.1f} GiB")
        if done % args.ckpt_every == 0 or done == stop:
            save_checkpoint(done)
            rate = (time.time() - t_run) / max(done - start, 1)
            log(f"cells {done}/{n_cells}  {rate:.2f} s/cell  eta {(stop - done) * rate / 60:.1f} min  "
                f"max rowsum dev so far {max(p['max_rowsum_dev'] for p in per_cell):.2e}")
    state["attention_stage_s_this_run"] = round(time.time() - t0, 1)
    atomic_write_text(state_file, json.dumps(state))
    return state


# --------------------------------------------------------------------- stage 3
def stage_finalize(args, out: Path, dicts: dict, prep: dict, t_start: float) -> dict:
    t0 = time.time()
    ckpt = out / "_checkpoint"
    state = json.loads((ckpt / "state.json").read_text())
    genes = pd.read_csv(out / "genes.tsv", sep="\t")
    cells_df = pd.read_csv(out / "cells.tsv", sep="\t")
    nG, n_cells = len(genes), len(cells_df)
    if state["n_done"] != n_cells:
        raise RuntimeError(f"attention stage incomplete: {state['n_done']}/{n_cells}")
    counts = np.load(ckpt / "counts.npy")
    layer_sum = np.load(ckpt / "layer_sum.npy")
    rank_sum = np.load(ckpt / "rank_sum.npy")
    checks = {}

    # co-presence consistency with tokenisation
    checks["diag_counts_equal_presence"] = bool(np.array_equal(np.diag(counts), genes.n_cells_present.values))
    checks["counts_symmetric"] = bool(np.array_equal(counts, counts.T))
    pos = counts > 0
    denom = np.where(pos, counts, 1).astype(np.float64)

    layer_mean = np.where(pos[None], layer_sum / denom[None], np.nan).astype(np.float32)
    del layer_sum
    rank_mean = np.where(pos, rank_sum / denom, np.nan).astype(np.float32)
    checks["rank_distance_symmetric"] = bool(np.allclose(np.nan_to_num(rank_mean), np.nan_to_num(rank_mean.T)))

    heads_sum = np.load(ckpt / "heads_sum.npy", mmap_mode="r")
    n_layers, n_heads = heads_sum.shape[:2]
    tmp16 = out / "attention_heads.npy.tmp.npy"
    tmp32 = out / "attention_heads_f32.npy.tmp.npy"
    heads_mean16 = np.lib.format.open_memmap(tmp16, mode="w+", dtype=np.float16, shape=heads_sum.shape)
    heads_mean32 = np.lib.format.open_memmap(tmp32, mode="w+", dtype=np.float32, shape=heads_sum.shape)
    max_layer_vs_headavg = 0.0
    f16_rel_err_normal, f16_abs_err, f16_below_normal, f16_zero, f16_total = 0.0, 0.0, 0, 0, 0
    tiny16 = float(np.finfo(np.float16).tiny)
    for l in range(n_layers):
        hm = np.where(pos[None], heads_sum[l] / denom[None], np.nan).astype(np.float32)
        max_layer_vs_headavg = max(max_layer_vs_headavg,
                                   float(np.nanmax(np.abs(hm.mean(0) - layer_mean[l]))))
        h16 = hm.astype(np.float16)
        heads_mean32[l] = hm
        heads_mean16[l] = h16
        v = hm[:, pos]
        v16 = h16[:, pos].astype(np.float32)
        nm = v >= tiny16
        f16_rel_err_normal = max(f16_rel_err_normal, float(np.max(np.abs(v16[nm] - v[nm]) / v[nm])))
        f16_abs_err = max(f16_abs_err, float(np.max(np.abs(v16 - v))))
        f16_below_normal += int(np.sum(~nm))
        f16_zero += int(np.sum((v16 == 0) & (v > 0)))
        f16_total += v.size
        del hm, h16, v, v16
    heads_mean16.flush(); heads_mean32.flush()
    checks["layer_mean_vs_mean_of_head_means_max_abs_diff"] = max_layer_vs_headavg
    checks["float16_heads_max_relative_error_normal_range"] = f16_rel_err_normal
    checks["float16_heads_max_abs_error"] = f16_abs_err
    checks["float16_heads_frac_below_normal_min_6.1e-5"] = f16_below_normal / max(f16_total, 1)
    checks["float16_heads_frac_flushed_to_zero"] = f16_zero / max(f16_total, 1)

    checks["no_nan_where_counts_pos_layer_mean"] = bool(not np.isnan(layer_mean[:, pos]).any())
    checks["no_nan_where_counts_pos_heads"] = bool(not any(np.isnan(heads_mean16[l][:, pos]).any()
                                                           for l in range(n_layers)))
    checks["no_nan_where_counts_pos_rank"] = bool(not np.isnan(rank_mean[pos]).any())
    checks["n_pairs_counts_zero"] = int((~pos).sum())
    checks["attention_values_in_0_1"] = bool(np.nanmin(layer_mean) >= 0 and np.nanmax(layer_mean) <= 1)
    per_cell = pd.DataFrame(state["per_cell"])
    checks["max_rowsum_dev_all_cells_layers"] = float(per_cell.max_rowsum_dev.max())
    checks["rowsum_check_passed_tol_1e-3"] = bool(per_cell.max_rowsum_dev.max() < 1e-3)
    checks["manual_layer_loop_vs_full_forward_max_abs_diff"] = state["manual_vs_full_max_abs_diff"]
    checks["per_cell_seq_len_matches_cells_tsv"] = bool(
        np.array_equal(per_cell.seq_len.values, cells_df.seq_len_with_special.values))
    checks["per_cell_nG_matches_cells_tsv"] = bool(
        np.array_equal(per_cell.n_G_present.values, cells_df.n_G_present.values))

    shapes = {
        "attention_layer_mean": list(layer_mean.shape),
        "attention_heads": list(heads_mean16.shape),
        "attention_heads_f32": list(heads_mean32.shape),
        "copresence_counts": list(counts.shape),
        "rank_distance_mean": list(rank_mean.shape),
    }
    assert layer_mean.shape == (n_layers, nG, nG) and heads_mean16.shape == (n_layers, n_heads, nG, nG)
    hard = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    if hard:
        raise RuntimeError(f"sanity checks failed: {hard}")

    atomic_save_npy(out / "attention_layer_mean.npy", layer_mean)
    del heads_mean16, heads_mean32
    os.replace(tmp16, out / "attention_heads.npy")
    os.replace(tmp32, out / "attention_heads_f32.npy")
    atomic_save_npy(out / "copresence_counts.npy", counts.astype(np.int32))
    atomic_save_npy(out / "rank_distance_mean.npy", rank_mean)
    pd.DataFrame(state["per_cell"]).to_csv(out / "attention_per_cell_log.tsv", sep="\t", index=False)

    model_dir = GF_HUB / MODEL_SNAPSHOT / MODEL_SUBDIR
    files = {}
    for p in sorted(out.iterdir()):
        if p.is_file() and not p.name.startswith(("kit_manifest", "README", ".")) and ".tmp" not in p.name:
            files[p.name] = {"bytes": p.stat().st_size, "md5": md5_file(p)}
    manifest = {
        "kit": "Geneformer V2-104M attention case-study kit (MI-Workbench)",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "builder_script": str(SCRIPT_PATH),
        "builder_script_md5": md5_file(SCRIPT_PATH),
        "workbench_git_head": git_head(WORKBENCH),
        "model": {
            "name": "ctheodoris/Geneformer " + MODEL_SUBDIR,
            "snapshot_commit_dir": MODEL_SNAPSHOT,
            "path": str(model_dir),
            "config_md5": md5_file(model_dir / "config.json"),
            "weights_md5": md5_file(model_dir / "model.safetensors"),
            "attn_implementation": "eager",
            "n_layers": int(n_layers), "n_heads": int(n_heads),
        },
        "dictionaries": dicts,
        "data": {
            "h5ad": str(H5AD), "h5ad_bytes": H5AD.stat().st_size,
            "h5ad_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(H5AD.stat().st_mtime)),
            "count_matrix_used": "raw/X (integer UMI counts; adata.X is log-normalised)",
            "sampled_obs_names_md5": prep["sampled_obs_names_md5"],
            "trrust": {"path": str(TRRUST), "md5": md5_file(TRRUST)},
            "dorothea": {"path": str(DOROTHEA), "md5": md5_file(DOROTHEA)},
        },
        "design": {
            "seed": SEED, "n_cells": int(n_cells), "assay_filter": prep["assay_filter"],
            "stratification": "cell_type, water-filling per-type cap",
            "per_type_cap": prep["per_type_cap"],
            "target_sum": TARGET_SUM, "model_input_size": MODEL_INPUT_SIZE,
            "special_tokens": "<cls> prepended, <eos> appended (V2 convention)",
            "n_top_genes": args.n_top_genes, "tf_min_presence": args.tf_min_presence,
            "max_genes": args.max_genes, "dorothea_levels": list(DOROTHEA_LEVELS),
            "per_head_accumulator": "float32 on device over all cells",
            "pytorch_mps_high_watermark_ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO"),
            "pytorch_mps_low_watermark_ratio": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO"),
            "layer_mean_accumulator": "float32 on device per <= %d-cell chunk, flushed into float64 on CPU" % args.ckpt_every,
        },
        "prepare": prep,
        "counts": {
            "n_cells": int(n_cells), "n_genes_G": int(nG),
            "n_tf_in_G": int(genes.is_tf.sum()),
            "n_trrust_edges_in_G": prep["n_trrust_edges_in_G"],
            "n_dorothea_abc_edges_in_G": prep["n_dorothea_abc_edges_in_G"],
            "n_pairs_copresence_ge50": int((counts >= 50).sum() - (np.diag(counts) >= 50).sum()),
        },
        "shapes": shapes,
        "dtypes": {"attention_layer_mean": "float32", "attention_heads": "float16", "attention_heads_f32": "float32",
                   "copresence_counts": "int32", "rank_distance_mean": "float32",
                   "coexpr_pearson": "float32", "coexpr_spearman": "float32"},
        "sanity_checks": checks,
        "runtime": {
            "device": state["device"],
            "attention_forward_s": round(state["elapsed_s"], 1),
            "attention_resumed_at": state.get("resumed_at", []),
            "sec_per_cell_median": float(per_cell.sec.median()),
            "prepare_s": prep["runtime_s"],
            "finalize_s": round(time.time() - t0, 1),
            "this_invocation_total_s": round(time.time() - t_start, 1),
        },
        "packages": pkg_versions(),
        "files": files,
    }
    atomic_write_text(out / "kit_manifest.json", json.dumps(manifest, indent=2, default=str))
    atomic_write_text(out / "README.md", readme_text(manifest))
    log(f"finalize done; sanity checks: {checks}")
    return manifest


def readme_text(m: dict) -> str:
    c, s, d = m["counts"], m["shapes"], m["design"]
    L, H, G = s["attention_heads"][0], s["attention_heads"][1], c["n_genes_G"]
    return f"""# Geneformer V2-104M attention case-study kit

Built by `{m['builder_script']}` (md5 `{m['builder_script_md5']}`) on {m['created']}.
Provenance, md5 sums, package versions, sanity checks and runtimes: `kit_manifest.json`.

Rebuild (resumable; stages whose outputs exist are skipped, `--force` rebuilds):

```bash
export TMPDIR="/Volumes/Crucial X6/tmp_miw"; export OMP_NUM_THREADS=4
/Users/ihorkendiukhov/anaconda3/envs/subproject02-evalbias-rev/bin/python \\
    automation/mi-workbench/experiments/case_study/build_kit.py
```

## Inputs
- Model: `{m['model']['path']}` (HF snapshot `{m['model']['snapshot_commit_dir']}`), BertForMaskedLM,
  {L} layers x {H} heads, eager attention, eval mode (no dropout).
- Dictionaries (gc104M; identical md5 across all HF snapshots holding them): see manifest.
- Data: `{m['data']['h5ad']}`; counts from `raw/X` (integer UMI counts; `X` in this file is log-normalised).
- Reference networks: TRRUST human (TF, target, mode, PMID) and DoRothEA human (confidence A-C kept).

## Design
- Cells: {c['n_cells']} cells with assay == `{d['assay_filter']}` (single UMI chemistry, so that co-expression is
  not driven by mixing Smart-seq2 read counts with 10x UMIs), stratified by `cell_type` with a water-filling cap
  of {d['per_type_cap']} cells per type (types with fewer cells contribute all of them; leftover slots go to randomly
  chosen larger types), seed {d['seed']}.
- Tokenisation (Geneformer V2 transcriptome tokenizer convention): per cell, counts / n_counts x {d['target_sum']}
  (n_counts = total raw counts over all 60,606 genes), divided by the gene's gc104M non-zero median; genes in the
  token dictionary with non-zero value are sorted descending (stable sort), truncated to {d['model_input_size'] - 2}
  genes, then `<cls>` (id 2) is prepended and `<eos>` (id 3) appended (max {d['model_input_size']} tokens).
  Ensembl IDs are used as-is (no version suffixes or duplicates exist in this file; the Geneformer Ensembl
  collapsing map is therefore the identity here).
- Gene rank r = token position in the sequence (1 = most highly ranked gene; `<cls>` is position 0).
- Gene set G ({G} genes): the {d['n_top_genes']} genes most frequently present in the 1,000 tokenised sequences (ties by
  mean normalised expression, then Ensembl ID), plus every TRRUST TF present in >= {d['tf_min_presence']:.0%} of cells, capped at
  {d['max_genes']} genes. Ordered by presence frequency (descending). Index i in every matrix = row i of `genes.tsv`.

## Files
All matrices are indexed by `gene_index` from `genes.tsv` (0-based). "Co-present in cell c" means both genes
appear in cell c's (truncated) token sequence.

| file | dtype / shape | definition |
|---|---|---|
| `genes.tsv` | {G} rows | `gene_index`, `ensembl_id`, `symbol`, `token_id`, `is_tf` (TRRUST source), `is_dorothea_tf`, `selection` (`top_frequency` or `trrust_tf_ge30pct`), `n_cells_present`, `presence_frac`, `mean_norm_expr` (mean counts/n_counts x 1e4 over all cells, zeros included), `mean_log1p_norm_expr`, `mean_median_scaled_expr`, `mean_rank_when_present` (mean token position over cells where present), `geneformer_median`, `trrust_out_degree_in_G`, `trrust_in_degree_in_G` (non-self TRRUST edges with both ends in G) |
| `cells.tsv` | {c['n_cells']} rows | `cell_index`, `obs_name`, `h5ad_row`, `cell_type`, `donor_id`, `assay`, `tissue`, `n_counts_raw_allgenes`, `obs_total_counts`, `n_tokenizable_nonzero` (before truncation), `seq_len_with_special`, `truncated`, `n_G_present` |
| `tokenized_cells.npz` | ragged | `input_ids` (concatenated sequences incl. special tokens), `offsets` (cell c = `input_ids[offsets[c]:offsets[c+1]]`), `g_index` / `g_position` / `g_offsets` (for cell c, the G genes present and their token positions) |
| `attention_layer_mean.npy` | float32 {s['attention_layer_mean']} | `A[l, i, j]` = mean attention weight from **query gene i** to **key gene j** in layer l, averaged over the {H} heads and over the cells where genes i and j are co-present. Rows are **not** renormalised (each full attention row sums to 1 over all tokens of that cell, including `<cls>`, `<eos>` and genes outside G). NaN where `copresence_counts == 0`. Diagonal = self-attention. |
| `attention_heads.npy` | float16 {s['attention_heads']} | `A[l, h, i, j]` = same as above for head h (no head averaging). float16: see manifest for quantisation error; values below 6.1e-5 are float16 subnormals with reduced relative precision. |
| `attention_heads_f32.npy` | float32 {s['attention_heads_f32']} | identical definition to `attention_heads.npy`, stored in float32 (added because a large fraction of per-head means fall below the float16 normal range; see manifest `sanity_checks`). Use this file for per-head rank statistics. |
| `copresence_counts.npy` | int32 {s['copresence_counts']} | `C[i, j]` = number of cells in which genes i and j are co-present (symmetric; diagonal = number of cells where gene i is present). |
| `rank_distance_mean.npy` | float32 {s['rank_distance_mean']} | mean over co-present cells of \\|r_i - r_j\\| (token-position distance). NaN where C == 0. |
| `coexpr_pearson.npy` | float32 ({G}, {G}) | Pearson correlation across the {c['n_cells']} cells of log1p(counts/n_counts x 1e4) (zeros included). |
| `coexpr_spearman.npy` | float32 ({G}, {G}) | Spearman correlation of the same vectors (average ranks for ties). |
| `trrust_edges.tsv` | {c['n_trrust_edges_in_G']} rows | unique TRRUST (TF -> target) pairs with both genes in G: symbols, Ensembl IDs, `tf_index`, `target_index`, `mode` (`;`-joined unique modes), `pmids`, `n_records`, `is_self` (autoregulation). |
| `dorothea_abc_edges.tsv` | {c['n_dorothea_abc_edges_in_G']} rows | unique DoRothEA (TF -> target) pairs with confidence A, B or C and both genes in G (best confidence kept), with indices and `is_self`. |
| `attention_per_cell_log.tsv` | {c['n_cells']} rows | per-cell sequence length, number of G genes present, max over layers/heads/rows of \\|sum_j A - 1\\| (row-sum sanity check before subsetting), seconds. |
| `prepare_summary.json` | | sampling strata, symbol mapping statistics, gene-set construction counts. |
| `_checkpoint/` | | raw accumulators (sums) used to resume the attention stage; not needed downstream. |

Symbol -> Ensembl mapping for TRRUST/DoRothEA uses `gene_name_id_dict_gc104M.pkl` first, then a unique match on the
h5ad `feature_name`.

## Accumulation
Per cell and layer, the attention tensor (heads x L x L) is sliced to the rows and columns of the G genes present.
Per-head sums are accumulated in float32 on the device over all cells; the head-averaged per-layer matrix is
accumulated in float32 per chunk of <= 100 cells and flushed into a float64 CPU accumulator. Means = sums / C.
"""


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-cells", type=int, default=N_CELLS)
    ap.add_argument("--assay", default=DEFAULT_ASSAY, help="assay filter; 'all' disables")
    ap.add_argument("--n-top-genes", type=int, default=N_TOP_GENES)
    ap.add_argument("--tf-min-presence", type=float, default=TF_MIN_PRESENCE)
    ap.add_argument("--max-genes", type=int, default=MAX_GENES)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu"])
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--max-cells", type=int, default=None, help="stop the attention stage early (smoke test)")
    ap.add_argument("--stage", default="all", choices=["all", "prepare", "attention", "finalize"])
    ap.add_argument("--force", action="store_true", help="rebuild all stages from scratch")
    args = ap.parse_args()
    t_start = time.time()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if "TMPDIR" not in os.environ or not os.environ["TMPDIR"].startswith("/Volumes/"):
        log("WARNING: TMPDIR is not on the external volume (MPS shader cache may fill the system disk)")
    if args.force:
        for p in list(out.glob("*")) + list((out / "_checkpoint").glob("*")):
            if p.is_file():
                p.unlink()
    dicts = resolve_dict_files()

    prep_file = out / "prepare_summary.json"
    if args.stage in ("all", "prepare") and not prep_file.exists():
        prep = stage_prepare(args, out, dicts)
    else:
        prep = json.loads(prep_file.read_text())
        log("prepare: outputs exist, skipped")
    if args.stage == "prepare":
        return
    if args.stage in ("all", "attention"):
        st = out / "_checkpoint" / "state.json"
        n_cells = len(pd.read_csv(out / "cells.tsv", sep="\t"))
        if not st.exists() or json.loads(st.read_text())["n_done"] < n_cells:
            stage_attention(args, out)
        else:
            log("attention: complete, skipped")
        if args.max_cells is not None:
            log("--max-cells given: stopping before finalize")
            return
    if args.stage in ("all", "finalize"):
        stage_finalize(args, out, dicts, prep, t_start)


if __name__ == "__main__":
    main()
