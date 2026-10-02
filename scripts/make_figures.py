#!/usr/bin/env python
"""PNG previews of the orchestration results (X6) from experiments/orchestration/*.json.

Reads:

- ``overhead.json`` from ``scripts/run_experiments.py``: control-plane
  overhead through the runner with a mock backend, one result per storage
  location;
- ``real_concurrency.json`` from ``scripts/real_concurrency.py``: concurrent
  Codex reviewer panels;
- ``mock_verification.json``: printed as a check summary only, no figure.

Writes to ``experiments/orchestration/figures/``:

- ``overhead.png``, in three panels:
  - (a) orchestration overhead per iteration, conservative definition
    (run wall time minus iterations x mock delay, so event-loop stalls count
    as overhead);
  - (b) throughput;
  - (c) overhead as a percentage of the median real reviewer-call latency.
- ``real_concurrency.png``, in three panels:
  - (a) panel wall time;
  - (b) per-call latency;
  - (c) throughput against ideal linear scaling.
- ``derived_summary.json``: the numbers the text quotes, such as overhead
  relative to real call latency.

These are previews only. The final PLOS TIFFs are produced later. matplotlib
is needed only for plotting; use the analysis env:

    /Users/ihorkendiukhov/anaconda3/bin/python scripts/make_figures.py
    python scripts/make_figures.py --in-dir experiments/orchestration --out-dir /tmp/figs
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IN = REPO_ROOT / "experiments" / "orchestration"

# Reference categorical slots 1-3 (validated all-pairs, light surface); ink tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def load(path: Path) -> Optional[dict]:
    return json.loads(path.read_text()) if path.is_file() else None


# ── data preparation (pure; tested without matplotlib) ──────────────────


def overhead_series(overhead: dict) -> list[dict]:
    """One series per (storage, mock delay): arrays over concurrency levels."""
    out = []
    results = overhead.get("results") or [overhead]
    for res in results:
        st = res.get("storage", {}) or {}
        label = st.get("label") or "default"
        fs = st.get("fs_type") or ""
        for sw in res.get("sweeps", []):
            rows = sw["rows"]
            out.append({
                "storage": label,
                "fs_type": fs,
                "delay": sw["adapter_delay_s"],
                "concurrency": [r["concurrency"] for r in rows],
                "overhead_p50": [r["overhead_ms_per_iteration"]["p50"] for r in rows],
                "overhead_p95": [r["overhead_ms_per_iteration"]["p95"] for r in rows],
                "overhead_upper_p50": [r["overhead_upper_ms_per_iteration"]["p50"] for r in rows],
                "overhead_upper_p95": [r["overhead_upper_ms_per_iteration"]["p95"] for r in rows],
                "throughput": [r["throughput_iters_per_s"]["mean"] for r in rows],
                "latency_p50": [r["latency_s"]["p50"] for r in rows],
                "latency_p95": [r["latency_s"]["p95"] for r in rows],
                "cpu_ms": [r["cpu_ms_per_iteration"]["mean"] for r in rows],
                "frac_p50": [r["overhead_fraction_of_run_wall"]["p50"] for r in rows],
                "integrity_ok": all(r["integrity_ok"] for r in rows),
                "loadavg": [r.get("loadavg_start") for r in rows],
            })
    return out


def real_call_latency(rc: Optional[dict]) -> Optional[float]:
    """Median latency (s) of successful reviewer calls at K=1 (unloaded provider)."""
    if not rc:
        return None
    for s in rc.get("summary_by_level", []):
        if s["k"] == 1 and s.get("call_latency_s_successful", {}).get("p50"):
            return float(s["call_latency_s_successful"]["p50"])
    return None


def concurrency_points(rc: dict) -> dict[str, Any]:
    levels = sorted(rc.get("levels", []), key=lambda lv: lv["k"])
    ks = [lv["k"] for lv in levels]
    return {
        "k": ks,
        "panel_walls": [[p["wall_seconds"] for p in lv["panels"]] for lv in levels],
        "call_ok": [[c["latency_s"] for c in lv["calls"] if c["success"]] for lv in levels],
        "call_failed": [[c["latency_s"] for c in lv["calls"] if not c["success"]] for lv in levels],
        "summary": [lv["summary"] for lv in levels],
    }


def derived_summary(overhead: Optional[dict], rc: Optional[dict]) -> dict:
    lat = real_call_latency(rc)
    out: dict[str, Any] = {"real_call_latency_p50_s_k1": lat}
    if overhead:
        rows = []
        for s in overhead_series(overhead):
            for i, c in enumerate(s["concurrency"]):
                row = {"storage": s["storage"], "fs_type": s["fs_type"],
                       "mock_delay_s": s["delay"], "concurrency": c,
                       "overhead_ms_p50": s["overhead_p50"][i],
                       "overhead_ms_p95": s["overhead_p95"][i],
                       "overhead_upper_ms_p50": s["overhead_upper_p50"][i],
                       "overhead_upper_ms_p95": s["overhead_upper_p95"][i],
                       "throughput_iters_per_s": s["throughput"][i],
                       "cpu_ms_per_iteration": s["cpu_ms"][i]}
                if lat:
                    for key in ("overhead_upper_ms_p50", "overhead_upper_ms_p95"):
                        row[key.replace("_ms_", "_pct_of_real_call_")] = round(
                            100.0 * row[key] / 1000.0 / lat, 3)
                rows.append(row)
        out["overhead"] = rows
        out["integrity_ok"] = bool(overhead.get("integrity_ok"))
    if rc:
        out["real_concurrency"] = [
            {k: s.get(k) for k in (
                "k", "wall_seconds", "panels_complete", "panels_partial", "panels_failed",
                "attempts", "retries", "failed_attempts", "rate_limit_errors",
                "throughput_panels_per_min", "throughput_successful_calls_per_min",
                "panel_wall_p50_vs_k1", "call_latency_p50_vs_k1")}
            | {"call_latency_p50_s": s.get("call_latency_s_successful", {}).get("p50"),
               "call_latency_p95_s": s.get("call_latency_s_successful", {}).get("p95"),
               "panel_wall_p50_s": s.get("panel_wall_s", {}).get("p50")}
            for s in rc.get("summary_by_level", [])
        ]
        out["real_model"] = rc.get("config", {}).get("model")
        out["real_effort"] = rc.get("config", {}).get("effort")
    return out


# ── plotting ────────────────────────────────────────────────────────────


def _style(plt) -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "lines.linewidth": 1.6,
    })


def _storage_colors(series: list[dict]) -> dict[str, str]:
    order: list[str] = []
    for s in series:
        if s["storage"] not in order:
            order.append(s["storage"])
    return {name: SERIES[i % len(SERIES)] for i, name in enumerate(order)}


def _storage_name(s: dict) -> str:
    return f"{s['storage']} ({s['fs_type']})" if s["fs_type"] else s["storage"]


def plot_overhead(overhead: dict, rc: Optional[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    series = overhead_series(overhead)
    colors = _storage_colors(series)
    lat = real_call_latency(rc)
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))
    for s in series:
        col = colors[s["storage"]]
        ls = "-" if s["delay"] == 0 else "--"
        mk = "o" if s["delay"] == 0 else "s"
        lab = f"{_storage_name(s)}, mock delay {s['delay']:g} s"
        x = s["concurrency"]
        y50, y95 = s["overhead_upper_p50"], s["overhead_upper_p95"]
        hi = [max(0.0, b - a) for a, b in zip(y50, y95)]
        axes[0].errorbar(x, y50, yerr=[[0] * len(hi), hi], color=col, ls=ls,
                         marker=mk, ms=4, capsize=2, elinewidth=0.8, label=lab)
        axes[1].plot(x, s["throughput"], color=col, ls=ls, marker=mk, ms=4, label=lab)
        if lat:
            pct = [v / 1000.0 / lat * 100.0 for v in y50]
            axes[2].plot(x, pct, color=col, ls=ls, marker=mk, ms=4, label=lab)
    axes[0].set_title("(a) Orchestration overhead")
    axes[0].set_ylabel("ms per iteration, p50 (bar: p95)")
    axes[1].set_title("(b) Throughput")
    axes[1].set_ylabel("iterations / s")
    axes[1].set_yscale("log")
    if lat:
        axes[2].set_title("(c) Relative to a real reviewer call")
        axes[2].set_ylabel(f"% of median call latency ({lat:.0f} s)")
    else:
        axes[2].set_visible(False)
    for ax in axes:
        ax.set_xlabel("concurrent runs")
        ax.set_xscale("log")
        ticks = sorted({c for s in series for c in s["concurrency"]})
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
    axes[0].set_ylim(bottom=0)
    if lat:
        axes[2].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_real_concurrency(rc: dict, out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)
    pts = concurrency_points(rc)
    ks = pts["k"]
    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.6))

    def strip(ax, groups, color, marker="o", label=None):
        first = True
        for k, vals in zip(ks, groups):
            n = len(vals)
            for i, v in enumerate(sorted(vals)):
                # multiplicative spread (+-12 %) so the strip width is constant on a log axis
                f = (1.0 + 0.24 * (i / (n - 1) - 0.5)) if n > 1 else 1.0
                ax.plot(k * f, v, marker=marker, ms=3.5, ls="none", color=color,
                        alpha=0.8, label=label if first else None)
                first = False

    strip(axes[0], pts["panel_walls"], SERIES[0], label="panel")
    axes[0].plot(ks, [median(v) if v else float("nan") for v in pts["panel_walls"]],
                 color=INK, lw=1.2, label="median")
    axes[0].set_title("(a) Panel wall time")
    axes[0].set_ylabel("seconds")

    strip(axes[1], pts["call_ok"], SERIES[0], label="successful call")
    if any(pts["call_failed"]):
        strip(axes[1], pts["call_failed"], SERIES[1], marker="x", label="failed attempt")
    axes[1].plot(ks, [median(v) if v else float("nan") for v in pts["call_ok"]],
                 color=INK, lw=1.2, label="median")
    axes[1].set_title("(b) Reviewer-call latency")
    axes[1].set_ylabel("seconds")

    thr = [s["throughput_successful_calls_per_min"] for s in pts["summary"]]
    axes[2].plot(ks, thr, color=SERIES[0], marker="o", ms=4, label="measured")
    if ks and ks[0] == 1 and thr[0]:
        axes[2].plot(ks, [thr[0] * k for k in ks], color=MUTED, ls="--", lw=1.0,
                     label="linear scaling")
    for k, s, y in zip(ks, pts["summary"], thr):
        note = f"r={s['retries']} f={s['failed_lens_calls_final']}"
        axes[2].annotate(note, (k, y), textcoords="offset points", xytext=(0, -11),
                         ha="center", fontsize=6.5, color=INK_2)
    axes[2].set_title("(c) Throughput")
    axes[2].set_ylabel("successful calls / min")
    for ax in axes:
        ax.set_xlabel("concurrent panels (3 calls each)")
        ax.set_xscale("log")
        ax.set_xticks(ks)
        ax.set_xticklabels([str(k) for k in ks])
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left")
    cfg = rc.get("config", {})
    fig.suptitle(f"{cfg.get('model', '')} (effort {cfg.get('effort', '')}), "
                 f"artifact {Path(cfg.get('artifact', '')).name}; "
                 "r = retries, f = failed lens calls", fontsize=7, color=INK_2, x=0.01,
                 ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-dir", default=str(DEFAULT_IN))
    ap.add_argument("--out-dir", default=None, help="default: <in-dir>/figures")
    args = ap.parse_args(argv)
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    overhead = load(in_dir / "overhead.json")
    rc = load(in_dir / "real_concurrency.json")
    mock = load(in_dir / "mock_verification.json")

    if mock:
        ms = mock["merge_semantics"]
        print(f"mock verification: all_passed={ms['all_passed']} checks={ms['checks']}")
    if overhead:
        plot_overhead(overhead, rc, out_dir / "overhead.png")
        print(f"Wrote {out_dir / 'overhead.png'}")
    else:
        print("overhead.json missing; skipped")
    if rc:
        plot_real_concurrency(rc, out_dir / "real_concurrency.png")
        print(f"Wrote {out_dir / 'real_concurrency.png'}")
    else:
        print("real_concurrency.json missing; skipped")
    summary = derived_summary(overhead, rc)
    (out_dir / "derived_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {out_dir / 'derived_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
