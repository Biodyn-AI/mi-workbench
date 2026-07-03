#!/usr/bin/env python
"""Generate manuscript figures from the experiment results JSON.

Usage:
    python scripts/make_figures.py [results.json] [out_dir]

Writes Fig2_scaling.pdf (orchestrator load scaling). PLOS requires figures as
separate files; export the PDF to TIFF/EPS at >=300 dpi for submission.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

INK = "#1b1b1b"
ACCENT = "#2b6cb0"


def main() -> None:
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/plos/data/results.json")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("paper/plos")
    results = json.loads(results_path.read_text())

    rows = results["load_benchmark"]["rows"]
    conc = [r["concurrency"] for r in rows]
    thr = [r["throughput_iters_per_s"] for r in rows]
    p95 = [r["latency_p95_s"] * 1000 for r in rows]  # ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.6))

    ax1.plot(conc, thr, "o-", color=ACCENT, lw=1.6, ms=5)
    ax1.set_xlabel("Concurrent runs")
    ax1.set_ylabel("Throughput (iterations/s)")
    ax1.set_title("(a) Throughput scaling", fontsize=9, color=INK)
    ax1.set_ylim(bottom=0)
    ax1.grid(True, axis="y", alpha=0.25)

    ax2.plot(conc, p95, "s-", color=INK, lw=1.6, ms=5)
    ax2.set_xlabel("Concurrent runs")
    ax2.set_ylabel("p95 per-run latency (ms)")
    ax2.set_title("(b) Tail latency", fontsize=9, color=INK)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    out = out_dir / "Fig2_scaling.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
