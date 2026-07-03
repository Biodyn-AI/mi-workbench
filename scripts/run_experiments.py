#!/usr/bin/env python
"""Run the MI-Workbench evaluation harness and write results to JSON.

Usage:
    python scripts/run_experiments.py [output.json]

Deterministic except for the wall-clock figures in the load benchmark.
"""
import json
import sys
from pathlib import Path

# Ensure the repo root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.analysis.experiments import run_all  # noqa: E402


def main() -> None:
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/plos/data/results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = run_all()
    out_path.write_text(json.dumps(results, indent=2))

    # Console summary.
    ab = results["consensus_ablation"]["rows"]
    print("Consensus ablation (fixed tier):")
    for r in ab:
        print(f"  panel={r['panel_size']}: distinct={r['distinct_critiques']} "
              f"multi_reviewer={r['multi_reviewer']} grade={r['grade']}")
    ov = results["role_overlap"]
    print(f"Role overlap: union={ov['union_size']} sum_individual={ov['sum_individual']} "
          f"pairwise_jaccard={[p['jaccard'] for p in ov['pairwise']]}")
    cp = results["convergence_profile"]
    print(f"Convergence: iters(min/med/max)="
          f"{cp['iterations_to_converge']} signals={cp['signal_counts']}")
    print("Load benchmark:")
    for r in results["load_benchmark"]["rows"]:
        print(f"  C={r['concurrency']:>2}: throughput={r['throughput_iters_per_s']} it/s "
              f"p95={r['latency_p95_s']}s wall={r['wall_seconds']}s")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
