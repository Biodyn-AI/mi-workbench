#!/usr/bin/env python
"""Run the orchestration experiments and write JSON to experiments/orchestration/.

Two groups (see backend/analysis/experiments.py):

* ``mock``: mock implementation verification of the merge semantics and loop
  control. Writes ``mock_verification.json``.
* ``overhead``: control-plane overhead of concurrent runs through
  ``runner.execute_run`` with a real SQLite DB, artifact writing and an
  integrity check after every batch. Mock backend with a fixed delay. Writes
  ``overhead.json``.

Usage (from automation/mi-workbench, env ``mi_workbench``; no numpy needed):

    python scripts/run_experiments.py                       # both groups
    python scripts/run_experiments.py --only mock
    python scripts/run_experiments.py --only overhead \\
        --storage internal_apfs=/private/tmp/miw_overhead \\
        --storage external_exfat="/Volumes/Crucial X6/tmp_miw"

``--storage LABEL=DIR`` may be repeated. Each run puts the database and the
artifacts on the device holding DIR, and the result is stored per label. The
default is one location, ``$TMPDIR``. The mock group is deterministic; the
overhead numbers are wall-clock measurements.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from backend.analysis.experiments import (  # noqa: E402
    control_plane_overhead,
    mock_verification,
)

DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "orchestration"


def _csv(value: str, cast):
    return [cast(v) for v in value.split(",") if v.strip()]


def filesystem_of(path: str) -> dict:
    """Mount point and filesystem type of ``path`` (from ``mount``; best effort)."""
    p = Path(path).resolve()
    try:
        out = subprocess.run(["mount"], capture_output=True, text=True, timeout=10).stdout
    except Exception as exc:  # noqa: BLE001
        return {"mount_point": "", "fs_type": "", "error": str(exc)}
    best = ("", "", "")
    for line in out.splitlines():
        # "<device> on <mount point> (<type>, opts...)"  (macOS)
        # "<device> on <mount point> type <type> (opts)" (Linux)
        if " on " not in line:
            continue
        dev, rest = line.split(" on ", 1)
        if " type " in rest:
            mp, rest2 = rest.split(" type ", 1)
            fs = rest2.split(" ", 1)[0]
        elif " (" in rest:
            mp, rest2 = rest.rsplit(" (", 1)
            fs = rest2.split(",", 1)[0].rstrip(")")
        else:
            continue
        mp = mp.strip()
        try:
            inside = p == Path(mp) or Path(mp) in p.parents
        except Exception:  # noqa: BLE001
            inside = False
        if inside and len(mp) > len(best[0]):
            best = (mp, fs, dev)
    return {"mount_point": best[0], "fs_type": best[1], "device": best[2]}


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def run_mock(out_dir: Path) -> dict:
    res = mock_verification()
    _write(out_dir / "mock_verification.json", res)
    ms = res["merge_semantics"]
    print(f"Mock merge-semantics checks: all_passed={ms['all_passed']} {ms['checks']}")
    for r in res["consensus_ablation"]["rows"]:
        print(f"  panel={r['panel_size']}: distinct={r['distinct_critiques']} "
              f"multi_reviewer={r['multi_reviewer']} grade={r['grade']}")
    ov = res["role_overlap"]
    print(f"  exact-string overlap: union={ov['union_size']} sum={ov['sum_individual']} "
          f"pairwise={[p['exact_string_set_overlap'] for p in ov['pairwise']]}")
    cp = res["convergence_profile"]
    print(f"  convergence (single deterministic trajectory): iterations={cp['iterations']} "
          f"stop={cp['stop_reason']} identical_across_seeds={cp['determinism_check']['identical']}")
    return res


def run_overhead(out_dir: Path, storages: list[tuple[str, str]], levels, iterations,
                 delays, repeats, preset, keep: bool) -> dict:
    results = []
    for label, parent in storages:
        fs = filesystem_of(parent)
        print(f"Overhead [{label}] on {parent} ({fs.get('fs_type') or '?'}) ...", flush=True)
        t0 = time.time()
        res = control_plane_overhead(
            concurrency_levels=levels, iterations_per_run=iterations,
            adapter_delays=delays, repeats=repeats, preset=preset,
            parent_dir=parent, keep_workdir=keep)
        res["storage"] = {"label": label, "parent_dir": parent, **fs}
        results.append(res)
        print(f"  done in {time.time() - t0:.1f}s, integrity_ok={res['integrity_ok']}")
        for sw in res["sweeps"]:
            print(f"  delay={sw['adapter_delay_s']}s")
            for r in sw["rows"]:
                print(f"    C={r['concurrency']:>2}: thr={r['throughput_iters_per_s']['mean']} it/s "
                      f"lat p50/p95={r['latency_s']['p50']}/{r['latency_s']['p95']}s "
                      f"ovh p50/p95={r['overhead_ms_per_iteration']['p50']}/"
                      f"{r['overhead_ms_per_iteration']['p95']} ms/it "
                      f"cpu={r['cpu_ms_per_iteration']['mean']} ms/it "
                      f"integrity={'ok' if r['integrity_ok'] else 'FAIL'}")
    out = {
        "schema": "miw-orchestration-overhead-set/1",
        "storages": [r["storage"]["label"] for r in results],
        "integrity_ok": all(r["integrity_ok"] for r in results),
        "results": results,
    }
    _write(out_dir / "overhead.json", out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["all", "mock", "overhead"], default="all")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--levels", default="1,2,5,10,20", help="concurrency levels")
    ap.add_argument("--iterations", type=int, default=6, help="fixed iterations per run")
    ap.add_argument("--delays", default="0,1.0", help="mock adapter delays (s)")
    ap.add_argument("--repeats", type=int, default=5, help="batches per level")
    ap.add_argument("--preset", default="reviewer_consensus")
    ap.add_argument("--storage", action="append", default=[],
                    help="LABEL=DIR: put DB + artifacts under DIR (repeatable)")
    ap.add_argument("--keep-workdir", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    ok = True
    if args.only in ("all", "mock"):
        res = run_mock(out_dir)
        ok = ok and res["merge_semantics"]["all_passed"]
    if args.only in ("all", "overhead"):
        storages = []
        for s in args.storage or [f"tmpdir={tempfile.gettempdir()}"]:
            label, _, path = s.partition("=")
            if not path:
                ap.error(f"--storage expects LABEL=DIR, got {s!r}")
            storages.append((label.strip(), path))
        res = run_overhead(out_dir, storages, _csv(args.levels, int), args.iterations,
                           _csv(args.delays, float), args.repeats, args.preset,
                           args.keep_workdir)
        ok = ok and res["integrity_ok"]
    print(f"\nWrote results to {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
