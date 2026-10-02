"""Freeze the planted-flaw benchmark: validate it and write benchmark_manifest.json.

Validation (any failure aborts with exit code 1 and writes nothing):
- every ground-truth file parses and has an artifact; flaw ids are unique;
- flawed items carry exactly 4 flaws from 4 distinct families, clean items none;
- each flaw's family matches its type prefix and the type is in the taxonomy;
- every ground-truth quote is a verbatim substring of its artifact;
- each of the 12 flaw types occurs exactly 4 times;
- every clean item has a flawed counterpart.

The manifest records the SHA-256 of every file in ``artifacts/``,
``ground_truth/`` and ``ANALYSIS_PLAN.md`` plus counts. An existing manifest is
never silently replaced: identical hashes -> "verified" (file untouched);
different hashes -> abort unless ``--force``.

Usage:
    python experiments/benchmark/freeze.py [--check] [--force]
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402


def validate(bench_dir: Path) -> tuple[list[str], dict]:
    """Return (problems, summary) for the benchmark in ``bench_dir``."""
    problems: list[str] = []
    items = common.load_items(bench_dir)
    artifacts = {p.stem for p in common.visible_files(bench_dir / "artifacts") if p.suffix == ".md"}
    gt_ids = set(items)
    for a in sorted(artifacts - gt_ids):
        problems.append(f"artifact without ground truth: {a}")
    type_counts: collections.Counter = collections.Counter()
    family_counts: collections.Counter = collections.Counter()
    seen_ids: set[str] = set()
    quote_occurrences: dict[str, int] = {}
    symmetry: list[str] = []
    for item_id, item in items.items():
        if not item.artifact_path.exists():
            problems.append(f"{item_id}: artifact file missing ({item.artifact_path.name})")
            continue
        if item.version not in ("flawed", "clean"):
            problems.append(f"{item_id}: version must be flawed|clean, got {item.version!r}")
        if not item.item_id.endswith(item.version):
            problems.append(f"{item_id}: id does not match version {item.version!r}")
        if item.version == "clean":
            if item.flaws:
                problems.append(f"{item_id}: clean item lists {len(item.flaws)} flaws")
            if f"{item.scenario}-flawed" not in items:
                problems.append(f"{item_id}: no flawed counterpart")
            continue
        if len(item.flaws) != 4:
            problems.append(f"{item_id}: {len(item.flaws)} flaws (expected 4)")
        families = [f.family for f in item.flaws]
        if len(set(families)) != len(families):
            problems.append(f"{item_id}: repeated family {families}")
        for f in item.flaws:
            if f.flaw_id in seen_ids:
                problems.append(f"duplicate flaw id {f.flaw_id}")
            seen_ids.add(f.flaw_id)
            if not f.flaw_id.startswith(item.scenario + "-"):
                problems.append(f"{f.flaw_id}: id does not start with {item.scenario}-")
            if f.type not in common.FLAW_TYPES:
                problems.append(f"{f.flaw_id}: unknown type {f.type}")
            elif common.FAMILY_OF_PREFIX[f.type[0]] != f.family:
                problems.append(f"{f.flaw_id}: family {f.family} does not match type {f.type}")
            for fld in ("quote", "description", "detection_criterion"):
                if not getattr(f, fld).strip():
                    problems.append(f"{f.flaw_id}: empty {fld}")
            n = item.artifact_text.count(f.quote) if f.quote else 0
            quote_occurrences[f.flaw_id] = n
            if n == 0:
                problems.append(f"{f.flaw_id}: quote is not a verbatim substring of {item.artifact_path.name}")
            type_counts[f.type] += 1
            family_counts[f.family] += 1
            if f.is_symmetry_instance:
                symmetry.append(f.flaw_id)
    for t in common.FLAW_TYPES:
        if type_counts.get(t, 0) != common.FLAWS_PER_TYPE:
            problems.append(f"type {t} occurs {type_counts.get(t, 0)} times (expected {common.FLAWS_PER_TYPE})")
    flawed = [i for i in items.values() if i.is_flawed]
    clean = [i for i in items.values() if not i.is_flawed]
    summary = {
        "items": len(items),
        "flawed_items": len(flawed),
        "clean_items": len(clean),
        "flaws": sum(len(i.flaws) for i in flawed),
        "flaws_per_type": {t: type_counts.get(t, 0) for t in common.FLAW_TYPES},
        "flaws_per_family": dict(sorted(family_counts.items())),
        "quote_occurrences": quote_occurrences,
        "t1_symmetry_instances": symmetry,
        "item_list": [
            {
                "item_id": i.item_id, "scenario": i.scenario, "version": i.version,
                "n_flaws": len(i.flaws),
                "flaw_types": [f.type for f in i.flaws],
            }
            for i in items.values()
        ],
    }
    return problems, summary


def reviewer_prompt_hashes() -> dict:
    """Informational: the reviewer prompts as resolved at freeze time."""
    out = {}
    for tag, (role, ref) in common.CALL_SPECS.items():
        if ref in out:
            continue
        info = common.load_system_prompt(ref, role)
        out[ref] = {"version": info.version, "sha256": info.sha256}
    return out


def build_manifest(bench_dir: Path, summary: dict) -> dict:
    return {
        "schema": "miw-benchmark-manifest/1",
        "created_utc": common.utc_now(),
        "hash_algorithm": "sha256",
        "files": common.current_file_hashes(bench_dir),
        "counts": {k: summary[k] for k in (
            "items", "flawed_items", "clean_items", "flaws", "flaws_per_type", "flaws_per_family")},
        "t1_symmetry_instances": summary["t1_symmetry_instances"],
        "quote_occurrences": summary["quote_occurrences"],
        "items": summary["item_list"],
        "task_statement": common.TASK_STATEMENT,
        "task_statement_sha256": common.sha256_text(common.TASK_STATEMENT),
        "reviewer_prompts_at_freeze": reviewer_prompt_hashes(),
        "note": ("files{} is the frozen contract (artifacts, ground truth, analysis plan); "
                 "reviewer_prompts_at_freeze is informational (each review record stores "
                 "its own system-prompt hash and version)."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench-dir", default=str(common.BENCH_DIR))
    ap.add_argument("--check", action="store_true",
                    help="validate and compare with the existing manifest; never write")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing manifest whose hashes differ")
    args = ap.parse_args(argv)
    bench_dir = Path(args.bench_dir)

    problems, summary = validate(bench_dir)
    if problems:
        print("FREEZE ABORTED: benchmark validation failed:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1
    print(f"validated: {summary['items']} items ({summary['flawed_items']} flawed, "
          f"{summary['clean_items']} clean), {summary['flaws']} flaws, "
          f"each of {len(common.FLAW_TYPES)} types x{common.FLAWS_PER_TYPE}; "
          f"T1 symmetry instances: {', '.join(summary['t1_symmetry_instances'])}")

    path = bench_dir / common.MANIFEST_NAME
    hashes = common.current_file_hashes(bench_dir)
    if path.exists():
        existing = common.read_json(path)
        if existing.get("files") == hashes:
            print(f"manifest verified (unchanged): {path}")
            return 0
        changed = sorted(
            k for k in set(hashes) | set(existing.get("files", {}))
            if hashes.get(k) != existing.get("files", {}).get(k)
        )
        msg = f"manifest exists and differs for {len(changed)} file(s): {', '.join(changed)}"
        if args.check or not args.force:
            print(f"FREEZE ABORTED: {msg} (use --force to re-freeze; every condition on a "
                  "changed artifact must then be re-run)", file=sys.stderr)
            return 1
        print(f"WARNING: {msg}; re-freezing (--force)")
    elif args.check:
        print(f"no manifest at {path}", file=sys.stderr)
        return 1
    common.atomic_write_json(path, build_manifest(bench_dir, summary))
    print(f"manifest written: {path} ({len(hashes)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
