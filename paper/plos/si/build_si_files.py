"""Build S1_File.zip, S2_File.zip and S3_File.zip for the PLOS ONE manuscript.

Run from anywhere: python paper/plos/si/build_si_files.py
macOS ._* files, __pycache__, archives and large binary arrays are excluded.
"""
from __future__ import annotations

import hashlib
import os
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "paper" / "plos" / "si"
SMALL = 5 * 1024 * 1024  # per-file cap for execution work directories (S3)

LICENCE = ("Data in this archive are released under CC BY 4.0; code is released under the MIT licence "
           "(see https://github.com/Biodyn-AI/mi-workbench).")


def keep(p: Path) -> bool:
    parts = p.parts
    return (not p.name.startswith("._") and "__pycache__" not in parts and "archive" not in parts
            and p.name != ".DS_Store" and not p.name.endswith(".pyc"))


def add_tree(z: zipfile.ZipFile, src: Path, arc: str, max_bytes: int | None = None, skipped: list | None = None,
             suffix_exclude: tuple = ()) -> None:
    if src.is_file():
        if keep(src):
            z.write(src, arc)
        return
    for p in sorted(src.rglob("*")):
        if not p.is_file() or not keep(p) or p.suffix in suffix_exclude:
            continue
        if max_bytes is not None and p.stat().st_size > max_bytes:
            if skipped is not None:
                skipped.append(f"{p.relative_to(ROOT)} ({p.stat().st_size / 1e6:.1f} MB)")
            continue
        z.write(p, f"{arc}/{p.relative_to(src)}")


def build(name: str, entries: list[tuple], readme: str) -> Path:
    path = OUT / name
    skipped: list[str] = []
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for e in entries:
            src, arc = e[0], e[1]
            opts = e[2] if len(e) > 2 else {}
            add_tree(z, ROOT / src, arc, skipped=skipped, **opts)
        text = readme + "\n\n" + LICENCE + "\n"
        if skipped:
            text += "\nFiles not included because of their size (regenerate with the scripts provided):\n" + \
                    "\n".join(f"- {s}" for s in skipped) + "\n"
        z.writestr("README.txt", text)
    return path


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    s1 = build("S1_File.zip", [
        ("experiments/benchmark/artifacts", "benchmark/artifacts"),
        ("experiments/benchmark/ground_truth", "benchmark/ground_truth"),
        ("experiments/benchmark/verification", "benchmark/verification"),
        ("experiments/benchmark/ANALYSIS_PLAN.md", "benchmark/ANALYSIS_PLAN.md"),
        ("experiments/benchmark/benchmark_manifest.json", "benchmark/benchmark_manifest.json"),
        ("experiments/benchmark/prompts/judge.md", "benchmark/prompts/judge.md"),
        ("experiments/benchmark/README.md", "benchmark/README.md"),
        ("prompts/reviewer/mi_reviewer.yaml", "reviewer_prompts/mi_reviewer.yaml"),
        ("prompts/reviewer/mi_reviewer_combined.yaml", "reviewer_prompts/mi_reviewer_combined.yaml"),
        ("prompts/adversarial_reviewer/adversarial_reviewer.yaml", "reviewer_prompts/adversarial_reviewer.yaml"),
        ("prompts/biological_plausibility/bio_plausibility_checker.yaml", "reviewer_prompts/bio_plausibility_checker.yaml"),
    ], "S1 File. Planted-flaw benchmark: 18 analysis write-ups (12 flawed, 6 clean), ground truth with "
       "detection criteria, independent verification notes, the analysis plan fixed before any reviewer call, the "
       "SHA-256 manifest, the judge rubric and the four reviewer prompts.")
    s2 = build("S2_File.zip", [
        ("experiments_data/benchmark_runs_v1/reviews", "benchmark_runs/reviews"),
        ("experiments_data/benchmark_runs_v1/judgments", "benchmark_runs/judgments"),
        ("experiments_data/benchmark_runs_v1/merge_eval", "benchmark_runs/merge_eval"),
        ("experiments_data/benchmark_runs_v1/logs", "benchmark_runs/logs"),
        ("experiments/benchmark/results", "benchmark_results"),
        ("experiments_data/stopping_runs", "stopping_runs"),
        ("experiments/stopping/results", "stopping_results"),
        ("experiments/benchmark", "scripts/benchmark", {"suffix_exclude": (".md", ".json")}),
        ("experiments/stopping", "scripts/stopping", {"suffix_exclude": (".json",)}),
        ("paper/plos/si/build_si_tables.py", "scripts/build_si_tables.py"),
    ], "S2 File. Raw reviewer outputs (with parsed critiques, token counts and timing), judge labels, LLM "
       "adjudications, logs and processed results of the planted-flaw benchmark and merge evaluation; trajectories, "
       "per-call records, state judgments and replay results of the stopping calibration; and the analysis scripts. "
       "Reproduce the tables with experiments/benchmark/analyze.py, merge_eval.py and experiments/stopping/replay.py "
       "from the repository.")
    case = ROOT / "experiments_data/case_study_agent_runs"
    s3 = build("S3_File.zip", [
        ("experiments/case_study/build_kit.py", "case_study/build_kit.py"),
        ("experiments/case_study/independent_validation.py", "case_study/independent_validation.py"),
        ("experiments/case_study/README.md", "case_study/README.md"),
        ("experiments/case_study/results", "case_study/independent_results"),
        ("experiments_data/case_study_kit/README.md", "case_study/kit/README.md"),
        ("experiments_data/case_study_kit/kit_manifest.json", "case_study/kit/kit_manifest.json"),
        ("experiments_data/case_study_kit/genes.tsv", "case_study/kit/genes.tsv"),
        ("experiments_data/case_study_kit/cells.tsv", "case_study/kit/cells.tsv"),
        ("experiments_data/case_study_kit/trrust_edges.tsv", "case_study/kit/trrust_edges.tsv"),
        ("experiments_data/case_study_kit/dorothea_abc_edges.tsv", "case_study/kit/dorothea_abc_edges.tsv"),
        ("experiments_data/case_study_kit/prepare_summary.json", "case_study/kit/prepare_summary.json"),
        ("experiments/case_study/agent_runs", "case_study/agent_runs"),
        ("experiments_data/case_study_agent_runs/workspaces", "case_study/agent_run_outputs/workspaces", {"max_bytes": SMALL}),
        ("experiments_data/case_study_agent_runs/exec_outputs", "case_study/agent_run_outputs/exec_outputs", {"max_bytes": SMALL}),
    ], "S3 File. Case study: data-kit construction and independent-analysis code with results and provenance; "
       "the small kit files (gene and cell tables, reference edges, manifest with checksums); the agent-run harness, "
       "task texts, number trace and verified synthesis; and the four agent runs (all iteration directories, run "
       "metadata and execution outputs). The large attention and co-expression arrays (about 2.6 GB) are not "
       "included; they are regenerated by build_kit.py from the public Tabula Sapiens data and Geneformer V2-104M "
       "(Hugging Face snapshot fcd26c45fc30fba1989e586bdc46bc366dda8655), and their checksums are in kit_manifest.json.")
    lines = ["# Supporting information files", ""]
    for p in [s1, s2, s3, OUT / "S1_Table.xlsx", OUT / "S2_Table.xlsx"]:
        lines.append(f"- {p.name}: {p.stat().st_size / 1e6:.1f} MB, sha256 {sha256(p)}")
    (OUT / "SI_MANIFEST.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
