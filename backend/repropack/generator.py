"""Reproducibility Package generator for MI-Workbench runs.

A package holds the run's top-level files, every iteration directory
(``iter_NNNN/``: full step outputs, parsed artifacts, ``CONSENSUS.json``,
``CODE_EXECUTION.json``, ``RUN_LOG.md``, ``<role>_feedback.md``, lens outputs)
and, when the run used ``code_execution_persist_dir``, the persisted
execution outputs (``code.py``, full ``stdout.txt`` / ``stderr.txt`` and the
files the code wrote). Files larger than the per-file cap
(``max_file_mb``; default 25 MB, env ``MIW_REPROPACK_MAX_FILE_MB``) are left
out and listed in the package README. Symlinks and other non-regular files
are never followed or included, and persisted outputs are only read from
inside the run's configured persist directory.
"""
from __future__ import annotations

import io
import json
import os
import platform
import re
import stat
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.models import RunState

#: Default per-file size cap for iteration files and persisted outputs.
DEFAULT_MAX_FILE_MB = 25.0
_ITER_DIR_RE = re.compile(r"^iter_\d{4,}$")
# OS metadata that is never packaged (macOS AppleDouble files on exFAT etc.).
_SKIP_NAMES = {".DS_Store", "Thumbs.db"}


def _default_max_file_mb() -> float:
    raw = os.environ.get("MIW_REPROPACK_MAX_FILE_MB", "")
    try:
        return float(raw) if raw.strip() else DEFAULT_MAX_FILE_MB
    except ValueError:
        return DEFAULT_MAX_FILE_MB


def _is_metadata_name(name: str) -> bool:
    return name in _SKIP_NAMES or name.startswith("._")


@dataclass
class PackEntry:
    """One file to package (``source``) or one left out (``reason``)."""

    arcname: str
    size: int
    source: Optional[Path] = None
    reason: str = ""


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


class ReproPackGenerator:
    """Generate a reproducibility package for a run.

    ``max_file_mb``: per-file cap (MB) for iteration files and persisted
    execution outputs; larger files are skipped and listed in the README.
    ``None`` = ``MIW_REPROPACK_MAX_FILE_MB`` or 25 MB; 0 = no cap.
    """

    def __init__(self, max_file_mb: Optional[float] = None):
        self.max_file_mb = _default_max_file_mb() if max_file_mb is None else float(max_file_mb)
        if self.max_file_mb < 0:
            raise ValueError("max_file_mb must be >= 0")

    def _cap_bytes(self, max_file_mb: Optional[float]) -> Optional[int]:
        mb = self.max_file_mb if max_file_mb is None else float(max_file_mb)
        if mb < 0:
            raise ValueError("max_file_mb must be >= 0")
        return None if mb == 0 else int(mb * 1024 * 1024)

    def generate(
        self,
        run_dir: str,
        workspace_path: str,
        run_state: Optional[RunState] = None,
        max_file_mb: Optional[float] = None,
    ) -> bytes:
        """Generate a zip file containing the full reproducibility package.

        Contents:
        - artifacts/ - Top-level files of the run directory (MECH.md, EVAL.md, etc.)
        - iterations/iter_NNNN/ - Every iteration directory of the run, in full
          (step outputs, parsed artifacts, CONSENSUS.json, CODE_EXECUTION.json,
          RUN_LOG.md, <role>_feedback.md, lens outputs)
        - code_execution/iter_NNNN/<unit>/ - Persisted execution outputs (code.py,
          stdout.txt, stderr.txt, work/...) when code_execution_persist_dir was used
        - logs/ - Raw provider transcripts
        - metadata/
          - run_meta.json - Run configuration and results
          - prompts_used.json - Resolved prompt templates
          - environment.json - Python version, packages, OS info
          - git_state.json - Current branch, commit hash, dirty status
          - timestamps.json - All timing information
        - methods/
          - METHOD.md - Methodology description (if exists)
          - dataset_card.md - Auto-generated dataset card
          - reproducibility_checklist.md - Auto-generated checklist
        - README.md - Human-readable summary of the package (lists the files
          left out because of the size cap or because they could not be read)

        ``max_file_mb`` overrides the generator's per-file cap for this package.
        """
        run_path = Path(run_dir)
        cap = self._cap_bytes(max_file_mb)
        run_meta = self._load_run_meta(run_path, run_state)
        artifacts = self._collect_artifacts(run_path)
        logs = self._collect_logs(run_path)
        entries = (self._collect_iterations(run_path, cap)
                   + self._collect_persisted_outputs(run_path, run_meta, cap))
        env_info = self._generate_environment_info()
        git_state = self._generate_git_state(workspace_path)

        included: list[PackEntry] = []
        skipped = [e for e in entries if e.source is None]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # artifacts/
            for arc_name, content in artifacts:
                zf.writestr(f"artifacts/{arc_name}", content)

            # iterations/ and code_execution/ (bytes as on disk)
            for e in entries:
                if e.source is None:
                    continue
                data, reason = self._read_capped(e.source, cap)
                if data is None:
                    skipped.append(PackEntry(e.arcname, e.size, reason=reason))
                    continue
                zf.writestr(e.arcname, data)
                included.append(PackEntry(e.arcname, len(data), e.source))

            # logs/
            for arc_name, content in logs:
                zf.writestr(f"logs/{arc_name}", content)

            # metadata/
            zf.writestr(
                "metadata/run_meta.json",
                json.dumps(run_meta, indent=2, default=str),
            )
            prompts_used = run_meta.get("prompts_used", {})
            zf.writestr(
                "metadata/prompts_used.json",
                json.dumps(prompts_used, indent=2, default=str),
            )
            zf.writestr(
                "metadata/environment.json",
                json.dumps(env_info, indent=2, default=str),
            )
            zf.writestr(
                "metadata/git_state.json",
                json.dumps(git_state, indent=2, default=str),
            )
            timestamps = self._extract_timestamps(run_meta)
            zf.writestr(
                "metadata/timestamps.json",
                json.dumps(timestamps, indent=2, default=str),
            )

            # methods/
            method_path = run_path / "METHOD.md"
            if method_path.exists():
                zf.writestr("methods/METHOD.md", method_path.read_text())

            artifacts_dict = {name: content for name, content in artifacts}
            zf.writestr(
                "methods/dataset_card.md",
                self._generate_dataset_card(artifacts_dict),
            )
            zf.writestr(
                "methods/reproducibility_checklist.md",
                self._generate_reproducibility_checklist(run_meta, artifacts_dict),
            )

            # README.md
            zf.writestr("README.md", self._generate_readme(
                run_meta, artifacts_dict, included=included, skipped=skipped,
                cap_bytes=cap))

        return buf.getvalue()

    def preview(self, run_dir: str, workspace_path: str,
                run_state: Optional[RunState] = None,
                max_file_mb: Optional[float] = None) -> list[dict]:
        """Preview what would be included: list of dicts with path and size.

        Files that would be left out carry ``"skipped": <reason>``."""
        run_path = Path(run_dir)
        cap = self._cap_bytes(max_file_mb)
        entries: list[dict] = []

        for name, content in self._collect_artifacts(run_path):
            entries.append({"path": f"artifacts/{name}", "size": len(content)})

        run_meta = self._load_run_meta(run_path, run_state)
        for e in (self._collect_iterations(run_path, cap)
                  + self._collect_persisted_outputs(run_path, run_meta, cap)):
            item: dict[str, Any] = {"path": e.arcname, "size": e.size}
            if e.source is None:
                item["skipped"] = e.reason
            entries.append(item)

        for name, content in self._collect_logs(run_path):
            entries.append({"path": f"logs/{name}", "size": len(content)})

        # metadata files (estimate sizes with placeholder)
        for meta_file in [
            "run_meta.json",
            "prompts_used.json",
            "environment.json",
            "git_state.json",
            "timestamps.json",
        ]:
            entries.append({"path": f"metadata/{meta_file}", "size": 0})

        method_path = run_path / "METHOD.md"
        if method_path.exists():
            entries.append({
                "path": "methods/METHOD.md",
                "size": method_path.stat().st_size,
            })

        entries.append({"path": "methods/dataset_card.md", "size": 0})
        entries.append({"path": "methods/reproducibility_checklist.md", "size": 0})
        entries.append({"path": "README.md", "size": 0})

        return entries

    # ── Internal helpers ────────────────────────────────────────────────

    def _load_run_meta(
        self, run_path: Path, run_state: Optional[RunState]
    ) -> dict:
        """Load run metadata from run_meta.json or RunState."""
        if run_state is not None:
            return json.loads(run_state.model_dump_json())

        meta_path = run_path / "run_meta.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())

        return {"run_id": run_path.name, "note": "No run metadata found"}

    def _collect_artifacts(self, run_dir: Path) -> list[tuple[str, str]]:
        """Collect all artifact files. Returns (path_in_zip, content) tuples."""
        results: list[tuple[str, str]] = []
        if not run_dir.exists():
            return results

        artifact_extensions = {".md", ".json", ".yaml", ".yml", ".txt", ".csv", ".tsv"}
        # Also look in an artifacts/ subdirectory
        search_dirs = [run_dir]
        artifacts_sub = run_dir / "artifacts"
        if artifacts_sub.is_dir():
            search_dirs.append(artifacts_sub)

        seen: set[str] = set()
        for search_dir in search_dirs:
            for f in sorted(search_dir.iterdir()):
                if (f.is_file() and f.suffix in artifact_extensions and f.name not in seen
                        and not _is_metadata_name(f.name)):
                    # Skip metadata files we handle separately
                    if f.name in ("run_meta.json",):
                        continue
                    try:
                        content = f.read_text(errors="replace")
                        results.append((f.name, content))
                        seen.add(f.name)
                    except Exception:
                        pass

        return results

    # ── iteration directories and persisted execution outputs ──────────

    @staticmethod
    def _walk_files(root: Path) -> list[tuple[Path, os.stat_result]]:
        """Regular files under ``root`` (sorted; symlinks never followed)."""
        out: list[tuple[Path, os.stat_result]] = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            dirnames[:] = sorted(d for d in dirnames if not _is_metadata_name(d))
            for name in sorted(filenames):
                if _is_metadata_name(name):
                    continue
                p = Path(dirpath) / name
                try:
                    st = os.lstat(p)
                except OSError:
                    continue
                out.append((p, st))
        return out

    def _entry(self, arcname: str, path: Path, st: os.stat_result,
               cap: Optional[int]) -> PackEntry:
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
            return PackEntry(arcname, 0, reason="not a regular file (symlink or special file)")
        if cap is not None and st.st_size > cap:
            return PackEntry(arcname, st.st_size,
                             reason=f"larger than the {_fmt_size(cap)} per-file cap")
        return PackEntry(arcname, st.st_size, source=path)

    @staticmethod
    def _iteration_dirs(run_dir: Path) -> list[Path]:
        if not run_dir.is_dir():
            return []
        return sorted(d for d in run_dir.iterdir()
                      if _ITER_DIR_RE.match(d.name) and d.is_dir() and not d.is_symlink())

    def _collect_iterations(self, run_dir: Path, cap: Optional[int]) -> list[PackEntry]:
        """Every file of every ``iter_NNNN`` directory, as ``iterations/...``."""
        entries: list[PackEntry] = []
        for d in self._iteration_dirs(run_dir):
            for p, st in self._walk_files(d):
                rel = p.relative_to(d).as_posix()
                entries.append(self._entry(f"iterations/{d.name}/{rel}", p, st, cap))
        return entries

    @staticmethod
    def _persist_root(run_meta: dict) -> Optional[str]:
        cfg = run_meta.get("config") if isinstance(run_meta, dict) else None
        raw = (cfg or {}).get("code_execution_persist_dir") if isinstance(cfg, dict) else None
        if not raw or not isinstance(raw, str):
            return None
        return os.path.realpath(os.path.expanduser(raw))

    def _collect_persisted_outputs(self, run_dir: Path, run_meta: dict,
                                   cap: Optional[int]) -> list[PackEntry]:
        """Persisted execution units listed in the iterations' CODE_EXECUTION.json
        (``blocks[*].persisted_to``), as ``code_execution/iter_NNNN/<unit>/...``.

        Only directories inside the run's ``code_execution_persist_dir`` are
        read; a listed unit outside it (or missing) is reported as skipped."""
        root = self._persist_root(run_meta)
        entries: list[PackEntry] = []
        for d in self._iteration_dirs(run_dir):
            report_path = d / "CODE_EXECUTION.json"
            if not report_path.is_file() or report_path.is_symlink():
                continue
            try:
                report = json.loads(report_path.read_text(errors="replace"))
            except (OSError, ValueError):
                continue
            seen: set[str] = set()
            for block in report.get("blocks") or []:
                unit = block.get("persisted_to") if isinstance(block, dict) else None
                if not unit or not isinstance(unit, str) or unit in seen:
                    continue
                seen.add(unit)
                unit_path = Path(unit)
                prefix = f"code_execution/{d.name}/{unit_path.name}"
                real = os.path.realpath(unit)
                if root is None:
                    entries.append(PackEntry(prefix + "/", 0, reason=(
                        "persisted outputs not read: the run's code_execution_persist_dir "
                        "is not recorded")))
                    continue
                if os.path.commonpath([real, root]) != root or real == root:
                    entries.append(PackEntry(prefix + "/", 0, reason=(
                        "persisted outputs not read: outside the run's "
                        "code_execution_persist_dir")))
                    continue
                if unit_path.is_symlink() or not os.path.isdir(real):
                    entries.append(PackEntry(prefix + "/", 0,
                                             reason="persisted outputs not found"))
                    continue
                for p, st in self._walk_files(Path(real)):
                    rel = p.relative_to(real).as_posix()
                    entries.append(self._entry(f"{prefix}/{rel}", p, st, cap))
        return entries

    @staticmethod
    def _read_capped(path: Path, cap: Optional[int]) -> tuple[Optional[bytes], str]:
        """Read a regular file without following symlinks, re-checking type and size."""
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
        except OSError as exc:
            return None, f"unreadable ({exc.strerror or exc})"
        try:
            st = os.fstat(fd)
            if not stat.S_ISREG(st.st_mode):
                return None, "not a regular file"
            if cap is not None and st.st_size > cap:
                return None, f"larger than the {_fmt_size(cap)} per-file cap"
            with os.fdopen(fd, "rb") as fh:
                fd = -1
                data = fh.read(cap + 1 if cap is not None else -1)
            if cap is not None and len(data) > cap:
                return None, f"larger than the {_fmt_size(cap)} per-file cap"
            return data, ""
        except OSError as exc:
            return None, f"unreadable ({exc.strerror or exc})"
        finally:
            if fd >= 0:
                os.close(fd)

    def _collect_logs(self, run_dir: Path) -> list[tuple[str, str]]:
        """Collect log files."""
        results: list[tuple[str, str]] = []
        logs_dir = run_dir / "logs"
        if not logs_dir.is_dir():
            return results

        for f in sorted(logs_dir.iterdir()):
            if f.is_file() and not _is_metadata_name(f.name):
                try:
                    content = f.read_text(errors="replace")
                    results.append((f.name, content))
                except Exception:
                    pass

        return results

    def _generate_environment_info(self) -> dict:
        """Capture environment: Python version, installed packages, OS, hardware."""
        import sys

        env: dict = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "os": platform.system(),
            "os_version": platform.version(),
            "captured_at": datetime.utcnow().isoformat(),
        }

        # Installed packages via pip freeze
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "freeze"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if result.returncode == 0:
                env["packages"] = result.stdout.strip().split("\n")
        except Exception:
            env["packages"] = []

        return env

    def _generate_git_state(self, workspace_path: str) -> dict:
        """Capture git branch, commit, dirty status."""
        git: dict = {"workspace_path": workspace_path}

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                git["is_git_repo"] = False
                return git

            git["is_git_repo"] = True

            # Branch
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            git["branch"] = result.stdout.strip() if result.returncode == 0 else "unknown"

            # Commit hash
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            git["commit"] = result.stdout.strip() if result.returncode == 0 else "unknown"

            # Dirty status
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=workspace_path,
                capture_output=True,
                text=True,
                timeout=5,
            )
            git["dirty"] = bool(result.stdout.strip()) if result.returncode == 0 else False

        except Exception:
            git["is_git_repo"] = False

        return git

    def _extract_timestamps(self, run_meta: dict) -> dict:
        """Extract all timing information from run metadata."""
        ts: dict = {}
        for key in ("created_at", "started_at", "stopped_at", "completed_at"):
            if key in run_meta and run_meta[key] is not None:
                ts[key] = str(run_meta[key])

        iterations = run_meta.get("iterations", [])
        if iterations:
            iter_times = []
            for it in iterations:
                entry = {"iteration_number": it.get("iteration_number")}
                if it.get("started_at"):
                    entry["started_at"] = str(it["started_at"])
                if it.get("completed_at"):
                    entry["completed_at"] = str(it["completed_at"])
                iter_times.append(entry)
            ts["iterations"] = iter_times

        ts["pack_generated_at"] = datetime.utcnow().isoformat()
        return ts

    def _generate_dataset_card(self, artifacts: dict[str, str]) -> str:
        """Auto-generate a dataset card from available information."""
        lines = [
            "# Dataset Card",
            "",
            "## Overview",
            "",
            "This dataset card was auto-generated from the run artifacts.",
            "",
        ]

        if artifacts:
            lines.append("## Available Artifacts")
            lines.append("")
            for name in sorted(artifacts.keys()):
                size = len(artifacts[name])
                lines.append(f"- **{name}** ({size:,} bytes)")
            lines.append("")

        lines.extend([
            "## Data Sources",
            "",
            "Please refer to the run metadata and METHOD.md for dataset details.",
            "",
            "## Preprocessing",
            "",
            "See the experiment scripts and logs for preprocessing steps.",
            "",
        ])

        return "\n".join(lines)

    def _generate_reproducibility_checklist(
        self, run_meta: dict, artifacts: dict[str, str]
    ) -> str:
        """Generate a checklist for reproducibility verification."""
        lines = [
            "# Reproducibility Checklist",
            "",
            "## Environment",
            "",
            "- [ ] All dependencies installed (see `metadata/environment.json`)",
            "- [ ] Python version matches (see `metadata/environment.json`)",
            "- [ ] OS/hardware requirements documented",
            "",
            "## Data",
            "",
            "- [ ] Dataset accessible at specified paths",
            "- [ ] Data preprocessing steps documented",
            "- [ ] Data checksums verified (if available)",
            "",
            "## Model",
            "",
            "- [ ] Model weights available",
            "- [ ] Model version/checkpoint recorded",
            "- [ ] Random seeds recorded",
            "",
            "## Analysis",
            "",
            "- [ ] Statistical tests reproducible",
            "- [ ] Figures regeneratable from data",
            "- [ ] All hyperparameters documented",
            "",
            "## Version Control",
            "",
            "- [ ] Git commit hash recorded (see `metadata/git_state.json`)",
            "- [ ] No uncommitted changes at time of run",
            "",
            "## Run Details",
            "",
        ]

        run_id = run_meta.get("run_id", "unknown")
        lines.append(f"- Run ID: `{run_id}`")

        provider = run_meta.get("provider", "unknown")
        lines.append(f"- Provider: `{provider}`")

        model = run_meta.get("model", "unknown")
        lines.append(f"- Model: `{model}`")

        status = run_meta.get("status", "unknown")
        lines.append(f"- Status: `{status}`")

        total_iterations = run_meta.get("total_iterations", run_meta.get("current_iteration", 0))
        lines.append(f"- Total iterations: {total_iterations}")

        lines.append("")
        lines.append(f"## Artifacts ({len(artifacts)} files)")
        lines.append("")
        for name in sorted(artifacts.keys()):
            lines.append(f"- [x] `{name}` included")
        lines.append("")

        return "\n".join(lines)

    def _generate_readme(
        self,
        run_meta: dict,
        artifacts: dict[str, str],
        included: Optional[list[PackEntry]] = None,
        skipped: Optional[list[PackEntry]] = None,
        cap_bytes: Optional[int] = None,
    ) -> str:
        """Generate a README for the repro pack."""
        run_id = run_meta.get("run_id", "unknown")
        task = run_meta.get("task", "No task description")
        provider = run_meta.get("provider", "unknown")
        model = run_meta.get("model", "")
        status = run_meta.get("status", "unknown")
        created = run_meta.get("created_at", "unknown")

        lines = [
            f"# Reproducibility Package - Run `{run_id}`",
            "",
            f"**Generated:** {datetime.utcnow().isoformat()}Z",
            "",
            "## Run Summary",
            "",
            f"- **Task:** {task}",
            f"- **Provider:** {provider}",
            f"- **Model:** {model or 'default'}",
            f"- **Status:** {status}",
            f"- **Created:** {created}",
            "",
            "## Package Contents",
            "",
            "```",
            "artifacts/        - Top-level run outputs (MECH.md, EVAL.md, etc.)",
            "iterations/       - Every iteration directory (iter_NNNN/: step outputs,",
            "                    parsed artifacts, CONSENSUS.json, CODE_EXECUTION.json,",
            "                    RUN_LOG.md, <role>_feedback.md, lens outputs)",
            "code_execution/   - Persisted execution outputs per iteration (code.py,",
            "                    stdout.txt, stderr.txt, work/), if persisted",
            "logs/             - Raw provider transcripts",
            "metadata/",
            "  run_meta.json   - Run configuration and results",
            "  prompts_used.json - Resolved prompt templates",
            "  environment.json  - Python version, packages, OS info",
            "  git_state.json    - Git branch, commit, dirty status",
            "  timestamps.json   - All timing information",
            "methods/",
            "  dataset_card.md            - Dataset documentation",
            "  reproducibility_checklist.md - Verification checklist",
            "README.md         - This file",
            "```",
            "",
        ]

        if artifacts:
            lines.append(f"## Artifacts ({len(artifacts)} files)")
            lines.append("")
            for name in sorted(artifacts.keys()):
                lines.append(f"- `{name}`")
            lines.append("")

        included = included or []
        skipped = skipped or []
        cap_text = "no cap" if cap_bytes is None else _fmt_size(cap_bytes)
        if included or skipped:
            iters = sorted({e.arcname.split("/")[1] for e in included
                            if e.arcname.startswith("iterations/")})
            n_exec = sum(1 for e in included if e.arcname.startswith("code_execution/"))
            lines.append("## Iterations and execution outputs")
            lines.append("")
            lines.append(f"- Iteration directories: {len(iters)} "
                         f"({sum(1 for e in included if e.arcname.startswith('iterations/'))} files)")
            lines.append(f"- Persisted execution output files: {n_exec}")
            lines.append(f"- Per-file size cap: {cap_text}")
            lines.append("")
        lines.append("## Files not included")
        lines.append("")
        if skipped:
            lines.append(f"{len(skipped)} file(s) were left out of this package:")
            lines.append("")
            lines.append("| Path | Size | Reason |")
            lines.append("|------|------|--------|")
            for e in sorted(skipped, key=lambda x: x.arcname):
                size = _fmt_size(e.size) if e.size else "-"
                lines.append(f"| `{e.arcname}` | {size} | {e.reason} |")
        else:
            lines.append(f"None (per-file size cap: {cap_text}).")
        lines.append("")

        lines.extend([
            "## How to Use",
            "",
            "1. Extract this zip file",
            "2. Review `methods/reproducibility_checklist.md`",
            "3. Install dependencies from `metadata/environment.json`",
            "4. Ensure datasets are accessible at documented paths",
            "5. Re-run the experiment using the recorded configuration",
            "",
            "---",
            "*Generated by MI-Workbench Reproducibility Package Generator*",
            "",
        ])

        return "\n".join(lines)
