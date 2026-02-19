"""Reproducibility Package generator for MI-Workbench runs."""
from __future__ import annotations

import io
import json
import platform
import subprocess
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from backend.models import RunState


class ReproPackGenerator:
    """Generate a reproducibility package for a run."""

    def generate(
        self,
        run_dir: str,
        workspace_path: str,
        run_state: Optional[RunState] = None,
    ) -> bytes:
        """Generate a zip file containing the full reproducibility package.

        Contents:
        - artifacts/ - All artifacts from the run (MECH.md, EVAL.md, etc.)
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
        - README.md - Human-readable summary of the package
        """
        run_path = Path(run_dir)
        run_meta = self._load_run_meta(run_path, run_state)
        artifacts = self._collect_artifacts(run_path)
        logs = self._collect_logs(run_path)
        env_info = self._generate_environment_info()
        git_state = self._generate_git_state(workspace_path)

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # artifacts/
            for arc_name, content in artifacts:
                zf.writestr(f"artifacts/{arc_name}", content)

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
            zf.writestr("README.md", self._generate_readme(run_meta, artifacts_dict))

        return buf.getvalue()

    def preview(self, run_dir: str, workspace_path: str) -> list[dict]:
        """Preview what would be included: list of dicts with path and size."""
        run_path = Path(run_dir)
        entries: list[dict] = []

        for name, content in self._collect_artifacts(run_path):
            entries.append({"path": f"artifacts/{name}", "size": len(content)})

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
                if f.is_file() and f.suffix in artifact_extensions and f.name not in seen:
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

    def _collect_logs(self, run_dir: Path) -> list[tuple[str, str]]:
        """Collect log files."""
        results: list[tuple[str, str]] = []
        logs_dir = run_dir / "logs"
        if not logs_dir.is_dir():
            return results

        for f in sorted(logs_dir.iterdir()):
            if f.is_file():
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

    def _generate_readme(self, run_meta: dict, artifacts: dict[str, str]) -> str:
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
            "artifacts/        - Run outputs (MECH.md, EVAL.md, etc.)",
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
