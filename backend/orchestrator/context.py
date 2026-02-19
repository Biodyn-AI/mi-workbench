"""Workspace context reader - injects existing artifacts into prompts."""
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

MAX_ARTIFACT_CHARS = 2000


class WorkspaceContextReader:
    """Reads existing artifacts from workspace to build context for prompts."""

    def __init__(self, workspace_path: str, run_id: str):
        self.workspace_path = Path(workspace_path)
        self.run_id = run_id

    def _run_dir(self) -> Path:
        return self.workspace_path / "runs" / self.run_id

    def _iter_dirs(self) -> list[tuple[int, Path]]:
        """Return sorted list of (iteration_number, path) for this run, ascending."""
        run_dir = self._run_dir()
        if not run_dir.exists():
            return []
        result = []
        for d in run_dir.iterdir():
            if d.is_dir() and d.name.startswith("iter_"):
                try:
                    num = int(d.name.split("_", 1)[1])
                    result.append((num, d))
                except (IndexError, ValueError):
                    continue
        result.sort(key=lambda x: x[0])
        return result

    def read_latest_artifacts(self) -> dict[str, str]:
        """Read the latest version of each artifact type from the run directory.

        Scans runs/{run_id}/iter_*/  in reverse order.
        For each artifact type (MECH.md, EVAL.md, XP.md, METHOD.md),
        returns the content from the highest iteration that has it.
        Also checks for canonical workspace-level artifacts.
        Returns: {"MECH.md": "content...", "EVAL.md": "content...", ...}
        """
        target_artifacts = {"MECH.md", "EVAL.md", "XP.md", "METHOD.md"}
        found: dict[str, tuple[int, str]] = {}  # name -> (iter_num, content)

        iter_dirs = self._iter_dirs()
        # Scan in reverse to find latest first
        for iter_num, iter_path in reversed(iter_dirs):
            for artifact_name in target_artifacts:
                if artifact_name in found:
                    continue
                artifact_path = iter_path / artifact_name
                if artifact_path.exists():
                    try:
                        content = artifact_path.read_text()
                        found[artifact_name] = (iter_num, content)
                    except Exception:
                        logger.debug("Failed to read %s", artifact_path)
                # Also check role-based names (e.g. executor_output.md)
                for f in iter_path.iterdir():
                    if f.is_file() and f.suffix == ".md":
                        norm = self._normalize_artifact_name(f.name)
                        if norm in target_artifacts and norm not in found:
                            try:
                                found[norm] = (iter_num, f.read_text())
                            except Exception:
                                logger.debug("Failed to read %s", f)

        return {name: content for name, (_, content) in found.items()}

    def read_run_artifacts(self, iteration: Optional[int] = None) -> dict[str, str]:
        """Read artifacts from a specific iteration (or latest if None)."""
        iter_dirs = self._iter_dirs()
        if not iter_dirs:
            return {}

        if iteration is not None:
            target_dirs = [(n, p) for n, p in iter_dirs if n == iteration]
        else:
            target_dirs = [iter_dirs[-1]]

        if not target_dirs:
            return {}

        _, target_path = target_dirs[0]
        result = {}
        if target_path.exists():
            for f in target_path.iterdir():
                if f.is_file() and f.suffix == ".md":
                    try:
                        result[f.name] = f.read_text()
                    except Exception:
                        logger.debug("Failed to read %s", f)
        return result

    def build_context_section(self) -> str:
        """Build a formatted context string for injection into prompts.

        Format:
        === Current Artifact State ===

        ### MECH.md (from iteration 3)
        [content truncated to last 2000 chars if too long]

        ### EVAL.md (from iteration 2)
        [content]

        === End Current State ===
        """
        target_artifacts = {"MECH.md", "EVAL.md", "XP.md", "METHOD.md"}
        found: dict[str, tuple[int, str]] = {}

        iter_dirs = self._iter_dirs()
        for iter_num, iter_path in reversed(iter_dirs):
            for artifact_name in target_artifacts:
                if artifact_name in found:
                    continue
                artifact_path = iter_path / artifact_name
                if artifact_path.exists():
                    try:
                        found[artifact_name] = (iter_num, artifact_path.read_text())
                    except Exception:
                        pass
                for f in iter_path.iterdir():
                    if f.is_file() and f.suffix == ".md":
                        norm = self._normalize_artifact_name(f.name)
                        if norm in target_artifacts and norm not in found:
                            try:
                                found[norm] = (iter_num, f.read_text())
                            except Exception:
                                pass

        if not found:
            return ""

        sections = ["=== Current Artifact State ===", ""]
        for name in sorted(found.keys()):
            iter_num, content = found[name]
            if len(content) > MAX_ARTIFACT_CHARS:
                content = "...[truncated]\n" + content[-MAX_ARTIFACT_CHARS:]
            sections.append(f"### {name} (from iteration {iter_num})")
            sections.append(content)
            sections.append("")

        sections.append("=== End Current State ===")
        return "\n".join(sections)

    def get_reviewer_feedback_from_last_iteration(self) -> Optional[str]:
        """Extract the reviewer's output from the most recent reviewer iteration.
        Looks for files matching *reviewer*.md in the latest iteration dirs.
        """
        iter_dirs = self._iter_dirs()
        for _, iter_path in reversed(iter_dirs):
            for f in iter_path.iterdir():
                if f.is_file() and "reviewer" in f.name.lower() and f.suffix == ".md":
                    try:
                        return f.read_text()
                    except Exception:
                        logger.debug("Failed to read reviewer feedback from %s", f)
        return None

    @staticmethod
    def _normalize_artifact_name(filename: str) -> str:
        """Map role-based filenames to canonical artifact names."""
        lower = filename.lower()
        if "mech" in lower or "executor" in lower:
            return "MECH.md"
        if "eval" in lower or "reviewer" in lower:
            return "EVAL.md"
        if "xp" in lower or "experiment" in lower or "adversar" in lower:
            return "XP.md"
        if "method" in lower:
            return "METHOD.md"
        return filename
