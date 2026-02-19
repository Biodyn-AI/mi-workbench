"""Git operations for workspace artifact management."""
from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from backend.models import GitMode


async def _run_git(workspace_path: str, *args: str) -> tuple[int, str, str]:
    """Run a git command in the workspace directory."""
    proc = await asyncio.create_subprocess_exec(
        "git", *args,
        cwd=workspace_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return proc.returncode, stdout.decode(), stderr.decode()


async def init_or_check_repo(workspace_path: str) -> bool:
    """Initialise a git repo if not already present. Returns True if repo exists/created."""
    git_dir = Path(workspace_path) / ".git"
    if git_dir.exists():
        return True
    rc, _, _ = await _run_git(workspace_path, "init")
    if rc == 0:
        # Initial commit so branches work
        await _run_git(workspace_path, "add", "-A")
        await _run_git(workspace_path, "commit", "-m", "Initial MI-Workbench workspace", "--allow-empty")
    return rc == 0


async def create_branch(workspace_path: str, branch_name: str) -> bool:
    """Create and checkout a new branch."""
    rc, _, _ = await _run_git(workspace_path, "checkout", "-b", branch_name)
    return rc == 0


async def get_current_branch(workspace_path: str) -> str:
    """Return the current branch name."""
    rc, stdout, _ = await _run_git(workspace_path, "rev-parse", "--abbrev-ref", "HEAD")
    if rc != 0:
        return ""
    return stdout.strip()


async def commit_artifacts(
    workspace_path: str,
    run_id: str,
    iteration: int,
    message: str,
    git_mode: GitMode,
) -> bool:
    """Stage and commit artifacts based on the git mode."""
    if git_mode == GitMode.NONE:
        return True

    # Ensure repo
    await init_or_check_repo(workspace_path)

    if git_mode == GitMode.BRANCH_PER_RUN:
        current = await get_current_branch(workspace_path)
        branch = f"run/{run_id}"
        if current != branch:
            await create_branch(workspace_path, branch)

    # Stage run directory
    run_dir = f"runs/{run_id}"
    await _run_git(workspace_path, "add", run_dir)

    # Commit
    rc, _, _ = await _run_git(workspace_path, "commit", "-m", message)
    return rc == 0
