"""Test-suite and packaging hygiene (revision work package E10).

Covers:
- the ``client`` fixture's background-task drain, which stops the aiosqlite
  connection leak that kept the pytest process alive after the summary;
- the session-end reaper for orphaned aiosqlite worker threads;
- the order-independence of ``test_convergence_disabled_in_engine``;
- pytest configuration, LICENSE, CITATION.cff and the requirements files.
"""
from __future__ import annotations

import ast
import asyncio
import gc
import re
import sys
import threading
import time
import warnings
from pathlib import Path

import aiosqlite
import pytest
import yaml
from httpx import AsyncClient

from backend.tests.conftest import (
    alive_aiosqlite_workers,
    drain_background_tasks,
    reap_orphaned_aiosqlite_threads,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ── Background-task drain (aiosqlite leak fix) ───────────────────────


@pytest.mark.asyncio
async def test_drain_stops_background_run_and_closes_connections(
    client: AsyncClient, tmp_path: Path
) -> None:
    baseline = set(alive_aiosqlite_workers())
    resp = await client.post(
        "/api/workspaces", json={"name": "drain-ws", "path": str(tmp_path / "ws")}
    )
    assert resp.status_code == 201
    ws_id = resp.json()["id"]
    resp = await client.post("/api/runs", json={
        "workspace_id": ws_id,
        "task": "drain test",
        "provider": "mock",
        "max_iterations": 50,
        "config_overrides": {"convergence_enabled": False},
    })
    assert resp.status_code == 201
    run_id = resp.json()["run_id"]

    # The run is still in flight: POST /api/runs returns before execute_run ends.
    me = asyncio.current_task()
    assert any(not t.done() for t in asyncio.all_tasks() if t is not me)

    hard_cancelled = await drain_background_tasks(timeout=30.0)

    assert hard_cancelled == []  # every task ended cooperatively
    assert [t for t in asyncio.all_tasks() if t is not me and not t.done()] == []
    resp = await client.get(f"/api/runs/{run_id}")
    assert resp.json()["status"] in ("stopped", "completed", "failed")
    # Every connection opened by the run was closed, so no new worker threads remain.
    assert _wait_until(lambda: set(alive_aiosqlite_workers()) <= baseline)


@pytest.mark.asyncio
async def test_drain_hard_cancels_after_timeout() -> None:
    never = asyncio.Event()

    async def stuck() -> None:
        await never.wait()

    task = asyncio.create_task(stuck(), name="stuck-task")
    await asyncio.sleep(0)
    hard_cancelled = await drain_background_tasks(timeout=0.1)
    assert hard_cancelled == ["stuck-task"]
    assert task.cancelled()


@pytest.mark.asyncio
async def test_drain_is_noop_without_background_tasks() -> None:
    assert await drain_background_tasks(timeout=1.0) == []


# ── Session-end reaper ────────────────────────────────────────────────


def test_reaper_unblocks_orphaned_aiosqlite_worker() -> None:
    before = set(alive_aiosqlite_workers())

    async def leak_connection() -> aiosqlite.Connection:
        db = await aiosqlite.connect(":memory:")
        await db.execute("SELECT 1")
        return db  # deliberately never closed

    leaked = asyncio.run(leak_connection())
    orphans = [t for t in alive_aiosqlite_workers() if t not in before]
    assert len(orphans) == 1
    assert not orphans[0].daemon  # this is what kept the interpreter alive

    reaped = reap_orphaned_aiosqlite_threads(orphans, join_timeout=5.0)

    assert reaped == [orphans[0].name]
    assert not orphans[0].is_alive()
    # aiosqlite warns (ResourceWarning) when an unclosed Connection is collected.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        del leaked
        gc.collect()


def test_reaper_ignores_non_aiosqlite_threads() -> None:
    other = threading.Thread(target=time.sleep, args=(0.2,), name="not-aiosqlite")
    other.start()
    try:
        assert reap_orphaned_aiosqlite_threads([other], join_timeout=5.0) == []
    finally:
        other.join()


# ── Order-dependent asyncio test (event-loop plumbing) ───────────────


def test_convergence_disabled_test_survives_missing_event_loop() -> None:
    """Reproduce the state pytest-asyncio leaves behind, then run the fixed test."""
    from backend.tests.test_convergence import TestConvergenceDetector

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.close()
    asyncio.set_event_loop(None)
    with pytest.raises(RuntimeError):
        asyncio.get_event_loop()  # the call the old test relied on

    TestConvergenceDetector().test_convergence_disabled_in_engine()


# ── pytest configuration ─────────────────────────────────────────────


def test_pytest_ini_is_active(request: pytest.FixtureRequest) -> None:
    cfg = request.config
    assert cfg.inipath is not None and cfg.inipath.name == "pytest.ini"
    assert cfg.getini("asyncio_mode") == "strict"
    assert cfg.getini("testpaths") == ["backend/tests"]
    filters = cfg.getini("filterwarnings")
    assert any("not an async function" in f for f in filters)
    assert not any(f.strip().startswith("error") for f in filters)


# ── LICENSE and CITATION.cff ─────────────────────────────────────────


def test_license_is_mit() -> None:
    text = (REPO_ROOT / "LICENSE").read_text()
    assert text.startswith("MIT License")
    assert "Copyright (c) 2026 Ihor Kendiukhov" in text
    assert "THE SOFTWARE IS PROVIDED \"AS IS\"" in text


def test_citation_cff_fields() -> None:
    raw = (REPO_ROOT / "CITATION.cff").read_text()
    cff = yaml.safe_load(raw)
    assert cff["cff-version"] == "1.2.0"
    assert cff["title"] == "MI-Workbench"
    assert cff["type"] == "software"
    assert cff["license"] == "MIT"
    assert cff["version"] == "1.1.0"
    assert cff["repository-code"] == "https://github.com/Biodyn-AI/mi-workbench"
    assert cff["authors"] == [{"family-names": "Kendiukhov", "given-names": "Ihor"}]
    assert "doi" not in cff
    pref = cff["preferred-citation"]
    assert pref["type"] == "article"
    assert pref["title"] == (
        "MI-Workbench: an autonomous multi-agent platform for mechanistic "
        "interpretability of biological foundation models"
    )
    assert pref["journal"] == "PLOS ONE"
    assert pref["status"] == "submitted"
    assert pref["authors"] == cff["authors"]
    assert "doi" not in pref
    assert re.search(r"^\s*#.*DOI.*accept", raw, flags=re.IGNORECASE | re.MULTILINE)


# ── Requirements files ───────────────────────────────────────────────


def _requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", line)
        assert match, f"unparseable requirement line in {path.name}: {line!r}"
        names.add(match.group(0).lower().replace("_", "-"))
    return names


def _pinned_lines(path: Path) -> list[str]:
    return [
        ln.split("#", 1)[0].strip()
        for ln in path.read_text().splitlines()
        if ln.split("#", 1)[0].strip() and not ln.strip().startswith("-")
    ]


def test_requirements_files_split() -> None:
    core = _requirement_names(REPO_ROOT / "requirements.txt")
    dev = _requirement_names(REPO_ROOT / "requirements-dev.txt")
    analysis = _requirement_names(REPO_ROOT / "requirements-analysis.txt")

    assert {"fastapi", "uvicorn", "aiosqlite", "pydantic", "pyyaml", "click", "httpx"} <= core
    assert {"pytest", "pytest-asyncio", "httpx", "hypothesis"} <= dev
    assert {"numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "statsmodels"} <= analysis
    # The backend must stay dependency-free: no scientific stack or test tools in core.
    assert not core & {"numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "statsmodels",
                       "sentence-transformers", "pytest", "pytest-asyncio", "hypothesis"}
    # sentence-transformers is optional and only mentioned as a comment.
    assert "sentence-transformers" not in analysis
    assert "sentence-transformers" in (REPO_ROOT / "requirements-analysis.txt").read_text()
    # Every requirement carries a lower bound.
    for name in ("requirements.txt", "requirements-dev.txt", "requirements-analysis.txt"):
        for line in _pinned_lines(REPO_ROOT / name):
            assert ">=" in line, f"{name}: {line!r} has no lower bound"


_IMPORT_TO_DIST = {"yaml": "pyyaml"}


def _unguarded_third_party_imports(path: Path) -> list[tuple[str, int]]:
    """Top-level module names imported outside any ``try:`` body."""

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.in_try = 0
            self.found: list[tuple[str, int]] = []

        def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
            self.in_try += 1
            for stmt in node.body:
                self.visit(stmt)
            self.in_try -= 1
            for sub in [*node.handlers, *node.orelse, *node.finalbody]:
                self.visit(sub)

        visit_TryStar = visit_Try  # noqa: N815

        def _add(self, module: str, lineno: int) -> None:
            top = module.split(".")[0]
            if self.in_try or top in sys.stdlib_module_names or top in ("backend", "cli"):
                return
            self.found.append((top, lineno))

        def visit_Import(self, node: ast.Import) -> None:  # noqa: N802
            for alias in node.names:
                self._add(alias.name, node.lineno)

        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # noqa: N802
            if node.level == 0 and node.module:
                self._add(node.module, node.lineno)

    visitor = Visitor()
    visitor.visit(ast.parse(path.read_text(), filename=str(path)))
    return visitor.found


def test_core_requirements_cover_backend_and_cli_imports() -> None:
    """Unguarded third-party imports in backend/ (non-test) and cli/ must be core deps."""
    core = _requirement_names(REPO_ROOT / "requirements.txt")
    missing: list[str] = []
    for top in ("backend", "cli"):
        for path in sorted((REPO_ROOT / top).rglob("*.py")):
            if path.name.startswith("._") or "tests" in path.parts or "__pycache__" in path.parts:
                continue
            for module, lineno in _unguarded_third_party_imports(path):
                dist = _IMPORT_TO_DIST.get(module, module).lower().replace("_", "-")
                if dist not in core:
                    missing.append(f"{path.relative_to(REPO_ROOT)}:{lineno} imports {module}")
    assert missing == [], (
        "Unguarded imports not declared in requirements.txt (declare them, or guard "
        "optional imports with try/except ImportError):\n" + "\n".join(missing)
    )
