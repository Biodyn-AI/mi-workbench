"""API / CLI surface of the revision-2 run settings: preset validation,
custom:<path> loops, reasoning effort, stop_reason and accounting fields."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest
from httpx import AsyncClient

from backend.orchestrator.presets import BUNDLED_LOOPS_DIR

REPO = Path(__file__).resolve().parents[2]


async def _ws(client: AsyncClient, tmp_path: Path, name: str = "cfg-ws") -> str:
    resp = await client.post("/api/workspaces", json={"name": name, "path": str(tmp_path / name)})
    assert resp.status_code == 201
    return resp.json()["id"]


async def _poll(client: AsyncClient, run_id: str, timeout: float = 30.0) -> dict:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        data = (await client.get(f"/api/runs/{run_id}")).json()
        if data["status"] not in ("pending", "running"):
            return data
        await asyncio.sleep(0.1)
    raise AssertionError("run did not finish")


@pytest.mark.asyncio
async def test_unknown_preset_rejected_at_creation(client: AsyncClient, tmp_path: Path):
    ws = await _ws(client, tmp_path)
    resp = await client.post("/api/runs", json={"workspace_id": ws, "task": "t",
                                                "loop_preset": "no_such_loop"})
    assert resp.status_code == 422
    assert "Unknown preset" in resp.json()["detail"]
    resp = await client.post("/api/runs", json={"workspace_id": ws, "task": "t",
                                                "loop_preset": "custom:/no/such/file.yaml"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_custom_loop_and_effort_run_to_completion(client: AsyncClient, tmp_path: Path):
    ws = await _ws(client, tmp_path)
    resp = await client.post("/api/runs", json={
        "workspace_id": ws, "task": "t", "provider": "mock",
        "loop_preset": f"custom:{BUNDLED_LOOPS_DIR / 'example_custom.yaml'}",
        "max_iterations": 2, "reasoning_effort": "low",
        "config_overrides": {"convergence_enabled": False, "grade_at_least": "A+"},
    })
    assert resp.status_code == 201, resp.text
    run = resp.json()
    assert run["config"]["reasoning_effort"] == "low"
    final = await _poll(client, run["run_id"])
    assert final["status"] == "completed", final.get("error")
    assert final["stop_reason"] == "max_iterations"
    assert final["total_input_tokens"] > 0
    its = final["iterations"]
    assert len(its) == 2
    assert all(it["reasoning_effort"] == "low" and it["provider"] == "mock" for it in its)
    one = (await client.get(f"/api/runs/{run['run_id']}/iterations/1")).json()
    assert one["input_tokens"] > 0 and one["cli_version"] == "mock"


@pytest.mark.asyncio
async def test_config_override_effort_wins(client: AsyncClient, tmp_path: Path):
    ws = await _ws(client, tmp_path, "cfg-ws2")
    resp = await client.post("/api/runs", json={
        "workspace_id": ws, "task": "t", "reasoning_effort": "low",
        "config_overrides": {"reasoning_effort": "high"}, "max_iterations": 1,
    })
    assert resp.status_code == 201
    assert resp.json()["config"]["reasoning_effort"] == "high"


@pytest.mark.asyncio
async def test_example_custom_by_name(client: AsyncClient, tmp_path: Path):
    ws = await _ws(client, tmp_path, "cfg-ws3")
    resp = await client.post("/api/runs", json={
        "workspace_id": ws, "task": "t", "loop_preset": "example_custom", "max_iterations": 2,
        "config_overrides": {"convergence_enabled": False},
    })
    assert resp.status_code == 201
    final = await _poll(client, resp.json()["run_id"])
    assert final["status"] == "completed"
    assert final["stop_reason"] in ("max_iterations", "stop_condition:grade_at_least")


# ── CLI payload helpers ──────────────────────────────────────────────


def _load_cli():
    spec = importlib.util.spec_from_file_location("miw_cli", REPO / "cli" / "miw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_config_pairs_and_payload():
    cli = _load_cli()
    cfg = cli.parse_config_pairs(["convergence_rule=all", "budget_max_tokens=2000000",
                                  "code_execution_enabled=true", "flags={\"halt\": false}",
                                  "convergence_signals=[\"grade_stable\"]", "note=plain text"])
    assert cfg == {"convergence_rule": "all", "budget_max_tokens": 2000000,
                   "code_execution_enabled": True, "flags": {"halt": False},
                   "convergence_signals": ["grade_stable"], "note": "plain text"}
    payload = cli.build_run_payload("ws1", "custom:/x/loop.yaml", "task", "codex_cli",
                                    "gpt-5.5", 10, "medium", ["convergence_window=2"])
    assert payload["loop_preset"] == "custom:/x/loop.yaml"
    assert payload["reasoning_effort"] == "medium"
    assert payload["config_overrides"] == {"convergence_window": 2}
    assert payload["max_iterations"] == 10
    import click
    with pytest.raises(click.BadParameter):
        cli.parse_config_pairs(["no_equals_sign"])


def test_cli_commands_expose_effort_and_config():
    from click.testing import CliRunner
    cli = _load_cli()
    runner = CliRunner()
    for cmd in (["run", "preset", "--help"], ["run", "loop", "--help"]):
        out = runner.invoke(cli.cli, cmd).output
        assert "--effort" in out and "--config" in out and "--max-iterations" in out
