#!/usr/bin/env python3
"""
MI-Workbench CLI (miw) - Command-line interface for the MI-Workbench platform.
Uses Click for command structure and httpx for backend API communication.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import click
import httpx

DEFAULT_BASE_URL = "http://localhost:8000/api"


def get_base_url() -> str:
    return os.environ.get("MIW_BASE_URL", DEFAULT_BASE_URL)


def api_client() -> httpx.Client:
    return httpx.Client(base_url=get_base_url(), timeout=30.0)


def print_json(data: dict | list) -> None:
    click.echo(json.dumps(data, indent=2, default=str))


def handle_response(resp: httpx.Response) -> dict | list:
    if resp.status_code >= 400:
        click.secho(f"Error {resp.status_code}: {resp.text}", fg="red", err=True)
        sys.exit(1)
    return resp.json()


# ── Main group ────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="0.1.0", prog_name="miw")
def cli():
    """MI-Workbench CLI - Manage workspaces, runs, prompts, and providers."""
    pass


# ── Workspace commands ────────────────────────────────────────────────

@cli.group()
def workspace():
    """Manage workspaces."""
    pass


@workspace.command("add")
@click.argument("path", type=click.Path(exists=True, resolve_path=True))
@click.option("--name", "-n", default=None, help="Workspace name (defaults to directory name)")
@click.option("--provider", "-p", default="mock", help="Default provider (claude_code, codex_cli, gemini_cli, mock)")
@click.option("--git-mode", "-g", default="none", help="Git mode (none, commit_per_run, commit_per_iteration, branch_per_run)")
def workspace_add(path: str, name: Optional[str], provider: str, git_mode: str):
    """Register a workspace directory."""
    if name is None:
        name = Path(path).name
    payload = {
        "name": name,
        "path": path,
        "default_provider": provider,
        "git_mode": git_mode,
    }
    with api_client() as client:
        resp = client.post("/workspaces", json=payload)
        data = handle_response(resp)
    click.secho(f"Workspace '{data['name']}' registered (id: {data['id']})", fg="green")


@workspace.command("list")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def workspace_list(json_output: bool):
    """List all registered workspaces."""
    with api_client() as client:
        resp = client.get("/workspaces")
        data = handle_response(resp)
    if json_output:
        print_json(data)
        return
    if not data:
        click.echo("No workspaces registered. Use 'miw workspace add <path>' to add one.")
        return
    click.echo(f"{'ID':<14} {'Name':<25} {'Provider':<15} {'Git Mode':<20} Path")
    click.echo("-" * 100)
    for ws in data:
        click.echo(
            f"{ws['id']:<14} {ws['name']:<25} {ws['default_provider']:<15} "
            f"{ws['git_mode']:<20} {ws['path']}"
        )


# ── Run commands ──────────────────────────────────────────────────────

@cli.group()
def run():
    """Manage runs."""
    pass


def parse_config_pairs(pairs: tuple[str, ...] | list[str]) -> dict:
    """Parse ``KEY=VALUE`` options into a run-config dict.

    Values are decoded as JSON when possible (``3``, ``0.5``, ``true``,
    ``["a","b"]``, ``{"flag": true}``), otherwise kept as strings.
    """
    config: dict = {}
    for pair in pairs or ():
        if "=" not in pair:
            raise click.BadParameter(f"expected KEY=VALUE, got {pair!r}", param_hint="--config")
        key, value = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise click.BadParameter(f"empty key in {pair!r}", param_hint="--config")
        try:
            config[key] = json.loads(value)
        except ValueError:
            config[key] = value
    return config


def build_run_payload(
    workspace_id: str,
    loop_preset: str,
    task: str,
    provider: str,
    model: str,
    max_iterations: int,
    effort: str = "",
    config_pairs: tuple[str, ...] | list[str] = (),
) -> dict:
    """Request body for ``POST /runs``."""
    payload = {
        "workspace_id": workspace_id,
        "loop_preset": loop_preset,
        "task": task,
        "provider": provider,
        "model": model,
        "max_iterations": max_iterations,
        "config_overrides": parse_config_pairs(config_pairs),
    }
    if effort:
        payload["reasoning_effort"] = effort
    return payload


_RUN_CONFIG_HELP = (
    "Run-config override KEY=VALUE (repeatable; JSON values), e.g. "
    "-c convergence_rule=all -c code_execution_enabled=true -c budget_max_tokens=2000000"
)


@run.command("preset")
@click.argument("preset_name")
@click.option("--workspace-id", "-w", required=True, help="Workspace ID")
@click.option("--task", "-t", required=True, help="Task description")
@click.option("--provider", "-p", default="mock", help="Provider to use")
@click.option("--model", "-m", default="", help="Model name (forwarded to the CLI adapter)")
@click.option("--effort", "-e", default="", help="Reasoning effort (e.g. low, medium, high)")
@click.option("--max-iterations", "-i", default=50, help="Maximum iterations")
@click.option("--config", "-c", "config_pairs", multiple=True, help=_RUN_CONFIG_HELP)
def run_preset(preset_name: str, workspace_id: str, task: str, provider: str, model: str,
               effort: str, max_iterations: int, config_pairs: tuple[str, ...]):
    """Start a run using a preset loop (executor_reviewer, reviewer_consensus,
    research_followups, example_custom, or any loops/<name>.yaml)."""
    payload = build_run_payload(workspace_id, preset_name, task, provider, model,
                                max_iterations, effort, config_pairs)
    with api_client() as client:
        resp = client.post("/runs", json=payload)
        data = handle_response(resp)
    click.secho(f"Run started (id: {data['run_id']}, preset: {preset_name})", fg="green")
    click.echo(f"Status: {data['status']}")


@run.command("loop")
@click.argument("yaml_path", type=click.Path(exists=True, resolve_path=True))
@click.option("--workspace-id", "-w", required=True, help="Workspace ID")
@click.option("--task", "-t", required=True, help="Task description")
@click.option("--provider", "-p", default="mock", help="Provider to use")
@click.option("--model", "-m", default="", help="Model name (forwarded to the CLI adapter)")
@click.option("--effort", "-e", default="", help="Reasoning effort (e.g. low, medium, high)")
@click.option("--max-iterations", "-i", default=50, help="Maximum iterations")
@click.option("--config", "-c", "config_pairs", multiple=True, help=_RUN_CONFIG_HELP)
def run_loop(yaml_path: str, workspace_id: str, task: str, provider: str, model: str,
             effort: str, max_iterations: int, config_pairs: tuple[str, ...]):
    """Start a run using a custom loop YAML definition (sent as custom:<path>)."""
    payload = build_run_payload(workspace_id, f"custom:{yaml_path}", task, provider, model,
                                max_iterations, effort, config_pairs)
    with api_client() as client:
        resp = client.post("/runs", json=payload)
        data = handle_response(resp)
    click.secho(f"Run started (id: {data['run_id']}, loop: {yaml_path})", fg="green")
    click.echo(f"Status: {data['status']}")


@run.command("list")
@click.option("--workspace-id", "-w", default=None, help="Filter by workspace ID")
@click.option("--status", "-s", default=None, help="Filter by status")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def run_list(workspace_id: Optional[str], status: Optional[str], json_output: bool):
    """List runs."""
    params = {}
    if workspace_id:
        params["workspace_id"] = workspace_id
    if status:
        params["status"] = status
    with api_client() as client:
        resp = client.get("/runs", params=params)
        data = handle_response(resp)
    if json_output:
        print_json(data)
        return
    if not data:
        click.echo("No runs found.")
        return
    click.echo(f"{'Run ID':<14} {'Status':<12} {'Preset':<22} {'Iter':<6} {'Cost':<10} Task")
    click.echo("-" * 100)
    for r in data:
        task_preview = r.get("task", "")[:40]
        click.echo(
            f"{r['run_id']:<14} {r['status']:<12} {r['loop_preset']:<22} "
            f"{r['current_iteration']:<6} ${r.get('total_cost', 0):<9.4f} {task_preview}"
        )


@run.command("repropack")
@click.argument("run_id")
@click.option("--output", "-o", default=None, help="Output file path (default: repropack-<run_id>.zip)")
@click.option("--preview", is_flag=True, help="Preview files without downloading")
@click.option("--max-file-mb", type=float, default=None,
              help="Per-file size cap in MB for iteration files and persisted execution "
                   "outputs (larger files are skipped and listed in the package README; "
                   "default: server setting, 25 MB; 0 = no cap)")
def run_repropack(run_id: str, output: Optional[str], preview: bool,
                  max_file_mb: Optional[float]):
    """Download a reproducibility package for a run."""
    params = {} if max_file_mb is None else {"max_file_mb": max_file_mb}
    if preview:
        with api_client() as client:
            resp = client.get(f"/repropack/{run_id}/preview", params=params)
            data = handle_response(resp)
        click.echo(f"Repro pack for run {run_id} ({data['total_files']} files):")
        click.echo("")
        for f in data["files"]:
            if f.get("skipped"):
                size_str = f"skipped: {f['skipped']}"
            else:
                size_str = f"{f['size']:,} bytes" if f["size"] > 0 else "generated"
            click.echo(f"  {f['path']:<50} {size_str}")
        return

    output_path = output or f"repropack-{run_id}.zip"
    click.echo(f"Generating repro pack for run {run_id}...")
    with httpx.Client(base_url=get_base_url(), timeout=120.0) as client:
        resp = client.post(f"/repropack/{run_id}", params=params)
        if resp.status_code >= 400:
            click.secho(f"Error {resp.status_code}: {resp.text}", fg="red", err=True)
            sys.exit(1)
        Path(output_path).write_bytes(resp.content)
    click.secho(f"Repro pack saved to {output_path} ({len(resp.content):,} bytes)", fg="green")


@cli.command("stop")
@click.argument("run_id")
def stop_run(run_id: str):
    """Stop a running run."""
    with api_client() as client:
        resp = client.post(f"/runs/{run_id}/stop")
        data = handle_response(resp)
    click.secho(f"Run {run_id} stopped.", fg="yellow")


# ── Prompt commands ───────────────────────────────────────────────────

@cli.group()
def prompts():
    """Manage prompt templates."""
    pass


@prompts.command("list")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def prompts_list(json_output: bool):
    """List all available prompt templates."""
    with api_client() as client:
        resp = client.get("/prompts")
        data = handle_response(resp)
    if json_output:
        print_json(data)
        return
    if not data:
        click.echo("No prompts found.")
        return
    click.echo(f"{'Role':<25} {'Name':<30} {'Version':<10} Description")
    click.echo("-" * 100)
    for p in data:
        desc = p.get("description", "")[:40]
        click.echo(f"{p['role']:<25} {p['name']:<30} {p['version']:<10} {desc}")


@prompts.command("edit")
@click.argument("role")
@click.argument("name")
def prompts_edit(role: str, name: str):
    """Open a prompt template in $EDITOR for editing."""
    # Resolve the prompt file path relative to the prompts directory
    prompts_dir = Path(__file__).parent.parent / "prompts"
    prompt_path = prompts_dir / role / f"{name}.yaml"
    if not prompt_path.exists():
        click.secho(f"Prompt file not found: {prompt_path}", fg="red", err=True)
        sys.exit(1)
    editor = os.environ.get("EDITOR", "vi")
    click.echo(f"Opening {prompt_path} in {editor}...")
    subprocess.run([editor, str(prompt_path)])


# ── Provider commands ─────────────────────────────────────────────────

@cli.group()
def providers():
    """Manage providers."""
    pass


@providers.command("list")
@click.option("--json-output", "-j", is_flag=True, help="Output as JSON")
def providers_list(json_output: bool):
    """List available providers and their status."""
    with api_client() as client:
        resp = client.get("/providers")
        # If the endpoint doesn't exist yet, fall back to a static list
        if resp.status_code == 404:
            data = [
                {"name": "claude_code", "available": True, "description": "Claude Code SDK adapter"},
                {"name": "codex_cli", "available": False, "description": "OpenAI Codex CLI adapter"},
                {"name": "gemini_cli", "available": False, "description": "Google Gemini CLI adapter"},
                {"name": "mock", "available": True, "description": "Mock adapter for testing"},
            ]
        else:
            data = handle_response(resp)
    if json_output:
        print_json(data)
        return
    click.echo(f"{'Name':<20} {'Available':<12} Description")
    click.echo("-" * 60)
    for p in data:
        status = click.style("yes", fg="green") if p.get("available") else click.style("no", fg="red")
        click.echo(f"{p['name']:<20} {status:<21} {p.get('description', '')}")


@providers.command("smoketest")
@click.argument("name")
def providers_smoketest(name: str):
    """Run a smoke test against a provider."""
    click.echo(f"Running smoke test for provider '{name}'...")
    with api_client() as client:
        resp = client.post(f"/providers/{name}/smoketest")
        if resp.status_code == 404:
            click.secho(f"Provider '{name}' not found or smoketest endpoint not available.", fg="red", err=True)
            sys.exit(1)
        data = handle_response(resp)
    if data.get("success"):
        click.secho(f"Smoke test passed for '{name}'.", fg="green")
        if data.get("output"):
            click.echo(f"Output: {data['output'][:200]}")
    else:
        click.secho(f"Smoke test failed for '{name}': {data.get('error', 'unknown error')}", fg="red")


# ── Subsystem commands ────────────────────────────────────────────────

@cli.group()
def subsystem():
    """Manage subsystems."""
    pass


@subsystem.command("init")
@click.argument("name")
def subsystem_init(name: str):
    """Scaffold a new subsystem with manifest and directory structure."""
    subsystems_dir = Path(__file__).parent.parent / "subsystems" / name
    if subsystems_dir.exists():
        click.secho(f"Subsystem '{name}' already exists at {subsystems_dir}", fg="yellow", err=True)
        sys.exit(1)
    subsystems_dir.mkdir(parents=True)
    manifest = {
        "name": name,
        "version": "0.1.0",
        "description": f"TODO: Describe the {name} subsystem",
        "roles": [],
        "artifact_types": [],
        "loop_presets": [],
        "validation_rules": [],
        "ui_panels": [],
    }
    manifest_path = subsystems_dir / "manifest.yaml"
    import yaml
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
    # Create placeholder directories
    (subsystems_dir / "prompts").mkdir()
    (subsystems_dir / "loops").mkdir()
    click.secho(f"Subsystem '{name}' scaffolded at {subsystems_dir}", fg="green")
    click.echo(f"  - {manifest_path}")
    click.echo(f"  - {subsystems_dir / 'prompts'}/")
    click.echo(f"  - {subsystems_dir / 'loops'}/")
    click.echo("Edit the manifest.yaml to configure your subsystem.")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
