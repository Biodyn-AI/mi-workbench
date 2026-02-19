"""Prompt template management API routes."""
from __future__ import annotations

import re
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.config import config
from backend.models import PromptMetadata, PromptTemplate, PromptUpdate

try:
    import yaml
except ImportError:
    import json as yaml  # fallback, but yaml should be installed

router = APIRouter(prefix="/prompts", tags=["prompts"])


def _prompts_root() -> Path:
    return Path(config.prompts_dir)


def _prompt_path(role: str, name: str) -> Path:
    return _prompts_root() / role / f"{name}.yaml"


def _load_prompt(role: str, name: str) -> PromptTemplate:
    path = _prompt_path(role, name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Prompt {role}/{name} not found")
    with open(path) as f:
        data = yaml.safe_load(f)
    # Ensure metadata has role/name
    if "metadata" not in data:
        data["metadata"] = {}
    data["metadata"]["role"] = role
    data["metadata"]["name"] = name
    return PromptTemplate(**data)


def _save_prompt(role: str, name: str, template: PromptTemplate) -> None:
    path = _prompt_path(role, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = template.model_dump(mode="json")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


@router.get("", response_model=list[PromptMetadata])
async def list_prompts() -> list[PromptMetadata]:
    root = _prompts_root()
    results: list[PromptMetadata] = []
    if not root.exists():
        return results
    for role_dir in sorted(root.iterdir()):
        if not role_dir.is_dir():
            continue
        for yaml_file in sorted(role_dir.glob("*.yaml")):
            try:
                with open(yaml_file) as f:
                    data = yaml.safe_load(f) or {}
                meta = data.get("metadata", {})
                meta["role"] = role_dir.name
                meta["name"] = yaml_file.stem
                results.append(PromptMetadata(**meta))
            except Exception:
                continue
    return results


@router.get("/{role}/{name}", response_model=PromptTemplate)
async def get_prompt(role: str, name: str) -> PromptTemplate:
    return _load_prompt(role, name)


@router.put("/{role}/{name}", response_model=PromptTemplate)
async def update_prompt(role: str, name: str, body: PromptUpdate) -> PromptTemplate:
    try:
        template = _load_prompt(role, name)
    except HTTPException:
        # Create new prompt
        template = PromptTemplate(
            metadata=PromptMetadata(role=role, name=name),
            system_prompt="",
        )
    if body.system_prompt is not None:
        template.system_prompt = body.system_prompt
    if body.developer_prompt is not None:
        template.developer_prompt = body.developer_prompt
    if body.variables is not None:
        template.variables = body.variables
    if body.output_schema is not None:
        template.output_schema = body.output_schema
    if body.changelog_entry:
        template.metadata.changelog.append(body.changelog_entry)
    _save_prompt(role, name, template)
    return template


@router.post("/{role}/{name}/validate", response_model=dict)
async def validate_prompt(role: str, name: str) -> dict:
    template = _load_prompt(role, name)
    issues: list[str] = []
    # Check that all required variables appear in the system prompt
    combined = template.system_prompt + template.developer_prompt
    for var in template.variables:
        pattern = r"\{\{\s*" + re.escape(var.name) + r"\s*\}\}"
        if var.required and not re.search(pattern, combined):
            issues.append(f"Required variable '{var.name}' not found in prompts")
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "variable_count": len(template.variables),
    }
