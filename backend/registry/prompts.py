"""
Prompt registry for MI-Workbench.
Loads, saves, lists, and resolves prompt templates from YAML files.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from backend.models import PromptMetadata, PromptOutputField, PromptTemplate, PromptVariable


class PromptRegistry:
    """Registry for managing prompt YAML templates."""

    def __init__(self, prompts_dir: Optional[str] = None) -> None:
        if prompts_dir is None:
            # Default to prompts/ relative to the project root
            self._dir = Path(__file__).parent.parent.parent / "prompts"
        else:
            self._dir = Path(prompts_dir)

    @property
    def prompts_dir(self) -> Path:
        return self._dir

    def _prompt_path(self, role: str, name: str) -> Path:
        return self._dir / role / f"{name}.yaml"

    def _parse_yaml(self, data: dict) -> PromptTemplate:
        """Parse a YAML dict into a PromptTemplate model."""
        meta_raw = data.get("metadata", {})
        metadata = PromptMetadata(
            role=meta_raw.get("role", ""),
            name=meta_raw.get("name", ""),
            version=meta_raw.get("version", "1.0.0"),
            description=meta_raw.get("description", ""),
            author=meta_raw.get("author", "system"),
            changelog=meta_raw.get("changelog", []),
            tags=meta_raw.get("tags", []),
        )
        variables = [
            PromptVariable(
                name=v["name"],
                description=v.get("description", ""),
                required=v.get("required", True),
                default=v.get("default"),
            )
            for v in data.get("variables", [])
        ]
        output_schema = [
            PromptOutputField(
                name=o["name"],
                description=o.get("description", ""),
                required=o.get("required", True),
            )
            for o in data.get("output_schema", [])
        ]
        return PromptTemplate(
            metadata=metadata,
            system_prompt=data.get("system_prompt", ""),
            developer_prompt=data.get("developer_prompt", ""),
            variables=variables,
            output_schema=output_schema,
            required_artifacts=data.get("required_artifacts", []),
            stop_conditions=data.get("stop_conditions", []),
        )

    def load_prompt(self, role: str, name: str) -> PromptTemplate:
        """Load a prompt template from its YAML file."""
        path = self._prompt_path(role, name)
        if not path.exists():
            raise FileNotFoundError(f"Prompt not found: {path}")
        with open(path) as f:
            data = yaml.safe_load(f)
        return self._parse_yaml(data)

    def save_prompt(self, role: str, name: str, template: PromptTemplate) -> Path:
        """Save a prompt template to its YAML file."""
        path = self._prompt_path(role, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "metadata": {
                "role": template.metadata.role,
                "name": template.metadata.name,
                "version": template.metadata.version,
                "description": template.metadata.description,
                "author": template.metadata.author,
                "changelog": template.metadata.changelog,
                "tags": template.metadata.tags,
            },
            "system_prompt": template.system_prompt,
            "developer_prompt": template.developer_prompt,
            "variables": [
                {
                    "name": v.name,
                    "description": v.description,
                    "required": v.required,
                    **({"default": v.default} if v.default is not None else {}),
                }
                for v in template.variables
            ],
            "output_schema": [
                {
                    "name": o.name,
                    "description": o.description,
                    "required": o.required,
                }
                for o in template.output_schema
            ],
            "required_artifacts": template.required_artifacts,
            "stop_conditions": template.stop_conditions,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        return path

    def list_prompts(self) -> list[tuple[str, str, str]]:
        """List all available prompts as (role, name, version) tuples."""
        results = []
        if not self._dir.exists():
            return results
        for role_dir in sorted(self._dir.iterdir()):
            if not role_dir.is_dir() or role_dir.name.startswith("."):
                continue
            for yaml_file in sorted(role_dir.glob("*.yaml")):
                try:
                    with open(yaml_file) as f:
                        data = yaml.safe_load(f)
                    meta = data.get("metadata", {})
                    results.append((
                        meta.get("role", role_dir.name),
                        meta.get("name", yaml_file.stem),
                        meta.get("version", "unknown"),
                    ))
                except Exception:
                    # Skip malformed files
                    results.append((role_dir.name, yaml_file.stem, "error"))
        return results

    def resolve_template(self, template: PromptTemplate, variables: dict[str, str]) -> str:
        """Resolve a template's system_prompt by replacing {{var}} placeholders."""
        result = template.system_prompt
        for var in template.variables:
            placeholder = "{{" + var.name + "}}"
            if var.name in variables:
                result = result.replace(placeholder, variables[var.name])
            elif var.default is not None:
                result = result.replace(placeholder, var.default)
        return result

    def validate_variables(
        self, template: PromptTemplate, variables: dict[str, str]
    ) -> list[str]:
        """Validate that all required variables are provided.

        Returns a list of error messages (empty if valid).
        """
        errors = []
        required_names = {v.name for v in template.variables if v.required}
        provided_names = set(variables.keys())
        defined_names = {v.name for v in template.variables}

        missing = required_names - provided_names
        # Exclude variables with defaults from "missing"
        defaults = {v.name for v in template.variables if v.default is not None}
        truly_missing = missing - defaults
        if truly_missing:
            for name in sorted(truly_missing):
                errors.append(f"Missing required variable: {name}")

        extra = provided_names - defined_names
        if extra:
            for name in sorted(extra):
                errors.append(f"Unknown variable: {name}")

        return errors
