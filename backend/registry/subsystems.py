"""
Subsystem registry for MI-Workbench.
Manages subsystem manifests: loading, listing, and scaffolding new subsystems.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from backend.models import SubsystemManifest


class SubsystemRegistry:
    """Registry for managing MI-Workbench subsystems."""

    def __init__(self, subsystems_dir: Optional[str] = None) -> None:
        if subsystems_dir is None:
            self._dir = Path(__file__).parent.parent.parent / "subsystems"
        else:
            self._dir = Path(subsystems_dir)

    @property
    def subsystems_dir(self) -> Path:
        return self._dir

    def load_subsystem(self, name: str) -> SubsystemManifest:
        """Load a subsystem manifest by name."""
        manifest_path = self._dir / name / "manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Subsystem manifest not found: {manifest_path}")
        with open(manifest_path) as f:
            data = yaml.safe_load(f)
        return SubsystemManifest(
            name=data.get("name", name),
            version=data.get("version", "0.1.0"),
            description=data.get("description", ""),
            roles=data.get("roles", []),
            artifact_types=data.get("artifact_types", []),
            loop_presets=data.get("loop_presets", []),
            validation_rules=data.get("validation_rules", []),
            ui_panels=data.get("ui_panels", []),
        )

    def list_subsystems(self) -> list[SubsystemManifest]:
        """List all subsystems with valid manifests."""
        results = []
        if not self._dir.exists():
            return results
        for sub_dir in sorted(self._dir.iterdir()):
            if not sub_dir.is_dir() or sub_dir.name.startswith("."):
                continue
            manifest_path = sub_dir / "manifest.yaml"
            if manifest_path.exists():
                try:
                    results.append(self.load_subsystem(sub_dir.name))
                except Exception:
                    # Skip malformed manifests
                    pass
        return results

    def init_subsystem(self, name: str) -> Path:
        """Scaffold a new subsystem with a minimal manifest and directory structure.

        Returns the path to the created manifest.
        Raises FileExistsError if the subsystem already exists.
        """
        sub_dir = self._dir / name
        if sub_dir.exists():
            raise FileExistsError(f"Subsystem '{name}' already exists at {sub_dir}")

        sub_dir.mkdir(parents=True)
        (sub_dir / "prompts").mkdir()
        (sub_dir / "loops").mkdir()

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
        manifest_path = sub_dir / "manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)

        return manifest_path
