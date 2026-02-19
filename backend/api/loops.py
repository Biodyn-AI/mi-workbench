"""Loop definition management API routes."""
from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.config import config
from backend.models import LoopDefinition

try:
    import yaml
except ImportError:
    import json as yaml

router = APIRouter(prefix="/loops", tags=["loops"])


def _loops_root() -> Path:
    return Path(config.loops_dir)


def _loop_path(name: str) -> Path:
    return _loops_root() / f"{name}.yaml"


@router.get("", response_model=list[LoopDefinition])
async def list_loops() -> list[LoopDefinition]:
    root = _loops_root()
    results: list[LoopDefinition] = []
    if not root.exists():
        return results
    for yaml_file in sorted(root.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f) or {}
            data.setdefault("name", yaml_file.stem)
            results.append(LoopDefinition(**data))
        except Exception:
            continue
    return results


@router.get("/{name}", response_model=LoopDefinition)
async def get_loop(name: str) -> LoopDefinition:
    path = _loop_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Loop '{name}' not found")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("name", name)
    return LoopDefinition(**data)


@router.put("/{name}", response_model=LoopDefinition)
async def save_loop(name: str, body: LoopDefinition) -> LoopDefinition:
    body.name = name
    path = _loop_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = body.model_dump(mode="json")
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return body
