"""Provider status and smoke-test API routes."""
from __future__ import annotations

import shutil

from fastapi import APIRouter, HTTPException

from backend.models import ProviderName

router = APIRouter(prefix="/providers", tags=["providers"])

# Map provider names to CLI executables for availability checks
_CLI_MAP: dict[ProviderName, str] = {
    ProviderName.CLAUDE_CODE: "claude",
    ProviderName.CODEX_CLI: "codex",
    ProviderName.GEMINI_CLI: "gemini",
    ProviderName.MOCK: "__mock__",
}


def _check_available(provider: ProviderName) -> bool:
    if provider == ProviderName.MOCK:
        return True
    exe = _CLI_MAP.get(provider)
    if exe is None:
        return False
    return shutil.which(exe) is not None


@router.get("", response_model=list[dict])
async def list_providers() -> list[dict]:
    results = []
    for p in ProviderName:
        results.append({
            "name": p.value,
            "available": _check_available(p),
        })
    return results


@router.post("/{name}/smoketest", response_model=dict)
async def smoke_test(name: str) -> dict:
    try:
        provider = ProviderName(name)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Unknown provider '{name}'")

    if provider == ProviderName.MOCK:
        return {"provider": name, "success": True, "message": "Mock provider always passes"}

    available = _check_available(provider)
    if not available:
        return {
            "provider": name,
            "success": False,
            "message": f"CLI for '{name}' not found on PATH",
        }

    return {
        "provider": name,
        "success": True,
        "message": f"CLI for '{name}' found on PATH",
    }
