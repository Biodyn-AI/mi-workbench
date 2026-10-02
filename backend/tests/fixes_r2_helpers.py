"""Shared helpers for the revision-2 regression tests (test_fixes_r2_*.py)."""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional

from backend.adapters.mock import MockAdapter
from backend.models import AdapterRunResult, ProviderName, RunState

LENSES = ("reviewer", "adversarial_reviewer", "bio_plausibility_checker")
REVIEW_ROLES = ("reviewer", "adversarial_reviewer", "adversarial", "bio_plausibility_checker")


def json_block(critiques: list[dict], extra: str = "") -> str:
    import json
    return "Notes.\n```json\n" + json.dumps({"critiques": critiques}) + "\n```" + extra


class ScriptedReviewAdapter:
    """Mock executor; review / lens roles answer with ``review_fn(role, request, n)``
    (``n`` = how many times this role was called so far, starting at 1). A
    returned string is a successful output; an ``AdapterRunResult`` is returned
    as is."""

    name = "scripted"

    def __init__(self, review_fn: Callable[[str, Any, int], Any],
                 roles: tuple[str, ...] = REVIEW_ROLES):
        self.inner = MockAdapter(min_delay=0, max_delay=0)
        self.review_fn = review_fn
        self.roles = set(roles)
        self.calls: dict[str, int] = {}
        self.requests: list[Any] = []

    async def run(self, request):
        self.requests.append(request)
        role = request.prompt_bundle.variables.get("role", "")
        if role in self.roles:
            self.calls[role] = self.calls.get(role, 0) + 1
            out = self.review_fn(role, request, self.calls[role])
            if isinstance(out, AdapterRunResult):
                return out
            return AdapterRunResult(success=True, output=out, provider="scripted",
                                    model=request.model or "m", input_tokens=10,
                                    output_tokens=5, token_usage=15)
        res = await self.inner.run(request)
        res.provider = "scripted"
        return res

    async def smoke_test(self):
        return {"status": "ok"}

    def is_available(self):
        return True

    def executor_requests(self):
        return [r for r in self.requests
                if r.prompt_bundle.variables.get("role") == "executor"]


def run_engine(adapter, loop, max_iterations: int = 6, config: Optional[dict] = None,
               events: Optional[list] = None, run_state: Optional[RunState] = None,
               resume_from: int = 0, workspace_path: str = "", artifact_writer=None,
               workspace_defaults: Optional[dict] = None, cancel_event=None):
    from backend.orchestrator.engine import LoopEngine

    MockAdapter.reset_iteration_count()

    async def go():
        eng = LoopEngine(
            adapter=adapter,
            on_event=(lambda t, d: events.append((t, d))) if events is not None else None,
            artifact_writer=artifact_writer,
            max_retries=0,
        )
        run = run_state or RunState(
            workspace_id="w", loop_preset="x", task="t", provider=ProviderName.MOCK,
            model="", max_iterations=max_iterations, config=dict(config or {}))
        final = await eng.run_loop(run, loop, cancel_event=cancel_event,
                                   workspace_path=workspace_path, resume_from=resume_from,
                                   workspace_defaults=workspace_defaults)
        return final, eng

    return asyncio.run(go())
