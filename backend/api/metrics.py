"""Prometheus-compatible metrics endpoint."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from backend.database import list_runs
from backend.orchestrator.runner import get_active_run_ids

router = APIRouter(tags=["metrics"])


class MetricsCollector:
    """Lightweight in-memory counters (no external dependency)."""

    def __init__(self) -> None:
        self.requests_total: int = 0
        self.iterations_total: int = 0
        self.adapter_errors_total: int = 0
        self.convergence_triggers: int = 0

    def inc_requests(self) -> None:
        self.requests_total += 1

    def inc_iterations(self) -> None:
        self.iterations_total += 1

    def inc_adapter_errors(self) -> None:
        self.adapter_errors_total += 1

    def inc_convergence(self) -> None:
        self.convergence_triggers += 1


_metrics = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _metrics


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """Return metrics in Prometheus text exposition format."""
    active_runs = len(get_active_run_ids())
    all_runs = await list_runs()
    total_runs = len(all_runs)
    completed = sum(1 for r in all_runs if r.status.value == "completed")
    failed = sum(1 for r in all_runs if r.status.value == "failed")

    lines = [
        "# HELP miw_active_runs Number of currently executing runs",
        "# TYPE miw_active_runs gauge",
        f"miw_active_runs {active_runs}",
        "",
        "# HELP miw_total_runs Total number of runs created",
        "# TYPE miw_total_runs counter",
        f"miw_total_runs {total_runs}",
        "",
        "# HELP miw_completed_runs Total completed runs",
        "# TYPE miw_completed_runs counter",
        f"miw_completed_runs {completed}",
        "",
        "# HELP miw_failed_runs Total failed runs",
        "# TYPE miw_failed_runs counter",
        f"miw_failed_runs {failed}",
        "",
        "# HELP miw_iterations_total Total iterations executed",
        "# TYPE miw_iterations_total counter",
        f"miw_iterations_total {_metrics.iterations_total}",
        "",
        "# HELP miw_adapter_errors_total Total adapter errors",
        "# TYPE miw_adapter_errors_total counter",
        f"miw_adapter_errors_total {_metrics.adapter_errors_total}",
        "",
        "# HELP miw_convergence_triggers_total Times convergence detection fired",
        "# TYPE miw_convergence_triggers_total counter",
        f"miw_convergence_triggers_total {_metrics.convergence_triggers}",
    ]

    return "\n".join(lines) + "\n"
