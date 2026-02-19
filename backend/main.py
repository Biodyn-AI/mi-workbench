"""MI-Workbench FastAPI application entry point."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.config import config, WORKBENCH_DIR
from backend.database import get_db, init_db
from backend.logging_config import setup_logging
from backend.orchestrator.recovery import recover_interrupted_runs
from backend.orchestrator.runner import get_active_run_ids, set_ws_broadcast

from backend.api.workspaces import router as workspaces_router
from backend.api.runs import router as runs_router
from backend.api.prompts import router as prompts_router
from backend.api.loops import router as loops_router
from backend.api.providers import router as providers_router
from backend.api.artifacts import router as artifacts_router
from backend.api.settings import router as settings_router
from backend.api.repropack import router as repropack_router
from backend.api.claims import router as claims_router
from backend.api.telemetry import router as telemetry_router
from backend.api.knowledge import router as knowledge_router
from backend.api.comparison import router as comparison_router
from backend.api.metrics import router as metrics_router

logger = logging.getLogger("mi-workbench")


# ── WebSocket connection manager ─────────────────────────────────────

class ConnectionManager:
    """Track WebSocket connections per run_id for log streaming."""

    def __init__(self) -> None:
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, run_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.setdefault(run_id, []).append(ws)

    def disconnect(self, run_id: str, ws: WebSocket) -> None:
        conns = self._connections.get(run_id, [])
        if ws in conns:
            conns.remove(ws)

    async def broadcast(self, run_id: str, message: dict) -> None:
        for ws in self._connections.get(run_id, []):
            try:
                await ws.send_json(message)
            except Exception:
                pass


ws_manager = ConnectionManager()


# ── Lifespan ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    # Startup
    json_format = os.environ.get("MIW_LOG_JSON", "1") != "0"
    setup_logging(level=config.log_level, json_format=json_format)
    logger.info("Initialising MI-Workbench backend...")
    await init_db()
    # Ensure key directories
    for d in [config.workspace_base_path, config.prompts_dir, config.loops_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)
    Path(config.db_path).parent.mkdir(parents=True, exist_ok=True)
    # Recover interrupted runs from previous server session
    recovered = await recover_interrupted_runs()
    if recovered:
        logger.info("Recovered %d interrupted runs: %s", len(recovered), recovered)
    # Wire WebSocket broadcast to the run executor
    set_ws_broadcast(ws_manager.broadcast)
    logger.info("MI-Workbench backend ready on %s:%s", config.host, config.port)
    yield
    # Shutdown
    logger.info("MI-Workbench backend shutting down.")


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="MI-Workbench",
    description="Mechanistic Interpretability Research Automation Platform",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(workspaces_router, prefix="/api")
app.include_router(runs_router, prefix="/api")
app.include_router(prompts_router, prefix="/api")
app.include_router(loops_router, prefix="/api")
app.include_router(providers_router, prefix="/api")
app.include_router(artifacts_router, prefix="/api")
app.include_router(settings_router, prefix="/api")
app.include_router(repropack_router, prefix="/api")
app.include_router(claims_router, prefix="/api")
app.include_router(telemetry_router, prefix="/api")
app.include_router(knowledge_router, prefix="/api")
app.include_router(comparison_router, prefix="/api")
app.include_router(metrics_router)  # Prometheus expects /metrics at root


# ── WebSocket for run streaming ──────────────────────────────────────

@app.websocket("/ws/runs/{run_id}")
async def run_websocket(websocket: WebSocket, run_id: str) -> None:
    await ws_manager.connect(run_id, websocket)
    try:
        while True:
            # Keep connection alive; client can send control messages
            data = await websocket.receive_text()
            # Echo back as acknowledgment
            await websocket.send_json({"type": "ack", "data": data})
    except WebSocketDisconnect:
        ws_manager.disconnect(run_id, websocket)


# ── Health check ─────────────────────────────────────────────────────

@app.get("/api/health")
async def health() -> dict:
    """Health check that verifies DB connectivity."""
    db_ok = False
    try:
        async with get_db() as db:
            await db.execute("SELECT 1")
            db_ok = True
    except Exception:
        pass

    active_runs = len(get_active_run_ids())

    return {
        "status": "ok" if db_ok else "degraded",
        "version": "0.1.0",
        "database": "ok" if db_ok else "error",
        "active_runs": active_runs,
    }


# ── CLI entry point ──────────────────────────────────────────────────

def start() -> None:
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=config.host,
        port=config.port,
        reload=True,
        log_level=config.log_level.lower(),
    )


if __name__ == "__main__":
    start()
