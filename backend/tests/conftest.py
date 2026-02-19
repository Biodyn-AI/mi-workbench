"""Shared test fixtures for MI-Workbench backend tests."""
from __future__ import annotations

import os
import tempfile
from typing import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Point config at a temp directory before importing anything else
_tmpdir = tempfile.mkdtemp(prefix="miw_test_")
os.environ["MIW_DB_PATH"] = os.path.join(_tmpdir, "test.db")
os.environ["MIW_WORKSPACE_BASE"] = os.path.join(_tmpdir, "workspaces")
os.environ["MIW_PROMPTS_DIR"] = os.path.join(_tmpdir, "prompts")
os.environ["MIW_LOOPS_DIR"] = os.path.join(_tmpdir, "loops")

# Re-initialise config singleton with test paths
from backend.config import config
config.db_path = os.environ["MIW_DB_PATH"]
config.workspace_base_path = os.environ["MIW_WORKSPACE_BASE"]
config.prompts_dir = os.environ["MIW_PROMPTS_DIR"]
config.loops_dir = os.environ["MIW_LOOPS_DIR"]

from backend.database import init_db
from backend.main import app


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """Provide an async HTTP client backed by the FastAPI app."""
    # Init fresh DB for each test
    await init_db()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
