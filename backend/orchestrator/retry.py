"""Retry wrapper for adapter calls with exponential backoff."""
import asyncio
import logging
import random
from typing import Optional

from backend.adapters.base import BaseAdapter
from backend.models import AdapterRunRequest, AdapterRunResult

logger = logging.getLogger(__name__)

import re

#: Adapter error categories (the ``[category]`` prefix written by
#: ``backend.adapters.cli_common.format_error``) that are worth retrying.
TRANSIENT_CATEGORIES = frozenset({"rate_limit", "timeout", "network", "overloaded"})
#: Categories that are never retried, whatever the detail text says.
PERMANENT_CATEGORIES = frozenset({"auth", "quota", "max_turns", "config"})

TRANSIENT_ERROR_KEYWORDS = [
    "timeout", "timed out", "rate limit", "rate_limit", "too many requests",
    "connection", "could not connect", "disconnected", "econnreset", "econnrefused",
    "503", "429", "502", "529", "overloaded", "unavailable",
    "resource_exhausted", "resource has been exhausted", "retryablequotaerror",
]
#: Messages that look transient but mean "stop" (quota / usage limits).
PERMANENT_ERROR_KEYWORDS = [
    "terminalquotaerror", "usage limit", "insufficient_quota",
]

_CATEGORY_PREFIX_RE = re.compile(r"^\s*\[([a-z_]+)\]")


def error_category(error: Optional[str]) -> str:
    """The ``[category]`` prefix of an adapter error, or "" if absent."""
    m = _CATEGORY_PREFIX_RE.match(error or "")
    return m.group(1) if m else ""


def is_transient_error(result: AdapterRunResult) -> bool:
    """Check if an error is likely transient and worth retrying.

    The adapters' own category prefix decides first: ``[rate_limit]``,
    ``[timeout]``, ``[network]`` and ``[overloaded]`` are transient;
    ``[auth]``, ``[quota]``, ``[max_turns]`` and ``[config]`` are permanent.
    Otherwise (``[unknown]``, ``[empty_output]``, no prefix) the error text is
    matched against keyword lists (usage-limit / terminal-quota messages are
    never retried).
    """
    if result.success:
        return False
    error = result.error or ""
    category = error_category(error)
    if category in TRANSIENT_CATEGORIES:
        return True
    if category in PERMANENT_CATEGORIES:
        return False
    low = error.lower()
    if any(kw in low for kw in PERMANENT_ERROR_KEYWORDS):
        return False
    return any(kw in low for kw in TRANSIENT_ERROR_KEYWORDS)


async def run_with_retry(
    adapter: BaseAdapter,
    request: AdapterRunRequest,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    on_retry: Optional[callable] = None,
) -> AdapterRunResult:
    """Run an adapter call with exponential backoff on transient failures.

    - Retries up to max_retries times on transient errors
    - Exponential backoff: base_delay * 2^attempt (capped at max_delay)
    - Adds jitter (+-25%) to avoid thundering herd
    - Calls on_retry(attempt, error, delay) before each retry
    - Returns immediately on success or permanent errors
    """
    last_result: Optional[AdapterRunResult] = None

    for attempt in range(max_retries + 1):
        result = await adapter.run(request)

        if result.success:
            return result

        if not is_transient_error(result):
            # Permanent error — don't retry
            return result

        last_result = result

        if attempt < max_retries:
            delay = min(base_delay * (2 ** attempt), max_delay)
            # Add jitter: +-25%
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay = max(0, delay + jitter)

            if on_retry:
                on_retry(attempt + 1, result.error, delay)

            logger.info(
                "Transient error (attempt %d/%d): %s — retrying in %.2fs",
                attempt + 1, max_retries, result.error, delay,
            )
            await asyncio.sleep(delay)

    # All retries exhausted — return the last failed result
    return last_result
