"""Retry wrapper for adapter calls with exponential backoff."""
import asyncio
import logging
import random
from typing import Optional

from backend.adapters.base import BaseAdapter
from backend.models import AdapterRunRequest, AdapterRunResult

logger = logging.getLogger(__name__)

TRANSIENT_ERROR_KEYWORDS = [
    "timeout", "rate limit", "connection", "503", "429", "502", "overloaded",
]


def is_transient_error(result: AdapterRunResult) -> bool:
    """Check if an error is likely transient and worth retrying."""
    if result.success:
        return False
    error = (result.error or "").lower()
    return any(kw in error for kw in TRANSIENT_ERROR_KEYWORDS)


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
