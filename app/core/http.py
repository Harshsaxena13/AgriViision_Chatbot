import asyncio
import logging
import random
from typing import Any

import httpx


logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 3
BASE_DELAY_SECONDS = 0.5
MAX_DELAY_SECONDS = 8.0


async def post_with_retries(client: httpx.AsyncClient, url: str, **kwargs: Any) -> httpx.Response:
    """POST, retrying 429 and 5xx responses with exponential backoff.

    The last response is returned as-is, so callers still decide how to handle errors.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = await client.post(url, **kwargs)
        if not is_retryable(response.status_code) or attempt == MAX_ATTEMPTS:
            return response
        delay = retry_delay(response, attempt)
        logger.warning(
            "Retrying POST %s status=%d attempt=%d/%d delay=%.2fs",
            url,
            response.status_code,
            attempt,
            MAX_ATTEMPTS,
            delay,
        )
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def is_retryable(status_code: int) -> bool:
    return status_code == 429 or status_code >= 500


def retry_delay(response: httpx.Response, attempt: int) -> float:
    # Honor a numeric Retry-After header (capped); otherwise back off exponentially with jitter.
    try:
        retry_after = float(response.headers.get("retry-after", ""))
    except ValueError:
        retry_after = -1.0
    if retry_after >= 0:
        return min(retry_after, MAX_DELAY_SECONDS)
    backoff = BASE_DELAY_SECONDS * 2 ** (attempt - 1)
    return min(backoff + random.uniform(0, BASE_DELAY_SECONDS), MAX_DELAY_SECONDS)
