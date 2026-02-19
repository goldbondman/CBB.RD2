"""
ESPN CBB Pipeline — HTTP Client
Thin fetch layer with retry/backoff. No business logic here.
"""

import time
import logging
from typing import Any, Dict

import requests

from espn_config import (
    DEFAULT_HEADERS,
    REQUEST_TIMEOUT,
    MAX_RETRIES,
    RETRY_INITIAL_DELAY,
    RETRY_BACKOFF,
    ESPN_SCOREBOARD_URL,
    ESPN_SUMMARY_URL,
)

log = logging.getLogger(__name__)


def fetch_with_retry(url: str, timeout: int = REQUEST_TIMEOUT) -> Dict[str, Any]:
    """
    GET a URL with exponential backoff retry.
    Raises RuntimeError if all attempts fail.
    """
    delay = RETRY_INITIAL_DELAY
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                log.warning(f"Attempt {attempt}/{MAX_RETRIES} failed for {url}: {exc} — retrying in {delay}s")
                time.sleep(delay)
                delay *= RETRY_BACKOFF
            else:
                log.error(f"All {MAX_RETRIES} attempts failed for {url}: {exc}")

    raise RuntimeError(f"fetch_with_retry failed after {MAX_RETRIES} attempts: {last_exc}") from last_exc


def fetch_scoreboard(date_yyyymmdd: str) -> Dict[str, Any]:
    """Fetch ESPN scoreboard for a single date (YYYYMMDD)."""
    url = ESPN_SCOREBOARD_URL.format(date=date_yyyymmdd)
    log.debug(f"Fetching scoreboard: {date_yyyymmdd}")
    return fetch_with_retry(url)


def fetch_summary(event_id: str) -> Dict[str, Any]:
    """Fetch ESPN game summary for a single event ID."""
    url = ESPN_SUMMARY_URL.format(event_id=event_id)
    log.debug(f"Fetching summary: {event_id}")
    return fetch_with_retry(url)
