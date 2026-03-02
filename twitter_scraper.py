"""
twitter_scraper.py
──────────────────
Fetch tweets for tracked handicapper handles.

Primary:  Twitter API v2 (Bearer token from TWITTER_BEARER_TOKEN env var)
Fallback: HTML scraping via requests + BeautifulSoup

Usage
-----
  python twitter_scraper.py                  # one-shot run
  python twitter_scraper.py --cron           # cron-friendly: quiet on success
  python twitter_scraper.py --handles CBB_Edge HoopsLock

Outputs
-------
  data/handicapper/raw_tweets.csv            # appended with new tweets
  data/handicapper/handicappers.csv          # updated tracking columns
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "handicapper"
HANDICAPPERS_CSV = DATA_DIR / "handicappers.csv"
RAW_TWEETS_CSV = DATA_DIR / "raw_tweets.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("twitter_scraper")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TWITTER_API_BASE = "https://api.twitter.com/2"
MAX_RESULTS = 100
MAX_RETRIES = 4
INITIAL_BACKOFF = 2.0          # seconds
RAW_TWEETS_FIELDS = [
    "tweet_id", "handicapper_id", "handle", "created_at", "text",
    "tweet_url", "ingested_at",
]
HANDICAPPERS_TRACKING_COLS = [
    "last_ingested_tweet_id",
    "last_scrape_time",
    "scrape_success_rate",
]


# ---------------------------------------------------------------------------
# Handicappers CSV helpers
# ---------------------------------------------------------------------------

def _ensure_tracking_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add tracking columns with defaults if they don't exist."""
    defaults = {
        "last_ingested_tweet_id": "",
        "last_scrape_time": "",
        "scrape_success_rate": 1.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    return df


def load_handicappers(handles: Optional[list[str]] = None) -> pd.DataFrame:
    """Load active handicappers, optionally filtered to *handles* (without @)."""
    df = pd.read_csv(HANDICAPPERS_CSV, dtype=str)
    df = _ensure_tracking_columns(df)
    active = df[df["status"].str.lower() == "active"].copy()
    if handles:
        normalised = {h.lstrip("@").lower() for h in handles}
        active = active[
            active["handle"].str.lstrip("@").str.lower().isin(normalised)
        ]
    return active.reset_index(drop=True)


def save_handicappers(df: pd.DataFrame) -> None:
    """Persist updated handicappers DataFrame."""
    df.to_csv(HANDICAPPERS_CSV, index=False)
    log.debug("Saved %s", HANDICAPPERS_CSV)


# ---------------------------------------------------------------------------
# Raw-tweets CSV helpers
# ---------------------------------------------------------------------------

def _existing_tweet_ids() -> set[str]:
    if not RAW_TWEETS_CSV.exists():
        return set()
    try:
        existing = pd.read_csv(RAW_TWEETS_CSV, dtype=str, usecols=["tweet_id"])
        return set(existing["tweet_id"].dropna().tolist())
    except Exception:
        return set()


def append_tweets(new_rows: list[dict]) -> int:
    """Append *new_rows* to raw_tweets.csv, skipping duplicates. Returns count written."""
    if not new_rows:
        return 0
    seen = _existing_tweet_ids()
    deduped = [r for r in new_rows if str(r["tweet_id"]) not in seen]
    if not deduped:
        return 0
    write_header = not RAW_TWEETS_CSV.exists()
    with open(RAW_TWEETS_CSV, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RAW_TWEETS_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(deduped)
    return len(deduped)


# ---------------------------------------------------------------------------
# Exponential backoff helper
# ---------------------------------------------------------------------------

def _backoff_request(func, *args, **kwargs):
    """Call *func(*args, **kwargs)* with exponential backoff on rate-limit / server errors."""
    delay = INITIAL_BACKOFF
    last_exc: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = func(*args, **kwargs)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", delay))
                log.warning("Rate-limited; sleeping %ds (attempt %d/%d)", retry_after, attempt, MAX_RETRIES)
                time.sleep(retry_after)
                delay *= 2
                continue
            if resp.status_code >= 500:
                log.warning("Server error %d; retrying in %ds (attempt %d/%d)",
                            resp.status_code, delay, attempt, MAX_RETRIES)
                time.sleep(delay)
                delay *= 2
                continue
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            log.warning("Request error %s; retrying in %ds (attempt %d/%d)",
                        exc, delay, attempt, MAX_RETRIES)
            time.sleep(delay)
            delay *= 2
    if last_exc:
        raise last_exc
    return None


# ---------------------------------------------------------------------------
# Twitter API v2 scraper
# ---------------------------------------------------------------------------

class TwitterAPIv2Scraper:
    """Fetch tweets using the Twitter API v2 Bearer token."""

    def __init__(self, bearer_token: str):
        self._headers = {"Authorization": f"Bearer {bearer_token}"}

    def _get_user_id(self, handle: str) -> Optional[str]:
        username = handle.lstrip("@")
        url = f"{TWITTER_API_BASE}/users/by/username/{username}"
        resp = _backoff_request(requests.get, url, headers=self._headers, timeout=15)
        if resp is None or resp.status_code != 200:
            log.error("Failed to look up user ID for @%s (status=%s)",
                      username, resp.status_code if resp else "N/A")
            return None
        data = resp.json().get("data", {})
        return data.get("id")

    def fetch(self, handle: str, since_id: Optional[str] = None) -> list[dict]:
        """Return list of tweet dicts for *handle*, newest-first."""
        user_id = self._get_user_id(handle)
        if not user_id:
            return []

        username = handle.lstrip("@")
        url = f"{TWITTER_API_BASE}/users/{user_id}/tweets"
        params: dict = {
            "max_results": MAX_RESULTS,
            "tweet.fields": "created_at,public_metrics",
            "exclude": "retweets,replies",
        }
        if since_id:
            params["since_id"] = since_id

        resp = _backoff_request(requests.get, url, headers=self._headers,
                                params=params, timeout=15)
        if resp is None or resp.status_code != 200:
            log.error("API fetch failed for @%s (status=%s)",
                      username, resp.status_code if resp else "N/A")
            return []

        payload = resp.json()
        tweets_data = payload.get("data", [])
        now_utc = datetime.now(timezone.utc).isoformat()
        results = []
        for tw in tweets_data:
            results.append({
                "tweet_id": tw["id"],
                "handle": f"@{username}",
                "created_at": tw.get("created_at", ""),
                "text": tw.get("text", ""),
                "tweet_url": f"https://twitter.com/{username}/status/{tw['id']}",
                "ingested_at": now_utc,
            })
        return results


# ---------------------------------------------------------------------------
# HTML / BeautifulSoup fallback scraper
# ---------------------------------------------------------------------------

class HTMLFallbackScraper:
    """
    Scrape tweet data from nitter.net (a Twitter proxy that doesn't require
    a login) when the Twitter API is unavailable.
    """

    NITTER_BASE = "https://nitter.net"

    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; CBB-RD2-scraper/1.0; "
            "+https://github.com/goldbondman/CBB.RD2)"
        )
    }

    def fetch(self, handle: str, since_id: Optional[str] = None) -> list[dict]:
        username = handle.lstrip("@")
        url = f"{self.NITTER_BASE}/{username}"
        try:
            resp = _backoff_request(
                requests.get, url, headers=self._HEADERS, timeout=20
            )
        except Exception as exc:
            log.error("HTML scrape request failed for @%s: %s", username, exc)
            return []

        if resp is None or resp.status_code != 200:
            log.error("HTML scrape failed for @%s (status=%s)",
                      username, resp.status_code if resp else "N/A")
            return []

        return self._parse(resp.text, username, since_id)

    def _parse(self, html: str, username: str, since_id: Optional[str]) -> list[dict]:
        soup = BeautifulSoup(html, "html.parser")
        now_utc = datetime.now(timezone.utc).isoformat()
        results = []
        since_int = int(since_id) if since_id and since_id.isdigit() else 0

        for item in soup.select(".timeline-item"):
            link_tag = item.select_one(".tweet-link")
            if not link_tag:
                continue
            href = link_tag.get("href", "")
            # href looks like /username/status/12345...
            parts = href.rstrip("/").split("/")
            tweet_id = parts[-1] if parts else ""
            if not tweet_id.isdigit():
                continue
            if since_int and int(tweet_id) <= since_int:
                continue

            text_tag = item.select_one(".tweet-content")
            tweet_text = text_tag.get_text(separator=" ", strip=True) if text_tag else ""

            time_tag = item.select_one(".tweet-date a")
            created_at = time_tag.get("title", "") if time_tag else ""

            results.append({
                "tweet_id": tweet_id,
                "handle": f"@{username}",
                "created_at": created_at,
                "text": tweet_text,
                "tweet_url": f"https://twitter.com/{username}/status/{tweet_id}",
                "ingested_at": now_utc,
            })

        return results


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _update_tracking(
    df: pd.DataFrame,
    idx: int,
    new_tweets: list[dict],
    success: bool,
) -> None:
    """Update the per-handicapper tracking columns in-place."""
    now = datetime.now(timezone.utc).isoformat()
    df.at[idx, "last_scrape_time"] = now

    if new_tweets:
        ids = [int(t["tweet_id"]) for t in new_tweets if str(t["tweet_id"]).isdigit()]
        if ids:
            df.at[idx, "last_ingested_tweet_id"] = str(max(ids))

    # Rolling exponential moving average of success
    prev_rate = df.at[idx, "scrape_success_rate"]
    try:
        prev_rate = float(prev_rate)
    except (ValueError, TypeError):
        prev_rate = 1.0
    new_rate = 0.9 * prev_rate + 0.1 * (1.0 if success else 0.0)
    df.at[idx, "scrape_success_rate"] = str(round(new_rate, 4))


def run(handles: Optional[list[str]] = None, cron: bool = False) -> int:
    """
    Run the scraper for all active handicappers (or a subset via *handles*).
    Returns total number of new tweets written.
    """
    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN", "").strip()

    if bearer_token:
        primary = TwitterAPIv2Scraper(bearer_token)
        log.info("Using Twitter API v2 (Bearer token present)")
    else:
        primary = None
        log.warning("TWITTER_BEARER_TOKEN not set; will use HTML fallback only")

    fallback = HTMLFallbackScraper()

    handicappers = load_handicappers(handles)
    if handicappers.empty:
        log.warning("No active handicappers found – nothing to do")
        return 0

    total_written = 0
    all_new: list[dict] = []

    for idx, row in handicappers.iterrows():
        handle = row["handle"]
        hid = row["handicapper_id"]
        since_id = row.get("last_ingested_tweet_id", "") or None

        log.info("Scraping @%s (id=%s, since_id=%s)", handle, hid, since_id)
        new_tweets: list[dict] = []
        success = False

        # --- primary: API v2 ---
        if primary is not None:
            try:
                new_tweets = primary.fetch(handle, since_id)
                success = True
                log.info("  API v2 → %d tweet(s)", len(new_tweets))
            except Exception as exc:
                log.warning("  API v2 error for %s: %s; trying fallback", handle, exc)
                new_tweets = []

        # --- fallback: HTML ---
        if not success:
            try:
                new_tweets = fallback.fetch(handle, since_id)
                success = True   # request completed regardless of tweet count
                log.info("  HTML fallback → %d tweet(s)", len(new_tweets))
            except Exception as exc:
                log.error("  HTML fallback error for %s: %s", handle, exc)
                success = False

        # Attach handicapper_id
        for tw in new_tweets:
            tw["handicapper_id"] = hid

        all_new.extend(new_tweets)
        _update_tracking(handicappers, idx, new_tweets, success)

    written = append_tweets(all_new)
    save_handicappers(handicappers)
    total_written += written

    if not cron:
        log.info("Done – %d new tweet(s) written to %s", total_written, RAW_TWEETS_CSV)
    else:
        if total_written:
            print(f"twitter_scraper: {total_written} new tweet(s) ingested")

    return total_written


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch tweets for CBB handicapper handles"
    )
    parser.add_argument(
        "--cron",
        action="store_true",
        help="Cron mode: suppress info logging; print count only when > 0",
    )
    parser.add_argument(
        "--handles",
        nargs="+",
        metavar="HANDLE",
        help="Limit to specific handles (with or without leading @)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    if args.cron:
        logging.getLogger().setLevel(logging.WARNING)
    run(handles=args.handles, cron=args.cron)


if __name__ == "__main__":
    main()
