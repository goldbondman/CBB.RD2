"""
batch_processor.py
──────────────────
Process raw tweets through the parse → map → backtest pipeline.

Usage
-----
  python batch_processor.py              # process new tweets from raw_tweets.csv
  python batch_processor.py --mock       # use built-in mock data for demo / testing
  python batch_processor.py --since-id 1845000000000000000

The script can be triggered automatically after twitter_scraper.py writes
new rows to data/handicapper/raw_tweets.csv.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "handicapper"
RAW_TWEETS_CSV = DATA_DIR / "raw_tweets.csv"
PICKS_CSV = DATA_DIR / "picks.csv"
RAW_PICKS_CSV = DATA_DIR / "raw_picks.csv"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("batch_processor")

# ---------------------------------------------------------------------------
# Mock data (used with --mock flag for demos / CI tests)
# ---------------------------------------------------------------------------

MOCK_TWEETS = [
    {
        "tweet_id": "9000000000000000001",
        "handicapper_id": "1",
        "handle": "@CBB_Edge",
        "created_at": "2026-03-01T18:00:00Z",
        "text": "Duke -5.5 tonight. Sharp action confirmed. #CBB",
        "tweet_url": "https://twitter.com/CBB_Edge/status/9000000000000000001",
        "ingested_at": "2026-03-01T18:02:00Z",
    },
    {
        "tweet_id": "9000000000000000002",
        "handicapper_id": "2",
        "handle": "@HoopsLock",
        "created_at": "2026-03-01T19:00:00Z",
        "text": "Kansas -7 at home vs Baylor. Back this one (2u)",
        "tweet_url": "https://twitter.com/HoopsLock/status/9000000000000000002",
        "ingested_at": "2026-03-01T19:02:00Z",
    },
    {
        "tweet_id": "9000000000000000003",
        "handicapper_id": "5",
        "handle": "@BracketBuster",
        "created_at": "2026-03-01T20:00:00Z",
        "text": "Houston/Memphis OVER 142.5 (1u)",
        "tweet_url": "https://twitter.com/BracketBuster/status/9000000000000000003",
        "ingested_at": "2026-03-01T20:02:00Z",
    },
]


# ---------------------------------------------------------------------------
# Tweet loader
# ---------------------------------------------------------------------------

def load_tweets(
    since_id: Optional[str] = None,
    use_mock: bool = False,
) -> pd.DataFrame:
    """
    Load raw tweets.

    Parameters
    ----------
    since_id:  If set, only return tweets with tweet_id > since_id.
    use_mock:  Use MOCK_TWEETS instead of the real CSV.
    """
    if use_mock:
        log.info("Loading mock tweets (%d rows)", len(MOCK_TWEETS))
        df = pd.DataFrame(MOCK_TWEETS)
    else:
        if not RAW_TWEETS_CSV.exists():
            log.warning("raw_tweets.csv not found – run twitter_scraper.py first")
            return pd.DataFrame(columns=["tweet_id", "handicapper_id", "text",
                                         "created_at", "ingested_at"])
        df = pd.read_csv(RAW_TWEETS_CSV, dtype=str)
        log.info("Loaded %d tweet(s) from %s", len(df), RAW_TWEETS_CSV)

    if since_id and not df.empty:
        try:
            since_int = int(since_id)
            df = df[pd.to_numeric(df["tweet_id"], errors="coerce") > since_int]
            log.info("Filtered to %d tweet(s) after tweet_id %s", len(df), since_id)
        except (ValueError, TypeError):
            pass

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Minimal parse / pick extraction
# ---------------------------------------------------------------------------

def _extract_pick(tweet_text: str) -> dict:
    """
    Very lightweight pick extractor.  Returns a dict with keys:
      pick_side, pick_spread, pick_units, pick_type
    Suitable for a quick demo; replace with a full NLP parser for production.
    """
    text = tweet_text.upper()
    pick: dict = {
        "pick_side": None,
        "pick_spread": None,
        "pick_units": None,
        "pick_type": "spread",
    }

    # Units: look for "(Nu)" pattern
    import re
    units_match = re.search(r"\((\d(?:\.\d)?)[Uu]\)", tweet_text)
    if units_match:
        pick["pick_units"] = float(units_match.group(1))

    # Totals: OVER / UNDER
    totals_match = re.search(r"\b(OVER|UNDER)\b\s+([\d.]+)", text)
    if totals_match:
        pick["pick_type"] = "total"
        pick["pick_side"] = totals_match.group(1).capitalize()
        pick["pick_spread"] = float(totals_match.group(2))
        return pick

    # Spread: <TEAM> [+-]N
    spread_match = re.search(
        r"([A-Z][A-Z\s]{2,20})\s+([+-]?\d+(?:\.\d+)?)\b", text
    )
    if spread_match:
        pick["pick_side"] = spread_match.group(1).strip().title()
        pick["pick_spread"] = float(spread_match.group(2))

    return pick


def parse_tweets(df: pd.DataFrame) -> pd.DataFrame:
    """Parse tweet rows into raw pick rows."""
    rows = []
    for _, tweet in df.iterrows():
        pick = _extract_pick(str(tweet.get("text", "")))
        rows.append(
            {
                "raw_pick_id": f"rp_{tweet['tweet_id']}",
                "tweet_id": tweet["tweet_id"],
                "handicapper_id": tweet["handicapper_id"],
                "raw_text": tweet.get("text", ""),
                "pick_side": pick["pick_side"],
                "pick_spread": pick["pick_spread"],
                "pick_units": pick["pick_units"],
                "pick_type": pick["pick_type"],
                "parse_status": "parsed",
                "created_at": tweet.get("created_at", ""),
            }
        )
    log.info("Parsed %d pick(s)", len(rows))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_raw_picks(df: pd.DataFrame) -> None:
    if df.empty:
        return
    write_header = not RAW_PICKS_CSV.exists()
    df.to_csv(RAW_PICKS_CSV, mode="a", header=write_header, index=False)
    log.info("Saved %d raw pick(s) to %s", len(df), RAW_PICKS_CSV)


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No picks processed.")
        return
    print(f"\n{'='*60}")
    print(f"  BATCH PROCESSOR SUMMARY – {datetime.now(timezone.utc).date()}")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        side = row.get("pick_side") or "?"
        spread = row.get("pick_spread")
        spread_str = f"{spread:+.1f}" if spread is not None else "ML"
        units = row.get("pick_units") or 1
        print(
            f"  [{row['handicapper_id']}] {side} {spread_str}  "
            f"({units}u)  [{row['pick_type']}]"
        )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------

def run(
    since_id: Optional[str] = None,
    use_mock: bool = False,
    trigger_scraper: bool = False,
) -> pd.DataFrame:
    """
    Full pipeline: optionally scrape → load tweets → parse → save picks.

    Parameters
    ----------
    since_id:        Process only tweets newer than this ID.
    use_mock:        Use built-in mock data instead of raw_tweets.csv.
    trigger_scraper: If True, call twitter_scraper.run() first.
    """
    if trigger_scraper and not use_mock:
        log.info("Triggering twitter_scraper …")
        try:
            from twitter_scraper import run as scraper_run
            written = scraper_run()
            log.info("Scraper wrote %d new tweet(s)", written)
        except Exception as exc:
            log.error("Scraper failed: %s", exc)

    tweets = load_tweets(since_id=since_id, use_mock=use_mock)
    if tweets.empty:
        log.info("No tweets to process")
        return pd.DataFrame()

    raw_picks = parse_tweets(tweets)
    save_raw_picks(raw_picks)
    print_summary(raw_picks)
    return raw_picks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse handicapper tweets into structured picks"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use built-in mock tweets (no CSV or API required)",
    )
    parser.add_argument(
        "--since-id",
        dest="since_id",
        metavar="TWEET_ID",
        help="Only process tweets newer than this tweet ID",
    )
    parser.add_argument(
        "--scrape",
        action="store_true",
        help="Run twitter_scraper first, then process new tweets",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)
    run(
        since_id=args.since_id,
        use_mock=args.mock,
        trigger_scraper=args.scrape,
    )


if __name__ == "__main__":
    main()
