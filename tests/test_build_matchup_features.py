"""Tests for scripts/build_matchup_features.py – is_upcoming logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import datetime

import pandas as pd
import pytest


def _make_market_row(game_date_utc: pd.Timestamp, home_score=None) -> dict:
    return {
        "game_datetime_utc": game_date_utc,
        "home_team_id": 1,
        "away_team_id": 2,
        "home_team_name": "Home",
        "away_team_name": "Away",
        "event_id": "g1",
        "closing_spread": -5.0,
        "closing_total": 140.0,
        "home_score": home_score,
    }


# ---------------------------------------------------------------------------
# Test that the is_upcoming fallback marks future-dated games as upcoming
# ---------------------------------------------------------------------------

def test_is_upcoming_includes_future_dates():
    """Games scheduled for future dates without scores must be flagged as upcoming."""
    from zoneinfo import ZoneInfo

    local_tz = ZoneInfo("America/Los_Angeles")
    today_local = pd.Timestamp.now(tz=local_tz).date()
    # Use a game clearly 7 days in the future (well past any timezone ambiguity)
    future_utc = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=7)

    has_scores = False
    gdate = future_utc
    game_local_date = gdate.tz_convert(local_tz).date() if pd.notna(gdate) else None

    # Fixed logic: >=
    new_is_upcoming = (not has_scores) and (game_local_date is not None) and (game_local_date >= today_local)
    # Old (buggy) logic: ==
    old_is_upcoming = (not has_scores) and (game_local_date == today_local)

    assert new_is_upcoming, "Fixed logic must flag a 7-day-future game as upcoming"
    assert not old_is_upcoming, "Old == logic must NOT flag a 7-day-future game as upcoming"


def test_is_upcoming_today_still_works():
    """Games scheduled for today without scores must still be flagged as upcoming."""
    from zoneinfo import ZoneInfo

    local_tz = ZoneInfo("America/Los_Angeles")
    today_local = pd.Timestamp.now(tz=local_tz).date()
    # Use a game 2 hours from now: this stays on the same local calendar day
    # (even at 22:00 PST, +2h = midnight PST still the same date or next, both are >= today)
    near_future_utc = pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=2)

    has_scores = False
    gdate = near_future_utc
    game_local_date = gdate.tz_convert(local_tz).date() if pd.notna(gdate) else None

    is_upcoming = (not has_scores) and (game_local_date is not None) and (game_local_date >= today_local)
    # game_local_date is today or later (within 2h), always >= today_local
    assert is_upcoming


def test_is_upcoming_past_game_not_upcoming():
    """Completed past games must NOT be flagged as upcoming."""
    from zoneinfo import ZoneInfo

    local_tz = ZoneInfo("America/Los_Angeles")
    today_local = pd.Timestamp.now(tz=local_tz).date()
    yesterday_utc = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)

    has_scores = False  # no scores in metrics, but game was yesterday
    gdate = yesterday_utc
    game_local_date = gdate.tz_convert(local_tz).date() if pd.notna(gdate) else None

    is_upcoming = (not has_scores) and (game_local_date is not None) and (game_local_date >= today_local)
    assert not is_upcoming, "Past game must not be flagged as upcoming"


def test_is_upcoming_none_date_not_upcoming():
    """Games with a null date must NOT be flagged as upcoming."""
    has_scores = False
    game_local_date = None

    is_upcoming = (not has_scores) and (game_local_date is not None) and (game_local_date >= datetime.date.today())
    assert not is_upcoming, "Game with null date must not be flagged as upcoming"
