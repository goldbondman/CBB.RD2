from datetime import date
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion.market_lines import (
    detect_book_disagreement,
    detect_reverse_line_movement,
    parse_espn_event,
    resolve_date_range,
)


def _espn_event(details: str) -> dict:
    return {
        "id": "401000001",
        "competitions": [
            {
                "date": "2026-03-05T23:00:00Z",
                "odds": [{"details": details, "overUnder": 145.5}],
                "competitors": [
                    {
                        "homeAway": "home",
                        "id": "111",
                        "team": {
                            "abbreviation": "IOWA",
                            "shortDisplayName": "Iowa",
                            "displayName": "Iowa Hawkeyes",
                        },
                    },
                    {
                        "homeAway": "away",
                        "id": "222",
                        "team": {
                            "abbreviation": "MSU",
                            "shortDisplayName": "Michigan State",
                            "displayName": "Michigan State Spartans",
                        },
                    },
                ],
            }
        ],
    }


def test_parse_espn_event_home_favored_sign():
    parsed = parse_espn_event(_espn_event("IOWA -4.5"))
    assert parsed is not None
    assert parsed["home_spread_current"] == -4.5


def test_parse_espn_event_away_favored_sign():
    parsed = parse_espn_event(_espn_event("MICHIGAN STATE -3.0"))
    assert parsed is not None
    assert parsed["home_spread_current"] == 3.0


def test_parse_espn_event_pick_em_zero():
    parsed = parse_espn_event(_espn_event("PK"))
    assert parsed is not None
    assert parsed["home_spread_current"] == 0.0


def test_detect_reverse_line_movement_home_public_line_toward_away_flags():
    # Public heavy on home, but home spread gets less favorable for home (toward away).
    out = detect_reverse_line_movement(75, -4.0, -2.5)
    assert out["rlm_flag"] is True
    assert out["rlm_sharp_side"] == "away"


def test_detect_reverse_line_movement_away_public_line_toward_home_flags():
    # Public heavy on away, but home spread gets more favorable for home (toward home).
    out = detect_reverse_line_movement(20, -2.0, -3.0)
    assert out["rlm_flag"] is True
    assert out["rlm_sharp_side"] == "home"


def test_detect_book_disagreement_sets_schema_keys():
    out = detect_book_disagreement(-1.0, 1.0)
    assert out["book_disagreement_flag"] is True
    assert out["book_sharp_side"] in {"home", "away"}
    assert isinstance(out.get("book_note"), str)


# --- resolve_date_range tests ---


def test_resolve_date_range_start_only_defaults_end_to_today():
    """start_date without end_date should span from start to today, not just one day."""
    today = date(2026, 3, 6)
    start, end = resolve_date_range("2025-11-01", None, None, today=today)
    assert start == date(2025, 11, 1)
    assert end == today


def test_resolve_date_range_start_and_end_explicit():
    today = date(2026, 3, 6)
    start, end = resolve_date_range("2025-12-01", "2026-01-15", None, today=today)
    assert start == date(2025, 12, 1)
    assert end == date(2026, 1, 15)


def test_resolve_date_range_days_back():
    today = date(2026, 3, 6)
    start, end = resolve_date_range(None, None, 7, today=today)
    assert start == date(2026, 2, 28)
    assert end == today


def test_resolve_date_range_no_args_defaults_to_today():
    today = date(2026, 3, 6)
    start, end = resolve_date_range(None, None, None, today=today)
    assert start == today
    assert end == today


def test_resolve_date_range_empty_strings_treated_as_none():
    today = date(2026, 3, 6)
    start, end = resolve_date_range("", "", "", today=today)
    assert start == today
    assert end == today


def test_resolve_date_range_compact_date_format():
    """YYYYMMDD format (no dashes) should also work."""
    today = date(2026, 3, 6)
    start, end = resolve_date_range("20251101", None, None, today=today)
    assert start == date(2025, 11, 1)
    assert end == today
