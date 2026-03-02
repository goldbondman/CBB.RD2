import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from game_mapper import GameMapper


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def games_df():
    return pd.DataFrame({
        "game_id": [123, 124, 125, 126],
        "date": pd.to_datetime([
            "2026-02-28 19:00:00",
            "2026-02-28 20:30:00",
            "2026-02-28 21:00:00",
            "2026-02-28 22:15:00",
        ]),
        "home_team": ["Illinois", "Kentucky", "Purdue", "Houston"],
        "away_team": ["Wisconsin", "UConn", "Gonzaga", "Tennessee"],
    })


@pytest.fixture()
def mapper():
    return GameMapper()


def _pick(raw_pick_id, team_raw, parsed_at):
    return pd.Series({
        "raw_pick_id": raw_pick_id,
        "team_raw": team_raw,
        "parsed_at": parsed_at,
    })


# ---------------------------------------------------------------------------
# map_raw_pick_to_game — happy path
# ---------------------------------------------------------------------------

def test_map_illinois_home_team(mapper, games_df):
    result = mapper.map_raw_pick_to_game(
        _pick(1, "Illinois", "2026-02-28 14:25:30"), games_df
    )
    assert result["game_id"] == 123
    assert result["mapping_status"] == "ok"
    assert result["team_canonical"] == "Illinois"
    assert result["tipoff_delta_minutes"] == pytest.approx(274.5, abs=0.1)


def test_map_uconn_away_team(mapper, games_df):
    result = mapper.map_raw_pick_to_game(
        _pick(2, "UConn", "2026-02-28 15:12:45"), games_df
    )
    assert result["game_id"] == 124
    assert result["mapping_status"] == "ok"
    assert result["team_canonical"] == "UConn"


def test_map_slash_separated_team_picks_first_match(mapper, games_df):
    """'UConn/Kentucky' should resolve to the first valid part ('UConn')."""
    result = mapper.map_raw_pick_to_game(
        _pick(2, "UConn/Kentucky", "2026-02-28 15:12:45"), games_df
    )
    assert result["game_id"] == 124
    assert result["mapping_status"] == "ok"
    assert result["team_canonical"] == "UConn"
    assert result["tipoff_delta_minutes"] == pytest.approx(317.25, abs=0.1)


def test_map_gonzaga_away_team(mapper, games_df):
    result = mapper.map_raw_pick_to_game(
        _pick(4, "Gonzaga", "2026-02-28 18:22:15"), games_df
    )
    assert result["game_id"] == 125
    assert result["mapping_status"] == "ok"
    assert result["team_canonical"] == "Gonzaga"


# ---------------------------------------------------------------------------
# map_raw_pick_to_game — edge cases
# ---------------------------------------------------------------------------

def test_map_no_team_name_returns_error(mapper, games_df):
    pick = pd.Series({"raw_pick_id": 99, "team_raw": "", "parsed_at": "2026-02-28 14:00:00"})
    result = mapper.map_raw_pick_to_game(pick, games_df)
    assert result["game_id"] is None
    assert result["mapping_status"] == "no_team_name"


def test_map_nan_team_name_returns_error(mapper, games_df):
    pick = pd.Series({"raw_pick_id": 99, "team_raw": float("nan"), "parsed_at": "2026-02-28 14:00:00"})
    result = mapper.map_raw_pick_to_game(pick, games_df)
    assert result["game_id"] is None
    assert result["mapping_status"] == "no_team_name"


def test_map_unknown_team_returns_no_match(mapper, games_df):
    # A nonsense team name should result in no successful mapping.
    result = mapper.map_raw_pick_to_game(
        _pick(10, "Xyzzy University", "2026-02-28 14:00:00"), games_df
    )
    assert result["game_id"] is None
    # Status is either a poor-match score or no_matching_game.
    assert result["mapping_status"].startswith("poor_team_match:") or \
           result["mapping_status"] == "no_matching_game"


def test_map_empty_games_df(mapper):
    pick = _pick(1, "Illinois", "2026-02-28 14:25:30")
    result = mapper.map_raw_pick_to_game(pick, pd.DataFrame())
    assert result["game_id"] is None
    assert result["mapping_status"] in ("no_matching_game", "poor_team_match:1.00")


def test_map_pick_outside_time_window(mapper, games_df):
    """A pick placed 13+ hours before tip-off should not be matched."""
    pick = _pick(1, "Illinois", "2026-02-27 05:00:00")  # >13 hours before any game
    result = mapper.map_raw_pick_to_game(pick, games_df)
    assert result["game_id"] is None
    assert result["mapping_status"] == "no_matching_game"


# ---------------------------------------------------------------------------
# batch_map_raw_picks
# ---------------------------------------------------------------------------

def test_batch_map_returns_correct_game_ids(mapper, games_df):
    picks = pd.DataFrame([
        {"raw_pick_id": 1, "team_raw": "Illinois", "parsed_at": "2026-02-28 14:25:30"},
        {"raw_pick_id": 2, "team_raw": "UConn/Kentucky", "parsed_at": "2026-02-28 15:12:45"},
        {"raw_pick_id": 4, "team_raw": "Gonzaga", "parsed_at": "2026-02-28 18:22:15"},
    ])
    results = mapper.batch_map_raw_picks(picks, games_df)

    assert list(results["game_id"]) == [123, 124, 125]
    assert list(results["mapping_status"]) == ["ok", "ok", "ok"]


def test_batch_map_preserves_raw_pick_ids(mapper, games_df):
    picks = pd.DataFrame([
        {"raw_pick_id": 7, "team_raw": "Illinois", "parsed_at": "2026-02-28 14:00:00"},
    ])
    results = mapper.batch_map_raw_picks(picks, games_df)
    assert results.iloc[0]["raw_pick_id"] == 7


def test_batch_map_empty_picks_returns_empty_df(mapper, games_df):
    results = mapper.batch_map_raw_picks(pd.DataFrame(), games_df)
    assert results.empty
