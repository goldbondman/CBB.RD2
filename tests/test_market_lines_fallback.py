from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enrichment.predictions_with_context import (
    _build_market_lines_fallback,
    build_predictions_with_context,
)


def test_build_market_lines_fallback_preserves_team_names(tmp_path: Path):
    market_path = tmp_path / "market_lines.csv"
    pred_df = pd.DataFrame(
        [
            {
                "event_id": "40123",
                "home_team": "Duke",
                "away_team": "UNC",
                "home_team_id": "150",
                "away_team_id": "153",
                "spread_line": -4.5,
                "total_line": 145.5,
                "home_ml": -190,
                "away_ml": 165,
            }
        ]
    )

    fallback = _build_market_lines_fallback(pred_df, market_path)

    assert not fallback.empty
    assert fallback.iloc[0]["home_team_name"] == "Duke"
    assert fallback.iloc[0]["away_team_name"] == "UNC"
    assert fallback.iloc[0]["home_team_id"] == "150"
    assert fallback.iloc[0]["away_team_id"] == "153"

    written = pd.read_csv(market_path, dtype=str)
    assert "home_team_name" in written.columns
    assert "away_team_name" in written.columns


def test_predictions_with_context_filters_to_today_pst(tmp_path: Path, monkeypatch):
    """predictions_with_context.csv must only include today (PST) and future games,
    not carry forward stale predictions from previous pipeline runs."""
    # Use a fixed reference date so the test is deterministic regardless of when it runs
    ref_date = pd.Timestamp("2026-03-03 12:00:00", tz="America/Los_Angeles")
    yesterday_utc = (ref_date - pd.Timedelta(days=1)).tz_convert("UTC")
    today_utc = ref_date.tz_convert("UTC")
    tomorrow_utc = (ref_date + pd.Timedelta(days=1)).tz_convert("UTC")

    preds_path = tmp_path / "predictions_combined_latest.csv"
    out_path = tmp_path / "predictions_with_context.csv"

    # Write a predictions file with games from yesterday, today, and tomorrow
    pd.DataFrame([
        {
            "game_id": "old_game_1",
            "event_id": "old_game_1",
            "game_datetime_utc": yesterday_utc.isoformat(),
            "home_team": "TeamA",
            "away_team": "TeamB",
            "home_team_id": "1",
            "away_team_id": "2",
            "pred_spread": -3.0,
        },
        {
            "game_id": "today_game_1",
            "event_id": "today_game_1",
            "game_datetime_utc": today_utc.isoformat(),
            "home_team": "TeamC",
            "away_team": "TeamD",
            "home_team_id": "3",
            "away_team_id": "4",
            "pred_spread": 2.5,
        },
        {
            "game_id": "tomorrow_game_1",
            "event_id": "tomorrow_game_1",
            "game_datetime_utc": tomorrow_utc.isoformat(),
            "home_team": "TeamE",
            "away_team": "TeamF",
            "home_team_id": "5",
            "away_team_id": "6",
            "pred_spread": -1.0,
        },
    ]).to_csv(preds_path, index=False)

    # Patch DATA_DIR so optional data files are looked for in tmp_path
    import enrichment.predictions_with_context as pwc
    monkeypatch.setattr(pwc, "DATA_DIR", tmp_path)

    build_predictions_with_context(
        predictions_path=preds_path,
        out_path=out_path,
        reference_date=ref_date,
    )

    result = pd.read_csv(out_path, dtype=str)
    game_ids = set(result["game_id"].tolist())

    assert "old_game_1" not in game_ids, "Stale game from yesterday should be filtered out"
    assert "today_game_1" in game_ids, "Today's game must be included"
    assert "tomorrow_game_1" in game_ids, "Tomorrow's game (within rolling window) must be included"


def test_predictions_with_context_fallback_when_all_games_in_past(tmp_path: Path, monkeypatch):
    """When all input games are in the past, output must include all rows (not be blank)."""
    ref_date = pd.Timestamp("2026-03-03 12:00:00", tz="America/Los_Angeles")
    last_week_utc = (ref_date - pd.Timedelta(days=7)).tz_convert("UTC")
    yesterday_utc = (ref_date - pd.Timedelta(days=1)).tz_convert("UTC")

    preds_path = tmp_path / "predictions_combined_latest.csv"
    out_path = tmp_path / "predictions_with_context.csv"

    pd.DataFrame([
        {
            "game_id": "past_game_1",
            "event_id": "past_game_1",
            "game_datetime_utc": last_week_utc.isoformat(),
            "home_team": "TeamA",
            "away_team": "TeamB",
            "home_team_id": "1",
            "away_team_id": "2",
            "pred_spread": -3.0,
        },
        {
            "game_id": "past_game_2",
            "event_id": "past_game_2",
            "game_datetime_utc": yesterday_utc.isoformat(),
            "home_team": "TeamC",
            "away_team": "TeamD",
            "home_team_id": "3",
            "away_team_id": "4",
            "pred_spread": 2.5,
        },
    ]).to_csv(preds_path, index=False)

    import enrichment.predictions_with_context as pwc
    monkeypatch.setattr(pwc, "DATA_DIR", tmp_path)

    build_predictions_with_context(
        predictions_path=preds_path,
        out_path=out_path,
        reference_date=ref_date,
    )

    result = pd.read_csv(out_path, dtype=str)
    assert len(result) == 2, (
        "When all input games are in the past, all rows must be preserved "
        "to avoid a blank output file"
    )
    game_ids = set(result["game_id"].tolist())
    assert "past_game_1" in game_ids
    assert "past_game_2" in game_ids
