import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from build_derived_csvs import build_upset_watch_csv


def test_upset_watch_uses_non_degenerate_model_spread(monkeypatch):
    predictions = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_date": "2099-02-25",
                "home_team": "Home A",
                "away_team": "Away A",
                "spread_line": -8.5,
                "mc_home_win_pct": 0.55,
                "mc_away_win_pct": 0.45,
                "ens_ens_spread": -3.15,
                "mc_spread_median": -6.2,
            },
            {
                "game_id": "g2",
                "game_date": "2099-02-25",
                "home_team": "Home B",
                "away_team": "Away B",
                "spread_line": 6.5,
                "mc_home_win_pct": 0.45,
                "mc_away_win_pct": 0.55,
                "ens_ens_spread": -3.15,
                "mc_spread_median": 1.8,
            },
        ]
    )

    captured = {}

    def _capture_write(df, stem, sources, dated_copy=False):
        captured["df"] = df.copy()
        captured["stem"] = stem

    monkeypatch.setattr("build_derived_csvs._write", _capture_write)
    monkeypatch.setattr("build_derived_csvs.WINDOW_AHEAD_HOURS", 1_000_000.0)

    build_upset_watch_csv(predictions)

    assert captured["stem"] == "upset_watch.csv"
    model_spreads = captured["df"]["model_spread"].tolist()
    assert model_spreads
    assert -3.15 not in model_spreads
    assert 1.8 in model_spreads
import pandas as pd

import build_derived_csvs as bdc


def test_enrich_ensemble_team_names_backfills_missing(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    ensemble = pd.DataFrame(
        [
            {"game_id": "1001", "home_team_id": "10", "away_team_id": "20", "home_team": "", "away_team": ""},
            {"game_id": "1002", "home_team_id": "11", "away_team_id": "21", "home_team": "Known Home", "away_team": "Known Away"},
        ]
    )
    preds = pd.DataFrame(
        [
            {"game_id": "1001", "home_team": "Duke", "away_team": "UNC"},
            {"game_id": "1002", "home_team": "Known Home", "away_team": "Known Away"},
        ]
    )

    ensemble.to_csv(data_dir / "ensemble_predictions_latest.csv", index=False)
    preds.to_csv(data_dir / "predictions_latest.csv", index=False)

    monkeypatch.setattr(bdc, "DATA", data_dir)

    bdc.enrich_ensemble_team_names()

    out = pd.read_csv(data_dir / "ensemble_predictions_latest.csv", dtype=str)
    row = out[out["game_id"] == "1001"].iloc[0]
    assert row["home_team"] == "Duke"
    assert row["away_team"] == "UNC"
    assert row["matchup"] == "UNC @ Duke"


def test_select_prediction_source_prefers_fresher_file(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    stale = pd.DataFrame(
        [
            {"game_id": "old", "game_datetime_utc": "2026-02-24T20:00:00Z"},
        ]
    )
    fresh = pd.DataFrame(
        [
            {"game_id": "new", "game_datetime_utc": "2026-02-27T20:00:00Z"},
        ]
    )

    stale.to_csv(data_dir / "predictions_mc_latest.csv", index=False)
    fresh.to_csv(data_dir / "predictions_latest.csv", index=False)

    monkeypatch.setattr(bdc, "DATA", data_dir)

    df, label = bdc._select_prediction_source()
    assert label == "predictions_latest"
    assert df is not None
    assert df.iloc[0]["game_id"] == "new"


def test_filter_upcoming_window_includes_todays_earlier_games(monkeypatch):
    """Games scheduled for today (PST) but earlier in the day must be included."""
    from datetime import datetime, timezone

    # Use a January date to ensure PST (UTC-8) is in effect, not PDT
    # Simulate running at 8 PM PST: 2099-01-15 20:00 PST = 2099-01-16 04:00 UTC
    now_utc = datetime(2099, 1, 16, 4, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(bdc, "NOW_UTC", now_utc)

    # Game scheduled at 1 PM PST today (7 hours before run time)
    today_morning_utc = "2099-01-15T21:00:00Z"   # 1 PM PST on 2099-01-15
    # Game from yesterday — should be excluded
    yesterday_utc = "2099-01-14T21:00:00Z"        # 1 PM PST on 2099-01-14
    # Game tomorrow — should be included
    tomorrow_utc = "2099-01-16T21:00:00Z"          # 1 PM PST on 2099-01-16

    df = pd.DataFrame([
        {"game_id": "today_morning", "game_datetime_utc": today_morning_utc},
        {"game_id": "yesterday",     "game_datetime_utc": yesterday_utc},
        {"game_id": "tomorrow",      "game_datetime_utc": tomorrow_utc},
    ])

    result = bdc._filter_upcoming_window(
        df,
        label="test",
        start_hours=0.0,
        behind_hours=0.0,
        ahead_hours=48.0,
        timezone_name="America/Los_Angeles",
    )

    ids = result["game_id"].tolist()
    assert "today_morning" in ids, "Game earlier today (PST) should be included"
    assert "yesterday" not in ids, "Yesterday's game should be excluded"
    assert "tomorrow" in ids, "Tomorrow's game should be included"
