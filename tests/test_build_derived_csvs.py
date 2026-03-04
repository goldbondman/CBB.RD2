import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

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


def test_filter_upcoming_window_excludes_past_games(monkeypatch):
    """Games scheduled before run time (even earlier today) must be excluded."""
    from datetime import datetime, timezone

    # Use a January date to ensure PST (UTC-8) is in effect, not PDT
    # Simulate running at 8 PM PST: 2099-01-15 20:00 PST = 2099-01-16 04:00 UTC
    now_utc = datetime(2099, 1, 16, 4, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(bdc, "NOW_UTC", now_utc)

    # Game scheduled at 1 PM PST today (7 hours before run time) — should be excluded
    today_morning_utc = "2099-01-15T21:00:00Z"   # 1 PM PST on 2099-01-15
    # Game from yesterday — should be excluded
    yesterday_utc = "2099-01-14T21:00:00Z"        # 1 PM PST on 2099-01-14
    # Game tomorrow — should be included
    tomorrow_utc = "2099-01-16T21:00:00Z"          # 1 PM PST on 2099-01-16
    # Game later tonight (9 PM PST) — should be included
    tonight_utc = "2099-01-16T05:00:00Z"           # 9 PM PST on 2099-01-15

    df = pd.DataFrame([
        {"game_id": "today_morning", "game_datetime_utc": today_morning_utc},
        {"game_id": "yesterday",     "game_datetime_utc": yesterday_utc},
        {"game_id": "tomorrow",      "game_datetime_utc": tomorrow_utc},
        {"game_id": "tonight",       "game_datetime_utc": tonight_utc},
    ])

    result = bdc._filter_upcoming_window(
        df,
        label="test",
        start_hours=0.0,
        behind_hours=0.0,
        ahead_hours=30.0,
        timezone_name="America/Los_Angeles",
    )

    ids = result["game_id"].tolist()
    assert "today_morning" not in ids, "Game before run time should be excluded"
    assert "yesterday" not in ids, "Yesterday's game should be excluded"
    assert "tomorrow" in ids, "Tomorrow's game should be included"
    assert "tonight" in ids, "Game later tonight should be included"


def test_filter_bet_recs_window_uses_half_open_40h_window(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(bdc, "NOW_UTC", datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc))

    df = pd.DataFrame([
        {"game_id": "prior_day", "home_team": "A", "away_team": "B", "game_datetime_utc": "2026-03-03T22:53:00Z"},
        {"game_id": "window_start", "home_team": "C", "away_team": "D", "game_datetime_utc": "2026-03-03T22:54:00Z"},
        {"game_id": "inside_window", "home_team": "E", "away_team": "F", "game_datetime_utc": "2026-03-05T14:53:00Z"},
        {"game_id": "window_end", "home_team": "G", "away_team": "H", "game_datetime_utc": "2026-03-05T14:54:00Z"},
    ])

    out = bdc._filter_bet_recs_window(df)
    ids = out["game_id"].tolist()

    assert "prior_day" not in ids
    assert "window_start" in ids
    assert "inside_window" in ids
    assert "window_end" not in ids


def test_filter_bet_recs_window_rejects_naive_and_logs_warning(monkeypatch, capsys):
    from datetime import datetime, timezone

    monkeypatch.setattr(bdc, "NOW_UTC", datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc))

    df = pd.DataFrame([
        {"game_id": "aware", "home_team": "A", "away_team": "B", "game_datetime_utc": "2026-03-04T01:00:00Z"},
        {"game_id": "aware2", "home_team": "AA", "away_team": "BB", "game_datetime_utc": "2026-03-04T03:00:00Z"},
        {"game_id": "aware3", "home_team": "X", "away_team": "Y", "game_datetime_utc": "2026-03-04T05:00:00Z"},
        {"game_id": "naive", "home_team": "C", "away_team": "D", "game_datetime_utc": "2026-03-04 01:00:00"},
    ])

    out = bdc._filter_bet_recs_window(df)
    captured = capsys.readouterr().out

    assert out["game_id"].tolist() == ["aware", "aware2", "aware3"]
    assert "game_id=naive" in captured
    assert "naive_timestamp_rejected" in captured


def test_filter_bet_recs_window_fails_when_parse_failures_exceed_30pct(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(bdc, "NOW_UTC", datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc))

    df = pd.DataFrame([
        {"game_id": "g1", "game_datetime_utc": "2026-03-04T01:00:00Z"},
        {"game_id": "g2", "game_datetime_utc": "2026-03-04 02:00:00"},
        {"game_id": "g3", "game_datetime_utc": "bad"},
    ])

    with pytest.raises(RuntimeError, match="parse failure ratio exceeds 30%"):
        bdc._filter_bet_recs_window(df)


def test_user_window_only_skips_stale_dated_ensemble_write(monkeypatch, tmp_path, capsys):
    from datetime import datetime, timezone

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    stale_path = data_dir / "ensemble_predictions_20260301.csv"
    pd.DataFrame(
        [
            {
                "game_id": "1001",
                "game_datetime_utc": "2026-03-01T20:00:00Z",
                "home_team": "",
                "away_team": "",
            }
        ]
    ).to_csv(stale_path, index=False)

    pd.DataFrame(
        [
            {"game_id": "1001", "home_team": "Duke", "away_team": "UNC"},
        ]
    ).to_csv(data_dir / "predictions_latest.csv", index=False)

    monkeypatch.setattr(bdc, "DATA", data_dir)
    monkeypatch.setattr(bdc, "NOW_UTC", datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc))
    monkeypatch.setenv("USER_WINDOW_ONLY", "1")

    bdc.enrich_ensemble_team_names()

    captured = capsys.readouterr().out
    out = pd.read_csv(stale_path, dtype=str)

    assert "[SKIP] ensemble_predictions_20260301.csv outside window" in captured
    assert out.loc[0, "home_team"] == ""
    assert out.loc[0, "away_team"] == ""


def test_filter_bet_recs_window_user_mode_does_not_raise_on_stale(monkeypatch):
    from datetime import datetime, timezone

    monkeypatch.setattr(bdc, "NOW_UTC", datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc))
    monkeypatch.setenv("USER_WINDOW_ONLY", "1")

    df = pd.DataFrame(
        [
            {"game_id": "stale", "game_datetime_utc": "2026-03-01T01:00:00Z"},
        ]
    )

    out = bdc._filter_bet_recs_window(df)
    assert out.empty
