from pathlib import Path

import pandas as pd

from cbb_backtester import grade_historical_predictions


def test_grade_historical_predictions_writes_partial_report_for_small_samples(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    predictions = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_datetime_utc": "2026-01-01T00:00:00Z",
                "pred_spread": -3.5,
                "pred_total": 140.5,
                "model_confidence": 0.62,
                "game_type": "REG",
            },
            {
                "game_id": "g2",
                "game_datetime_utc": "2026-01-02T00:00:00Z",
                "pred_spread": 2.0,
                "pred_total": 138.0,
                "model_confidence": 0.55,
                "game_type": "REG",
            },
            {
                "game_id": "g3",
                "game_datetime_utc": "2026-01-03T00:00:00Z",
                "pred_spread": -1.0,
                "pred_total": 142.0,
                "model_confidence": 0.58,
                "game_type": "REG",
            },
        ]
    )
    predictions.to_csv(data_dir / "predictions_20260104.csv", index=False)

    weighted = pd.DataFrame(
        [
            {
                "event_id": "g1",
                "game_datetime_utc": "2026-01-01T00:00:00Z",
                "home_away": "home",
                "points_for": 75,
                "points_against": 70,
                "home_team": "A",
                "away_team": "B",
                "home_conference": "ACC",
                "away_conference": "SEC",
            },
            {
                "event_id": "g2",
                "game_datetime_utc": "2026-01-02T00:00:00Z",
                "home_away": "home",
                "points_for": 68,
                "points_against": 73,
                "home_team": "C",
                "away_team": "D",
                "home_conference": "AAC",
                "away_conference": "MVC",
            },
            {
                "event_id": "g3",
                "game_datetime_utc": "2026-01-03T00:00:00Z",
                "home_away": "home",
                "points_for": 80,
                "points_against": 78,
                "home_team": "E",
                "away_team": "F",
                "home_conference": "Big Ten",
                "away_conference": "WCC",
            },
        ]
    )
    weighted_path = data_dir / "team_game_weighted.csv"
    weighted.to_csv(weighted_path, index=False)

    monkeypatch.setattr("cbb_backtester.WEIGHTED_CSV", weighted_path)
    monkeypatch.setattr("cbb_backtester.DATA_CSV_DIR", data_dir / "csv")

    report_df, dim_df = grade_historical_predictions(data_dir)

    assert len(report_df) == 3
    assert (data_dir / "model_accuracy_report.csv").exists()
    assert dim_df.empty


def test_grade_historical_predictions_uses_event_id_and_writes_dq_audit(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    predictions = pd.DataFrame(
        [
            {
                "event_id": "g1",
                "game_datetime_utc": "2026-01-01T00:00:00Z",
                "pred_spread": -3.5,
                "pred_total": 140.5,
            },
            {
                "event_id": "g_missing_score",
                "game_datetime_utc": "2026-01-02T00:00:00Z",
                "pred_spread": 2.0,
                "pred_total": 138.0,
            },
        ]
    )
    predictions.to_csv(data_dir / "predictions_20260104.csv", index=False)
def test_grade_historical_predictions_falls_back_to_predictions_latest(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    predictions_latest = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_datetime_utc": "2026-01-01T00:00:00Z",
                "pred_spread": -3.5,
                "pred_total": 140.5,
                "model_confidence": 0.62,
                "game_type": "REG",
            }
        ]
    )
    predictions_latest.to_csv(data_dir / "predictions_latest.csv", index=False)

    weighted = pd.DataFrame(
        [
            {
                "event_id": "g1",
                "game_datetime_utc": "2026-01-01T00:00:00Z",
                "home_away": "home",
                "points_for": 75,
                "points_against": 70,
                "home_team": "A",
                "away_team": "B",
            },
            {
                "event_id": "g_missing_score",
                "game_datetime_utc": "2026-01-02T00:00:00Z",
                "home_away": "home",
                "points_for": None,
                "points_against": None,
                "home_team": "C",
                "away_team": "D",
            },
                "home_conference": "ACC",
                "away_conference": "SEC",
            }
        ]
    )
    weighted_path = data_dir / "team_game_weighted.csv"
    weighted.to_csv(weighted_path, index=False)

    monkeypatch.setattr("cbb_backtester.WEIGHTED_CSV", weighted_path)
    monkeypatch.setattr("cbb_backtester.DATA_CSV_DIR", data_dir / "csv")

    report_df, _ = grade_historical_predictions(data_dir)

    assert len(report_df) == 1
    audit = pd.read_csv(data_dir / "dq_audit.csv")
    assert "missing_final_score" in set(audit["reason_codes"].astype(str))
