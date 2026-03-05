from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import enrichment.predictions_with_context as pwc


def test_predictions_with_context_line_reason_codes(tmp_path: Path, monkeypatch):
    preds_path = tmp_path / "predictions_combined_latest.csv"
    out_path = tmp_path / "predictions_with_context.csv"

    preds = pd.DataFrame(
        [
            {
                "event_id": "1001",
                "game_id": "1001",
                "home_team": "Home A",
                "away_team": "Away A",
                "home_team_id": "1",
                "away_team_id": "2",
                "game_datetime_utc": "2026-03-05T22:00:00Z",
                "pred_spread": -3.5,
                "model_confidence": 0.6,
            },
            {
                "event_id": "1002",
                "game_id": "1002",
                "home_team": "Home B",
                "away_team": "Away B",
                "home_team_id": "3",
                "away_team_id": "4",
                "game_datetime_utc": "2026-03-05T23:00:00Z",
                "pred_spread": 2.0,
                "model_confidence": 0.55,
            },
            {
                "event_id": "1003",
                "game_id": "1003",
                "home_team": "Home C",
                "away_team": "Away C",
                "home_team_id": "5",
                "away_team_id": "6",
                "game_datetime_utc": "2026-03-06T01:00:00Z",
                "pred_spread": 1.0,
                "model_confidence": 0.58,
            },
        ]
    )
    preds.to_csv(preds_path, index=False)

    pd.DataFrame(
        [
            {
                "event_id": "1001",
                "captured_at_utc": "2026-03-05T20:00:00Z",
                "home_spread_current": -4.0,
                "total_current": 145.5,
                "line_movement": 0.5,
            }
        ]
    ).to_csv(tmp_path / "market_lines_latest.csv", index=False)

    pd.DataFrame(
        [
            {
                "event_id": "1002",
                "captured_at_utc": "2026-03-05T20:05:00Z",
                "home_spread_current": "BAD_PARSE",
                "total_current": 140.0,
            }
        ]
    ).to_csv(tmp_path / "odds_snapshot.csv", index=False)

    monkeypatch.setattr(pwc, "DATA_DIR", tmp_path)

    pwc.build_predictions_with_context(
        predictions_path=preds_path,
        out_path=out_path,
        reference_date=pd.Timestamp("2026-03-05 12:00:00", tz="America/Los_Angeles"),
    )

    out = pd.read_csv(out_path, low_memory=False)
    by_id = {str(r["event_id"]): r for _, r in out.iterrows()}

    assert by_id["1001"]["line_status"] == "OK"
    assert pd.isna(by_id["1001"]["line_missing_reason"])

    assert by_id["1002"]["line_status"] in {"PARTIAL", "MISSING"}
    assert str(by_id["1002"]["line_missing_reason"]) in {"PARSE_ERROR", "PARTIAL", "NO_ODDS_RETURNED"}

    assert by_id["1003"]["line_status"] == "MISSING"
    assert by_id["1003"]["line_missing_reason"] == "NO_MATCHING_GAME_ID"

    unmatched_path = Path(tmp_path).parent / "debug" / "unmatched_predictions_keys.csv"
    assert unmatched_path.exists()
    unmatched = pd.read_csv(unmatched_path, low_memory=False)
    assert "1003" in unmatched["event_id"].astype(str).tolist()
