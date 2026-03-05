from pathlib import Path

import pandas as pd

from build_training_data import build_canonical_market_table, build_canonical_prediction_table


def test_build_canonical_market_table_prefers_closing(tmp_path: Path):
    latest = pd.DataFrame(
        [
            {
                "event_id": "401",
                "capture_type": "opening",
                "captured_at_utc": "2026-03-01T00:00:00Z",
                "home_spread_open": -4.5,
                "home_spread_current": -4.5,
                "total_current": 140.5,
            },
            {
                "event_id": "401",
                "capture_type": "closing",
                "captured_at_utc": "2026-03-01T05:00:00Z",
                "home_spread_current": -6.0,
                "total_current": 142.0,
            },
        ]
    )
    latest.to_csv(tmp_path / "market_lines_latest.csv", index=False)

    market, diag = build_canonical_market_table(tmp_path)
    assert len(diag) == 4
    assert len(market) == 1
    row = market.iloc[0]
    assert row["game_id"] == "401"
    assert row["closing_spread"] == -6.0
    assert row["total_line"] == 142.0


def test_build_canonical_prediction_table_uses_priority_and_latest(tmp_path: Path):
    hist = pd.DataFrame(
        [
            {"game_id": "401", "pred_spread": -3.5, "pred_total": 141.0, "predicted_at_utc": "2026-03-01T01:00:00Z"},
        ]
    )
    latest = pd.DataFrame(
        [
            {"game_id": "401", "pred_spread": -5.0, "pred_total": 145.0, "predicted_at_utc": "2026-03-01T02:00:00Z"},
            {"game_id": "402", "pred_spread": 2.0, "pred_total": 138.0, "predicted_at_utc": "2026-03-01T02:00:00Z"},
        ]
    )
    hist.to_csv(tmp_path / "predictions_history.csv", index=False)
    latest.to_csv(tmp_path / "predictions_latest.csv", index=False)

    preds, _ = build_canonical_prediction_table(tmp_path)
    by_gid = {row["game_id"]: row for _, row in preds.iterrows()}
    assert by_gid["401"]["pred_spread"] == -3.5
    assert by_gid["401"]["pred_total"] == 141.0
    assert by_gid["402"]["pred_spread"] == 2.0
