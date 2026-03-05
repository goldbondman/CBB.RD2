from pathlib import Path

import pandas as pd

from pipeline.advanced_metrics import advanced_metrics_formulas as f
from pipeline.advanced_metrics import build_advanced_metrics as builder


def test_core_formula_math():
    assert round(f.odi_star(0.05, 0.03, 0.02, 0.01), 4) == 0.0330
    assert round(f.pei(0.33, 0.28, 0.19, 0.16), 4) == 0.0800
    assert round(f.posw(0.08, 68.5), 4) == 0.0548
    assert round(f.svi(1.0, 0.6, -0.2), 4) == 1.2000
    assert round(f.pxp(0.7, 0.2, 0.6), 4) == 0.5200


def test_safe_zscore_zero_std():
    assert f.safe_zscore(10.0, 10.0, 0.0) == 0.0
    s = pd.Series([5.0, 5.0, 5.0])
    z = f.zscore_series(s)
    assert (z.fillna(0.0) == 0.0).all()


def test_builder_blocked_metrics_do_not_raise(tmp_path: Path, monkeypatch):
    team_path = tmp_path / "team_game_metrics.csv"
    out_path = tmp_path / "advanced_metrics.csv"

    # Intentionally sparse inputs: enough keys to run, missing many metric inputs.
    pd.DataFrame(
        [
            {
                "event_id": "1",
                "team_id": "A",
                "opponent_id": "B",
                "home_away": "home",
                "game_datetime_utc": "2026-03-05T03:00:00Z",
            },
            {
                "event_id": "1",
                "team_id": "B",
                "opponent_id": "A",
                "home_away": "away",
                "game_datetime_utc": "2026-03-05T03:00:00Z",
            },
        ]
    ).to_csv(team_path, index=False)

    monkeypatch.setattr(builder, "TEAM_GAME_METRICS", team_path)
    monkeypatch.setattr(builder, "TEAM_GAME_WEIGHTED", tmp_path / "missing_weighted.csv")
    monkeypatch.setattr(builder, "PLAYER_GAME_METRICS", tmp_path / "missing_players.csv")
    monkeypatch.setattr(builder, "ROTATION_FEATURES", tmp_path / "missing_rotation.csv")
    monkeypatch.setattr(builder, "SITUATIONAL_FEATURES", tmp_path / "missing_situational.csv")
    monkeypatch.setattr(builder, "TRAVEL_FATIGUE", tmp_path / "missing_travel.csv")
    monkeypatch.setattr(builder, "GAMES", tmp_path / "missing_games.csv")

    out = builder.build_advanced_metrics(output_path=out_path)

    assert out_path.exists()
    assert len(out) == 2
    assert "metric_status_SME" in out.columns
    assert out["metric_status_SME"].astype(str).str.startswith("BLOCKED_MISSING_INPUT").all()
    assert out["metric_status_ALT"].astype(str).str.startswith("BLOCKED_MISSING_INPUT").all()
