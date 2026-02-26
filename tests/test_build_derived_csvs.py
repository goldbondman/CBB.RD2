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
                "game_date": "2026-02-25",
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
                "game_date": "2026-02-25",
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

    build_upset_watch_csv(predictions)

    assert captured["stem"] == "upset_watch.csv"
    model_spreads = captured["df"]["model_spread"].tolist()
    assert model_spreads
    assert -3.15 not in model_spreads
    assert 1.8 in model_spreads
