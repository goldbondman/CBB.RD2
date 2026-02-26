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
