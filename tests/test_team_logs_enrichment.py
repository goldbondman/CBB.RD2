import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest

from espn_pipeline import _enrich_team_rows_from_scoreboard, _validate_team_log_enrichment


def test_scoreboard_enrichment_backfills_missing_team_log_columns():
    df_team = pd.DataFrame([
        {"event_id": "1", "team_id": "10", "home_away": "home", "home_conference": None, "spread": None},
        {"event_id": "1", "team_id": "20", "home_away": "away", "home_conference": None, "spread": None},
    ])
    games_df = pd.DataFrame([
        {
            "game_id": "1",
            "home_team_id": "10",
            "away_team_id": "20",
            "home_conference": "ACC",
            "away_conference": "SEC",
            "home_rank": 7,
            "away_rank": 21,
            "spread": -3.5,
            "over_under": 145.5,
            "home_ml": -160,
            "away_ml": 140,
            "odds_provider": "TestBook",
            "odds_details": "Home -3.5",
            "home_h1": 35,
            "home_h2": 37,
            "away_h1": 31,
            "away_h2": 30,
            "home_wins": 20,
            "home_losses": 4,
            "away_wins": 18,
            "away_losses": 6,
            "home_conf_wins": 10,
            "home_conf_losses": 2,
            "away_conf_wins": 9,
            "away_conf_losses": 3,
        }
    ])

    enriched = _enrich_team_rows_from_scoreboard(df_team, games_df)

    home = enriched.loc[enriched["team_id"] == "10"].iloc[0]
    away = enriched.loc[enriched["team_id"] == "20"].iloc[0]

    assert home["home_conference"] == "ACC"
    assert away["away_conference"] == "SEC"
    assert float(home["spread"]) == -3.5
    assert int(home["h1_pts"]) == 35
    assert int(home["h1_pts_against"]) == 31
    assert int(away["h1_pts"]) == 31
    assert int(away["h1_pts_against"]) == 35
    assert int(home["conf_wins"]) == 10
    assert int(away["conf_wins"]) == 9


def test_validation_rejects_all_null_enrichment_groups():
    df = pd.DataFrame([
        {
            "completed": "true",
            "home_conference": None,
            "away_conference": None,
            "home_h1": None,
            "home_h2": None,
            "away_h1": None,
            "away_h2": None,
            "home_wins": None,
            "home_losses": None,
            "spread": None,
            "over_under": None,
            "home_ml": None,
            "away_ml": None,
            "odds_provider": None,
            "odds_details": None,
        }
    ])

    with pytest.raises(ValueError):
        _validate_team_log_enrichment(df)


def test_validation_passes_when_columns_populated():
    df = pd.DataFrame([
        {
            "completed": "true",
            "home_conference": "ACC",
            "away_conference": "SEC",
            "home_h1": 40,
            "home_h2": 35,
            "away_h1": 33,
            "away_h2": 32,
            "home_wins": 15,
            "home_losses": 5,
            "spread": -2.5,
            "over_under": 140.5,
            "home_ml": -130,
            "away_ml": 110,
            "odds_provider": "Book",
            "odds_details": "test",
        }
    ])

    _validate_team_log_enrichment(df)
