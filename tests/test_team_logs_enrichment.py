import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
import pytest

from espn_pipeline import (
    _append_dedupe_write,
    _enrich_team_rows_from_scoreboard,
    _validate_team_log_enrichment,
)


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


def test_validation_treats_string_placeholders_as_nulls():
    df = pd.DataFrame([
        {
            "completed": "true",
            "home_conference": "None",
            "away_conference": "nan",
            "home_h1": "",
            "home_h2": "None",
            "away_h1": "nan",
            "away_h2": "",
            "home_wins": "None",
            "home_losses": "nan",
            "spread": "None",
            "over_under": "nan",
            "home_ml": "",
            "away_ml": "None",
            "odds_provider": "nan",
            "odds_details": "",
        }
    ])

    with pytest.raises(ValueError):
        _validate_team_log_enrichment(df)


def test_scoreboard_enrichment_maps_home_away_record_splits_correctly():
    """home_wins/losses/away_wins/losses should use the per-team record splits,
    not the overall wins column which was the previous (buggy) behaviour."""
    df_team = pd.DataFrame([
        {"event_id": "1", "team_id": "10", "home_away": "home",
         "home_wins": None, "home_losses": None,
         "away_wins": None, "away_losses": None,
         "conf_wins": None, "conf_losses": None, "win_pct": None},
        {"event_id": "1", "team_id": "20", "home_away": "away",
         "home_wins": None, "home_losses": None,
         "away_wins": None, "away_losses": None,
         "conf_wins": None, "conf_losses": None, "win_pct": None},
    ])
    games_df = pd.DataFrame([
        {
            "game_id": "1",
            "home_team_id": "10", "away_team_id": "20",
            "home_conference": "ACC", "away_conference": "SEC",
            # Overall records
            "home_wins": 20, "home_losses": 4,
            "away_wins": 18, "away_losses": 6,
            # Home-court / road record splits (double-prefix from parse_scoreboard_event)
            "home_home_wins": 12, "home_home_losses": 1,
            "home_away_wins": 8,  "home_away_losses": 3,
            "away_home_wins": 10, "away_home_losses": 2,
            "away_away_wins": 8,  "away_away_losses": 4,
            # Conference records
            "home_conf_wins": 10, "home_conf_losses": 2,
            "away_conf_wins": 9,  "away_conf_losses": 3,
        }
    ])

    enriched = _enrich_team_rows_from_scoreboard(df_team, games_df)
    home = enriched.loc[enriched["team_id"] == "10"].iloc[0]
    away = enriched.loc[enriched["team_id"] == "20"].iloc[0]

    # Home team should get home_home_wins (12) not home_wins (20)
    assert int(home["home_wins"]) == 12
    assert int(home["home_losses"]) == 1
    assert int(home["away_wins"]) == 8
    assert int(home["away_losses"]) == 3

    # Away team should get away_home_wins (10) not home_wins (20)
    assert int(away["home_wins"]) == 10
    assert int(away["home_losses"]) == 2
    assert int(away["away_wins"]) == 8
    assert int(away["away_losses"]) == 4

    assert int(home["conf_wins"]) == 10
    assert int(away["conf_wins"]) == 9


def test_enrichment_handles_null_like_strings_from_csv_roundtrip():
    """When games_df has been through CSV write/read, None becomes 'None' string.
    The enrichment should treat these null-like strings as actual nulls."""
    df_team = pd.DataFrame([
        {"event_id": "1", "team_id": "10", "home_away": "home",
         "spread": "None", "home_conference": "nan", "wins": "20", "losses": "4",
         "win_pct": "None"},
    ])
    games_df = pd.DataFrame([
        {
            "game_id": "1",
            "home_team_id": "10", "away_team_id": "20",
            "home_conference": "ACC", "away_conference": "SEC",
            "spread": "-3.5",
            "home_wins": "20", "home_losses": "4",
        }
    ])

    enriched = _enrich_team_rows_from_scoreboard(df_team, games_df)
    row = enriched.iloc[0]

    assert row["home_conference"] == "ACC"
    assert float(row["spread"]) == -3.5
    assert float(row["win_pct"]) == pytest.approx(0.833, abs=0.001)


def test_enrichment_populates_all_target_columns():
    """All target null-guard columns should be populated after enrichment
    when both summary and scoreboard have data."""
    df_team = pd.DataFrame([
        {
            "event_id": "1", "team_id": "10", "home_away": "home",
            # Simulate summary providing some data but missing enrichment fields
            "conference": "ACC", "wins": 20, "losses": 4,
            "home_conference": None, "away_conference": None,
            "home_rank": None, "away_rank": None,
            "spread": None, "over_under": None,
            "home_ml": None, "away_ml": None,
            "odds_provider": None, "odds_details": None,
            "home_h1": None, "home_h2": None,
            "away_h1": None, "away_h2": None,
            "home_wins": None, "home_losses": None,
            "away_wins": None, "away_losses": None,
            "conf_wins": None, "conf_losses": None,
            "win_pct": None,
            "h1_pts": None, "h2_pts": None,
            "h1_pts_against": None, "h2_pts_against": None,
        },
    ])
    games_df = pd.DataFrame([
        {
            "game_id": "1",
            "home_team_id": "10", "away_team_id": "20",
            "home_conference": "ACC", "away_conference": "SEC",
            "home_rank": 7, "away_rank": 21,
            "spread": -3.5, "over_under": 145.5,
            "home_ml": -160, "away_ml": 140,
            "odds_provider": "TestBook", "odds_details": "Home -3.5",
            "home_h1": 35, "home_h2": 37, "away_h1": 31, "away_h2": 30,
            "home_wins": 20, "home_losses": 4,
            "away_wins": 18, "away_losses": 6,
            "home_home_wins": 12, "home_home_losses": 1,
            "home_away_wins": 8, "home_away_losses": 3,
            "home_conf_wins": 10, "home_conf_losses": 2,
            "away_conf_wins": 9, "away_conf_losses": 3,
        }
    ])

    enriched = _enrich_team_rows_from_scoreboard(df_team, games_df)
    row = enriched.iloc[0]

    assert row["home_conference"] == "ACC"
    assert row["away_conference"] == "SEC"
    assert float(row["spread"]) == -3.5
    assert float(row["over_under"]) == 145.5
    assert float(row["home_ml"]) == -160
    assert float(row["away_ml"]) == 140
    assert row["odds_provider"] == "TestBook"
    assert row["odds_details"] == "Home -3.5"
    assert int(row["home_h1"]) == 35
    assert int(row["home_h2"]) == 37
    assert int(row["away_h1"]) == 31
    assert int(row["away_h2"]) == 30
    assert int(row["home_wins"]) == 12
    assert int(row["home_losses"]) == 1
    assert int(row["away_wins"]) == 8
    assert int(row["away_losses"]) == 3
    assert int(row["conf_wins"]) == 10
    assert int(row["conf_losses"]) == 2
    assert int(row["h1_pts"]) == 35
    assert int(row["h2_pts"]) == 37
    assert int(row["h1_pts_against"]) == 31
    assert int(row["h2_pts_against"]) == 30
    assert float(row["win_pct"]) == pytest.approx(0.833, abs=0.001)


def test_append_dedupe_sanitizes_none_strings(tmp_path):
    """_append_dedupe_write should convert 'None'/'nan' strings to actual NaN
    so CSVs don't contain misleading text."""
    path = tmp_path / "test.csv"
    df = pd.DataFrame([{"a": None, "b": 5, "c": float("nan"), "pulled_at_utc": "2026-01-01T00:00:00+00:00"}])
    result = _append_dedupe_write(path, df, unique_keys=["b"])

    assert pd.isna(result.iloc[0]["a"])
    assert result.iloc[0]["b"] == "5"
    assert pd.isna(result.iloc[0]["c"])


def test_append_dedupe_write_persist_false_does_not_touch_disk(tmp_path):
    path = tmp_path / "team_game_logs.csv"
    existing = pd.DataFrame([{"event_id": "1", "team_id": "10", "value": "A", "pulled_at_utc": "2026-01-01T00:00:00+00:00"}])
    existing.to_csv(path, index=False)

    new = pd.DataFrame([{"event_id": "1", "team_id": "10", "value": "B", "pulled_at_utc": "2026-01-02T00:00:00+00:00"}])
    combined = _append_dedupe_write(
        path,
        new,
        unique_keys=["event_id", "team_id"],
        persist=False,
    )

    on_disk = pd.read_csv(path, dtype=str)
    assert on_disk.iloc[0]["value"] == "A"
    assert combined.iloc[0]["value"] == "B"
