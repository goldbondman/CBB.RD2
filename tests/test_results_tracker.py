"""
Tests for cbb_results_tracker.py — focused on the CSV production fixes.

Validates that process_date correctly matches predictions to completed game
results and writes output CSVs, including edge cases where:
- Predictions are for wrong dates (fallback to *_latest.csv)
- Results DataFrame is missing optional columns
- Merge produces matches when game_ids overlap
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from cbb_results_tracker import (
    ResultsTracker,
    load_predictions,
    load_games_results,
    load_results_log,
    GameOutcome,
    compute_outcomes,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_predictions_csv(path: Path, game_ids: list, date_str: str = "20260220"):
    """Write a minimal predictions CSV that the tracker can load."""
    rows = []
    for gid in game_ids:
        rows.append({
            "game_id": str(gid),
            "home_team": "TeamA",
            "away_team": "TeamB",
            "home_team_id": "1",
            "away_team_id": "2",
            "game_datetime_utc": f"2026-02-{date_str[-2:]}T19:00:00+00:00",
            "neutral_site": 0,
            "pred_spread": -3.5,
            "pred_total": 145.0,
            "model_confidence": 0.65,
            "edge_flag": 0,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_games_csv(path: Path, game_ids: list, completed: bool = True,
                    include_spread: bool = True):
    """Write a minimal games.csv with completed games."""
    rows = []
    for gid in game_ids:
        row = {
            "game_id": str(gid),
            "home_team": "TeamA",
            "away_team": "TeamB",
            "home_team_id": "1",
            "away_team_id": "2",
            "game_datetime_utc": "2026-02-20T19:00:00+00:00",
            "neutral_site": 0,
            "completed": str(completed),
            "home_score": 75,
            "away_score": 70,
        }
        if include_spread:
            row["spread"] = -2.5
            row["over_under"] = 148.0
            row["home_ml"] = -150
            row["away_ml"] = 130
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


# ── Tests ────────────────────────────────────────────────────────────────────

class TestProcessDate:
    """Tests for ResultsTracker.process_date() method."""

    def test_produces_csv_when_predictions_match_completed_games(self, tmp_path, monkeypatch):
        """process_date should produce results_log.csv when predictions match completed games."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        game_ids = ["401700001", "401700002", "401700003"]
        _make_predictions_csv(data_dir / "predictions_combined_latest.csv", game_ids)
        _make_games_csv(data_dir / "games.csv", game_ids)

        # Patch module-level paths
        monkeypatch.setattr("cbb_results_tracker.DATA_DIR", data_dir)
        monkeypatch.setattr("cbb_results_tracker.PREDICTIONS_CSV", data_dir / "predictions_combined_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.ENSEMBLE_CSV", data_dir / "ensemble_predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.PRIMARY_CSV", data_dir / "predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.GAMES_CSV", data_dir / "games.csv")
        monkeypatch.setattr("cbb_results_tracker.RESULTS_LOG", data_dir / "results_log.csv")

        tracker = ResultsTracker(output_dir=data_dir)
        outcomes, alerts = tracker.process_date(target_date="20260220")

        assert len(outcomes) == 3, f"Expected 3 outcomes, got {len(outcomes)}"
        assert (data_dir / "results_log.csv").exists(), "results_log.csv should be created"

        log_df = pd.read_csv(data_dir / "results_log.csv")
        assert len(log_df) == 3

    def test_no_csv_when_no_predictions_exist(self, tmp_path, monkeypatch):
        """process_date should return empty when no prediction files exist."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monkeypatch.setattr("cbb_results_tracker.DATA_DIR", data_dir)
        monkeypatch.setattr("cbb_results_tracker.PREDICTIONS_CSV", data_dir / "predictions_combined_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.ENSEMBLE_CSV", data_dir / "ensemble_predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.PRIMARY_CSV", data_dir / "predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.GAMES_CSV", data_dir / "games.csv")
        monkeypatch.setattr("cbb_results_tracker.RESULTS_LOG", data_dir / "results_log.csv")

        tracker = ResultsTracker(output_dir=data_dir)
        outcomes, alerts = tracker.process_date(target_date="20260220")

        assert outcomes == []
        assert alerts == []

    def test_matches_when_latest_contains_target_game_ids(self, tmp_path, monkeypatch):
        """When date-specific predictions don't exist but latest has the right game_ids,
        the tracker should still match and produce CSV."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        game_ids = ["401700010", "401700011"]
        # Only create _latest.csv, no date-specific file
        _make_predictions_csv(data_dir / "predictions_combined_latest.csv", game_ids)
        _make_games_csv(data_dir / "games.csv", game_ids)

        monkeypatch.setattr("cbb_results_tracker.DATA_DIR", data_dir)
        monkeypatch.setattr("cbb_results_tracker.PREDICTIONS_CSV", data_dir / "predictions_combined_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.ENSEMBLE_CSV", data_dir / "ensemble_predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.PRIMARY_CSV", data_dir / "predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.GAMES_CSV", data_dir / "games.csv")
        monkeypatch.setattr("cbb_results_tracker.RESULTS_LOG", data_dir / "results_log.csv")

        tracker = ResultsTracker(output_dir=data_dir)
        outcomes, _ = tracker.process_date(target_date="20260220")

        assert len(outcomes) == 2
        assert (data_dir / "results_log.csv").exists()

    def test_handles_missing_spread_columns_in_games(self, tmp_path, monkeypatch):
        """process_date should not crash when games.csv is missing spread/ml columns."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        game_ids = ["401700020"]
        _make_predictions_csv(data_dir / "predictions_combined_latest.csv", game_ids)
        _make_games_csv(data_dir / "games.csv", game_ids, include_spread=False)

        monkeypatch.setattr("cbb_results_tracker.DATA_DIR", data_dir)
        monkeypatch.setattr("cbb_results_tracker.PREDICTIONS_CSV", data_dir / "predictions_combined_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.ENSEMBLE_CSV", data_dir / "ensemble_predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.PRIMARY_CSV", data_dir / "predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.GAMES_CSV", data_dir / "games.csv")
        monkeypatch.setattr("cbb_results_tracker.RESULTS_LOG", data_dir / "results_log.csv")

        tracker = ResultsTracker(output_dir=data_dir)
        outcomes, _ = tracker.process_date(target_date="20260220")

        # Should produce outcomes even without spread columns
        assert len(outcomes) == 1
        assert (data_dir / "results_log.csv").exists()

    def test_loads_all_completed_games_not_just_predicted(self, tmp_path, monkeypatch):
        """process_date should load ALL completed games, not just those matching
        prediction game_ids, so that the inner merge correctly finds overlap."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Predictions have games 1 and 2
        pred_ids = ["401700030", "401700031"]
        # Games.csv has games 1, 2, and 3 (extra completed game)
        all_ids = ["401700030", "401700031", "401700032"]

        _make_predictions_csv(data_dir / "predictions_combined_latest.csv", pred_ids)
        _make_games_csv(data_dir / "games.csv", all_ids)

        monkeypatch.setattr("cbb_results_tracker.DATA_DIR", data_dir)
        monkeypatch.setattr("cbb_results_tracker.PREDICTIONS_CSV", data_dir / "predictions_combined_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.ENSEMBLE_CSV", data_dir / "ensemble_predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.PRIMARY_CSV", data_dir / "predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.GAMES_CSV", data_dir / "games.csv")
        monkeypatch.setattr("cbb_results_tracker.RESULTS_LOG", data_dir / "results_log.csv")

        tracker = ResultsTracker(output_dir=data_dir)
        outcomes, _ = tracker.process_date(target_date="20260220")

        # Should match only the 2 predicted games, not the 3rd
        assert len(outcomes) == 2


class TestLoadPredictions:
    """Tests for load_predictions() function."""

    def test_fallback_to_latest_when_date_file_missing(self, tmp_path, monkeypatch):
        """When date-specific prediction file doesn't exist, should fallback to latest."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        _make_predictions_csv(data_dir / "predictions_combined_latest.csv",
                              ["401700040", "401700041"])

        monkeypatch.setattr("cbb_results_tracker.DATA_DIR", data_dir)
        monkeypatch.setattr("cbb_results_tracker.PREDICTIONS_CSV", data_dir / "predictions_combined_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.ENSEMBLE_CSV", data_dir / "ensemble_predictions_latest.csv")
        monkeypatch.setattr("cbb_results_tracker.PRIMARY_CSV", data_dir / "predictions_latest.csv")

        preds = load_predictions(date_filter="20260220")
        assert not preds.empty
        assert len(preds) == 2

    def test_returns_empty_when_no_files_exist(self, tmp_path, monkeypatch):
        """When no prediction files exist at all, should return empty DataFrame."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        monkeypatch.setattr("cbb_results_tracker.DATA_DIR", data_dir)
        monkeypatch.setattr("cbb_results_tracker.PREDICTIONS_CSV", data_dir / "nope.csv")
        monkeypatch.setattr("cbb_results_tracker.ENSEMBLE_CSV", data_dir / "nope2.csv")
        monkeypatch.setattr("cbb_results_tracker.PRIMARY_CSV", data_dir / "nope3.csv")

        preds = load_predictions(date_filter="20260220")
        assert preds.empty


class TestLoadGamesResults:
    """Tests for load_games_results() function."""

    def test_loads_completed_games(self, tmp_path, monkeypatch):
        """Should only return completed games."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Mix of completed and incomplete games
        rows = [
            {"game_id": "1", "home_team": "A", "away_team": "B",
             "home_team_id": "1", "away_team_id": "2",
             "completed": "True", "home_score": 80, "away_score": 70,
             "game_datetime_utc": "2026-02-20T19:00:00+00:00"},
            {"game_id": "2", "home_team": "C", "away_team": "D",
             "home_team_id": "3", "away_team_id": "4",
             "completed": "False", "home_score": "", "away_score": "",
             "game_datetime_utc": "2026-02-21T19:00:00+00:00"},
        ]
        pd.DataFrame(rows).to_csv(data_dir / "games.csv", index=False)

        monkeypatch.setattr("cbb_results_tracker.GAMES_CSV", data_dir / "games.csv")

        results = load_games_results()
        assert len(results) == 1
        assert results.iloc[0]["game_id"] == "1"

    def test_returns_empty_when_no_games_csv(self, tmp_path, monkeypatch):
        """Should return empty DataFrame when games.csv doesn't exist."""
        monkeypatch.setattr("cbb_results_tracker.GAMES_CSV", tmp_path / "nonexistent.csv")

        results = load_games_results()
        assert results.empty
