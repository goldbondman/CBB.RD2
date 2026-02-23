"""
Tests for build_backtest_csvs.py — validates grading logic, CSV generation,
and edge case handling for the backtest reporting system.
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.build_backtest_csvs import (
    grade_prediction,
    grade_all,
    compute_period_stats,
    build_summary,
    build_by_model,
    build_weekly,
    build_by_conference,
    build_calibration,
    build_by_edge,
    build_model_matrix,
    compute_streak,
    compute_longest_streaks,
    validate_results_log,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_graded_row(**overrides) -> pd.Series:
    """Create a single prediction row with sensible defaults."""
    row = {
        "game_id": "401700001",
        "game_date": "2026-01-15",
        "home_team": "Duke",
        "away_team": "UNC",
        "pred_spread": -5.0,
        "actual_spread": -3.0,
        "spread_line": -4.0,
        "total_line": 145.0,
        "pred_total": 148.0,
        "actual_total": 150.0,
        "model_confidence": 0.68,
        "edge_flag": 1,
        "sub_model": "M1",
        "conference": "ACC",
        "week_label": "W3",
    }
    row.update(overrides)
    return pd.Series(row)


def _make_results_df(n=50, **col_overrides) -> pd.DataFrame:
    """Create a DataFrame mimicking results_log.csv with n rows."""
    np.random.seed(42)
    now = datetime.now(timezone.utc)
    rows = []
    for i in range(n):
        pred_spread = np.random.uniform(-15, 15)
        actual_spread = pred_spread + np.random.normal(0, 5)
        spread_line = pred_spread + np.random.uniform(-3, 3)
        pred_total = np.random.uniform(130, 165)
        actual_total = pred_total + np.random.normal(0, 8)
        total_line = pred_total + np.random.uniform(-4, 4)
        game_date = (now - timedelta(days=np.random.randint(0, 90))).strftime("%Y-%m-%d")

        rows.append({
            "game_id": f"40170{i:04d}",
            "game_date": game_date,
            "home_team": f"Team{i%10}",
            "away_team": f"Team{(i+5)%10}",
            "pred_spread": round(pred_spread, 1),
            "actual_spread": round(actual_spread, 1),
            "spread_line": round(spread_line, 1),
            "total_line": round(total_line, 1),
            "pred_total": round(pred_total, 1),
            "actual_total": round(actual_total, 1),
            "model_confidence": round(np.random.uniform(0.5, 0.95), 3),
            "edge_flag": int(np.random.random() > 0.7),
            "sub_model": f"M{(i % 7) + 1}",
            "conference": ["ACC", "SEC", "Big Ten", "Big 12", "Pac-12"][i % 5],
            "week_label": f"W{(i // 7) + 1}",
        })
    df = pd.DataFrame(rows)
    for col, val in col_overrides.items():
        df[col] = val
    return df


# ── Tests: Grading Logic (Section A) ────────────────────────────────────────

class TestGradePrediction:
    """Tests for the grade_prediction function."""

    def test_home_team_wins_correctly_graded(self):
        """Model predicts home win (neg spread), home actually wins."""
        row = _make_graded_row(pred_spread=-5.0, actual_spread=-3.0)
        result = grade_prediction(row)
        assert result["predicted_winner"] == "home"
        assert result["actual_winner"] == "home"
        assert result["winner_correct"] is True

    def test_away_team_wins_correctly_graded(self):
        """Model predicts away win (pos spread), away actually wins."""
        row = _make_graded_row(pred_spread=5.0, actual_spread=3.0)
        result = grade_prediction(row)
        assert result["predicted_winner"] == "away"
        assert result["actual_winner"] == "away"
        assert result["winner_correct"] is True

    def test_wrong_winner_prediction(self):
        """Model predicts home, away actually wins."""
        row = _make_graded_row(pred_spread=-5.0, actual_spread=3.0)
        result = grade_prediction(row)
        assert result["predicted_winner"] == "home"
        assert result["actual_winner"] == "away"
        assert result["winner_correct"] is False

    def test_ats_cover_home(self):
        """Model favors home more than line → home bet covers."""
        # pred_spread=-8.5 < spread_line=-6.0 → bet home
        # actual_spread=-7.0, diff = -7.0 - (-6.0) = -1.0 < 0 → cover
        row = _make_graded_row(pred_spread=-8.5, spread_line=-6.0, actual_spread=-7.0)
        result = grade_prediction(row)
        assert result["ats_correct"] is True
        assert result["ats_result"] == "WIN"

    def test_ats_no_cover_home(self):
        """Model favors home more than line → home bet does NOT cover."""
        # pred_spread=-8.5 < spread_line=-6.0 → bet home
        # actual_spread=-5.0, diff = -5.0 - (-6.0) = 1.0 > 0 → no cover
        row = _make_graded_row(pred_spread=-8.5, spread_line=-6.0, actual_spread=-5.0)
        result = grade_prediction(row)
        assert result["ats_correct"] is False
        assert result["ats_result"] == "LOSS"

    def test_ats_push(self):
        """Actual spread lands exactly on line → push."""
        row = _make_graded_row(pred_spread=-8.5, spread_line=-6.0, actual_spread=-6.0)
        result = grade_prediction(row)
        assert result["ats_correct"] is None
        assert result["ats_result"] == "PUSH"

    def test_ats_no_line(self):
        """No spread line available."""
        row = _make_graded_row(spread_line=None)
        result = grade_prediction(row)
        assert result["ats_correct"] is None
        assert result["ats_result"] == "NO_LINE"

    def test_totals_over_correct(self):
        """Model predicts OVER and total goes over."""
        row = _make_graded_row(pred_total=150.0, total_line=145.0, actual_total=148.0)
        result = grade_prediction(row)
        assert result["total_pick"] == "OVER"
        assert result["ou_correct"] is True
        assert result["ou_result"] == "WIN"

    def test_totals_under_correct(self):
        """Model predicts UNDER and total goes under."""
        row = _make_graded_row(pred_total=140.0, total_line=145.0, actual_total=142.0)
        result = grade_prediction(row)
        assert result["total_pick"] == "UNDER"
        assert result["ou_correct"] is True
        assert result["ou_result"] == "WIN"

    def test_totals_push(self):
        """Total lands exactly on the line."""
        row = _make_graded_row(pred_total=150.0, total_line=145.0, actual_total=145.0)
        result = grade_prediction(row)
        assert result["ou_correct"] is None
        assert result["ou_result"] == "PUSH"

    def test_roi_calculation_win(self):
        """Win → +0.909 units."""
        row = _make_graded_row(pred_spread=-8.5, spread_line=-6.0, actual_spread=-7.0)
        result = grade_prediction(row)
        assert result["ats_roi"] == 0.909

    def test_roi_calculation_loss(self):
        """Loss → -1.0 units."""
        row = _make_graded_row(pred_spread=-8.5, spread_line=-6.0, actual_spread=-5.0)
        result = grade_prediction(row)
        assert result["ats_roi"] == -1.0

    def test_roi_calculation_push(self):
        """Push → 0 units."""
        row = _make_graded_row(pred_spread=-8.5, spread_line=-6.0, actual_spread=-6.0)
        result = grade_prediction(row)
        assert result["ats_roi"] == 0.0

    def test_spread_error_computed(self):
        """Spread error is abs(pred - actual)."""
        row = _make_graded_row(pred_spread=-5.0, actual_spread=-3.0)
        result = grade_prediction(row)
        assert result["spread_error"] == 2.0

    def test_total_error_computed(self):
        """Total error is abs(pred - actual)."""
        row = _make_graded_row(pred_total=148.0, actual_total=150.0)
        result = grade_prediction(row)
        assert result["total_error"] == 2.0


class TestGradeAll:
    """Tests for grade_all() function."""

    def test_adds_grading_columns(self):
        """grade_all should add all grading columns to the DataFrame."""
        df = _make_results_df(20)
        graded = grade_all(df)
        expected_cols = ["predicted_winner", "actual_winner", "winner_correct",
                         "ats_correct", "ats_result", "ou_correct", "ou_result",
                         "total_pick", "ats_roi", "ou_roi", "spread_error", "total_error"]
        for col in expected_cols:
            assert col in graded.columns, f"Missing column: {col}"

    def test_preserves_original_columns(self):
        """Original columns should still be present after grading."""
        df = _make_results_df(10)
        graded = grade_all(df)
        for col in ["game_id", "pred_spread", "actual_spread"]:
            assert col in graded.columns


# ── Tests: Streak Computation ────────────────────────────────────────────────

class TestStreaks:
    """Tests for streak computation helpers."""

    def test_win_streak(self):
        results = pd.Series([False, True, True, True])
        assert compute_streak(results) == 3

    def test_loss_streak(self):
        results = pd.Series([True, False, False])
        assert compute_streak(results) == -2

    def test_empty_streak(self):
        assert compute_streak(pd.Series(dtype=bool)) == 0

    def test_single_win(self):
        assert compute_streak(pd.Series([True])) == 1

    def test_single_loss(self):
        assert compute_streak(pd.Series([False])) == -1

    def test_longest_streaks(self):
        results = pd.Series([True, True, True, False, False, True, True])
        win, loss = compute_longest_streaks(results)
        assert win == 3
        assert loss == 2


# ── Tests: Period Stats (Section B) ─────────────────────────────────────────

class TestComputePeriodStats:
    """Tests for compute_period_stats function."""

    def test_basic_stats(self):
        df = _make_results_df(50)
        df = grade_all(df)
        stats = compute_period_stats(df, "season")
        assert stats["period"] == "season"
        assert stats["total_games"] == 50
        assert stats["ats_graded"] > 0
        assert 0 <= stats["ats_pct"] <= 1 if stats["ats_pct"] is not None else True

    def test_empty_df(self):
        df = _make_results_df(0)
        # Manually add required columns for empty case
        for col in ["ats_result", "ou_result", "winner_correct",
                     "ats_roi", "ou_roi", "spread_error", "total_error"]:
            df[col] = pd.Series(dtype=float)
        stats = compute_period_stats(df, "empty")
        assert stats["total_games"] == 0


class TestBuildSummary:
    """Tests for build_summary function."""

    def test_produces_5_periods(self):
        df = grade_all(_make_results_df(50))
        summary = build_summary(df)
        assert len(summary) == 5
        assert list(summary["period"]) == ["season", "l60", "l30", "l14", "l7"]


# ── Tests: Per-Model Stats (Section C) ──────────────────────────────────────

class TestBuildByModel:
    """Tests for build_by_model function."""

    def test_produces_rows_for_available_models(self, monkeypatch):
        monkeypatch.setattr("build_backtest_csvs.WEIGHTS_JSON",
                            Path("/tmp/nonexistent_weights.json"))
        df = grade_all(_make_results_df(70))
        result = build_by_model(df)
        assert len(result) > 0
        assert "model_id" in result.columns
        assert "season_ats_pct" in result.columns

    def test_handles_missing_sub_model(self, monkeypatch):
        monkeypatch.setattr("build_backtest_csvs.WEIGHTS_JSON",
                            Path("/tmp/nonexistent_weights.json"))
        df = grade_all(_make_results_df(50))
        df = df.drop(columns=["sub_model"], errors="ignore")
        result = build_by_model(df)
        # Should either produce proxy-based rows or empty df
        assert isinstance(result, pd.DataFrame)


# ── Tests: Weekly (Section D) ───────────────────────────────────────────────

class TestBuildWeekly:
    """Tests for build_weekly function."""

    def test_produces_weekly_rows(self):
        df = grade_all(_make_results_df(100))
        weekly = build_weekly(df)
        assert isinstance(weekly, pd.DataFrame)
        if not weekly.empty:
            assert "week_label" in weekly.columns
            assert "ats_pct" in weekly.columns


# ── Tests: Conference (Section E) ───────────────────────────────────────────

class TestBuildByConference:
    """Tests for build_by_conference function."""

    def test_produces_conference_rows(self):
        df = grade_all(_make_results_df(100))
        conf = build_by_conference(df)
        assert isinstance(conf, pd.DataFrame)
        if not conf.empty:
            assert "conference" in conf.columns

    def test_handles_missing_conference_column(self):
        df = grade_all(_make_results_df(50))
        df = df.drop(columns=["conference"], errors="ignore")
        conf = build_by_conference(df)
        assert conf.empty


# ── Tests: Calibration (Section F) ──────────────────────────────────────────

class TestBuildCalibration:
    """Tests for build_calibration function."""

    def test_produces_calibration_rows(self):
        df = grade_all(_make_results_df(200))
        calib = build_calibration(df)
        assert isinstance(calib, pd.DataFrame)
        if not calib.empty:
            assert "confidence_bucket" in calib.columns
            assert "calibration_status" in calib.columns
            valid_statuses = {"WELL_CALIBRATED", "OVERCONFIDENT", "UNDERCONFIDENT"}
            assert set(calib["calibration_status"].unique()).issubset(valid_statuses)

    def test_handles_missing_confidence(self):
        df = grade_all(_make_results_df(50))
        df = df.drop(columns=["model_confidence"], errors="ignore")
        calib = build_calibration(df)
        assert calib.empty


# ── Tests: Edge Tiers (Section G) ───────────────────────────────────────────

class TestBuildByEdge:
    """Tests for build_by_edge function."""

    def test_produces_edge_rows(self):
        df = grade_all(_make_results_df(100))
        edge = build_by_edge(df)
        assert isinstance(edge, pd.DataFrame)
        assert len(edge) > 0
        assert "edge_tier" in edge.columns

    def test_handles_missing_spread_line(self):
        df = grade_all(_make_results_df(50))
        df = df.drop(columns=["spread_line"], errors="ignore")
        edge = build_by_edge(df)
        assert edge.empty


# ── Tests: Model Matrix (Section H) ─────────────────────────────────────────

class TestBuildModelMatrix:
    """Tests for build_model_matrix function."""

    def test_produces_matrix_rows(self):
        df = grade_all(_make_results_df(140))
        matrix = build_model_matrix(df)
        assert isinstance(matrix, pd.DataFrame)
        if not matrix.empty:
            assert "model_a" in matrix.columns
            assert "model_b" in matrix.columns

    def test_handles_missing_sub_model(self):
        df = grade_all(_make_results_df(50))
        df = df.drop(columns=["sub_model"], errors="ignore")
        matrix = build_model_matrix(df)
        assert matrix.empty


# ── Tests: Validation ───────────────────────────────────────────────────────

class TestValidation:
    """Tests for validate_results_log."""

    def test_valid_df_passes(self):
        df = _make_results_df(20)
        assert validate_results_log(df) is True

    def test_missing_required_column_fails(self):
        df = _make_results_df(20).drop(columns=["pred_spread"])
        assert validate_results_log(df) is False


# ── Tests: Bootstrap (missing results_log.csv) ─────────────────────────────

class TestBootstrapMissingResultsLog:
    """Tests that main() exits cleanly when results_log.csv is missing."""

    def test_main_exits_zero_when_results_log_missing(self, tmp_path, monkeypatch):
        """main() should exit 0 and write an empty graded log placeholder
        when results_log.csv does not exist."""
        import evaluation.build_backtest_csvs as mod

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        csv_dir = data_dir / "csv"

        monkeypatch.setattr(mod, "RESULTS_LOG", data_dir / "results_log.csv")
        monkeypatch.setattr(mod, "GRADED_LOG", data_dir / "results_log_graded.csv")
        monkeypatch.setattr(mod, "CSV_DIR", csv_dir)
        monkeypatch.setattr("sys.argv", ["build_backtest_csvs.py", "--section", "grade-only"])

        with pytest.raises(SystemExit) as exc_info:
            mod.main()

        assert exc_info.value.code == 0
        assert (data_dir / "results_log_graded.csv").exists()
