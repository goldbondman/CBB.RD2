"""
Tests for backtest_engine.py — BacktestEngine class, grade_picks,
compute_alignment_stats, run_full_backtest, and CLI command helpers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest_engine import BacktestEngine, cmd_backtest, cmd_live_signals


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def sample_data_dir(tmp_path: Path) -> Path:
    """Create minimal CSV fixtures matching the picks + games schema."""
    (tmp_path / "handicappers.csv").write_text(
        "handicapper_id,handle,tier,status,lifetime_roi,win_pct,total_picks\n"
        "1,@CBB_Edge,sharp,active,0.032,0.562,245\n"
        "2,@HoopsLock,sharp,active,0.018,0.548,198\n"
    )
    (tmp_path / "raw_tweets.csv").write_text(
        "tweet_id,handicapper_id,text,created_at,ingested_at\n"
        "1001,1,Duke -5.5,2026-02-28T18:30:00Z,2026-02-28T18:35:00Z\n"
    )
    (tmp_path / "raw_picks.csv").write_text(
        "raw_pick_id,handicapper_id,tweet_id,raw_text,line,units,parse_status\n"
        "1,1,1001,Duke -5.5,-5.5,1.0,parsed\n"
        "2,2,1002,Kansas -7.0,-7.0,1.0,parsed\n"
    )
    (tmp_path / "picks.csv").write_text(
        "pick_id,raw_pick_id,handicapper_id,game_id,team,line,units,mapping_status\n"
        "301,1,1,401900101,Duke,-5.5,1.0,mapped\n"
        "302,2,2,401900102,Kansas,-7.0,1.0,mapped\n"
        "303,1,1,401900103,Gonzaga,-6.0,2.0,mapped\n"
    )
    (tmp_path / "games.csv").write_text(
        "game_id,date,home_team,away_team,closing_spread,total_line\n"
        "401900101,2026-02-28,Duke Blue Devils,Syracuse Orange,-5.5,138.5\n"
        "401900102,2026-02-28,Kansas Jayhawks,Baylor Bears,-7.0,148.0\n"
        "401900103,2026-03-01,Gonzaga Bulldogs,Saint Mary's Gaels,-6.0,145.5\n"
    )
    return tmp_path


@pytest.fixture()
def engine(sample_data_dir: Path) -> BacktestEngine:
    return BacktestEngine(str(sample_data_dir))


# ── Tests: BacktestEngine.grade_picks ────────────────────────────────────────

class TestGradePicks:
    def test_returns_dataframe(self, engine: BacktestEngine) -> None:
        graded = engine.grade_picks()
        assert isinstance(graded, pd.DataFrame)

    def test_has_required_columns(self, engine: BacktestEngine) -> None:
        graded = engine.grade_picks()
        for col in ('pick_id', 'handicapper_id', 'game_id', 'payout', 'roi'):
            assert col in graded.columns, f"Missing column: {col}"

    def test_row_count_matches_picks(self, engine: BacktestEngine) -> None:
        graded = engine.grade_picks()
        assert len(graded) == len(engine.data['picks'])

    def test_payout_sign_covered(self, engine: BacktestEngine) -> None:
        """Covered picks should have positive payout."""
        graded = engine.grade_picks()
        covered = graded[graded['covered'].fillna(False).astype(bool)]
        assert (covered['payout'] > 0).all()

    def test_payout_sign_not_covered(self, engine: BacktestEngine) -> None:
        """Uncovered picks should have negative payout."""
        graded = engine.grade_picks()
        lost = graded[graded['covered'].fillna(True) == False]
        assert (lost['payout'] < 0).all()

    def test_roi_equals_payout_over_units(self, engine: BacktestEngine) -> None:
        graded = engine.grade_picks()
        expected = graded['payout'] / graded['units']
        pd.testing.assert_series_equal(graded['roi'], expected, check_names=False)

    def test_empty_picks_returns_empty(self, tmp_path: Path) -> None:
        """Empty picks → empty DataFrame."""
        (tmp_path / "handicappers.csv").write_text(
            "handicapper_id,handle,tier,status,lifetime_roi,win_pct,total_picks\n"
        )
        (tmp_path / "raw_tweets.csv").write_text(
            "tweet_id,handicapper_id,text,created_at,ingested_at\n"
        )
        (tmp_path / "raw_picks.csv").write_text(
            "raw_pick_id,handicapper_id,tweet_id,raw_text,line,units,parse_status\n"
        )
        (tmp_path / "picks.csv").write_text(
            "pick_id,raw_pick_id,handicapper_id,game_id,team,line,units,mapping_status\n"
        )
        (tmp_path / "games.csv").write_text(
            "game_id,date,home_team,away_team,closing_spread,total_line\n"
        )
        eng = BacktestEngine(str(tmp_path))
        graded = eng.grade_picks()
        assert graded.empty


# ── Tests: BacktestEngine.compute_alignment_stats ────────────────────────────

class TestComputeAlignmentStats:
    def test_returns_dataframe(self, engine: BacktestEngine) -> None:
        graded = engine.grade_picks()
        alignment = engine.compute_alignment_stats(graded)
        assert isinstance(alignment, pd.DataFrame)

    def test_has_alignment_column(self, engine: BacktestEngine) -> None:
        graded = engine.grade_picks()
        alignment = engine.compute_alignment_stats(graded)
        assert 'model_crowd_alignment' in alignment.columns

    def test_alignment_values_valid(self, engine: BacktestEngine) -> None:
        graded = engine.grade_picks()
        alignment = engine.compute_alignment_stats(graded)
        valid = {'model_plus_crowd', 'model_vs_crowd', 'mixed'}
        assert set(alignment['model_crowd_alignment'].unique()).issubset(valid)

    def test_empty_input_returns_empty(self, engine: BacktestEngine) -> None:
        result = engine.compute_alignment_stats(pd.DataFrame())
        assert result.empty


# ── Tests: BacktestEngine.run_full_backtest ───────────────────────────────────

class TestRunFullBacktest:
    def test_returns_expected_keys(self, engine: BacktestEngine) -> None:
        results = engine.run_full_backtest()
        expected_keys = {'graded_picks', 'capper_stats', 'alignment_stats', 'summary'}
        assert expected_keys == set(results.keys())

    def test_summary_has_expected_fields(self, engine: BacktestEngine) -> None:
        results = engine.run_full_backtest()
        summary = results['summary']
        for field in ('total_picks', 'overall_roi', 'overall_win_pct',
                      'model_plus_crowd_roi', 'model_vs_crowd_roi'):
            assert field in summary, f"Missing summary field: {field}"

    def test_total_picks_matches_graded(self, engine: BacktestEngine) -> None:
        results = engine.run_full_backtest()
        assert results['summary']['total_picks'] == len(results['graded_picks'])

    def test_capper_stats_indexed_by_handicapper(self, engine: BacktestEngine) -> None:
        results = engine.run_full_backtest()
        capper_stats = results['capper_stats']
        assert 'total_picks' in capper_stats.columns
        assert 'win_pct' in capper_stats.columns
        assert 'roi_pct' in capper_stats.columns


# ── Tests: CLI helpers ────────────────────────────────────────────────────────

class TestCLICommands:
    def test_cmd_backtest_runs(self, sample_data_dir: Path, capsys) -> None:
        class _Args:
            data_dir = str(sample_data_dir)

        cmd_backtest(_Args())
        captured = capsys.readouterr()
        assert 'HANDICAPPER WISDOM BACKTEST RESULTS' in captured.out

    def test_cmd_backtest_saves_csv(self, sample_data_dir: Path) -> None:
        class _Args:
            data_dir = str(sample_data_dir)

        cmd_backtest(_Args())
        assert (sample_data_dir / "backtest_results.csv").exists()

    def test_cmd_live_signals_runs(self, sample_data_dir: Path, capsys) -> None:
        class _Args:
            data_dir = str(sample_data_dir)

        cmd_live_signals(_Args())
        captured = capsys.readouterr()
        # Either prints signals or "No upcoming games found."
        assert captured.out.strip() != ''

    def test_cmd_live_signals_saves_csv(self, sample_data_dir: Path) -> None:
        class _Args:
            data_dir = str(sample_data_dir)

        cmd_live_signals(_Args())
        # live_signals.csv is only written when upcoming games exist;
        # our fixture has no home_score column so all games are considered upcoming
        assert (sample_data_dir / "live_signals.csv").exists()
