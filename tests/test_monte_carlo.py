"""
tests/test_monte_carlo.py — Tests for cbb_monte_carlo.py

Validates:
  - GameSimInput / SimResult dataclass creation
  - Variance model (spread_std and total_std)
  - Single-game simulation produces valid outputs
  - Batch simulation (simulate_slate)
  - Cover/over/under probabilities populated correctly
  - Win probability and upset probability logic
  - Model alignment classification
  - Confidence tier assignment
  - Edge enrichment flags
  - Calibration adjustment
  - results_to_mc_columns output format
  - build_game_sim_inputs from CSV
  - Rankings loader fallback
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cbb_monte_carlo import (
    GameSimInput,
    SimResult,
    compute_spread_std,
    compute_total_std,
    simulate_game,
    simulate_slate,
    apply_calibration,
    results_to_mc_columns,
    build_game_sim_inputs,
    load_team_profiles_for_sim,
    build_game_cards,
    LEAGUE_AVG_PROFILES,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def home_favored_input():
    """Home team clearly favored (negative spread)."""
    return GameSimInput(
        game_id="401000001",
        home_team="Duke",
        away_team="LowMajor",
        home_team_id="1",
        away_team_id="2",
        ensemble_spread=-10.0,
        ensemble_total=145.0,
        home_cage_em=18.0,
        away_cage_em=-8.0,
        home_consistency=65.0,
        away_consistency=45.0,
        home_floor_em=8.0,
        away_floor_em=-15.0,
        home_ceiling_em=28.0,
        away_ceiling_em=0.0,
        home_net_rtg_l5=16.0,
        away_net_rtg_l5=-10.0,
        home_cage_t=70.0,
        away_cage_t=68.0,
        home_suffocation=60.0,
        away_suffocation=40.0,
        spread_line=-7.5,
        total_line=142.5,
    )


@pytest.fixture
def even_game_input():
    """Evenly matched game with no line."""
    return GameSimInput(
        game_id="401000002",
        home_team="TeamA",
        away_team="TeamB",
        home_team_id="3",
        away_team_id="4",
        ensemble_spread=-1.0,
        ensemble_total=140.0,
        home_consistency=50.0,
        away_consistency=50.0,
        home_cage_t=68.0,
        away_cage_t=68.0,
        home_suffocation=50.0,
        away_suffocation=50.0,
    )


@pytest.fixture
def away_favored_input():
    """Away team favored (positive spread)."""
    return GameSimInput(
        game_id="401000003",
        home_team="WeakHome",
        away_team="StrongAway",
        home_team_id="5",
        away_team_id="6",
        ensemble_spread=8.0,
        ensemble_total=135.0,
        spread_line=6.5,
        total_line=137.0,
    )


# ── Variance Model Tests ─────────────────────────────────────────────────────

class TestVarianceModel:
    def test_base_std_with_league_avg(self, even_game_input):
        """With all league-average inputs, should produce expected value."""
        std = compute_spread_std(even_game_input, base_std=10.5)
        # Avg consistency=50 → adj=1.25, pace=68 → adj=1.0, suff=50 → adj=1.0
        # So output ≈ 10.5 * 1.25 = 13.125
        assert 12.5 < std < 13.8

    def test_low_consistency_increases_variance(self):
        """Low consistency teams should produce higher variance."""
        low_cons = GameSimInput(
            game_id="x", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=0.0, ensemble_total=140.0,
            home_consistency=20.0, away_consistency=20.0,
            home_cage_t=68.0, away_cage_t=68.0,
            home_suffocation=50.0, away_suffocation=50.0,
        )
        high_cons = GameSimInput(
            game_id="x", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=0.0, ensemble_total=140.0,
            home_consistency=90.0, away_consistency=90.0,
            home_cage_t=68.0, away_cage_t=68.0,
            home_suffocation=50.0, away_suffocation=50.0,
        )
        assert compute_spread_std(low_cons) > compute_spread_std(high_cons)

    def test_elite_defense_reduces_variance(self):
        """Elite suffocation defense should compress variance."""
        elite_def = GameSimInput(
            game_id="x", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=0.0, ensemble_total=140.0,
            home_suffocation=85.0, away_suffocation=50.0,
            home_cage_t=68.0, away_cage_t=68.0,
        )
        avg_def = GameSimInput(
            game_id="x", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=0.0, ensemble_total=140.0,
            home_suffocation=50.0, away_suffocation=50.0,
            home_cage_t=68.0, away_cage_t=68.0,
        )
        assert compute_spread_std(elite_def) < compute_spread_std(avg_def)

    def test_total_std_positive(self, even_game_input):
        std = compute_total_std(even_game_input)
        assert std > 0

    def test_spread_std_always_positive(self):
        """Spread std should always be positive regardless of inputs."""
        extreme = GameSimInput(
            game_id="x", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=0.0, ensemble_total=140.0,
            home_consistency=100.0, away_consistency=100.0,
            home_cage_t=80.0, away_cage_t=80.0,
            home_suffocation=100.0, away_suffocation=100.0,
        )
        assert compute_spread_std(extreme) > 0


# ── Single Game Simulation Tests ──────────────────────────────────────────────

class TestSimulateGame:
    def test_returns_sim_result(self, home_favored_input):
        result = simulate_game(home_favored_input, n_sims=1000, seed=42)
        assert isinstance(result, SimResult)
        assert result.game_id == "401000001"
        assert result.n_sims == 1000

    def test_spread_mean_near_ensemble(self, home_favored_input):
        """Mean of simulated spreads should be close to ensemble spread."""
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert abs(result.spread_mean - home_favored_input.ensemble_spread) < 1.0

    def test_total_mean_near_ensemble(self, home_favored_input):
        """Mean of simulated totals should be close to ensemble total."""
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert abs(result.total_mean - home_favored_input.ensemble_total) < 1.5

    def test_percentiles_ordered(self, home_favored_input):
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert result.spread_p10 <= result.spread_p25
        assert result.spread_p25 <= result.spread_median
        assert result.spread_median <= result.spread_p75
        assert result.spread_p75 <= result.spread_p90
        assert result.total_p10 <= result.total_median <= result.total_p90

    def test_win_probabilities_sum_to_one(self, home_favored_input):
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert abs(result.home_win_pct + result.away_win_pct - 1.0) < 0.001

    def test_home_favored_high_home_win_pct(self, home_favored_input):
        """Home team favored by 10 pts should win most sims."""
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert result.home_win_pct > 0.7

    def test_away_favored_upset_is_home_win(self, away_favored_input):
        """When away is favored, upset = home wins."""
        result = simulate_game(away_favored_input, n_sims=5000, seed=42)
        assert result.upset_probability == result.home_win_pct

    def test_home_favored_upset_is_away_win(self, home_favored_input):
        """When home is favored, upset = away wins."""
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert result.upset_probability == result.away_win_pct

    def test_cover_probabilities_with_line(self, home_favored_input):
        """Cover probs should be populated when spread_line exists."""
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert result.home_covers_pct is not None
        assert result.away_covers_pct is not None
        assert result.push_pct is not None
        assert result.cover_probability is not None
        # Should sum to approximately 1
        total = result.home_covers_pct + result.away_covers_pct + result.push_pct
        assert abs(total - 1.0) < 0.01

    def test_cover_probabilities_none_without_line(self, even_game_input):
        """Cover probs should be None when no spread_line."""
        result = simulate_game(even_game_input, n_sims=1000, seed=42)
        assert result.home_covers_pct is None
        assert result.away_covers_pct is None
        assert result.push_pct is None
        assert result.cover_probability is None

    def test_over_under_with_line(self, home_favored_input):
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert result.over_pct is not None
        assert result.under_pct is not None
        assert 0.0 <= result.over_pct <= 1.0
        assert 0.0 <= result.under_pct <= 1.0

    def test_over_under_none_without_line(self, even_game_input):
        result = simulate_game(even_game_input, n_sims=1000, seed=42)
        assert result.over_pct is None
        assert result.under_pct is None

    def test_confidence_range(self, home_favored_input):
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert 0.50 <= result.mc_confidence <= 1.0

    def test_confidence_tier_assignment(self, home_favored_input):
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert result.mc_confidence_tier in ("LOW", "MEDIUM", "HIGH", "ELITE")

    def test_model_alignment_values(self, home_favored_input):
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert result.model_alignment in ("STRONG", "LEAN", "SPLIT")

    def test_strong_favorite_strong_alignment(self):
        """Very heavy favorite should show STRONG alignment (all percentiles same sign)."""
        inp = GameSimInput(
            game_id="strong", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=-20.0, ensemble_total=140.0,
            spread_std=6.0,  # tight std ensures p90 stays negative
        )
        result = simulate_game(inp, n_sims=5000, seed=42)
        assert result.model_alignment == "STRONG"

    def test_reproducibility_with_seed(self, home_favored_input):
        """Same seed should produce identical results."""
        r1 = simulate_game(home_favored_input, n_sims=1000, seed=123)
        r2 = simulate_game(home_favored_input, n_sims=1000, seed=123)
        assert r1.spread_mean == r2.spread_mean
        assert r1.home_win_pct == r2.home_win_pct

    def test_different_seeds_different_results(self, home_favored_input):
        """Different seeds should produce different results."""
        r1 = simulate_game(home_favored_input, n_sims=1000, seed=1)
        r2 = simulate_game(home_favored_input, n_sims=1000, seed=2)
        # Means will be similar but not identical with different seeds
        assert r1.spread_mean != r2.spread_mean

    def test_variance_flags(self):
        """Test high/low variance flag thresholds."""
        # Use custom std to control variance
        high_var_input = GameSimInput(
            game_id="hv", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=0.0, ensemble_total=140.0,
            spread_std=15.0,  # force high std
        )
        result = simulate_game(high_var_input, n_sims=5000, seed=42)
        assert result.high_variance_flag is True
        assert result.low_variance_flag is False

        low_var_input = GameSimInput(
            game_id="lv", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=0.0, ensemble_total=140.0,
            spread_std=5.0,  # force low std
        )
        result = simulate_game(low_var_input, n_sims=5000, seed=42)
        assert result.low_variance_flag is True
        assert result.high_variance_flag is False

    def test_provided_std_overrides_computed(self):
        """When spread_std is provided, it should be used instead of computed."""
        inp_custom = GameSimInput(
            game_id="c", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=-5.0, ensemble_total=140.0,
            spread_std=3.0,  # very tight
            total_std=3.0,
        )
        result = simulate_game(inp_custom, n_sims=5000, seed=42)
        # Realized std should be close to provided std
        assert abs(result.spread_std_realized - 3.0) < 0.5

    def test_outputs_are_decimals_not_percentages(self, home_favored_input):
        """All percentage outputs should be stored as decimals (0.67 not 67)."""
        result = simulate_game(home_favored_input, n_sims=5000, seed=42)
        assert 0.0 <= result.home_win_pct <= 1.0
        assert 0.0 <= result.away_win_pct <= 1.0
        assert 0.0 <= result.upset_probability <= 1.0
        if result.cover_probability is not None:
            assert 0.0 <= result.cover_probability <= 1.0
        if result.over_pct is not None:
            assert 0.0 <= result.over_pct <= 1.0


# ── Edge Enrichment Tests ─────────────────────────────────────────────────────

class TestEdgeEnrichment:
    def test_edge_confirmed(self):
        """Edge confirmed when cover_probability > 0.60 AND edge_flag=1."""
        inp = GameSimInput(
            game_id="e1", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=-15.0, ensemble_total=140.0,
            spread_line=-7.0,
            edge_flag=1,
        )
        result = simulate_game(inp, n_sims=5000, seed=42)
        # Heavy favorite covering a moderate line → should confirm
        assert result.mc_edge_confirmed is True

    def test_edge_not_confirmed_without_flag(self):
        """Edge should not be confirmed if edge_flag=0."""
        inp = GameSimInput(
            game_id="e2", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=-15.0, ensemble_total=140.0,
            spread_line=-7.0,
            edge_flag=0,
        )
        result = simulate_game(inp, n_sims=5000, seed=42)
        assert result.mc_edge_confirmed is False

    def test_edge_contradicted(self):
        """Edge contradicted when edge_flag=1 but cover_probability < 0.52."""
        inp = GameSimInput(
            game_id="e3", home_team="A", away_team="B",
            home_team_id="1", away_team_id="2",
            ensemble_spread=-5.0, ensemble_total=140.0,
            spread_std=5.0,  # tight std
            spread_line=-5.0,  # line matches spread → ~50/50 cover
            edge_flag=1,
        )
        result = simulate_game(inp, n_sims=5000, seed=42)
        # Spread and line match → cover prob near 0.50 → < 0.52
        assert result.mc_edge_contradicted is True


# ── Calibration Tests ─────────────────────────────────────────────────────────

class TestCalibration:
    def test_apply_calibration_basic(self):
        table = {"50": 1.05, "60": 0.95, "70": 1.0}
        assert apply_calibration(0.55, table) == pytest.approx(0.55 * 1.05, abs=0.01)
        assert apply_calibration(0.65, table) == pytest.approx(0.65 * 0.95, abs=0.01)

    def test_apply_calibration_clamped(self):
        """Output should be clamped to [0.50, 0.99]."""
        table = {"90": 1.5}  # would push 0.95 to 1.425
        assert apply_calibration(0.95, table) == 0.99

        table = {"50": 0.5}  # would push 0.52 to 0.26
        assert apply_calibration(0.52, table) == 0.50

    def test_missing_bucket_uses_1(self):
        """If bucket not in table, multiplier defaults to 1.0."""
        table = {"60": 0.9}
        assert apply_calibration(0.55, table) == pytest.approx(0.55, abs=0.01)


# ── Batch Simulation Tests ────────────────────────────────────────────────────

class TestSimulateSlate:
    def test_returns_list_of_results(self, home_favored_input, even_game_input):
        results = simulate_slate(
            [home_favored_input, even_game_input],
            n_sims=500,
            verbose=False,
        )
        assert len(results) == 2
        assert all(isinstance(r, SimResult) for r in results)

    def test_game_ids_preserved(self, home_favored_input, even_game_input):
        results = simulate_slate(
            [home_favored_input, even_game_input],
            n_sims=500,
            verbose=False,
        )
        assert results[0].game_id == "401000001"
        assert results[1].game_id == "401000002"

    def test_empty_slate(self):
        results = simulate_slate([], n_sims=500, verbose=False)
        assert results == []

    def test_performance_50_games(self):
        """50 games × 5,000 sims should complete in under 10 seconds."""
        import time
        games = [
            GameSimInput(
                game_id=str(i), home_team=f"Home{i}", away_team=f"Away{i}",
                home_team_id=str(i), away_team_id=str(100 + i),
                ensemble_spread=-5.0, ensemble_total=140.0,
            )
            for i in range(50)
        ]
        t0 = time.time()
        results = simulate_slate(games, n_sims=5000, verbose=False)
        elapsed = time.time() - t0
        assert len(results) == 50
        assert elapsed < 10.0, f"50 games took {elapsed:.1f}s (target: <10s)"


# ── Output Format Tests ──────────────────────────────────────────────────────

class TestOutputFormat:
    def test_results_to_mc_columns(self, home_favored_input):
        result = simulate_game(home_favored_input, n_sims=1000, seed=42)
        df = results_to_mc_columns([result])
        assert len(df) == 1
        assert "game_id" in df.columns
        assert "mc_spread_median" in df.columns
        assert "mc_cover_probability" in df.columns
        assert "mc_confidence" in df.columns
        assert "mc_confidence_tier" in df.columns
        assert "mc_n_sims" in df.columns

    def test_all_mc_columns_present(self, home_favored_input):
        result = simulate_game(home_favored_input, n_sims=1000, seed=42)
        df = results_to_mc_columns([result])
        expected_cols = [
            "game_id",
            "mc_spread_median", "mc_spread_p10", "mc_spread_p25",
            "mc_spread_p75", "mc_spread_p90", "mc_spread_std",
            "mc_total_median", "mc_total_p10", "mc_total_p90",
            "mc_home_win_pct", "mc_away_win_pct",
            "mc_cover_probability", "mc_over_pct", "mc_under_pct",
            "mc_upset_probability", "mc_confidence", "mc_confidence_tier",
            "mc_high_variance", "mc_low_variance", "mc_model_alignment",
            "mc_edge_confirmed", "mc_edge_contradicted", "mc_n_sims",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"


# ── Build Game Sim Inputs Tests ──────────────────────────────────────────────

class TestBuildGameSimInputs:
    def test_basic_build(self):
        df = pd.DataFrame([{
            "game_id": "401000001",
            "home_team": "Duke",
            "away_team": "UNC",
            "home_team_id": "1",
            "away_team_id": "2",
            "pred_spread": -5.0,
            "pred_total": 145.0,
            "spread_line": -4.5,
            "total_line": 143.0,
        }])
        inputs = build_game_sim_inputs(df, {})
        assert len(inputs) == 1
        assert inputs[0].game_id == "401000001"
        assert inputs[0].ensemble_spread == -5.0
        assert inputs[0].spread_line == -4.5

    def test_skips_rows_without_spread(self):
        df = pd.DataFrame([{
            "game_id": "401000001",
            "home_team": "Duke",
            "away_team": "UNC",
            "home_team_id": "1",
            "away_team_id": "2",
        }])
        inputs = build_game_sim_inputs(df, {})
        assert len(inputs) == 0

    def test_uses_ensemble_spread_when_available(self):
        df = pd.DataFrame([{
            "game_id": "401000001",
            "home_team": "Duke",
            "away_team": "UNC",
            "home_team_id": "1",
            "away_team_id": "2",
            "pred_spread": -5.0,
            "ens_ens_spread": -7.0,
            "pred_total": 145.0,
        }])
        inputs = build_game_sim_inputs(df, {})
        assert len(inputs) == 1
        assert inputs[0].ensemble_spread == -7.0

    def test_team_profiles_applied(self):
        df = pd.DataFrame([{
            "game_id": "401000001",
            "home_team": "Duke",
            "away_team": "UNC",
            "home_team_id": "1",
            "away_team_id": "2",
            "pred_spread": -5.0,
            "pred_total": 145.0,
        }])
        profiles = {
            "1": {"consistency_score": 75.0, "cage_em": 15.0},
            "2": {"consistency_score": 60.0, "cage_em": 8.0},
        }
        inputs = build_game_sim_inputs(df, profiles)
        assert inputs[0].home_consistency == 75.0
        assert inputs[0].away_cage_em == 8.0


# ── Rankings Loader Tests ─────────────────────────────────────────────────────

class TestLoadTeamProfiles:
    def test_missing_file_returns_empty(self, tmp_path):
        profiles = load_team_profiles_for_sim(
            str(tmp_path / "nonexistent.csv")
        )
        assert profiles == {}

    def test_loads_from_csv(self, tmp_path):
        csv_path = tmp_path / "rankings.csv"
        csv_path.write_text(
            "team_id,consistency_score,cage_em,floor_em,ceiling_em,net_rtg_l5,cage_t,suffocation\n"
            "1,70.0,15.0,5.0,25.0,12.0,72.0,65.0\n"
            "2,40.0,-5.0,-15.0,5.0,-8.0,66.0,45.0\n"
        )
        profiles = load_team_profiles_for_sim(str(csv_path))
        assert len(profiles) == 2
        assert profiles["1"]["consistency_score"] == 70.0
        assert profiles["2"]["cage_em"] == -5.0


# ── Game Cards Tests ──────────────────────────────────────────────────────────

class TestBuildGameCards:
    def test_basic_cards(self):
        combined = pd.DataFrame([{
            "game_id": "401000001",
            "home_team": "Duke",
            "away_team": "UNC",
            "pred_spread": -5.0,
            "pred_total": 145.0,
            "spread_line": -4.5,
            "total_line": 143.0,
        }])
        mc = pd.DataFrame([{
            "game_id": "401000001",
            "mc_cover_probability": 0.63,
            "mc_confidence_tier": "MEDIUM",
            "mc_spread_p25": -9.0,
            "mc_spread_p75": -2.0,
            "mc_total_p10": 130.0,
            "mc_total_p90": 155.0,
            "mc_home_win_pct": 0.72,
            "mc_away_win_pct": 0.28,
            "mc_upset_probability": 0.28,
            "mc_model_alignment": "LEAN",
            "mc_high_variance": False,
            "mc_edge_confirmed": False,
            "mc_edge_contradicted": False,
        }])
        cards = build_game_cards(combined, mc)
        assert len(cards) == 1
        assert "mc_cover_probability" in cards.columns
        assert "mc_spread_range" in cards.columns
        assert "generated_at" in cards.columns
