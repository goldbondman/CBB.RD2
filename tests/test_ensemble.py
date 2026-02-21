"""
tests/test_ensemble.py — Tests for cbb_ensemble.py

Validates:
  - TeamProfile creation and field completeness
  - Each of the 7 sub-models produces valid output
  - EnsemblePredictor aggregation (weighted averaging, agreement, edge flags)
  - EnsembleConfig.from_optimized() with and without weights file
  - to_flat_dict() serialisation
  - Spread sign convention (negative = home favoured)
  - Model diversity (models produce different spreads)
  - Confidence bounds (always 0–1)
  - Neutral-site handling (no HCA)
"""

import json
import math
import tempfile
from pathlib import Path

import pytest

from cbb_ensemble import (
    TeamProfile,
    ModelPrediction,
    EnsembleResult,
    EnsembleConfig,
    EnsemblePredictor,
    FourFactorsModel,
    AdjustedEfficiencyModel,
    PythagoreanModel,
    MomentumModel,
    SituationalModel,
    CAGERankingsModel,
    RegressedEfficiencyModel,
    load_team_profiles,
    results_to_csv,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def strong_home():
    """A clearly strong home team."""
    return TeamProfile(
        team_id="1", team_name="Duke", conference="ACC",
        games_before=25,
        cage_em=18.0, cage_o=118.0, cage_d=100.0, cage_t=70.0,
        barthag=0.92, pythagorean_win_pct=0.88,
        efg_pct=55.0, tov_pct=14.0, orb_pct=34.0, ftr=32.0,
        efg_vs_opp=5.0, tov_vs_opp=-3.0, orb_vs_opp=4.0, ftr_vs_opp=2.0,
        net_rtg_l5=16.0, net_rtg_l10=17.0,
        ortg_l5=116.0, ortg_l10=117.0,
        drtg_l5=100.0, drtg_l10=100.0,
        cage_power_index=80.0, suffocation=60.0, clutch_rating=65.0,
        resume_score=75.0, dna_score=62.0,
        home_wpct=0.92, away_wpct=0.70, close_wpct=0.65,
        win_streak=5.0, consistency_score=65.0,
    )


@pytest.fixture
def weak_away():
    """A clearly weak away team."""
    return TeamProfile(
        team_id="2", team_name="LowMajor", conference="SoCon",
        games_before=22,
        cage_em=-8.0, cage_o=98.0, cage_d=106.0, cage_t=68.0,
        barthag=0.30, pythagorean_win_pct=0.30,
        efg_pct=46.0, tov_pct=20.0, orb_pct=27.0, ftr=25.0,
        efg_vs_opp=-4.0, tov_vs_opp=2.0, orb_vs_opp=-3.0, ftr_vs_opp=-2.0,
        net_rtg_l5=-10.0, net_rtg_l10=-8.0,
        ortg_l5=96.0, ortg_l10=97.0,
        drtg_l5=106.0, drtg_l10=105.0,
        cage_power_index=30.0, suffocation=40.0, clutch_rating=42.0,
        resume_score=25.0, dna_score=38.0,
        home_wpct=0.60, away_wpct=0.25, close_wpct=0.40,
        win_streak=-2.0, consistency_score=45.0,
    )


@pytest.fixture
def even_teams():
    """Two nearly identical teams for testing symmetry."""
    base = TeamProfile(
        team_id="3", team_name="TeamA", games_before=20,
        cage_em=5.0, cage_o=108.0, cage_d=103.0, cage_t=70.0,
        barthag=0.65, pythagorean_win_pct=0.60,
        net_rtg_l5=5.0, net_rtg_l10=5.0,
        ortg_l5=108.0, ortg_l10=108.0,
        cage_power_index=55.0,
    )
    return base, TeamProfile(
        team_id="4", team_name="TeamB", games_before=20,
        cage_em=5.0, cage_o=108.0, cage_d=103.0, cage_t=70.0,
        barthag=0.65, pythagorean_win_pct=0.60,
        net_rtg_l5=5.0, net_rtg_l10=5.0,
        ortg_l5=108.0, ortg_l10=108.0,
        cage_power_index=55.0,
    )


# ── TeamProfile Tests ─────────────────────────────────────────────────────────

class TestTeamProfile:
    def test_default_creation(self):
        """TeamProfile should be creatable with all defaults."""
        tp = TeamProfile()
        assert tp.team_id == ""
        assert tp.cage_em == 0.0
        assert tp.games_before == 0

    def test_all_backtester_fields_present(self):
        """All fields from build_team_state_before() must exist."""
        required = [
            "team_id", "team_name", "conference", "games_before",
            "cage_em", "cage_o", "cage_d", "cage_t", "barthag",
            "efg_pct", "tov_pct", "orb_pct", "drb_pct", "ftr", "ft_pct",
            "three_pct", "three_par", "opp_efg_pct", "opp_tov_pct", "opp_ftr",
            "efg_vs_opp", "tov_vs_opp", "orb_vs_opp", "ftr_vs_opp",
            "net_rtg_l5", "net_rtg_l10", "ortg_l5", "ortg_l10",
            "drtg_l5", "drtg_l10", "pace_l5", "pace_l10",
            "net_rtg_std_l10", "efg_std_l10", "consistency_score",
            "suffocation", "momentum", "clutch_rating",
            "floor_em", "ceiling_em", "dna_score", "star_risk",
            "regression_risk", "resume_score", "cage_power_index",
            "luck", "pythagorean_win_pct", "actual_win_pct",
            "home_wpct", "away_wpct", "close_wpct", "win_streak",
            "sos", "opp_avg_net_rtg", "wab",
            "opp_avg_ortg", "opp_avg_drtg", "opp_orb_pct",
            "rest_days", "games_l7",
        ]
        for f in required:
            assert f in TeamProfile.__dataclass_fields__, f"Missing field: {f}"

    def test_construct_from_dict(self):
        """Backtester constructs TeamProfile from dict via __dataclass_fields__."""
        state = {"team_id": "99", "cage_em": 12.5, "unknown_field": "ignore"}
        tp = TeamProfile(**{
            k: state[k]
            for k in TeamProfile.__dataclass_fields__
            if k in state
        })
        assert tp.team_id == "99"
        assert tp.cage_em == 12.5


# ── Individual Model Tests ────────────────────────────────────────────────────

class TestSubModels:
    """Each model should produce valid spread, total, confidence."""

    MODELS = [
        FourFactorsModel,
        AdjustedEfficiencyModel,
        PythagoreanModel,
        MomentumModel,
        SituationalModel,
        CAGERankingsModel,
        RegressedEfficiencyModel,
    ]

    @pytest.mark.parametrize("ModelClass", MODELS)
    def test_model_returns_model_prediction(self, ModelClass, strong_home, weak_away):
        model = ModelClass()
        pred = model.predict(strong_home, weak_away, neutral=False)
        assert isinstance(pred, ModelPrediction)
        assert isinstance(pred.spread, float)
        assert isinstance(pred.total, float)
        assert isinstance(pred.confidence, float)

    @pytest.mark.parametrize("ModelClass", MODELS)
    def test_confidence_bounds(self, ModelClass, strong_home, weak_away):
        model = ModelClass()
        pred = model.predict(strong_home, weak_away, neutral=False)
        assert 0.0 <= pred.confidence <= 1.0

    @pytest.mark.parametrize("ModelClass", MODELS)
    def test_total_positive(self, ModelClass, strong_home, weak_away):
        model = ModelClass()
        pred = model.predict(strong_home, weak_away, neutral=False)
        assert pred.total > 0

    @pytest.mark.parametrize("ModelClass", MODELS)
    def test_strong_home_favoured(self, ModelClass, strong_home, weak_away):
        """With a clearly stronger home team, spread should be negative."""
        model = ModelClass()
        pred = model.predict(strong_home, weak_away, neutral=False)
        assert pred.spread < 0, (
            f"{ModelClass.name} should favour strong home "
            f"(spread={pred.spread})"
        )

    @pytest.mark.parametrize("ModelClass", MODELS)
    def test_model_name_matches(self, ModelClass):
        model = ModelClass()
        pred = model.predict(TeamProfile(games_before=10),
                             TeamProfile(games_before=10))
        assert pred.model_name == model.name

    @pytest.mark.parametrize("ModelClass", MODELS)
    def test_neutral_site_reduces_home_advantage(
        self, ModelClass, strong_home, weak_away
    ):
        model = ModelClass()
        pred_home = model.predict(strong_home, weak_away, neutral=False)
        pred_neutral = model.predict(strong_home, weak_away, neutral=True)
        # On neutral site, home advantage removed → spread less negative
        assert pred_neutral.spread >= pred_home.spread - 0.01


# ── Ensemble Predictor Tests ─────────────────────────────────────────────────

class TestEnsemblePredictor:
    def test_basic_prediction(self, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away, neutral=False)
        assert isinstance(result, EnsembleResult)
        assert result.spread < 0  # home favoured
        assert result.total > 0
        assert 0.0 <= result.confidence <= 1.0
        assert result.model_agreement in ("STRONG", "MODERATE", "SPLIT")

    def test_seven_model_predictions(self, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        assert len(result.model_predictions) == 7

    def test_model_diversity(self, strong_home, weak_away):
        """Models should NOT all produce identical spreads."""
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        spreads = [mp.spread for mp in result.model_predictions]
        assert len(set(spreads)) > 1, "All models produced identical spread"

    def test_strong_agreement_label(self, strong_home, weak_away):
        """Clear mismatch should produce STRONG agreement."""
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        assert result.model_agreement == "STRONG"

    def test_spread_std_positive(self, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        assert result.spread_std >= 0.0

    def test_cage_edge_computed(self, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        expected = strong_home.cage_em - weak_away.cage_em
        assert abs(result.cage_edge - expected) < 0.01

    def test_barthag_diff_computed(self, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        expected = strong_home.barthag - weak_away.barthag
        assert abs(result.barthag_diff - expected) < 0.001

    def test_edge_flag_spread(self, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(
            strong_home, weak_away, spread_line=0.0
        )
        # With strong home, spread should be far from 0 → edge flag
        assert result.edge_flag_spread

    def test_edge_flag_not_set_when_no_line(self, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        assert result.edge_flag_spread is False
        assert result.edge_flag_total is False

    def test_custom_weights(self, strong_home, weak_away):
        """Custom weights should produce different output than defaults."""
        default_pred = EnsemblePredictor().predict(strong_home, weak_away)
        custom_config = EnsembleConfig(
            spread_weights={
                "FourFactors": 0.50,
                "AdjEfficiency": 0.10,
                "Pythagorean": 0.10,
                "Momentum": 0.10,
                "Situational": 0.05,
                "CAGERankings": 0.10,
                "RegressedEff": 0.05,
            }
        )
        custom_pred = EnsemblePredictor(custom_config).predict(
            strong_home, weak_away
        )
        # Should be different (FourFactors heavily weighted now)
        assert default_pred.spread != custom_pred.spread

    def test_even_teams_neutral_near_zero(self, even_teams):
        """Identical teams on neutral site should have spread near zero."""
        home, away = even_teams
        predictor = EnsemblePredictor()
        result = predictor.predict(home, away, neutral=True)
        assert abs(result.spread) < 2.0, (
            f"Even teams on neutral should be near 0, got {result.spread}"
        )


# ── EnsembleResult Tests ─────────────────────────────────────────────────────

class TestEnsembleResult:
    def test_to_flat_dict_keys(self, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        flat = result.to_flat_dict()

        assert "ens_spread" in flat
        assert "ens_total" in flat
        assert "ens_confidence" in flat
        assert "ens_agreement" in flat
        assert "fourfactors_spread" in flat
        assert "adjefficiency_spread" in flat
        assert "pythagorean_spread" in flat
        assert "momentum_spread" in flat
        assert "situational_spread" in flat
        assert "cagerankings_spread" in flat
        assert "regressedeff_spread" in flat

    def test_to_flat_dict_model_totals(self, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        flat = result.to_flat_dict()

        for name in ["fourfactors", "adjefficiency", "pythagorean",
                      "momentum", "situational", "cagerankings", "regressedeff"]:
            assert f"{name}_total" in flat
            assert f"{name}_conf" in flat


# ── EnsembleConfig Tests ─────────────────────────────────────────────────────

class TestEnsembleConfig:
    def test_default_weights(self):
        config = EnsembleConfig()
        assert len(config.spread_weights) == 7
        assert abs(sum(config.spread_weights.values()) - 1.0) < 0.01

    def test_from_optimized_no_file(self, tmp_path, monkeypatch):
        """from_optimized() should return defaults when no file exists."""
        monkeypatch.setattr(
            "cbb_ensemble.WEIGHTS_PATH", tmp_path / "nonexistent.json"
        )
        config = EnsembleConfig.from_optimized()
        assert len(config.spread_weights) == 7

    def test_from_optimized_with_file(self, tmp_path, monkeypatch):
        """from_optimized() should load weights from JSON."""
        weights_file = tmp_path / "weights.json"
        weights_file.write_text(json.dumps({
            "weights": {"FourFactors": 0.30, "AdjEfficiency": 0.25}
        }))
        monkeypatch.setattr("cbb_ensemble.WEIGHTS_PATH", weights_file)
        config = EnsembleConfig.from_optimized()
        assert config.spread_weights["FourFactors"] == 0.30
        assert config.spread_weights["AdjEfficiency"] == 0.25
        # Others should keep defaults
        assert config.spread_weights["Pythagorean"] == 0.14


# ── Integration Tests ─────────────────────────────────────────────────────────

class TestIntegration:
    def test_results_to_csv(self, tmp_path, strong_home, weak_away):
        predictor = EnsemblePredictor()
        result = predictor.predict(strong_home, weak_away)
        out_path = tmp_path / "test_output.csv"
        results_to_csv(
            [result], out_path,
            game_metadata=[{"game_id": "12345", "home_team": "Duke"}],
        )
        assert out_path.exists()
        import pandas as pd
        df = pd.read_csv(out_path)
        assert len(df) == 1
        assert "ens_spread" in df.columns
        assert "game_id" in df.columns
