#!/usr/bin/env python3
"""
cbb_ensemble.py — 7-Model Ensemble Predictor for College Basketball

Implements seven analytically-distinct sub-models that each predict a
spread and total for a CBB matchup.  ``EnsemblePredictor`` aggregates
them via configurable weighted averaging to produce a consensus line.

Model Descriptions
──────────────────────────────────────────────────────────────────────
  M1  FourFactors       Dean Oliver's four factors (eFG%, TOV%, ORB%, FTR)
  M2  AdjEfficiency     Adjusted offensive / defensive efficiency edge
  M3  Pythagorean       Pythagorean expected win % → implied margin
  M4  Momentum          Recency-weighted L5 trend vs season baseline
  M5  Situational       Rest, home/away splits, scheduling fatigue
  M6  CAGERankings      Composite CAGE power-index rating system
  M7  RegressedEff      Mean-regressed efficiency toward league average

Pipeline Integration
──────────────────────────────────────────────────────────────────────
  * ``EnsemblePredictor.predict()`` → ``EnsembleResult``
  * ``load_team_profiles()`` reads ``team_pretournament_snapshot.csv``
    or ``team_game_weighted.csv`` to build ``TeamProfile`` objects.
  * ``results_to_csv()`` serialises ensemble rows to CSV.
  * ``EnsembleConfig.from_optimized()`` loads
    ``data/backtest_optimized_weights.json`` when available.

Consumed by:
  cbb_backtester.py   (imports models + TeamProfile for historical replay)
  cbb_predictions_rolling.yml  (GitHub Actions workflow inline Python)
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from pipeline_csv_utils import normalize_numeric_dtypes

from models.alpha_evaluator import evaluate_alpha as detect_alpha

try:
    from espn_config import get_game_tier
except ImportError:
    def get_game_tier(home_conf: str, away_conf: str) -> str:
        return "unknown"

from cbb_config import (
    LEAGUE_AVG_ORTG,
    LEAGUE_AVG_DRTG,
    LEAGUE_AVG_PACE,
    LEAGUE_AVG_EFG,
    LEAGUE_AVG_TOV,
    LEAGUE_AVG_FTR,
    LEAGUE_AVG_ORB,
    LEAGUE_AVG_DRB,
    HCA,
    PYTH_EXP,
    SIGMA,
    DEFAULT_SPREAD_WEIGHTS,
    DEFAULT_TOTAL_WEIGHTS,
    WEIGHTS_PATH,
)

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
STACKING_PATH = DATA_DIR / "stacking_coefficients.json"

DEFAULT_WEIGHTS = {
    "w_schedule":      0.25,
    "w_four_factors":  0.20,
    "w_bidirectional": 0.25,
    "w_ats":           0.15,
    "w_situational":   0.10,
}


def load_model_weights(weights_path: Path = Path("data/model_weights.json")) -> Dict[str, float]:
    log.info("Attempting to load model weights from %s", weights_path)
    if weights_path.exists():
        try:
            with open(weights_path) as f:
                loaded = json.load(f)
            total = sum(loaded.values())
            if abs(total - 1.0) > 0.02:
                raise ValueError(f"Weights sum to {total:.3f}, not 1.0")
            return loaded
        except Exception as e:
            log.warning("Could not load model_weights.json (%s) — using defaults", e)
    return DEFAULT_WEIGHTS.copy()

def load_stacking_params(stacking_path: Path = STACKING_PATH) -> Optional[Dict[str, object]]:
    """Load ridge stacking coefficients when available and well-formed."""
    if not stacking_path.exists() or stacking_path.stat().st_size <= 10:
        return None
    try:
        payload = json.loads(stacking_path.read_text())
    except (OSError, json.JSONDecodeError):
        log.warning("Failed to parse stacking coefficients at %s", stacking_path, exc_info=True)
        return None

    required = {"coef", "intercept", "features"}
    if not required.issubset(payload):
        log.warning("Ignoring stacking coefficients missing required keys: %s", required - set(payload))
        return None

    if not isinstance(payload.get("features"), list) or not payload["features"]:
        log.warning("Ignoring stacking coefficients with empty feature list")
        return None

    return payload



# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TeamProfile:
    """Pre-game team state consumed by all sub-models.

    Fields mirror ``build_team_state_before()`` in ``cbb_backtester.py``
    so that a dict produced there can be unpacked directly via
    ``TeamProfile(**{k: state[k] for k in TeamProfile.__dataclass_fields__
    if k in state})``.
    """

    # Identity
    team_id:          str   = ""
    team_name:        str   = ""
    conference:       str   = ""
    games_before:     int   = 0

    # CAGE adjusted efficiency
    cage_em:          float = 0.0
    cage_o:           float = LEAGUE_AVG_ORTG
    cage_d:           float = LEAGUE_AVG_DRTG
    cage_t:           float = LEAGUE_AVG_PACE
    barthag:          float = 0.500

    # Four factors (season)
    efg_pct:          float = LEAGUE_AVG_EFG
    tov_pct:          float = LEAGUE_AVG_TOV
    orb_pct:          float = 30.0
    drb_pct:          float = 70.0
    ftr:              float = LEAGUE_AVG_FTR
    ft_pct:           float = 71.0
    three_pct:        float = 33.5
    three_par:        float = 35.0
    opp_efg_pct:      float = LEAGUE_AVG_EFG
    opp_tov_pct:      float = LEAGUE_AVG_TOV
    opp_ftr:          float = LEAGUE_AVG_FTR

    # Opponent context
    efg_vs_opp:       float = 0.0
    tov_vs_opp:       float = 0.0
    orb_vs_opp:       float = 0.0
    ftr_vs_opp:       float = 0.0

    # Rolling windows (L5 / L10)
    net_rtg_l5:       float = 0.0
    net_rtg_l10:      float = 0.0
    ortg_l5:          float = LEAGUE_AVG_ORTG
    ortg_l10:         float = LEAGUE_AVG_ORTG
    drtg_l5:          float = LEAGUE_AVG_DRTG
    drtg_l10:         float = LEAGUE_AVG_DRTG
    pace_l5:          float = LEAGUE_AVG_PACE
    pace_l10:         float = LEAGUE_AVG_PACE
    efg_l5:           float = LEAGUE_AVG_EFG
    efg_l10:          float = LEAGUE_AVG_EFG
    tov_l5:           float = LEAGUE_AVG_TOV
    tov_l10:          float = LEAGUE_AVG_TOV
    three_pct_l5:     float = 33.5
    three_pct_l10:    float = 33.5

    # Variance
    net_rtg_std_l10:  float = 8.0
    efg_std_l10:      float = 5.0
    consistency_score: float = 50.0

    # CAGE composites
    suffocation:      float = 50.0
    momentum:         float = 50.0
    clutch_rating:    float = 50.0
    floor_em:         float = -8.0
    ceiling_em:       float = 8.0
    dna_score:        float = 50.0
    star_risk:        float = 50.0
    regression_risk:  int   = 0
    resume_score:     float = 50.0
    cage_power_index: float = 50.0

    # Luck & record
    luck:                float = 0.0
    pythagorean_win_pct: float = 0.5
    actual_win_pct:      float = 0.5
    home_wpct:           float = 0.65
    away_wpct:           float = 0.40
    close_wpct:          float = 0.50
    win_streak:          float = 0.0
    sos:                 float = 0.0
    opp_avg_net_rtg:     float = 0.0

    # Opponent context for specific models
    wab:              float = 0.0
    opp_avg_ortg:     float = LEAGUE_AVG_ORTG
    opp_avg_drtg:     float = LEAGUE_AVG_DRTG
    opp_orb_pct:      float = 30.0

    # Situational (injected at predict-time)
    rest_days:        float = 3.0
    games_l7:         float = 2.0
    fatigue_index:    float = 0.0

    # ATS / bias context
    cover_rate_season: float = 0.5
    cover_rate_l10:    float = 0.5
    ats_margin_l10:    float = 0.0
    cover_margin:      float = 0.0
    cover_streak:      float = 0.0
    momentum_tier:     str   = ""
    ha_net_rtg_l10:    float = 0.0


@dataclass
class ModelPrediction:
    """Output of a single sub-model."""
    model_name: str
    spread:     float       # negative = home favored (market convention)
    total:      float
    confidence: float       # 0–1


@dataclass
class EnsembleResult:
    """Aggregated ensemble output."""
    spread:            float = 0.0
    total:             float = 0.0
    confidence:        float = 0.0
    model_agreement:   str   = ""      # "STRONG", "MODERATE", "SPLIT"
    spread_std:        float = 0.0
    cage_edge:         float = 0.0
    barthag_diff:      float = 0.0
    model_predictions: List[ModelPrediction] = field(default_factory=list)

    # Edge flags
    edge_flag_spread:  bool  = False
    edge_flag_total:   bool  = False
    spread_edge_pts:   float = 0.0
    total_edge_pts:    float = 0.0
    bias_corrections_applied: List[str] = field(default_factory=list)
    pre_correction_prediction: float = 0.0
    stacked_spread: float = 0.0
    stacked_spread_weighted: float = 0.0

    def to_flat_dict(self) -> Dict:
        """Flatten for CSV output — one row per game."""
        d = {
            "ens_spread":          round(self.spread, 2),
            "ens_total":           round(self.total, 1),
            "ens_confidence":      round(self.confidence, 3),
            "ens_agreement":       self.model_agreement,
            "ens_spread_std":      round(self.spread_std, 2),
            "cage_edge":           round(self.cage_edge, 2),
            "barthag_diff":        round(self.barthag_diff, 3),
            "edge_flag_spread":    int(self.edge_flag_spread),
            "edge_flag_total":     int(self.edge_flag_total),
            "spread_edge_pts":     round(self.spread_edge_pts, 2),
            "total_edge_pts":      round(self.total_edge_pts, 2),
            "pre_correction_prediction": round(self.pre_correction_prediction, 2),
            "stacked_spread": round(self.stacked_spread, 2),
            "stacked_spread_weighted": round(self.stacked_spread_weighted, 2),
            "ensemble_spread": round(self.stacked_spread_weighted, 2),
            "bias_corrections_applied": "|".join(self.bias_corrections_applied),
        }
        for mp in self.model_predictions:
            name = mp.model_name.lower()
            d[f"{name}_spread"] = round(mp.spread, 2)
            d[f"{name}_total"]  = round(mp.total, 1)
            d[f"{name}_conf"]   = round(mp.confidence, 3)

        # Compatibility columns consumed by optimize_weights.py
        # (historical M1–M5 naming contract used in graded predictions)
        model_spreads = [round(mp.spread, 2) for mp in self.model_predictions]
        d["model1_schedule_pred"] = model_spreads[0] if len(model_spreads) > 0 else np.nan
        d["model2_four_factors_pred"] = model_spreads[1] if len(model_spreads) > 1 else np.nan
        d["model3_bidirectional_pred"] = model_spreads[2] if len(model_spreads) > 2 else np.nan
        d["model4_ats_pred"] = model_spreads[3] if len(model_spreads) > 3 else np.nan
        d["model5_situational_pred"] = model_spreads[4] if len(model_spreads) > 4 else np.nan
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnsembleConfig:
    """Weights and thresholds for ensemble aggregation."""

    spread_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SPREAD_WEIGHTS)
    )
    total_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TOTAL_WEIGHTS)
    )
    edge_threshold_spread: float = 3.0     # pts from market to flag edge
    edge_threshold_total:  float = 4.0
    agreement_strong:      float = 0.70    # fraction of models on same side
    agreement_moderate:    float = 0.55
    use_stacking:          bool = False

    @classmethod
    def from_optimized(cls) -> "EnsembleConfig":
        """Load backtest-optimized weights if available, else defaults."""
        config = cls()
        if WEIGHTS_PATH.exists() and WEIGHTS_PATH.stat().st_size > 10:
            try:
                payload = json.loads(WEIGHTS_PATH.read_text())
                if isinstance(payload.get("weights"), dict):
                    config.spread_weights.update(payload["weights"])
                if isinstance(payload.get("total_weights"), dict):
                    config.total_weights.update(payload["total_weights"])
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        stacking = load_stacking_params()
        if stacking and bool(stacking.get("use_stacking_recommended", False)):
            config.use_stacking = True
        return config


# ═══════════════════════════════════════════════════════════════════════════════
# SUB-MODEL BASE
# ═══════════════════════════════════════════════════════════════════════════════

class _BaseModel:
    """Interface for all sub-models."""

    name: str = "Base"

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        raise NotImplementedError

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _hca(neutral: bool) -> float:
        return 0.0 if neutral else HCA

    @staticmethod
    def _expected_pace(home: TeamProfile, away: TeamProfile) -> float:
        return (home.cage_t + away.cage_t) / 2.0

    @staticmethod
    def _eff_to_total(
        home_off: float, away_off: float, pace: float
    ) -> float:
        return (home_off + away_off) * pace / 100.0

    @staticmethod
    def _off_vs_def_total(
        home_off: float,
        away_off: float,
        home_def: float,
        away_def: float,
        pace: float,
    ) -> float:
        """Estimate total using each offense blended against opponent defense."""
        home_pp100 = (home_off + away_def) / 2.0
        away_pp100 = (away_off + home_def) / 2.0
        return (home_pp100 + away_pp100) * pace / 100.0

    @staticmethod
    def _confidence_from_games(
        home: TeamProfile, away: TeamProfile
    ) -> float:
        """Sample-size based confidence (0–1)."""
        min_full = 8
        h = min(home.games_before / min_full, 1.0)
        a = min(away.games_before / min_full, 1.0)
        return (h + a) / 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# M1 — FOUR FACTORS MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class FourFactorsModel(_BaseModel):
    """
    Dean Oliver's Four Factors applied as opponent-adjusted deltas.
    Uses vs-opponent season metrics to isolate performance above
    expectation.  Weights: eFG% (40%), TOV% (25%), ORB% (20%), FTR (15%)
    """

    name = "FourFactors"

    W_EFG = 0.40
    W_TOV = 0.25
    W_ORB = 0.20
    W_FTR = 0.15

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        use_vs_opp = any(
            abs(v) > 1e-6
            for v in (
                home.efg_vs_opp,
                away.efg_vs_opp,
                home.tov_vs_opp,
                away.tov_vs_opp,
                home.orb_vs_opp,
                away.orb_vs_opp,
                home.ftr_vs_opp,
                away.ftr_vs_opp,
            )
        )

        def _to_unit_rate(v: float) -> float:
            v = float(v)
            return v / 100.0 if abs(v) > 1.5 else v

        if use_vs_opp:
            efg_delta = _to_unit_rate(home.efg_vs_opp - away.efg_vs_opp)
            tov_delta = -_to_unit_rate(home.tov_vs_opp - away.tov_vs_opp)
            orb_delta = _to_unit_rate(home.orb_vs_opp - away.orb_vs_opp)
            ftr_delta = _to_unit_rate(home.ftr_vs_opp - away.ftr_vs_opp)
        else:
            # Fallback to raw Four Factors vs each opponent profile if
            # vs-opponent columns are missing in source data.
            efg_delta = _to_unit_rate(
                (home.efg_pct - away.opp_efg_pct)
                - (away.efg_pct - home.opp_efg_pct)
            )
            tov_delta = -_to_unit_rate(
                (home.tov_pct - away.opp_tov_pct)
                - (away.tov_pct - home.opp_tov_pct)
            )
            orb_delta = _to_unit_rate(home.orb_pct - away.orb_pct)
            ftr_delta = _to_unit_rate(
                (home.ftr - away.opp_ftr) - (away.ftr - home.opp_ftr)
            )
            # Inject a modest efficiency anchor when four-factor deltas are sparse.
            efg_delta += (home.cage_em - away.cage_em) * 0.05

        composite = (
            self.W_EFG * efg_delta
            + self.W_TOV * tov_delta
            + self.W_ORB * orb_delta
            + self.W_FTR * ftr_delta
        )

        margin = composite * 0.8 + self._hca(neutral)
        spread = -margin

        pace  = self._expected_pace(home, away)
        total = self._off_vs_def_total(
            home_off=home.cage_o,
            away_off=away.cage_o,
            home_def=home.cage_d,
            away_def=away.cage_d,
            pace=pace,
        )

        sample_conf = self._confidence_from_games(home, away)
        consistency = 1.0 - (
            abs(home.efg_vs_opp) + abs(away.efg_vs_opp)
        ) / 20.0
        conf = max(
            0.10, min(0.95, 0.6 * sample_conf + 0.4 * max(0.0, consistency))
        )

        return ModelPrediction(
            self.name, round(spread, 2), round(total, 1), round(conf, 3)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M2 — ADJUSTED EFFICIENCY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class AdjustedEfficiencyModel(_BaseModel):
    """
    Adjusted net efficiency differential (CAGE EM) — the most
    signal-dense single predictor in college basketball.
    """

    name = "AdjEfficiency"

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        eff_edge = home.cage_em - away.cage_em
        pace = self._expected_pace(home, away)
        margin = eff_edge * (pace / 100.0) + self._hca(neutral)
        spread = -margin

        base_total = self._eff_to_total(home.cage_o, away.cage_o, pace)
        luck_drag = (abs(home.luck) + abs(away.luck)) * 0.15
        total = base_total - luck_drag

        sample_conf = self._confidence_from_games(home, away)
        variance_penalty = min(
            (home.net_rtg_std_l10 + away.net_rtg_std_l10) / 40.0, 0.3
        )
        conf = max(0.10, min(0.95, sample_conf - variance_penalty))

        return ModelPrediction(
            self.name, round(spread, 2), round(total, 1), round(conf, 3)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M3 — PYTHAGOREAN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class PythagoreanModel(_BaseModel):
    """
    Converts Pythagorean expected win % into an implied margin using
    the normal CDF (inverse).
    """

    name = "Pythagorean"

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        h_wp = np.clip(home.pythagorean_win_pct, 0.01, 0.99)
        a_wp = np.clip(away.pythagorean_win_pct, 0.01, 0.99)

        h_implied = scipy_stats.norm.ppf(h_wp) * SIGMA
        a_implied = scipy_stats.norm.ppf(a_wp) * SIGMA

        margin = (h_implied - a_implied) + self._hca(neutral)
        spread = -margin

        pace  = self._expected_pace(home, away)
        style_adj = (home.cover_margin + away.cover_margin) * 0.15
        total = self._eff_to_total(home.cage_o, away.cage_o, pace) + style_adj

        luck_penalty = (abs(home.luck) + abs(away.luck)) / 20.0
        sample_conf = self._confidence_from_games(home, away)
        conf = max(0.10, min(0.95, sample_conf - luck_penalty))

        return ModelPrediction(
            self.name, round(spread, 2), round(total, 1), round(conf, 3)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M4 — MOMENTUM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumModel(_BaseModel):
    """
    Recency bias model: weights L5 rolling performance more than
    season averages.  Captures streaky play, mid-season breakouts,
    and injuries that changed the rotation.
    """

    name = "Momentum"

    L5_WEIGHT  = 0.65
    L10_WEIGHT = 0.35

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        h_net = (
            self.L5_WEIGHT * home.net_rtg_l5
            + self.L10_WEIGHT * home.net_rtg_l10
        )
        a_net = (
            self.L5_WEIGHT * away.net_rtg_l5
            + self.L10_WEIGHT * away.net_rtg_l10
        )

        h_trend = home.net_rtg_l5 - home.net_rtg_l10
        a_trend = away.net_rtg_l5 - away.net_rtg_l10
        trend_edge = (h_trend - a_trend) * 0.25

        pace = self._expected_pace(home, away)
        raw_margin = (h_net - a_net) * (pace / 100.0) + trend_edge
        margin = raw_margin + self._hca(neutral)
        spread = -margin

        h_off = (
            self.L5_WEIGHT * home.ortg_l5 + self.L10_WEIGHT * home.ortg_l10
        )
        a_off = (
            self.L5_WEIGHT * away.ortg_l5 + self.L10_WEIGHT * away.ortg_l10
        )
        total = self._eff_to_total(h_off, a_off, pace)

        h_var = abs(home.net_rtg_l5 - home.net_rtg_l10)
        a_var = abs(away.net_rtg_l5 - away.net_rtg_l10)
        variance_penalty = (h_var + a_var) / 40.0
        sample_conf = self._confidence_from_games(home, away)
        conf = max(0.10, min(0.95, sample_conf - variance_penalty))

        return ModelPrediction(
            self.name, round(spread, 2), round(total, 1), round(conf, 3)
        )


class ATSIntelligenceModel(_BaseModel):
    """ATS behavior model using cover rates, margin and streak context."""

    name = "ATSIntelligence"

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        cover_a = float(home.cover_rate_season or home.cover_rate_l10 or 0.5)
        cover_b = float(away.cover_rate_season or away.cover_rate_l10 or 0.5)
        ats_a = float(home.ats_margin_l10 or home.cover_margin or 0.0)
        ats_b = float(away.ats_margin_l10 or away.cover_margin or 0.0)
        streak_a = float(home.cover_streak or 0.0)
        streak_b = float(away.cover_streak or 0.0)

        margin = (
            (cover_a - cover_b) * 3.0
            + (ats_a - ats_b) * 0.4
            + (streak_a - streak_b) * 0.15
            + self._hca(neutral)
        )
        spread = -margin
        pace = self._expected_pace(home, away)
        pace_adj = (home.momentum + away.momentum - 100.0) * 0.03
        total = self._eff_to_total(home.cage_o, away.cage_o, pace) + pace_adj
        conf = max(0.10, min(0.90, self._confidence_from_games(home, away)))
        return ModelPrediction(self.name, round(spread, 2), round(total, 1), round(conf, 3))


# ═══════════════════════════════════════════════════════════════════════════════
# M5 — SITUATIONAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SituationalModel(_BaseModel):
    """
    Captures non-efficiency signals: rest days, home/away splits,
    close-game performance, and win streaks.
    """

    name = "Situational"

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        eff_edge = (home.cage_em - away.cage_em) * 0.5
        rest_edge = self._rest_adjustment(home.rest_days, away.rest_days)
        fatigue_edge = (away.fatigue_index - home.fatigue_index) * 2.0

        if neutral:
            split_edge = 0.0
        else:
            h_strength = home.home_wpct - 0.50
            a_strength = away.away_wpct - 0.50
            split_edge = (h_strength - a_strength) * 4.0

        close_edge = (home.close_wpct - away.close_wpct) * 2.0
        streak_edge = np.clip(
            home.win_streak - away.win_streak, -5, 5
        ) * 0.3

        pace = self._expected_pace(home, away)
        margin = eff_edge + rest_edge + fatigue_edge + split_edge + close_edge + streak_edge
        if not neutral:
            margin += HCA * 0.5
        spread = -margin

        total = self._off_vs_def_total(
            home_off=home.cage_o,
            away_off=away.cage_o,
            home_def=home.cage_d,
            away_def=away.cage_d,
            pace=pace,
        )

        sample_conf = self._confidence_from_games(home, away)
        conf = max(0.10, min(0.90, sample_conf * 0.8))

        return ModelPrediction(
            self.name, round(spread, 2), round(total, 1), round(conf, 3)
        )

    @staticmethod
    def _rest_adjustment(home_rest: float, away_rest: float) -> float:
        """Rest-day edge in points."""
        def _rest_value(days: float) -> float:
            if days >= 3:
                return 0.0
            if days >= 2:
                return -0.5
            if days >= 1:
                return -1.5
            return -3.0
        return _rest_value(home_rest) - _rest_value(away_rest)


# ═══════════════════════════════════════════════════════════════════════════════
# M6 — CAGE RANKINGS MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class CAGERankingsModel(_BaseModel):
    """
    Uses the full CAGE composite rating system: power index,
    suffocation, clutch, resume, and tournament DNA.
    """

    name = "CAGERankings"

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        power_edge  = (home.cage_power_index - away.cage_power_index) / 10.0
        suff_edge   = (home.suffocation - away.suffocation) / 50.0
        clutch_edge = (home.clutch_rating - away.clutch_rating) / 50.0
        resume_edge = (home.resume_score - away.resume_score) / 50.0
        dna_edge    = (home.dna_score - away.dna_score) / 50.0

        base_margin = (
            0.45 * power_edge
            + 0.15 * suff_edge
            + 0.15 * clutch_edge
            + 0.15 * resume_edge
            + 0.10 * dna_edge
        )

        if all(
            abs(v - 50.0) < 1e-6
            for v in (
                home.cage_power_index, away.cage_power_index,
                home.suffocation, away.suffocation,
                home.clutch_rating, away.clutch_rating,
                home.resume_score, away.resume_score,
                home.dna_score, away.dna_score,
            )
        ):
            base_margin = (home.cage_em - away.cage_em) * 0.45

        margin = base_margin + self._hca(neutral)
        spread = -margin

        pace = self._expected_pace(home, away)
        h_form_off = home.ortg_l10 if abs(home.ortg_l10) > 1e-6 else home.cage_o
        a_form_off = away.ortg_l10 if abs(away.ortg_l10) > 1e-6 else away.cage_o
        total = self._eff_to_total(h_form_off, a_form_off, pace)

        sample_conf = self._confidence_from_games(home, away)
        risk_penalty = (home.star_risk + away.star_risk - 100.0) / 200.0
        conf = max(0.10, min(0.95, sample_conf - max(0.0, risk_penalty)))

        return ModelPrediction(
            self.name, round(spread, 2), round(total, 1), round(conf, 3)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M7 — REGRESSED EFFICIENCY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RegressedEfficiencyModel(_BaseModel):
    """
    Mean-regresses adjusted efficiency toward league average based on
    sample size and consistency.  Conservative early in the season,
    converges toward AdjEfficiency as data accumulates.
    """

    name = "RegressedEff"

    FULL_SEASON_GAMES = 25.0

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        h_reg = self._regress(home)
        a_reg = self._regress(away)

        pace = self._expected_pace(home, away)
        margin = (h_reg - a_reg) * (pace / 100.0) + self._hca(neutral)
        spread = -margin

        h_ortg_reg = LEAGUE_AVG_ORTG + (
            (home.cage_o - LEAGUE_AVG_ORTG) * self._reg_factor(home)
        )
        a_ortg_reg = LEAGUE_AVG_ORTG + (
            (away.cage_o - LEAGUE_AVG_ORTG) * self._reg_factor(away)
        )
        total = self._eff_to_total(h_ortg_reg, a_ortg_reg, pace)

        sample_conf = self._confidence_from_games(home, away)
        avg_reg = (self._reg_factor(home) + self._reg_factor(away)) / 2.0
        conf = max(0.10, min(0.90, sample_conf * avg_reg))

        return ModelPrediction(
            self.name, round(spread, 2), round(total, 1), round(conf, 3)
        )

    def _reg_factor(self, team: TeamProfile) -> float:
        """0–1 regression factor: higher = less regression."""
        sample = min(team.games_before / self.FULL_SEASON_GAMES, 1.0)
        consist = team.consistency_score / 100.0
        return sample * np.clip(consist, 0.3, 1.0)

    def _regress(self, team: TeamProfile) -> float:
        """Regress CAGE EM toward zero (league average)."""
        return team.cage_em * self._reg_factor(team)


class LuckRegressionModel(_BaseModel):
    """Regression-to-mean model using Pythagorean win% and luck."""

    name = "LuckRegression"

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        pace = self._expected_pace(home, away)
        margin = (
            (home.pythagorean_win_pct - away.pythagorean_win_pct) * 25.0
            - (home.luck - away.luck) * 0.3
            + self._hca(neutral)
        )
        spread = -margin
        luck_drag = (abs(home.luck) + abs(away.luck)) * 0.2
        total = self._eff_to_total(home.cage_o, away.cage_o, pace) - luck_drag
        conf = max(0.10, min(0.90, self._confidence_from_games(home, away)))
        return ModelPrediction(self.name, round(spread, 2), round(total, 1), round(conf, 3))


class VarianceModel(_BaseModel):
    """Volatility-aware model that regresses efficiency edges for high-variance teams."""

    name = "Variance"

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        pace = self._expected_pace(home, away)
        h_std = max(home.net_rtg_std_l10, 0.0)
        a_std = max(away.net_rtg_std_l10, 0.0)
        eff_edge = home.cage_em - away.cage_em
        confidence_adj = float(np.clip(1.0 - (h_std + a_std) / 40.0, 0.35, 1.0))
        margin = eff_edge * confidence_adj * (pace / 100.0) + self._hca(neutral)
        spread = -margin
        volatility_drag = (h_std + a_std) * 0.12
        total = self._eff_to_total(home.cage_o, away.cage_o, pace) - volatility_drag
        conf = max(0.10, min(0.90, self._confidence_from_games(home, away) * confidence_adj))
        return ModelPrediction(self.name, round(spread, 2), round(total, 1), round(conf, 3))


class HomeAwayFormModel(_BaseModel):
    """Location-form model based on home/away adjusted net rating."""

    name = "HomeAwayForm"

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        pace = self._expected_pace(home, away)
        h_loc = home.ha_net_rtg_l10 if abs(home.ha_net_rtg_l10) > 1e-6 else home.cage_em
        a_loc = away.ha_net_rtg_l10 if abs(away.ha_net_rtg_l10) > 1e-6 else away.cage_em
        location_edge = h_loc - a_loc
        margin = location_edge * (pace / 100.0)
        if not neutral:
            margin += HCA * 0.5
        spread = -margin
        h_form_off = home.ortg_l10 if abs(home.ortg_l10) > 1e-6 else home.cage_o
        a_form_off = away.ortg_l10 if abs(away.ortg_l10) > 1e-6 else away.cage_o
        total = self._eff_to_total(h_form_off, a_form_off, pace)
        conf = max(0.10, min(0.85, self._confidence_from_games(home, away)))
        return ModelPrediction(self.name, round(spread, 2), round(total, 1), round(conf, 3))


def _apply_bias_corrections(
    prediction: float,
    home_conf: str,
    away_conf: str,
    home_momentum_tier: str = None,
    bias_path: Path = Path("data/model_bias_table.csv"),
) -> tuple[float, list[str]]:
    """Apply actionable bias corrections with ±4.0 total cap."""
    if not bias_path.exists():
        return prediction, []

    try:
        bt = pd.read_csv(bias_path)
        actionable = bt[bt["actionable"] == True]
    except Exception:
        # Bugfix: loading/casting bias rows can fail on malformed CSV.
        # Return unmodified prediction, but emit traceback at warning level
        # so this data-quality failure is no longer silent in production logs.
        log.warning("Failed to load actionable bias corrections from %s", bias_path, exc_info=True)
        return prediction, []

    game_tier = get_game_tier(home_conf, away_conf)
    game_features = {
        "conference_tier": game_tier,
        "cross_tier_matchup": game_tier,
        "momentum_tier": home_momentum_tier,
    }

    applied = []
    total = 0.0
    for _, row in actionable.iterrows():
        dim = str(row.get("dimension", ""))
        group = str(row.get("group", ""))
        corr = float(row.get("correction", 0.0) or 0.0)
        if game_features.get(dim) == group:
            total += corr
            applied.append(f"{dim}={group}:{corr:+.2f}")

    total = float(np.clip(total, -4.0, 4.0))
    return round(prediction + total, 1), applied


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

def apply_stacking(model_spreads: Dict[str, float], aux: Dict[str, float], params: Dict[str, object]) -> float:
    """Apply ridge stacking coefficients to model spread outputs."""
    features = params.get("features", [])
    coef = np.array(params.get("coef", []), dtype=float)
    intercept = float(params.get("intercept", 0.0))

    if len(features) != len(coef):
        raise ValueError("Stacking features/coefs length mismatch")

    values = []
    for feature in features:
        if feature in model_spreads:
            values.append(float(model_spreads.get(feature, 0.0) or 0.0))
        else:
            values.append(float(aux.get(feature, 0.0) or 0.0))

    X = np.array([values], dtype=float)
    return float(X @ coef + intercept)


class EnsemblePredictor:
    """
    Aggregates the 7 sub-model predictions into a consensus line.

    Aggregation method: weighted average of spread and total,
    with separate weight vectors for each.

    Model agreement categories:
      STRONG   — ≥70% of models agree on the side (home/away)
      MODERATE — ≥55% agree
      SPLIT    — neither side reaches 55%
    """

    MODELS = [
        FourFactorsModel,
        AdjustedEfficiencyModel,
        PythagoreanModel,
        SituationalModel,
        CAGERankingsModel,
        LuckRegressionModel,
        VarianceModel,
        HomeAwayFormModel,
    ]

    def _apply_bias_corrections(
        self,
        prediction: float,
        home: TeamProfile,
        away: TeamProfile,
        bias_table_path: Path,
    ) -> Tuple[float, List[str]]:
        """Apply additive post-prediction bias corrections from bias table."""
        if not bias_table_path.exists():
            return prediction, []

        try:
            bt = pd.read_csv(bias_table_path)
            actionable = bt[
                bt["actionable"].astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})
            ]
        except Exception:
            # Bugfix: this affects final model values; keep fallback but surface traceback.
            log.warning("Unable to load bias table %s", bias_table_path, exc_info=True)
            return prediction, []

        game_tier = get_game_tier(home.conference, away.conference)
        if str(home.momentum_tier).strip():
            momentum_tier = str(home.momentum_tier).strip().upper()
        else:
            momentum_tier = pd.cut(
                pd.Series([home.momentum]),
                bins=[-999, -10, -2, 4, 10, 999],
                labels=["COLD", "NEUTRAL", "WARM", "HOT", "ELITE"],
            ).astype(str).iloc[0]

        line_for_bias = getattr(self, "_current_spread_line", None)
        if line_for_bias is None or pd.isna(line_for_bias):
            line_for_bias = prediction

        spread_abs = abs(float(line_for_bias))
        if spread_abs < 3:
            spread_bucket = "0-3"
        elif spread_abs < 6:
            spread_bucket = "3-6"
        elif spread_abs < 10:
            spread_bucket = "6-10"
        else:
            spread_bucket = "10+"

        favorite_side = "home_fav" if float(line_for_bias) < 0 else "away_fav"

        game_features = {
            "conference_tier": game_tier,
            "spread_bucket": spread_bucket,
            "game_tier": game_tier,
            "momentum_tier": momentum_tier,
            "favorite_side": favorite_side,
        }

        applied: List[str] = []
        total_correction = 0.0
        for _, row in actionable.iterrows():
            dim = row.get("dimension")
            group = row.get("group")
            corr = float(row.get("correction", 0.0))
            if dim in game_features and game_features[dim] == group:
                total_correction += corr
                applied.append(f"{dim}={group}: {corr:+.2f}pts")

        total_correction = float(np.clip(total_correction, -4.0, 4.0))
        corrected = prediction + total_correction
        if applied:
            log.debug("Bias corrections applied: %s → total %+.2f", applied, total_correction)

        return round(corrected, 1), applied

    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models = [cls() for cls in self.MODELS]
        self.model_weights = load_model_weights()
        self.stacking_params = load_stacking_params()

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
        spread_line: Optional[float] = None,
        total_line: Optional[float] = None,
    ) -> EnsembleResult:
        """Run all sub-models, aggregate, and return EnsembleResult."""

        preds: List[ModelPrediction] = []
        for model in self.models:
            try:
                mp = model.predict(home, away, neutral)
                preds.append(mp)
            except Exception:
                # Bugfix: a model failure changes ensemble composition and outputs.
                # Log at warning with traceback instead of debug-only silent degradation.
                log.warning("Model %s failed during ensemble prediction", model.name, exc_info=True)

        if not preds:
            return EnsembleResult()

        # ── Weighted average ──────────────────────────────────────────────
        sw = self.config.spread_weights
        tw = self.config.total_weights

        spread_sum, spread_w = 0.0, 0.0
        total_sum,  total_w  = 0.0, 0.0

        for mp in preds:
            ws = sw.get(mp.model_name, 1.0 / len(preds))
            wt = tw.get(mp.model_name, 1.0 / len(preds))
            spread_sum += mp.spread * ws
            spread_w   += ws
            total_sum  += mp.total * wt
            total_w    += wt

        ens_spread = spread_sum / spread_w if spread_w > 0 else 0.0
        ens_total  = total_sum  / total_w  if total_w  > 0 else 140.0
        weighted_spread = ens_spread
        model_map = {mp.model_name: mp.spread for mp in preds}

        model_spreads_for_stacking = {
            "m1_spread": float(model_map.get("FourFactors", ens_spread)),
            "m2_spread": float(model_map.get("AdjEfficiency", ens_spread)),
            "m3_spread": float(model_map.get("Pythagorean", ens_spread)),
            "m4_spread": float(model_map.get("Situational", ens_spread)),
            "m5_spread": float(model_map.get("CAGERankings", ens_spread)),
            "m6_spread": float(model_map.get("LuckRegression", ens_spread)),
            "m7_spread": float(model_map.get("HomeAwayForm", model_map.get("Variance", ens_spread))),
        }
        aux_features = {
            "cage_edge": float(home.cage_em - away.cage_em),
            "barthag_diff": float(home.barthag - away.barthag),
        }

        stacked_spread = weighted_spread
        if self.config.use_stacking and self.stacking_params:
            try:
                stacked_spread = apply_stacking(model_spreads_for_stacking, aux_features, self.stacking_params)
            except Exception:
                log.warning("Failed to apply stacking meta-model; falling back to weighted spread", exc_info=True)
                stacked_spread = weighted_spread

        ens_spread = stacked_spread

        # pre_correction: 5-model blend stored for compatibility columns
        # ens_spread: 8-model weighted average from config.spread_weights
        # (ens_spread from spread_sum/spread_w above is the authoritative output)
        m1 = model_map.get("Situational", ens_spread)
        m2 = model_map.get("FourFactors", ens_spread)
        m3 = model_map.get("AdjEfficiency", ens_spread)
        m4 = model_map.get("ATSIntelligence", ens_spread)
        m5 = model_map.get("Momentum", ens_spread)
        w = self.model_weights
        pre_correction = (
            w.get("w_schedule", DEFAULT_WEIGHTS["w_schedule"]) * m1
            + w.get("w_four_factors", DEFAULT_WEIGHTS["w_four_factors"]) * m2
            + w.get("w_bidirectional", DEFAULT_WEIGHTS["w_bidirectional"]) * m3
            + w.get("w_ats", DEFAULT_WEIGHTS["w_ats"]) * m4
            + w.get("w_situational", DEFAULT_WEIGHTS["w_situational"]) * m5
        )
        # NOTE: ens_spread retains the 8-model weighted average.
        # pre_correction is stored in EnsembleResult for diagnostic
        # use and backward-compat columns only.

        # ── Confidence & agreement ────────────────────────────────────────
        spreads = np.array([mp.spread for mp in preds])
        confs   = np.array([mp.confidence for mp in preds])
        spread_std = float(np.std(spreads)) if len(spreads) > 1 else 0.0

        home_frac = float(np.mean(spreads < 0))
        away_frac = 1.0 - home_frac
        majority = max(home_frac, away_frac)

        if majority >= self.config.agreement_strong:
            agreement = "STRONG"
        elif majority >= self.config.agreement_moderate:
            agreement = "MODERATE"
        else:
            agreement = "SPLIT"

        avg_conf = float(np.mean(confs))
        agreement_bonus = (
            0.05 if agreement == "STRONG"
            else (-0.05 if agreement == "SPLIT" else 0.0)
        )
        ensemble_conf = max(0.05, min(0.95, avg_conf + agreement_bonus))

        # ── CAGE edge & barthag diff (metadata) ──────────────────────────
        cage_edge    = home.cage_em - away.cage_em
        if cage_edge == 0.0 and home.cage_em == 0.0 and away.cage_em == 0.0:
            log.debug(
                "Both teams have cage_em=0.0 for %s vs %s — "
                "6 of 8 sub-models will return pure HCA. "
                "Check load_team_profiles() column mapping.",
                home.team_name, away.team_name,
            )
        barthag_diff = home.barthag - away.barthag

        # ── Edge flags ────────────────────────────────────────────────────
        edge_spread = False
        edge_total  = False
        spread_edge_pts = 0.0
        total_edge_pts  = 0.0

        if spread_line is not None:
            spread_edge_pts = abs(ens_spread - spread_line)
            edge_spread = (
                spread_edge_pts >= self.config.edge_threshold_spread
            )

        if total_line is not None:
            total_edge_pts = abs(ens_total - total_line)
            edge_total = (
                total_edge_pts >= self.config.edge_threshold_total
            )

        self._current_spread_line = spread_line
        corrected_spread, bias_applied = self._apply_bias_corrections(
            ens_spread,
            home,
            away,
            bias_table_path=DATA_DIR / "model_bias_table.csv",
        )
        self._current_spread_line = None

        return EnsembleResult(
            spread=round(corrected_spread, 2),
            total=round(ens_total, 1),
            confidence=round(ensemble_conf, 3),
            model_agreement=agreement,
            spread_std=round(spread_std, 2),
            cage_edge=round(cage_edge, 2),
            barthag_diff=round(barthag_diff, 3),
            model_predictions=preds,
            edge_flag_spread=edge_spread,
            edge_flag_total=edge_total,
            spread_edge_pts=round(spread_edge_pts, 2),
            total_edge_pts=round(total_edge_pts, 2),
            bias_corrections_applied=bias_applied,
            pre_correction_prediction=round(pre_correction, 2),
            stacked_spread=round(stacked_spread, 2),
            stacked_spread_weighted=round(weighted_spread, 2),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM PROFILE LOADER (for workflow integration)
# ═══════════════════════════════════════════════════════════════════════════════

def load_team_profiles(
    snapshot_path: Optional[Path] = None,
    weighted_path: Optional[Path] = None,
) -> Dict[str, TeamProfile]:
    """
    Load team profiles from CSV for ensemble predictions.

    Priority:
      1. team_pretournament_snapshot.csv (one row per team, latest game)
      2. team_game_weighted.csv (take last row per team)

    Returns dict keyed by team_id.
    """
    snapshot_path = snapshot_path or DATA_DIR / "team_pretournament_snapshot.csv"
    weighted_path = weighted_path or DATA_DIR / "team_game_weighted.csv"

    df = None
    for path in [weighted_path, snapshot_path]:
        if path.exists() and path.stat().st_size > 100:
            candidate = pd.read_csv(path, dtype=str, low_memory=False)
            candidate = normalize_numeric_dtypes(candidate)
            signal_cols = [
                "adj_net_rtg", "net_rtg", "net_rtg_l10", "cover_rate_season",
                "ha_net_rtg_l10", "pythagorean_win_pct", "luck_score",
            ]
            signal_score = int(sum(candidate.get(c, pd.Series(dtype=float)).notna().sum() for c in signal_cols if c in candidate.columns))
            if df is None or signal_score > 0:
                df = candidate
                log.info("Loading team profiles from %s (signal score=%d)", path.name, signal_score)
                if signal_score > 0:
                    break

    if df is None or df.empty:
        log.warning("No team data found for profiles")
        return {}

    # Numeric coercion
    str_cols = {
        "team_id", "team", "opponent_id", "opponent", "home_away",
        "conference", "conf_id", "event_id", "game_id",
        "game_datetime_utc", "game_datetime_pst", "venue", "state",
        "source", "parse_version", "t_offensive_archetype",
    }
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Count actual games per team from full history
    _team_game_count: dict[str, int] = {}
    df_all = df.copy()
    if "team_id" in df_all.columns:
        _counts = df_all["team_id"].astype(str).value_counts()
        _team_game_count = _counts.to_dict()
        log.info(
            "Game counts: %d teams, median %d games, max %d games",
            len(_team_game_count),
            int(_counts.median()) if len(_counts) > 0 else 0,
            int(_counts.max()) if len(_counts) > 0 else 0,
        )

    # Keep last row per team (most recent game)
    if "game_datetime_utc" in df.columns:
        df["game_datetime_utc"] = pd.to_datetime(
            df["game_datetime_utc"], utc=True, errors="coerce"
        )
        df = df.sort_values("game_datetime_utc")
    df_all = df.copy()   # preserve full history for aggregation
    df = df.drop_duplicates("team_id", keep="last")

    # Try efficiency columns in confirmed priority order.
    _team_agg: dict[str, float] = {}
    _team_o_agg: dict[str, float] = {}
    _team_d_agg: dict[str, float] = {}
    _team_t_agg: dict[str, float] = {}

    _agg_candidates = ["adj_net_rtg", "net_eff", "net_rtg", "cage_em"]
    _agg_col = next((c for c in _agg_candidates if c in df_all.columns), None)
    _o_col = next((c for c in ["adj_ortg", "ortg", "off_rtg", "off_eff", "cage_o"] if c in df_all.columns), None)
    _d_col = next((c for c in ["adj_drtg", "drtg", "def_rtg", "def_eff", "cage_d"] if c in df_all.columns), None)
    _t_col = next((c for c in ["adj_pace", "pace", "poss", "possessions", "cage_t"] if c in df_all.columns), None)

    if _agg_col and "team_id" in df_all.columns:
        _team_agg = pd.to_numeric(df_all[_agg_col], errors="coerce").groupby(df_all["team_id"].astype(str)).mean().dropna().to_dict()
    if _o_col and "team_id" in df_all.columns:
        _team_o_agg = pd.to_numeric(df_all[_o_col], errors="coerce").groupby(df_all["team_id"].astype(str)).mean().dropna().to_dict()
    if _d_col and "team_id" in df_all.columns:
        _team_d_agg = pd.to_numeric(df_all[_d_col], errors="coerce").groupby(df_all["team_id"].astype(str)).mean().dropna().to_dict()
    if _t_col and "team_id" in df_all.columns:
        _team_t_agg = pd.to_numeric(df_all[_t_col], errors="coerce").groupby(df_all["team_id"].astype(str)).mean().dropna().to_dict()

    if _agg_col:
        _non_zero = sum(1 for v in _team_agg.values() if v != 0.0)
        log.info(
            "[DIAG] load_team_profiles | Season %s aggregated: %d teams, %d non-zero (range: %.1f to %.1f)",
            _agg_col, len(_team_agg), _non_zero,
            min(_team_agg.values()) if _team_agg else 0,
            max(_team_agg.values()) if _team_agg else 0,
        )
    else:
        log.warning(
            "[DIAG] No net efficiency column found for cage_em aggregation. "
            "All cage_em will be 0.0. Columns checked: %s. Available: %s",
            _agg_candidates,
            [c for c in df_all.columns if any(x in c.lower() for x in ["net", "eff", "rtg", "em"])][:15],
        )

    profiles: Dict[str, TeamProfile] = {}

    def col(row, *names, default=0.0):
        """Try column names in priority order, return first non-null."""
        for name in names:
            v = row.get(name)
            try:
                if v is not None and not pd.isna(float(v)):
                    return float(v)
            except (TypeError, ValueError):
                continue
        return default

    def _g(row, col, default=0.0):
        v = row.get(col, default)
        try:
            return float(v) if pd.notna(v) else default
        except (TypeError, ValueError):
            return default

    for _, row in df.iterrows():
        tid = str(row.get("team_id", ""))
        if not tid:
            continue

        tp = TeamProfile(
            team_id=tid,
            team_name=str(row.get("team", "")),
            conference=str(row.get("conference", "")),
            games_before=(
                int(_g(row, "wins", 0) + _g(row, "losses", 0))
                or _team_game_count.get(tid)
                or int(_g(row, "games_played", 0))
                or int(_g(row, "game_number", 0))
                or 0
            ),

            # Use `tid in _agg` not `_agg.get(tid) or ...`
            # A legitimately zero-efficiency team would be skipped by `or`.
            cage_em=(_team_agg[tid] if tid in _team_agg else col(row, "adj_net_rtg", "net_eff", "cage_em", "adj_em", default=0.0)),
            cage_o=(_team_o_agg[tid] if tid in _team_o_agg else col(row, "adj_ortg", "ortg", "off_rtg", "off_eff", "cage_o", default=LEAGUE_AVG_ORTG)),
            cage_d=(_team_d_agg[tid] if tid in _team_d_agg else col(row, "adj_drtg", "drtg", "def_rtg", "def_eff", "cage_d", default=LEAGUE_AVG_DRTG)),
            cage_t=(_team_t_agg[tid] if tid in _team_t_agg else col(row, "adj_pace", "pace", "poss", "possessions", "cage_t", default=LEAGUE_AVG_PACE)),
            barthag=col(row, "barthag", "barthag_score", default=0.5),

            # Four factors — try both naming conventions
            efg_pct=col(row, "efg_pct", "eff_fg_pct", default=LEAGUE_AVG_EFG),
            tov_pct=col(row, "tov_pct", "to_pct", "tov_rate", default=LEAGUE_AVG_TOV),
            orb_pct=col(row, "orb_pct", "off_reb_pct", default=30.0),
            drb_pct=col(row, "drb_pct", "def_reb_pct", default=70.0),
            ftr=col(row, "ftr", "ft_rate", "free_throw_rate", default=LEAGUE_AVG_FTR),
            ft_pct=col(row, "ft_pct", "ftm_pct", default=71.0),
            three_pct=col(row, "three_pct", "fg3_pct", "tp_pct", default=33.5),
            three_par=col(row, "three_par", "fg3_rate", "tp_rate", default=35.0),

            opp_efg_pct=col(row, "opp_avg_efg_season", "opp_efg_pct",
                            "opp_eff_fg_pct", default=LEAGUE_AVG_EFG),
            opp_tov_pct=col(row, "opp_avg_tov_season", "opp_tov_pct",
                            "opp_to_pct", default=LEAGUE_AVG_TOV),
            opp_ftr=col(row, "opp_avg_ftr_season", "opp_ftr",
                        "opp_ft_rate", default=LEAGUE_AVG_FTR),

            efg_vs_opp=col(row, "efg_vs_opp_season", "efg_vs_opp",
                           "off_efg_vs_opp", default=0.0),
            tov_vs_opp=col(row, "tov_vs_opp_season", "tov_vs_opp",
                           "off_tov_vs_opp", default=0.0),
            orb_vs_opp=col(row, "orb_vs_opp_season", "orb_vs_opp",
                           "off_orb_vs_opp", default=0.0),
            ftr_vs_opp=col(row, "ftr_vs_opp_season", "ftr_vs_opp",
                           "off_ftr_vs_opp", default=0.0),

            # Rolling windows — try l5/l10 suffixed variants
            net_rtg_l5=col(row, "net_rtg_l5", "net_eff_l5", "form_rating",
                           "adj_net_rtg_l5", default=0.0),
            net_rtg_l10=col(row, "net_rtg_l10", "net_eff_l10", "momentum_score",
                            "adj_net_rtg_l10", default=0.0),
            ortg_l5=col(row, "ortg_l5", "off_rtg_l5",
                        default=LEAGUE_AVG_ORTG),
            ortg_l10=col(row, "ortg_l10", "off_rtg_l10",
                         default=LEAGUE_AVG_ORTG),
            drtg_l5=col(row, "drtg_l5", "def_rtg_l5",
                        default=LEAGUE_AVG_DRTG),
            drtg_l10=col(row, "drtg_l10", "def_rtg_l10",
                         default=LEAGUE_AVG_DRTG),
            pace_l5=col(row, "pace_l5", "poss_l5", default=LEAGUE_AVG_PACE),
            pace_l10=col(row, "pace_l10", "poss_l10", default=LEAGUE_AVG_PACE),
            efg_l5=col(row, "efg_l5", "eff_fg_l5", default=LEAGUE_AVG_EFG),
            efg_l10=col(row, "efg_l10", "eff_fg_l10", default=LEAGUE_AVG_EFG),
            tov_l5=col(row, "tov_l5", "to_pct_l5", default=LEAGUE_AVG_TOV),
            tov_l10=col(row, "tov_l10", "to_pct_l10", default=LEAGUE_AVG_TOV),
            three_pct_l5=col(row, "three_pct_l5", "fg3_pct_l5", default=33.5),
            three_pct_l10=col(row, "three_pct_l10", "fg3_pct_l10", default=33.5),

            net_rtg_std_l10=col(row, "net_rtg_std_l10", "net_eff_std_l10",
                                "net_rtg_std", default=8.0),
            efg_std_l10=col(row, "efg_std_l10", "eff_fg_std", default=5.0),
            consistency_score=col(
                row,
                "consistency_score",
                "consistency",
                default=max(20.0, 100.0 - col(row, "net_rtg_std_l10", default=8.0) * 5.0),
            ),

            # CAGE composites
            suffocation=col(row, "t_suffocation_rating", "suffocation",
                            "suffocation_rating", default=50.0),
            momentum=col(row, "t_momentum_quality_rating", "momentum",
                         "momentum_rating", "momentum_score", default=50.0),
            clutch_rating=col(row, "clutch_rating", "clutch", default=50.0),
            floor_em=col(row, "floor_em", "floor_net_rtg", default=-8.0),
            ceiling_em=col(row, "ceiling_em", "ceiling_net_rtg", default=8.0),
            dna_score=col(row, "t_tournament_dna_score", "dna_score",
                          "tournament_dna", default=50.0),
            star_risk=col(row, "t_star_reliance_risk", "star_risk",
                          "star_reliance", default=50.0),
            regression_risk=int(col(row, "t_regression_risk_flag",
                                    "regression_risk", default=0)),
            resume_score=col(row, "resume_score", "resume", default=50.0),
            cage_power_index=col(row, "cage_power_index", "power_index",
                                 "cage_pi", default=50.0),

            # Luck & record
            luck=col(row, "luck_score", "luck", default=0.0),
            pythagorean_win_pct=col(
                row,
                "pythagorean_win_pct",
                "pyth_win_pct",
                "pyth_wp",
                default=_g(row, "wins", 0) / max(1.0, _g(row, "wins", 0) + _g(row, "losses", 0)),
            ),
            actual_win_pct=col(row, "season_win_pct", "win_pct", default=0.5),
            home_wpct=col(row, "home_win_pct", "home_wp", default=0.65),
            away_wpct=col(row, "away_win_pct", "away_wp", default=0.40),
            close_wpct=col(row, "close_game_win_pct", "close_wp", default=0.50),
            win_streak=col(row, "win_streak", default=0.0),
            sos=col(row, "opp_avg_net_rtg_season", "sos", "strength_of_schedule",
                    default=0.0),
            opp_avg_net_rtg=col(row, "opp_avg_net_rtg_season",
                                "opp_avg_net_rtg", default=0.0),
            wab=col(row, "wab", default=0.0),
            opp_avg_ortg=col(row, "opp_avg_ortg_season", "opp_avg_ortg",
                             default=LEAGUE_AVG_ORTG),
            opp_avg_drtg=col(row, "opp_avg_drtg_season", "opp_avg_drtg",
                             default=LEAGUE_AVG_DRTG),
            opp_orb_pct=col(row, "opp_avg_orb_season", "opp_orb_pct", default=30.0),
            rest_days=col(row, "rest_days", default=3.0),
            games_l7=col(row, "games_l7", default=2.0),
            fatigue_index=col(row, "fatigue_index", default=0.0),

            # ATS
            cover_rate_season=col(row, "cover_rate_season", "cover_wtd_qual_rate_l10",
                                  "ats_cover_rate", default=0.5),
            cover_rate_l10=col(row, "cover_rate_l10", "cover_wtd_qual_rate_l10", default=0.5),
            ats_margin_l10=col(row, "ats_margin_l10", "cover_margin_l10", default=0.0),
            cover_margin=col(row, "cover_margin", default=0.0),
            cover_streak=col(row, "cover_streak", default=0.0),
            momentum_tier=str(row.get("momentum_tier") or
                              row.get("t_momentum_tier") or ""),
            ha_net_rtg_l10=col(
                row,
                "ha_net_rtg_l10",
                "ha_net_eff_l10",
                "home_net_rtg_season",
                "away_net_rtg_season",
                "net_rtg",
                default=0.0,
            ),
        )
        profiles[tid] = tp

    # Log how many teams have non-default cage_em (detects column mismatch)
    non_default = sum(1 for p in profiles.values() if p.cage_em != 0.0)
    median_games = int(pd.Series(_team_game_count.values()).median()) if _team_game_count else 0
    if non_default == 0:
        log.warning(
            "[DIAG] ALL %d team profiles have cage_em=0.0 — column name mismatch likely. "
            "Expected 'adj_net_rtg' or 'net_eff' in source CSV. Actual columns: %s",
            len(profiles), [c for c in df.columns if 'net' in c.lower() or 'eff' in c.lower()][:10],
        )
    else:
        _em_vals = [p.cage_em for p in profiles.values()]
        log.info(
            # Bugfix: prior malformed log args concatenated two messages and caused
            # syntax/runtime failure. Use one explicit, correctly-parameterized message.
            "[DIAG] load_team_profiles | cage_em nonzero: %d/%d | games_before median: %d | "
            "range: %.1f to %.1f",
            non_default, len(profiles), median_games, min(_em_vals), max(_em_vals)
        )

    return profiles




def _coalesce_pred_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee pred_spread is populated before write.
    Tries aliases in priority order — first non-null wins.
    Logs a warning if recovery was needed so it is visible in run logs.
    Never silently writes a fully-null pred_spread column.
    """
    if "pred_spread" not in df.columns or df["pred_spread"].isna().all():
        # Try to find a source for pred_spread
        source_found = False
        for alias in ["ens_ens_spread", "predicted_spread", "ensemble_spread", "ens_spread"]:
            if alias in df.columns and df[alias].notna().any():
                df["pred_spread"] = df[alias]
                source_found = True
                log.warning(
                    "[INTEGRITY] pred_spread was null — recovered from '%s' (%d rows)",
                    alias, df["pred_spread"].notna().sum()
                )
                break
        else:
            log.error(
                "[INTEGRITY] pred_spread is null and no alias found. "
                "Aliases checked: ens_ens_spread, predicted_spread, ensemble_spread. "
                "Available columns: %s",
                sorted(df.columns.tolist())
            )

    # Final gate — raise if still null after recovery attempts
    if "pred_spread" not in df.columns or df["pred_spread"].isna().all():
        raise RuntimeError(
            "pred_spread is missing or fully null after all recovery attempts. "
            "Cannot write predictions file with no spread values."
        )

    null_pct = df["pred_spread"].isna().mean() * 100

    if null_pct > 50:
        raise RuntimeError(
            f"pred_spread is {null_pct:.0f}% null after all recovery attempts. "
            "Integrity gate failed (>50% null)."
        )

    if null_pct > 10:
        log.warning(
            "[INTEGRITY] pred_spread is %.0f%% null after coalesce — "
            "check model output for games with insufficient history",
            null_pct
        )
    return df

def results_to_csv(
    results: List[EnsembleResult],
    path: Path,
    game_metadata: Optional[List[Dict]] = None,
) -> None:
    """Write ensemble results to CSV, optionally merging game metadata."""
    rows = []
    for i, res in enumerate(results):
        flat = res.to_flat_dict()
        if game_metadata and i < len(game_metadata):
            flat.update(game_metadata[i])
        rows.append(flat)

    if rows:
        out_df = _coalesce_pred_spread(pd.DataFrame(rows))

        model_spread_cols = [c for c in out_df.columns if c.endswith("_spread") and c not in {"ens_spread", "pred_spread"}]
        model_total_cols = [c for c in out_df.columns if c.endswith("_total") and c != "ens_total"]
        if model_spread_cols:
            same_spread_ratio = float((out_df[model_spread_cols].nunique(axis=1) <= 1).mean())
            if same_spread_ratio >= 0.90:
                log.warning(
                    "[INTEGRITY] %.0f%% of rows have identical per-model spreads. Check profile lookup/id normalization.",
                    same_spread_ratio * 100,
                )
        if model_total_cols:
            same_total_ratio = float((out_df[model_total_cols].nunique(axis=1) <= 1).mean())
            if same_total_ratio >= 0.90:
                log.warning(
                    "[INTEGRITY] %.0f%% of rows have identical per-model totals. Check model input diversity.",
                    same_total_ratio * 100,
                )

        out_df.to_csv(path, index=False)
        log.info("Wrote %d ensemble predictions to %s", len(rows), path)
