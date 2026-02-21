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
        }
        for mp in self.model_predictions:
            name = mp.model_name.lower()
            d[f"{name}_spread"] = round(mp.spread, 2)
            d[f"{name}_total"]  = round(mp.total, 1)
            d[f"{name}_conf"]   = round(mp.confidence, 3)
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
        efg_delta = home.efg_vs_opp - away.efg_vs_opp
        tov_delta = -(home.tov_vs_opp - away.tov_vs_opp)
        orb_delta = home.orb_vs_opp - away.orb_vs_opp
        ftr_delta = home.ftr_vs_opp - away.ftr_vs_opp

        composite = (
            self.W_EFG * efg_delta
            + self.W_TOV * tov_delta
            + self.W_ORB * orb_delta
            + self.W_FTR * ftr_delta
        )

        margin = composite * 0.8 + self._hca(neutral)
        spread = -margin

        pace  = self._expected_pace(home, away)
        total = self._eff_to_total(home.cage_o, away.cage_o, pace)

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

        total = self._eff_to_total(home.cage_o, away.cage_o, pace)

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
        total = self._eff_to_total(home.cage_o, away.cage_o, pace)

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
        margin = eff_edge + rest_edge + split_edge + close_edge + streak_edge
        if not neutral:
            margin += HCA * 0.5
        spread = -margin

        total = self._eff_to_total(home.cage_o, away.cage_o, pace)

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

        margin = (
            0.45 * power_edge
            + 0.15 * suff_edge
            + 0.15 * clutch_edge
            + 0.15 * resume_edge
            + 0.10 * dna_edge
        ) + self._hca(neutral)
        spread = -margin

        pace = self._expected_pace(home, away)
        total = self._eff_to_total(home.cage_o, away.cage_o, pace)

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


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════

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
        MomentumModel,
        SituationalModel,
        CAGERankingsModel,
        RegressedEfficiencyModel,
    ]

    def __init__(self, config: EnsembleConfig = None):
        self.config = config or EnsembleConfig()
        self.models = [cls() for cls in self.MODELS]

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
            except Exception as exc:
                log.debug("Model %s failed: %s", model.name, exc)

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

        return EnsembleResult(
            spread=round(ens_spread, 2),
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
    for path in [snapshot_path, weighted_path]:
        if path.exists() and path.stat().st_size > 100:
            log.info("Loading team profiles from %s", path.name)
            df = pd.read_csv(path, dtype=str, low_memory=False)
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

    # Keep last row per team (most recent game)
    if "game_datetime_utc" in df.columns:
        df["game_datetime_utc"] = pd.to_datetime(
            df["game_datetime_utc"], utc=True, errors="coerce"
        )
        df = df.sort_values("game_datetime_utc")
    df = df.drop_duplicates("team_id", keep="last")

    profiles: Dict[str, TeamProfile] = {}

    for _, row in df.iterrows():
        tid = str(row.get("team_id", ""))
        if not tid:
            continue

        def g(col, default=0.0):
            v = row.get(col, default)
            try:
                return float(v) if pd.notna(v) else default
            except (TypeError, ValueError):
                return default

        tp = TeamProfile(
            team_id=tid,
            team_name=str(row.get("team", "")),
            conference=str(row.get("conference", "")),
            games_before=int(g("games_played", 0))
            or int(g("game_number", 0)),
            cage_em=g("adj_net_rtg"),
            cage_o=g("adj_ortg", LEAGUE_AVG_ORTG),
            cage_d=g("adj_drtg", LEAGUE_AVG_DRTG),
            cage_t=g("adj_pace", LEAGUE_AVG_PACE),
            barthag=g("barthag", 0.5),
            efg_pct=g("efg_pct", LEAGUE_AVG_EFG),
            tov_pct=g("tov_pct", LEAGUE_AVG_TOV),
            orb_pct=g("orb_pct", 30.0),
            drb_pct=g("drb_pct", 70.0),
            ftr=g("ftr", LEAGUE_AVG_FTR),
            ft_pct=g("ft_pct", 71.0),
            three_pct=g("three_pct", 33.5),
            three_par=g("three_par", 35.0),
            opp_efg_pct=g("opp_avg_efg_season", LEAGUE_AVG_EFG),
            opp_tov_pct=g("opp_avg_tov_season", LEAGUE_AVG_TOV),
            opp_ftr=g("opp_avg_ftr_season", LEAGUE_AVG_FTR),
            efg_vs_opp=g("efg_vs_opp_season", 0.0),
            tov_vs_opp=g("tov_vs_opp_season", 0.0),
            orb_vs_opp=g("orb_vs_opp_season", 0.0),
            ftr_vs_opp=g("ftr_vs_opp_season", 0.0),
            net_rtg_l5=g("net_rtg_l5", 0.0),
            net_rtg_l10=g("net_rtg_l10", 0.0),
            ortg_l5=g("ortg_l5", LEAGUE_AVG_ORTG),
            ortg_l10=g("ortg_l10", LEAGUE_AVG_ORTG),
            drtg_l5=g("drtg_l5", LEAGUE_AVG_DRTG),
            drtg_l10=g("drtg_l10", LEAGUE_AVG_DRTG),
            pace_l5=g("pace_l5", LEAGUE_AVG_PACE),
            pace_l10=g("pace_l10", LEAGUE_AVG_PACE),
            efg_l5=g("efg_l5", LEAGUE_AVG_EFG),
            efg_l10=g("efg_l10", LEAGUE_AVG_EFG),
            tov_l5=g("tov_l5", LEAGUE_AVG_TOV),
            tov_l10=g("tov_l10", LEAGUE_AVG_TOV),
            three_pct_l5=g("three_pct_l5", 33.5),
            three_pct_l10=g("three_pct_l10", 33.5),
            net_rtg_std_l10=g("net_rtg_std_l10", 8.0),
            efg_std_l10=g("efg_std_l10", 5.0),
            consistency_score=g("consistency_score", 50.0),
            suffocation=g("t_suffocation_rating", 50.0),
            momentum=g("t_momentum_quality_rating", 50.0),
            clutch_rating=g("clutch_rating", 50.0),
            floor_em=g("floor_em", -8.0),
            ceiling_em=g("ceiling_em", 8.0),
            dna_score=g("t_tournament_dna_score", 50.0),
            star_risk=g("t_star_reliance_risk", 50.0),
            regression_risk=int(g("t_regression_risk_flag", 0)),
            resume_score=g("resume_score", 50.0),
            cage_power_index=g("cage_power_index", 50.0),
            luck=g("luck_score", 0.0),
            pythagorean_win_pct=g("pythagorean_win_pct", 0.5),
            actual_win_pct=g("season_win_pct", 0.5),
            home_wpct=g("home_win_pct", 0.65),
            away_wpct=g("away_win_pct", 0.40),
            close_wpct=g("close_game_win_pct", 0.50),
            win_streak=g("win_streak", 0.0),
            sos=g("opp_avg_net_rtg_season", 0.0),
            opp_avg_net_rtg=g("opp_avg_net_rtg_season", 0.0),
            wab=g("wab", 0.0),
            opp_avg_ortg=g("opp_avg_ortg_season", LEAGUE_AVG_ORTG),
            opp_avg_drtg=g("opp_avg_drtg_season", LEAGUE_AVG_DRTG),
            opp_orb_pct=g("opp_avg_orb_season", 30.0),
        )
        profiles[tid] = tp

    log.info("Loaded %d team profiles", len(profiles))
    return profiles


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
        pd.DataFrame(rows).to_csv(path, index=False)
        log.info("Wrote %d ensemble predictions to %s", len(rows), path)
