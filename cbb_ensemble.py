#!/usr/bin/env python3
"""
ESPN CBB Pipeline — Ensemble Prediction Engine
cbb_ensemble.py

Seven independent sub-models, each producing a point spread and game total.
A configurable weighted ensemble combines them into a final prediction.

ENSEMBLE MEMBERS
─────────────────────────────────────────────────────────────────────────────
M1  FourFactorsModel          Dean Oliver's framework. The statistical bedrock.
M2  AdjustedEfficiencyModel   Pure adj_ortg/drtg differential via CAGE_EM.
M3  PythagoreanModel          Win probability → spread via Log5 method.
M4  MomentumModel             L5/L10 trend-weighted form, regression-adjusted.
M5  SituationalModel          Rest, schedule density, home/road splits.
M6  CAGERankingsModel         CAGE Power Index, BARTHAG, resume — full profile.
M7  RegressedEfficiencyModel  Luck-stripped, consistency-weighted ratings.

Default spread weights:   M1=0.12  M2=0.22  M3=0.14  M4=0.16  M5=0.10  M6=0.18  M7=0.08
Default total weights:    M1=0.15  M2=0.24  M3=0.10  M4=0.14  M5=0.12  M6=0.17  M7=0.08

Pipeline position: runs after espn_rankings.py (needs cbb_rankings.csv).
Outputs: data/ensemble_predictions_YYYYMMDD.csv
"""

import argparse
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from espn_config import (
    OUT_RANKINGS, OUT_TOURNAMENT_SNAPSHOT, OUT_WEIGHTED,
    LEAGUE_AVG_ORTG, LEAGUE_AVG_DRTG, LEAGUE_AVG_PACE,
    LEAGUE_AVG_EFG, LEAGUE_AVG_TOV, LEAGUE_AVG_FTR,
    LEAGUE_AVG_ORB, LEAGUE_AVG_DRB, PYTHAGOREAN_EXP, DEFAULT_HCA,
)
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
# P1 FIX: scipy imported at module level — not deferred inside predict()
# M3 PythagoreanModel uses scipy.stats.norm.ppf for WP→spread conversion.
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TZ = ZoneInfo("America/Los_Angeles")

EFF_TO_PTS = LEAGUE_AVG_PACE / 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM PROFILE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TeamProfile:
    """
    Unified team data container. All sub-models read from this object.
    Populated from cbb_rankings.csv (richest) or team_game_weighted.csv.
    All fields have sensible league-average defaults.
    """
    team_id:     str = ""
    team_name:   str = "Unknown"
    conference:  str = ""
    record:      str = ""

    cage_em:     float = 0.0
    cage_o:      float = LEAGUE_AVG_ORTG
    cage_d:      float = LEAGUE_AVG_DRTG
    cage_t:      float = LEAGUE_AVG_PACE
    cage_power_index: float = 50.0
    barthag:     float = 0.500
    wab:         float = 0.0
    resume_score: float = 50.0
    eff_grade:   str = "B-"
    sos:         float = 0.0

    efg_pct:     float = LEAGUE_AVG_EFG
    tov_pct:     float = LEAGUE_AVG_TOV
    orb_pct:     float = LEAGUE_AVG_ORB
    drb_pct:     float = LEAGUE_AVG_DRB
    ftr:         float = LEAGUE_AVG_FTR
    ft_pct:      float = 71.0
    three_par:   float = 35.0
    three_pct:   float = 33.5

    opp_efg_pct: float = LEAGUE_AVG_EFG
    opp_tov_pct: float = LEAGUE_AVG_TOV
    opp_orb_pct: float = LEAGUE_AVG_ORB
    opp_ftr:     float = LEAGUE_AVG_FTR

    efg_vs_opp:  float = 0.0
    tov_vs_opp:  float = 0.0
    orb_vs_opp:  float = 0.0
    ftr_vs_opp:  float = 0.0

    net_rtg_l5:  float = 0.0
    net_rtg_l10: float = 0.0
    ortg_l5:     float = LEAGUE_AVG_ORTG
    ortg_l10:    float = LEAGUE_AVG_ORTG
    drtg_l5:     float = LEAGUE_AVG_DRTG
    drtg_l10:    float = LEAGUE_AVG_DRTG
    pace_l5:     float = LEAGUE_AVG_PACE
    pace_l10:    float = LEAGUE_AVG_PACE
    efg_l5:      float = LEAGUE_AVG_EFG
    efg_l10:     float = LEAGUE_AVG_EFG
    tov_l5:      float = LEAGUE_AVG_TOV
    tov_l10:     float = LEAGUE_AVG_TOV
    three_pct_l5:  float = 33.5
    three_pct_l10: float = 33.5

    net_rtg_std_l10: float = 8.0
    efg_std_l10:     float = 5.0
    consistency_score: float = 50.0

    suffocation:    float = 50.0
    momentum:       float = 50.0
    clutch_rating:  float = 50.0
    dna_score:      float = 50.0
    floor_em:       float = -8.0
    ceiling_em:     float = 8.0
    star_risk:      float = 50.0
    regression_risk: int  = 0

    luck:            float = 0.0
    pythagorean_win_pct: float = 0.5
    actual_win_pct:  float = 0.5
    home_wpct:       float = 0.65
    away_wpct:       float = 0.40
    close_wpct:      float = 0.50
    rest_days:       float = 3.0
    games_l7:        float = 2.0
    win_streak:      float = 0.0

    opp_avg_net_rtg: float = 0.0
    opp_avg_ortg:    float = LEAGUE_AVG_ORTG
    opp_avg_drtg:    float = LEAGUE_AVG_DRTG

    @property
    def trend_delta(self) -> float:
        return self.net_rtg_l5 - self.net_rtg_l10

    @property
    def three_pct_gap(self) -> float:
        return self.three_pct_l5 - self.three_pct_l10

    @property
    def fatigue_factor(self) -> float:
        rest_score    = min(self.rest_days / 4.0, 1.0)
        density_score = max(0.0, 1.0 - (self.games_l7 - 2) * 0.15)
        return round(0.6 * rest_score + 0.4 * density_score, 3)


@dataclass
class ModelPrediction:
    model_name:  str
    spread:      float
    total:       float
    confidence:  float
    home_score:  float = field(init=False)
    away_score:  float = field(init=False)
    notes:       str = ""

    def __post_init__(self):
        self.home_score = round(self.total / 2 - self.spread / 2, 1)
        self.away_score = round(self.total / 2 + self.spread / 2, 1)


@dataclass
class EnsemblePrediction:
    home_team:   str
    away_team:   str
    neutral_site: bool
    spread:      float
    total:       float
    confidence:  float
    home_score:  float
    away_score:  float
    model_predictions: List[ModelPrediction] = field(default_factory=list)
    spread_lo:   float = 0.0
    spread_hi:   float = 0.0
    total_lo:    float = 0.0
    total_hi:    float = 0.0
    spread_std:  float = 0.0
    total_std:   float = 0.0
    cage_edge:   float = 0.0
    barthag_diff: float = 0.0
    model_agreement: str = "STRONG"
    line_value:      str = ""
    predicted_at:    str = ""

    def to_flat_dict(self) -> Dict:
        row = {
            "home_team":      self.home_team,
            "away_team":      self.away_team,
            "neutral_site":   int(self.neutral_site),
            "ensemble_spread": self.spread,
            "ensemble_total":  self.total,
            "ensemble_confidence": self.confidence,
            "home_score_proj": self.home_score,
            "away_score_proj": self.away_score,
            "spread_range":   f"{self.spread_lo:+.1f} to {self.spread_hi:+.1f}",
            "total_range":    f"{self.total_lo:.1f} to {self.total_hi:.1f}",
            "spread_std":     self.spread_std,
            "total_std":      self.total_std,
            "model_agreement": self.model_agreement,
            "cage_edge":      self.cage_edge,
            "barthag_diff":   self.barthag_diff,
            "predicted_at":   self.predicted_at,
        }
        for mp in self.model_predictions:
            prefix = mp.model_name.lower().replace(" ", "_")
            row[f"{prefix}_spread"] = mp.spread
            row[f"{prefix}_total"]  = mp.total
            row[f"{prefix}_conf"]   = mp.confidence
        return row


# ═══════════════════════════════════════════════════════════════════════════════
# BASE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class BaseModel(ABC):
    name: str = "Base"

    @abstractmethod
    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:
        ...

    def _hca(self, neutral: bool) -> float:
        return 0.0 if neutral else DEFAULT_HCA

    def _clamp_spread(self, spread: float) -> float:
        return float(np.clip(spread, -35.0, 35.0))

    def _clamp_total(self, total: float) -> float:
        return float(np.clip(total, 110.0, 185.0))


# ═══════════════════════════════════════════════════════════════════════════════
# M1 — FOUR FACTORS MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class FourFactorsModel(BaseModel):
    """
    M1 — Dean Oliver's Four Factors framework.
    Factors: eFG% 0.40, TOV% 0.25, ORB% 0.20, FTR 0.15.
    Uses efg_vs_opp / tov_vs_opp (opponent-context metrics) blended 60/40 with raw.
    Blind spots: pace, recency, situational context.
    """
    name = "FourFactors"
    EFG_W = 0.40
    TOV_W = 0.25
    ORB_W = 0.20
    FTR_W = 0.15
    RAW_BLEND = 0.40
    CTX_BLEND = 0.60

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:
        raw_efg   = (home.efg_pct - away.efg_pct) - (home.opp_efg_pct - away.opp_efg_pct)
        ctx_efg   = home.efg_vs_opp - away.efg_vs_opp
        efg_edge  = self.RAW_BLEND * raw_efg + self.CTX_BLEND * ctx_efg

        raw_tov   = (away.tov_pct - home.tov_pct) - (home.opp_tov_pct - away.opp_tov_pct)
        ctx_tov   = home.tov_vs_opp - away.tov_vs_opp
        tov_edge  = self.RAW_BLEND * raw_tov + self.CTX_BLEND * ctx_tov

        raw_orb   = home.orb_pct - away.orb_pct
        ctx_orb   = home.orb_vs_opp - away.orb_vs_opp
        orb_edge  = self.RAW_BLEND * raw_orb + self.CTX_BLEND * ctx_orb

        raw_ftr   = (home.ftr - away.ftr) - (home.opp_ftr - away.opp_ftr)
        ctx_ftr   = home.ftr_vs_opp - away.ftr_vs_opp
        ftr_edge  = self.RAW_BLEND * raw_ftr + self.CTX_BLEND * ctx_ftr

        factor_edge_eff = (
            self.EFG_W * efg_edge * 1.0 +
            self.TOV_W * tov_edge * 1.0 +
            self.ORB_W * orb_edge * 0.6 +
            self.FTR_W * ftr_edge * 0.5
        )

        pace   = (home.cage_t + away.cage_t) / 2.0
        spread = -(factor_edge_eff * pace / 100.0 + self._hca(neutral))
        spread = self._clamp_spread(spread)

        home_off_eff = (
            home.efg_pct * 1.8 +
            (LEAGUE_AVG_TOV - home.tov_pct) * 0.8 +
            home.orb_pct * 0.3 +
            home.ftr * home.ft_pct / 100.0 * 0.4
        )
        away_off_eff = (
            away.efg_pct * 1.8 +
            (LEAGUE_AVG_TOV - away.tov_pct) * 0.8 +
            away.orb_pct * 0.3 +
            away.ftr * away.ft_pct / 100.0 * 0.4
        )

        home_def_factor = away.opp_efg_pct / LEAGUE_AVG_EFG
        away_def_factor = home.opp_efg_pct / LEAGUE_AVG_EFG

        home_pts = home_off_eff * away_def_factor * pace / 100.0
        away_pts = away_off_eff * home_def_factor * pace / 100.0
        total    = self._clamp_total(home_pts + away_pts)

        has_ctx = (abs(home.efg_vs_opp) + abs(away.efg_vs_opp)) > 0.5
        conf    = 0.82 if has_ctx else 0.65

        return ModelPrediction(
            model_name=self.name,
            spread=round(spread, 2),
            total=round(total, 1),
            confidence=conf,
            notes=f"efg_edge={efg_edge:.2f} tov_edge={tov_edge:.2f} orb_edge={orb_edge:.2f}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M2 — ADJUSTED EFFICIENCY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class AdjustedEfficiencyModel(BaseModel):
    """
    M2 — Pure CAGE_EM differential → spread. Most direct CAGE-powered model.
    Blind spots: recency, momentum, situational factors.
    """
    name = "AdjEfficiency"

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:
        pace = (home.cage_t + away.cage_t) / 2.0

        cage_em_diff = home.cage_em - away.cage_em
        edge_pts     = cage_em_diff * pace / 100.0
        spread       = self._clamp_spread(-(edge_pts + self._hca(neutral)))

        home_def_factor = LEAGUE_AVG_DRTG / max(away.cage_d, 70.0)
        away_def_factor = LEAGUE_AVG_DRTG / max(home.cage_d, 70.0)

        home_pts = home.cage_o * home_def_factor * pace / 100.0
        away_pts = away.cage_o * away_def_factor * pace / 100.0
        total    = self._clamp_total(home_pts + away_pts)

        em_magnitude = min(abs(cage_em_diff) / 10.0, 1.0)
        conf = 0.70 + 0.18 * em_magnitude

        return ModelPrediction(
            model_name=self.name,
            spread=round(spread, 2),
            total=round(total, 1),
            confidence=round(conf, 3),
            notes=f"cage_em_diff={cage_em_diff:+.2f} pace={pace:.1f}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M3 — PYTHAGOREAN / LOG5 MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class PythagoreanModel(BaseModel):
    """
    M3 — BARTHAG → Log5 win probability → point spread via inverse normal.
    Uses scipy_stats.norm.ppf (module-level import — P1 fix applied).
    Blind spots: pace, real-time form, situational variation.
    """
    name = "Pythagorean"
    SIGMA = 11.0

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:
        h_b  = np.clip(home.barthag, 0.001, 0.999)
        a_b  = np.clip(away.barthag, 0.001, 0.999)
        log5 = (h_b - h_b * a_b) / (h_b + a_b - 2 * h_b * a_b)
        log5 = float(np.clip(log5, 0.001, 0.999))

        if not neutral:
            hca_wp = 3.2 / (self.SIGMA * np.sqrt(2 * np.pi) * 0.4)
            log5   = float(np.clip(log5 + hca_wp * 0.5, 0.001, 0.999))

        # P1 FIX: use module-level scipy_stats (not local import)
        implied_edge = scipy_stats.norm.ppf(log5) * self.SIGMA
        spread = self._clamp_spread(-implied_edge)

        def _barthag_to_ortg(b: float) -> float:
            return 85.0 + b * 40.0

        home_ortg = _barthag_to_ortg(h_b)
        away_ortg = _barthag_to_ortg(a_b)

        home_net  = home.cage_em if home.cage_em != 0 else (home.barthag - 0.5) * 30
        away_net  = away.cage_em if away.cage_em != 0 else (away.barthag - 0.5) * 30
        home_drtg = home_ortg - home_net
        away_drtg = away_ortg - away_net

        pace      = (home.cage_t + away.cage_t) / 2.0
        home_pts  = home_ortg * (LEAGUE_AVG_DRTG / max(away_drtg, 70)) * pace / 100.0
        away_pts  = away_ortg * (LEAGUE_AVG_DRTG / max(home_drtg, 70)) * pace / 100.0
        total     = self._clamp_total(home_pts + away_pts)

        barth_diff = abs(h_b - a_b)
        conf = 0.65 + 0.20 * min(barth_diff / 0.3, 1.0)

        return ModelPrediction(
            model_name=self.name,
            spread=round(spread, 2),
            total=round(total, 1),
            confidence=round(conf, 3),
            notes=f"log5_wp={log5:.3f} home_barthag={h_b:.4f} away_barthag={a_b:.4f}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M4 — MOMENTUM / TREND MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MomentumModel(BaseModel):
    """
    M4 — L5/L10 recency-weighted form with regression adjustment.
    Regression: pulls L5 toward L10 for teams on unsustainable shooting heaters.
    CAGE momentum score adds ±1 pt directional signal.
    Blind spots: opponent quality of recent games.
    """
    name = "Momentum"
    L5_WEIGHT  = 0.70
    L10_WEIGHT = 0.30
    HEATER_REGRESSION = 0.40
    TREND_SCALE = 0.8

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:
        def _regressed_l5(team: TeamProfile) -> float:
            net_l5 = team.net_rtg_l5
            net_l10 = team.net_rtg_l10
            if team.three_pct_gap > 5.0 and team.regression_risk:
                net_l5 = net_l5 - self.HEATER_REGRESSION * (net_l5 - net_l10)
            return net_l5

        home_l5_adj = _regressed_l5(home)
        away_l5_adj = _regressed_l5(away)

        home_form = self.L5_WEIGHT * home_l5_adj  + self.L10_WEIGHT * home.net_rtg_l10
        away_form = self.L5_WEIGHT * away_l5_adj  + self.L10_WEIGHT * away.net_rtg_l10

        home_trend = home.trend_delta
        away_trend = away.trend_delta
        trend_edge = (home_trend - away_trend) * self.TREND_SCALE

        home_mom_adj = (home.momentum - 50.0) / 50.0 * 1.0
        away_mom_adj = (away.momentum - 50.0) / 50.0 * 1.0
        cage_mom_edge = home_mom_adj - away_mom_adj

        form_diff  = home_form - away_form
        pace       = (home.pace_l5 + away.pace_l5) / 2.0
        edge_pts   = form_diff * pace / 100.0 + trend_edge + cage_mom_edge
        spread     = self._clamp_spread(-(edge_pts + self._hca(neutral)))

        home_ortg_form = home.ortg_l5 * self.L5_WEIGHT + home.ortg_l10 * self.L10_WEIGHT
        away_ortg_form = away.ortg_l5 * self.L5_WEIGHT + away.ortg_l10 * self.L10_WEIGHT
        home_drtg_form = home.drtg_l5 * self.L5_WEIGHT + home.drtg_l10 * self.L10_WEIGHT
        away_drtg_form = away.drtg_l5 * self.L5_WEIGHT + away.drtg_l10 * self.L10_WEIGHT

        home_pts  = home_ortg_form * (LEAGUE_AVG_DRTG / max(away_drtg_form, 70)) * pace / 100.0
        away_pts  = away_ortg_form * (LEAGUE_AVG_DRTG / max(home_drtg_form, 70)) * pace / 100.0
        total     = self._clamp_total(home_pts + away_pts)

        home_regressed = home.three_pct_gap > 5.0 and bool(home.regression_risk)
        away_regressed = away.three_pct_gap > 5.0 and bool(away.regression_risk)
        conf = 0.74 - 0.08 * home_regressed - 0.08 * away_regressed

        return ModelPrediction(
            model_name=self.name,
            spread=round(spread, 2),
            total=round(total, 1),
            confidence=round(conf, 3),
            notes=(f"home_form={home_form:+.1f} away_form={away_form:+.1f} "
                   f"trend_edge={trend_edge:+.1f} mom_edge={cage_mom_edge:+.2f}"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M5 — SITUATIONAL MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class SituationalModel(BaseModel):
    """
    M5 — Rest, fatigue, schedule density, home/away splits.
    Intentionally small spread effect (±4 pts max) — adjustment model, not primary.
    Blind spots: all quality-related factors.
    """
    name = "Situational"
    REST_MAX     = 1.5
    DENSITY_MAX  = 1.2
    SPLIT_MAX    = 2.0
    STREAK_MAX   = 0.8
    CLUTCH_MAX   = 0.6

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:
        rest_diff  = home.rest_days - away.rest_days
        rest_edge  = float(np.clip(rest_diff * 0.25, -self.REST_MAX, self.REST_MAX))

        home_fatigue = home.fatigue_factor
        away_fatigue = away.fatigue_factor
        density_edge = float(np.clip(
            (home_fatigue - away_fatigue) * self.DENSITY_MAX * 2,
            -self.DENSITY_MAX, self.DENSITY_MAX
        ))

        if neutral:
            split_edge = 0.0
        else:
            home_home_boost = home.home_wpct - home.actual_win_pct
            away_road_drag  = away.actual_win_pct - away.away_wpct
            split_edge = float(np.clip(
                (home_home_boost + away_road_drag) * 5.0,
                -self.SPLIT_MAX, self.SPLIT_MAX
            ))

        home_streak_pts = float(np.clip(home.win_streak * 0.12, -self.STREAK_MAX, self.STREAK_MAX))
        away_streak_pts = float(np.clip(away.win_streak * 0.12, -self.STREAK_MAX, self.STREAK_MAX))
        streak_edge = home_streak_pts - away_streak_pts

        home_clutch_adj = (home.clutch_rating - 50.0) / 50.0 * self.CLUTCH_MAX
        away_clutch_adj = (away.clutch_rating - 50.0) / 50.0 * self.CLUTCH_MAX
        clutch_edge = home_clutch_adj - away_clutch_adj

        total_edge = rest_edge + density_edge + split_edge + streak_edge + clutch_edge
        hca        = self._hca(neutral)
        spread     = self._clamp_spread(-(total_edge + hca))

        home_fatigue_pts = (1.0 - home_fatigue) * 3.0
        away_fatigue_pts = (1.0 - away_fatigue) * 3.0

        pace       = (home.cage_t + away.cage_t) / 2.0
        home_base  = home.cage_o * (LEAGUE_AVG_DRTG / max(away.cage_d, 70)) * pace / 100.0
        away_base  = away.cage_o * (LEAGUE_AVG_DRTG / max(home.cage_d, 70)) * pace / 100.0
        total      = self._clamp_total(
            (home_base - home_fatigue_pts) + (away_base - away_fatigue_pts)
        )

        conf = 0.58 + 0.10 * (abs(rest_diff) > 1) + 0.08 * (abs(total_edge) > 2.0)

        return ModelPrediction(
            model_name=self.name,
            spread=round(spread, 2),
            total=round(total, 1),
            confidence=round(conf, 3),
            notes=(f"rest={rest_edge:+.2f} density={density_edge:+.2f} "
                   f"split={split_edge:+.2f} streak={streak_edge:+.2f} clutch={clutch_edge:+.2f}"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M6 — CAGE RANKINGS MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class CAGERankingsModel(BaseModel):
    """
    M6 — Full CAGE profile: CAGE_EM + Power Index + Resume + Suffocation + BARTHAG.
    Most holistic model. Blind spots: recency, pace context.
    """
    name = "CAGERankings"
    EM_W       = 0.40
    PI_W       = 0.25
    RESUME_W   = 0.15
    SUFF_W     = 0.12
    BARTH_W    = 0.08

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:
        pace        = (home.cage_t + away.cage_t) / 2.0
        em_edge     = (home.cage_em - away.cage_em) * pace / 100.0
        pi_edge     = (home.cage_power_index - away.cage_power_index) * 0.04
        resume_edge = (home.resume_score - away.resume_score) * 0.025
        suff_edge   = (home.suffocation - away.suffocation) * 0.025
        barth_edge  = (home.barthag - away.barthag) * 15.0

        composite = (
            self.EM_W     * em_edge     +
            self.PI_W     * pi_edge     +
            self.RESUME_W * resume_edge +
            self.SUFF_W   * suff_edge   +
            self.BARTH_W  * barth_edge
        )

        spread = self._clamp_spread(-(composite + self._hca(neutral)))

        home_def_factor = LEAGUE_AVG_DRTG / max(away.cage_d, 70.0)
        away_def_factor = LEAGUE_AVG_DRTG / max(home.cage_d, 70.0)

        home_pts = home.cage_o * home_def_factor * pace / 100.0
        away_pts = away.cage_o * away_def_factor * pace / 100.0

        avg_suffocation = (home.suffocation + away.suffocation) / 2.0
        suff_total_adj  = (avg_suffocation - 50.0) / 50.0 * (-2.5)
        total = self._clamp_total(home_pts + away_pts + suff_total_adj)

        has_full_cage = (
            home.cage_power_index != 50.0 and
            away.cage_power_index != 50.0 and
            home.resume_score     != 50.0 and
            away.resume_score     != 50.0
        )
        conf = 0.85 if has_full_cage else 0.70

        return ModelPrediction(
            model_name=self.name,
            spread=round(spread, 2),
            total=round(total, 1),
            confidence=round(conf, 3),
            notes=(f"em_edge={em_edge:+.2f} pi_edge={pi_edge:+.2f} "
                   f"resume_edge={resume_edge:+.2f} suff_edge={suff_edge:+.2f}"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# M7 — REGRESSED EFFICIENCY MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class RegressedEfficiencyModel(BaseModel):
    """
    M7 — Luck-stripped, consistency-weighted efficiency. The "sober" model.
    Strips luck score and regresses toward floor/ceiling midpoint.
    Blind spots: clutch teams who legitimately beat Pythagorean.
    """
    name = "RegressedEff"
    LUCK_REGRESSION = 0.40
    FLOOR_TILT = 0.55
    CEILING_TILT = 0.45

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:
        def _luck_adjusted_em(team: TeamProfile) -> float:
            em = team.cage_em
            if abs(team.luck) > 0.02:
                luck_correction = team.luck * self.LUCK_REGRESSION * 20.0
                em = em - luck_correction
            return em

        home_adj_em = _luck_adjusted_em(home)
        away_adj_em = _luck_adjusted_em(away)

        def _consistency_weight(team: TeamProfile) -> float:
            c = team.consistency_score / 100.0
            return 0.60 + 0.40 * c

        home_em_w   = home_adj_em * _consistency_weight(home)
        away_em_w   = away_adj_em * _consistency_weight(away)

        home_mid = home.floor_em * self.FLOOR_TILT + home.ceiling_em * self.CEILING_TILT
        away_mid = away.floor_em * self.FLOOR_TILT + away.ceiling_em * self.CEILING_TILT

        home_final_em = 0.60 * home_em_w + 0.40 * home_mid
        away_final_em = 0.60 * away_em_w + 0.40 * away_mid

        pace       = (home.cage_t + away.cage_t) / 2.0
        em_diff    = home_final_em - away_final_em
        edge_pts   = em_diff * pace / 100.0
        spread     = self._clamp_spread(-(edge_pts + self._hca(neutral)))

        def _regressed_ortg(team: TeamProfile) -> float:
            return team.cage_o - team.luck * self.LUCK_REGRESSION * 15.0

        def _regressed_drtg(team: TeamProfile) -> float:
            return team.cage_d + team.luck * self.LUCK_REGRESSION * 15.0

        home_pts  = _regressed_ortg(home) * (LEAGUE_AVG_DRTG / max(_regressed_drtg(away), 70)) * pace / 100.0
        away_pts  = _regressed_ortg(away) * (LEAGUE_AVG_DRTG / max(_regressed_drtg(home), 70)) * pace / 100.0
        total     = self._clamp_total(home_pts + away_pts)

        luck_penalty = min(abs(home.luck) + abs(away.luck), 0.20)
        conf = 0.72 - luck_penalty * 1.5

        return ModelPrediction(
            model_name=self.name,
            spread=round(spread, 2),
            total=round(total, 1),
            confidence=round(conf, 3),
            notes=(f"home_adj_em={home_adj_em:+.2f}(luck={home.luck:+.3f}) "
                   f"away_adj_em={away_adj_em:+.2f}(luck={away.luck:+.3f})"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE WEIGHTING & COMBINER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnsembleConfig:
    spread_weights: Dict[str, float] = field(default_factory=lambda: {
        "FourFactors":    0.12,
        "AdjEfficiency":  0.22,
        "Pythagorean":    0.14,
        "Momentum":       0.16,
        "Situational":    0.10,
        "CAGERankings":   0.18,
        "RegressedEff":   0.08,
    })
    total_weights: Dict[str, float] = field(default_factory=lambda: {
        "FourFactors":    0.15,
        "AdjEfficiency":  0.24,
        "Pythagorean":    0.10,
        "Momentum":       0.14,
        "Situational":    0.12,
        "CAGERankings":   0.17,
        "RegressedEff":   0.08,
    })
    min_confidence_threshold: float = 0.60
    strong_agreement_std:   float = 2.0
    moderate_agreement_std: float = 4.0

    def normalize(self) -> None:
        s_sum = sum(self.spread_weights.values())
        t_sum = sum(self.total_weights.values())
        if s_sum > 0:
            self.spread_weights = {k: v / s_sum for k, v in self.spread_weights.items()}
        if t_sum > 0:
            self.total_weights  = {k: v / t_sum for k, v in self.total_weights.items()}

    @classmethod
    def from_optimized(cls, path: Path = Path("data/backtest_optimized_weights.json")):
        cfg = cls()
        try:
            if path.exists() and path.stat().st_size > 10:
                payload = json.loads(path.read_text())
                optimized = payload.get("weights") if isinstance(payload, dict) else None
                if isinstance(optimized, dict) and optimized:
                    cfg.spread_weights.update(optimized)
                    cfg.total_weights.update(optimized)
        except Exception:
            pass
        cfg.normalize()
        return cfg


class EnsemblePredictor:
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
        self.config.normalize()
        self.models = [M() for M in self.MODELS]

    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
        spread_line: Optional[float] = None,
        total_line:  Optional[float] = None,
    ) -> EnsemblePrediction:
        raw_predictions: List[ModelPrediction] = []
        for model in self.models:
            try:
                pred = model.predict(home, away, neutral)
                raw_predictions.append(pred)
            except Exception as exc:
                log.warning(f"Model {model.name} failed: {exc}")

        if not raw_predictions:
            raise RuntimeError("All sub-models failed — cannot produce ensemble")

        by_name = {p.model_name: p for p in raw_predictions}

        def _effective_weight(name: str, weight: float) -> float:
            pred = by_name.get(name)
            if pred is None:
                return 0.0
            return weight * 0.5 if pred.confidence < self.config.min_confidence_threshold else weight

        eff_spread_w = {k: _effective_weight(k, v) for k, v in self.config.spread_weights.items()}
        eff_total_w  = {k: _effective_weight(k, v) for k, v in self.config.total_weights.items()}

        sw_sum = sum(eff_spread_w.values()) or 1.0
        tw_sum = sum(eff_total_w.values())  or 1.0
        eff_spread_w = {k: v / sw_sum for k, v in eff_spread_w.items()}
        eff_total_w  = {k: v / tw_sum for k, v in eff_total_w.items()}

        ensemble_spread = sum(
            by_name[n].spread * w for n, w in eff_spread_w.items() if n in by_name
        )
        ensemble_total = sum(
            by_name[n].total * w for n, w in eff_total_w.items() if n in by_name
        )
        ensemble_conf = sum(
            by_name[n].confidence * eff_spread_w.get(n, 0) for n in by_name
        )

        spreads = [p.spread for p in raw_predictions]
        totals  = [p.total  for p in raw_predictions]
        spread_std = float(np.std(spreads))
        total_std  = float(np.std(totals))

        if spread_std <= self.config.strong_agreement_std:
            agreement = "STRONG"
        elif spread_std <= self.config.moderate_agreement_std:
            agreement = "MODERATE"
        else:
            agreement = "SPLIT"

        cage_edge   = (home.cage_em - away.cage_em) * (home.cage_t + away.cage_t) / 200.0
        barth_diff  = home.barthag - away.barthag

        line_value = ""
        if spread_line is not None:
            diff = ensemble_spread - spread_line
            if abs(diff) >= 3.0:
                side = home.team_name if diff < 0 else away.team_name
                line_value = f"{side} VALUE ({diff:+.1f} vs line)"
            elif abs(diff) >= 1.5:
                line_value = f"SLIGHT LEAN ({diff:+.1f})"
            else:
                line_value = "NEAR LINE"

        home_score = round(ensemble_total / 2 - ensemble_spread / 2, 1)
        away_score = round(ensemble_total / 2 + ensemble_spread / 2, 1)

        return EnsemblePrediction(
            home_team    = home.team_name,
            away_team    = away.team_name,
            neutral_site = neutral,
            spread       = round(ensemble_spread, 2),
            total        = round(ensemble_total,  1),
            confidence   = round(ensemble_conf,   3),
            home_score   = home_score,
            away_score   = away_score,
            model_predictions = raw_predictions,
            spread_lo    = round(min(spreads), 2),
            spread_hi    = round(max(spreads), 2),
            total_lo     = round(min(totals),  1),
            total_hi     = round(max(totals),  1),
            spread_std   = round(spread_std,   2),
            total_std    = round(total_std,    2),
            cage_edge    = round(cage_edge,    2),
            barthag_diff = round(barth_diff,   4),
            model_agreement = agreement,
            line_value   = line_value,
            predicted_at = datetime.now(TZ).isoformat(),
        )

    def predict_batch(
        self,
        matchups: List[Tuple[TeamProfile, TeamProfile, bool]],
        spread_lines: Optional[List[Optional[float]]] = None,
        total_lines:  Optional[List[Optional[float]]] = None,
    ) -> List[EnsemblePrediction]:
        if spread_lines is None:
            spread_lines = [None] * len(matchups)
        if total_lines is None:
            total_lines  = [None] * len(matchups)
        results = []
        for (home, away, neutral), sl, tl in zip(matchups, spread_lines, total_lines):
            try:
                results.append(self.predict(home, away, neutral, sl, tl))
            except Exception as exc:
                log.error(f"Ensemble failed for {home.team_name} vs {away.team_name}: {exc}")
        return results


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — CAGE RANKINGS → TeamProfile
# ═══════════════════════════════════════════════════════════════════════════════

def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    v = row.get(col, default)
    try:
        return float(v) if pd.notna(v) else default
    except (TypeError, ValueError):
        return default


def _parse_record_wpct(record_str: str) -> Tuple[float, float, float]:
    """Parse 'W-L' or 'W-L (H W-L, A W-L)' into (overall, home, away) win pcts."""
    try:
        parts = str(record_str).strip().split()[0]  # Take first token
        w, l  = parts.split("-")
        total = int(w) + int(l)
        return (int(w) / total) if total > 0 else 0.5, 0.65, 0.40
    except Exception:
        return 0.5, 0.65, 0.40


def _load_from_rankings(rankings_path: Path = OUT_RANKINGS) -> Dict[str, "TeamProfile"]:
    """
    Load all teams from cbb_rankings.csv into TeamProfile objects.
    Returns dict keyed by team_id (str).

    Column name mapping (P2, P4, P5 fixes applied):
      - luck_score (not luck) — matches espn_rankings.py output
      - efg_vs_opp / tov_vs_opp (not dna_efg_diff / dna_tov_diff) — matches espn_sos.py
      - Full L5/L10 rolling cols read explicitly; fall back to season cage_o/d if absent
      - home_wpct parsed from record string (not split home_wins/home_losses cols)
    """
    if not rankings_path.exists():
        log.warning(f"Rankings CSV not found at {rankings_path} — using snapshot fallback")
        return _load_from_snapshot()

    df = pd.read_csv(rankings_path, dtype=str, low_memory=False)
    str_cols = {
        "team_id", "team", "conference", "record", "home_record", "away_record",
        "eff_grade", "trend_arrow", "offensive_archetype", "t_offensive_archetype",
        "q1_record", "q2_record", "q3_record", "q4_record", "updated_at",
    }
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    profiles = {}
    for _, row in df.iterrows():
        tid = str(row.get("team_id", ""))
        if not tid:
            continue

        # P5 FIX: Try explicit rolling column names from team_game_weighted.csv
        # espn_rankings.py passes these through if it joins on team_id.
        # If missing, fall back to season cage_o/cage_d (degraded M4 signal).
        ortg_l5  = _safe(row, "ortg_l5",  _safe(row, "cage_o", LEAGUE_AVG_ORTG))
        ortg_l10 = _safe(row, "ortg_l10", _safe(row, "cage_o", LEAGUE_AVG_ORTG))
        drtg_l5  = _safe(row, "drtg_l5",  _safe(row, "cage_d", LEAGUE_AVG_DRTG))
        drtg_l10 = _safe(row, "drtg_l10", _safe(row, "cage_d", LEAGUE_AVG_DRTG))
        pace_l5  = _safe(row, "pace_l5",  _safe(row, "cage_t", LEAGUE_AVG_PACE))
        pace_l10 = _safe(row, "pace_l10", _safe(row, "cage_t", LEAGUE_AVG_PACE))
        efg_l5   = _safe(row, "efg_l5",   _safe(row, "efg_pct", LEAGUE_AVG_EFG))
        efg_l10  = _safe(row, "efg_l10",  _safe(row, "efg_pct", LEAGUE_AVG_EFG))
        tov_l5   = _safe(row, "tov_l5",   _safe(row, "tov_pct", LEAGUE_AVG_TOV))
        tov_l10  = _safe(row, "tov_l10",  _safe(row, "tov_pct", LEAGUE_AVG_TOV))
        three_l5 = _safe(row, "three_pct_l5",  _safe(row, "three_pct", 33.5))
        three_l10= _safe(row, "three_pct_l10", _safe(row, "three_pct", 33.5))

        # P5 FIX: net_rtg_l5/l10 — try weighted columns first, then raw
        net_l5  = _safe(row, "net_rtg_l5",  _safe(row, "net_rtg_wtd_qual_l5",  0.0))
        net_l10 = _safe(row, "net_rtg_l10", _safe(row, "net_rtg_wtd_qual_l10", 0.0))

        # P5 FIX: home/away win% — parse from record string rather than split cols
        overall_wpct, home_wpct, away_wpct = _parse_record_wpct(str(row.get("record", "")))
        # Try split home/away records if available
        if row.get("home_record") and str(row.get("home_record")) not in ("nan", ""):
            hw, _, = _parse_record_wpct(str(row.get("home_record", "")))
            home_wpct = hw if hw != 0.5 else home_wpct
        if row.get("away_record") and str(row.get("away_record")) not in ("nan", ""):
            aw, _, = _parse_record_wpct(str(row.get("away_record", "")))
            away_wpct = aw if aw != 0.5 else away_wpct

        p = TeamProfile(
            team_id   = tid,
            team_name = str(row.get("team",        "")),
            conference= str(row.get("conference",  "")),
            record    = str(row.get("record",      "")),

            cage_em   = _safe(row, "cage_em",    0.0),
            cage_o    = _safe(row, "cage_o",     LEAGUE_AVG_ORTG),
            cage_d    = _safe(row, "cage_d",     LEAGUE_AVG_DRTG),
            cage_t    = _safe(row, "cage_t",     LEAGUE_AVG_PACE),
            cage_power_index = _safe(row, "cage_power_index", 50.0),
            barthag   = _safe(row, "barthag",    0.500),
            wab       = _safe(row, "wab",        0.0),
            resume_score = _safe(row, "resume_score", 50.0),
            eff_grade = str(row.get("eff_grade", "B-")),
            sos       = _safe(row, "sos",        0.0),

            efg_pct   = _safe(row, "efg_pct",    LEAGUE_AVG_EFG),
            tov_pct   = _safe(row, "tov_pct",    LEAGUE_AVG_TOV),
            orb_pct   = _safe(row, "orb_pct",    LEAGUE_AVG_ORB),
            drb_pct   = _safe(row, "drb_pct",    LEAGUE_AVG_DRB),
            ftr       = _safe(row, "ftr",        LEAGUE_AVG_FTR),
            ft_pct    = _safe(row, "ft_pct",     71.0),
            three_par = _safe(row, "three_par",  35.0),
            three_pct = _safe(row, "three_pct",  33.5),

            opp_efg_pct = _safe(row, "opp_efg_pct", LEAGUE_AVG_EFG),
            opp_tov_pct = _safe(row, "opp_tov_pct", LEAGUE_AVG_TOV),
            opp_orb_pct = _safe(row, "opp_orb_pct", LEAGUE_AVG_ORB),
            opp_ftr     = _safe(row, "opp_ftr",     LEAGUE_AVG_FTR),

            # P4 FIX: efg_vs_opp / tov_vs_opp — correct column names from espn_sos.py
            # Original code incorrectly used "dna_efg_diff" / "dna_tov_diff"
            efg_vs_opp  = _safe(row, "efg_vs_opp",  0.0),
            tov_vs_opp  = _safe(row, "tov_vs_opp",  0.0),
            orb_vs_opp  = _safe(row, "orb_vs_opp",  0.0),
            ftr_vs_opp  = _safe(row, "ftr_vs_opp",  0.0),

            # P5 FIX: Rolling columns
            net_rtg_l5  = net_l5,
            net_rtg_l10 = net_l10,
            ortg_l5     = ortg_l5,
            ortg_l10    = ortg_l10,
            drtg_l5     = drtg_l5,
            drtg_l10    = drtg_l10,
            pace_l5     = pace_l5,
            pace_l10    = pace_l10,
            efg_l5      = efg_l5,
            efg_l10     = efg_l10,
            tov_l5      = tov_l5,
            tov_l10     = tov_l10,
            three_pct_l5  = three_l5,
            three_pct_l10 = three_l10,

            net_rtg_std_l10 = _safe(row, "net_rtg_std_l10", 8.0),
            efg_std_l10     = _safe(row, "efg_std_l10",     5.0),
            consistency_score = _safe(row, "consistency_score", 50.0),

            suffocation   = _safe(row, "suffocation",   50.0),
            momentum      = _safe(row, "momentum",      50.0),
            clutch_rating = _safe(row, "clutch_rating", 50.0),
            dna_score     = _safe(row, "dna_score",     50.0),
            floor_em      = _safe(row, "floor_em",     -8.0),
            ceiling_em    = _safe(row, "ceiling_em",    8.0),
            star_risk     = _safe(row, "star_risk",    50.0),
            regression_risk = int(_safe(row, "regression_risk", 0)),

            # P2 FIX: luck_score (not luck) — matches espn_rankings.py output
            luck          = _safe(row, "luck_score",       _safe(row, "luck", 0.0)),
            pythagorean_win_pct = _safe(row, "expected_win_pct", overall_wpct),
            actual_win_pct      = _safe(row, "actual_win_pct",   overall_wpct),
            home_wpct    = home_wpct,
            away_wpct    = away_wpct,
            close_wpct   = _safe(row, "close_wpct", 0.5),
            rest_days    = 3.0,   # populated at prediction time
            games_l7     = 2.0,
            win_streak   = 0.0,

            opp_avg_net_rtg = _safe(row, "sos", 0.0),
        )
        profiles[tid] = p

    log.info(f"Loaded {len(profiles)} team profiles from rankings")
    return profiles


def _load_from_snapshot(snapshot_path: Path = OUT_TOURNAMENT_SNAPSHOT) -> Dict[str, "TeamProfile"]:
    """Fallback: load from tournament snapshot (fewer CAGE columns but same shape)."""
    if not snapshot_path.exists():
        log.error("Neither rankings nor snapshot CSV found — cannot load profiles")
        return {}
    df = pd.read_csv(snapshot_path, dtype=str, low_memory=False)
    str_cols = {"team_id", "team", "conference", "t_offensive_archetype"}
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    profiles = {}
    for _, row in df.iterrows():
        tid = str(row.get("team_id", ""))
        p = TeamProfile(
            team_id   = tid,
            team_name = str(row.get("team", "")),
            conference= str(row.get("conference", "")),
            cage_em   = _safe(row, "adj_net_rtg",  0.0),
            cage_o    = _safe(row, "adj_ortg",      LEAGUE_AVG_ORTG),
            cage_d    = _safe(row, "adj_drtg",      LEAGUE_AVG_DRTG),
            cage_t    = _safe(row, "adj_pace",      LEAGUE_AVG_PACE),
            barthag   = _safe(row, "barthag",       0.5),
            luck      = _safe(row, "luck_score",    0.0),  # P2 FIX: luck_score
            suffocation   = _safe(row, "t_suffocation_rating",      50.0),
            momentum      = _safe(row, "t_momentum_quality_rating", 50.0),
            clutch_rating = _safe(row, "clutch_rating",             50.0),
            consistency_score = _safe(row, "consistency_score",     50.0),
            floor_em   = _safe(row, "floor_em",   -8.0),
            ceiling_em = _safe(row, "ceiling_em",  8.0),
            net_rtg_l5 = _safe(row, "net_rtg_l5",  0.0),
            net_rtg_l10= _safe(row, "net_rtg_l10", 0.0),
        )
        profiles[tid] = p

    log.info(f"Loaded {len(profiles)} profiles from snapshot (fallback)")
    return profiles


def load_team_profiles(rankings_path: Path = OUT_RANKINGS) -> Dict[str, "TeamProfile"]:
    """Public loader — tries rankings first, falls back to snapshot."""
    if rankings_path.exists():
        return _load_from_rankings(rankings_path)
    return _load_from_snapshot()


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_ensemble_result(result: EnsemblePrediction) -> None:
    fav = result.home_team if result.spread <= 0 else result.away_team
    dog = result.away_team if result.spread <= 0 else result.home_team
    fav_spread = abs(result.spread)

    print()
    print("=" * 72)
    print(f"  {result.home_team}  vs  {result.away_team}")
    site = "NEUTRAL SITE" if result.neutral_site else "HOME: " + result.home_team
    print(f"  {site}")
    print("=" * 72)
    print(f"  ENSEMBLE SPREAD:  {fav} -{fav_spread:.1f}  (spread: {result.spread:+.1f})")
    print(f"  ENSEMBLE TOTAL:   {result.total:.1f}  ({result.home_team} {result.home_score} — {result.away_team} {result.away_score})")
    print(f"  CONFIDENCE:       {result.confidence:.0%}")
    print(f"  MODEL AGREEMENT:  {result.model_agreement}  (spread std = {result.spread_std:.1f} pts)")
    if result.line_value:
        print(f"  LINE VALUE:       {result.line_value}")
    print()
    print(f"  CAGE EDGE (raw):  {result.cage_edge:+.1f} pts  |  BARTHAG DIFF: {result.barthag_diff:+.4f}")
    print()
    print(f"  {'MODEL':<18} {'SPREAD':>8} {'TOTAL':>8} {'CONF':>7}")
    print(f"  {'-'*46}")
    for mp in result.model_predictions:
        print(f"  {mp.model_name:<18} {mp.spread:>+8.1f} {mp.total:>8.1f} {mp.confidence:>7.0%}")
    print(f"  {'─'*46}")
    print(f"  {'ENSEMBLE':<18} {result.spread:>+8.1f} {result.total:>8.1f} {result.confidence:>7.0%}")
    print(f"  Spread range: {result.spread_lo:+.1f} to {result.spread_hi:+.1f}")
    print(f"  Total range:  {result.total_lo:.1f} to {result.total_hi:.1f}")
    print("=" * 72)
    print()


def results_to_csv(results: List[EnsemblePrediction], path: Path) -> None:
    if not results:
        return
    rows = [r.to_flat_dict() for r in results]
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info(f"Wrote {len(rows)} ensemble predictions → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CBB Ensemble Prediction Engine")
    parser.add_argument("--home-id",   type=str,   help="Home team ESPN team_id")
    parser.add_argument("--away-id",   type=str,   help="Away team ESPN team_id")
    parser.add_argument("--home-name", type=str,   help="Home team name (fuzzy match)")
    parser.add_argument("--away-name", type=str,   help="Away team name (fuzzy match)")
    parser.add_argument("--neutral",   action="store_true")
    parser.add_argument("--spread-line", type=float, default=None)
    parser.add_argument("--total-line",  type=float, default=None)
    parser.add_argument("--rankings-csv", type=Path, default=OUT_RANKINGS)
    parser.add_argument("--output",    type=Path,   default=None)
    args = parser.parse_args()

    profiles = load_team_profiles(args.rankings_csv)
    if not profiles:
        log.error("No team profiles loaded — run espn_rankings.py first")
        return

    def _find_team(team_id: Optional[str], name: Optional[str]) -> Optional[TeamProfile]:
        if team_id and team_id in profiles:
            return profiles[team_id]
        if name:
            # P6 FIX: prefer exact word-boundary match over broad substring
            name_lower = name.lower().strip()
            # Pass 1: exact full-name match
            for p in profiles.values():
                if p.team_name.lower().strip() == name_lower:
                    return p
            # Pass 2: name is a complete word within team_name
            import re
            for p in profiles.values():
                if re.search(r"\b" + re.escape(name_lower) + r"\b", p.team_name.lower()):
                    return p
            # Pass 3: substring fallback (original behavior) — warns on ambiguity
            matches = [p for p in profiles.values() if name_lower in p.team_name.lower()]
            if len(matches) == 1:
                return matches[0]
            elif len(matches) > 1:
                log.warning(f"Ambiguous team name '{name}' matches {len(matches)} teams: "
                            f"{[m.team_name for m in matches[:5]]}. Using first match.")
                return matches[0]
        return None

    home = _find_team(args.home_id, args.home_name)
    away = _find_team(args.away_id, args.away_name)

    if not home or not away:
        log.info("Teams not found — running demo with synthetic profiles")
        home = TeamProfile(
            team_id="demo_1", team_name="Elite 1-Seed",
            cage_em=22.5, cage_o=118.0, cage_d=95.5, cage_t=69.0,
            barthag=0.940, cage_power_index=88.0, resume_score=82.0,
            suffocation=78.0, momentum=65.0, clutch_rating=72.0,
            consistency_score=71.0, floor_em=14.0, ceiling_em=31.0,
            luck=-0.02, net_rtg_l5=24.1, net_rtg_l10=22.8,
            ortg_l5=118.5, ortg_l10=117.8, drtg_l5=94.4, drtg_l10=95.0,
            pace_l5=68.8, pace_l10=69.1, three_pct_l5=37.2, three_pct_l10=36.1,
            home_wpct=0.88, away_wpct=0.72, close_wpct=0.67, win_streak=5.0,
        )
        away = TeamProfile(
            team_id="demo_2", team_name="Scrappy 9-Seed",
            cage_em=3.2, cage_o=107.5, cage_d=104.3, cage_t=71.5,
            barthag=0.558, cage_power_index=52.0, resume_score=48.0,
            suffocation=51.0, momentum=70.0, clutch_rating=61.0,
            consistency_score=44.0, floor_em=-4.0, ceiling_em=10.5,
            luck=0.06, net_rtg_l5=6.8, net_rtg_l10=3.1,
            ortg_l5=110.2, ortg_l10=107.8, drtg_l5=103.4, drtg_l10=104.7,
            pace_l5=72.1, pace_l10=71.6, three_pct_l5=39.1, three_pct_l10=33.8,
            home_wpct=0.71, away_wpct=0.42, close_wpct=0.58, win_streak=3.0,
            rest_days=4.0,
        )

    predictor = EnsemblePredictor()
    result    = predictor.predict(
        home, away,
        neutral=args.neutral,
        spread_line=args.spread_line,
        total_line=args.total_line,
    )

    print_ensemble_result(result)

    if args.output:
        results_to_csv([result], args.output)
    else:
        dated = DATA_DIR / f"ensemble_predictions_{datetime.now(TZ).strftime('%Y%m%d')}.csv"
        DATA_DIR.mkdir(exist_ok=True)
        results_to_csv([result], dated)


if __name__ == "__main__":
    main()
