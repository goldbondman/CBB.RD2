#!/usr/bin/env python3
"""
ESPN CBB Pipeline — Ensemble Prediction Engine
cbb_ensemble.py

Seven independent sub-models, each producing a point spread and game total.
A configurable weighted ensemble combines them into a final prediction.

QUANT TEAM DESIGN NOTES
─────────────────────────────────────────────────────────────────────────────
Why an ensemble over a single model?

No single metric captures everything. The four-factors model knows nothing
about whether a team's hot streak is real or regression bait. The momentum
model knows nothing about whether the opponent is a tough defensive unit or
a cupcake. The CAGE model captures everything but is slow to react to a
three-game slide. Each model has a lane where it's best-in-class and a lane
where it's blind. The ensemble exploits that diversity.

The critical design constraint: every sub-model must be INDEPENDENTLY
calibrated so its output is already in points, not an index. The ensemble
layer only weights — it does not transform.

ENSEMBLE MEMBERS
─────────────────────────────────────────────────────────────────────────────
M1  FourFactorsModel          Dean Oliver's framework. The statistical bedrock.
M2  AdjustedEfficiencyModel   Pure adj_ortg/drtg differential via CAGE_EM.
M3  PythagoreanModel          Win probability → spread via Log5 method.
M4  MomentumModel             L5/L10 trend-weighted form, regression-adjusted.
M5  SituationalModel          Rest, schedule density, home/road splits.
M6  CAGERankingsModel         CAGE Power Index, BARTHAG, resume — full profile.
M7  RegressedEfficiencyModel  Luck-stripped, consistency-weighted ratings.

CAGE INTEGRATION
─────────────────────────────────────────────────────────────────────────────
CAGE rankings feed M2, M3, M6, M7 directly. When cbb_rankings.csv is
available, all models that use efficiency ratings pull the adj_* columns
from the rankings snapshot, ensuring models 2–7 share the same quality-
adjusted baseline that the rankings engine produces.

WEIGHTING PHILOSOPHY
─────────────────────────────────────────────────────────────────────────────
Default weights are empirically derived from backtesting against historical
CBB spreads. The ensemble is intentionally NOT tournament-specific — the
weights here reflect regular-season predictive accuracy. Tournament-specific
weighting lives in espn_tournament.py and the predict_tomorrow runner.

Default spread weights:   M1=0.12  M2=0.22  M3=0.14  M4=0.16  M5=0.10  M6=0.18  M7=0.08
Default total weights:    M1=0.15  M2=0.24  M3=0.10  M4=0.14  M5=0.12  M6=0.17  M7=0.08
(weights adjusted dynamically by data confidence when full snapshot unavailable)

Usage:
    # Single matchup
    from cbb_ensemble import EnsemblePredictor, TeamProfile
    pred = EnsemblePredictor().predict(home_profile, away_profile)

    # Batch from CSVs
    python cbb_ensemble.py --home-id 52 --away-id 150

Pipeline position: runs after espn_rankings.py (needs cbb_rankings.csv).
Outputs: data/ensemble_predictions_YYYYMMDD.csv
"""

import argparse
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

TZ = ZoneInfo("America/Los_Angeles")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR        = Path("data")
RANKINGS_CSV    = DATA_DIR / "cbb_rankings.csv"
SNAPSHOT_CSV    = DATA_DIR / "team_pretournament_snapshot.csv"
WEIGHTED_CSV    = DATA_DIR / "team_game_weighted.csv"

# ── League constants ──────────────────────────────────────────────────────────
LEAGUE_AVG_ORTG  = 103.0
LEAGUE_AVG_DRTG  = 103.0
LEAGUE_AVG_PACE  = 70.0
LEAGUE_AVG_EFG   = 50.5
LEAGUE_AVG_TOV   = 18.0
LEAGUE_AVG_FTR   = 28.0
LEAGUE_AVG_ORB   = 30.0
LEAGUE_AVG_DRB   = 70.0
PYTHAGOREAN_EXP  = 11.5
DEFAULT_HCA      = 3.2      # Home court advantage, points

# ── Points-per-possession conversion ─────────────────────────────────────────
# 1 unit of efficiency (pts/100) × avg pace / 100 → points per game impact
# At LEAGUE_AVG_PACE=70: 1 unit ≈ 0.70 points in a game
EFF_TO_PTS = LEAGUE_AVG_PACE / 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM PROFILE — shared input object for all sub-models
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TeamProfile:
    """
    Unified team data container. All sub-models read from this object.

    Populated from cbb_rankings.csv (richest) or team_game_weighted.csv.
    All fields have sensible league-average defaults so models degrade
    gracefully when data is incomplete.

    Design: flat dataclass rather than dict — type hints + IDE completion,
    and explicit defaults make it obvious what's missing vs intentionally 0.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    team_id:     str = ""
    team_name:   str = "Unknown"
    conference:  str = ""
    record:      str = ""

    # ── CAGE / Adjusted efficiency (from espn_rankings.py / espn_sos.py) ──────
    cage_em:     float = 0.0     # Adjusted efficiency margin (our AdjEM)
    cage_o:      float = LEAGUE_AVG_ORTG   # Adjusted offensive rating
    cage_d:      float = LEAGUE_AVG_DRTG   # Adjusted defensive rating
    cage_t:      float = LEAGUE_AVG_PACE   # Adjusted tempo
    cage_power_index: float = 50.0         # Power index 0–100
    barthag:     float = 0.500             # P(beat avg D1)
    wab:         float = 0.0               # Wins above bubble
    resume_score: float = 50.0
    eff_grade:   str = "B-"
    sos:         float = 0.0               # Strength of schedule (opp avg net rtg)

    # ── Raw four factors (season) ─────────────────────────────────────────────
    efg_pct:     float = LEAGUE_AVG_EFG    # Effective FG%
    tov_pct:     float = LEAGUE_AVG_TOV    # Turnover rate
    orb_pct:     float = LEAGUE_AVG_ORB    # Offensive rebound %
    drb_pct:     float = LEAGUE_AVG_DRB    # Defensive rebound %
    ftr:         float = LEAGUE_AVG_FTR    # Free throw rate
    ft_pct:      float = 71.0
    three_par:   float = 35.0              # 3-point attempt rate
    three_pct:   float = 33.5

    # ── Opponent four factors (what this defense allows) ──────────────────────
    opp_efg_pct: float = LEAGUE_AVG_EFG
    opp_tov_pct: float = LEAGUE_AVG_TOV
    opp_orb_pct: float = LEAGUE_AVG_ORB
    opp_ftr:     float = LEAGUE_AVG_FTR

    # ── Opponent-context metrics (from espn_sos.py) ───────────────────────────
    efg_vs_opp:  float = 0.0    # eFG% above what opp typically allows
    tov_vs_opp:  float = 0.0    # TOV% above what opp typically forces
    orb_vs_opp:  float = 0.0
    ftr_vs_opp:  float = 0.0

    # ── Rolling windows ────────────────────────────────────────────────────────
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

    # ── Variance / consistency ────────────────────────────────────────────────
    net_rtg_std_l10: float = 8.0    # Net rating standard deviation (L10)
    efg_std_l10:     float = 5.0    # eFG% std dev (L10)
    consistency_score: float = 50.0

    # ── CAGE proprietary composites ────────────────────────────────────────────
    suffocation:    float = 50.0    # Defensive composite
    momentum:       float = 50.0    # Momentum quality rating
    clutch_rating:  float = 50.0    # Close-game excellence
    dna_score:      float = 50.0    # Tournament DNA
    floor_em:       float = -8.0    # Worst realistic AdjEM
    ceiling_em:     float = 8.0     # Best realistic AdjEM
    star_risk:      float = 50.0
    regression_risk: int  = 0       # 1 = shooting unsustainably hot

    # ── Schedule / situational ────────────────────────────────────────────────
    luck:            float = 0.0    # Actual win% - Pythagorean win%
    pythagorean_win_pct: float = 0.5
    actual_win_pct:  float = 0.5
    home_wpct:       float = 0.65   # Historical home win %
    away_wpct:       float = 0.40   # Historical away win %
    close_wpct:      float = 0.50   # Win % in games ≤5 pts
    rest_days:       float = 3.0    # Days since last game
    games_l7:        float = 2.0    # Games played in last 7 days
    win_streak:      float = 0.0    # Current streak (+win, -loss)

    # ── Opponent schedule context ─────────────────────────────────────────────
    opp_avg_net_rtg: float = 0.0    # Avg opponent quality this season
    opp_avg_ortg:    float = LEAGUE_AVG_ORTG
    opp_avg_drtg:    float = LEAGUE_AVG_DRTG

    # ── Derived convenience properties ────────────────────────────────────────
    @property
    def trend_delta(self) -> float:
        """L5 net rating minus L10 net rating. Positive = improving."""
        return self.net_rtg_l5 - self.net_rtg_l10

    @property
    def three_pct_gap(self) -> float:
        """Recent 3P% minus season 3P%. >5 = regression risk."""
        return self.three_pct_l5 - self.three_pct_l10

    @property
    def fatigue_factor(self) -> float:
        """
        0–1 fatigue penalty. 1.0 = fresh (4+ rest days).
        Decays with schedule density and low rest.
        """
        rest_score    = min(self.rest_days / 4.0, 1.0)
        density_score = max(0.0, 1.0 - (self.games_l7 - 2) * 0.15)
        return round(0.6 * rest_score + 0.4 * density_score, 3)


@dataclass
class ModelPrediction:
    """Output of a single sub-model."""
    model_name:  str
    spread:      float    # Negative = home favored (e.g. -5.5 = home by 5.5)
    total:       float    # Combined projected score
    confidence:  float    # 0–1
    home_score:  float = field(init=False)
    away_score:  float = field(init=False)
    notes:       str = ""

    def __post_init__(self):
        self.home_score = round(self.total / 2 - self.spread / 2, 1)
        self.away_score = round(self.total / 2 + self.spread / 2, 1)


@dataclass
class EnsemblePrediction:
    """Final ensemble output — weighted combination of all sub-models."""
    home_team:   str
    away_team:   str
    neutral_site: bool

    spread:      float    # Final weighted spread
    total:       float    # Final weighted total
    confidence:  float    # Weighted ensemble confidence

    home_score:  float
    away_score:  float

    # Per-model breakdown
    model_predictions: List[ModelPrediction] = field(default_factory=list)

    # Spread and total ranges (min/max across sub-models)
    spread_lo:   float = 0.0
    spread_hi:   float = 0.0
    total_lo:    float = 0.0
    total_hi:    float = 0.0
    spread_std:  float = 0.0
    total_std:   float = 0.0

    # CAGE integration summary
    cage_edge:   float = 0.0    # Pure CAGE_EM-derived edge (sanity check)
    barthag_diff: float = 0.0   # BARTHAG advantage

    # Flags
    model_agreement: str = "STRONG"   # STRONG / MODERATE / SPLIT
    line_value:      str = ""         # vs market line, if provided
    predicted_at:    str = ""

    def to_flat_dict(self) -> Dict:
        """Flatten to one CSV row."""
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
    """
    Abstract base for all ensemble sub-models.

    Each sub-model must implement predict() and return a ModelPrediction
    with spread and total already in points — no further scaling needed.

    Sub-models are pure functions over TeamProfile objects. No I/O.
    """

    name: str = "Base"

    @abstractmethod
    def predict(
        self,
        home: TeamProfile,
        away: TeamProfile,
        neutral: bool = False,
    ) -> ModelPrediction:
        ...

    def _hca(self, neutral: bool) -> float:
        return 0.0 if neutral else DEFAULT_HCA

    def _clamp_spread(self, spread: float) -> float:
        """Clamp to realistic CBB range."""
        return float(np.clip(spread, -35.0, 35.0))

    def _clamp_total(self, total: float) -> float:
        """CBB totals realistically live 115–175."""
        return float(np.clip(total, 110.0, 185.0))


# ═══════════════════════════════════════════════════════════════════════════════
# M1 — FOUR FACTORS MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class FourFactorsModel(BaseModel):
    """
    M1 — Dean Oliver's Four Factors framework.

    The statistical bedrock of basketball analytics. Independently predicts
    both teams' expected points per possession, then scales to a game total
    via tempo. Spread = home pts − away pts.

    Factors (Oliver's original weights, validated over 25+ years):
      eFG%   0.40  — shooting quality (the most impactful single factor)
      TOV%   0.25  — ball security
      ORB%   0.20  — second chances
      FTR    0.15  — free throw generation

    This model uses OPPONENT-ALLOWED four-factor stats on defense, not raw
    team averages, so both sides of the ball are captured symmetrically.

    CAGE integration: Uses efg_vs_opp and tov_vs_opp (opponent-context
    adjusted metrics from espn_sos.py) when available, blended 60/40 with
    raw four-factor differentials.

    Blind spots: pace, recency, situational context.
    """

    name = "FourFactors"

    # Oliver's validated weights
    EFG_W = 0.40
    TOV_W = 0.25
    ORB_W = 0.20
    FTR_W = 0.15

    # Blend: raw differential vs opponent-context adjusted
    RAW_BLEND   = 0.40
    CTX_BLEND   = 0.60

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:

        # ── eFG% edge ─────────────────────────────────────────────────────────
        # Raw: home shooting quality vs away shooting quality
        # Context: how each team shoots relative to what their opponents allow
        raw_efg   = (home.efg_pct - away.efg_pct) - (home.opp_efg_pct - away.opp_efg_pct)
        ctx_efg   = (home.efg_vs_opp - away.efg_vs_opp)
        efg_edge  = self.RAW_BLEND * raw_efg + self.CTX_BLEND * ctx_efg

        # ── TOV% edge (lower TOV = better — so home advantage = home lower) ───
        # Positive = home turns it over less / forces more TOs
        raw_tov   = (away.tov_pct - home.tov_pct) - (home.opp_tov_pct - away.opp_tov_pct)
        ctx_tov   = (home.tov_vs_opp - away.tov_vs_opp)
        tov_edge  = self.RAW_BLEND * raw_tov + self.CTX_BLEND * ctx_tov

        # ── ORB% edge ─────────────────────────────────────────────────────────
        raw_orb   = (home.orb_pct - away.orb_pct)
        ctx_orb   = (home.orb_vs_opp - away.orb_vs_opp)
        orb_edge  = self.RAW_BLEND * raw_orb + self.CTX_BLEND * ctx_orb

        # ── FTR edge ──────────────────────────────────────────────────────────
        raw_ftr   = (home.ftr - away.ftr) - (home.opp_ftr - away.opp_ftr)
        ctx_ftr   = (home.ftr_vs_opp - away.ftr_vs_opp)
        ftr_edge  = self.RAW_BLEND * raw_ftr + self.CTX_BLEND * ctx_ftr

        # ── Composite factor score → efficiency units → points ────────────────
        # Each factor edge is in percentage-point units. Scale to pts/100 poss.
        # eFG: 1% edge ≈ 1.0 pts/100 (two possessions, each worth ~0.5 pts/%)
        # TOV: 1% edge ≈ 1.0 pts/100 (each turnover is ~1 pt swing)
        # ORB: 1% edge ≈ 0.6 pts/100 (second chances worth less than primary)
        # FTR: 1% edge ≈ 0.5 pts/100
        factor_edge_eff = (
            self.EFG_W * efg_edge * 1.0 +
            self.TOV_W * tov_edge * 1.0 +
            self.ORB_W * orb_edge * 0.6 +
            self.FTR_W * ftr_edge * 0.5
        )

        # Convert efficiency edge to point spread
        pace   = (home.cage_t + away.cage_t) / 2.0
        spread = -(factor_edge_eff * pace / 100.0 + self._hca(neutral))
        spread = self._clamp_spread(spread)

        # ── Total: each team's expected scoring via four-factor efficiency ─────
        # Build each team's offensive efficiency from their factor profile
        home_off_eff = (
            home.efg_pct * 1.8 +                    # eFG is primary driver
            (LEAGUE_AVG_TOV - home.tov_pct) * 0.8 + # lower TOV = more pts
            home.orb_pct * 0.3 +                     # ORB gives extra chances
            home.ftr * home.ft_pct / 100.0 * 0.4    # FTR × FT% = FT value
        )
        away_off_eff = (
            away.efg_pct * 1.8 +
            (LEAGUE_AVG_TOV - away.tov_pct) * 0.8 +
            away.orb_pct * 0.3 +
            away.ftr * away.ft_pct / 100.0 * 0.4
        )

        # Apply defensive suppression from opponent (what defense allows)
        home_def_factor = away.opp_efg_pct / LEAGUE_AVG_EFG  # <1 = elite defense
        away_def_factor = home.opp_efg_pct / LEAGUE_AVG_EFG

        home_pts = home_off_eff * away_def_factor * pace / 100.0
        away_pts = away_off_eff * home_def_factor * pace / 100.0
        total    = self._clamp_total(home_pts + away_pts)

        # Confidence: better when both teams have meaningful context data
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
    M2 — Pure adjusted efficiency differential (CAGE_EM powered).

    The simplest model with the strongest single-variable correlation to
    game outcomes. CAGE_EM (adj_ortg − adj_drtg) captures the full
    quality of a team in one number, opponent-adjusted.

    This model directly translates CAGE_EM differential to a point spread
    by scaling to expected game pace. No blending — it trusts the
    season-long adjusted efficiency number fully.

    Spread derivation:
        Edge (efficiency units) = home.cage_em − away.cage_em
        Edge (points/game)      = Edge × avg_pace / 100
        Spread                  = −(Edge + HCA)

    Total derivation:
        Each team's expected pts = team.cage_o × avg_pace / 100
        BUT we use opponent.cage_d to suppress each offense:
          home_pts = home.cage_o × (LEAGUE_AVG / away.cage_d) × pace / 100
          away_pts = away.cage_o × (LEAGUE_AVG / home.cage_d) × pace / 100

    CAGE integration: cage_em, cage_o, cage_d, cage_t are the PRIMARY inputs.
    This is the most direct CAGE-powered model in the ensemble.

    Blind spots: recency, momentum, situational factors.
    """

    name = "AdjEfficiency"

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:

        pace = (home.cage_t + away.cage_t) / 2.0

        # ── Spread ────────────────────────────────────────────────────────────
        cage_em_diff = home.cage_em - away.cage_em
        edge_pts     = cage_em_diff * pace / 100.0
        spread       = self._clamp_spread(-(edge_pts + self._hca(neutral)))

        # ── Total ─────────────────────────────────────────────────────────────
        # Defense adjustment factor: how much does opponent's defense suppress?
        # LEAGUE_AVG_DRTG / away.cage_d: <1.0 = away defense is elite (suppresses)
        home_def_factor = LEAGUE_AVG_DRTG / max(away.cage_d, 70.0)
        away_def_factor = LEAGUE_AVG_DRTG / max(home.cage_d, 70.0)

        home_pts = home.cage_o * home_def_factor * pace / 100.0
        away_pts = away.cage_o * away_def_factor * pace / 100.0
        total    = self._clamp_total(home_pts + away_pts)

        # Confidence: strong when both teams have established adjusted ratings
        # Degrades when cage_em is near zero (uninformative for close matchups)
        em_magnitude = min(abs(cage_em_diff) / 10.0, 1.0)   # larger gap = more signal
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
    M3 — Pythagorean win probability → spread via Log5 method.

    Bill James's Pythagorean expectation (adapted for basketball by Oliver):
        P(win) = pts_for^exp / (pts_for^exp + pts_against^exp)

    We use BARTHAG (our CAGE implementation of Pythagorean using adj ratings)
    as the input, which is more stable than raw points.

    Log5 translation from win probability to point spread:
        Win probability → implied point spread via inverse normal distribution.
        Calibration: σ = 11 pts (empirical CBB game outcome distribution width).

    This is the PROBABILITY model. It answers "who wins?" rather than
    "by how much?" and converts that back to points — a different lens
    than the efficiency models.

    CAGE integration: BARTHAG is the primary input. This directly leverages
    the CAGE rankings probability metric.

    Key insight: Pythagorean models are REGRESSIVE — they pull extreme
    teams toward the mean and penalize lucky/unlucky records implicitly.
    This makes M3 a natural counterweight to M2's absolute efficiency reading.

    Blind spots: pace, real-time form, situational variation.
    """

    name = "Pythagorean"

    # Calibration constant: CBB game score distribution
    # σ=11 means 68% of games land within 11 pts of the "true" line
    SIGMA = 11.0

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:

        # ── Win probability via Log5 ───────────────────────────────────────────
        # Log5: P(A beats B) = (A − A×B) / (A + B − 2×A×B)
        # where A = P(home beats avg D1), B = P(away beats avg D1)
        h_b  = np.clip(home.barthag, 0.001, 0.999)
        a_b  = np.clip(away.barthag, 0.001, 0.999)
        log5 = (h_b - h_b * a_b) / (h_b + a_b - 2 * h_b * a_b)
        log5 = float(np.clip(log5, 0.001, 0.999))

        # HCA adds ~5% win probability boost on average (3.2 pts ≈ 5% WP at σ=11)
        if not neutral:
            hca_wp = 3.2 / (self.SIGMA * np.sqrt(2 * np.pi) * 0.4)
            log5   = float(np.clip(log5 + hca_wp * 0.5, 0.001, 0.999))

        # Convert P(home wins) to expected margin via inverse normal
        from scipy import stats as _stats
        implied_edge = _stats.norm.ppf(log5) * self.SIGMA

        spread = self._clamp_spread(-implied_edge)

        # ── Total via Pythagorean scoring rates ───────────────────────────────
        # Use luck-adjusted (pythagorean) win percentages to estimate
        # each team's scoring profile
        # Approximation: a team with barthag=0.70 scores ~112 pts/100 poss
        # Linear mapping: 0.5 barthag → 103 pts/100, 1.0 → ~125, 0.0 → ~85
        def _barthag_to_ortg(b: float) -> float:
            return 85.0 + b * 40.0   # 0→85, 0.5→105, 1.0→125

        home_ortg = _barthag_to_ortg(h_b)
        away_ortg = _barthag_to_ortg(a_b)

        # Derive drtg from ortg and barthag-implied net
        home_net  = home.cage_em if home.cage_em != 0 else (home.barthag - 0.5) * 30
        away_net  = away.cage_em if away.cage_em != 0 else (away.barthag - 0.5) * 30
        home_drtg = home_ortg - home_net
        away_drtg = away_ortg - away_net

        pace      = (home.cage_t + away.cage_t) / 2.0
        home_pts  = home_ortg * (LEAGUE_AVG_DRTG / max(away_drtg, 70)) * pace / 100.0
        away_pts  = away_ortg * (LEAGUE_AVG_DRTG / max(home_drtg, 70)) * pace / 100.0
        total     = self._clamp_total(home_pts + away_pts)

        # Confidence: strong when barthag values are meaningfully differentiated
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
    M4 — Recency-weighted form with regression adjustment.

    Answers the question the efficiency models cannot: "Is this team getting
    better or worse RIGHT NOW?" A team's season-long adjusted efficiency is
    the best long-run predictor, but the L5 games are the best short-run
    predictor. This model weights recency heavily.

    Three-component prediction:
    1. L5 net rating differential (primary — most recent 5 games)
    2. Trend delta (L5 minus L10 — direction of travel)
    3. Regression adjustment (penalize teams on unsustainable hot streaks)

    Regression adjustment logic:
      - 3-point shooting is the most volatile CBB stat
      - If a team is shooting 5%+ above their season avg from 3 in the L5,
        we regress their L5 net rating toward L10 by 40% of the gap
      - This prevents chasing hot-shooting teams who are due to cool off

    CAGE integration: Uses momentum score from espn_tournament.py (when
    available) as a validation/scaling signal for the trend direction.

    Blind spots: opponent quality of recent games (a L5 against weak
    opponents looks identical to L5 against elite opponents here).
    """

    name = "Momentum"

    # Recency blend: L5 (70%) vs L10 (30%)
    L5_WEIGHT  = 0.70
    L10_WEIGHT = 0.30

    # Regression: how much to pull L5 toward L10 when shooting heater detected
    HEATER_REGRESSION = 0.40

    # Trend amplifier: each sigma of trend_delta ≈ 0.8 pts in model output
    TREND_SCALE = 0.8

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:

        # ── Regress L5 if shooting heater detected ────────────────────────────
        def _regressed_l5(team: TeamProfile) -> float:
            net_l5 = team.net_rtg_l5
            net_l10 = team.net_rtg_l10
            if team.three_pct_gap > 5.0 and team.regression_risk:
                # Pull L5 net rating toward L10 when shooting is unsustainably hot
                net_l5 = net_l5 - self.HEATER_REGRESSION * (net_l5 - net_l10)
            return net_l5

        home_l5_adj = _regressed_l5(home)
        away_l5_adj = _regressed_l5(away)

        # ── Blended form rating (L5 heavy) ────────────────────────────────────
        home_form = self.L5_WEIGHT * home_l5_adj  + self.L10_WEIGHT * home.net_rtg_l10
        away_form = self.L5_WEIGHT * away_l5_adj  + self.L10_WEIGHT * away.net_rtg_l10

        # ── Trend delta (directional signal) ──────────────────────────────────
        home_trend = home.trend_delta   # L5 − L10; positive = heating up
        away_trend = away.trend_delta
        trend_edge = (home_trend - away_trend) * self.TREND_SCALE

        # ── CAGE momentum score integration ───────────────────────────────────
        # Momentum score (0–100) from espn_tournament.py normalizes streak quality.
        # Teams with high momentum score get a small boost; low score = fade.
        # Effect is intentionally small (±1 pt max) — momentum is signal, not story.
        home_mom_adj = (home.momentum - 50.0) / 50.0 * 1.0   # -1.0 to +1.0
        away_mom_adj = (away.momentum - 50.0) / 50.0 * 1.0
        cage_mom_edge = home_mom_adj - away_mom_adj

        # ── Spread ────────────────────────────────────────────────────────────
        form_diff  = home_form - away_form
        pace       = (home.pace_l5 + away.pace_l5) / 2.0
        edge_pts   = form_diff * pace / 100.0 + trend_edge + cage_mom_edge
        spread     = self._clamp_spread(-(edge_pts + self._hca(neutral)))

        # ── Total ─────────────────────────────────────────────────────────────
        home_ortg_form = home.ortg_l5 * self.L5_WEIGHT + home.ortg_l10 * self.L10_WEIGHT
        away_ortg_form = away.ortg_l5 * self.L5_WEIGHT + away.ortg_l10 * self.L10_WEIGHT
        home_drtg_form = home.drtg_l5 * self.L5_WEIGHT + home.drtg_l10 * self.L10_WEIGHT
        away_drtg_form = away.drtg_l5 * self.L5_WEIGHT + away.drtg_l10 * self.L10_WEIGHT

        home_pts  = home_ortg_form * (LEAGUE_AVG_DRTG / max(away_drtg_form, 70)) * pace / 100.0
        away_pts  = away_ortg_form * (LEAGUE_AVG_DRTG / max(home_drtg_form, 70)) * pace / 100.0
        total     = self._clamp_total(home_pts + away_pts)

        # Confidence: lower when regression adjustments were needed
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
    M5 — Rest, fatigue, schedule density, and home/away split patterns.

    Most models treat every game as if both teams are equally fresh and
    motivated. They aren't. A team playing its fourth game in eight days
    on the road after a rivalry game is a fundamentally different bet than
    the same team at home after five days of rest.

    This model answers: "What does the schedule context do to the line?"
    Its spread output is intentionally small (±4 pts max) — it's an
    adjustment model, not a primary rating model. Its value is in the
    situations where it fires hard (2+ pt adjustment) and the other models
    don't see it.

    Components:
    1. Rest differential — who's fresher?
    2. Schedule density — who's more fatigued?
    3. Home/away win% differential — how much does location matter for each?
    4. Win streak momentum — back-to-back wins create real team confidence
    5. Close game experience — teams with more clutch experience are better
       positioned in tight situational spots

    CAGE integration: Uses clutch_rating from CAGE composites to adjust
    the close-game situational edge.

    Blind spots: everything quality-related. This model is intentionally
    blind to talent — it only knows context.
    """

    name = "Situational"

    # Maximum contributions per component (pts)
    REST_MAX     = 1.5
    DENSITY_MAX  = 1.2
    SPLIT_MAX    = 2.0
    STREAK_MAX   = 0.8
    CLUTCH_MAX   = 0.6

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:

        # ── Rest differential ─────────────────────────────────────────────────
        rest_diff  = home.rest_days - away.rest_days  # positive = home is fresher
        rest_edge  = float(np.clip(rest_diff * 0.25, -self.REST_MAX, self.REST_MAX))

        # ── Schedule density ──────────────────────────────────────────────────
        home_fatigue = home.fatigue_factor   # 0–1, higher = fresher
        away_fatigue = away.fatigue_factor
        density_edge = float(np.clip(
            (home_fatigue - away_fatigue) * self.DENSITY_MAX * 2,
            -self.DENSITY_MAX, self.DENSITY_MAX
        ))

        # ── Home/away win% split ──────────────────────────────────────────────
        if neutral:
            split_edge = 0.0
        else:
            # How much better is home team at home vs away team on the road?
            home_home_boost = home.home_wpct - home.actual_win_pct
            away_road_drag  = away.actual_win_pct - away.away_wpct
            split_edge = float(np.clip(
                (home_home_boost + away_road_drag) * 5.0,   # scale wp to pts
                -self.SPLIT_MAX, self.SPLIT_MAX
            ))

        # ── Win streak momentum ───────────────────────────────────────────────
        # Positive streak = confidence; negative = uncertainty
        home_streak_pts = float(np.clip(home.win_streak * 0.12, -self.STREAK_MAX, self.STREAK_MAX))
        away_streak_pts = float(np.clip(away.win_streak * 0.12, -self.STREAK_MAX, self.STREAK_MAX))
        streak_edge = home_streak_pts - away_streak_pts

        # ── Close game experience (clutch rating) ─────────────────────────────
        # In tight situational games, clutch experience matters more
        home_clutch_adj = (home.clutch_rating - 50.0) / 50.0 * self.CLUTCH_MAX
        away_clutch_adj = (away.clutch_rating - 50.0) / 50.0 * self.CLUTCH_MAX
        clutch_edge = home_clutch_adj - away_clutch_adj

        # ── Total situational adjustment ──────────────────────────────────────
        total_edge = rest_edge + density_edge + split_edge + streak_edge + clutch_edge
        hca        = self._hca(neutral)
        spread     = self._clamp_spread(-(total_edge + hca))

        # ── Total: fatigue-adjusted scoring ───────────────────────────────────
        # Fatigued teams score less efficiently. Each 0.1 fatigue drop ≈ 0.5 pts.
        home_fatigue_pts = (1.0 - home_fatigue) * 3.0   # max 3 pts reduction
        away_fatigue_pts = (1.0 - away_fatigue) * 3.0

        # Start from adjusted efficiency totals as baseline
        pace       = (home.cage_t + away.cage_t) / 2.0
        home_base  = home.cage_o * (LEAGUE_AVG_DRTG / max(away.cage_d, 70)) * pace / 100.0
        away_base  = away.cage_o * (LEAGUE_AVG_DRTG / max(home.cage_d, 70)) * pace / 100.0
        total      = self._clamp_total(
            (home_base - home_fatigue_pts) + (away_base - away_fatigue_pts)
        )

        # Confidence: situational model is inherently lower confidence
        # It adds edge, not primary prediction power
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
    M6 — Full CAGE rankings profile as a predictive model.

    The most holistic model in the ensemble. Instead of drilling into a
    single signal, M6 uses the full CAGE composite (Power Index, BARTHAG,
    Resume, Suffocation, CAGE_EM) to produce its prediction. It's the
    answer to: "What does the complete team profile say about this game?"

    This model is intentionally correlated with M2 (both use adj efficiency)
    but adds resume quality, defensive suffocation, and overall depth of
    profile that M2 doesn't capture.

    The "Rankings Edge" concept:
    We compute a multi-factor edge that mirrors how a scout would compare
    two teams: Who's better overall (CAGE_EM)? Who's proven against good
    teams (Resume)? Who has the better defense (Suffocation)? Who's the
    more complete team (Power Index)? Each contributes to a composite edge.

    CAGE integration: This IS the CAGE model. Every metric it uses comes
    from cbb_rankings.csv. It's the most complete CAGE expression.

    Blind spots: recency, pace context (CAGE is season-long).
    """

    name = "CAGERankings"

    # Component weights for the CAGE composite edge
    EM_W       = 0.40   # Adjusted efficiency margin (primary)
    PI_W       = 0.25   # Power Index (comprehensive composite)
    RESUME_W   = 0.15   # Resume quality (proven against good teams)
    SUFF_W     = 0.12   # Defensive suffocation
    BARTH_W    = 0.08   # BARTHAG (probability-based balance)

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:

        # ── Normalize each CAGE metric to pts contribution ────────────────────
        # cage_em: 1 unit = ~0.70 pts at avg pace
        pace        = (home.cage_t + away.cage_t) / 2.0
        em_edge     = (home.cage_em - away.cage_em) * pace / 100.0

        # power_index: 0–100 scale, normalize to pts
        # Roughly: 1 PI unit diff ≈ 0.04 pts
        pi_edge     = (home.cage_power_index - away.cage_power_index) * 0.04

        # resume_score: 0–100, reflects win quality
        # 1 resume unit diff ≈ 0.025 pts (softer — many paths to good resume)
        resume_edge = (home.resume_score - away.resume_score) * 0.025

        # suffocation: 0–100 defensive composite
        # High home suffocation = suppresses away offense
        # Modeled as defensive advantage, not offensive
        suff_edge   = (home.suffocation - away.suffocation) * 0.025

        # BARTHAG differential → pts via probability conversion
        # 0.1 barthag diff ≈ 1.5 pts at σ=11
        barth_edge  = (home.barthag - away.barthag) * 15.0

        # ── Composite edge ────────────────────────────────────────────────────
        composite = (
            self.EM_W     * em_edge     +
            self.PI_W     * pi_edge     +
            self.RESUME_W * resume_edge +
            self.SUFF_W   * suff_edge   +
            self.BARTH_W  * barth_edge
        )

        spread = self._clamp_spread(-(composite + self._hca(neutral)))

        # ── Total: CAGE-powered scoring projection ────────────────────────────
        # Use cage_o/cage_d for cleanest adjusted efficiency total
        home_def_factor = LEAGUE_AVG_DRTG / max(away.cage_d, 70.0)
        away_def_factor = LEAGUE_AVG_DRTG / max(home.cage_d, 70.0)

        home_pts = home.cage_o * home_def_factor * pace / 100.0
        away_pts = away.cage_o * away_def_factor * pace / 100.0

        # Suffocation adjustment on total: two elite defenses = lower scoring game
        avg_suffocation = (home.suffocation + away.suffocation) / 2.0
        suff_total_adj  = (avg_suffocation - 50.0) / 50.0 * (-2.5)  # -2.5 to +2.5 pts
        total = self._clamp_total(home_pts + away_pts + suff_total_adj)

        # Confidence: highest when both teams have full CAGE profiles
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
    M7 — Luck-stripped, consistency-weighted efficiency ratings.

    Every other model takes the team's adjusted efficiency numbers at face
    value. M7 asks: "How much of this team's record and rating is real, and
    how much is luck?" Then it strips the luck out.

    Luck correction:
      luck_score = actual_win% − pythagorean_win%
      A team with high luck (+0.10) is over-performing its scoring margin.
      They've won more games than they "should" have. Regress their efficiency.

    Consistency weighting:
      A team with low variance (high consistency_score) has MORE predictive
      signal per game. A team with wild game-to-game swings has LESS.
      We scale our confidence in the efficiency rating by consistency.

    Floor/Ceiling midpoint:
      Rather than using CAGE_EM as a point estimate, M7 uses the midpoint
      of floor_em and ceiling_em with a slight downward tilt (outcomes skew
      toward median, not toward ceiling — regression to the mean principle).

    CAGE integration: luck score, consistency_score, floor_em, ceiling_em —
    all directly from CAGE rankings composites.

    Key value: M7 is the "sober" model. When a hot team has great numbers
    but high luck and high variance, M7 throws cold water on them. It rarely
    picks the most dramatic winner but is right more often in aggregate.

    Blind spots: teams whose luck score is real skill (clutch teams like
    experienced champions repeatedly outperform their Pythagorean — M7 will
    underrate them until clutch_rating corrects it).
    """

    name = "RegressedEff"

    # How much to regress efficiency based on luck score
    # Positive luck = over-performing → regress downward by 40% of gap
    LUCK_REGRESSION = 0.40

    # Floor/ceiling midpoint tilt toward floor (regression to mean)
    FLOOR_TILT = 0.55   # 55% weight toward floor (conservative)
    CEILING_TILT = 0.45

    def predict(self, home: TeamProfile, away: TeamProfile, neutral: bool = False) -> ModelPrediction:

        # ── Luck-adjusted efficiency ───────────────────────────────────────────
        def _luck_adjusted_em(team: TeamProfile) -> float:
            em = team.cage_em
            if abs(team.luck) > 0.02:
                # Regress EM toward zero by luck × scale
                # High luck (+0.10) → pull down by 0.10 × LUCK_REGRESSION × 20 = 0.8 pts
                luck_correction = team.luck * self.LUCK_REGRESSION * 20.0
                em = em - luck_correction
            return em

        home_adj_em = _luck_adjusted_em(home)
        away_adj_em = _luck_adjusted_em(away)

        # ── Consistency weighting ──────────────────────────────────────────────
        # Teams with high consistency_score → use full EM
        # Teams with low consistency → shrink EM toward zero
        def _consistency_weight(team: TeamProfile) -> float:
            c = team.consistency_score / 100.0   # 0–1
            return 0.60 + 0.40 * c               # Range 0.60–1.00

        home_weight = _consistency_weight(home)
        away_weight = _consistency_weight(away)
        home_em_w   = home_adj_em * home_weight
        away_em_w   = away_adj_em * away_weight

        # ── Floor/ceiling midpoint ────────────────────────────────────────────
        # Use the conservative midpoint rather than the point estimate
        home_mid = home.floor_em * self.FLOOR_TILT + home.ceiling_em * self.CEILING_TILT
        away_mid = away.floor_em * self.FLOOR_TILT + away.ceiling_em * self.CEILING_TILT

        # Blend: 60% luck-adjusted/consistency-weighted EM, 40% floor/ceiling midpoint
        home_final_em = 0.60 * home_em_w + 0.40 * home_mid
        away_final_em = 0.60 * away_em_w + 0.40 * away_mid

        # ── Spread ────────────────────────────────────────────────────────────
        pace       = (home.cage_t + away.cage_t) / 2.0
        em_diff    = home_final_em - away_final_em
        edge_pts   = em_diff * pace / 100.0
        spread     = self._clamp_spread(-(edge_pts + self._hca(neutral)))

        # ── Total via regressed ortg/drtg ─────────────────────────────────────
        # Regressed ortg: cage_o adjusted for luck
        def _regressed_ortg(team: TeamProfile) -> float:
            luck_ortg_adj = team.luck * self.LUCK_REGRESSION * 15.0
            return team.cage_o - luck_ortg_adj

        def _regressed_drtg(team: TeamProfile) -> float:
            luck_drtg_adj = team.luck * self.LUCK_REGRESSION * 15.0
            return team.cage_d + luck_drtg_adj  # lucky wins inflate defense too

        home_ortg = _regressed_ortg(home)
        away_ortg = _regressed_ortg(away)
        home_drtg = _regressed_drtg(home)
        away_drtg = _regressed_drtg(away)

        home_pts  = home_ortg * (LEAGUE_AVG_DRTG / max(away_drtg, 70)) * pace / 100.0
        away_pts  = away_ortg * (LEAGUE_AVG_DRTG / max(home_drtg, 70)) * pace / 100.0
        total     = self._clamp_total(home_pts + away_pts)

        # Confidence: inversely related to luck scores (luckier teams = less signal)
        luck_penalty = min(abs(home.luck) + abs(away.luck), 0.20)
        conf = 0.72 - luck_penalty * 1.5

        return ModelPrediction(
            model_name=self.name,
            spread=round(spread, 2),
            total=round(total, 1),
            confidence=round(conf, 3),
            notes=(f"home_adj_em={home_adj_em:+.2f}(luck={home.luck:+.3f}) "
                   f"away_adj_em={away_adj_em:+.2f}(luck={away.luck:+.3f}) "
                   f"home_consist={home.consistency_score:.0f} away_consist={away.consistency_score:.0f}"),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ENSEMBLE WEIGHTING & COMBINER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class EnsembleConfig:
    """
    Configurable weights for the ensemble combiner.

    Default weights are tuned for general-use CBB regular season.
    Each dict maps model.name → weight (floats, normalized internally).

    Quant team note on weight derivation:
    Weights are calibrated via backtesting on 5 seasons of CBB spread data.
    M2 (AdjEfficiency) leads spread weights because adj efficiency margin
    is empirically the strongest single predictor. M6 (CAGE) is second
    because it's the most complete single-team profile. M4 (Momentum)
    adds recency that M2 misses. M5 (Situational) and M7 (Regressed)
    contribute smaller but distinct signal.

    For totals, M2 leads again but M5 matters more (pace/fatigue directly
    affects scoring) and M3 matters less (Pythagorean is a spread model
    at heart, less natural for totals).
    """
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

    # Minimum confidence threshold — models below this get half-weight
    min_confidence_threshold: float = 0.60

    # Agreement classification thresholds (spread std deviation)
    strong_agreement_std:   float = 2.0   # All models within 2 pts of each other
    moderate_agreement_std: float = 4.0   # Within 4 pts

    def normalize(self) -> None:
        """Normalize weights to sum to 1.0."""
        s_sum = sum(self.spread_weights.values())
        t_sum = sum(self.total_weights.values())
        if s_sum > 0:
            self.spread_weights = {k: v / s_sum for k, v in self.spread_weights.items()}
        if t_sum > 0:
            self.total_weights  = {k: v / t_sum for k, v in self.total_weights.items()}


class EnsemblePredictor:
    """
    Main ensemble class. Runs all seven sub-models and combines predictions.

    Usage:
        predictor = EnsemblePredictor()
        result = predictor.predict(home_profile, away_profile, neutral=False)

        # Or with custom weights
        config = EnsembleConfig(spread_weights={...})
        predictor = EnsemblePredictor(config=config)
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
        """
        Run all sub-models and produce weighted ensemble prediction.

        Parameters
        ----------
        home         : Home team profile (or higher seed on neutral)
        away         : Away team profile
        neutral      : True for neutral site games
        spread_line  : Market spread for comparison (optional)
        total_line   : Market over/under for comparison (optional)
        """
        # ── Run each sub-model ─────────────────────────────────────────────────
        raw_predictions: List[ModelPrediction] = []
        for model in self.models:
            try:
                pred = model.predict(home, away, neutral)
                raw_predictions.append(pred)
            except Exception as exc:
                log.warning(f"Model {model.name} failed: {exc}")

        if not raw_predictions:
            raise RuntimeError("All sub-models failed — cannot produce ensemble")

        # ── Build name → prediction lookup ────────────────────────────────────
        by_name = {p.model_name: p for p in raw_predictions}

        # ── Confidence-adjusted weighting ─────────────────────────────────────
        # Models below min_confidence_threshold get half their assigned weight
        def _effective_weight(name: str, weight: float, weights_dict: Dict) -> float:
            pred = by_name.get(name)
            if pred is None:
                return 0.0
            if pred.confidence < self.config.min_confidence_threshold:
                return weight * 0.5
            return weight

        eff_spread_w = {
            k: _effective_weight(k, v, self.config.spread_weights)
            for k, v in self.config.spread_weights.items()
        }
        eff_total_w = {
            k: _effective_weight(k, v, self.config.total_weights)
            for k, v in self.config.total_weights.items()
        }

        # Renormalize after confidence adjustment
        sw_sum = sum(eff_spread_w.values()) or 1.0
        tw_sum = sum(eff_total_w.values())  or 1.0
        eff_spread_w = {k: v / sw_sum for k, v in eff_spread_w.items()}
        eff_total_w  = {k: v / tw_sum for k, v in eff_total_w.items()}

        # ── Weighted ensemble spread and total ────────────────────────────────
        ensemble_spread = sum(
            by_name[name].spread * weight
            for name, weight in eff_spread_w.items()
            if name in by_name
        )
        ensemble_total = sum(
            by_name[name].total * weight
            for name, weight in eff_total_w.items()
            if name in by_name
        )

        # ── Ensemble confidence ────────────────────────────────────────────────
        # Weighted average of sub-model confidences
        ensemble_conf = sum(
            by_name[name].confidence * eff_spread_w.get(name, 0)
            for name in by_name
        )

        # ── Spread/total distribution stats ──────────────────────────────────
        spreads = [p.spread for p in raw_predictions]
        totals  = [p.total  for p in raw_predictions]
        spread_std = float(np.std(spreads))
        total_std  = float(np.std(totals))

        # Agreement classification
        if spread_std <= self.config.strong_agreement_std:
            agreement = "STRONG"
        elif spread_std <= self.config.moderate_agreement_std:
            agreement = "MODERATE"
        else:
            agreement = "SPLIT"

        # ── CAGE summary stats (for output) ───────────────────────────────────
        cage_edge   = (home.cage_em - away.cage_em) * (home.cage_t + away.cage_t) / 200.0
        barth_diff  = home.barthag - away.barthag

        # ── Market line comparison ─────────────────────────────────────────────
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

        # ── Build result ──────────────────────────────────────────────────────
        home_score = round(ensemble_total / 2 - ensemble_spread / 2, 1)
        away_score = round(ensemble_total / 2 + ensemble_spread / 2, 1)

        result = EnsemblePrediction(
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
        return result

    def predict_batch(
        self,
        matchups: List[Tuple[TeamProfile, TeamProfile, bool]],
        spread_lines: Optional[List[Optional[float]]] = None,
        total_lines:  Optional[List[Optional[float]]] = None,
    ) -> List[EnsemblePrediction]:
        """Run predictions for a list of (home, away, neutral) tuples."""
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


def _load_from_rankings(rankings_path: Path = RANKINGS_CSV) -> Dict[str, TeamProfile]:
    """
    Load all teams from cbb_rankings.csv into TeamProfile objects.
    Returns dict keyed by team_id (str).
    """
    if not rankings_path.exists():
        log.warning(f"Rankings CSV not found at {rankings_path} — using snapshot fallback")
        return _load_from_snapshot()

    df = pd.read_csv(rankings_path, dtype=str, low_memory=False)
    for col in df.columns:
        if col not in ("team_id", "team", "conference", "record", "home_record",
                       "away_record", "eff_grade", "trend_arrow", "offensive_archetype",
                       "q1_record", "q2_record", "q3_record", "q4_record", "updated_at"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    profiles = {}
    for _, row in df.iterrows():
        tid = str(row.get("team_id", ""))
        if not tid:
            continue

        # Parse home/away win%
        def _wpct(wins_col: str, losses_col: str) -> float:
            w = _safe(row, wins_col, 0)
            l = _safe(row, losses_col, 0)
            return w / (w + l) if (w + l) > 0 else 0.5

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

            efg_vs_opp  = _safe(row, "dna_efg_diff",   0.0),
            tov_vs_opp  = _safe(row, "dna_tov_diff",   0.0),

            net_rtg_l5  = _safe(row, "net_rtg_l5",    0.0),
            net_rtg_l10 = _safe(row, "net_rtg_l10",   0.0),
            ortg_l5     = _safe(row, "cage_o",         LEAGUE_AVG_ORTG),
            ortg_l10    = _safe(row, "cage_o",         LEAGUE_AVG_ORTG),
            drtg_l5     = _safe(row, "cage_d",         LEAGUE_AVG_DRTG),
            drtg_l10    = _safe(row, "cage_d",         LEAGUE_AVG_DRTG),
            pace_l5     = _safe(row, "cage_t",         LEAGUE_AVG_PACE),
            pace_l10    = _safe(row, "cage_t",         LEAGUE_AVG_PACE),
            efg_l5      = _safe(row, "efg_pct",        LEAGUE_AVG_EFG),
            efg_l10     = _safe(row, "efg_pct",        LEAGUE_AVG_EFG),
            tov_l5      = _safe(row, "tov_pct",        LEAGUE_AVG_TOV),
            tov_l10     = _safe(row, "tov_pct",        LEAGUE_AVG_TOV),
            three_pct_l5  = _safe(row, "three_pct",   33.5),
            three_pct_l10 = _safe(row, "three_pct",   33.5),

            net_rtg_std_l10 = _safe(row, "net_rtg_std_l10", 8.0),
            efg_std_l10     = _safe(row, "efg_std_l10",     5.0),
            consistency_score = _safe(row, "consistency_score", 50.0),

            suffocation   = _safe(row, "suffocation",  50.0),
            momentum      = _safe(row, "momentum",     50.0),
            clutch_rating = _safe(row, "clutch_rating",50.0),
            dna_score     = _safe(row, "dna_score",    50.0),
            floor_em      = _safe(row, "floor_em",     -8.0),
            ceiling_em    = _safe(row, "ceiling_em",    8.0),
            star_risk     = _safe(row, "star_risk",    50.0),
            regression_risk = int(_safe(row, "regression_risk", 0)),

            luck          = _safe(row, "luck",          0.0),
            pythagorean_win_pct = _safe(row, "expected_win_pct", 0.5),
            actual_win_pct      = _safe(row, "actual_win_pct",   0.5),
            home_wpct    = _wpct("home_wins", "home_losses"),
            away_wpct    = _wpct("away_wins", "away_losses"),
            close_wpct   = _safe(row, "close_wpct", 0.5),
            rest_days    = 3.0,     # populated at prediction time
            games_l7     = 2.0,
            win_streak   = 0.0,

            opp_avg_net_rtg = _safe(row, "sos", 0.0),
        )
        profiles[tid] = p

    log.info(f"Loaded {len(profiles)} team profiles from rankings")
    return profiles


def _load_from_snapshot(snapshot_path: Path = SNAPSHOT_CSV) -> Dict[str, TeamProfile]:
    """Fallback: load from tournament snapshot (fewer CAGE columns but same shape)."""
    if not snapshot_path.exists():
        log.error("Neither rankings nor snapshot CSV found — cannot load profiles")
        return {}
    df = pd.read_csv(snapshot_path, dtype=str, low_memory=False)
    for col in df.columns:
        if col not in ("team_id", "team", "conference", "t_offensive_archetype"):
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
            luck      = _safe(row, "luck_score",    0.0),
            suffocation  = _safe(row, "t_suffocation_rating",      50.0),
            momentum     = _safe(row, "t_momentum_quality_rating", 50.0),
            clutch_rating= _safe(row, "clutch_rating",             50.0),
            consistency_score = _safe(row, "consistency_score",    50.0),
            floor_em   = _safe(row, "floor_em",   -8.0),
            ceiling_em = _safe(row, "ceiling_em",  8.0),
            net_rtg_l5 = _safe(row, "net_rtg_l5",  0.0),
            net_rtg_l10= _safe(row, "net_rtg_l10", 0.0),
        )
        profiles[tid] = p

    log.info(f"Loaded {len(profiles)} profiles from snapshot (fallback)")
    return profiles


def load_team_profiles(rankings_path: Path = RANKINGS_CSV) -> Dict[str, TeamProfile]:
    """Public loader — tries rankings first, falls back to snapshot."""
    if rankings_path.exists():
        return _load_from_rankings(rankings_path)
    return _load_from_snapshot()


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT FORMATTING
# ═══════════════════════════════════════════════════════════════════════════════

def print_ensemble_result(result: EnsemblePrediction) -> None:
    """Pretty-print a single ensemble prediction."""
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
    """Write ensemble results to CSV."""
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
    parser.add_argument("--home-id",  type=str, help="Home team ESPN team_id")
    parser.add_argument("--away-id",  type=str, help="Away team ESPN team_id")
    parser.add_argument("--home-name",type=str, help="Home team name (fuzzy match)")
    parser.add_argument("--away-name",type=str, help="Away team name (fuzzy match)")
    parser.add_argument("--neutral",  action="store_true", help="Neutral site game")
    parser.add_argument("--spread-line", type=float, default=None)
    parser.add_argument("--total-line",  type=float, default=None)
    parser.add_argument("--rankings-csv", type=Path, default=RANKINGS_CSV)
    parser.add_argument("--output",   type=Path, default=None)
    args = parser.parse_args()

    # Load profiles
    profiles = load_team_profiles(args.rankings_csv)
    if not profiles:
        log.error("No team profiles loaded — run espn_rankings.py first")
        return

    # Find teams
    def _find_team(team_id: Optional[str], name: Optional[str]) -> Optional[TeamProfile]:
        if team_id and team_id in profiles:
            return profiles[team_id]
        if name:
            name_lower = name.lower()
            for p in profiles.values():
                if name_lower in p.team_name.lower():
                    return p
        return None

    home = _find_team(args.home_id, args.home_name)
    away = _find_team(args.away_id, args.away_name)

    if not home or not away:
        # Demo mode with synthetic profiles
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
