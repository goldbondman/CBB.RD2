#!/usr/bin/env python3
"""
cbb_backtester.py — Historical Prediction Backtester

Replays every completed game in the dataset as if it were predicted in real time,
using ONLY data that was available before tipoff. Produces per-model and ensemble
accuracy metrics, calibration curves, and empirically-derived weight suggestions.

THE LEAKAGE PROBLEM (and how we solve it)
─────────────────────────────────────────────────────────────────────────────
The naive approach — load team_game_weighted.csv, grab a team's row for game G,
use its rolling metrics to predict game G — is WRONG. Those rolling metrics
include game G itself. A team's adj_net_rtg row for game G was computed after
the game was played.

Our solution: for each game G, filter the team's game log to rows where
game_datetime_utc < G's start time. Then take the most recent row as the
pre-game team state. This row contains metrics through game G-1 — everything
the model legitimately knew before tipoff.

This is the same discipline as walk-forward validation in finance. The pipeline
already stores one row per team per game; we just read backward.

OUTPUTS
─────────────────────────────────────────────────────────────────────────────
data/backtest_results_YYYYMMDD.csv     — per-game prediction vs actual
data/backtest_model_report_YYYYMMDD.csv — per-model accuracy metrics
data/backtest_optimized_weights.json   — suggested ensemble weights from optimizer
data/backtest_calibration_YYYYMMDD.csv — confidence vs actual win rate bins

METRICS COMPUTED
─────────────────────────────────────────────────────────────────────────────
ATS%        : Against-the-spread win rate (>52.4% = beating the vig)
O/U%        : Over/under accuracy
MAE         : Mean absolute error on predicted margin
RMSE        : Root mean squared error on predicted margin
Brier Score : Calibration metric for win probability (lower = better)
ROI (sim)   : Simulated return if flat-betting $1 at -110 on every edge flag
Coverage    : % of games we had enough data to predict

Usage:
    python cbb_backtester.py
    python cbb_backtester.py --min-games 8 --start-date 20250101 --top-n 25
    python cbb_backtester.py --optimize-weights --output-dir data/
"""

import argparse
import json
import logging
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
WEIGHTED_CSV = DATA_DIR / "team_game_weighted.csv"
METRICS_CSV  = DATA_DIR / "team_game_metrics.csv"
GAMES_CSV    = DATA_DIR / "games.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
LEAGUE_AVG_ORTG = 103.0
LEAGUE_AVG_DRTG = 103.0
LEAGUE_AVG_PACE = 70.0
LEAGUE_AVG_EFG  = 50.5
LEAGUE_AVG_TOV  = 18.0
LEAGUE_AVG_FTR  = 28.0
VIG_BREAK_EVEN  = 52.38   # % ATS needed to break even at -110 juice


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG & DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BacktestConfig:
    """Controls backtest behavior."""
    min_games_required:  int   = 5      # Min games for each team before predicting
    start_date:          Optional[str] = None   # YYYYMMDD — skip earlier games
    end_date:            Optional[str] = None
    edge_threshold:      float = 3.0    # pts from line to flag as edge
    confidence_bins:     int   = 10     # Bins for calibration curve
    optimize_weights:    bool  = True   # Run weight optimizer after backtest
    optimizer_method:    str   = "Nelder-Mead"
    optimizer_metric:    str   = "ats"  # "ats", "mae", "brier"
    min_edge_sample:     int   = 20     # Min edge-flag games for ROI calc
    top_n_display:       int   = 25     # Games to show in printed table


@dataclass
class GameResult:
    """Actual outcome of a game."""
    game_id:          str
    game_datetime:    pd.Timestamp
    home_team_id:     str
    away_team_id:     str
    home_team:        str
    away_team:        str
    home_score:       float
    away_score:       float
    actual_margin:    float    # home − away (positive = home won)
    neutral_site:     bool


@dataclass
class PredictionRecord:
    """
    One game's full prediction vs actual outcome.
    Stored per-model AND as ensemble.
    """
    game_id:          str
    game_datetime:    str
    home_team:        str
    away_team:        str
    home_team_id:     str
    away_team_id:     str
    neutral_site:     bool
    actual_margin:    float    # home − away
    actual_total:     float
    home_score_actual: float
    away_score_actual: float

    # Per-model predictions (keyed by model name)
    model_spreads:    Dict[str, float] = field(default_factory=dict)
    model_totals:     Dict[str, float] = field(default_factory=dict)
    model_confs:      Dict[str, float] = field(default_factory=dict)

    # Ensemble
    ens_spread:       float = 0.0
    ens_total:        float = 0.0
    ens_confidence:   float = 0.0
    ens_agreement:    str   = ""
    spread_std:       float = 0.0

    # Market line (from games.csv if available)
    market_spread:    Optional[float] = None
    market_total:     Optional[float] = None

    # Pre-game team state metadata
    home_games_before: int = 0
    away_games_before: int = 0
    data_quality:      str = "full"   # "full", "limited", "skipped"


@dataclass
class ModelMetrics:
    """Accuracy metrics for one model (or ensemble)."""
    model_name:        str
    n_games:           int = 0

    # Spread accuracy
    spread_mae:        float = 0.0    # Mean absolute error
    spread_rmse:       float = 0.0
    spread_bias:       float = 0.0    # Systematic over/under-prediction
    spread_r2:         float = 0.0    # Correlation with actual margin

    # ATS (requires market spread)
    n_ats_games:       int   = 0
    ats_correct:       int   = 0
    ats_pct:           float = 0.0    # >52.38 = profitable at -110

    # Over/under (requires market total)
    n_ou_games:        int   = 0
    ou_correct:        int   = 0
    ou_pct:            float = 0.0

    # Win prediction accuracy
    win_correct:       int   = 0
    win_pct:           float = 0.0

    # Brier score (calibration)
    brier_score:       float = 0.0    # Lower = better calibrated; 0.25 = random

    # Edge flag ROI simulation
    n_edge_games:      int   = 0
    edge_ats_pct:      float = 0.0
    edge_roi_sim:      float = 0.0    # Simulated ROI at -110 on all edge games

    # Confidence correlation
    conf_ats_corr:     float = 0.0    # Do higher-confidence picks win more often?

    def to_dict(self) -> Dict:
        return asdict(self)

    @property
    def ats_vs_vig(self) -> float:
        """Percentage points above break-even."""
        return self.ats_pct - VIG_BREAK_EVEN


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

STR_COLS = {
    "team_id", "team", "opponent_id", "opponent", "home_away",
    "conference", "conf_id", "event_id", "game_id",
    "game_datetime_utc", "game_datetime_pst", "venue", "state",
    "source", "parse_version", "t_offensive_archetype",
}


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if col not in STR_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_game_log() -> pd.DataFrame:
    """
    Load full season game log. Prefers team_game_weighted (richest),
    falls back to team_game_metrics.
    Returns sorted by team_id, game_datetime_utc.
    """
    for path in [WEIGHTED_CSV, METRICS_CSV]:
        if path.exists() and path.stat().st_size > 100:
            log.info(f"Loading game log: {path.name}")
            df = pd.read_csv(path, dtype=str, low_memory=False)
            df["game_datetime_utc"] = pd.to_datetime(
                df.get("game_datetime_utc", pd.NaT), utc=True, errors="coerce"
            )
            df = _coerce_numeric(df)
            df = df.sort_values(["team_id", "game_datetime_utc"])
            log.info(f"  {len(df):,} team-game rows, {df['team_id'].nunique()} teams")
            return df

    raise FileNotFoundError(
        "No game log found. Run espn_pipeline.py first.\n"
        f"Looked for: {WEIGHTED_CSV}, {METRICS_CSV}"
    )


def load_completed_games(game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Build a table of completed games with actual scores.
    We reconstruct this from the game log by pivoting home/away rows.
    Returns one row per game with home_score, away_score, actual_margin.
    """
    gl = game_log.copy()
    gl["team_id"]  = gl["team_id"].astype(str)
    gl["event_id"] = gl["event_id"].astype(str)

    # Filter to completed games (have real scores)
    gl = gl.dropna(subset=["points_for", "points_against"])
    gl = gl[gl["points_for"] > 0]

    # Split home / away rows
    home = gl[gl["home_away"].astype(str).str.lower() == "home"].copy()
    away = gl[gl["home_away"].astype(str).str.lower() == "away"].copy()

    if home.empty or away.empty:
        log.warning("Could not split home/away rows — using fallback pivot")
        # Fallback: each game appears twice (once per team), dedupe
        gl["actual_margin"] = gl["points_for"] - gl["points_against"]
        return gl[["event_id", "game_datetime_utc", "team_id", "team",
                   "opponent_id", "opponent", "points_for", "points_against",
                   "actual_margin"]].drop_duplicates("event_id")

    home = home.rename(columns={
        "team_id": "home_team_id", "team": "home_team",
        "opponent_id": "away_team_id", "opponent": "away_team",
        "points_for": "home_score", "points_against": "away_score",
    })
    away = away.rename(columns={
        "team_id": "away_team_id_check",
    })

    games = home[["event_id", "game_datetime_utc",
                  "home_team_id", "home_team",
                  "away_team_id", "away_team",
                  "home_score", "away_score"]].copy()

    games["actual_margin"] = games["home_score"] - games["away_score"]
    games["actual_total"]  = games["home_score"] + games["away_score"]
    games["home_won"]      = (games["actual_margin"] > 0).astype(int)
    games["neutral_site"]  = home.get("neutral_site", pd.Series(0, index=home.index)).values

    games = games.drop_duplicates("event_id")
    log.info(f"Completed games: {len(games):,}")
    return games.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM STATE RECONSTRUCTION (ZERO LOOKAHEAD)
# ═══════════════════════════════════════════════════════════════════════════════

def build_team_state_before(
    team_id: str,
    before_dt: pd.Timestamp,
    game_log: pd.DataFrame,
    min_games: int = 5,
) -> Optional[Dict]:
    """
    Reconstruct team state using only games that completed BEFORE before_dt.

    This is the core leakage prevention. We take the most recent game-log row
    for this team with game_datetime_utc < before_dt. That row contains rolling
    metrics (adj_net_rtg, net_rtg_l5, etc.) computed through game G-1.

    Returns None if fewer than min_games are available.
    """
    team_rows = game_log[
        (game_log["team_id"].astype(str) == str(team_id)) &
        (game_log["game_datetime_utc"] < before_dt)
    ].sort_values("game_datetime_utc")

    if len(team_rows) < min_games:
        return None

    # Most recent row = team's state immediately before this game
    latest = team_rows.iloc[-1]

    def g(col, default=0.0):
        v = latest.get(col, default)
        try:
            return float(v) if pd.notna(v) else default
        except (TypeError, ValueError):
            return default

    return {
        "team_id":        str(team_id),
        "team_name":      str(latest.get("team", "")),
        "conference":     str(latest.get("conference", "")),
        "games_before":   len(team_rows),

        # CAGE efficiency
        "cage_em":        g("adj_net_rtg"),
        "cage_o":         g("adj_ortg",  LEAGUE_AVG_ORTG),
        "cage_d":         g("adj_drtg",  LEAGUE_AVG_DRTG),
        "cage_t":         g("adj_pace",  LEAGUE_AVG_PACE),
        "barthag":        g("barthag",   0.500),

        # Four factors
        "efg_pct":        g("efg_pct",   LEAGUE_AVG_EFG),
        "tov_pct":        g("tov_pct",   LEAGUE_AVG_TOV),
        "orb_pct":        g("orb_pct",   30.0),
        "drb_pct":        g("drb_pct",   70.0),
        "ftr":            g("ftr",       LEAGUE_AVG_FTR),
        "ft_pct":         g("ft_pct",    71.0),
        "three_pct":      g("three_pct", 33.5),
        "three_par":      g("three_par", 35.0),
        "opp_efg_pct":    g("opp_avg_efg_season", LEAGUE_AVG_EFG),
        "opp_tov_pct":    g("opp_avg_tov_season",  LEAGUE_AVG_TOV),
        "opp_ftr":        g("opp_avg_ftr_season",   LEAGUE_AVG_FTR),

        # Opponent context
        "efg_vs_opp":     g("efg_vs_opp_season",  0.0),
        "tov_vs_opp":     g("tov_vs_opp_season",  0.0),
        "orb_vs_opp":     g("orb_vs_opp_season",  0.0),
        "ftr_vs_opp":     g("ftr_vs_opp_season",  0.0),

        # Rolling windows
        "net_rtg_l5":     g("net_rtg_l5",   0.0),
        "net_rtg_l10":    g("net_rtg_l10",  0.0),
        "ortg_l5":        g("ortg_l5",      LEAGUE_AVG_ORTG),
        "ortg_l10":       g("ortg_l10",     LEAGUE_AVG_ORTG),
        "drtg_l5":        g("drtg_l5",      LEAGUE_AVG_DRTG),
        "drtg_l10":       g("drtg_l10",     LEAGUE_AVG_DRTG),
        "pace_l5":        g("pace_l5",      LEAGUE_AVG_PACE),
        "pace_l10":       g("pace_l10",     LEAGUE_AVG_PACE),
        "efg_l5":         g("efg_l5",       LEAGUE_AVG_EFG),
        "efg_l10":        g("efg_l10",      LEAGUE_AVG_EFG),
        "tov_l5":         g("tov_l5",       LEAGUE_AVG_TOV),
        "tov_l10":        g("tov_l10",      LEAGUE_AVG_TOV),
        "three_pct_l5":   g("three_pct_l5", 33.5),
        "three_pct_l10":  g("three_pct_l10",33.5),

        # Variance
        "net_rtg_std_l10":    g("net_rtg_std_l10", 8.0),
        "efg_std_l10":        g("efg_std_l10",      5.0),
        "consistency_score":  g("consistency_score",50.0),

        # CAGE composites (from tournament module if available)
        "suffocation":    g("t_suffocation_rating",      50.0),
        "momentum":       g("t_momentum_quality_rating", 50.0),
        "clutch_rating":  g("clutch_rating",             50.0),
        "floor_em":       g("floor_em",   -8.0),
        "ceiling_em":     g("ceiling_em",  8.0),
        "dna_score":      g("t_tournament_dna_score",    50.0),
        "star_risk":      g("t_star_reliance_risk",      50.0),
        "regression_risk":int(g("t_regression_risk_flag", 0)),
        "resume_score":   g("resume_score", 50.0),
        "cage_power_index": g("cage_power_index", 50.0),

        # Luck & record
        "luck":               g("luck_score",           0.0),
        "pythagorean_win_pct":g("pythagorean_win_pct",  0.5),
        "actual_win_pct":     g("season_win_pct",       0.5),
        "home_wpct":          g("home_win_pct",         0.65),
        "away_wpct":          g("away_win_pct",         0.40),
        "close_wpct":         g("close_game_win_pct",   0.50),
        "win_streak":         g("win_streak",           0.0),
        "sos":                g("opp_avg_net_rtg_season", 0.0),
        "opp_avg_net_rtg":    g("opp_avg_net_rtg_season", 0.0),

        # Opponent context — M6 (CAGERankings) reads wab; M1 (FourFactors) reads opp_orb_pct
        "wab":          g("wab",                  0.0),
        "opp_avg_ortg": g("opp_avg_ortg_season",  LEAGUE_AVG_ORTG),
        "opp_avg_drtg": g("opp_avg_drtg_season",  LEAGUE_AVG_DRTG),
        "opp_orb_pct":  g("opp_avg_orb_season",   30.0),
    }


def _state_to_profile(state: Dict):
    """Convert a raw state dict to a TeamProfile for the ensemble models."""
    from cbb_ensemble import TeamProfile
    return TeamProfile(**{
        k: state[k] for k in TeamProfile.__dataclass_fields__
        if k in state
    })


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION ENGINE (single-game replay)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_game_historical(
    home_state: Dict,
    away_state: Dict,
    neutral: bool,
    weights: Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Run all 7 ensemble sub-models on pre-game team states.
    Returns dict with per-model predictions + ensemble.

    weights: if provided, overrides EnsembleConfig defaults (for optimizer).
    """
    from cbb_ensemble import (
        EnsemblePredictor, EnsembleConfig,
        FourFactorsModel, AdjustedEfficiencyModel, PythagoreanModel,
        MomentumModel, SituationalModel, CAGERankingsModel, RegressedEfficiencyModel,
        TeamProfile,
    )

    home = _state_to_profile(home_state)
    away = _state_to_profile(away_state)

    # Inject rest/situational defaults (not available in historical data)
    home.rest_days = 3.0
    away.rest_days = 3.0
    home.games_l7  = 2.0
    away.games_l7  = 2.0

    # Build ensemble config (use custom weights if optimizing)
    if weights:
        config = EnsembleConfig(
            spread_weights=weights.copy(),
            total_weights=weights.copy(),
        )
    else:
        config = EnsembleConfig()

    predictor = EnsemblePredictor(config)
    result    = predictor.predict(home, away, neutral=neutral)

    out = {
        "ens_spread":     result.spread,
        "ens_total":      result.total,
        "ens_confidence": result.confidence,
        "ens_agreement":  result.model_agreement,
        "spread_std":     result.spread_std,
        "cage_edge":      result.cage_edge,
        "barthag_diff":   result.barthag_diff,
    }

    for mp in result.model_predictions:
        name = mp.model_name.lower()
        out[f"{name}_spread"] = mp.spread
        out[f"{name}_total"]  = mp.total
        out[f"{name}_conf"]   = mp.confidence

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_model_metrics(
    records: pd.DataFrame,
    spread_col: str,
    total_col: str,
    conf_col: str,
    model_name: str,
    edge_threshold: float = 3.0,
) -> ModelMetrics:
    """
    Compute full accuracy metrics for one model using all backtest records.

    Parameters
    ----------
    records     : BacktestResults DataFrame
    spread_col  : Column with this model's predicted spread (home perspective)
    total_col   : Column with this model's predicted total
    conf_col    : Column with this model's confidence
    model_name  : Display name
    edge_threshold : Pts from market line to flag as an edge
    """
    m = ModelMetrics(model_name=model_name)

    df = records.dropna(subset=[spread_col, "actual_margin"]).copy()
    m.n_games = len(df)

    if m.n_games == 0:
        return m

    actual     = df["actual_margin"].values
    predicted  = -df[spread_col].values   # Convert spread to margin (invert sign)
    residuals  = actual - predicted

    # ── Margin accuracy ───────────────────────────────────────────────────────
    m.spread_mae  = float(np.mean(np.abs(residuals)))
    m.spread_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    m.spread_bias = float(np.mean(residuals))   # + = we underpredict home, − = overpredict

    corr, _ = scipy_stats.pearsonr(predicted, actual)
    m.spread_r2 = float(corr ** 2) if not np.isnan(corr) else 0.0

    # ── Win prediction ────────────────────────────────────────────────────────
    pred_home_wins = (predicted > 0).astype(int)
    actual_home_wins = (actual > 0).astype(int)
    m.win_correct = int((pred_home_wins == actual_home_wins).sum())
    m.win_pct     = round(m.win_correct / m.n_games * 100, 1)

    # ── ATS (vs market spread) ────────────────────────────────────────────────
    ats_df = df.dropna(subset=["market_spread"])
    m.n_ats_games = len(ats_df)

    if m.n_ats_games > 0:
        mkt_spread  = ats_df["market_spread"].values    # negative = home favored
        act_margin  = ats_df["actual_margin"].values
        pred_margin = -ats_df[spread_col].values

        # ATS: did actual margin beat the market spread?
        # e.g. market_spread=-7, actual_margin=+3: home +3 vs -7 line → home covers
        # Our spread is home-perspective (negative = home favored, matches market convention)
        home_cover = act_margin > (-mkt_spread)     # push = loss
        away_cover = act_margin < (-mkt_spread)
        push       = act_margin == (-mkt_spread)

        # We pick home if our model thinks home covers (pred_margin > −mkt_spread means
        # we think home does better than the line says)
        model_picks_home = pred_margin > (-mkt_spread)

        correct = (
            (model_picks_home & home_cover) |
            (~model_picks_home & away_cover)
        )
        # Pushes don't count
        non_push_mask = ~push
        if non_push_mask.sum() > 0:
            m.ats_correct = int(correct[non_push_mask].sum())
            m.ats_pct     = round(m.ats_correct / non_push_mask.sum() * 100, 1)

        # ── Edge flag ROI ──────────────────────────────────────────────────────
        edge_mask = np.abs(pred_margin - (-mkt_spread)) >= edge_threshold
        m.n_edge_games = int(edge_mask.sum())

        if m.n_edge_games >= 5:
            edge_correct = int(correct[edge_mask & non_push_mask].sum())
            edge_total   = int(non_push_mask[edge_mask].sum())
            m.edge_ats_pct = round(edge_correct / max(edge_total, 1) * 100, 1)
            # Flat-bet ROI at -110: win 0.909 units, lose 1.0 unit
            wins_roi  = edge_correct * 0.909
            losses_roi = (edge_total - edge_correct) * 1.0
            m.edge_roi_sim = round((wins_roi - losses_roi) / max(edge_total, 1) * 100, 1)

    # ── Over/Under ────────────────────────────────────────────────────────────
    ou_df = df.dropna(subset=["market_total", total_col, "actual_total"])
    m.n_ou_games = len(ou_df)

    if m.n_ou_games > 0:
        pred_totals = ou_df[total_col].values
        mkt_totals  = ou_df["market_total"].values
        act_totals  = ou_df["actual_total"].values

        model_over  = pred_totals > mkt_totals
        actual_over = act_totals  > mkt_totals
        actual_push = act_totals == mkt_totals
        non_push    = ~actual_push

        ou_correct  = ((model_over == actual_over) & non_push)
        m.ou_correct = int(ou_correct.sum())
        m.ou_pct    = round(m.ou_correct / non_push.sum() * 100, 1) if non_push.sum() > 0 else 0.0

    # ── Brier Score ───────────────────────────────────────────────────────────
    # Convert predicted margin to win probability via normal CDF
    sigma = 11.0  # empirical CBB game distribution width
    win_probs = scipy_stats.norm.cdf(predicted / sigma)
    outcomes  = (actual > 0).astype(float)
    m.brier_score = round(float(np.mean((win_probs - outcomes) ** 2)), 4)

    # ── Confidence vs ATS correlation ────────────────────────────────────────
    if conf_col in df.columns and m.n_ats_games > 0:
        conf_df = ats_df.dropna(subset=[conf_col])
        if len(conf_df) > 10:
            conf_vals   = conf_df[conf_col].values
            pred_m_ats  = -conf_df[spread_col].values
            mkt_m_ats   = -conf_df["market_spread"].values
            correct_ats = (pred_m_ats > mkt_m_ats) == (conf_df["actual_margin"].values > mkt_m_ats)
            corr, _     = scipy_stats.pointbiserialr(conf_vals, correct_ats.astype(float))
            m.conf_ats_corr = round(float(corr) if not np.isnan(corr) else 0.0, 3)

    return m


def build_calibration_curve(
    records: pd.DataFrame,
    conf_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Group predictions by confidence decile, compute actual ATS win rate per bin.
    A well-calibrated model: high confidence bins → higher ATS win rate.
    Returns DataFrame with columns: conf_bin_lo, conf_bin_hi, n_games, ats_pct.
    """
    df = records.dropna(subset=[conf_col, "market_spread", "actual_margin",
                                  "ens_spread"]).copy()
    if len(df) < n_bins * 3:
        return pd.DataFrame()

    df["conf_bin"] = pd.cut(df[conf_col], bins=n_bins, labels=False)
    pred_margin    = -df["ens_spread"].values
    mkt_spread     = df["market_spread"].values
    act_margin     = df["actual_margin"].values

    rows = []
    for bin_id in range(n_bins):
        mask = df["conf_bin"] == bin_id
        if mask.sum() < 3:
            continue
        bin_df  = df[mask]
        bin_pred = pred_margin[mask.values]
        bin_mkt  = mkt_spread[mask.values]
        bin_act  = act_margin[mask.values]

        home_cover  = bin_act > (-bin_mkt)
        model_home  = bin_pred > (-bin_mkt)
        push        = bin_act == (-bin_mkt)
        non_push    = ~push
        correct     = (model_home == home_cover) & non_push

        rows.append({
            "conf_bin_lo": round(bin_df[conf_col].min(), 3),
            "conf_bin_hi": round(bin_df[conf_col].max(), 3),
            "conf_mean":   round(bin_df[conf_col].mean(), 3),
            "n_games":     int(non_push.sum()),
            "ats_correct": int(correct.sum()),
            "ats_pct":     round(correct.sum() / max(non_push.sum(), 1) * 100, 1),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT OPTIMIZER
# ═══════════════════════════════════════════════════════════════════════════════

MODEL_NAMES = [
    "fourfactors", "adjefficiency", "pythagorean",
    "momentum", "situational", "cagerankings", "regressedeff",
]

DEFAULT_WEIGHTS = np.array([0.12, 0.22, 0.14, 0.16, 0.10, 0.18, 0.08])


def _ensemble_from_weights(records: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Compute ensemble spread from per-model spreads with given weights."""
    w = np.abs(weights) / np.abs(weights).sum()   # Normalize, force positive
    cols = [f"{n}_spread" for n in MODEL_NAMES]

    available = [c for c in cols if c in records.columns]
    if not available:
        raise ValueError("No per-model spread columns found in records")

    mat = records[[c for c in cols if c in records.columns]].fillna(0).values
    # Pad with zeros if some models missing
    if mat.shape[1] < len(MODEL_NAMES):
        pad = np.zeros((mat.shape[0], len(MODEL_NAMES) - mat.shape[1]))
        mat = np.hstack([mat, pad])
        w_use = w[:mat.shape[1]]
        w_use = w_use / w_use.sum()
    else:
        w_use = w

    return mat @ w_use


def _loss_ats(weights: np.ndarray, records: pd.DataFrame) -> float:
    """
    Loss function for optimizer: negative ATS win rate (we minimize, so
    optimizer maximizes ATS accuracy).
    """
    try:
        ens_pred    = _ensemble_from_weights(records, weights)
        pred_margin = -ens_pred     # Convert spread → margin
        mkt_spread  = records["market_spread"].values
        act_margin  = records["actual_margin"].values

        home_cover = act_margin > (-mkt_spread)
        model_home = pred_margin > (-mkt_spread)
        push       = act_margin == (-mkt_spread)
        non_push   = ~push

        correct = ((model_home == home_cover) & non_push)
        ats_pct = correct[non_push].mean() if non_push.sum() > 0 else 0.5
        return -float(ats_pct)    # Negative because we minimize
    except Exception:
        return 0.0    # Return no-loss on failure


def _loss_mae(weights: np.ndarray, records: pd.DataFrame) -> float:
    """Loss: mean absolute error on predicted margin."""
    try:
        ens_pred    = _ensemble_from_weights(records, weights)
        pred_margin = -ens_pred
        act_margin  = records["actual_margin"].values
        return float(np.mean(np.abs(act_margin - pred_margin)))
    except Exception:
        return 20.0


def _loss_brier(weights: np.ndarray, records: pd.DataFrame) -> float:
    """Loss: Brier score (calibration quality)."""
    try:
        ens_pred    = _ensemble_from_weights(records, weights)
        pred_margin = -ens_pred
        win_probs   = scipy_stats.norm.cdf(pred_margin / 11.0)
        outcomes    = (records["actual_margin"].values > 0).astype(float)
        return float(np.mean((win_probs - outcomes) ** 2))
    except Exception:
        return 0.25


def optimize_weights(
    records: pd.DataFrame,
    metric: str = "ats",
    method: str = "Nelder-Mead",
    n_restarts: int = 8,
) -> Tuple[Dict[str, float], float]:
    """
    Find ensemble weights that maximize ATS accuracy (or minimize MAE/Brier).

    Uses multiple random restarts to avoid local minima.
    Returns (weight_dict, best_metric_value).

    The optimizer runs on the full historical backtest — in production you'd
    want a holdout set. But even with in-sample optimization, weights that
    are drastically different from prior suggest models need recalibration.
    """
    ats_df = records.dropna(subset=["market_spread", "actual_margin"]).copy()

    # Need at least one per-model spread column
    model_cols = [f"{n}_spread" for n in MODEL_NAMES if f"{n}_spread" in ats_df.columns]
    if len(model_cols) < 3:
        log.warning("Too few model columns for weight optimization")
        return {n: w for n, w in zip(MODEL_NAMES, DEFAULT_WEIGHTS)}, 0.0

    loss_fn = {"ats": _loss_ats, "mae": _loss_mae, "brier": _loss_brier}.get(metric, _loss_ats)

    best_weights = DEFAULT_WEIGHTS.copy()
    best_loss    = float("inf")

    np.random.seed(42)
    for i in range(n_restarts):
        if i == 0:
            w0 = DEFAULT_WEIGHTS.copy()
        else:
            w0 = np.random.dirichlet(np.ones(len(MODEL_NAMES)))

        result = minimize(
            loss_fn,
            w0,
            args=(ats_df,),
            method=method,
            options={"maxiter": 2000, "xatol": 1e-4, "fatol": 1e-4},
        )

        if result.fun < best_loss:
            best_loss    = result.fun
            best_weights = result.x

    # Normalize to sum to 1.0
    best_weights = np.abs(best_weights)
    best_weights = best_weights / best_weights.sum()

    weight_dict = {n: round(float(w), 4) for n, w in zip(MODEL_NAMES, best_weights)}
    metric_val  = -best_loss if metric == "ats" else best_loss

    log.info(f"Optimized weights ({metric}): {weight_dict}")
    log.info(f"Best {metric}: {metric_val:.4f}  (default: {-loss_fn(DEFAULT_WEIGHTS, ats_df):.4f})")

    return weight_dict, round(metric_val, 4)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class BacktestEngine:
    """
    Iterates through historical games, reconstructs pre-game team states,
    runs predictions, and computes accuracy metrics.
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    def run(
        self,
        game_log: pd.DataFrame,
        completed_games: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Main backtest loop. Returns DataFrame of PredictionRecords.

        For each completed game:
        1. Get pre-game team states (leakage-free)
        2. Run all 7 sub-models
        3. Record predicted vs actual
        """
        cfg    = self.config
        records = []
        skipped = {"too_few_games": 0, "no_state": 0, "model_error": 0}

        # Apply date filters
        games = completed_games.copy()
        if cfg.start_date:
            start = pd.Timestamp(f"{cfg.start_date[:4]}-{cfg.start_date[4:6]}-{cfg.start_date[6:]}", tz="UTC")
            games = games[games["game_datetime_utc"] >= start]
        if cfg.end_date:
            end   = pd.Timestamp(f"{cfg.end_date[:4]}-{cfg.end_date[4:6]}-{cfg.end_date[6:]}", tz="UTC")
            games = games[games["game_datetime_utc"] <= end]

        total = len(games)
        log.info(f"Backtesting {total:,} games...")

        for idx, row in games.iterrows():
            if (idx % 500) == 0 and idx > 0:
                log.info(f"  Progress: {idx:,}/{total:,}  ({idx/total*100:.0f}%)")

            game_dt    = row["game_datetime_utc"]
            home_id    = str(row["home_team_id"])
            away_id    = str(row["away_team_id"])
            event_id   = str(row["event_id"])
            neutral    = bool(row.get("neutral_site", False))

            # ── Build pre-game team states (ZERO LOOKAHEAD) ──────────────────
            home_state = build_team_state_before(
                home_id, game_dt, game_log, cfg.min_games_required
            )
            away_state = build_team_state_before(
                away_id, game_dt, game_log, cfg.min_games_required
            )

            if home_state is None or away_state is None:
                skipped["too_few_games"] += 1
                continue

            # ── Run prediction ────────────────────────────────────────────────
            try:
                pred = predict_game_historical(home_state, away_state, neutral)
            except Exception as exc:
                log.debug(f"Model error on {event_id}: {exc}")
                skipped["model_error"] += 1
                continue

            # ── Build record ──────────────────────────────────────────────────
            rec = {
                "game_id":          event_id,
                "game_datetime":    str(game_dt),
                "home_team":        str(row.get("home_team", "")),
                "away_team":        str(row.get("away_team", "")),
                "home_team_id":     home_id,
                "away_team_id":     away_id,
                "neutral_site":     int(neutral),
                "actual_margin":    float(row["actual_margin"]),
                "actual_total":     float(row["actual_total"]),
                "home_score_actual":float(row["home_score"]),
                "away_score_actual":float(row["away_score"]),
                "home_games_before": home_state["games_before"],
                "away_games_before": away_state["games_before"],
                "home_cage_em":      round(home_state["cage_em"], 2),
                "away_cage_em":      round(away_state["cage_em"], 2),
                "cage_em_diff":      round(home_state["cage_em"] - away_state["cage_em"], 2),

                # Market lines (populated later if available)
                "market_spread":    None,
                "market_total":     None,
            }
            rec.update(pred)
            records.append(rec)

        log.info(f"Backtest complete: {len(records):,} predicted | "
                 f"skipped: {sum(skipped.values())} "
                 f"(insufficient history: {skipped['too_few_games']}, "
                 f"model errors: {skipped['model_error']})")

        return pd.DataFrame(records)

    def attach_market_lines(
        self,
        records: pd.DataFrame,
        games_csv_path: Path = GAMES_CSV,
    ) -> pd.DataFrame:
        """
        Join market spread / over-under from games.csv onto backtest records.
        Many historical games won't have lines — that's expected.
        """
        if not games_csv_path.exists():
            log.warning("games.csv not found — skipping market line attachment")
            return records

        gdf = pd.read_csv(games_csv_path, dtype=str)
        for col in ["spread", "over_under", "home_ml", "away_ml"]:
            if col in gdf.columns:
                gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

        gdf["game_id"] = gdf["game_id"].astype(str)
        records["game_id"] = records["game_id"].astype(str)

        records = records.merge(
            gdf[["game_id", "spread", "over_under"]].rename(
                columns={"spread": "market_spread", "over_under": "market_total"}
            ),
            on="game_id",
            how="left",
            suffixes=("", "_from_csv"),
        )

        # Use _from_csv version if market_spread was previously None
        for col in ["market_spread", "market_total"]:
            dup = f"{col}_from_csv"
            if dup in records.columns:
                records[col] = records[col].combine_first(records[dup])
                records = records.drop(columns=[dup])

        lined = records["market_spread"].notna().sum()
        log.info(f"Market lines attached: {lined:,}/{len(records):,} games have a spread")
        return records


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_full_report(
    records: pd.DataFrame,
    config: BacktestConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-model metrics report and calibration curve.
    Returns (model_report_df, calibration_df).
    """
    model_map = {
        "FourFactors":   ("fourfactors_spread",   "fourfactors_total",   "fourfactors_conf"),
        "AdjEfficiency": ("adjefficiency_spread",  "adjefficiency_total", "adjefficiency_conf"),
        "Pythagorean":   ("pythagorean_spread",    "pythagorean_total",   "pythagorean_conf"),
        "Momentum":      ("momentum_spread",       "momentum_total",      "momentum_conf"),
        "Situational":   ("situational_spread",    "situational_total",   "situational_conf"),
        "CAGERankings":  ("cagerankings_spread",   "cagerankings_total",  "cagerankings_conf"),
        "RegressedEff":  ("regressedeff_spread",   "regressedeff_total",  "regressedeff_conf"),
        "Ensemble":      ("ens_spread",            None,                   "ens_confidence"),
    }

    report_rows = []
    for name, (sp_col, tot_col, conf_col) in model_map.items():
        if sp_col not in records.columns:
            continue
        m = compute_model_metrics(
            records, sp_col,
            tot_col or "ens_spread",
            conf_col, name,
            config.edge_threshold,
        )
        report_rows.append(m.to_dict())

    report_df = pd.DataFrame(report_rows)

    # Calibration curve on ensemble
    calib_df = build_calibration_curve(records, "ens_confidence", config.confidence_bins)

    return report_df, calib_df


def print_report(report_df: pd.DataFrame, records: pd.DataFrame) -> None:
    """Print backtest summary to stdout."""
    print()
    print("=" * 100)
    print("  BACKTEST RESULTS — All Models")
    print(f"  {len(records):,} games backtested  |  "
          f"{records['market_spread'].notna().sum():,} with market lines")
    print("=" * 100)
    print(
        f"  {'MODEL':<16} {'N':>5} {'MAE':>6} {'RMSE':>6} {'BIAS':>6} "
        f"{'WIN%':>6} {'ATS%':>6} {'O/U%':>6} {'BRIER':>6} "
        f"{'EDGE_N':>7} {'EDGE_ATS':>9} {'EDGE_ROI':>9}"
    )
    print("  " + "-" * 96)

    for _, row in report_df.iterrows():
        ats_flag = " ⚡" if float(row.get("ats_pct", 0)) > VIG_BREAK_EVEN else ""
        print(
            f"  {str(row['model_name']):<16} "
            f"{int(row.get('n_games', 0)):>5} "
            f"{float(row.get('spread_mae', 0)):>6.2f} "
            f"{float(row.get('spread_rmse', 0)):>6.2f} "
            f"{float(row.get('spread_bias', 0)):>+6.2f} "
            f"{float(row.get('win_pct', 0)):>5.1f}% "
            f"{float(row.get('ats_pct', 0)):>5.1f}%{ats_flag:2} "
            f"{float(row.get('ou_pct', 0)):>5.1f}% "
            f"{float(row.get('brier_score', 0)):>6.4f} "
            f"{int(row.get('n_edge_games', 0)):>7} "
            f"{float(row.get('edge_ats_pct', 0)):>8.1f}% "
            f"{float(row.get('edge_roi_sim', 0)):>+8.1f}%"
        )

    print("  " + "=" * 96)
    print(f"  ATS break-even at -110 juice: {VIG_BREAK_EVEN:.2f}%  |  ⚡ = beats vig")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run(config: BacktestConfig = None, output_dir: Path = DATA_DIR) -> Dict[str, Path]:
    """Full backtest pipeline. Returns paths to output files."""
    config = config or BacktestConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    today  = datetime.now().strftime("%Y%m%d")

    # ── Load data ────────────────────────────────────────────────────────────
    game_log       = load_game_log()
    completed_games = load_completed_games(game_log)

    if completed_games.empty:
        log.error("No completed games found")
        return {}

    # ── Run backtest ─────────────────────────────────────────────────────────
    engine  = BacktestEngine(config)
    records = engine.run(game_log, completed_games)

    if records.empty:
        log.error("No records produced — check data and min_games threshold")
        return {}

    records = engine.attach_market_lines(records)

    # ── Build reports ─────────────────────────────────────────────────────────
    report_df, calib_df = build_full_report(records, config)
    print_report(report_df, records)

    # ── Weight optimization ───────────────────────────────────────────────────
    optimized_weights = {}
    if config.optimize_weights:
        log.info(f"Optimizing ensemble weights (metric={config.optimizer_metric})...")
        opt_weights, opt_metric = optimize_weights(
            records, metric=config.optimizer_metric, method=config.optimizer_method
        )
        optimized_weights = {
            "weights":          opt_weights,
            "metric":           config.optimizer_metric,
            "value":            opt_metric,
            "default_weights":  {n: round(float(w), 4)
                                  for n, w in zip(MODEL_NAMES, DEFAULT_WEIGHTS)},
            "n_games_used":     int(records["market_spread"].notna().sum()),
            "generated_at":     datetime.now().isoformat(),
        }

        print()
        print("  OPTIMIZED ENSEMBLE WEIGHTS")
        print("  " + "-" * 50)
        print(f"  {'MODEL':<16} {'DEFAULT':>9} {'OPTIMIZED':>10}")
        default_dict = dict(zip(MODEL_NAMES, DEFAULT_WEIGHTS))
        for name, opt_w in opt_weights.items():
            def_w = default_dict.get(name, 0.0)
            delta = opt_w - def_w
            print(f"  {name:<16} {def_w:>9.4f} {opt_w:>10.4f}  ({delta:+.4f})")
        print(f"\n  Optimized {config.optimizer_metric.upper()}: {opt_metric:.4f}")
        print()

    # ── Write outputs ─────────────────────────────────────────────────────────
    outputs = {}

    results_path = output_dir / f"backtest_results_{today}.csv"
    records.to_csv(results_path, index=False)
    outputs["results"] = results_path
    log.info(f"Results → {results_path}  ({len(records):,} rows)")

    report_path = output_dir / f"backtest_model_report_{today}.csv"
    report_df.to_csv(report_path, index=False)
    outputs["report"] = report_path
    log.info(f"Report  → {report_path}")

    if not calib_df.empty:
        calib_path = output_dir / f"backtest_calibration_{today}.csv"
        calib_df.to_csv(calib_path, index=False)
        outputs["calibration"] = calib_path
        log.info(f"Calibration → {calib_path}")

    if optimized_weights:
        weights_path = output_dir / "backtest_optimized_weights.json"
        with open(weights_path, "w") as f:
            json.dump(optimized_weights, f, indent=2)
        outputs["weights"] = weights_path
        log.info(f"Weights → {weights_path}")

    return outputs


def main():
    parser = argparse.ArgumentParser(description="CBB Prediction Backtester")
    parser.add_argument("--min-games",    type=int, default=5)
    parser.add_argument("--start-date",   type=str, default=None, help="YYYYMMDD")
    parser.add_argument("--end-date",     type=str, default=None)
    parser.add_argument("--edge-threshold", type=float, default=3.0)
    parser.add_argument("--optimize-weights", action="store_true", default=True)
    parser.add_argument("--optimizer-metric", choices=["ats","mae","brier"], default="ats")
    parser.add_argument("--output-dir",   type=Path, default=DATA_DIR)
    parser.add_argument("--no-optimize",  action="store_true",
                        help="Skip weight optimization (faster)")
    args = parser.parse_args()

    config = BacktestConfig(
        min_games_required  = args.min_games,
        start_date          = args.start_date,
        end_date            = args.end_date,
        edge_threshold      = args.edge_threshold,
        optimize_weights    = not args.no_optimize,
        optimizer_metric    = args.optimizer_metric,
    )

    run(config, args.output_dir)


if __name__ == "__main__":
    main()
