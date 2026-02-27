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
import random
import json
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from config.logging_config import get_logger
from cbb_config import (
    ENSEMBLE_MODEL_NAMES,
    DEFAULT_SPREAD_WEIGHTS as CONFIG_DEFAULT_SPREAD_WEIGHTS,
)
from espn_config import PIPELINE_RUN_ID
from pipeline_csv_utils import safe_write_csv
from pipeline.id_utils import canonicalize_espn_game_id

warnings.filterwarnings("ignore")

log = get_logger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR     = Path("data")
DATA_CSV_DIR = DATA_DIR / "csv"
WEIGHTED_CSV = DATA_DIR / "team_game_weighted.csv"
METRICS_CSV  = DATA_DIR / "team_game_metrics.csv"
GAMES_CSV    = DATA_DIR / "games.csv"
GRADED_RESULTS_CSV = DATA_DIR / "results_log_graded.csv"
TIER_CLASSIFICATIONS_CSV = DATA_DIR / "tier_classifications.csv"
MODEL_ACCURACY_REPORT_CSV = DATA_DIR / "model_accuracy_report.csv"
MODEL_ACCURACY_BY_DIMENSION_CSV = DATA_DIR / "model_accuracy_by_dimension.csv"
STACKING_COEFFICIENTS_JSON = DATA_DIR / "stacking_coefficients.json"


def _resolve_data_path(primary: Path) -> Path:
    """Resolve files that may live under data/ or data/csv/."""
    if primary.exists():
        return primary

    fallback = DATA_CSV_DIR / primary.name
    if fallback.exists():
        log.info(f"Using fallback data path: {fallback}")
        return fallback

    return primary

# ── Constants ─────────────────────────────────────────────────────────────────
LEAGUE_AVG_ORTG = 103.0
LEAGUE_AVG_DRTG = 103.0
LEAGUE_AVG_PACE = 70.0
LEAGUE_AVG_EFG  = 50.5
LEAGUE_AVG_TOV  = 18.0
LEAGUE_AVG_FTR  = 28.0
VIG_BREAK_EVEN  = 52.38   # % ATS needed to break even at -110 juice
UNIT_WIN = 100 / 110

MIN_SAMPLE = {
    "ats_pct": 10,
    "clv_positive_rate": 10,
    "home_ats_pct": 5,
    "edge_ats_pct": 5,
    "ou_pct": 10,
    "calibration_error": 20,
}

CONFERENCE_BASE_TIER = {
    "HIGH": 3,
    "MID": 2,
    "LOW": 1,
    "UNKNOWN": 0,
}

HIGH_TIER_CONFERENCES = {
    "atlantic coast conference", "acc", "big ten", "big 12", "big east",
    "southeastern conference", "sec", "pac-12", "pac 12", "pac-10", "pac 10",
}
MID_TIER_CONFERENCES = {
    "american athletic conference", "aac", "atlantic 10", "a-10", "a10",
    "mountain west", "west coast conference", "wcc", "missouri valley conference", "mvc",
}


def _append_weight_history_snapshot(history_path: Path, current_weights: dict) -> None:
    snapshots = []
    if history_path.exists() and history_path.stat().st_size > 0:
        try:
            snapshots = json.loads(history_path.read_text())
            if not isinstance(snapshots, list):
                snapshots = []
        except Exception:
            snapshots = []
    snapshots.append(current_weights)
    history_path.write_text(json.dumps(snapshots, indent=2))


def train_stacking_meta_model(backtest_results: pd.DataFrame, output_dir: Path) -> Optional[Dict[str, object]]:
    """Train ridge stacker on model outputs and validate vs weighted average on holdout."""
    model_alias_map = {
        "m1_spread": ["fourfactors_spread"],
        "m2_spread": ["adjefficiency_spread"],
        "m3_spread": ["pythagorean_spread"],
        "m4_spread": ["situational_spread"],
        "m5_spread": ["cagerankings_spread"],
        "m6_spread": ["luckregression_spread"],
        "m7_spread": ["variance_spread"],
        "m8_spread": ["homeawayform_spread"],
    }

    stack_df = backtest_results.copy()
    for alias, candidates in model_alias_map.items():
        if alias in stack_df.columns:
            continue
        src = next((col for col in candidates if col in stack_df.columns), None)
        if src:
            stack_df[alias] = pd.to_numeric(stack_df[src], errors="coerce")

    model_cols = ["m1_spread", "m2_spread", "m3_spread", "m4_spread", "m5_spread", "m6_spread", "m7_spread", "m8_spread"]
    aux_cols = ["cage_edge", "barthag_diff"]
    feature_cols = [c for c in model_cols + aux_cols if c in stack_df.columns]
    stacker_df = stack_df[feature_cols + ["actual_margin", "ens_spread"]].dropna()

    if len(stacker_df) < 80 or not feature_cols:
        log.info(
            "Skipping stacking meta-model training: need >=80 rows and non-empty features (rows=%d, features=%d)",
            len(stacker_df),
            len(feature_cols),
        )
        return None

    holdout_n = max(16, int(len(stacker_df) * 0.20))
    train_df = stacker_df.iloc[:-holdout_n].copy()
    holdout_df = stacker_df.iloc[-holdout_n:].copy()

    if len(train_df) < 50 or holdout_df.empty:
        log.info(
            "Skipping stacking validation split: train=%d holdout=%d",
            len(train_df),
            len(holdout_df),
        )
        return None

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["actual_margin"].to_numpy()
    X_holdout = holdout_df[feature_cols].to_numpy()
    y_holdout = holdout_df["actual_margin"].to_numpy()

    stacker = Ridge(alpha=1.0)
    stacker.fit(X_train, y_train)

    train_mae = float(np.abs(stacker.predict(X_train) - y_train).mean())
    ridge_holdout_mae = float(np.abs(stacker.predict(X_holdout) - y_holdout).mean())
    weighted_holdout_mae = float(np.abs(holdout_df["ens_spread"].to_numpy() - y_holdout).mean())
    improvement = weighted_holdout_mae - ridge_holdout_mae
    use_stacking_recommended = improvement >= 0.3

    stacking_params = {
        "coef": stacker.coef_.tolist(),
        "intercept": float(stacker.intercept_),
        "features": feature_cols,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "n_samples": len(stacker_df),
        "train_samples": len(train_df),
        "holdout_samples": len(holdout_df),
        "train_mae": train_mae,
        "holdout_mae_ridge": ridge_holdout_mae,
        "holdout_mae_weighted": weighted_holdout_mae,
        "holdout_mae_improvement": improvement,
        "use_stacking_recommended": use_stacking_recommended,
    }
    output_path = output_dir / STACKING_COEFFICIENTS_JSON.name
    output_path.write_text(json.dumps(stacking_params, indent=2))
    log.info(
        "Stacking meta-model trained: train_MAE=%.2f | holdout ridge=%.2f vs weighted=%.2f | use_stacking=%s",
        train_mae,
        ridge_holdout_mae,
        weighted_holdout_mae,
        use_stacking_recommended,
    )

    return stacking_params


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
    for path in [_resolve_data_path(WEIGHTED_CSV), _resolve_data_path(METRICS_CSV)]:
        if path.exists() and path.stat().st_size > 100:
            log.info(f"Loading game log: {path.name}")
            df = pd.read_csv(path, dtype=str, low_memory=False)
            required = {"team_id", "event_id", "game_datetime_utc", "home_away", "points_for", "points_against"}
            missing = sorted(required - set(df.columns))
            if missing:
                raise ValueError(
                    f"{path} is missing required columns for backtesting: {missing}. "
                    f"Available columns: {sorted(df.columns)[:25]}..."
                )
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
    gl["event_id"] = gl["event_id"].apply(canonicalize_espn_game_id)

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

    # Defensive drop before renaming to avoid duplicate columns
    for col in ["home_team_id", "home_team", "away_team_id", "away_team", "conference"]:
        if col in home.columns:
            home = home.drop(columns=[col])
    # Avoid duplicate columns by dropping existing ones before rename
    drop_cols = ["home_team_id", "away_team_id", "home_team", "away_team", "home_score", "away_score"]
    home = home.drop(columns=[c for c in drop_cols if c in home.columns])

    home = home.rename(columns={
        "team_id": "home_team_id", "team": "home_team",
        "opponent_id": "away_team_id", "opponent": "away_team",
        "points_for": "home_score", "points_against": "away_score",
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

    if len(team_rows) < max(1, min_games):
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
    Run all active ensemble sub-models on pre-game team states.
    Returns dict with per-model predictions + ensemble.

    weights: if provided, overrides EnsembleConfig defaults (for optimizer).
    """
    from cbb_ensemble import (
        EnsemblePredictor, EnsembleConfig,
        FourFactorsModel, AdjustedEfficiencyModel, PythagoreanModel,
        MomentumModel, SituationalModel, CAGERankingsModel,
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
        mkt_spread  = ats_df["market_spread"].values    # negative = home favored (convention)
        act_margin  = ats_df["actual_margin"].values
        pred_margin = -ats_df[spread_col].values        # pred_margin = -spread (positive = home win by)

        # ATS: did actual margin beat the market spread?
        # Convention: market_spread -7 means home is favorite by 7.
        # Home covers if actual_margin > 7 (i.e. actual_margin > -market_spread)
        home_cover = act_margin > (-mkt_spread)     # push = loss in this calc
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

# NOTE: If cbb_ensemble model composition changes (add/remove/merge models),
# retrain this stacking/meta-model and refresh MODEL_NAMES/DEFAULT_WEIGHTS before backtesting.

MODEL_NAMES = list(ENSEMBLE_MODEL_NAMES)

DEFAULT_WEIGHTS = np.array([CONFIG_DEFAULT_SPREAD_WEIGHTS[name] for name in MODEL_NAMES], dtype=float)


def _ensemble_from_weights(records: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """Compute ensemble spread from per-model spreads with given weights."""
    w = np.abs(weights) / np.abs(weights).sum()   # Normalize, force positive
    cols = [f"{n.lower()}_spread" for n in MODEL_NAMES]

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
    model_cols = [f"{n.lower()}_spread" for n in MODEL_NAMES if f"{n.lower()}_spread" in ats_df.columns]
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
        skip_samples = []

        # History rows must be truly completed (same definition as completed_games)
        history_log = game_log.dropna(subset=["points_for", "points_against"]).copy()
        history_log = history_log[history_log["points_for"] > 0]

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

        for i, (_, row) in enumerate(games.iterrows(), start=1):
            if (i % 500) == 0 or i == total:
                pct = (i / total * 100.0) if total else 100.0
                log.info(f"  Progress: {i:,}/{total:,} games ({pct:.0f}%)")

            game_dt    = row["game_datetime_utc"]
            home_id    = str(row["home_team_id"])
            away_id    = str(row["away_team_id"])
            event_id   = str(row["event_id"])
            neutral    = bool(row.get("neutral_site", False))

            # ── Build pre-game team states (ZERO LOOKAHEAD) ──────────────────
            home_state = build_team_state_before(
                home_id, game_dt, history_log, cfg.min_games_required
            )
            away_state = build_team_state_before(
                away_id, game_dt, history_log, cfg.min_games_required
            )

            if home_state is None or away_state is None:
                skipped["too_few_games"] += 1
                if len(skip_samples) < 10:
                    home_prior = history_log[(history_log["team_id"].astype(str) == home_id) & (history_log["game_datetime_utc"] < game_dt)]
                    away_prior = history_log[(history_log["team_id"].astype(str) == away_id) & (history_log["game_datetime_utc"] < game_dt)]
                    skip_samples.append({
                        "event_id": event_id,
                        "game_datetime": str(game_dt),
                        "home_team": str(row.get("home_team", "")),
                        "away_team": str(row.get("away_team", "")),
                        "home_prior_games": int(len(home_prior)),
                        "away_prior_games": int(len(away_prior)),
                        "reason": f"min_games_required={cfg.min_games_required}",
                    })
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
                "game_id":          canonicalize_espn_game_id(event_id),
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

        if not records:
            eval_min = games["game_datetime_utc"].min() if not games.empty else pd.NaT
            eval_max = games["game_datetime_utc"].max() if not games.empty else pd.NaT
            hist_min = history_log["game_datetime_utc"].min() if not history_log.empty else pd.NaT
            hist_max = history_log["game_datetime_utc"].max() if not history_log.empty else pd.NaT
            log.error("No predictions produced diagnostics:")
            log.error("  eval date range: %s → %s", eval_min, eval_max)
            log.error("  history date range: %s → %s", hist_min, hist_max)
            log.error("  min-games threshold: %s", cfg.min_games_required)
            if skip_samples:
                log.error("  first skipped games sample: %s", skip_samples)

        return pd.DataFrame(records)

    def attach_market_lines(
        self,
        records: pd.DataFrame,
        closing_lines_path: Path = DATA_DIR / "market_lines_closing.csv",
        fallback_lines_path: Path = DATA_DIR / "market_lines.csv",
    ) -> pd.DataFrame:
        """
        Join market spread / over-under onto backtest records.
        Prefers market_lines_closing.csv, falls back to market_lines.csv when needed.
        Also computes ATS and Total margins for actual and predicted.
        """
        gdf: Optional[pd.DataFrame] = None
        source_label = ""

        if closing_lines_path.exists():
            gdf = pd.read_csv(closing_lines_path, dtype={"espn_game_id": str})
            gdf = gdf.rename(columns={
                "close_home_spread": "home_market_spread",
                "close_total": "market_total",
            })
            source_label = closing_lines_path.name
        elif fallback_lines_path.exists():
            gdf = pd.read_csv(fallback_lines_path, dtype={"event_id": str}, low_memory=False)
            gdf = gdf.rename(columns={
                "event_id": "espn_game_id",
                "home_spread_current": "home_market_spread",
                "spread": "home_market_spread_alt",
                "total_current": "market_total",
                "over_under": "market_total_alt",
                "total": "market_total_alt2",
            })
            if "home_market_spread" not in gdf.columns:
                gdf["home_market_spread"] = np.nan
            if "home_market_spread_alt" in gdf.columns:
                gdf["home_market_spread"] = pd.to_numeric(gdf["home_market_spread"], errors="coerce").fillna(
                    pd.to_numeric(gdf["home_market_spread_alt"], errors="coerce")
                )
            if "market_total" not in gdf.columns:
                gdf["market_total"] = np.nan
            for alt_col in ["market_total_alt", "market_total_alt2"]:
                if alt_col in gdf.columns:
                    gdf["market_total"] = pd.to_numeric(gdf["market_total"], errors="coerce").fillna(
                        pd.to_numeric(gdf[alt_col], errors="coerce")
                    )

            # Deterministic dedupe: keep the latest capture per game_id.
            ts_col = next((c for c in ["captured_at_utc", "pulled_at_utc"] if c in gdf.columns), None)
            if ts_col:
                gdf[ts_col] = pd.to_datetime(gdf[ts_col], errors="coerce", utc=True)
                gdf = gdf.sort_values(["espn_game_id", ts_col]).drop_duplicates("espn_game_id", keep="last")
            else:
                gdf = gdf.drop_duplicates("espn_game_id", keep="last")
            source_label = fallback_lines_path.name
        else:
            log.warning(
                "%s and %s not found — skipping market line attachment",
                closing_lines_path,
                fallback_lines_path,
            )
            return records

        if "espn_game_id" not in gdf.columns:
            log.warning("Market line source %s missing game-id column — skipping attachment", source_label)
            return records

        gdf["espn_game_id"] = gdf["espn_game_id"].apply(canonicalize_espn_game_id)

        merge_cols = ["espn_game_id", "home_market_spread", "market_total"]
        gdf = gdf[[c for c in merge_cols if c in gdf.columns]].copy()

        # Merge on canonical espn_game_id
        records["game_id"] = records["game_id"].apply(canonicalize_espn_game_id)

        records = records.merge(
            gdf[["espn_game_id", "home_market_spread", "market_total"]],
            left_on="game_id",
            right_on="espn_game_id",
            how="left"
        )

        # Cleanup
        if "espn_game_id" in records.columns:
            records = records.drop(columns=["espn_game_id"])

        # Compute extended schema
        # A) away_market_spread
        records["away_market_spread"] = -records["home_market_spread"]

        # B) Actual margins
        # actual_margin_ATS = actual_home_margin + home_market_spread
        # (Using + because home_market_spread is negative for home favorites)
        records["actual_margin_ATS"] = records["actual_margin"] + records["home_market_spread"]
        records["actual_margin_total"] = records["actual_total"] - records["market_total"]

        # C) Predicted vs Market
        # home_market_spread is from HOME perspective (negative = home favorite)
        # ens_spread is ALSO from HOME perspective
        records["pred_home_spread"] = records["ens_spread"]
        records["pred_away_spread"] = -records["pred_home_spread"]
        records["pred_total"] = records["ens_total"]

        # pred_margin_ATS = pred_home_margin + home_market_spread
        # (where pred_home_margin = -ens_spread)
        records["pred_margin_ATS"] = (-records["ens_spread"]) + records["home_market_spread"]
        records["pred_margin_total"] = records["pred_total"] - records["market_total"]

        # Legacy field removal/bridging
        if "market_spread" in records.columns:
            records = records.drop(columns=["market_spread"])
        # We'll use home_market_spread as the primary for metrics computation below
        records["market_spread"] = records["home_market_spread"]

        # D) Column ordering
        # Insert new headers immediately to the RIGHT of: cage_em_diff
        cols = records.columns.tolist()
        if "cage_em_diff" in cols:
            idx = cols.index("cage_em_diff") + 1
            new_cols_ordered = [
                "home_market_spread", "away_market_spread", "market_total",
                "actual_margin_ATS", "actual_margin_total",
                "pred_home_spread", "pred_away_spread", "pred_total",
                "pred_margin_ATS", "pred_margin_total"
            ]
            # Remove them from their current positions first
            other_cols = [c for c in cols if c not in new_cols_ordered]
            # Re-insert
            idx = other_cols.index("cage_em_diff") + 1
            final_cols = other_cols[:idx] + new_cols_ordered + other_cols[idx:]
            records = records[final_cols]

        lined = records["home_market_spread"].notna().sum()
        log.info(
            "Market lines attached (%s): %d/%d games have a spread",
            source_label,
            lined,
            len(records),
        )
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
        "FourFactors":    ("fourfactors_spread",    "fourfactors_total",    "fourfactors_conf"),
        "AdjEfficiency":  ("adjefficiency_spread",   "adjefficiency_total",  "adjefficiency_conf"),
        "Pythagorean":    ("pythagorean_spread",     "pythagorean_total",    "pythagorean_conf"),
        "Situational":    ("situational_spread",     "situational_total",    "situational_conf"),
        "CAGERankings":   ("cagerankings_spread",    "cagerankings_total",   "cagerankings_conf"),
        "LuckRegression": ("luckregression_spread",  "luckregression_total", "luckregression_conf"),
        "Variance":       ("variance_spread",        "variance_total",       "variance_conf"),
        "HomeAwayForm":   ("homeawayform_spread",    "homeawayform_total",   "homeawayform_conf"),
        "Ensemble":       ("ens_spread",             None,                     "ens_confidence"),
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
    print("=" * 100)
    print("  BACKTEST RESULTS — All Models")
    mkt_count = records['home_market_spread'].notna().sum() if 'home_market_spread' in records.columns else 0
    print(f"  {len(records):,} games backtested  |  "
          f"{mkt_count:,} with market lines")
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


def _resolve_optional_csv(path: Path) -> Optional[Path]:
    resolved = _resolve_data_path(path)
    if resolved.exists() and resolved.stat().st_size > 10:
        return resolved
    return None


def _safe_rate(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, np.nan)
    return numerator / denom


def _wilson_ci(wins: float, losses: float, z: float = 1.96) -> Tuple[float, float]:
    n = wins + losses
    if n <= 0:
        return (np.nan, np.nan)
    p = wins / n
    denom = 1 + (z ** 2 / n)
    center = (p + z ** 2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n) + (z ** 2 / (4 * n ** 2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _add_sample_guards(backtest_df: pd.DataFrame) -> pd.DataFrame:
    df = backtest_df.copy()
    sample_map = {
        "ats_pct": "ats_sample",
        "clv_positive_rate": "clv_sample",
        "home_ats_pct": "home_ats_sample",
        "edge_ats_pct": "edge_ats_sample",
        "ou_pct": "ou_sample",
        "calibration_error": "calibration_sample",
    }

    for metric, threshold in MIN_SAMPLE.items():
        sample_col = sample_map.get(metric)
        if metric in df.columns and sample_col in df.columns:
            below_min = df[sample_col].fillna(0) < threshold
            df.loc[below_min, metric] = np.nan

    return df


def _assert_consistency(df: pd.DataFrame) -> None:
    checks = [
        ("home_wins + away_wins + neutral_wins", "wins", 2),
        ("ats_wins + ats_losses", "wins + losses", 5),
    ]

    for left_expr, right_expr, tolerance in checks:
        try:
            left = df.eval(left_expr)
            right = df.eval(right_expr)
            mismatch = (left - right).abs() > tolerance
            if mismatch.any():
                log.warning(
                    "Consistency check failed for %s ~ %s (%d rows)",
                    left_expr,
                    right_expr,
                    int(mismatch.sum()),
                )
        except Exception as exc:
            log.warning("Consistency check skipped for %s ~ %s: %s", left_expr, right_expr, exc)


def build_team_backtest_csv(output_dir: Path) -> Optional[Path]:
    metrics_path = _resolve_optional_csv(METRICS_CSV)
    if metrics_path is None:
        metrics_path = _resolve_optional_csv(WEIGHTED_CSV)
    if metrics_path is None:
        log.warning("Skipping team_season_summary.csv build: no team_game_metrics/team_game_weighted CSV found")
        return None

    tgm = pd.read_csv(metrics_path, low_memory=False)
    if tgm.empty:
        log.warning("Skipping team_season_summary.csv build: team game metrics file is empty")
        return None

    tgm["team_id"] = tgm["team_id"].astype(str)
    tgm["home_away_norm"] = tgm.get("home_away", "").astype(str).str.lower()
    tgm["conference"] = tgm.get("conference", np.nan)
    tgm["opp_rest_days"] = pd.to_numeric(tgm.get("opp_rest_days"), errors="coerce")
    tgm["rest_days"] = pd.to_numeric(tgm.get("rest_days"), errors="coerce")
    tgm["cover"] = pd.to_numeric(tgm.get("cover"), errors="coerce")
    tgm["ats_push"] = pd.to_numeric(tgm.get("ats_push"), errors="coerce").fillna(0)
    tgm["game_datetime_utc"] = pd.to_datetime(tgm.get("game_datetime_utc"), errors="coerce", utc=True)
    tgm["win"] = pd.to_numeric(tgm.get("win"), errors="coerce").fillna(0)
    tgm["loss_flag"] = (tgm["win"] == 0).astype(int)
    tgm["neutral_site"] = pd.to_numeric(tgm.get("neutral_site"), errors="coerce").fillna(0)

    # Calculate win flags for aggregation (Q2)
    tgm["home_win_flag"] = ((tgm["home_away_norm"] == "home") & (tgm["win"] == 1)).astype(int)
    tgm["away_win_flag"] = ((tgm["home_away_norm"] == "away") & (tgm["win"] == 1)).astype(int)
    tgm["neutral_win_flag"] = ((tgm["neutral_site"] == 1) & (tgm["win"] == 1)).astype(int)

    required = {"team_id", "team", "wins", "losses", "points_for", "points_against"}
    missing_required = sorted(required - set(tgm.columns))
    if missing_required:
        raise ValueError(
            f"Cannot build team_season_summary.csv: missing required columns {missing_required}. "
            f"Source file: {metrics_path}"
        )

    grouped = tgm.groupby("team_id", dropna=False)

    def _series(col_name: str, default=np.nan) -> pd.Series:
        if col_name in tgm.columns:
            return tgm[col_name]
        return pd.Series(default, index=tgm.index)

    # Correct win counting: group by team and sum 'win' column where condition matches
    if "win" not in tgm.columns:
        tgm["win"] = (tgm["points_for"] > tgm["points_against"]).astype(int)

    home_wins = tgm[tgm["home_away_norm"] == "home"].groupby("team_id")["win"].sum()
    away_wins = tgm[tgm["home_away_norm"] == "away"].groupby("team_id")["win"].sum()
    neutral_wins = tgm[pd.to_numeric(tgm["neutral_site"], errors="coerce") == 1].groupby("team_id")["win"].sum()

    agg_spec = {
        "team": ("team", "last"),
        "conference": ("conference", "last"),
        "wins": ("win", "sum"),
        "losses": ("loss_flag", "sum"),
        "home_wins": ("home_win_flag", "sum"),
        "away_wins": ("away_win_flag", "sum"),
        "neutral_wins": ("neutral_win_flag", "sum"),
        "ats_wins": ("cover", lambda s: int((s == 1).sum())),
        "ats_losses": ("cover", lambda s: int((s == 0).sum())),
        "ats_pushes": ("ats_push", "sum"),
        "avg_total_scored": ("points_for", "mean"),
        "avg_ou_line": ("over_under", "mean"),
        "h2_avg_pts": ("h2_pts", "mean"),
        "h2_avg_pts_against": ("h2_pts_against", "mean"),
        "ortg_l5": ("ortg_l5", "last"),
        "ortg_l10": ("ortg_l10", "last"),
        "drtg_l5": ("drtg_l5", "last"),
        "drtg_l10": ("drtg_l10", "last"),
        "net_rtg_l5": ("net_rtg_l5", "last"),
        "net_rtg_l10": ("net_rtg_l10", "last"),
        "net_rtg_std": ("net_rtg_std_l10", "last"),
        "luck_score": ("luck_score", "last"),
        "pyth_win_pct": ("pyth_win_pct_season", "last"),
        "avg_margin_l10": ("margin_l10", "last"),
        "avg_margin_l5": ("margin_l5", "last"),
        "avg_opp_net_rtg": ("opp_avg_net_rtg_season", "last"),
        "data_through_date": ("game_datetime_utc", "max"),
        "cover_streak": ("cover_streak", "last"),
    }
    missing_optional = []
    resolved_agg = {}
    for out_col, agg in agg_spec.items():
        source_col = agg[0]
        if source_col in tgm.columns:
            resolved_agg[out_col] = agg
        elif out_col == "avg_opp_net_rtg" and "opp_avg_ortg_season" in tgm.columns:
            resolved_agg[out_col] = ("opp_avg_ortg_season", "last")
        else:
            missing_optional.append(source_col)

    if missing_optional:
        # Q5: Downgrade to debug; some columns like opp_avg_net_rtg_season may be joined later
        log.debug("build_team_backtest_csv missing optional columns (may be available post-merge): %s", sorted(set(missing_optional)))

    base = grouped.agg(
        **resolved_agg,
    ).reset_index()

    base["ats_sample"] = base["ats_wins"] + base["ats_losses"]
    if "avg_opp_net_rtg" not in base.columns:
        base["avg_opp_net_rtg"] = np.nan
    base["ats_pct"] = _safe_rate(base["ats_wins"], base["ats_sample"])
    base["avg_ou_margin"] = base["avg_total_scored"] - base["avg_ou_line"]

    splits = []
    split_specs = {
        "home_ats_pct": tgm["home_away_norm"] == "home",
        "away_ats_pct": tgm["home_away_norm"] == "away",
        "close_game_ats_pct": pd.to_numeric(_series("close_game_flag"), errors="coerce") == 1,
        "blowout_ats_pct": pd.to_numeric(_series("blowout_flag"), errors="coerce") == 1,
        "conf_game_ats_pct": pd.to_numeric(_series("conf_game_flag"), errors="coerce") == 1,
        "nonconf_ats_pct": pd.to_numeric(_series("conf_game_flag"), errors="coerce") == 0,
        "favorite_ats_pct": pd.to_numeric(_series("spread"), errors="coerce") < 0,
        "underdog_ats_pct": pd.to_numeric(_series("spread"), errors="coerce") > 0,
        "rest_advantage_ats_pct": tgm["rest_days"] > tgm["opp_rest_days"],
    }
    for metric, condition in split_specs.items():
        subset = tgm[condition.fillna(False)].copy()
        if subset.empty:
            continue
        agg = subset.groupby("team_id")["cover"].agg(
            **{f"{metric}_wins": lambda s: int((s == 1).sum()), f"{metric}_losses": lambda s: int((s == 0).sum())}
        ).reset_index()
        agg[f"{metric}_sample"] = agg[f"{metric}_wins"] + agg[f"{metric}_losses"]
        agg[metric] = _safe_rate(agg[f"{metric}_wins"], agg[f"{metric}_sample"])
        splits.append(agg[["team_id", metric, f"{metric}_sample"]])

    for split_df in splits:
        base = base.merge(split_df, on="team_id", how="left")

    # Re-inject corrected win counts
    base["home_wins"] = base["team_id"].map(home_wins).fillna(0)
    base["away_wins"] = base["team_id"].map(away_wins).fillna(0)
    base["neutral_wins"] = base["team_id"].map(neutral_wins).fillna(0)

    # Post-merge data quality check for opp_avg_net_rtg_season
    if "avg_opp_net_rtg" not in base.columns or base["avg_opp_net_rtg"].isna().all():
        log.warning("build_team_backtest_csv: avg_opp_net_rtg (opp_avg_net_rtg_season) is missing post-merge")

    ats_l10 = grouped["cover"].apply(lambda s: (s.tail(10) == 1).mean()).rename("ats_pct_l10")
    ats_l20 = grouped["cover"].apply(lambda s: (s.tail(20) == 1).mean()).rename("ats_pct_l20")
    base = base.merge(ats_l10, on="team_id", how="left").merge(ats_l20, on="team_id", how="left")
    base["ortg_trend"] = pd.to_numeric(base["ortg_l5"], errors="coerce") - pd.to_numeric(base["ortg_l10"], errors="coerce")
    base["drtg_trend"] = pd.to_numeric(base["drtg_l5"], errors="coerce") - pd.to_numeric(base["drtg_l10"], errors="coerce")
    base["net_rtg_trend"] = pd.to_numeric(base["net_rtg_l5"], errors="coerce") - pd.to_numeric(base["net_rtg_l10"], errors="coerce")
    base["regression_risk_flag"] = (
        pd.to_numeric(base["net_rtg_l5"], errors="coerce") >
        (pd.to_numeric(base["net_rtg_l10"], errors="coerce") + (2 * pd.to_numeric(base["net_rtg_std"], errors="coerce")))
    )
    base["sos_rank"] = pd.to_numeric(base["avg_opp_net_rtg"], errors="coerce").rank(ascending=False, method="average")

    if "actual_total" in tgm.columns:
        tgm["actual_total"] = pd.to_numeric(tgm["actual_total"], errors="coerce")
    else:
        tgm["actual_total"] = pd.to_numeric(tgm.get("points_for"), errors="coerce") + pd.to_numeric(tgm.get("points_against"), errors="coerce")
    tgm["ou_line"] = pd.to_numeric(tgm.get("over_under"), errors="coerce")
    tgm["ou_result"] = np.where(
        tgm["actual_total"] > tgm["ou_line"],
        1,
        np.where(tgm["actual_total"] < tgm["ou_line"], 0, np.nan),
    )
    tgm["h1_ou_result"] = np.where(
        (pd.to_numeric(tgm.get("h1_pts"), errors="coerce") + pd.to_numeric(tgm.get("h1_pts_against"), errors="coerce")) > (tgm["ou_line"] / 2),
        1,
        np.where((pd.to_numeric(tgm.get("h1_pts"), errors="coerce") + pd.to_numeric(tgm.get("h1_pts_against"), errors="coerce")) < (tgm["ou_line"] / 2), 0, np.nan),
    )
    ou_agg = tgm.groupby("team_id").agg(
        ou_wins=("ou_result", lambda s: int((s == 1).sum())),
        ou_losses=("ou_result", lambda s: int((s == 0).sum())),
        h1_ou_wins=("h1_ou_result", lambda s: int((s == 1).sum())),
        h1_ou_losses=("h1_ou_result", lambda s: int((s == 0).sum())),
    ).reset_index()
    ou_agg["ou_sample"] = ou_agg["ou_wins"] + ou_agg["ou_losses"]
    ou_agg["ou_pct"] = _safe_rate(ou_agg["ou_wins"], ou_agg["ou_sample"])
    ou_agg["h1_ou_sample"] = ou_agg["h1_ou_wins"] + ou_agg["h1_ou_losses"]
    ou_agg["h1_ou_pct"] = _safe_rate(ou_agg["h1_ou_wins"], ou_agg["h1_ou_sample"])
    base = base.merge(ou_agg[["team_id", "ou_pct", "ou_sample", "h1_ou_pct", "h1_ou_sample"]], on="team_id", how="left")

    graded_path = _resolve_optional_csv(GRADED_RESULTS_CSV)
    if graded_path is not None:
        rl = pd.read_csv(graded_path, low_memory=False)
        if "team_id" in rl.columns and not rl.empty:
            rl["team_id"] = rl["team_id"].astype(str)
            rl["cover"] = pd.to_numeric(rl.get("cover"), errors="coerce")
            rl["edge_flag"] = pd.to_numeric(rl.get("edge_flag"), errors="coerce").fillna(0)
            rl["model_confidence"] = pd.to_numeric(rl.get("model_confidence"), errors="coerce")
            rl["predicted_prob"] = rl.get("predicted_prob", rl["model_confidence"])
            rl["actual_outcome"] = np.where(pd.to_numeric(rl.get("actual_spread"), errors="coerce") < 0, 1.0, 0.0)

            rl_agg = rl.groupby("team_id").agg(
                directional_accuracy=("winner_correct", "mean"),
                ats_wins_rl=("ats_result", lambda s: int((s == "WIN").sum())),
                ats_losses_rl=("ats_result", lambda s: int((s == "LOSS").sum())),
                avg_edge_size=("edge_size", "mean"),
                edge_wins=("ats_result", lambda s: int(((s == "WIN") & (rl.loc[s.index, "edge_flag"] == 1)).sum())),
                edge_losses=("ats_result", lambda s: int(((s == "LOSS") & (rl.loc[s.index, "edge_flag"] == 1)).sum())),
                calibration_error=("predicted_prob", lambda s: float(np.abs(s - rl.loc[s.index, "actual_outcome"]).mean())),
                brier_score=("predicted_prob", lambda s: float(np.mean((s - rl.loc[s.index, "actual_outcome"]) ** 2))),
                avg_clv=("clv_vs_close", "mean"),
                clv_positive_wins=("clv_vs_close", lambda s: int((s > 0).sum())),
                clv_sample=("clv_vs_close", lambda s: int(s.notna().sum())),
                avg_clv_edge_games=("clv_vs_close", lambda s: float(s[rl.loc[s.index, "edge_flag"] == 1].mean())),
                clv_grade=("beat_closing_line", "mean"),
            ).reset_index()
            rl_agg["ats_sample"] = rl_agg["ats_wins_rl"] + rl_agg["ats_losses_rl"]
            rl_agg["ats_pct"] = _safe_rate(rl_agg["ats_wins_rl"], rl_agg["ats_sample"])
            rl_agg["edge_sample"] = rl_agg["edge_wins"] + rl_agg["edge_losses"]
            rl_agg["edge_ats_pct"] = _safe_rate(rl_agg["edge_wins"], rl_agg["edge_sample"])
            rl_agg["clv_positive_rate"] = _safe_rate(rl_agg["clv_positive_wins"], rl_agg["clv_sample"])
            rl_agg["roi_units"] = (rl_agg["ats_wins_rl"] * UNIT_WIN) - rl_agg["ats_losses_rl"]
            rl_agg["roi_units_edge_only"] = (rl_agg["edge_wins"] * UNIT_WIN) - rl_agg["edge_losses"]
            rl_agg["clv_roi"] = np.where(rl_agg["clv_positive_rate"].notna(), (rl_agg["clv_positive_rate"] * UNIT_WIN) - (1 - rl_agg["clv_positive_rate"]), np.nan)

            base = base.merge(
                rl_agg[[
                    "team_id", "directional_accuracy", "ats_pct", "roi_units", "roi_units_edge_only",
                    "avg_edge_size", "edge_ats_pct", "calibration_error", "brier_score", "avg_clv",
                    "clv_positive_rate", "clv_roi", "avg_clv_edge_games", "clv_grade", "clv_sample", "edge_sample",
                ]],
                on="team_id",
                how="left",
            )

    # Confidence intervals on ATS
    wilson = base.apply(lambda row: _wilson_ci(float(row.get("ats_wins", 0)), float(row.get("ats_losses", 0))), axis=1)
    base["ats_pct_95ci_low"] = wilson.apply(lambda x: x[0])
    base["ats_pct_95ci_high"] = wilson.apply(lambda x: x[1])

    # Tier join/fallback
    tier_path = _resolve_optional_csv(TIER_CLASSIFICATIONS_CSV)
    if tier_path is not None:
        tiers = pd.read_csv(tier_path, low_memory=False)
        if "team_id" in tiers.columns:
            tiers["team_id"] = tiers["team_id"].astype(str)
            cols = [c for c in ["team_id", "tier", "tier_score", "conference_base_tier"] if c in tiers.columns]
            base = base.merge(tiers[cols], on="team_id", how="left")

    if "tier" not in base.columns:
        from espn_config import get_conference_tier

        base["tier"] = base["conference"].apply(get_conference_tier)
        base["conference_base_tier"] = base["tier"]
        base["tier_score"] = base["tier"].map(CONFERENCE_BASE_TIER).fillna(0)
    else:
        if "conference_base_tier" not in base.columns:
            base["conference_base_tier"] = base["tier"]
        if "tier_score" not in base.columns:
            base["tier_score"] = base["conference_base_tier"].map(CONFERENCE_BASE_TIER).fillna(0)

    base = _add_sample_guards(base)
    _assert_consistency(base)

    out_path = output_dir / "team_season_summary.csv"
    safe_write_csv(base, out_path, index=False, label="team_season_summary", allow_empty=True)
    safe_write_csv(base, output_dir / "team_season_summary_latest.csv", index=False, label="team_season_summary_latest", allow_empty=True)
    log.info(
        "build_team_backtest_csv end: %d teams summarized | "
        "median home_wins: %.1f | neutral_wins range: %.1f-%.1f",
        len(base),
        base["home_wins"].median(),
        base["neutral_wins"].min(),
        base["neutral_wins"].max()
    )
    return out_path


def train_platt_calibration(backtest_results: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Fit Platt calibration and persist coefficients when quality is acceptable."""
    calib_features = ["ens_confidence", "spread_std", "cage_edge"]
    required_cols = calib_features + ["market_spread", "actual_margin", "ens_spread"]

    missing = [c for c in required_cols if c not in backtest_results.columns]
    if missing:
        log.warning("Skipping calibration: missing required columns %s", missing)
        return None

    calib_df = backtest_results[required_cols].dropna().copy()
    if calib_df.empty:
        log.warning("Skipping calibration: no complete rows after dropna")
        return None

    pred_margin = -calib_df["ens_spread"].values
    mkt_margin = -calib_df["market_spread"].values
    act_margin = calib_df["actual_margin"].values

    push_mask = act_margin == mkt_margin
    calib_df = calib_df.loc[~push_mask].copy()
    if len(calib_df) < 50:
        log.info("Skipping calibration: only %d ATS-labeled rows (need >= 50)", len(calib_df))
        return None

    pred_margin = -calib_df["ens_spread"].values
    mkt_margin = -calib_df["market_spread"].values
    act_margin = calib_df["actual_margin"].values
    model_home = pred_margin > mkt_margin
    home_cover = act_margin > mkt_margin
    calib_df["ats_correct"] = (model_home == home_cover).astype(int)

    X = calib_df[calib_features].values
    y = calib_df["ats_correct"].astype(int).values

    clf = LogisticRegression(C=1.0, max_iter=500)
    clf.fit(X, y)

    y_prob = clf.predict_proba(X)[:, 1]
    brier = float(brier_score_loss(y, y_prob))
    log.info("Calibration Brier score: %.4f", brier)

    if brier > 0.28:
        log.warning("Calibration rejected: Brier score %.4f > 0.28 baseline", brier)
        return None

    params = {
        "coef": clf.coef_[0].tolist(),
        "intercept": float(clf.intercept_[0]),
        "features": calib_features,
        "trained_at": pd.Timestamp.utcnow().isoformat(),
        "n_samples": len(calib_df),
        "brier_score": brier,
    }
    calib_path = output_dir / "calibration_params.json"
    calib_path.write_text(json.dumps(params, indent=2))
    log.info("Calibration layer trained on %d samples", len(calib_df))
    return calib_path


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
        outputs = {}
        backtest_path = build_team_backtest_csv(output_dir)
        if backtest_path is not None:
            outputs["team_season_summary"] = backtest_path
        return outputs

    # ── Run backtest ─────────────────────────────────────────────────────────
    engine  = BacktestEngine(config)
    records = engine.run(game_log, completed_games)

    if records.empty:
        log.error("No records produced — check data and min_games threshold")
        outputs = {}
        backtest_path = build_team_backtest_csv(output_dir)
        if backtest_path is not None:
            outputs["team_season_summary"] = backtest_path
        return outputs

    records = engine.attach_market_lines(records)

    stacking_params = train_stacking_meta_model(records, output_dir)

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
        optimized_total_weights = {n: round(float(w), 4) for n, w in zip(MODEL_NAMES, DEFAULT_WEIGHTS)}  # TODO: optimize total weights
        optimized_weights = {
            "weights":          opt_weights,
            "total_weights":    optimized_total_weights,
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

    # ── Train confidence calibration ─────────────────────────────────────────
    calibration_path = train_platt_calibration(records, output_dir)

    # ── Write outputs ─────────────────────────────────────────────────────────
    outputs = {}
    if calibration_path is not None:
        outputs["calibration_params"] = calibration_path

    results_path = output_dir / f"backtest_results_{today}.csv"
    safe_write_csv(records, results_path, index=False, label="backtest_results", allow_empty=False)
    latest_results_path = output_dir / "backtest_results_latest.csv"
    safe_write_csv(records, latest_results_path, index=False, label="backtest_results_latest", allow_empty=False)

    # Prediction records should NOT go to final context output as it overwrites enrichment output
    # Rename to backtest_context_records.csv if needed for debugging
    context_path = output_dir / "backtest_context_records.csv"
    safe_write_csv(records, context_path, index=False, label="backtest_context_records", allow_empty=False)

    outputs["results"] = results_path
    outputs["results_latest"] = latest_results_path
    log.info(f"Results -> {results_path}  ({len(records):,} rows)")

    report_path = output_dir / f"backtest_model_report_{today}.csv"
    safe_write_csv(report_df, report_path, index=False, label="backtest_model_report", allow_empty=True)
    safe_write_csv(report_df, output_dir / "backtest_model_report_latest.csv", index=False, label="backtest_model_report_latest", allow_empty=True)
    outputs["report"] = report_path
    log.info(f"Report  -> {report_path}")

    if stacking_params:
        outputs["stacking"] = output_dir / STACKING_COEFFICIENTS_JSON.name
        log.info("Stacking coefficients -> %s", outputs["stacking"])

    if not calib_df.empty:
        calib_path = output_dir / f"backtest_calibration_{today}.csv"
        safe_write_csv(calib_df, calib_path, index=False, label="backtest_calibration", allow_empty=True)
        safe_write_csv(calib_df, output_dir / "backtest_calibration_latest.csv", index=False, label="backtest_calibration_latest", allow_empty=True)
        outputs["calibration"] = calib_path
        log.info(f"Calibration -> {calib_path}")

    if optimized_weights:
        weights_payload = {
            **optimized_weights,
            "pipeline_run_id": PIPELINE_RUN_ID,
            "seed": SEED,
            "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
        }

        weights_path = output_dir / "backtest_optimized_weights.json"
        model_weights_path = output_dir / "model_weights.json"
        history_path = output_dir / "model_weights_history.json"

        for candidate in (weights_path, model_weights_path):
            if candidate.exists() and candidate.stat().st_size > 0:
                try:
                    _append_weight_history_snapshot(history_path, json.loads(candidate.read_text()))
                except Exception:
                    log.warning("Could not snapshot existing weights at %s", candidate)
            candidate.write_text(json.dumps(weights_payload, indent=2))

        outputs["weights"] = weights_path
        log.info(f"Weights → {weights_path}")

        # Append to weight history CSV
        weight_history_path = output_dir / "model_weight_history.csv"
        history_row = {
            "run_date": today,
            "optimizer_metric": config.optimizer_metric,
            "optimized_value": round(opt_metric, 4),
            "n_games_used": len(records),
            **{f"{name}_weight": round(w, 4) for name, w in opt_weights.items()},
        }
        history_df = pd.DataFrame([history_row])
        if weight_history_path.exists() and weight_history_path.stat().st_size > 0:
            existing = pd.read_csv(weight_history_path)
            history_df = pd.concat([existing, history_df], ignore_index=True)
        safe_write_csv(history_df, weight_history_path, index=False, label="model_weight_history", allow_empty=True)
        log.info(f"Weight history appended -> {weight_history_path}")

    backtest_path = build_team_backtest_csv(output_dir)
    if backtest_path is not None:
        outputs["team_season_summary"] = backtest_path

    return outputs


def verify_backtest_output(path: Path) -> bool:
    """Read back a freshly written backtest CSV and fail loud on bad writes."""
    if not path.exists():
        log.error("[VERIFY FAIL] %s does not exist after write", path)
        return False

    if path.stat().st_size < 500:
        log.error("[VERIFY FAIL] %s is suspiciously small: %s bytes", path, path.stat().st_size)
        return False

    df = pd.read_csv(path)
    if df.empty:
        log.error("[VERIFY FAIL] %s has 0 rows", path)
        return False

    critical_cols = ["game_id", "actual_margin", "ens_spread", "home_team_id", "away_team_id"]
    missing = [col for col in critical_cols if col not in df.columns]
    if missing:
        log.error("[VERIFY FAIL] %s missing critical columns: %s", path, missing)
        return False

    null_failures = [col for col in critical_cols if df[col].isna().mean() == 1.0]
    if null_failures:
        log.error("[VERIFY FAIL] %s has 100%% null critical columns: %s", path, null_failures)
        return False

    log.info("[VERIFY PASS] %s: %s rows, critical columns populated", path, len(df))
    return True


def _write_empty_accuracy_outputs(report_path: Path, dimension_path: Path) -> None:
    report_cols = [
        "game_id", "game_datetime_utc", "home_team", "away_team", "pred_spread", "actual_margin",
        "spread_error", "abs_error", "covered", "correct_side", "model_confidence", "game_tier",
        "home_conference", "away_conference", "total_line", "pred_total", "actual_total",
    ]
    dim_cols = [
        "dimension", "group", "n_games", "ats_wins", "ats_losses", "ats_pushes", "ats_win_rate",
        "mean_abs_error", "mean_signed_error", "correct_side_rate", "mean_confidence", "sufficient_sample",
    ]
    safe_write_csv(pd.DataFrame(columns=report_cols), report_path, index=False, label="model_accuracy_report", allow_empty=True)
    safe_write_csv(pd.DataFrame(columns=dim_cols), dimension_path, index=False, label="model_accuracy_by_dimension", allow_empty=True)


def _append_dq_audit_rows(data_dir: Path, rows: List[dict]) -> None:
    """Append deterministic quality-gate failures to data/dq_audit.csv."""
    if not rows:
        return

    audit_path = data_dir / "dq_audit.csv"
    incoming = pd.DataFrame(rows)
    incoming["created_at"] = pd.Timestamp.utcnow().isoformat()
    incoming["pipeline_run_id"] = PIPELINE_RUN_ID

    if audit_path.exists() and audit_path.stat().st_size > 0:
        existing = pd.read_csv(audit_path)
        out = pd.concat([existing, incoming], ignore_index=True)
    else:
        out = incoming

    dedupe_cols = ["entity_type", "entity_id", "severity", "reason_codes", "details", "pipeline_run_id"]
    dedupe_cols = [col for col in dedupe_cols if col in out.columns]
    if dedupe_cols:
        out = out.drop_duplicates(subset=dedupe_cols, keep="last")

    safe_write_csv(out, audit_path, index=False, label="dq_audit", allow_empty=True)


def _normalize_conference_tier(home_conference: object, away_conference: object) -> str:
    confs = []
    for conf in (home_conference, away_conference):
        if pd.isna(conf):
            continue
        conf_txt = str(conf).strip().lower()
        if not conf_txt or conf_txt in {"nan", "none"}:
            continue
        confs.append(conf_txt)

    if not confs:
        return "UNKNOWN"
    if any(c in HIGH_TIER_CONFERENCES for c in confs):
        return "HIGH"
    if any(c in MID_TIER_CONFERENCES for c in confs):
        return "MID"
    return "LOW"


def _build_dimension_rows(report_df: pd.DataFrame) -> pd.DataFrame:
    dims: List[pd.DataFrame] = []

    def _agg_dimension(df: pd.DataFrame, dim_name: str, group_col: str) -> None:
        grouped = df.groupby(group_col, dropna=False)
        out = grouped.agg(
            n_games=("game_id", "size"),
            ats_wins=("ats_result", lambda s: int((s == "W").sum())),
            ats_losses=("ats_result", lambda s: int((s == "L").sum())),
            ats_pushes=("ats_result", lambda s: int((s == "P").sum())),
            mean_abs_error=("abs_error", "mean"),
            mean_signed_error=("spread_error", "mean"),
            correct_side_rate=("correct_side", "mean"),
            mean_confidence=("model_confidence", "mean"),
        ).reset_index()

        out = out[out["n_games"] >= 5].copy()
        if out.empty:
            return

        denom = out["ats_wins"] + out["ats_losses"]
        out["ats_win_rate"] = np.where(denom > 0, out["ats_wins"] / denom, np.nan)
        out["dimension"] = dim_name
        out["group"] = out[group_col].astype(str)
        out["sufficient_sample"] = out["n_games"] >= 20
        dims.append(out[[
            "dimension", "group", "n_games", "ats_wins", "ats_losses", "ats_pushes", "ats_win_rate",
            "mean_abs_error", "mean_signed_error", "correct_side_rate", "mean_confidence", "sufficient_sample",
        ]])

    _agg_dimension(report_df, "conference_tier", "conference_tier")
    _agg_dimension(report_df, "spread_bucket", "spread_bucket")
    _agg_dimension(report_df, "favorite_side", "favorite_side")
    _agg_dimension(report_df, "day_of_week", "day_of_week")
    _agg_dimension(report_df, "month", "month")

    if not dims:
        return pd.DataFrame(columns=[
            "dimension", "group", "n_games", "ats_wins", "ats_losses", "ats_pushes", "ats_win_rate",
            "mean_abs_error", "mean_signed_error", "correct_side_rate", "mean_confidence", "sufficient_sample",
        ])

    return pd.concat(dims, ignore_index=True).sort_values(["dimension", "group"]).reset_index(drop=True)


def grade_historical_predictions(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    prediction_files = sorted(data_dir.glob("predictions_*.csv"))
    report_path = data_dir / "model_accuracy_report.csv"
    dimension_path = data_dir / "model_accuracy_by_dimension.csv"

    if not prediction_files:
        log.warning("No predictions_*.csv files found in %s", data_dir)
        _write_empty_accuracy_outputs(report_path, dimension_path)
        return pd.DataFrame(), pd.DataFrame()

    pred_frames = []
    for p in prediction_files:
        try:
            df = pd.read_csv(p, low_memory=False)
            df["prediction_file"] = p.name
            pred_frames.append(df)
        except Exception as exc:
            log.warning("Skipping unreadable prediction file %s: %s", p, exc)

    if not pred_frames:
        _write_empty_accuracy_outputs(report_path, dimension_path)
        return pd.DataFrame(), pd.DataFrame()

    predictions = pd.concat(pred_frames, ignore_index=True)

    if "game_id" not in predictions.columns or predictions["game_id"].isna().all():
        for fallback_id_col in ("event_id", "espn_event_id"):
            if fallback_id_col in predictions.columns:
                predictions["game_id"] = predictions[fallback_id_col]
                log.info("Resolved missing game_id from %s in prediction files.", fallback_id_col)
                break

    gate_total = len(predictions)
    total_predictions = len(predictions)

    weighted_path = _resolve_data_path(WEIGHTED_CSV)
    if not weighted_path.exists():
        log.warning("Missing weighted game file at %s", weighted_path)
        _write_empty_accuracy_outputs(report_path, dimension_path)
        return pd.DataFrame(), pd.DataFrame()

    wg = pd.read_csv(weighted_path, low_memory=False)
    if "event_id" not in wg.columns:
        log.error("Missing event_id in weighted games file")
        return pd.DataFrame(), pd.DataFrame()
    wg["event_id"] = wg["event_id"].astype(str)
    wg["game_datetime_utc"] = pd.to_datetime(wg["game_datetime_utc"], errors="coerce", utc=True)

    if "home_away" in wg.columns:
        home_rows = wg[wg["home_away"].astype(str).str.lower().eq("home")].copy()
    else:
        home_rows = wg.copy()

    if home_rows.empty:
        home_rows = wg.copy()

    games = home_rows.sort_values("game_datetime_utc").drop_duplicates(subset=["event_id"], keep="last").copy()
    games["home_score"] = pd.to_numeric(games.get("points_for"), errors="coerce")
    games["away_score"] = pd.to_numeric(games.get("points_against"), errors="coerce")
    games = games.rename(columns={"event_id": "game_id"})
    for optional_col in ("home_conference", "away_conference"):
        if optional_col not in games.columns:
            games[optional_col] = pd.NA

    predictions["game_id"] = predictions.get("game_id", pd.Series(index=predictions.index, dtype=object)).astype(str)
    predictions["game_datetime_utc"] = pd.to_datetime(predictions.get("game_datetime_utc"), errors="coerce", utc=True)
    predictions["pred_spread"] = pd.to_numeric(predictions.get("pred_spread"), errors="coerce")
    predictions["pred_total"] = pd.to_numeric(predictions.get("pred_total"), errors="coerce")
    predictions["model_confidence"] = pd.to_numeric(predictions.get("model_confidence"), errors="coerce")
    predictions["total_line"] = pd.to_numeric(predictions.get("total_line"), errors="coerce")

    merged = predictions.merge(
        games[["game_id", "game_datetime_utc", "home_team", "away_team", "home_conference", "away_conference", "home_score", "away_score"]],
        on="game_id",
        how="left",
        suffixes=("_pred", ""),
    )

    now_utc = pd.Timestamp.now(tz="UTC")
    merged["game_datetime_utc_final"] = merged["game_datetime_utc"].combine_first(merged["game_datetime_utc_pred"])
    merged["actual_margin"] = merged["home_score"] - merged["away_score"]
    merged["actual_total"] = merged["home_score"] + merged["away_score"]

    gate_counts = {
        "total_predictions": gate_total,
        "missing_game_id": int(merged["game_id"].isna().sum()),
        "missing_pred_spread": int(merged["pred_spread"].isna().sum()),
        "missing_game_datetime": int(merged["game_datetime_utc_final"].isna().sum()),
        "future_games": int((merged["game_datetime_utc_final"] >= now_utc).fillna(False).sum()),
        "missing_home_score": int(merged["home_score"].isna().sum()),
        "missing_away_score": int(merged["away_score"].isna().sum()),
    }
    log.info("Accuracy grading integrity gates: %s", gate_counts)

    gradeable_mask = (
        merged["pred_spread"].notna()
        & merged["game_datetime_utc_final"].notna()
        & (merged["game_datetime_utc_final"] < now_utc)
        & merged["home_score"].notna()
        & merged["away_score"].notna()
    )
    gradeable = merged[gradeable_mask].copy()

    rejected = merged[~gradeable_mask].copy()
    if not rejected.empty:
        rejected["reason_codes"] = ""
        rejected.loc[rejected["pred_spread"].isna(), "reason_codes"] += "missing_pred_spread;"
        rejected.loc[rejected["game_datetime_utc_final"].isna(), "reason_codes"] += "missing_game_datetime;"
        rejected.loc[
            (rejected["game_datetime_utc_final"] >= now_utc).fillna(False),
            "reason_codes",
        ] += "game_not_final;"
        rejected.loc[rejected["home_score"].isna() | rejected["away_score"].isna(), "reason_codes"] += "missing_final_score;"
        rejected["reason_codes"] = rejected["reason_codes"].str.strip(";")
        rejected["reason_codes"] = rejected["reason_codes"].replace("", "unknown")
        rejected_summary = rejected.groupby("reason_codes", dropna=False).size().to_dict()
        log.warning("Rejected predictions during grading: %s", rejected_summary)
        _append_dq_audit_rows(
            data_dir,
            [
                {
                    "entity_type": "model_accuracy_report",
                    "entity_id": str(k),
                    "severity": "warning",
                    "reason_codes": str(k),
                    "details": json.dumps({"count": int(v)}),
                }
                for k, v in sorted(rejected_summary.items(), key=lambda item: item[0])
            ],
        )

    if gradeable.empty:
        log.warning("Found 0 gradeable games. Writing empty accuracy outputs.")
        _write_empty_accuracy_outputs(report_path, dimension_path)
        log.info("Total predictions found: %s", total_predictions)
        log.info("Total gradeable games: %s", len(gradeable))
        return pd.DataFrame(), pd.DataFrame()

    if len(gradeable) < 10:
        log.warning(
            "Found only %s gradeable games (<10). Writing partial accuracy outputs with limited sample.",
            len(gradeable),
        )

    mae_neg_home_favored = (-gradeable["pred_spread"] - gradeable["actual_margin"]).abs().mean()
    mae_pos_home_favored = (gradeable["pred_spread"] - gradeable["actual_margin"]).abs().mean()
    use_negative_home_favored = mae_neg_home_favored <= mae_pos_home_favored
    if not use_negative_home_favored:
        gradeable["pred_spread"] = -gradeable["pred_spread"]
    log.info(
        "Detected spread sign convention: %s (mae_neg=%.3f, mae_pos=%.3f)",
        "negative=home_favored" if use_negative_home_favored else "positive=home_favored",
        mae_neg_home_favored,
        mae_pos_home_favored,
    )

    gradeable["spread_error"] = gradeable["pred_spread"] - (-gradeable["actual_margin"])
    gradeable["abs_error"] = gradeable["spread_error"].abs()
    margin_plus_spread = gradeable["actual_margin"] + gradeable["pred_spread"]
    gradeable["covered"] = margin_plus_spread > 0

    pred_sign = np.sign(gradeable["pred_spread"])
    actual_sign = np.sign(-gradeable["actual_margin"])
    gradeable["correct_side"] = (pred_sign == actual_sign) & (pred_sign != 0) & (actual_sign != 0)
    gradeable["ats_result"] = np.where(
        margin_plus_spread > 0,
        "W",
        np.where(margin_plus_spread < 0, "L", "P"),
    )

    gradeable["game_tier"] = gradeable.get("game_tier")
    if gradeable["game_tier"] is None:
        gradeable["game_tier"] = gradeable.get("game_type")
    if gradeable["game_tier"] is None:
        gradeable["game_tier"] = "UNKNOWN"
    if hasattr(gradeable["game_tier"], "fillna"):
        gradeable["game_tier"] = gradeable["game_tier"].fillna("UNKNOWN")
    gradeable["conference_tier"] = gradeable.apply(
        lambda r: _normalize_conference_tier(r.get("home_conference"), r.get("away_conference")), axis=1
    )
    abs_spread = gradeable["pred_spread"].abs()
    gradeable["spread_bucket"] = np.select(
        [abs_spread < 3, abs_spread < 6, abs_spread < 10, abs_spread >= 10],
        ["0-3", "3-6", "6-10", "10+"],
        default="UNKNOWN",
    )
    gradeable["favorite_side"] = np.where(gradeable["pred_spread"] < 0, "home_fav", "away_fav")
    dt = pd.to_datetime(gradeable["game_datetime_utc_final"], utc=True)
    gradeable["day_of_week"] = dt.dt.strftime("%a")
    gradeable["month"] = dt.dt.strftime("%b")

    report_df = gradeable[[
        "game_id", "game_datetime_utc_final", "home_team", "away_team", "pred_spread", "actual_margin",
        "spread_error", "abs_error", "covered", "correct_side", "model_confidence", "game_tier",
        "home_conference", "away_conference", "total_line", "pred_total", "actual_total",
        "conference_tier", "spread_bucket", "favorite_side", "day_of_week", "month", "ats_result",
    ]].rename(columns={"game_datetime_utc_final": "game_datetime_utc"})

    safe_write_csv(
        report_df[[
            "game_id", "game_datetime_utc", "home_team", "away_team", "pred_spread", "actual_margin", "spread_error",
            "abs_error", "covered", "correct_side", "model_confidence", "game_tier", "home_conference",
            "away_conference", "total_line", "pred_total", "actual_total",
        ]],
        report_path,
        index=False,
        label="model_accuracy_report",
        allow_empty=True,
    )

    dim_df = _build_dimension_rows(report_df)
    safe_write_csv(dim_df, dimension_path, index=False, label="model_accuracy_by_dimension", allow_empty=True)

    ats_wins = int((report_df["ats_result"] == "W").sum())
    ats_losses = int((report_df["ats_result"] == "L").sum())
    ats_pushes = int((report_df["ats_result"] == "P").sum())
    overall_mae = float(report_df["abs_error"].mean())

    best_group = "N/A"
    worst_group = "N/A"
    eligible = dim_df[dim_df["n_games"] >= 10]
    if not eligible.empty:
        best_row = eligible.loc[eligible["mean_abs_error"].idxmin()]
        worst_row = eligible.loc[eligible["mean_abs_error"].idxmax()]
        best_group = f"{best_row['dimension']}:{best_row['group']} (MAE={best_row['mean_abs_error']:.3f}, n={int(best_row['n_games'])})"
        worst_group = f"{worst_row['dimension']}:{worst_row['group']} (MAE={worst_row['mean_abs_error']:.3f}, n={int(worst_row['n_games'])})"

    log.info("Total predictions found: %s", total_predictions)
    log.info("Total gradeable games: %s", len(report_df))
    log.info("Overall ATS record (W-L-P): %s-%s-%s", ats_wins, ats_losses, ats_pushes)
    log.info("Overall mean absolute error: %.4f", overall_mae)
    log.info("Best performing dimension group: %s", best_group)
    log.info("Worst performing dimension group: %s", worst_group)

    return report_df, dim_df


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

    outputs = run(config, args.output_dir)
    if not outputs:
        sys.exit(1)

    verify_path = args.output_dir / "backtest_results_latest.csv"
    if "results_latest" not in outputs:
        log.warning(
            "Skipping backtest results verification and grading because no predictions were produced. "
            "Try lowering --min-games or widening the evaluation window."
        )
        return

    if not verify_backtest_output(verify_path):
        sys.exit(1)

    grade_historical_predictions(DATA_DIR)


if __name__ == "__main__":
    main()
