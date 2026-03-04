#!/usr/bin/env python3
"""
cbb_results_tracker.py — Daily Prediction vs Actual Outcome Tracker

Runs after games complete to record how our predictions performed.
The forward-looking complement to cbb_backtester.py — where the backtester
replays history, this tracks the current season in real time.

ARCHITECTURE
─────────────────────────────────────────────────────────────────────────────
Each day after games finish (trigger: ~8AM UTC, 12AM PST):

1. Load predictions_combined_latest.csv (what we predicted)
2. Load games.csv (what actually happened)
3. Match predictions to final scores by game_id
4. Compute per-prediction outcomes: ATS, O/U, margin error, model agreement
5. Append to running performance log (data/results_log.csv)
6. Compute rolling accuracy windows: L7, L30, season
7. Detect alerts: model drift, edge flag calibration, agreement accuracy
8. Write summary report to data/results_summary.csv

ALERT TRIGGERS
─────────────────────────────────────────────────────────────────────────────
  MODEL_DRIFT      : Rolling L14 ATS% drops >5pts from season average
  OVERCONFIDENT    : High-confidence picks (>75%) going <45% ATS over L14
  UNDERCONFIDENT   : Low-confidence picks (<60%) going >58% ATS (model is
                     too conservative — raise confidence thresholds)
  EDGE_FLAG_COLD   : Edge-flagged games going <48% ATS over L14
  ENSEMBLE_SPLIT   : SPLIT-agreement games going at different rate than STRONG
  BIAS_DETECTED    : Running systematic over/under prediction of home team
  TOTAL_DRIFT      : O/U accuracy dropping below 45% over L20

Outputs:
    data/results_log.csv           — Permanent append-only per-game record
    data/results_summary.csv       — Rolling window accuracy report
    data/results_alerts.csv        — Active alerts
    data/results_model_split.csv   — Per-model L7/L30/season ATS breakdown

Usage:
    python cbb_results_tracker.py                     # process yesterday's games
    python cbb_results_tracker.py --date 20250315     # specific date
    python cbb_results_tracker.py --reprocess-days 30 # reprocess last N days
    python cbb_results_tracker.py --summary           # print summary only, no write
"""
# home/away splits

import argparse
import json
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pipeline_csv_utils import normalize_game_id, normalize_numeric_dtypes, safe_write_csv
from config.logging_config import get_logger

warnings.filterwarnings("ignore")

log = get_logger(__name__)

TZ = ZoneInfo("America/Los_Angeles")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR         = Path("data")
DATA_CSV_DIR     = DATA_DIR / "csv"
PREDICTIONS_CSV  = DATA_DIR / "predictions_combined_latest.csv"
PREDICTIONS_GRADED_CSV = DATA_DIR / "predictions_graded.csv"
PREDICTIONS_CONTEXT_CSV = DATA_DIR / "predictions_with_context.csv"
ENSEMBLE_CSV     = DATA_DIR / "ensemble_predictions_latest.csv"
PRIMARY_CSV      = DATA_DIR / "predictions_latest.csv"
PREDICTIONS_HISTORY_CSV = DATA_DIR / "predictions_history.csv"
GAMES_CSV        = DATA_DIR / "games.csv"
RESULTS_LOG      = DATA_DIR / "results_log.csv"
RESULTS_LOG_BY_TEAM = DATA_DIR / "results_log_by_team.csv"
RESULTS_SUMMARY  = DATA_DIR / "results_summary.csv"
RESULTS_ALERTS   = DATA_DIR / "results_alerts.csv"
MODEL_SPLIT_CSV  = DATA_DIR / "results_model_split.csv"
DYNAMIC_WEIGHTS_JSON = DATA_DIR / "dynamic_model_weights.json"
BACKTEST_WEIGHTS_JSON = DATA_DIR / "backtest_optimized_weights.json"


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
VIG_BREAK_EVEN  = 52.38
MODEL_NAMES     = [
    "fourfactors", "adjefficiency", "pythagorean",
    "momentum", "situational", "cagerankings", "regressedeff",
]
MODEL_ID_MAP = {
    "m1": "fourfactors",
    "m2": "adjefficiency",
    "m3": "pythagorean",
    "m4": "momentum",
    "m5": "situational",
    "m6": "cagerankings",
    "m7": "regressedeff",
}

# Alert thresholds
DRIFT_THRESHOLD       = 5.0    # % pts drop from season avg to trigger alert
OVERCONF_THRESHOLD    = 45.0   # % ATS below which high-confidence picks are "cold"
UNDERCONF_THRESHOLD   = 58.0   # % ATS above which low-confidence picks are "hot"
EDGE_COLD_THRESHOLD   = 48.0   # % ATS edge flag alert
BIAS_THRESHOLD        = 3.0    # Points of systematic directional bias
TOTAL_DRIFT_THRESHOLD = 45.0   # O/U% below which to alert
MIN_SAMPLE_FOR_ALERT  = 12     # Minimum games in window before alerting


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GameOutcome:
    """
    Single game result record — everything we predicted plus what happened.
    Stored as one row in results_log.csv.
    """
    # Identity
    game_id:          str
    game_date:        str          # YYYYMMDD
    game_datetime_utc:str
    home_team:        str
    away_team:        str
    home_team_id:     str
    away_team_id:     str
    neutral_site:     int

    # Actuals
    home_score_actual: float
    away_score_actual: float
    actual_margin:     float       # home − away
    actual_total:      float
    home_won:          int

    # Primary model prediction
    pred_spread:       Optional[float] = None    # Negative = home favored
    pred_spread_source: str = ""
    pred_total:        Optional[float] = None
    pred_home_score:   Optional[float] = None
    pred_away_score:   Optional[float] = None
    primary_confidence:Optional[float] = None

    # Ensemble prediction
    ens_spread:        Optional[float] = None
    ens_total:         Optional[float] = None
    ens_confidence:    Optional[float] = None
    ens_model_agreement: Optional[str]  = None
    ens_spread_std:    Optional[float] = None

    # Market lines
    market_spread:     Optional[float] = None
    market_total:      Optional[float] = None
    home_ml:           Optional[float] = None
    away_ml:           Optional[float] = None

    # Primary model outcomes
    primary_ats_correct:  Optional[int]   = None    # 1=correct, 0=wrong, None=no line
    primary_ou_correct:   Optional[int]   = None
    primary_margin_error: Optional[float] = None    # predicted_margin - actual_margin
    primary_wins_game:    Optional[int]   = None    # Picked correct winner?
    primary_edge_flagged: Optional[int]   = None
    primary_edge_ats:     Optional[int]   = None

    # Ensemble outcomes
    ens_ats_correct:   Optional[int]   = None
    ens_ou_correct:    Optional[int]   = None
    ens_margin_error:  Optional[float] = None
    ens_wins_game:     Optional[int]   = None

    # Per-model spreads (for individual model tracking)
    fourfactors_spread:   Optional[float] = None
    adjefficiency_spread: Optional[float] = None
    pythagorean_spread:   Optional[float] = None
    momentum_spread:      Optional[float] = None
    situational_spread:   Optional[float] = None
    cagerankings_spread:  Optional[float] = None
    regressedeff_spread:  Optional[float] = None
    m1_spread:            Optional[float] = None
    m2_spread:            Optional[float] = None
    m3_spread:            Optional[float] = None
    m4_spread:            Optional[float] = None
    m5_spread:            Optional[float] = None
    m6_spread:            Optional[float] = None
    m7_spread:            Optional[float] = None
    spread_line:          Optional[float] = None

    # Per-model ATS outcomes
    fourfactors_ats:   Optional[int] = None
    adjefficiency_ats: Optional[int] = None
    pythagorean_ats:   Optional[int] = None
    momentum_ats:      Optional[int] = None
    situational_ats:   Optional[int] = None
    cagerankings_ats:  Optional[int] = None
    regressedeff_ats:  Optional[int] = None

    # Metadata
    cage_em_diff:      Optional[float] = None
    barthag_diff:      Optional[float] = None
    clv_vs_consensus:  Optional[float] = None
    clv_vs_open:       Optional[float] = None
    clv_vs_pinnacle:   Optional[float] = None
    clv_vs_dk:         Optional[float] = None
    beat_consensus:    Optional[bool] = None
    beat_pinnacle:     Optional[bool] = None
    clv_direction:     Optional[str] = None
    home_spread_current: Optional[float] = None
    home_spread_open:  Optional[float] = None
    processed_at:      str = ""

    def to_dict(self) -> Dict:
        row = asdict(self)
        # event_id is the primary key used by pipeline_csv_utils deduplication and
        # results_log schema — must be present and non-null.
        row["event_id"] = row["game_id"]
        # Backward-compatible alias required by results_log schema validators.
        if "predicted_spread" not in row:
            row["predicted_spread"] = row.get("pred_spread")
        # Aliases for downstream compatibility
        row["ats_correct"] = row.get("primary_ats_correct")
        row["cover"] = row.get("primary_ats_correct")
        row["ou_correct"] = row.get("primary_ou_correct")
        row["spread_line"] = row.get("market_spread")
        row["actual_spread"] = row.get("actual_margin")
        return row


def _expand_to_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Expand game-level rows to team-level rows for downstream compat."""
    rows: List[Dict] = []
    for _, r in df.iterrows():
        base = r.to_dict()
        ats_val = _safe_int(r.get("ats_correct"), default=None)
        away_ats = None if ats_val is None else (1 - ats_val)
        home = {
            **base,
            "team_id": r.get("home_team_id"),
            "opponent_id": r.get("away_team_id"),
            "home_away": "home",
            "team_ats_correct": r.get("ats_correct"),
            "team_ou_correct": r.get("ou_correct"),
        }
        away = {
            **base,
            "team_id": r.get("away_team_id"),
            "opponent_id": r.get("home_team_id"),
            "home_away": "away",
            "team_ats_correct": away_ats,
            "team_ou_correct": r.get("ou_correct"),
        }
        rows.extend([home, away])
    return pd.DataFrame(rows)


@dataclass
class Alert:
    """A performance alert triggered by the results tracker."""
    alert_type:   str     # MODEL_DRIFT, OVERCONFIDENT, EDGE_FLAG_COLD, etc.
    severity:     str     # WARNING, CRITICAL
    message:      str
    metric_value: float
    threshold:    float
    window:       str     # "L7", "L14", "L30", "SEASON"
    n_games:      int
    detected_at:  str = ""

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RollingStats:
    """Accuracy stats over a rolling window."""
    window:       str
    n_games:      int = 0
    ats_correct:  int = 0
    ats_pct:      float = 0.0
    ou_correct:   int = 0
    ou_pct:       float = 0.0
    win_correct:  int = 0
    win_pct:      float = 0.0
    margin_mae:   float = 0.0
    margin_bias:  float = 0.0    # Positive = we over-predict home
    edge_n:       int = 0
    edge_ats_pct: float = 0.0
    high_conf_ats:float = 0.0    # ATS% when confidence > 0.70
    low_conf_ats: float = 0.0    # ATS% when confidence < 0.60


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _safe_float(val, default=None) -> Optional[float]:
    try:
        v = float(val)
        return v if not np.isnan(v) else default
    except (TypeError, ValueError):
        return default


def _safe_int(val, default=None) -> Optional[int]:
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return default


def load_predictions(date_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Load the most recent prediction file. Prefers combined (primary + ensemble),
    falls back to ensemble-only, then primary-only.

    date_filter: YYYYMMDD — if provided, try to load predictions_{date}.csv first.
    """
    candidates = []
    if date_filter:
        candidates += [
            DATA_DIR / f"predictions_combined_{date_filter}.csv",
            DATA_DIR / f"ensemble_predictions_{date_filter}.csv",
            DATA_DIR / f"predictions_{date_filter}.csv",
        ]
    candidates += [
        PREDICTIONS_CSV,
        ENSEMBLE_CSV,
        PRIMARY_CSV,
    ]

    for path in candidates:
        if path.exists() and path.stat().st_size > 50:
            log.info(f"Loading predictions: {path.name}")
            df = pd.read_csv(path, dtype=str, low_memory=False)
            df = normalize_numeric_dtypes(df)
            for col in df.columns:
                if col not in {"game_id","home_team","away_team","home_team_id",
                               "away_team_id","game_datetime_utc","neutral_site",
                               "ens_model_agreement","ens_offensive_archetype",
                               "predicted_at","game_type","_source_date"}:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            spread_candidates = [
                "pred_spread", "predicted_spread", "ensemble_spread", "ens_spread"
            ]
            existing_spread_cols = [c for c in spread_candidates if c in df.columns]
            if existing_spread_cols:
                any_spread_data = df[existing_spread_cols].notna().any(axis=None)
                if not any_spread_data:
                    log.warning(
                        "No spread predictions in %s: all spread columns are null; "
                        "loading anyway for game result tracking",
                        path.name,
                    )

            log.info(f"  {len(df)} predictions loaded")
            return df

    log.warning("No prediction files found")
    return pd.DataFrame()


def load_games_results(game_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load completed game results from games.csv.
    Returns only completed games with final scores.
    """
    games_path = _resolve_data_path(GAMES_CSV)
    if not games_path.exists():
        log.error("games.csv not found — cannot match results")
        return pd.DataFrame()

    df = pd.read_csv(games_path, dtype=str, low_memory=False)
    df = normalize_numeric_dtypes(df)
    df["game_datetime_utc"] = pd.to_datetime(
        df.get("game_datetime_utc", pd.NaT), utc=True, errors="coerce"
    )

    for col in ["home_score", "away_score", "spread", "over_under", "home_ml", "away_ml"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filter to completed games
    completed_mask = df["completed"].astype(str).str.lower().isin(["true", "1", "yes"])
    df = df[completed_mask].copy()

    # Filter to specific game IDs if provided
    if game_ids:
        df = df[df["game_id"].astype(str).isin([str(g) for g in game_ids])]

    # Compute margin and total from scores
    if "home_score" in df.columns and "away_score" in df.columns:
        df["actual_margin"] = df["home_score"] - df["away_score"]
        df["actual_total"]  = df["home_score"] + df["away_score"]
        df["home_won"]      = (df["actual_margin"] > 0).astype(int)

    log.info(f"Completed games loaded: {len(df):,}")
    return df


def load_results_log() -> pd.DataFrame:
    """Load existing results log, or return empty DataFrame with schema."""
    if RESULTS_LOG.exists() and RESULTS_LOG.stat().st_size > 50:
        df = pd.read_csv(RESULTS_LOG, dtype=str, low_memory=False)
        df = normalize_numeric_dtypes(df)
        for col in df.columns:
            if col not in {"game_id","game_date","game_datetime_utc","home_team",
                           "away_team","home_team_id","away_team_id",
                           "ens_model_agreement","processed_at"}:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        log.info(f"Results log loaded: {len(df):,} historical records")
        return df

    log.info("No existing results log — starting fresh")
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# OUTCOME COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_ats(pred_spread: Optional[float],
                 market_spread: Optional[float],
                 actual_margin: float) -> Optional[int]:
    """
    Return 1 (correct), 0 (wrong), None (push or no line).
    Convention: spread is home-perspective (negative = home favored).
    ATS: we pick home if our spread < market spread (home covers more).
    """
    if pred_spread is None or market_spread is None:
        return None

    pred_margin   = -pred_spread      # Convert spread → expected margin
    mkt_margin    = -market_spread    # Market's expected home margin

    home_cover    = actual_margin > mkt_margin
    away_cover    = actual_margin < mkt_margin
    push          = actual_margin == mkt_margin

    if push:
        return None

    model_home = pred_margin > mkt_margin
    if model_home:
        return 1 if home_cover else 0
    else:
        return 1 if away_cover else 0


def _compute_ou(pred_total: Optional[float],
                market_total: Optional[float],
                actual_total: float) -> Optional[int]:
    """Return 1 (correct), 0 (wrong), None (push or no line)."""
    if pred_total is None or market_total is None:
        return None
    if actual_total == market_total:
        return None
    model_over  = pred_total > market_total
    actual_over = actual_total > market_total
    return 1 if model_over == actual_over else 0


def _compute_margin_error(pred_spread: Optional[float],
                           actual_margin: float) -> Optional[float]:
    """Predicted margin minus actual margin. Positive = over-predicted home."""
    if pred_spread is None:
        return None
    return round(-pred_spread - actual_margin, 2)


def _win_correct(pred_spread: Optional[float], actual_margin: float) -> Optional[int]:
    """Did we pick the correct winner?"""
    if pred_spread is None:
        return None
    pred_home_wins = pred_spread < 0
    actual_home_wins = actual_margin > 0
    return 1 if pred_home_wins == actual_home_wins else 0


def _resolve_pred_spread(pred_row: pd.Series) -> Tuple[Optional[float], str]:
    """Resolve pred_spread from canonical field or known aliases."""
    candidates = [
        "pred_spread",
        "predicted_spread",
        "ensemble_spread",
        "ens_ens_spread",
        "ens_spread",
    ]
    for col in candidates:
        if col in pred_row.index:
            val = _safe_float(pred_row.get(col))
            if val is not None:
                return val, col
    return None, ""


def compute_outcomes(
    pred_row: pd.Series,
    result_row: pd.Series,
) -> GameOutcome:
    """
    Compute all outcomes for one matched prediction + result pair.
    """
    game_id = str(pred_row.get("game_id", ""))
    game_dt = str(result_row.get("game_datetime_utc", ""))

    home_score  = _safe_float(result_row.get("home_score"))
    away_score  = _safe_float(result_row.get("away_score"))
    act_margin  = _safe_float(result_row.get("actual_margin"), 0.0)
    act_total   = _safe_float(result_row.get("actual_total"))
    if act_total is None and home_score and away_score:
        act_total = home_score + away_score

    mkt_spread  = _safe_float(result_row.get("spread"))
    mkt_total   = _safe_float(result_row.get("over_under"))
    home_ml     = _safe_float(result_row.get("home_ml"))
    away_ml     = _safe_float(result_row.get("away_ml"))

    # Primary model fields
    pred_spread, pred_spread_source = _resolve_pred_spread(pred_row)
    pred_total    = _safe_float(pred_row.get("pred_total"))
    prim_conf     = _safe_float(pred_row.get("model_confidence"))
    edge_flag     = _safe_int(pred_row.get("edge_flag"), 0)

    # Ensemble fields — check multiple column naming conventions
    ens_spread    = _safe_float(
        pred_row.get("ens_spread") or pred_row.get("ensemble_spread")
        or pred_row.get("ens_ens_spread") or pred_row.get("ens_ensemble_spread")
    )
    ens_total     = _safe_float(
        pred_row.get("ens_total") or pred_row.get("ensemble_total")
        or pred_row.get("ens_ens_total") or pred_row.get("ens_ensemble_total")
    )
    ens_conf      = _safe_float(
        pred_row.get("ens_confidence") or pred_row.get("ens_ens_confidence")
    )
    ens_agree     = str(pred_row.get("ens_model_agreement", ""))
    ens_std       = _safe_float(pred_row.get("ens_spread_std"))

    # Per-model spreads
    model_spreads = {
        name: _safe_float(pred_row.get(f"ens_{name}_spread") or pred_row.get(f"{name}_spread"))
        for name in MODEL_NAMES
    }

    # ── Per-model ATS ────────────────────────────────────────────────────────
    model_ats = {
        name: _compute_ats(model_spreads[name], mkt_spread, act_margin)
        for name in MODEL_NAMES
    }

    _raw_date = str(pred_row.get("game_date", "")).strip()
    if len(_raw_date) == 8 and _raw_date.isdigit():
        _normalized_game_date = f"{_raw_date[:4]}-{_raw_date[4:6]}-{_raw_date[6:]}"
    else:
        _normalized_game_date = _raw_date

    outcome = GameOutcome(
        game_id           = game_id,
        game_date         = _normalized_game_date,
        game_datetime_utc = game_dt,
        home_team         = str(pred_row.get("home_team", result_row.get("home_team", ""))),
        away_team         = str(pred_row.get("away_team", result_row.get("away_team", ""))),
        home_team_id      = str(pred_row.get("home_team_id", "")),
        away_team_id      = str(pred_row.get("away_team_id", "")),
        neutral_site      = int(_safe_float(pred_row.get("neutral_site"), 0)),

        home_score_actual = float(home_score or 0),
        away_score_actual = float(away_score or 0),
        actual_margin     = float(act_margin),
        actual_total      = float(act_total or 0),
        home_won          = int(_safe_int(result_row.get("home_won"), int(act_margin > 0))),

        pred_spread        = pred_spread,
        pred_total         = pred_total,
        pred_home_score    = _safe_float(pred_row.get("pred_home_score")),
        pred_away_score    = _safe_float(pred_row.get("pred_away_score")),
        primary_confidence = prim_conf,
        pred_spread_source  = pred_spread_source,

        ens_spread         = ens_spread,
        ens_total          = ens_total,
        ens_confidence     = ens_conf,
        ens_model_agreement= ens_agree,
        ens_spread_std     = ens_std,

        market_spread      = mkt_spread,
        market_total       = mkt_total,
        home_ml            = home_ml,
        away_ml            = away_ml,

        # Primary model outcomes
        primary_ats_correct  = _compute_ats(pred_spread, mkt_spread, act_margin),
        primary_ou_correct   = _compute_ou(pred_total, mkt_total, act_total or 0),
        primary_margin_error = _compute_margin_error(pred_spread, act_margin),
        primary_wins_game    = _win_correct(pred_spread, act_margin),
        primary_edge_flagged = edge_flag,
        primary_edge_ats     = _compute_ats(pred_spread, mkt_spread, act_margin)
                               if edge_flag else None,

        # Ensemble outcomes
        ens_ats_correct  = _compute_ats(ens_spread, mkt_spread, act_margin),
        ens_ou_correct   = _compute_ou(ens_total, mkt_total, act_total or 0),
        ens_margin_error = _compute_margin_error(ens_spread, act_margin),
        ens_wins_game    = _win_correct(ens_spread, act_margin),

        # Per-model
        fourfactors_spread   = model_spreads["fourfactors"],
        adjefficiency_spread = model_spreads["adjefficiency"],
        pythagorean_spread   = model_spreads["pythagorean"],
        momentum_spread      = model_spreads["momentum"],
        situational_spread   = model_spreads["situational"],
        cagerankings_spread  = model_spreads["cagerankings"],
        regressedeff_spread  = model_spreads["regressedeff"],
        m1_spread            = model_spreads["fourfactors"],
        m2_spread            = model_spreads["adjefficiency"],
        m3_spread            = model_spreads["pythagorean"],
        m4_spread            = model_spreads["momentum"],
        m5_spread            = model_spreads["situational"],
        m6_spread            = model_spreads["cagerankings"],
        m7_spread            = model_spreads["regressedeff"],
        spread_line          = mkt_spread,

        fourfactors_ats   = model_ats["fourfactors"],
        adjefficiency_ats = model_ats["adjefficiency"],
        pythagorean_ats   = model_ats["pythagorean"],
        momentum_ats      = model_ats["momentum"],
        situational_ats   = model_ats["situational"],
        cagerankings_ats  = model_ats["cagerankings"],
        regressedeff_ats  = model_ats["regressedeff"],

        cage_em_diff  = _safe_float(pred_row.get("cage_em_diff") or pred_row.get("ens_cage_edge")),
        barthag_diff  = _safe_float(pred_row.get("barthag_diff") or pred_row.get("ens_barthag_diff")),
        clv_vs_consensus = _safe_float(pred_row.get("clv_vs_consensus")),
        clv_vs_open      = _safe_float(pred_row.get("clv_vs_open")),
        clv_vs_pinnacle  = _safe_float(pred_row.get("clv_vs_pinnacle")),
        clv_vs_dk        = _safe_float(pred_row.get("clv_vs_dk")),
        beat_consensus   = pred_row.get("beat_consensus"),
        beat_pinnacle    = pred_row.get("beat_pinnacle"),
        clv_direction    = pred_row.get("clv_direction"),
        home_spread_current = _safe_float(pred_row.get("home_spread_current")),
        home_spread_open    = _safe_float(pred_row.get("home_spread_open")),
        processed_at  = datetime.now(TZ).isoformat(),
    )

    return outcome


# ═══════════════════════════════════════════════════════════════════════════════
# ROLLING STATS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _ats_pct(df: pd.DataFrame, col: str) -> float:
    """ATS% from a column of 1/0/None."""
    valid = pd.to_numeric(df[col], errors="coerce").dropna()
    return round(valid.mean() * 100, 1) if len(valid) > 0 else 0.0


def _safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(vals.mean()), 3) if len(vals) > 0 else 0.0


def compute_team_records(games_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministically derive team wins/losses from per-game win flag."""
    games_df = normalize_numeric_dtypes(games_df)
    if games_df is None or games_df.empty or "team_id" not in games_df.columns:
        return pd.DataFrame(columns=["team_id", "wins", "losses", "games_count"])

    df = games_df.copy()
    if "win" in df.columns:
        df["win_flag"] = pd.to_numeric(df["win"], errors="coerce")
    elif {"home_won", "home_away"}.issubset(df.columns):
        home_won = pd.to_numeric(df["home_won"], errors="coerce")
        is_home = df["home_away"].astype(str).str.lower().eq("home")
        df["win_flag"] = np.where(is_home, home_won, 1 - home_won)
    else:
        df["win_flag"] = pd.NA

    key_cols = [c for c in ["team_id", "event_id", "game_id", "game_datetime_utc", "home_away"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="last")

    coverage = float(df["win_flag"].notna().mean()) if len(df) else 0.0
    if coverage <= 0.80:
        log.warning("win_flag coverage %.1f%% below 80%% threshold", coverage * 100)
        return pd.DataFrame(columns=["team_id", "wins", "losses", "games_count"])

    valid = df.dropna(subset=["win_flag"]).copy()
    valid["win_flag"] = valid["win_flag"].astype(int)
    records = (
        valid.groupby("team_id", as_index=False)
        .agg(
            wins=("win_flag", "sum"),
            games_count=("win_flag", "count"),
        )
    )
    records["losses"] = records["games_count"] - records["wins"]
    return records[["team_id", "wins", "losses", "games_count"]]


def compute_rolling_stats(
    log_df: pd.DataFrame,
    window_days: Optional[int] = None,
    window_label: str = "SEASON",
    source: str = "primary",   # "primary" or "ensemble"
) -> RollingStats:
    """
    Compute rolling accuracy stats over the last N days (or full season).
    source: which model's outcomes to use ("primary" or "ensemble").
    """
    log_df = normalize_numeric_dtypes(log_df)
    df = log_df.copy()

    if window_days is not None:
        df["game_datetime_utc"] = pd.to_datetime(
            df["game_datetime_utc"], utc=True, errors="coerce"
        )
        cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=window_days)
        df = df[df["game_datetime_utc"] >= cutoff]

    if df.empty:
        return RollingStats(window=window_label)

    prefix  = "primary" if source == "primary" else "ens"
    ats_col = f"{prefix}_ats_correct"
    ou_col  = f"{prefix}_ou_correct"
    win_col = f"{prefix}_wins_game"
    err_col = f"{prefix}_margin_error"
    conf_col= f"{prefix}_confidence" if source == "ensemble" else "primary_confidence"

    stats = RollingStats(window=window_label)
    stats.n_games = len(df)

    # ATS
    ats_df = df.dropna(subset=[ats_col])
    if len(ats_df) > 0:
        stats.ats_correct = int(pd.to_numeric(ats_df[ats_col], errors="coerce").sum())
        stats.ats_pct     = _ats_pct(ats_df, ats_col)

    # O/U
    ou_df = df.dropna(subset=[ou_col])
    if len(ou_df) > 0:
        stats.ou_correct = int(pd.to_numeric(ou_df[ou_col], errors="coerce").sum())
        stats.ou_pct     = _ats_pct(ou_df, ou_col)

    # Win prediction
    win_df = df.dropna(subset=[win_col])
    if len(win_df) > 0:
        stats.win_correct = int(pd.to_numeric(win_df[win_col], errors="coerce").sum())
        stats.win_pct     = _ats_pct(win_df, win_col)

    # Margin error
    if err_col in df.columns:
        errors = pd.to_numeric(df[err_col], errors="coerce").dropna()
        stats.margin_mae  = round(float(errors.abs().mean()), 2) if len(errors) > 0 else 0.0
        stats.margin_bias = round(float(errors.mean()), 2) if len(errors) > 0 else 0.0

    # Edge flag
    if "primary_edge_flagged" in df.columns and "primary_edge_ats" in df.columns:
        edge_df = df[df["primary_edge_flagged"] == 1].dropna(subset=["primary_edge_ats"])
        stats.edge_n = len(edge_df)
        if stats.edge_n > 0:
            stats.edge_ats_pct = _ats_pct(edge_df, "primary_edge_ats")

    # Confidence-stratified ATS
    if conf_col in df.columns and ats_col in df.columns:
        conf_df = df.dropna(subset=[conf_col, ats_col])
        conf_vals = pd.to_numeric(conf_df[conf_col], errors="coerce")

        high_conf = conf_df[conf_vals >= 0.70]
        low_conf  = conf_df[conf_vals <  0.60]

        if len(high_conf) >= 5:
            stats.high_conf_ats = _ats_pct(high_conf, ats_col)
        if len(low_conf) >= 5:
            stats.low_conf_ats  = _ats_pct(low_conf, ats_col)

    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# ALERT DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_alerts(log_df: pd.DataFrame) -> List[Alert]:
    """
    Run all alert detection rules against the results log.
    Returns list of active Alert objects.
    """
    log_df = normalize_numeric_dtypes(log_df)
    alerts   = []
    now_str  = datetime.now(TZ).isoformat()

    season   = compute_rolling_stats(log_df, window_days=None, window_label="SEASON")
    l14      = compute_rolling_stats(log_df, window_days=14,   window_label="L14")
    l7       = compute_rolling_stats(log_df, window_days=7,    window_label="L7")

    # ── MODEL DRIFT ───────────────────────────────────────────────────────────
    # L14 ATS% dropped significantly below season average
    if season.ats_pct > 0 and l14.n_games >= MIN_SAMPLE_FOR_ALERT:
        drift = season.ats_pct - l14.ats_pct
        if drift > DRIFT_THRESHOLD:
            alerts.append(Alert(
                alert_type   = "MODEL_DRIFT",
                severity     = "CRITICAL" if drift > 10 else "WARNING",
                message      = (f"L14 ATS% ({l14.ats_pct:.1f}%) is {drift:.1f}pts below "
                                f"season avg ({season.ats_pct:.1f}%). Model may need recalibration."),
                metric_value = l14.ats_pct,
                threshold    = season.ats_pct - DRIFT_THRESHOLD,
                window       = "L14",
                n_games      = l14.n_games,
                detected_at  = now_str,
            ))

    # ── OVERCONFIDENT ─────────────────────────────────────────────────────────
    if l14.n_games >= MIN_SAMPLE_FOR_ALERT and l14.high_conf_ats > 0:
        if l14.high_conf_ats < OVERCONF_THRESHOLD:
            alerts.append(Alert(
                alert_type   = "OVERCONFIDENT",
                severity     = "WARNING",
                message      = (f"High-confidence picks (≥70%) going "
                                f"{l14.high_conf_ats:.1f}% ATS over L14. "
                                f"Model confidence scores may be inflated."),
                metric_value = l14.high_conf_ats,
                threshold    = OVERCONF_THRESHOLD,
                window       = "L14",
                n_games      = l14.n_games,
                detected_at  = now_str,
            ))

    # ── UNDERCONFIDENT ────────────────────────────────────────────────────────
    if l14.n_games >= MIN_SAMPLE_FOR_ALERT and l14.low_conf_ats > 0:
        if l14.low_conf_ats > UNDERCONF_THRESHOLD:
            alerts.append(Alert(
                alert_type   = "UNDERCONFIDENT",
                severity     = "WARNING",
                message      = (f"Low-confidence picks (<60%) going "
                                f"{l14.low_conf_ats:.1f}% ATS over L14. "
                                f"Model confidence thresholds may be too conservative."),
                metric_value = l14.low_conf_ats,
                threshold    = UNDERCONF_THRESHOLD,
                window       = "L14",
                n_games      = l14.n_games,
                detected_at  = now_str,
            ))

    # ── EDGE FLAG COLD ────────────────────────────────────────────────────────
    if l14.edge_n >= 5:
        if l14.edge_ats_pct < EDGE_COLD_THRESHOLD:
            alerts.append(Alert(
                alert_type   = "EDGE_FLAG_COLD",
                severity     = "CRITICAL",
                message      = (f"Edge-flagged games going {l14.edge_ats_pct:.1f}% ATS "
                                f"over L14 ({l14.edge_n} games). "
                                f"Edge detection threshold may need adjustment."),
                metric_value = l14.edge_ats_pct,
                threshold    = EDGE_COLD_THRESHOLD,
                window       = "L14",
                n_games      = l14.edge_n,
                detected_at  = now_str,
            ))

    # ── SYSTEMATIC BIAS ───────────────────────────────────────────────────────
    if abs(l14.margin_bias) > BIAS_THRESHOLD and l14.n_games >= MIN_SAMPLE_FOR_ALERT:
        direction = "over-predicting home" if l14.margin_bias > 0 else "over-predicting away"
        alerts.append(Alert(
            alert_type   = "BIAS_DETECTED",
            severity     = "WARNING",
            message      = (f"Systematic {direction}: margin bias = {l14.margin_bias:+.1f} pts "
                            f"over L14. Home court advantage may need recalibration."),
            metric_value = l14.margin_bias,
            threshold    = BIAS_THRESHOLD,
            window       = "L14",
            n_games      = l14.n_games,
            detected_at  = now_str,
        ))

    # ── TOTAL DRIFT ───────────────────────────────────────────────────────────
    if l14.n_games >= MIN_SAMPLE_FOR_ALERT and l14.ou_pct > 0:
        if l14.ou_pct < TOTAL_DRIFT_THRESHOLD:
            alerts.append(Alert(
                alert_type   = "TOTAL_DRIFT",
                severity     = "WARNING",
                message      = (f"O/U accuracy dropped to {l14.ou_pct:.1f}% over L14. "
                                f"Pace or scoring environment may have shifted."),
                metric_value = l14.ou_pct,
                threshold    = TOTAL_DRIFT_THRESHOLD,
                window       = "L14",
                n_games      = l14.n_games,
                detected_at  = now_str,
            ))

    # ── ENSEMBLE MODEL SPLIT ACCURACY ─────────────────────────────────────────
    # Do STRONG-agreement games outperform SPLIT-agreement games?
    if "ens_model_agreement" in log_df.columns and "primary_ats_correct" in log_df.columns:
        strong = log_df[log_df["ens_model_agreement"] == "STRONG"]
        split  = log_df[log_df["ens_model_agreement"] == "SPLIT"]
        if len(strong) >= 10 and len(split) >= 10:
            strong_ats = _ats_pct(strong.dropna(subset=["primary_ats_correct"]),
                                   "primary_ats_correct")
            split_ats  = _ats_pct(split.dropna(subset=["primary_ats_correct"]),
                                   "primary_ats_correct")
            if split_ats > strong_ats + 3.0:
                alerts.append(Alert(
                    alert_type   = "ENSEMBLE_SPLIT_WINS",
                    severity     = "WARNING",
                    message      = (f"SPLIT-agreement games going {split_ats:.1f}% ATS vs "
                                    f"STRONG games at {strong_ats:.1f}%. "
                                    f"Model disagreement may carry predictive signal."),
                    metric_value = split_ats - strong_ats,
                    threshold    = 3.0,
                    window       = "SEASON",
                    n_games      = len(split),
                    detected_at  = now_str,
                ))

    return alerts


# ═══════════════════════════════════════════════════════════════════════════════
# PER-MODEL SPLIT TABLE
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_split_table(log_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-model ATS accuracy across L7 / L30 / SEASON.
    This shows which models are currently hot vs. which are dragging.
    """
    log_df = normalize_numeric_dtypes(log_df)
    rows = []
    windows = [
        ("L7",     7),
        ("L30",    30),
        ("SEASON", None),
    ]

    for model_name in MODEL_NAMES + ["primary", "ensemble"]:
        row = {"model": model_name}

        if model_name == "primary":
            ats_col = "primary_ats_correct"
        elif model_name == "ensemble":
            ats_col = "ens_ats_correct"
        else:
            ats_col = f"{model_name}_ats"

        if ats_col not in log_df.columns:
            continue

        for label, days in windows:
            df_w = log_df.copy()
            if days is not None:
                df_w["game_datetime_utc"] = pd.to_datetime(
                    df_w["game_datetime_utc"], utc=True, errors="coerce"
                )
                cutoff = pd.Timestamp.now(tz="UTC") - timedelta(days=days)
                df_w   = df_w[df_w["game_datetime_utc"] >= cutoff]

            valid = df_w[ats_col].dropna()
            n     = len(valid)
            pct   = round(float(pd.to_numeric(valid, errors="coerce").mean()) * 100, 1) if n > 0 else 0.0

            row[f"{label}_n"]   = n
            row[f"{label}_ats"] = pct
            row[f"{label}_vs_vig"] = round(pct - VIG_BREAK_EVEN, 1)

        rows.append(row)

    return pd.DataFrame(rows)


def _load_backtest_weights() -> Dict[str, float]:
    default = {mid: 1.0 / len(MODEL_ID_MAP) for mid in MODEL_ID_MAP}
    if not BACKTEST_WEIGHTS_JSON.exists() or BACKTEST_WEIGHTS_JSON.stat().st_size <= 2:
        return default
    try:
        payload = json.loads(BACKTEST_WEIGHTS_JSON.read_text())
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        log.warning("Could not parse %s (%s); using equal backtest fallback", BACKTEST_WEIGHTS_JSON, exc)
        return default

    out: Dict[str, float] = {}
    spread_weights = payload.get("weights") if isinstance(payload, dict) else None
    for mid, model_name in MODEL_ID_MAP.items():
        key = f"{mid}_weight"
        val = payload.get(key) if isinstance(payload, dict) else None
        if val is None and isinstance(spread_weights, dict):
            for candidate in (model_name, model_name.title(), model_name.capitalize()):
                if candidate in spread_weights:
                    val = spread_weights[candidate]
                    break
        try:
            out[mid] = float(val) if val is not None else default[mid]
        except (TypeError, ValueError):
            out[mid] = default[mid]

    total = sum(out.values())
    if total <= 0:
        return default
    return {k: v / total for k, v in out.items()}


def write_dynamic_model_weights(log_df: pd.DataFrame) -> None:
    """Compute + persist L14 dynamic model weights blended with backtest priors."""
    if log_df.empty:
        log.warning("Skipping dynamic model weights: empty results log")
        return

    results_log = log_df.copy()
    if "spread_line" not in results_log.columns and "market_spread" in results_log.columns:
        results_log["spread_line"] = results_log["market_spread"]

    required_cols = {"actual_margin", "spread_line"}
    if not required_cols.issubset(results_log.columns):
        log.warning("Skipping dynamic model weights: missing required columns %s", sorted(required_cols - set(results_log.columns)))
        return

    model_ids = list(MODEL_ID_MAP.keys())
    dynamic_weights: Dict[str, float] = {}
    for mid in model_ids:
        spread_col = f"{mid}_spread"
        if spread_col not in results_log.columns:
            continue
        recent = results_log.tail(14).copy()
        recent = recent.dropna(subset=[spread_col, "actual_margin", "spread_line"])
        if recent.empty:
            continue
        recent["covered"] = (
            (pd.to_numeric(recent[spread_col], errors="coerce") > 0)
            == (pd.to_numeric(recent["actual_margin"], errors="coerce") > pd.to_numeric(recent["spread_line"], errors="coerce"))
        )
        ats_pct = recent["covered"].mean()
        dynamic_weights[mid] = max(0.02, float(ats_pct))

    if not dynamic_weights:
        log.warning("Skipping dynamic model weights: no model spread columns with usable L14 rows")
        return

    total_dyn = sum(dynamic_weights.values())
    dynamic_weights = {k: round(v / total_dyn, 4) for k, v in dynamic_weights.items()}

    backtest_weights = _load_backtest_weights()
    final_weights: Dict[str, float] = {}
    for mid in model_ids:
        bt = backtest_weights.get(mid, 1 / 7)
        dyn = dynamic_weights.get(mid, 1 / 7)
        final_weights[mid] = round(0.70 * bt + 0.30 * dyn, 4)

    total_blend = sum(final_weights.values())
    if total_blend > 0:
        final_weights = {k: round(v / total_blend, 4) for k, v in final_weights.items()}

    payload = {
        "computed_at": datetime.now(tz=ZoneInfo("UTC")).isoformat(),
        "l14_window": 14,
        "dynamic_weights": dynamic_weights,
        "backtest_weights": {k: round(v, 4) for k, v in backtest_weights.items()},
        "blended_weights": final_weights,
    }
    DYNAMIC_WEIGHTS_JSON.write_text(json.dumps(payload, indent=2))
    log.info("Dynamic model weights written → %s", DYNAMIC_WEIGHTS_JSON)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN TRACKER
# ═══════════════════════════════════════════════════════════════════════════════

class ResultsTracker:
    """
    Daily results processor. Matches predictions to completed games,
    computes outcomes, updates the results log, and fires alerts.
    """

    def __init__(self, output_dir: Path = DATA_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_date(
        self,
        target_date: Optional[str] = None,
        dry_run: bool = False,
    ) -> Tuple[List[GameOutcome], List[Alert]]:
        """
        Process results for a specific date (or yesterday if None).
        Returns (outcomes, alerts).
        """
        if target_date is None:
            # Default: yesterday's games
            target_date = (datetime.now(TZ) - timedelta(days=1)).strftime("%Y%m%d")

        log.info(f"Processing results for: {target_date}")

        # ── Load predictions for this date ───────────────────────────────────
        preds = load_predictions(date_filter=target_date)
        if preds.empty:
            log.warning(f"No predictions found for {target_date}")
            return [], []

        # ── Load actual results ───────────────────────────────────────────────
        # Load ALL completed games (not filtered by prediction game_ids).
        # Date-based filtering and game_id matching happen via the merge.
        results = load_games_results(game_ids=None)

        if results.empty:
            log.warning("No completed game results found")
            return [], []

        # ── Match predictions to results ──────────────────────────────────────
        # Alias normalization: legacy sources may emit event_id; normalize to game_id for joins.
        if "event_id" in preds.columns and "game_id" not in preds.columns:
            preds = preds.rename(columns={"event_id": "game_id"})
        if "event_id" in results.columns and "game_id" not in results.columns:
            results = results.rename(columns={"event_id": "game_id"})
        preds["game_id"] = preds["game_id"].map(normalize_game_id)
        results["game_id"] = results["game_id"].map(normalize_game_id)

        # Ensure all required merge columns exist in results (fill missing
        # with NaN so the column selection never raises KeyError).
        merge_cols = ["game_id", "home_score", "away_score", "actual_margin",
                     "actual_total", "home_won", "spread", "over_under",
                     "home_ml", "away_ml", "game_datetime_utc", "neutral_site"]
        for col in merge_cols:
            if col not in results.columns:
                results[col] = np.nan

        matched = preds.merge(
            results[merge_cols],
            on="game_id",
            how="inner",
            suffixes=("_pred","_result"),
        )

        log.info(f"Matched {len(matched)} predictions to results "
                 f"({len(preds)} predicted, {len(results)} completed)")

        if matched.empty:
            log.warning("No matched predictions — game_ids may not align. "
                        f"Prediction game_ids sample: "
                        f"{preds['game_id'].head(3).tolist()}, "
                        f"Results game_ids sample: "
                        f"{results['game_id'].head(3).tolist()}")
            return [], []

        # ── Compute outcomes ──────────────────────────────────────────────────
        outcomes = []
        for _, row in matched.iterrows():
            # Resolve column name conflicts from merge
            result_row = pd.Series({
                "home_score":      row.get("home_score"),
                "away_score":      row.get("away_score"),
                "actual_margin":   row.get("actual_margin"),
                "actual_total":    row.get("actual_total"),
                "home_won":        row.get("home_won"),
                "spread":          row.get("spread"),
                "over_under":      row.get("over_under"),
                "home_ml":         row.get("home_ml"),
                "away_ml":         row.get("away_ml"),
                "game_datetime_utc": row.get("game_datetime_utc_result",
                                             row.get("game_datetime_utc", "")),
                "neutral_site":    row.get("neutral_site_result",
                                           row.get("neutral_site", 0)),
            })
            outcome = compute_outcomes(row, result_row)
            outcomes.append(outcome)

        log.info(f"Computed outcomes for {len(outcomes)} games")

        if dry_run:
            log.info("[DRY RUN] Not writing to results log")
            self._print_daily_summary(outcomes, [])
            return outcomes, []

        # ── Update results log ────────────────────────────────────────────────
        existing_log = load_results_log()
        new_rows     = pd.DataFrame([o.to_dict() for o in outcomes])
        pred_missing = int(new_rows.get("pred_spread", pd.Series(dtype=float)).isna().sum()) if not new_rows.empty else 0
        pred_total = len(new_rows)
        pred_filled_from_alias = int((new_rows.get("pred_spread_source", "") != "pred_spread").sum()) if "pred_spread_source" in new_rows.columns else 0
        log.info("[INTEGRITY] results_log pred_spread check: total=%s missing=%s alias_resolved=%s", pred_total, pred_missing, pred_filled_from_alias)

        if not existing_log.empty and "game_id" in existing_log.columns:
            # Deduplicate: overwrite existing game_ids with fresh outcomes
            existing_log["game_id"] = existing_log["game_id"].astype(str)
            new_rows["game_id"]     = new_rows["game_id"].astype(str)
            existing_clean = existing_log[
                ~existing_log["game_id"].isin(new_rows["game_id"])
            ]
            updated_log = pd.concat([existing_clean, new_rows], ignore_index=True)
        else:
            updated_log = new_rows

        # Sort by date
        if "game_datetime_utc" in updated_log.columns:
            updated_log["game_datetime_utc"] = pd.to_datetime(
                updated_log["game_datetime_utc"], utc=True, errors="coerce"
            )
            updated_log = updated_log.sort_values("game_datetime_utc")

        results_log_path = self.output_dir / "results_log.csv"
        safe_write_csv(updated_log, results_log_path, index=False)
        log.info(f"Results log updated: {len(updated_log):,} total records → {results_log_path}")

        by_team_df = _expand_to_team_rows(updated_log)
        by_team_path = self.output_dir / "results_log_by_team.csv"
        safe_write_csv(by_team_df, by_team_path, index=False, label="results_log_by_team", allow_empty=True)
        log.info(f"Results log by team updated: {len(by_team_df):,} total records → {by_team_path}")

        # ── Detect alerts ─────────────────────────────────────────────────────
        alerts = detect_alerts(updated_log)

        # ── Build and write summary ───────────────────────────────────────────
        self._write_summary(updated_log, alerts)

        # ── Print daily summary ───────────────────────────────────────────────
        self._print_daily_summary(outcomes, alerts)

        return outcomes, alerts

    def reprocess(self, days_back: int = 30) -> None:
        """
        Reprocess the last N days of predictions against results.
        Useful when prediction files have been updated or results corrected.
        """
        log.info(f"Reprocessing last {days_back} days...")
        start_date = datetime.now(TZ) - timedelta(days=days_back)

        all_outcomes = []
        for d in range(days_back, 0, -1):
            date_str = (datetime.now(TZ) - timedelta(days=d)).strftime("%Y%m%d")
            # Only process dates that have a date-specific prediction file.
            # Avoid falling back to *_latest.csv during reprocess — that would
            # produce duplicate entries for every day without a specific file.
            date_specific_files = [
                DATA_DIR / f"predictions_combined_{date_str}.csv",
                DATA_DIR / f"ensemble_predictions_{date_str}.csv",
                DATA_DIR / f"predictions_{date_str}.csv",
            ]
            if not any(f.exists() and f.stat().st_size > 50 for f in date_specific_files):
                continue
            outcomes, _ = self.process_date(target_date=date_str, dry_run=True)
            all_outcomes.extend(outcomes)

        if all_outcomes:
            new_rows = pd.DataFrame([o.to_dict() for o in all_outcomes])
            new_rows["game_id"] = new_rows["game_id"].astype(str)

            # Merge with entries outside the reprocess window rather than overwriting
            existing_log = load_results_log()
            if not existing_log.empty and "game_id" in existing_log.columns:
                existing_log["game_id"] = existing_log["game_id"].astype(str)
                outside_window = existing_log[
                    ~existing_log["game_id"].isin(new_rows["game_id"])
                ]
                merged = pd.concat([outside_window, new_rows], ignore_index=True)
            else:
                merged = new_rows

            if "game_datetime_utc" in merged.columns:
                merged["game_datetime_utc"] = pd.to_datetime(
                    merged["game_datetime_utc"], utc=True, errors="coerce"
                )
                # Sort with NaT values placed at the end
                merged = merged.sort_values("game_datetime_utc", na_position="last")

            safe_write_csv(merged, self.output_dir / "results_log.csv", index=False, label="results_log", allow_empty=True)
            by_team_df = _expand_to_team_rows(merged)
            safe_write_csv(by_team_df, self.output_dir / "results_log_by_team.csv", index=False, label="results_log_by_team", allow_empty=True)
            log.info(f"Reprocessed {len(all_outcomes)} outcomes → {len(merged)} total records")

            updated_log = load_results_log()
            alerts      = detect_alerts(updated_log)
            self._write_summary(updated_log, alerts)

    def _write_summary(self, log_df: pd.DataFrame, alerts: List[Alert]) -> None:
        """Write rolling window summary and alert files."""
        windows = [
            ("L7",     7),
            ("L14",    14),
            ("L30",    30),
            ("SEASON", None),
        ]

        summary_rows = []
        for label, days in windows:
            for source in ["primary", "ensemble"]:
                stats = compute_rolling_stats(log_df, days, label, source)
                row   = asdict(stats)
                row["source"] = source
                summary_rows.append(row)

        summary_df = pd.DataFrame(summary_rows)
        safe_write_csv(summary_df, self.output_dir / "results_summary.csv", index=False, label="results_summary", allow_empty=True)
        log.info(f"Summary written → {self.output_dir / 'results_summary.csv'}")

        # Alerts
        if alerts:
            alert_df = pd.DataFrame([a.to_dict() for a in alerts])
            safe_write_csv(alert_df, self.output_dir / "results_alerts.csv", index=False, label="results_alerts", allow_empty=True)
            log.info(f"Alerts written: {len(alerts)} active → {self.output_dir / 'results_alerts.csv'}")
        else:
            safe_write_csv(pd.DataFrame(), self.output_dir / "results_alerts.csv", index=False, label="results_alerts", allow_empty=True)

        # Per-model split table
        model_split = build_model_split_table(log_df)
        safe_write_csv(model_split, self.output_dir / "results_model_split.csv", index=False, label="results_model_split", allow_empty=True)

        write_dynamic_model_weights(log_df)

    def _print_daily_summary(
        self,
        outcomes: List[GameOutcome],
        alerts: List[Alert],
    ) -> None:
        """Print yesterday's results to stdout."""
        if not outcomes:
            print("No outcomes to display")
            return

        has_lines   = [o for o in outcomes if o.market_spread is not None]
        ats_correct = [o for o in has_lines if o.primary_ats_correct == 1]
        ens_correct = [o for o in has_lines if o.ens_ats_correct == 1]
        win_correct = [o for o in outcomes if o.primary_wins_game == 1]

        print()
        print("=" * 80)
        print(f"  DAILY RESULTS — {outcomes[0].game_date if outcomes else 'N/A'}")
        print("=" * 80)
        print(f"  Games processed:     {len(outcomes)}")
        print(f"  Games with lines:    {len(has_lines)}")

        if has_lines:
            prim_ats = len(ats_correct) / len(has_lines) * 100
            ens_ats  = len(ens_correct) / len(has_lines) * 100
            print(f"  Primary ATS:         {len(ats_correct)}/{len(has_lines)} ({prim_ats:.1f}%)")
            print(f"  Ensemble ATS:        {len(ens_correct)}/{len(has_lines)} ({ens_ats:.1f}%)")

        if outcomes:
            win_pct = len(win_correct) / len(outcomes) * 100
            print(f"  Winner picks:        {len(win_correct)}/{len(outcomes)} ({win_pct:.1f}%)")

        errors = [o.primary_margin_error for o in outcomes if o.primary_margin_error is not None]
        if errors:
            print(f"  Margin MAE:          {np.mean(np.abs(errors)):.1f} pts")

        print()
        print(f"  {'MATCHUP':<36} {'ACT':>6} {'PRED':>6} {'ERR':>6} "
              f"{'ATS':>4} {'ENS_ATS':>7}")
        print("  " + "-" * 65)

        for o in sorted(outcomes, key=lambda x: abs(x.primary_margin_error or 0), reverse=True):
            matchup  = f"{o.home_team[:16]} vs {o.away_team[:14]}"
            act_str  = f"{o.actual_margin:+.0f}"
            pred_str = f"{-o.pred_spread:+.0f}" if o.pred_spread is not None else "  N/A"
            err_str  = f"{o.primary_margin_error:+.0f}" if o.primary_margin_error is not None else "  N/A"
            ats_str  = {1: " ✓", 0: " ✗", None: " -"}[o.primary_ats_correct]
            ens_str  = {1: "  ✓", 0: "  ✗", None: "  -"}[o.ens_ats_correct]
            print(f"  {matchup:<36} {act_str:>6} {pred_str:>6} {err_str:>6} {ats_str:>4} {ens_str:>7}")

        if alerts:
            print()
            print(f"  ⚠️  {len(alerts)} ALERT(s) DETECTED:")
            for a in alerts:
                sev_icon = "🚨" if a.severity == "CRITICAL" else "⚠️ "
                print(f"  {sev_icon} [{a.alert_type}] {a.message}")

        print("=" * 80)
        print()

    def print_season_summary(self) -> None:
        """Load results log and print full season performance table."""
        log_df = load_results_log()
        if log_df.empty:
            print("No results in log yet")
            return

        print()
        print("=" * 90)
        print("  SEASON PERFORMANCE SUMMARY")
        print("=" * 90)

        for label, days in [("L7",7),("L14",14),("L30",30),("SEASON",None)]:
            stats = compute_rolling_stats(log_df, days, label, "primary")
            ens   = compute_rolling_stats(log_df, days, label, "ensemble")

            if stats.n_games == 0:
                continue

            ats_indicator = " ⚡" if stats.ats_pct > VIG_BREAK_EVEN else "  "
            print(f"\n  {label}  ({stats.n_games} games)")
            print(f"  {'':4} {'WIN%':>6} {'ATS%':>6} {'O/U%':>6} {'MAE':>6} "
                  f"{'BIAS':>6} {'EDGE_N':>7} {'EDGE%':>6} {'HI_CONF':>8} {'LO_CONF':>8}")
            print(f"  PRIM {stats.win_pct:>6.1f} {stats.ats_pct:>6.1f}{ats_indicator} "
                  f"{stats.ou_pct:>6.1f} {stats.margin_mae:>6.2f} {stats.margin_bias:>+6.2f} "
                  f"{stats.edge_n:>7} {stats.edge_ats_pct:>6.1f} "
                  f"{stats.high_conf_ats:>8.1f} {stats.low_conf_ats:>8.1f}")
            print(f"  ENS  {ens.win_pct:>6.1f} {ens.ats_pct:>6.1f}   "
                  f"{ens.ou_pct:>6.1f} {ens.margin_mae:>6.2f} {ens.margin_bias:>+6.2f}")

        # Per-model split
        model_split = build_model_split_table(log_df)
        if not model_split.empty:
            print()
            print("  PER-MODEL ATS%  (season / L30 / L7)")
            print(f"  {'MODEL':<16} {'SEASON':>8} {'L30':>8} {'L7':>8}")
            print("  " + "-" * 42)
            for _, row in model_split.iterrows():
                sea = row.get("SEASON_ats", 0)
                l30 = row.get("L30_ats", 0)
                l7  = row.get("L7_ats", 0)
                flag = " ⚡" if float(sea) > VIG_BREAK_EVEN else "  "
                print(f"  {str(row['model']):<16} {float(sea):>7.1f}%{flag} "
                      f"{float(l30):>7.1f}% {float(l7):>7.1f}%")

        print("=" * 90)
        print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="CBB Prediction Results Tracker")
    parser.add_argument("--date",            type=str, default=None,
                        help="Process specific date YYYYMMDD (default: yesterday)")
    parser.add_argument("--reprocess-days",  type=int, default=None,
                        help="Reprocess last N days of results")
    parser.add_argument("--summary",         action="store_true",
                        help="Print season summary only, no processing")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Compute outcomes but don't write to log")
    parser.add_argument("--output-dir",      type=Path, default=DATA_DIR)
    args = parser.parse_args()

    tracker = ResultsTracker(output_dir=args.output_dir)

    if args.summary:
        tracker.print_season_summary()
        return

    if args.reprocess_days:
        tracker.reprocess(days_back=args.reprocess_days)
        tracker.print_season_summary()
        return

    outcomes, alerts = tracker.process_date(
        target_date=args.date,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        tracker.print_season_summary()


if __name__ == "__main__":
    main()
