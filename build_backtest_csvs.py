#!/usr/bin/env python3
"""
build_backtest_csvs.py
======================
Reads data/results_log.csv and produces all backtest reporting CSVs.
Designed to run in under 10 seconds on a full season of data (~1,500 games).

Outputs:
  data/results_log_graded.csv         — Fully graded prediction log
  data/csv/backtest_summary.csv       — Overall performance by period
  data/csv/backtest_by_model.csv      — Per-model M1–M7 performance
  data/csv/backtest_weekly.csv        — Weekly trend per model
  data/csv/backtest_by_conference.csv — Conference-level breakdown
  data/csv/backtest_calibration.csv   — Confidence calibration table
  data/csv/backtest_by_edge.csv       — Edge tier performance
  data/csv/backtest_model_matrix.csv  — Head-to-head model comparison

CLI:
  python build_backtest_csvs.py                    # build all CSVs
  python build_backtest_csvs.py --section summary  # only backtest_summary.csv
  python build_backtest_csvs.py --section models   # only backtest_by_model.csv
  python build_backtest_csvs.py --section weekly   # only backtest_weekly.csv
  python build_backtest_csvs.py --section grade-only  # only grade + write graded log
  python build_backtest_csvs.py --since 2025-11-01 # filter to games after date
  python build_backtest_csvs.py --dry-run          # compute and print, don't write
  python build_backtest_csvs.py --validate         # check results_log integrity only
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import traceback
from datetime import datetime, timezone, timedelta
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd

from pipeline_csv_utils import normalize_game_id

# ── Paths ──────────────────────────────────────────────────────────────────
DATA        = pathlib.Path("data")
CSV_DIR     = DATA / "csv"
RESULTS_LOG = DATA / "results_log.csv"
GRADED_LOG  = DATA / "results_log_graded.csv"
WEIGHTS_JSON = DATA / "backtest_optimized_weights.json"
RANKINGS_CSV = DATA / "cbb_rankings.csv"
CLOSING_LINES = DATA / "market_lines_closing.csv"

NOW_UTC = datetime.now(timezone.utc)
NOW_ISO = NOW_UTC.strftime("%Y-%m-%dT%H:%M:%SZ")

# ── Model definitions ──────────────────────────────────────────────────────
MODELS = {
    "M1": "Four Factors",
    "M2": "Adj Efficiency",
    "M3": "Pythagorean",
    "M4": "Momentum",
    "M5": "Situational",
    "M6": "CAGE Rankings",
    "M7": "Regressed Efficiency",
}

EDGE_TIERS = [
    ("NO_LINE",  None,  None),
    ("SMALL",    0.0,   1.99),
    ("MEDIUM",   2.0,   3.74),
    ("VALUE",    3.75,  4.99),
    ("STRONG",   5.0,   6.99),
    ("LOCK",     7.0,   8.99),
    ("SCREAMER", 9.0,   None),
]

CONFIDENCE_BUCKETS = [
    ("50-54",  0.50, 0.54, 0.52),
    ("55-59",  0.55, 0.59, 0.57),
    ("60-64",  0.60, 0.64, 0.62),
    ("65-69",  0.65, 0.69, 0.67),
    ("70-74",  0.70, 0.74, 0.72),
    ("75-79",  0.75, 0.79, 0.77),
    ("80-84",  0.80, 0.84, 0.82),
    ("85-89",  0.85, 0.89, 0.87),
    ("90+",    0.90, 1.00, 0.95),
]

MIN_SAMPLE_MODEL   = 15
MIN_SAMPLE_CONF    = 10
MIN_SAMPLE_WEEKLY  = 5
MIN_SAMPLE_MATRIX  = 10
PUSH_THRESHOLD     = 0.1
CALIBRATION_TOLERANCE = 0.03
MODEL_COMPARISON_EVEN_THRESHOLD = 0.02
MIN_NUMERIC_RATIO  = 0.5


def compute_clv(pred_spread: float, opening_line: float, closing_line: float) -> dict:
    """Compute CLV vs open/close and whether we beat the close."""
    clv_vs_open = pred_spread - opening_line if pd.notna(opening_line) else None
    clv_vs_close = pred_spread - closing_line if pd.notna(closing_line) else None
    beat_close = None
    if pd.notna(closing_line):
        beat_close = abs(pred_spread) < abs(closing_line)

    return {
        "clv_vs_open": round(clv_vs_open, 2) if clv_vs_open is not None else None,
        "clv_vs_close": round(clv_vs_close, 2) if clv_vs_close is not None else None,
        "beat_closing_line": beat_close,
    }


def compute_weekly_sharpe(df: pd.DataFrame) -> Optional[float]:
    """Weekly Sharpe ratio of flat-bet ATS ROI."""
    working = df.copy()
    if "game_datetime_utc" not in working.columns:
        return None
    working["week"] = pd.to_datetime(working["game_datetime_utc"], errors="coerce", utc=True).dt.to_period("W")
    ats = working[working["ats_result"].isin(["WIN", "LOSS"])].copy()
    if ats.empty:
        return None

    ats["ats_win"] = (ats["ats_result"] == "WIN").astype(int)
    weekly_pnl = ats.groupby("week")["ats_win"].apply(
        lambda x: x.sum() * (100 / 110) - (len(x) - x.sum())
    )

    if len(weekly_pnl) < 4:
        return None

    mean_pnl = weekly_pnl.mean()
    std_pnl = weekly_pnl.std()
    if std_pnl == 0 or pd.isna(std_pnl):
        return None

    sharpe = (mean_pnl / std_pnl) * (30 ** 0.5)
    return round(float(sharpe), 3)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _load(path: pathlib.Path, label: str = "") -> Optional[pd.DataFrame]:
    """Load a CSV; return None with a warning if missing or empty."""
    if not path.exists() or path.stat().st_size < 10:
        print(f"[WARN] {label or path.name}: source file missing or empty — {path}")
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            print(f"[WARN] {label or path.name}: loaded but empty")
            return None
        if "game_id" in df.columns:
            df["game_id"] = df["game_id"].map(normalize_game_id)
        if "event_id" in df.columns:
            df["event_id"] = df["event_id"].map(normalize_game_id)
        return df
    except Exception as exc:
        print(f"[WARN] {label or path.name}: failed to load — {exc}")
        return None


def _write(df: pd.DataFrame, path: pathlib.Path, label: str, dry_run: bool = False) -> int:
    """Write CSV; return row count."""
    df = df.copy()
    df["generated_at"] = NOW_ISO
    n = len(df)
    if dry_run:
        print(f"[DRY]  {label}: {n} rows (not written)")
        return n
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[OK]   {label}: {n} rows")
    return n


def _safe_pct(series: pd.Series) -> Optional[float]:
    """Compute mean of a boolean series, return None if empty."""
    if series.empty:
        return None
    return round(float(series.mean()), 4)


def _safe_round(val, decimals=2):
    """Round a value; return None if NaN."""
    if pd.isna(val):
        return None
    return round(float(val), decimals)


def _parse_date_col(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise game date into '_date' column as UTC datetime.

    Handles three common formats stored in game_date:
      - ISO string  "2026-02-26" / "2026-02-26T..." → parse directly
      - YYYYMMDD integer 20260226 → convert to "2026-02-26" first;
        pd.to_datetime(20260226) wrongly treats it as nanoseconds → 1969-12-29
      - null / empty → fall through to game_datetime_utc
    """
    if "game_date" in df.columns and df["game_date"].notna().any():
        raw = df["game_date"].astype(str).str.strip()
        # Detect 8-digit YYYYMMDD integers stored as strings or floats
        yyyymmdd_mask = raw.str.match(r"^\d{8}$")
        if yyyymmdd_mask.any():
            raw = raw.where(
                ~yyyymmdd_mask,
                raw.str[:4] + "-" + raw.str[4:6] + "-" + raw.str[6:],
            )
        df["_date"] = pd.to_datetime(raw.replace({"nan": None, "": None}),
                                     errors="coerce", utc=True)
    elif "game_datetime_utc" in df.columns:
        df["_date"] = pd.to_datetime(df["game_datetime_utc"], errors="coerce", utc=True)
    else:
        df["_date"] = pd.NaT

    # If game_date was present but entirely null, fall back to game_datetime_utc
    if df.get("_date", pd.Series(dtype="object")).isna().all() and "game_datetime_utc" in df.columns:
        df["_date"] = pd.to_datetime(df["game_datetime_utc"], errors="coerce", utc=True)

    return df


def compute_streak(results: pd.Series) -> int:
    """
    results is a boolean Series (True=win, False=loss), chronological order.
    Returns positive int for current win streak, negative for loss streak.
    """
    if results.empty:
        return 0
    current = results.iloc[-1]
    streak = 0
    for r in reversed(results.tolist()):
        if r == current:
            streak += 1
        else:
            break
    return streak if current else -streak


def compute_longest_streaks(results: pd.Series):
    """Return (longest_win_streak, longest_loss_streak)."""
    if results.empty:
        return 0, 0
    max_win = 0
    max_loss = 0
    cur = 0
    prev = None
    for r in results.tolist():
        if r == prev:
            cur += 1
        else:
            cur = 1
            prev = r
        if r is True:
            max_win = max(max_win, cur)
        elif r is False:
            max_loss = max(max_loss, cur)
    return max_win, max_loss


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — GRADE EACH PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════

def grade_prediction(row: pd.Series) -> dict:
    """Returns grading columns for one prediction row."""

    # ── Winner (ML direction) ──────────────────────────────────────────
    predicted_winner = "home" if row["pred_spread"] < 0 else "away"
    actual_winner    = "home" if row["actual_spread"] < 0 else "away"
    winner_correct   = predicted_winner == actual_winner

    # ── ATS ────────────────────────────────────────────────────────────
    if pd.isna(row.get("spread_line")):
        ats_correct = None
        ats_result  = "NO_LINE"
    else:
        diff = row["actual_spread"] - row["spread_line"]
        if abs(diff) < PUSH_THRESHOLD:
            ats_correct = None
            ats_result  = "PUSH"
        elif row["pred_spread"] < row["spread_line"]:
            # model favors home more than line → bet home
            ats_correct = diff < 0
            ats_result  = "WIN" if ats_correct else "LOSS"
        else:
            # model favors away → bet away
            ats_correct = diff > 0
            ats_result  = "WIN" if ats_correct else "LOSS"

    # ── Totals ─────────────────────────────────────────────────────────
    if pd.isna(row.get("total_line")) or pd.isna(row.get("actual_total")):
        ou_correct = None
        ou_result  = "NO_LINE"
        total_pick = None
    else:
        total_pick = "OVER" if row["pred_total"] > row["total_line"] else "UNDER"
        total_diff = row["actual_total"] - row["total_line"]
        if abs(total_diff) < PUSH_THRESHOLD:
            ou_correct = None
            ou_result  = "PUSH"
        elif total_pick == "OVER":
            ou_correct = total_diff > 0
            ou_result  = "WIN" if ou_correct else "LOSS"
        else:
            ou_correct = total_diff < 0
            ou_result  = "WIN" if ou_correct else "LOSS"

    # ── ROI (flat 1 unit bet, -110 juice) ──────────────────────────────
    ats_roi = 0.909 if ats_correct is True else (-1.0 if ats_correct is False else 0.0)
    ou_roi  = 0.909 if ou_correct  is True else (-1.0 if ou_correct  is False else 0.0)

    # ── Spread error ───────────────────────────────────────────────────
    spread_error = abs(row["pred_spread"] - row["actual_spread"]) if pd.notna(row.get("actual_spread")) else None
    total_error  = abs(row["pred_total"]  - row["actual_total"])  if pd.notna(row.get("actual_total"))  else None

    opening_line = row.get("spread_line")
    closing_line = row.get("closing_spread_line", opening_line)
    clv = compute_clv(row["pred_spread"], opening_line, closing_line)

    return {
        "predicted_winner": predicted_winner,
        "actual_winner":    actual_winner,
        "winner_correct":   winner_correct,
        "ats_correct":      ats_correct,
        "ats_result":       ats_result,
        "ou_correct":       ou_correct,
        "ou_result":        ou_result,
        "total_pick":       total_pick,
        "ats_roi":          ats_roi,
        "ou_roi":           ou_roi,
        "spread_error":     spread_error,
        "total_error":      total_error,
        **clv,
    }


def grade_all(df: pd.DataFrame) -> pd.DataFrame:
    """Apply grading to every row; return df with grading columns added."""
    required = ["pred_spread", "actual_spread"]
    for col in required:
        if col not in df.columns:
            print(f"[FAIL] results_log.csv missing required column: {col}")
            sys.exit(1)

    # Ensure numeric
    for col in ["pred_spread", "actual_spread", "spread_line", "total_line",
                "pred_total", "actual_total", "model_confidence"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    grading = df.apply(grade_prediction, axis=1, result_type="expand")
    for col in grading.columns:
        df[col] = grading[col]
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — OVERALL PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_period_stats(df: pd.DataFrame, label: str) -> dict:
    """Compute stats for a period slice of graded data."""
    graded_ats = df[df["ats_result"].isin(["WIN", "LOSS"])]
    graded_ou  = df[df["ou_result"].isin(["WIN", "LOSS"])]
    graded_ml  = df[df["winner_correct"].notna()]

    edge_series = df["edge_flag"] if "edge_flag" in df.columns else pd.Series(False, index=df.index)
    edge_games = graded_ats[edge_series.reindex(graded_ats.index).fillna(False).astype(bool)]

    high_conf = graded_ats[graded_ats["model_confidence"] >= 0.75] if "model_confidence" in graded_ats.columns else pd.DataFrame()

    return {
        "period":              label,
        "total_games":         len(df),

        # Winner / ML direction
        "ml_graded":           len(graded_ml),
        "ml_correct":          int(graded_ml["winner_correct"].sum()) if len(graded_ml) else 0,
        "ml_pct":              _safe_pct(graded_ml["winner_correct"]),

        # ATS
        "ats_graded":          len(graded_ats),
        "ats_wins":            int((graded_ats["ats_result"] == "WIN").sum()),
        "ats_losses":          int((graded_ats["ats_result"] == "LOSS").sum()),
        "ats_pushes":          int((df["ats_result"] == "PUSH").sum()),
        "ats_pct":             _safe_pct(graded_ats["ats_result"] == "WIN"),
        "ats_roi_units":       _safe_round(graded_ats["ats_roi"].sum()) if len(graded_ats) else 0.0,

        # Totals
        "ou_graded":           len(graded_ou),
        "ou_wins":             int((graded_ou["ou_result"] == "WIN").sum()),
        "ou_losses":           int((graded_ou["ou_result"] == "LOSS").sum()),
        "ou_pushes":           int((df["ou_result"] == "PUSH").sum()),
        "ou_pct":              _safe_pct(graded_ou["ou_result"] == "WIN"),
        "ou_roi_units":        _safe_round(graded_ou["ou_roi"].sum()) if len(graded_ou) else 0.0,

        # Edge games
        "edge_graded":         len(edge_games),
        "edge_ats_wins":       int((edge_games["ats_result"] == "WIN").sum()) if len(edge_games) else 0,
        "edge_ats_pct":        _safe_pct(edge_games["ats_result"] == "WIN") if len(edge_games) else None,
        "edge_roi_units":      _safe_round(edge_games["ats_roi"].sum()) if len(edge_games) else None,

        # Accuracy
        "avg_spread_error":    _safe_round(df["spread_error"].mean()),
        "avg_total_error":     _safe_round(df["total_error"].mean()),
        "median_spread_error": _safe_round(df["spread_error"].median()),
        "mean_clv_vs_open":    _safe_round(df["clv_vs_open"].mean()) if "clv_vs_open" in df.columns else None,
        "mean_clv_vs_close":   _safe_round(df["clv_vs_close"].mean()) if "clv_vs_close" in df.columns else None,
        "beat_close_pct":      _safe_pct(df["beat_closing_line"]) if "beat_closing_line" in df.columns else None,
        "sharpe_ratio":        compute_weekly_sharpe(df),

        # Confidence calibration
        "avg_confidence":      _safe_round(df["model_confidence"].mean(), 3) if "model_confidence" in df.columns else None,
        "high_conf_ats_pct":   _safe_pct(high_conf["ats_result"] == "WIN") if len(high_conf) else None,
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build Section B: overall summary across time periods."""
    df = _parse_date_col(df)
    now = pd.Timestamp.now(tz="UTC")

    periods = [
        ("season", df),
        ("l60",    df[df["_date"] >= now - pd.Timedelta(days=60)]),
        ("l30",    df[df["_date"] >= now - pd.Timedelta(days=30)]),
        ("l14",    df[df["_date"] >= now - pd.Timedelta(days=14)]),
        ("l7",     df[df["_date"] >= now - pd.Timedelta(days=7)]),
    ]

    rows = [compute_period_stats(sub, label) for label, sub in periods]
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — PER-MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_sub_model_source(df: pd.DataFrame):
    """Determine how to split by sub_model. Returns (source, df_with_model_col)."""
    if "sub_model" in df.columns:
        valid = df["sub_model"].dropna()
        valid_models = valid[valid.str.match(r"^M[1-7]$")]
        if len(valid_models) >= 10:
            return "sub_model_column", df
        if len(valid) > 0:
            return "mixed", df

    # Fallback: use confidence quintiles as proxy
    if "model_confidence" in df.columns and df["model_confidence"].notna().sum() > 20:
        print("[WARN] sub_model missing — creating synthetic M1-M7 labels from confidence quintiles")
        df = df.copy()
        labels = [f"M{i}" for i in range(1, min(8, max(3, len(df) // 10 + 1)))]
        df["sub_model"] = pd.qcut(
            df["model_confidence"].rank(method="first"),
            q=min(7, max(2, len(df) // 10)),
            labels=labels,
            duplicates="drop",
        )
        # Guardrail: avoid presenting confidence-derived bins as real model IDs.
        df["sub_model"] = "PROXY_" + df["sub_model"].astype(str)
        return "confidence_proxy", df

    return "unknown", df


def compute_model_stats(df_all: pd.DataFrame, model_df: pd.DataFrame,
                        model_id: str, model_name: str,
                        data_source: str, weights: dict) -> dict:
    """Compute per-model stats (Section C)."""
    model_df = _parse_date_col(model_df)
    now = pd.Timestamp.now(tz="UTC")

    # Period slices
    season = model_df
    l30 = model_df[model_df["_date"] >= now - pd.Timedelta(days=30)]
    l14 = model_df[model_df["_date"] >= now - pd.Timedelta(days=14)]
    l7  = model_df[model_df["_date"] >= now - pd.Timedelta(days=7)]

    # Graded subsets
    def _ats_graded(d):
        return d[d["ats_result"].isin(["WIN", "LOSS"])]

    def _ou_graded(d):
        return d[d["ou_result"].isin(["WIN", "LOSS"])]

    def _ml_graded(d):
        return d[d["winner_correct"].notna()]

    s_ats = _ats_graded(season)
    s_ou  = _ou_graded(season)
    s_ml  = _ml_graded(season)
    insufficient = len(s_ats) < MIN_SAMPLE_MODEL

    def _pct_or_null(series, min_n=MIN_SAMPLE_MODEL):
        if len(series) < min_n:
            return None
        return _safe_pct(series)

    # vs ensemble comparison (simplified — compare against overall season)
    all_ats = df_all[df_all["ats_result"].isin(["WIN", "LOSS"])]
    beats_ensemble_ats = None
    agrees_with_ensemble = None
    if "ens_ensemble_spread" in df_all.columns and len(s_ats) >= MIN_SAMPLE_MODEL:
        model_ids_in_all = model_df.index.intersection(all_ats.index)
        if len(model_ids_in_all) > 0:
            overlap = all_ats.loc[model_ids_in_all]
            if "ens_ats_correct" in overlap.columns:
                model_correct = overlap["ats_result"] == "WIN"
                ens_correct   = overlap["ens_ats_correct"].astype(bool)
                beats = (model_correct & ~ens_correct).sum()
                beats_ensemble_ats = round(float(beats / len(overlap)), 4) if len(overlap) else None
                agrees_with_ensemble = round(float((model_correct == ens_correct).mean()), 4) if len(overlap) else None

    # Spread accuracy
    se = season["spread_error"].dropna()

    # Weights
    w_spread = weights.get("weights", {}).get(f"model_{model_id.lower()}_spread",
               weights.get("weights", {}).get(model_id, None))
    w_total  = weights.get("weights", {}).get(f"model_{model_id.lower()}_total", None)

    # Weight justified
    season_ats_pct = _safe_pct(s_ats["ats_result"] == "WIN") if len(s_ats) else None
    l30_ats_pct    = _safe_pct(_ats_graded(l30)["ats_result"] == "WIN") if len(_ats_graded(l30)) >= MIN_SAMPLE_MODEL else None
    weight_justified = None
    if season_ats_pct is not None and l30_ats_pct is not None:
        weight_justified = l30_ats_pct > season_ats_pct

    # Streaks
    ats_results_bool = s_ats.sort_values("_date")["ats_result"].map({"WIN": True, "LOSS": False}).dropna()
    current_streak   = compute_streak(ats_results_bool)
    longest_win, longest_loss = compute_longest_streaks(ats_results_bool)

    # Conference breakdown
    best_conf = worst_conf = None
    best_conf_pct = worst_conf_pct = None
    if "conference" in season.columns:
        conf_stats = []
        for conf, cdf in _ats_graded(season).groupby("conference"):
            if len(cdf) >= 10:
                pct = float((cdf["ats_result"] == "WIN").mean())
                conf_stats.append((conf, pct))
        if conf_stats:
            conf_stats.sort(key=lambda x: x[1], reverse=True)
            best_conf, best_conf_pct = conf_stats[0]
            worst_conf, worst_conf_pct = conf_stats[-1]

    data_quality_notes = None
    if insufficient:
        data_quality_notes = "INSUFFICIENT_SAMPLE"
    elif data_source == "confidence_proxy":
        data_quality_notes = "sub_model derived from confidence quintiles"
    elif data_source == "unknown":
        data_quality_notes = "sub_model column not available"

    return {
        "model_id":               model_id,
        "model_name":             model_name,
        "data_source":            data_source,
        "data_quality_notes":     data_quality_notes,
        # Season
        "season_ml_graded":       len(s_ml),
        "season_ml_pct":          _pct_or_null(s_ml["winner_correct"]),
        "season_ats_graded":      len(s_ats),
        "season_ats_wins":        int((s_ats["ats_result"] == "WIN").sum()),
        "season_ats_losses":      int((s_ats["ats_result"] == "LOSS").sum()),
        "season_ats_pct":         _pct_or_null(s_ats["ats_result"] == "WIN"),
        "season_ats_roi":         _safe_round(s_ats["ats_roi"].sum()) if len(s_ats) else 0.0,
        "season_ou_graded":       len(s_ou),
        "season_ou_wins":         int((s_ou["ou_result"] == "WIN").sum()),
        "season_ou_losses":       int((s_ou["ou_result"] == "LOSS").sum()),
        "season_ou_pct":          _pct_or_null(s_ou["ou_result"] == "WIN"),
        "season_ou_roi":          _safe_round(s_ou["ou_roi"].sum()) if len(s_ou) else 0.0,
        "season_avg_spread_error": _safe_round(se.mean()) if len(se) else None,
        "season_within_3pts_pct": round(float((se <= 3).mean()), 4) if len(se) else None,
        "season_within_7pts_pct": round(float((se <= 7).mean()), 4) if len(se) else None,
        # L30
        "l30_ml_pct":             _pct_or_null(_ml_graded(l30)["winner_correct"]),
        "l30_ats_graded":         len(_ats_graded(l30)),
        "l30_ats_pct":            l30_ats_pct,
        "l30_ats_roi":            _safe_round(_ats_graded(l30)["ats_roi"].sum()) if len(_ats_graded(l30)) else 0.0,
        "l30_ou_pct":             _pct_or_null(_ou_graded(l30)["ou_result"] == "WIN"),
        # L14
        "l14_ats_pct":            _pct_or_null(_ats_graded(l14)["ats_result"] == "WIN"),
        "l14_ou_pct":             _pct_or_null(_ou_graded(l14)["ou_result"] == "WIN"),
        # L7
        "l7_ats_pct":             _pct_or_null(_ats_graded(l7)["ats_result"] == "WIN"),
        "l7_ou_pct":              _pct_or_null(_ou_graded(l7)["ou_result"] == "WIN"),
        # vs ensemble
        "beats_ensemble_ats":     beats_ensemble_ats,
        "agrees_with_ensemble":   agrees_with_ensemble,
        # weights
        "current_weight_spread":  w_spread,
        "current_weight_total":   w_total,
        "weight_justified":       weight_justified,
        # streaks
        "current_ats_streak":     current_streak,
        "longest_win_streak":     longest_win,
        "longest_loss_streak":    longest_loss,
        # conference
        "best_conf":              best_conf,
        "best_conf_ats_pct":      round(best_conf_pct, 4) if best_conf_pct is not None else None,
        "worst_conf":             worst_conf,
        "worst_conf_ats_pct":     round(worst_conf_pct, 4) if worst_conf_pct is not None else None,
    }


def build_by_model(df: pd.DataFrame) -> pd.DataFrame:
    """Build Section C: per-model performance."""
    # Load weights
    weights = {}
    if WEIGHTS_JSON.exists():
        try:
            with open(WEIGHTS_JSON) as f:
                weights = json.load(f)
        except Exception:
            pass

    source, df = _detect_sub_model_source(df)

    if source == "unknown" or "sub_model" not in df.columns:
        print("[WARN] sub_model column not available — cannot build per-model breakdown")
        return pd.DataFrame()

    warns = []
    rows = []
    model_ids = [m for m in sorted(df["sub_model"].dropna().astype(str).unique())]
    for mid in model_ids:
        mname = MODELS.get(mid, mid)
        mdf = df[df["sub_model"] == mid]
        if mdf.empty:
            continue
        row = compute_model_stats(df, mdf, mid, mname, source, weights)
        rows.append(row)
        if row.get("data_quality_notes") == "INSUFFICIENT_SAMPLE":
            warns.append(mid)

    if warns:
        for m in warns:
            ats_n = len(df[(df["sub_model"] == m) & df["ats_result"].isin(["WIN", "LOSS"])])
            print(f"[WARN] {m} has only {ats_n} graded games — stats marked INSUFFICIENT_SAMPLE")

    result = pd.DataFrame(rows)
    if not result.empty and "season_ats_pct" in result.columns:
        result = result.sort_values("season_ats_pct", ascending=False, na_position="last")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — WEEKLY PERFORMANCE LOG
# ═══════════════════════════════════════════════════════════════════════════════

def build_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Build Section D: weekly performance per model."""
    df = _parse_date_col(df)
    if df["_date"].isna().all():
        print("[WARN] No valid dates — cannot build weekly breakdown")
        return pd.DataFrame()

    if "week_label" not in df.columns:
        df["week_label"] = df["_date"].dt.isocalendar().week.astype(str)

    # Group by week
    # Handle mixed timezones by converting all to UTC naive if aware
    dates_no_tz = df["_date"].apply(
        lambda dt: dt.tz_convert(None) if hasattr(dt, "tz") and dt.tz is not None else dt
    )
    df["_week_start"] = dates_no_tz.dt.to_period("W").apply(
        lambda p: p.start_time if pd.notna(p) else pd.NaT
    )

    rows = []
    for (wl, ws), wdf in df.groupby(["week_label", "_week_start"], dropna=False):
        ats_g = wdf[wdf["ats_result"].isin(["WIN", "LOSS"])]
        ou_g  = wdf[wdf["ou_result"].isin(["WIN", "LOSS"])]
        ml_g  = wdf[wdf["winner_correct"].notna()]

        if len(ats_g) < MIN_SAMPLE_WEEKLY and len(ml_g) < MIN_SAMPLE_WEEKLY:
            continue

        edge_series = wdf["edge_flag"] if "edge_flag" in wdf.columns else pd.Series(False, index=wdf.index)
        edge_g = ats_g[edge_series.reindex(ats_g.index).fillna(False).astype(bool)]

        row = {
            "week_label":       wl,
            "week_start_date":  ws.strftime("%Y-%m-%d") if pd.notna(ws) else None,
            "total_games":      len(wdf),
            "ats_wins":         int((ats_g["ats_result"] == "WIN").sum()),
            "ats_losses":       int((ats_g["ats_result"] == "LOSS").sum()),
            "ats_pct":          _safe_pct(ats_g["ats_result"] == "WIN"),
            "ats_roi":          _safe_round(ats_g["ats_roi"].sum()) if len(ats_g) else 0.0,
            "ou_wins":          int((ou_g["ou_result"] == "WIN").sum()),
            "ou_losses":        int((ou_g["ou_result"] == "LOSS").sum()),
            "ou_pct":           _safe_pct(ou_g["ou_result"] == "WIN"),
            "ou_roi":           _safe_round(ou_g["ou_roi"].sum()) if len(ou_g) else 0.0,
            "ml_correct":       int(ml_g["winner_correct"].sum()) if len(ml_g) else 0,
            "ml_pct":           _safe_pct(ml_g["winner_correct"]),
            "edge_games":       len(edge_g),
            "edge_ats_pct":     _safe_pct(edge_g["ats_result"] == "WIN") if len(edge_g) else None,
            "avg_spread_error": _safe_round(wdf["spread_error"].mean()),
        }

        # Per-model columns
        if "sub_model" in wdf.columns:
            for mid in MODELS:
                mdf_ats = ats_g[ats_g["sub_model"] == mid]
                mdf_ou  = ou_g[ou_g["sub_model"] == mid]
                mdf_ml  = ml_g[ml_g["sub_model"] == mid]
                row[f"{mid}_ats_pct"] = _safe_pct(mdf_ats["ats_result"] == "WIN") if len(mdf_ats) else None
                row[f"{mid}_ou_pct"]  = _safe_pct(mdf_ou["ou_result"] == "WIN") if len(mdf_ou) else None
                row[f"{mid}_ml_pct"]  = _safe_pct(mdf_ml["winner_correct"]) if len(mdf_ml) else None
                row[f"{mid}_ats_roi"] = _safe_round(mdf_ats["ats_roi"].sum()) if len(mdf_ats) else None
        else:
            for mid in MODELS:
                row[f"{mid}_ats_pct"] = None
                row[f"{mid}_ou_pct"]  = None
                row[f"{mid}_ml_pct"]  = None
                row[f"{mid}_ats_roi"] = None

        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty and "week_start_date" in result.columns:
        result = result.sort_values("week_start_date", ascending=True)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION E — CONFERENCE PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def build_by_conference(df: pd.DataFrame) -> pd.DataFrame:
    """Build Section E: conference breakdown."""
    if "conference" not in df.columns:
        print("[WARN] No conference column — cannot build conference breakdown")
        return pd.DataFrame()

    rows = []
    for conf, cdf in df.groupby("conference"):
        ats_g = cdf[cdf["ats_result"].isin(["WIN", "LOSS"])]
        ou_g  = cdf[cdf["ou_result"].isin(["WIN", "LOSS"])]
        ml_g  = cdf[cdf["winner_correct"].notna()]

        if len(ats_g) < MIN_SAMPLE_CONF:
            continue

        edge_series = cdf["edge_flag"] if "edge_flag" in cdf.columns else pd.Series(False, index=cdf.index)
        edge_g = ats_g[edge_series.reindex(ats_g.index).fillna(False).astype(bool)]

        # Best/worst model in this conference
        best_model = worst_model = None
        best_model_pct = worst_model_pct = None
        if "sub_model" in cdf.columns:
            model_stats = []
            for mid in MODELS:
                m_ats = ats_g[ats_g["sub_model"] == mid]
                if len(m_ats) >= 5:
                    pct = float((m_ats["ats_result"] == "WIN").mean())
                    model_stats.append((mid, pct))
            if model_stats:
                model_stats.sort(key=lambda x: x[1], reverse=True)
                best_model, best_model_pct = model_stats[0]
                worst_model, worst_model_pct = model_stats[-1]

        rows.append({
            "conference":        conf,
            "ats_graded":        len(ats_g),
            "ats_wins":          int((ats_g["ats_result"] == "WIN").sum()),
            "ats_losses":        int((ats_g["ats_result"] == "LOSS").sum()),
            "ats_pct":           _safe_pct(ats_g["ats_result"] == "WIN"),
            "ats_roi":           _safe_round(ats_g["ats_roi"].sum()),
            "ou_graded":         len(ou_g),
            "ou_pct":            _safe_pct(ou_g["ou_result"] == "WIN"),
            "ou_roi":            _safe_round(ou_g["ou_roi"].sum()) if len(ou_g) else 0.0,
            "ml_pct":            _safe_pct(ml_g["winner_correct"]),
            "edge_graded":       len(edge_g),
            "edge_ats_pct":      _safe_pct(edge_g["ats_result"] == "WIN") if len(edge_g) else None,
            "avg_spread_error":  _safe_round(cdf["spread_error"].mean()),
            "best_model":        best_model,
            "best_model_ats_pct": round(best_model_pct, 4) if best_model_pct is not None else None,
            "worst_model":       worst_model,
            "worst_model_ats_pct": round(worst_model_pct, 4) if worst_model_pct is not None else None,
            "games_this_season": len(cdf),
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("ats_pct", ascending=False, na_position="last")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION F — CONFIDENCE CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def build_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Build Section F: confidence calibration table."""
    if "model_confidence" not in df.columns:
        print("[WARN] No model_confidence column — cannot build calibration table")
        return pd.DataFrame()

    rows = []
    for label, lo, hi, midpoint in CONFIDENCE_BUCKETS:
        mask = (df["model_confidence"] >= lo) & (df["model_confidence"] <= hi)
        bdf = df[mask]
        ats_g = bdf[bdf["ats_result"].isin(["WIN", "LOSS"])]
        ou_g  = bdf[bdf["ou_result"].isin(["WIN", "LOSS"])]
        ml_g  = bdf[bdf["winner_correct"].notna()]

        if len(ats_g) < MIN_SAMPLE_CONF:
            continue

        ats_pct = float((ats_g["ats_result"] == "WIN").mean())
        cal_error = round(ats_pct - midpoint, 4)

        if abs(cal_error) <= CALIBRATION_TOLERANCE:
            cal_status = "WELL_CALIBRATED"
        elif cal_error < 0:
            cal_status = "OVERCONFIDENT"
        else:
            cal_status = "UNDERCONFIDENT"

        rows.append({
            "confidence_bucket":  label,
            "bucket_midpoint":    midpoint,
            "ats_graded":         len(ats_g),
            "ats_pct":            round(ats_pct, 4),
            "ou_pct":             _safe_pct(ou_g["ou_result"] == "WIN"),
            "ml_pct":             _safe_pct(ml_g["winner_correct"]),
            "expected_win_rate":  midpoint,
            "calibration_error":  cal_error,
            "calibration_status": cal_status,
            "roi_units":          _safe_round(ats_g["ats_roi"].sum()),
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION G — EDGE TIER PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════

def build_by_edge(df: pd.DataFrame) -> pd.DataFrame:
    """Build Section G: edge tier performance."""
    if "spread_line" not in df.columns:
        print("[WARN] No spread_line column — cannot compute edge tiers")
        return pd.DataFrame()

    # Compute edge size (absolute difference between pred_spread and spread_line)
    df = df.copy()
    df["_edge_size"] = np.where(
        df["spread_line"].notna(),
        (df["pred_spread"] - df["spread_line"]).abs(),
        np.nan,
    )

    rows = []
    for tier_name, lo, hi in EDGE_TIERS:
        if tier_name == "NO_LINE":
            mask = df["spread_line"].isna()
            edge_range = "N/A"
        elif hi is None:
            mask = df["_edge_size"] >= lo
            edge_range = f"{lo:.1f}+"
        else:
            mask = (df["_edge_size"] >= lo) & (df["_edge_size"] <= hi)
            edge_range = f"{lo:.2f}-{hi:.2f}"

        tdf = df[mask]
        ats_g = tdf[tdf["ats_result"].isin(["WIN", "LOSS"])]
        ou_g  = tdf[tdf["ou_result"].isin(["WIN", "LOSS"])]
        ml_g  = tdf[tdf["winner_correct"].notna()]

        rows.append({
            "edge_tier":              tier_name,
            "edge_range":             edge_range,
            "ats_graded":             len(ats_g),
            "ats_wins":               int((ats_g["ats_result"] == "WIN").sum()),
            "ats_losses":             int((ats_g["ats_result"] == "LOSS").sum()),
            "ats_pushes":             int((tdf["ats_result"] == "PUSH").sum()),
            "ats_pct":                _safe_pct(ats_g["ats_result"] == "WIN"),
            "ats_roi_units":          _safe_round(ats_g["ats_roi"].sum()) if len(ats_g) else 0.0,
            "ou_pct":                 _safe_pct(ou_g["ou_result"] == "WIN"),
            "ml_pct":                 _safe_pct(ml_g["winner_correct"]),
            "avg_actual_spread_error": _safe_round(tdf["spread_error"].mean()),
            "avg_kelly_units":        _safe_round(tdf["kelly_units"].mean()) if "kelly_units" in tdf.columns else None,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION H — HEAD-TO-HEAD MODEL COMPARISON MATRIX
# ═══════════════════════════════════════════════════════════════════════════════

def build_model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Build Section H: head-to-head model comparison."""
    work = df.copy()

    # Identify if we need to melt columns (per-game format with M1..M7 spreads as columns)
    model_cols = [f"{m.lower()}_spread" for m in MODELS.keys()]
    present_cols = [c for c in model_cols if c in work.columns]

    if present_cols and ("sub_model" not in work.columns or work["sub_model"].isna().all()):
        if "game_id" not in work.columns:
            print("[WARN] Cannot build model matrix without game_id column for wide format")
            return pd.DataFrame()

        melted = []
        for m_id in MODELS.keys():
            m_col = f"{m_id.lower()}_spread"
            if m_col in work.columns:
                m_df = work.copy()
                m_df["pred_spread"] = m_df[m_col]
                # Re-grade just for this sub-model to get its specific ats_result
                grading = m_df.apply(grade_prediction, axis=1, result_type="expand")
                m_df["ats_result"] = grading["ats_result"]
                m_df["sub_model"] = m_id
                melted.append(m_df[["game_id", "sub_model", "ats_result"]])

        ats_g = pd.concat(melted)
        ats_g = ats_g[ats_g["ats_result"].isin(["WIN", "LOSS"])]
    else:
        if "sub_model" not in work.columns:
            print("[WARN] No sub_model column — cannot build model matrix")
            return pd.DataFrame()
        ats_g = work[work["ats_result"].isin(["WIN", "LOSS"])]

    available_models = sorted(set(ats_g["sub_model"].dropna().astype(str)))

    if len(available_models) < 2:
        print("[WARN] Fewer than 2 models available — cannot build comparison matrix")
        return pd.DataFrame()

    if "game_id" not in ats_g.columns:
        print("[WARN] No game_id column — cannot build comparison matrix")
        return pd.DataFrame()

    matrix = ats_g.pivot_table(
        index="game_id",
        columns="sub_model",
        values="ats_result",
        aggfunc="first",
    )
    matrix = matrix[[c for c in available_models if c in matrix.columns]]

    if matrix.shape[1] < 2:
        print("[WARN] sub_model missing or only one unique value — cannot build comparison matrix")
        return pd.DataFrame()

    rows = []
    for ma, mb in combinations(available_models, 2):
        if ma not in matrix.columns or mb not in matrix.columns:
            rows.append({
                "model_a": ma, "model_b": mb,
                "agreement_rate": None, "disagreement_games": 0,
                "model_a_wins_when_split": None,
                "model_b_wins_when_split": None,
                "better_model_when_split": None,
            })
            continue

        pair = matrix[[ma, mb]].dropna()
        if pair.empty:
            rows.append({
                "model_a": ma, "model_b": mb,
                "agreement_rate": None, "disagreement_games": 0,
                "model_a_wins_when_split": None,
                "model_b_wins_when_split": None,
                "better_model_when_split": None,
            })
            continue

        ma_res = pair[ma]
        mb_res = pair[mb]

        agree = (ma_res == mb_res).sum()
        agreement_rate = round(float(agree / len(pair)), 4)

        disagree_mask = ma_res != mb_res
        n_disagree = int(disagree_mask.sum())

        a_wins = b_wins = None
        better = None
        if n_disagree >= MIN_SAMPLE_MATRIX:
            a_wins = round(float((ma_res[disagree_mask] == "WIN").mean()), 4)
            b_wins = round(float((mb_res[disagree_mask] == "WIN").mean()), 4)
            if abs(a_wins - b_wins) < MODEL_COMPARISON_EVEN_THRESHOLD:
                better = "EVEN"
            elif a_wins > b_wins:
                better = "A"
            else:
                better = "B"

        rows.append({
            "model_a":                 ma,
            "model_b":                 mb,
            "agreement_rate":          agreement_rate,
            "disagreement_games":      n_disagree,
            "model_a_wins_when_split": a_wins,
            "model_b_wins_when_split": b_wins,
            "better_model_when_split": better,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# HISTORICAL MARKET LINE ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def enrich_spread_lines_from_closing(
    df: pd.DataFrame,
    closing_path: Optional[pathlib.Path] = None,
) -> pd.DataFrame:
    """Back-fill missing spread_line / total_line from market_lines_closing.csv.

    Rows that already have a non-null ``spread_line`` are left unchanged.
    When ``market_lines_closing.csv`` is absent or empty the DataFrame is
    returned unmodified.
    """
    if closing_path is None:
        closing_path = CLOSING_LINES
    if not closing_path.exists() or closing_path.stat().st_size < 10:
        return df

    try:
        closing = pd.read_csv(closing_path, dtype={"espn_game_id": str}, low_memory=False)
    except Exception as exc:
        print(f"[WARN] enrich_spread_lines_from_closing: could not load {closing_path} — {exc}")
        return df

    if closing.empty:
        return df

    # Normalise the game-id column name
    if "espn_game_id" in closing.columns:
        closing = closing.rename(columns={"espn_game_id": "_closing_id"})
    else:
        print("[WARN] enrich_spread_lines_from_closing: no espn_game_id column — skipping")
        return df

    closing["_closing_id"] = closing["_closing_id"].map(normalize_game_id)

    # Identify the canonical game-id column in the results DataFrame
    id_col = None
    for candidate in ("game_id", "event_id"):
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        print("[WARN] enrich_spread_lines_from_closing: no game_id/event_id column — skipping")
        return df

    df = df.copy()
    df["_norm_id"] = df[id_col].map(normalize_game_id)

    # Determine which rows are missing spread_line
    needs_spread = "spread_line" not in df.columns or df["spread_line"].isna().any()
    needs_total  = "total_line"  not in df.columns or df["total_line"].isna().any()

    if not needs_spread and not needs_total:
        df = df.drop(columns=["_norm_id"])
        return df

    # Build a lookup from the closing file
    lookup_cols = {"_closing_id": "_closing_id"}
    if "close_home_spread" in closing.columns and needs_spread:
        lookup_cols["close_home_spread"] = "_close_spread"
    if "close_total" in closing.columns and needs_total:
        lookup_cols["close_total"] = "_close_total"

    if len(lookup_cols) == 1:  # only id col
        df = df.drop(columns=["_norm_id"])
        return df

    closing_lookup = closing[list(lookup_cols.keys())].rename(columns=lookup_cols).drop_duplicates("_closing_id")

    df = df.merge(closing_lookup, left_on="_norm_id", right_on="_closing_id", how="left")

    if "_close_spread" in df.columns:
        if "spread_line" not in df.columns:
            df["spread_line"] = pd.to_numeric(df["_close_spread"], errors="coerce")
        else:
            df["spread_line"] = pd.to_numeric(df["spread_line"], errors="coerce").fillna(
                pd.to_numeric(df["_close_spread"], errors="coerce")
            )
        df = df.drop(columns=["_close_spread"])

    if "_close_total" in df.columns:
        if "total_line" not in df.columns:
            df["total_line"] = pd.to_numeric(df["_close_total"], errors="coerce")
        else:
            df["total_line"] = pd.to_numeric(df["total_line"], errors="coerce").fillna(
                pd.to_numeric(df["_close_total"], errors="coerce")
            )
        df = df.drop(columns=["_close_total"])

    df = df.drop(columns=[c for c in ["_norm_id", "_closing_id"] if c in df.columns])

    filled = int(df["spread_line"].notna().sum()) if "spread_line" in df.columns else 0
    print(f"[BACKTEST] enrich_spread_lines_from_closing: {filled} rows now have spread_line "
          f"(source: {closing_path.name})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_results_log(df: pd.DataFrame) -> bool:
    """Check results_log.csv integrity."""
    ok = True
    print(f"[BACKTEST] Validating results_log.csv: {len(df)} rows")

    required = ["pred_spread", "actual_spread"]
    for col in required:
        if col not in df.columns:
            print(f"[FAIL] Missing required column: {col}")
            ok = False

    optional = ["spread_line", "total_line", "pred_total", "actual_total",
                "model_confidence", "edge_flag", "conference", "sub_model",
                "game_id", "game_date", "home_team", "away_team"]
    for col in optional:
        if col not in df.columns:
            print(f"[WARN] Optional column missing: {col}")

    # Numeric checks
    for col in ["pred_spread", "actual_spread"]:
        if col in df.columns:
            nonnull = pd.to_numeric(df[col], errors="coerce").notna().sum()
            if nonnull < len(df) * MIN_NUMERIC_RATIO:
                print(f"[WARN] {col}: only {nonnull}/{len(df)} numeric values")

    if ok:
        print("[OK]   results_log.csv passed validation")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build backtest performance CSVs")
    parser.add_argument("--section", type=str, default=None,
                        choices=["summary", "models", "weekly", "conference",
                                 "calibration", "edge", "matrix", "grade-only"],
                        help="Only build a specific section")
    parser.add_argument("--since", type=str, default=None,
                        help="Filter to games after this date (YYYY-MM-DD)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print, don't write files")
    parser.add_argument("--validate", action="store_true",
                        help="Only check results_log integrity")
    args = parser.parse_args()

    # Load results log
    df = _load(RESULTS_LOG, "results_log.csv")
    if df is None:
        if args.section == "grade-only":
            print("[WARN] results_log.csv missing — writing empty graded log placeholder.")
            GRADED_LOG.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(GRADED_LOG, index=False)
            sys.exit(0)
        print("[ERROR] results_log.csv missing or empty — cannot build backtest CSVs.")
        sys.exit(1)

    print(f"[BACKTEST] Loading results_log.csv: {len(df):,} rows")

    alias_map = {
        "primary_ats_correct": "ats_correct",
        "primary_ou_correct": "ou_correct",
        "market_spread": "spread_line",
        "actual_margin": "actual_spread",
        "ens_ens_spread": "pred_spread",
    }
    for src, dst in alias_map.items():
        if src in df.columns and dst not in df.columns:
            df[dst] = df[src]

    # pred_spread fallback for backtester grading
    if "pred_spread" not in df.columns or df["pred_spread"].isna().all():
        for col in ["predicted_spread", "ens_ens_spread", "ensemble_spread"]:
            if col in df.columns and df[col].notna().any():
                df["pred_spread"] = df[col]
                break

    # Enrich missing spread_line / total_line from historical closing lines
    df = enrich_spread_lines_from_closing(df)

    # Validate only mode
    if args.validate:
        ok = validate_results_log(df)
        sys.exit(0 if ok else 1)

    # ── Grade all predictions ──────────────────────────────────────────
    df = grade_all(df)

    # Stats summary
    ats_gradeable = df[df["ats_result"].isin(["WIN", "LOSS"])]
    ou_gradeable  = df[df["ou_result"].isin(["WIN", "LOSS"])]
    ml_gradeable  = df[df["winner_correct"].notna()]

    ats_wins = int((ats_gradeable["ats_result"] == "WIN").sum())
    ats_losses = int((ats_gradeable["ats_result"] == "LOSS").sum())
    ats_pushes = int((df["ats_result"] == "PUSH").sum())
    ats_pct = ats_wins / max(1, len(ats_gradeable)) * 100

    ou_pct = float((ou_gradeable["ou_result"] == "WIN").mean() * 100) if len(ou_gradeable) else 0
    ml_pct = float(ml_gradeable["winner_correct"].mean() * 100) if len(ml_gradeable) else 0

    print(f"[BACKTEST] Graded: {len(ats_gradeable):,} ATS-gradeable, "
          f"{len(ou_gradeable):,} O/U-gradeable, {len(ml_gradeable):,} ML-gradeable")
    print(f"[BACKTEST] Season ATS: {ats_pct:.1f}% ({ats_wins}-{ats_losses}-{ats_pushes}) "
          f"| O/U: {ou_pct:.1f}% | ML: {ml_pct:.1f}%")

    # Write graded log
    _write(df, GRADED_LOG, "results_log_graded.csv", dry_run=args.dry_run)

    if not args.dry_run:
        try:
            from evaluation.walk_forward import walk_forward_validation

            wf = walk_forward_validation(df)
            wf.to_csv(DATA / "walk_forward_results.csv", index=False)
            if not wf.empty:
                print(f"[OK]   walk_forward_results.csv: {len(wf)} rows")
        except Exception as exc:
            print(f"[WARN] walk-forward validation skipped: {exc}")

        try:
            from evaluation.feature_audit import audit_feature_predictiveness

            if len(df) >= 100:
                feat = audit_feature_predictiveness(df)
                feat.to_csv(DATA / "feature_audit.csv", index=False)
                print(f"[OK]   feature_audit.csv: {len(feat)} rows")
        except Exception as exc:
            print(f"[WARN] feature audit skipped: {exc}")

    if args.section == "grade-only":
        return

    # ── Apply date filter ──────────────────────────────────────────────
    if args.since:
        df = _parse_date_col(df)
        cutoff = pd.Timestamp(args.since, tz="UTC")
        before = len(df)
        df = df[df["_date"] >= cutoff]
        print(f"[BACKTEST] --since {args.since}: {before} → {len(df)} rows")
        if df.empty:
            print("[WARN] No rows after date filter")
            return

    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # ── Section B: Summary ─────────────────────────────────────────────
    sections = args.section
    if sections is None or sections == "summary":
        summary = build_summary(df)
        _write(summary, CSV_DIR / "backtest_summary.csv",
               "backtest_summary.csv", dry_run=args.dry_run)

    # ── Section C: By Model ────────────────────────────────────────────
    if sections is None or sections == "models":
        by_model = build_by_model(df)
        if not by_model.empty:
            _write(by_model, CSV_DIR / "backtest_by_model.csv",
                   "backtest_by_model.csv", dry_run=args.dry_run)

    # ── Section D: Weekly ──────────────────────────────────────────────
    if sections is None or sections == "weekly":
        weekly = build_weekly(df)
        if not weekly.empty:
            _write(weekly, CSV_DIR / "backtest_weekly.csv",
                   "backtest_weekly.csv", dry_run=args.dry_run)

    # ── Section E: Conference ──────────────────────────────────────────
    if sections is None or sections == "conference":
        by_conf = build_by_conference(df)
        if not by_conf.empty:
            _write(by_conf, CSV_DIR / "backtest_by_conference.csv",
                   "backtest_by_conference.csv", dry_run=args.dry_run)

    # ── Section F: Calibration ─────────────────────────────────────────
    if sections is None or sections == "calibration":
        calib = build_calibration(df)
        if not calib.empty:
            _write(calib, CSV_DIR / "backtest_calibration.csv",
                   "backtest_calibration.csv", dry_run=args.dry_run)

    # ── Section G: Edge Tiers ──────────────────────────────────────────
    if sections is None or sections == "edge":
        by_edge = build_by_edge(df)
        if not by_edge.empty:
            _write(by_edge, CSV_DIR / "backtest_by_edge.csv",
                   "backtest_by_edge.csv", dry_run=args.dry_run)
            _write(by_edge, DATA / "edge_history.csv",
                   "edge_history.csv", dry_run=args.dry_run)

    # ── Section H: Model Matrix ────────────────────────────────────────
    if sections is None or sections == "matrix":
        matrix = build_model_matrix(df)
        if not matrix.empty:
            _write(matrix, CSV_DIR / "backtest_model_matrix.csv",
                   "backtest_model_matrix.csv", dry_run=args.dry_run)


if __name__ == "__main__":
    main()
