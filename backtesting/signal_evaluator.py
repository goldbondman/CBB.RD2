#!/usr/bin/env python3
"""
backtesting/signal_evaluator.py
================================
Evaluate backtested signals against current team metrics to produce
per-game signal columns for the prediction pipeline.

Called by apply_signals.py after predictions_combined_latest.csv is built.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from .direction_map import SIGNAL_DIRECTION

ROOT = pathlib.Path(".")
DATA = pathlib.Path("data")

# Metrics for which we compute home - away diff
DIFF_METRICS = [
    "ANE_season", "ANE_L4", "ANE_L7", "ANE_L10", "ANE_L12",
    "ANE_trend_short", "ANE_trend_med",
    "SVI_season", "SVI_L4", "SVI_L7", "SVI_L10", "SVI_L12",
    "PEQ_season", "PEQ_L4", "PEQ_L7", "PEQ_L10", "PEQ_L12",
    "WL_season",  "WL_L4",  "WL_L7",  "WL_L10",  "WL_L12",
    "DPC_season", "DPC_L4", "DPC_L7", "DPC_L10", "DPC_L12", "DPC_trend",
    "FFC_season", "FFC_L4", "FFC_L7", "FFC_L10", "FFC_L12",
    "PXP_season", "PXP_L4", "PXP_L7", "PXP_L10", "PXP_L12",
    "ODI_season", "ODI_L4", "ODI_L7", "ODI_L10", "ODI_L12",
    "eFG_ODI", "TO_ODI", "ORB_ODI", "FTR_ODI",
    "SCI",
]

# Metrics for which we compute home + away sum
SUM_METRICS = ["DPC_season", "DPC_L10", "ODI_season", "ODI_L10"]


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_active_signals(
    path: pathlib.Path = ROOT / "user_signals.csv",
) -> pd.DataFrame:
    """Load non-stale signals sorted by holdout SU descending."""
    if not path.exists() or path.stat().st_size < 10:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "stale" not in df.columns:
        return pd.DataFrame()
    active = df[df["stale"] == False].copy()  # noqa: E712
    return active.sort_values("holdout_su", ascending=False).reset_index(drop=True)


def load_team_latest_metrics(
    adv_path: pathlib.Path = DATA / "team_game_metrics_advanced.csv",
) -> pd.DataFrame:
    """
    Load team_game_metrics_advanced.csv and return one row per team
    (the most recent game), indexed by team_id.
    """
    df = pd.read_csv(adv_path, low_memory=False)
    df["game_dt"] = pd.to_datetime(df["game_dt"], utc=True, errors="coerce")
    latest = df.sort_values("game_dt").groupby("team_id").last().reset_index()
    return latest.set_index("team_id")


# ─────────────────────────────────────────────────────────────────────────────
# Matchup row builder
# ─────────────────────────────────────────────────────────────────────────────

def build_matchup_row(
    home_row: pd.Series,
    away_row: pd.Series,
) -> dict[str, Any]:
    """
    Given the latest per-team metric rows for home and away,
    compute the matchup-level _diff and _sum columns that signals reference.
    """
    row: dict[str, Any] = {}

    for metric in DIFF_METRICS:
        h = home_row.get(metric, np.nan)
        a = away_row.get(metric, np.nan)
        if pd.notna(h) and pd.notna(a):
            row[f"{metric}_diff"] = float(h) - float(a)
        else:
            row[f"{metric}_diff"] = np.nan

    for metric in SUM_METRICS:
        h = home_row.get(metric, np.nan)
        a = away_row.get(metric, np.nan)
        if pd.notna(h) and pd.notna(a):
            row[f"{metric}_sum"] = float(h) + float(a)
        else:
            row[f"{metric}_sum"] = np.nan

    # MTI: matchup tension = avg |adj_net_rtg| both teams
    h_anr = home_row.get("adj_net_rtg", np.nan)
    a_anr = away_row.get("adj_net_rtg", np.nan)
    if pd.notna(h_anr) and pd.notna(a_anr):
        row["MTI"] = (abs(float(h_anr)) + abs(float(a_anr))) / 2.0
    else:
        row["MTI"] = np.nan

    # SCI: use home team's value as primary (per matchup builder convention)
    row["SCI"]   = home_row.get("SCI", np.nan)
    row["SCI_A"] = home_row.get("SCI", np.nan)
    row["SCI_B"] = away_row.get("SCI", np.nan)

    # Rest flags (not available at prediction time without schedule data — skip)

    return row


# ─────────────────────────────────────────────────────────────────────────────
# Signal evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _signal_fires(sig: str, val: float, thresh: float) -> bool:
    direction = SIGNAL_DIRECTION.get(sig, "A")
    if pd.isna(val):
        return False
    if direction == "A":
        return float(val) > float(thresh)
    elif direction == "B":
        return float(val) < float(thresh)
    elif direction == "X":
        return abs(float(val)) > float(thresh)
    return False


def evaluate_signals_for_matchup(
    matchup_row: dict[str, Any],
    active_signals: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Evaluate all active signals against a pre-built matchup row dict.
    Returns a summary dict of fired signals.
    """
    if active_signals.empty:
        return {
            "signal_n_fired": 0,
            "signal_best_su": None,
            "signal_best_holdout_su": None,
            "signal_best_combo": None,
            "signal_top_combos": None,
        }

    fired = []
    for _, sig_row in active_signals.iterrows():
        try:
            signals = sig_row["signals"].split("|")
            thresholds = json.loads(sig_row.get("thresholds", "{}"))
        except Exception:
            continue

        all_fire = True
        for sig in signals:
            val = matchup_row.get(sig, np.nan)
            thresh = thresholds.get(sig, 0.0)
            if not _signal_fires(sig, float(val) if pd.notna(val) else np.nan, thresh):
                all_fire = False
                break

        if all_fire:
            fired.append({
                "combo":        sig_row["signals"],
                "hit_rate_su":  sig_row.get("hit_rate_su"),
                "holdout_su":   sig_row.get("holdout_su"),
                "n":            sig_row.get("n"),
                "fire_rate":    sig_row.get("fire_rate"),
            })

    if not fired:
        return {
            "signal_n_fired": 0,
            "signal_best_su": None,
            "signal_best_holdout_su": None,
            "signal_best_combo": None,
            "signal_top_combos": None,
        }

    fired.sort(key=lambda x: x.get("holdout_su") or 0, reverse=True)
    best = fired[0]

    return {
        "signal_n_fired":        len(fired),
        "signal_best_su":        round(float(best["hit_rate_su"]), 4) if best["hit_rate_su"] else None,
        "signal_best_holdout_su": round(float(best["holdout_su"]), 4) if best["holdout_su"] else None,
        "signal_best_combo":     best["combo"],
        "signal_top_combos":     "; ".join(f["combo"] for f in fired[:top_n]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main overlay runner
# ─────────────────────────────────────────────────────────────────────────────

def run_signal_overlay(
    predictions_path: pathlib.Path,
    adv_path: pathlib.Path = DATA / "team_game_metrics_advanced.csv",
    signals_path: pathlib.Path = ROOT / "user_signals.csv",
    output_path: pathlib.Path | None = None,
) -> pd.DataFrame:
    """
    Load predictions, evaluate signals per game, return augmented DataFrame.
    Appended columns:
        signal_n_fired          — count of signals that fired
        signal_best_su          — in-sample SU of best fired signal
        signal_best_holdout_su  — holdout SU of best fired signal
        signal_best_combo       — signal names of best combo (pipe-separated)
        signal_top_combos       — top 5 fired combos (semicolon-separated)
        signal_agrees_model     — True if signal(s) agree with model's pick
    """
    print("[signals] Loading predictions...")
    preds = pd.read_csv(predictions_path)
    print(f"[signals] {len(preds)} games to evaluate")

    print("[signals] Loading team latest metrics...")
    team_metrics = load_team_latest_metrics(adv_path)

    print("[signals] Loading active signals...")
    active = load_active_signals(signals_path)
    print(f"[signals] {len(active)} active signals loaded")

    # Initialise new columns
    preds["signal_n_fired"]          = 0
    preds["signal_best_su"]          = None
    preds["signal_best_holdout_su"]  = None
    preds["signal_best_combo"]       = None
    preds["signal_top_combos"]       = None
    preds["signal_agrees_model"]     = None

    n_matched = 0
    for idx, game in preds.iterrows():
        htid = game.get("home_team_id")
        atid = game.get("away_team_id")

        if htid not in team_metrics.index or atid not in team_metrics.index:
            continue

        home_row = team_metrics.loc[htid]
        away_row = team_metrics.loc[atid]

        matchup = build_matchup_row(home_row, away_row)
        result  = evaluate_signals_for_matchup(matchup, active)

        for col, val in result.items():
            preds.at[idx, col] = val

        if result["signal_n_fired"] > 0:
            n_matched += 1
            # Signals fire when home team has the advantage — compare to model
            model_favors_home = float(game.get("home_win_prob", 0.5) or 0.5) > 0.5
            preds.at[idx, "signal_agrees_model"] = bool(model_favors_home)

    n_fired = int((preds["signal_n_fired"] > 0).sum())
    print(f"[signals] Games with >= 1 signal fired: {n_fired} / {len(preds)}")
    if n_fired:
        avg = preds.loc[preds["signal_n_fired"] > 0, "signal_best_holdout_su"].mean()
        print(f"[signals] Avg best holdout SU on fired games: {avg:.3f}")
        agree = preds.loc[preds["signal_n_fired"] > 0, "signal_agrees_model"].mean()
        print(f"[signals] Signal-model agreement rate: {agree:.1%}")

    if output_path:
        preds.to_csv(output_path, index=False)
        print(f"[signals] Saved to {output_path}")

    return preds
