#!/usr/bin/env python3
"""
backtesting/stale.py
=====================
Stale signal detection and re-validation.

On every future run:
1. Load user_signals.csv where stale == False
2. Re-validate each signal against latest data
3. If holdout SU < 0.52 → set stale = True, log reason
4. Print stale report before running new combos
5. Never delete stale rows — preserve full history
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

ROOT = pathlib.Path(".")
USER_SIGNALS_PATH = ROOT / "user_signals.csv"
STALE_HOLDOUT_THRESH = 0.52


def load_active_signals(path: pathlib.Path = USER_SIGNALS_PATH) -> pd.DataFrame:
    """Load non-stale signals from user_signals.csv."""
    if not path.exists() or path.stat().st_size < 10:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "stale" not in df.columns:
        return pd.DataFrame()
    return df[df["stale"] == False].copy()  # noqa: E712


def revalidate_signal(
    signal_row: pd.Series,
    matchup_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Re-test a single signal against the latest matchup data.
    Returns updated fields.
    """
    from .direction_map import SIGNAL_DIRECTION

    try:
        combo = tuple(signal_row["signals"].split("|"))
        thresholds = json.loads(signal_row.get("thresholds", "{}"))
    except Exception:
        return {"stale": True, "stale_reason": "parse_error"}

    n_total = len(matchup_df)
    if n_total == 0:
        return {}

    fires_mask = pd.Series(True, index=matchup_df.index)
    for sig in combo:
        if sig not in matchup_df.columns:
            return {"stale": True, "stale_reason": f"missing_col:{sig}"}
        direction = SIGNAL_DIRECTION.get(sig, "A")
        thresh = thresholds.get(sig, 0.0)
        val_series = matchup_df[sig]
        null_mask = val_series.isna()
        if direction == "A":
            fires_mask &= (val_series > thresh) & ~null_mask
        elif direction == "B":
            fires_mask &= (val_series < thresh) & ~null_mask
        elif direction == "X":
            fires_mask &= (val_series.abs() > thresh) & ~null_mask

    n_fires = fires_mask.sum()
    if n_fires < 10:
        return {}  # not enough data for re-validation yet

    fired = matchup_df[fires_mask]
    new_holdout_su = fired["home_win"].mean()

    old_holdout_su = signal_row.get("holdout_su")
    if pd.notna(new_holdout_su) and new_holdout_su < STALE_HOLDOUT_THRESH:
        reason = f"holdout decay: {old_holdout_su:.3f} → {new_holdout_su:.3f}"
        return {
            "stale": True,
            "stale_reason": reason,
            "holdout_su": round(float(new_holdout_su), 4),
        }

    return {
        "holdout_su": round(float(new_holdout_su), 4),
    }


def check_stale_signals(
    matchup_df: pd.DataFrame,
    path: pathlib.Path = USER_SIGNALS_PATH,
) -> int:
    """
    Re-validate all active signals. Mark stale ones.
    Returns count of newly stale signals.
    """
    if not path.exists() or path.stat().st_size < 10:
        print("[stale] No user_signals.csv found — skipping stale check")
        return 0

    df = pd.read_csv(path)
    if "stale" not in df.columns:
        print("[stale] 'stale' column missing — skipping")
        return 0

    active = df["stale"] == False  # noqa: E712
    n_active = active.sum()
    if n_active == 0:
        print("[stale] No active signals to re-validate")
        return 0

    print(f"[stale] Re-validating {n_active} active signals...")
    n_newly_stale = 0

    for idx in df[active].index:
        row = df.loc[idx]
        updates = revalidate_signal(row, matchup_df)
        if not updates:
            continue
        for col, val in updates.items():
            df.at[idx, col] = val
        if updates.get("stale"):
            n_newly_stale += 1
            print(f"  STALE: {row.get('signals', '')} — {updates.get('stale_reason', '')}")

    df.to_csv(path, index=False)
    print(f"[stale] {n_newly_stale} signals newly marked stale | {n_active - n_newly_stale} still active")
    return n_newly_stale
