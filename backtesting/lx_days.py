#!/usr/bin/env python3
"""
backtesting/lx_days.py
=======================
Rest flag derivation for matchup-level data.

All flags are already computed in compute_metrics.py build_matchup_metrics().
This module provides the canonical definitions and a standalone builder
for re-computing flags if needed.
"""

from __future__ import annotations
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# Threshold definitions
# ─────────────────────────────────────────────────────────────────────────────

SHORT_REST_DAYS    = 4   # <= N days = short rest (fatigue risk)
EXTENDED_REST_DAYS = 7   # >= N days = extended rest (rhythm risk)


def build_rest_flags(matchup: pd.DataFrame) -> pd.DataFrame:
    """
    (Re)compute all rest flags from rest_days_A and rest_days_B columns.
    No-ops gracefully if columns are missing.
    """
    m = matchup.copy()

    if "rest_days_A" not in m.columns or "rest_days_B" not in m.columns:
        return m

    ra = m["rest_days_A"]
    rb = m["rest_days_B"]

    m["short_rest_A"]    = (ra <= SHORT_REST_DAYS).astype(int)
    m["extended_rest_A"] = (ra >= EXTENDED_REST_DAYS).astype(int)
    m["short_rest_B"]    = (rb <= SHORT_REST_DAYS).astype(int)
    m["extended_rest_B"] = (rb >= EXTENDED_REST_DAYS).astype(int)

    # rest_advantage: A fully rested, B fatigued → favors A
    m["rest_advantage"]    = (m["extended_rest_A"].astype(bool) & m["short_rest_B"].astype(bool)).astype(int)
    # rest_disadvantage: A fatigued, B rested → direction 'B' (signal fires for B = bad for A)
    m["rest_disadvantage"] = (m["short_rest_A"].astype(bool) & m["extended_rest_B"].astype(bool)).astype(int)

    return m


def rest_flag_summary(matchup: pd.DataFrame) -> None:
    """Print fire rates for all rest flags."""
    flags = [
        "rest_advantage", "rest_disadvantage",
        "short_rest_A", "short_rest_B",
        "extended_rest_A", "extended_rest_B",
    ]
    n = len(matchup)
    print("[rest flags]")
    for f in flags:
        if f in matchup.columns:
            rate = matchup[f].sum() / n
            print(f"  {f}: {matchup[f].sum()} fires ({rate:.1%})")
