#!/usr/bin/env python3
"""
backtesting/thresholds.py
==========================
Median threshold computation (Pass 1) and full grid definition (Pass 2).
"""

from __future__ import annotations
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Full grid for Pass 2 refinement (55%+ pairs only)
# Keys = matchup-level _diff column names
# ─────────────────────────────────────────────────────────────────────────────

FULL_GRID: dict[str, list[float]] = {
    # ANE diff — net rating units (pts/100 poss)
    "ANE_season_diff": [-5, -2, 0, 2, 5],
    "ANE_L4_diff":     [-5, -2, 0, 2, 5],
    "ANE_L7_diff":     [-5, -2, 0, 2, 5],
    "ANE_L10_diff":    [-5, -2, 0, 2, 5],
    "ANE_L12_diff":    [-5, -2, 0, 2, 5],
    "ANE_trend_short_diff": [-3, -1, 0, 1, 3],
    "ANE_trend_med_diff":   [-3, -1, 0, 1, 3],

    # SVI diff — quality-weighted win rate units
    "SVI_season_diff": [-1, -0.5, 0, 0.5, 1],
    "SVI_L4_diff":     [-1, -0.5, 0, 0.5, 1],
    "SVI_L7_diff":     [-1, -0.5, 0, 0.5, 1],
    "SVI_L10_diff":    [-1, -0.5, 0, 0.5, 1],
    "SVI_L12_diff":    [-1, -0.5, 0, 0.5, 1],

    # PEQ diff — SOS-adjusted net rating units
    "PEQ_season_diff": [-5, -2, 0, 2, 5],
    "PEQ_L4_diff":     [-5, -2, 0, 2, 5],
    "PEQ_L7_diff":     [-5, -2, 0, 2, 5],
    "PEQ_L10_diff":    [-5, -2, 0, 2, 5],
    "PEQ_L12_diff":    [-5, -2, 0, 2, 5],

    # WL diff — win rate difference (0–1 scale)
    "WL_season_diff": [-0.2, -0.1, 0, 0.1, 0.2],
    "WL_L4_diff":     [-0.5, -0.25, 0, 0.25, 0.5],
    "WL_L7_diff":     [-0.5, -0.25, 0, 0.25, 0.5],
    "WL_L10_diff":    [-0.4, -0.2, 0, 0.2, 0.4],
    "WL_L12_diff":    [-0.4, -0.2, 0, 0.2, 0.4],

    # DPC diff — defensive efficiency units (inverted drtg)
    "DPC_season_diff": [-2, -1, 0, 1, 2],
    "DPC_L4_diff":     [-3, -1, 0, 1, 3],
    "DPC_L7_diff":     [-3, -1, 0, 1, 3],
    "DPC_L10_diff":    [-2, -1, 0, 1, 2],
    "DPC_L12_diff":    [-2, -1, 0, 1, 2],
    "DPC_trend_diff":  [-1, -0.5, 0, 0.5, 1],
    "DPC_L10_std":     [1, 2, 5],   # magnitude — lower std = more consistent
    "DPC_season_sum":  [0, 5, 10],
    "DPC_L10_sum":     [0, 5, 10],

    # FFC diff — four factors composite (centered percentage point units)
    "FFC_season_diff": [-3, -1, 0, 1, 3],
    "FFC_L4_diff":     [-5, -2, 0, 2, 5],
    "FFC_L7_diff":     [-5, -2, 0, 2, 5],
    "FFC_L10_diff":    [-3, -1, 0, 1, 3],
    "FFC_L12_diff":    [-3, -1, 0, 1, 3],

    # PXP diff — win vs pythagorean expectation
    "PXP_season_diff": [-0.1, -0.05, 0, 0.05, 0.1],
    "PXP_L4_diff":     [-0.5, -0.25, 0, 0.25, 0.5],
    "PXP_L7_diff":     [-0.5, -0.25, 0, 0.25, 0.5],
    "PXP_L10_diff":    [-0.3, -0.15, 0, 0.15, 0.3],
    "PXP_L12_diff":    [-0.3, -0.15, 0, 0.15, 0.3],

    # ODI diff — opponent-adj differential (perf_vs_exp_net units)
    "ODI_season_diff": [-10, -3, 0, 3, 10],
    "ODI_L4_diff":     [-15, -5, 0, 5, 15],
    "ODI_L7_diff":     [-15, -5, 0, 5, 15],
    "ODI_L10_diff":    [-10, -3, 0, 3, 10],
    "ODI_L12_diff":    [-10, -3, 0, 3, 10],
    "ODI_season_sum":  [0, 10, 20],
    "ODI_L10_sum":     [0, 10, 20],

    # Four-factor ODIs (percentage point units)
    "eFG_ODI":         [-2, 0, 2],
    "TO_ODI":          [-2, 0, 2],
    "ORB_ODI":         [-2, 0, 2],
    "FTR_ODI":         [-5, 0, 5],
    "eFG_ODI_diff":    [-2, 0, 2],
    "TO_ODI_diff":     [-2, 0, 2],
    "ORB_ODI_diff":    [-2, 0, 2],
    "FTR_ODI_diff":    [-5, 0, 5],

    # MTI — matchup tension (magnitude, game-level)
    "MTI": [2, 5, 10],

    # SCI — signal confidence (cover rate deviation)
    "SCI":   [0.02, 0.05, 0.10],
    "SCI_A": [0.02, 0.05, 0.10],

    # Rest flags — boolean (threshold always 0.5, i.e. fires when =1)
    "rest_advantage":    [0.5],
    "rest_disadvantage": [0.5],
    "short_rest_A":      [0.5],
    "short_rest_B":      [0.5],
    "extended_rest_A":   [0.5],
    "extended_rest_B":   [0.5],
}


def build_median_thresholds(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute the season median of each signal column in the matchup dataframe.
    Boolean/flag columns (0/1) default to threshold 0.5.
    """
    bool_flags = {
        "rest_advantage", "rest_disadvantage",
        "short_rest_A", "short_rest_B",
        "extended_rest_A", "extended_rest_B",
    }
    thresholds: dict[str, float] = {}

    for col in df.columns:
        if col in bool_flags:
            thresholds[col] = 0.5
        elif col in FULL_GRID or any(
            col.endswith(sfx) for sfx in ("_diff", "_sum")
        ):
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                if pd.notna(median_val):
                    thresholds[col] = float(median_val)
                else:
                    thresholds[col] = 0.0
        elif col in ("MTI", "SCI", "SCI_A", "SCI_B"):
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                thresholds[col] = float(median_val) if pd.notna(median_val) else 0.0

    return thresholds
