#!/usr/bin/env python3
"""
backtesting/direction_map.py
=============================
SIGNAL_DIRECTION registry and signal_fires() function.

Direction codes:
  'A'  — higher value = Team A (home) wins
  'B'  — lower  value = Team B (away) wins (i.e., high value = bad for A)
  'X'  — magnitude filter only; no directional prediction
"""

from __future__ import annotations
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Signal Direction Map (explicit — do not infer)
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_DIRECTION: dict[str, str] = {
    # ANE — Adjusted Net Efficiency (rolling windows)
    "ANE_season":      "A",
    "ANE_L4":          "A",
    "ANE_L7":          "A",
    "ANE_L10":         "A",
    "ANE_L12":         "A",
    "ANE_trend_short": "A",
    "ANE_trend_med":   "A",
    # ANE diff (matchup-level)
    "ANE_season_diff": "A",
    "ANE_L4_diff":     "A",
    "ANE_L7_diff":     "A",
    "ANE_L10_diff":    "A",
    "ANE_L12_diff":    "A",
    "ANE_trend_short_diff": "A",
    "ANE_trend_med_diff":   "A",

    # SVI — Schedule-adjusted Victory Index
    "SVI_season":      "A",
    "SVI_L4":          "A",
    "SVI_L7":          "A",
    "SVI_L10":         "A",
    "SVI_L12":         "A",
    "SVI_season_diff": "A",
    "SVI_L4_diff":     "A",
    "SVI_L7_diff":     "A",
    "SVI_L10_diff":    "A",
    "SVI_L12_diff":    "A",

    # PEQ — Performance Efficiency Quotient
    "PEQ_season":      "A",
    "PEQ_L4":          "A",
    "PEQ_L7":          "A",
    "PEQ_L10":         "A",
    "PEQ_L12":         "A",
    "PEQ_season_diff": "A",
    "PEQ_L4_diff":     "A",
    "PEQ_L7_diff":     "A",
    "PEQ_L10_diff":    "A",
    "PEQ_L12_diff":    "A",

    # WL — Win-Loss Rate
    "WL_season":       "A",
    "WL_L4":           "A",
    "WL_L7":           "A",
    "WL_L10":          "A",
    "WL_L12":          "A",
    "WL_season_diff":  "A",
    "WL_L4_diff":      "A",
    "WL_L7_diff":      "A",
    "WL_L10_diff":     "A",
    "WL_L12_diff":     "A",

    # DPC — Defensive Performance Composite
    "DPC_season":      "A",
    "DPC_L4":          "A",
    "DPC_L7":          "A",
    "DPC_L10":         "A",
    "DPC_L12":         "A",
    "DPC_trend":       "A",
    "DPC_L10_std":     "X",   # variability only
    "DPC_season_diff": "A",
    "DPC_L4_diff":     "A",
    "DPC_L7_diff":     "A",
    "DPC_L10_diff":    "A",
    "DPC_L12_diff":    "A",
    "DPC_trend_diff":  "A",
    "DPC_season_sum":  "X",
    "DPC_L10_sum":     "X",

    # FFC — Four Factors Composite
    "FFC_season":      "A",
    "FFC_L4":          "A",
    "FFC_L7":          "A",
    "FFC_L10":         "A",
    "FFC_L12":         "A",
    "FFC_season_diff": "A",
    "FFC_L4_diff":     "A",
    "FFC_L7_diff":     "A",
    "FFC_L10_diff":    "A",
    "FFC_L12_diff":    "A",

    # PXP — Performance vs Expected
    "PXP_season":      "A",
    "PXP_L4":          "A",
    "PXP_L7":          "A",
    "PXP_L10":         "A",
    "PXP_L12":         "A",
    "PXP_season_diff": "A",
    "PXP_L4_diff":     "A",
    "PXP_L7_diff":     "A",
    "PXP_L10_diff":    "A",
    "PXP_L12_diff":    "A",

    # ODI — Opponent-adjusted Differential Index
    "ODI_season":      "A",
    "ODI_L4":          "A",
    "ODI_L7":          "A",
    "ODI_L10":         "A",
    "ODI_L12":         "A",
    "ODI_season_diff": "A",
    "ODI_L4_diff":     "A",
    "ODI_L7_diff":     "A",
    "ODI_L10_diff":    "A",
    "ODI_L12_diff":    "A",
    "ODI_season_sum":  "X",
    "ODI_L10_sum":     "X",

    # Four-factor ODIs (opponent-adjusted components)
    "eFG_ODI":         "A",
    "TO_ODI":          "A",
    "ORB_ODI":         "A",
    "FTR_ODI":         "A",
    "eFG_ODI_L10":     "A",
    "TO_ODI_L10":      "A",
    "ORB_ODI_L10":     "A",
    "FTR_ODI_L10":     "A",
    "eFG_ODI_diff":    "A",
    "TO_ODI_diff":     "A",
    "ORB_ODI_diff":    "A",
    "FTR_ODI_diff":    "A",

    # MTI — Matchup Tension Index (magnitude filter, game-level)
    "MTI":             "X",

    # SCI — Signal Confidence Index (magnitude filter)
    "SCI":             "X",
    "SCI_A":           "X",
    "SCI_B":           "X",

    # Rest flags (from lx_days.py)
    "rest_advantage":    "A",   # home rested, away fatigued → favors A
    "rest_disadvantage": "B",   # home fatigued, away rested → favors B
    "short_rest_A":      "B",   # bad for A
    "short_rest_B":      "A",   # bad for B = good for A
    "extended_rest_A":   "A",   # good for A
    "extended_rest_B":   "B",   # good for B = contextual for A
}


def signal_fires(
    signal_name: str,
    row: "pd.Series",
    threshold_dict: dict[str, float],
) -> bool:
    """
    Returns True if the signal fires for Team A winning.

    Direction A: fires when value > threshold
    Direction B: fires when value < threshold  (high value = bad for A)
    Direction X: fires when abs(value) > threshold (magnitude filter)
    """
    if signal_name not in SIGNAL_DIRECTION:
        raise KeyError(f"Signal '{signal_name}' not in SIGNAL_DIRECTION registry")

    direction = SIGNAL_DIRECTION[signal_name]
    val = row.get(signal_name, None)
    if val is None or (hasattr(val, "__float__") and pd.isna(float(val))):
        return False

    thresh = threshold_dict.get(signal_name, 0.0)

    if direction == "A":
        return float(val) > thresh
    elif direction == "B":
        return float(val) < thresh
    elif direction == "X":
        return abs(float(val)) > thresh
    return False


def combo_fires(
    combo: tuple[str, ...],
    row: "pd.Series",
    threshold_dict: dict[str, float],
) -> bool:
    """Returns True when ALL signals in the combo fire."""
    return all(signal_fires(sig, row, threshold_dict) for sig in combo)


def validate_registry(df_columns: list[str]) -> dict[str, list[str]]:
    """
    Check which registered signals are present/missing in a dataframe.
    Returns {'present': [...], 'missing': [...]}.
    """
    present = [s for s in SIGNAL_DIRECTION if s in df_columns]
    missing = [s for s in SIGNAL_DIRECTION if s not in df_columns]
    return {"present": present, "missing": missing}
