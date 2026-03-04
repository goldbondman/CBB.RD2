#!/usr/bin/env python3
"""
backtesting/combos.py
======================
Combo builder, collinearity filter, and trio pruning.

Rules:
- Collinearity: skip any combo where 2+ signals share the same base metric
  AND the same Lx window. ANE_L4_diff + ANE_L10_diff = OK. ANE_L10_diff x2 = illegal.
- Trio pool: built only from pairs that achieved >= 55% SU hit rate.
- Quads: hardcoded priority list only (no brute force).
"""

from __future__ import annotations
import re
from itertools import combinations
from typing import Iterator


# ─────────────────────────────────────────────────────────────────────────────
# Base signal pool (matchup-level columns to test as pair signals)
# ─────────────────────────────────────────────────────────────────────────────

BASE_METRICS = [
    "ANE_season_diff", "ANE_L4_diff", "ANE_L7_diff", "ANE_L10_diff", "ANE_L12_diff",
    "ANE_trend_short_diff", "ANE_trend_med_diff",
    "SVI_season_diff", "SVI_L4_diff", "SVI_L7_diff", "SVI_L10_diff", "SVI_L12_diff",
    "PEQ_season_diff", "PEQ_L4_diff", "PEQ_L7_diff", "PEQ_L10_diff", "PEQ_L12_diff",
    "WL_season_diff",  "WL_L4_diff",  "WL_L7_diff",  "WL_L10_diff",  "WL_L12_diff",
    "DPC_season_diff", "DPC_L4_diff", "DPC_L7_diff", "DPC_L10_diff", "DPC_L12_diff",
    "DPC_trend_diff",  "DPC_L10_std",
    "FFC_season_diff", "FFC_L4_diff", "FFC_L7_diff", "FFC_L10_diff", "FFC_L12_diff",
    "PXP_season_diff", "PXP_L4_diff", "PXP_L7_diff", "PXP_L10_diff", "PXP_L12_diff",
    "ODI_season_diff", "ODI_L4_diff", "ODI_L7_diff", "ODI_L10_diff", "ODI_L12_diff",
    "eFG_ODI",  "TO_ODI",  "ORB_ODI",  "FTR_ODI",
    "MTI", "SCI",
    "rest_advantage", "rest_disadvantage",
    "short_rest_A",   "short_rest_B",
    "extended_rest_A","extended_rest_B",
]

# Non-directional sum features — may only appear in trios/quads WITH a _diff signal
SUM_ONLY_METRICS = {"DPC_season_sum", "DPC_L10_sum", "ODI_season_sum", "ODI_L10_sum"}

# ─────────────────────────────────────────────────────────────────────────────
# Priority Lx smoke-test combos (Step 3 in the plan)
# ─────────────────────────────────────────────────────────────────────────────

SMOKE_TEST_COMBOS: list[tuple[str, ...]] = [
    # Pairs
    ("rest_advantage", "ANE_L4_diff"),
    ("rest_advantage", "DPC_L10_diff"),
    ("rest_advantage", "ODI_L7_diff"),
    ("rest_advantage", "PXP_season_diff"),
    ("short_rest_B",   "FFC_L7_diff"),
    ("extended_rest_A","SVI_L4_diff"),
    ("extended_rest_A","WL_L7_diff"),
    ("extended_rest_A","ANE_L4_diff"),
    # Trios
    ("rest_advantage", "ANE_L10_diff", "DPC_L10_diff"),
    ("rest_advantage", "ODI_season_diff", "SVI_season_diff"),
    ("short_rest_B",   "FFC_L10_diff", "PXP_season_diff"),
    ("extended_rest_A","ANE_L4_diff", "eFG_ODI"),
    ("rest_advantage", "WL_season_diff", "ORB_ODI"),
    ("short_rest_A",   "MTI", "PEQ_season_diff"),
    ("rest_advantage", "DPC_trend_diff", "ANE_season_diff"),
    # Quads (priority list)
    ("rest_advantage", "ANE_L10_diff", "DPC_season_diff", "ODI_season_diff"),
    ("short_rest_B",   "FFC_L7_diff", "PXP_season_diff", "WL_season_diff"),
    ("rest_advantage", "SVI_L4_diff", "eFG_ODI", "ORB_ODI"),
    ("extended_rest_A","ANE_L4_diff", "DPC_L10_diff", "PEQ_season_diff"),
    ("rest_advantage", "ODI_season_diff", "SCI", "MTI"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Collinearity detection
# ─────────────────────────────────────────────────────────────────────────────

def _parse_signal(signal: str) -> tuple[str, str | None]:
    """
    Parse a signal name into (base_metric, lx_window).
    Examples:
      'ANE_L4_diff' → ('ANE', 'L4')
      'DPC_season_diff' → ('DPC', 'season')
      'rest_advantage' → ('rest_advantage', None)
      'eFG_ODI' → ('eFG_ODI', None)
      'MTI' → ('MTI', None)
    """
    # Match base_metric_Lx or base_metric_season
    m = re.match(r"^([A-Z_a-z]+?)_(L\d+|season)(_diff|_sum)?$", signal)
    if m:
        return m.group(1), m.group(2)
    return signal, None


def has_collinear_signals(combo: tuple[str, ...]) -> bool:
    """
    Returns True if 2+ signals share the same (base_metric, lx_window).
    Same base metric + different Lx = allowed.
    Same base metric + same Lx = collinear → skip.
    """
    seen: set[tuple[str, str]] = set()
    for sig in combo:
        base, lx = _parse_signal(sig)
        if lx is None:
            continue  # no-Lx signals can't collinear with each other this way
        key = (base, lx)
        if key in seen:
            return True
        seen.add(key)
    return False


def has_sum_without_diff(combo: tuple[str, ...]) -> bool:
    """
    Returns True if any _sum metric appears without any _diff metric in the combo.
    Sum metrics may only appear with at least one _diff signal.
    """
    has_sum = any(s in SUM_ONLY_METRICS for s in combo)
    has_diff = any("_diff" in s for s in combo)
    return has_sum and not has_diff


# ─────────────────────────────────────────────────────────────────────────────
# Combo generators
# ─────────────────────────────────────────────────────────────────────────────

def available_signals(df_columns: list[str]) -> list[str]:
    """Return BASE_METRICS that actually exist in the dataframe."""
    return [s for s in BASE_METRICS if s in df_columns]


def generate_pairs(
    signal_pool: list[str],
    df_columns: list[str],
) -> Iterator[tuple[str, str]]:
    """
    Generate all valid pairs from signal_pool.
    Filters: collinearity, sum-without-diff.
    """
    pool = [s for s in signal_pool if s in df_columns]
    for combo in combinations(pool, 2):
        if has_collinear_signals(combo):
            continue
        if has_sum_without_diff(combo):
            continue
        yield combo


def generate_trios_from_pairs(
    qualifying_pairs: list[dict],
    df_columns: list[str],
) -> Iterator[tuple[str, str, str]]:
    """
    Build trios only from signals that appeared in >= 55% SU qualifying pairs.
    """
    qualifying_signals: set[str] = set()
    for pair in qualifying_pairs:
        qualifying_signals.update(pair["combo"])

    pool = sorted(s for s in qualifying_signals if s in df_columns)

    for combo in combinations(pool, 3):
        if has_collinear_signals(combo):
            continue
        if has_sum_without_diff(combo):
            continue
        yield combo


def count_combos(signal_pool: list[str], df_columns: list[str]) -> dict[str, int]:
    """Count expected combos per type (for reporting)."""
    pool = [s for s in signal_pool if s in df_columns]
    pairs = sum(
        1 for c in combinations(pool, 2)
        if not has_collinear_signals(c) and not has_sum_without_diff(c)
    )
    return {"pairs_estimated": pairs}
