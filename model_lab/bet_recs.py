"""Canonical bet recommendation logic.

Single source of truth for edge thresholds and rec string generation.
Imported by both joint_models.py and build_derived_csvs.py.
"""

from __future__ import annotations

import numpy as np

# Edge thresholds — change here, takes effect everywhere
BET_REC_SPREAD_THRESHOLD = 1.5   # minimum |spread_edge| points to recommend
BET_REC_TOTAL_THRESHOLD = 2.0    # minimum |total_edge| points to recommend
BET_REC_ML_THRESHOLD = 0.05      # minimum ml_edge (probability) to recommend


def generate_bet_recs(
    spread_edge: float,
    total_edge: float,
    ml_edge_home: float = float("nan"),
) -> str:
    """Return a pipe-delimited string of actionable bet recommendations.

    Each token is one of:
      SPREAD:HOME+{edge}  — home covers, positive edge
      SPREAD:AWAY+{edge}  — away covers, positive edge
      TOTAL:OVER+{edge}   — over, positive edge
      TOTAL:UNDER+{edge}  — under, positive edge
      ML:HOME+{edge}      — home ML edge above threshold
      PASS                — no edge exceeds any threshold

    Args:
        spread_edge: pred_margin_final - closing_spread_line (home perspective, points)
        total_edge:  pred_total - closing_total_line (points)
        ml_edge_home: model_win_prob_home - market_win_prob_home (probability units)

    Returns:
        Pipe-delimited recommendation string, never empty, never None.
    """
    recs: list[str] = []

    se = float(spread_edge) if np.isfinite(float(spread_edge) if not isinstance(spread_edge, float) else spread_edge) else float("nan")
    te = float(total_edge) if np.isfinite(float(total_edge) if not isinstance(total_edge, float) else total_edge) else float("nan")
    me = float(ml_edge_home) if np.isfinite(float(ml_edge_home) if not isinstance(ml_edge_home, float) else ml_edge_home) else float("nan")

    import math
    if math.isfinite(se) and abs(se) >= BET_REC_SPREAD_THRESHOLD:
        side = "HOME" if se > 0 else "AWAY"
        recs.append(f"SPREAD:{side}+{abs(se):.1f}")

    if math.isfinite(te) and abs(te) >= BET_REC_TOTAL_THRESHOLD:
        side = "OVER" if te > 0 else "UNDER"
        recs.append(f"TOTAL:{side}+{abs(te):.1f}")

    if math.isfinite(me) and me >= BET_REC_ML_THRESHOLD:
        recs.append(f"ML:HOME+{me:.3f}")

    return "|".join(recs) if recs else "PASS"
