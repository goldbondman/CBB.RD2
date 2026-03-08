"""
kelly.py — Shared fractional Kelly unit-sizing logic.

Unit scale: 0.5u (min) to 5u (max, reserved for elite spots <30/season).

Sizing is driven by:
  1. Model edge magnitude  (pred vs market, in points)
  2. CAGE alignment        (CONFIRMS / NEUTRAL / DIVERGES)
  3. Trend alignment       (True / False — trend direction matches model pick)

Kelly math (informational, used to set tier midpoints):
  At -110 juice: full_kelly = (1.909 * p - 1) / 0.909
  We use 25% fractional Kelly, mapped to the 0.5–5u discrete scale.

Tier table (before alignment adjustments):
  Edge < 2.0 pts  →  0u  (no bet, below model edge threshold)
  Edge 2.0–3.9    →  1u  base
  Edge 4.0–5.9    →  2u  base
  Edge 6.0–7.9    →  3u  base
  Edge 8.0–9.9    →  4u  base
  Edge ≥ 10.0     →  4u  base (cap before signal bonus)

Alignment adjustments (additive, rounded to nearest 0.5u, capped at 5u):
  CAGE CONFIRMS   → +0.5u
  CAGE DIVERGES   → −0.5u  (signal opposes pick, reduce exposure)
  Trend aligns    → +0.5u

5u elite-spot criteria (all required):
  • Edge ≥ 6.0 pts
  • CAGE CONFIRMS
  • Trend aligns
  Expected frequency: <30 games / season at current model calibration.
"""

from __future__ import annotations

__all__ = ["kelly_units", "kelly_fraction", "EDGE_MIN"]

EDGE_MIN = 2.0          # minimum model edge to place any bet
_JUICE   = 110.0        # standard juice


def kelly_fraction(p: float, juice: float = _JUICE) -> float:
    """Full Kelly fraction at -110 juice given win probability p."""
    b = 100.0 / juice          # net profit per unit risked
    return max(0.0, (b * p - (1.0 - p)) / b)


def kelly_units(
    edge: float,
    cage_validates: str = "NEUTRAL",
    trend_aligns: bool = False,
) -> float:
    """Return bet size in units (0.0 = no bet, 0.5–5.0 otherwise).

    Parameters
    ----------
    edge           : model edge vs market in points (absolute value expected by
                     caller; sign determines direction, this function uses abs)
    cage_validates : 'CONFIRMS', 'NEUTRAL', or 'DIVERGES'
    trend_aligns   : True when the net-rtg trend supports the model pick direction
    """
    edge = abs(float(edge))

    # No bet below threshold
    if edge < EDGE_MIN:
        return 0.0

    # Base tier
    if edge < 4.0:
        base = 1.0
    elif edge < 6.0:
        base = 2.0
    elif edge < 8.0:
        base = 3.0
    elif edge < 10.0:
        base = 4.0
    else:
        base = 4.0      # cap base; alignment bonus can push to 5u

    # Alignment adjustments
    bonus = 0.0
    if cage_validates == "CONFIRMS":
        bonus += 0.5
    elif cage_validates == "DIVERGES":
        bonus -= 0.5
    if trend_aligns:
        bonus += 0.5

    raw = base + bonus
    # Round to nearest 0.5u, clamp to [0.5, 5.0]
    units = round(raw * 2) / 2
    return float(max(0.5, min(5.0, units)))


def is_elite_spot(edge: float, cage_validates: str, trend_aligns: bool) -> bool:
    """True when all three criteria for a 5u elite spot are met."""
    return (
        abs(float(edge)) >= 6.0
        and cage_validates == "CONFIRMS"
        and trend_aligns
    )
