"""
Single source of truth for alpha detection, kelly sizing, and
edge classification. Called at two points in the pipeline:

  1. espn_prediction_runner.py  — at prediction time, with
     situational context (trap, revenge) but no market data yet.
     Run with market_context=None to get base alpha.

  2. predictions_with_context.py — after market data is merged.
     Re-run with full market_context to get final alpha.
     The second call OVERRIDES the first.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path("data")


def _as_bool(val) -> bool:
    """Parse bool-like CSV values safely (True/False/1/0/yes/no)."""
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return False
        return val != 0
    return str(val).strip().lower() in {"1", "true", "t", "yes", "y"}


def _as_side(val: Optional[str]) -> Optional[str]:
    """Normalize market side values to home/away when possible."""
    if val is None:
        return None
    side = str(val).strip().lower()
    if side in {"home", "away"}:
        return side
    return None


def kelly_fraction_calc(
    model_confidence: float,
    edge_pts: float = 0.0,
    juice: float = -110,
    multiplier: float = 1.0,
) -> float:
    """
    Quarter-Kelly bankroll fraction using edge-adjusted win probability.

    model_confidence: data reliability score (0–1), NOT win probability.
    edge_pts: |model_spread - market_spread| in points.

    Win probability is estimated from edge size, then scaled by
    model_confidence as a reliability discount.

    Edge → base win probability:
      0.0 pts → 50.0%  (no edge)
      1.0 pts → 51.5%
      2.0 pts → 53.0%
      3.0 pts → 54.5%
      4.0 pts → 56.0%
      5.0 pts → 57.0%
      6.0 pts → 58.0%
      7.0+ pts → 59.0% (cap)

    Scaled by model_confidence:
      Low confidence → regress win prob toward 50%
      High confidence → use full edge-implied prob
    """
    if juice < 0:
        decimal_odds = 1 + (100 / abs(juice))
    else:
        decimal_odds = 1 + (juice / 100)
    b = decimal_odds - 1.0

    # Edge → base win probability (capped at 59%)
    edge = max(0.0, float(edge_pts))
    base_win_prob = 0.50 + min(edge * 0.015, 0.09)

    # Confidence discount: regress toward 50% when uncertain
    conf = max(0.0, min(1.0, float(model_confidence)))
    p = 0.50 + (base_win_prob - 0.50) * conf

    q = 1.0 - p
    if b <= 0:
        return 0.0

    kelly = (b * p - q) / b
    quarter_kelly = kelly * 0.25
    result = max(0.0, quarter_kelly) * float(multiplier)
    return round(result, 4)


def evaluate_alpha(
    pred_spread: float,
    spread_line: Optional[float],
    model_confidence: float,
    trap_for_favorite: bool = False,
    revenge_info: Optional[Dict] = None,
    market_context: Optional[Dict] = None,
    game_id: str = "",
    home_team: str = "",
    away_team: str = "",
) -> Dict:
    """
    Unified alpha evaluator. Call twice:
      - At prediction time: market_context=None
      - After market merge: market_context=row from enriched CSV
    """
    revenge_info = revenge_info or {}
    market_context = market_context or {}
    reasoning: List[str] = []
    edge_types: List[str] = []
    is_alpha = False
    kelly_mult = 1.0

    model_side = "home" if pred_spread < 0 else "away"

    edge_pts = 0.0
    if spread_line is not None:
        try:
            edge_pts = abs(float(pred_spread) - float(spread_line))
        except (TypeError, ValueError):
            pass

    if edge_pts >= 3.0:
        is_alpha = True
        edge_types.append("MODEL_EDGE")
        reasoning.append(
            f"Model disagrees with line by {edge_pts:.1f}pts "
            f"({'home' if pred_spread < spread_line else 'away'} side)"
        )

    if trap_for_favorite:
        reasoning.append(
            "⚠️ TRAP GAME — ranked team vs weak opponent "
            "between quality games. Fade favorite historically profitable."
        )
        kelly_mult *= 0.70

    if revenge_info.get("revenge_flag"):
        team = revenge_info.get("revenge_team", "")
        margin = revenge_info.get("revenge_margin")
        reasoning.append(
            f"REVENGE SPOT: {team} team lost last meeting "
            f"by {margin} points — motivated opponent."
        )
        if team == model_side:
            is_alpha = True
            edge_types.append("REVENGE_CONFIRMED")

    market_evaluated = bool(market_context)

    if market_evaluated:
        steam = _as_bool(market_context.get("steam_flag", False))
        line_move = float(market_context.get("line_movement") or 0)

        # Guard: real steam requires actual line movement.
        # Reject false positives from ingestion where line_movement=0.0
        if steam and abs(line_move) < 0.5:
            steam = False
            log.debug(
                "Steam flag overridden for game_id=%s: "
                "line_movement=%.2f < 0.5 (ingestion false positive)",
                game_id,
                line_move,
            )

        if steam:
            steam_side = "home" if line_move > 0 else "away"
            if steam_side == model_side:
                reasoning.append(
                    f"🔥 Steam CONFIRMS model: sharp money "
                    f"moved line {abs(line_move):.1f}pts toward "
                    f"{model_side.upper()}"
                )
                is_alpha = True
                edge_types.append("STEAM_CONFIRMED")
            else:
                kelly_mult = 0.0
                reasoning.append(
                    f"🔥 STEAM AGAINST MODEL: sharp syndicate moved "
                    f"line {abs(line_move):.1f}pts toward "
                    f"{steam_side.upper()} — standing down."
                )

        rlm_side = _as_side(market_context.get("rlm_sharp_side"))
        if rlm_side:
            if rlm_side == model_side:
                is_alpha = True
                edge_types.append("RLM_CONFIRMED")
                reasoning.append(
                    f"Reverse line movement CONFIRMS model: "
                    f"public fading {model_side} but sharp money "
                    f"pushed line toward {model_side}"
                )
            else:
                kelly_mult = min(kelly_mult, 0.40)
                reasoning.append(
                    f"⚠️ RLM AGAINST MODEL: sharps on "
                    f"{rlm_side.upper()}, model likes "
                    f"{model_side.upper()} — reducing to 40%"
                )

        book_sharp = _as_side(market_context.get("book_sharp_side"))
        book_diff = float(market_context.get("book_spread_diff") or 0)
        if book_sharp:
            if book_sharp == model_side:
                reasoning.append(
                    f"Pinnacle vs DraftKings disagrees "
                    f"{book_diff:.1f}pts in model direction — "
                    f"sharp book agrees"
                )
                if book_diff >= 1.5:
                    is_alpha = True
                    edge_types.append("BOOK_DISAGREE_CONFIRMED")
            else:
                kelly_mult = min(kelly_mult, 0.60)
                reasoning.append(
                    f"⚠️ Book disagreement AGAINST model: "
                    f"Pinnacle pointing {book_sharp.upper()}, "
                    f"model likes {model_side.upper()}"
                )

        if _as_bool(market_context.get("line_freeze_flag", False)):
            kelly_mult = min(kelly_mult, 0.50)
            reasoning.append(
                "⚠️ LINE FREEZE: heavy public action, line unmoved — "
                "books comfortable. Model edge likely illusory."
            )

        home_tickets = market_context.get("home_tickets_pct")
        home_money = market_context.get("home_money_pct")
        if home_tickets is not None:
            try:
                tickets = float(home_tickets)
                if tickets >= 70 and model_side == "away":
                    reasoning.append(
                        f"Public fade: {tickets:.0f}% of tickets on home, "
                        f"model likes away — contrarian signal"
                    )
                elif tickets <= 30 and model_side == "home":
                    reasoning.append(
                        f"Public fade: only {tickets:.0f}% on home, "
                        f"model likes home — contrarian signal"
                    )
            except (TypeError, ValueError):
                pass

        if home_tickets is not None and home_money is not None:
            try:
                t = float(home_tickets)
                m = float(home_money)
                divergence = abs(m - t)
                if divergence >= 20:
                    sharp_on_home = m > t
                    sharp_side = "home" if sharp_on_home else "away"
                    if sharp_side == model_side:
                        reasoning.append(
                            f"Money/ticket split: {m:.0f}% money vs "
                            f"{t:.0f}% tickets on home — sharp money "
                            f"on {sharp_side} agrees with model"
                        )
                    else:
                        kelly_mult = min(kelly_mult, 0.70)
                        reasoning.append(
                            f"Money/ticket split AGAINST model: "
                            f"sharp money on {sharp_side} "
                            f"(divergence {divergence:.0f}pp)"
                        )
            except (TypeError, ValueError):
                pass

    try:
        summary_path = DATA_DIR / "model_accuracy_summary.csv"
        if summary_path.exists():
            s = pd.read_csv(summary_path).iloc[-1]

            beat_pinn = s.get("beat_pinnacle_pct")
            clv_n = float(s.get("clv_vs_pinnacle_n", 0) or 0)
            clv_with = s.get("clv_with_sharp_mean")
            ats_confirm = s.get("ats_market_confirmed")
            confirm_n = float(s.get("market_confirmed_n", 0) or 0)

            if clv_n >= 50:
                if beat_pinn and float(beat_pinn) > 52 and edge_pts > 2.0:
                    is_alpha = True
                    edge_types.append("CLV_VS_PINNACLE")
                    reasoning.append(
                        f"Model beats Pinnacle {float(beat_pinn):.0f}% "
                        f"of the time ({clv_n:.0f} games). "
                        f"Current gap {edge_pts:.1f}pts is signal."
                    )

                clv_dir = (
                    market_context.get("clv_direction")
                    if market_evaluated
                    else None
                )
                if (
                    clv_with
                    and float(clv_with) > 0.3
                    and clv_dir == "WITH_SHARP"
                ):
                    reasoning.append(
                        f"Model CLV +{float(clv_with):.2f}pts when "
                        f"aligned with sharp movement — this game qualifies."
                    )

                if (
                    ats_confirm
                    and float(ats_confirm) > 55
                    and confirm_n >= 20
                    and market_context.get("market_confirmed")
                ):
                    is_alpha = True
                    edge_types.append("MARKET_CONFIRMED_HISTORICAL")
                    reasoning.append(
                        f"Market-confirmed picks: "
                        f"{float(ats_confirm):.0f}% ATS "
                        f"(n={confirm_n:.0f}). This game confirmed."
                    )
    except Exception as e:
        log.debug("CLV criterion skipped: %s", e)

    if kelly_mult == 0.0:
        is_alpha = False

    final_kelly = kelly_fraction_calc(
        model_confidence=model_confidence,
        edge_pts=edge_pts,
        multiplier=kelly_mult,
    )
    kelly_units = round(final_kelly * 100, 2)

    return {
        "is_alpha": is_alpha,
        "edge_types": "|".join(edge_types),
        "alpha_reasoning": " | ".join(reasoning),
        "kelly_fraction": final_kelly,
        "kelly_units": kelly_units,
        "kelly_multiplier": round(kelly_mult, 3),
        "market_evaluated": market_evaluated,
        "edge_pts": round(edge_pts, 2),
    }
