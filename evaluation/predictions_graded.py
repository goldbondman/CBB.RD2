#!/usr/bin/env python3
"""Build graded predictions dataset with CLV + market signal context."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pipeline_csv_utils import normalize_game_id

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "predictions_graded.csv"


_CANDIDATES = [
    Path("data/predictions_with_context.csv"),
    Path("data/results_log_graded.csv"),
    Path("data/predictions_combined_latest.csv"),
]


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def grade_row(row: pd.Series) -> Dict[str, Any]:
    pred_spread = _safe_float(row.get("predicted_spread", row.get("pred_spread")))
    if pred_spread is None:
        pred_spread = _safe_float(row.get("pred_spread"))

    actual_margin = _safe_float(row.get("actual_margin"))
    if actual_margin is None:
        home_score = _safe_float(row.get("home_score"))
        away_score = _safe_float(row.get("away_score"))
        if home_score is not None and away_score is not None:
            actual_margin = home_score - away_score

    spread_line = _safe_float(row.get("home_spread_current", row.get("spread_line")))
    if spread_line is None:
        spread_line = _safe_float(row.get("spread_line"))

    spread_error = None
    if pred_spread is not None and actual_margin is not None:
        spread_error = round(abs(pred_spread - actual_margin), 3)

    home_covered_pred = None
    if pred_spread is not None and spread_line is not None and actual_margin is not None:
        predicted_home_covers = pred_spread > spread_line
        actual_home_covers = actual_margin > spread_line
        home_covered_pred = bool(predicted_home_covers == actual_home_covers)

    pred_spread = row.get("predicted_spread", row.get("pred_spread"))
    closing_line = row.get("home_spread_current")
    opening_line = row.get("home_spread_open")
    pinn_line = row.get("pinnacle_spread")
    dk_line = row.get("draftkings_spread")

    def _clv(pred: Any, line: Any) -> Optional[float]:
        """CLV convention: line - pred (positive => model found value)."""
        if pred is None or line is None:
            return None
        try:
            # CLV convention: clv_vs_consensus = line - pred (positive => model found value).
            return round(float(line) - float(pred), 3)
        except (TypeError, ValueError):
            return None

    clv_vs_consensus = _clv(pred_spread, closing_line)
    clv_vs_pinnacle = _clv(pred_spread, pinn_line)
    clv_vs_dk = _clv(pred_spread, dk_line)
    clv_vs_open = _clv(pred_spread, opening_line)

    def _beat_line(pred: Any, line: Any) -> Optional[bool]:
        if pred is None or line is None:
            return None
        try:
            return abs(float(pred)) < abs(float(line))
        except (TypeError, ValueError):
            return None

    beat_consensus = _beat_line(pred_spread, closing_line)
    beat_pinnacle = _beat_line(pred_spread, pinn_line)

    line_move = row.get("line_movement")
    clv_direction = None
    if pred_spread is not None and line_move is not None:
        try:
            model_dir = 1 if float(pred_spread) > 0 else -1
            line_dir = 1 if float(line_move) > 0 else -1
            clv_direction = "WITH_SHARP" if model_dir == line_dir else "AGAINST_SHARP"
        except (TypeError, ValueError):
            pass

    steam = bool(row.get("steam_flag", False))
    rlm_flag = bool(row.get("rlm_flag", False))
    rlm_side = row.get("rlm_sharp_side")
    book_dis = bool(row.get("book_disagreement_flag", False))
    book_side = row.get("book_sharp_side")
    freeze = bool(row.get("line_freeze_flag", False))

    try:
        model_side = "home" if float(pred_spread or 0) > 0 else "away"
    except (TypeError, ValueError):
        model_side = None

    market_confirmed = bool(
        (rlm_flag and rlm_side == model_side) or (book_dis and book_side == model_side)
    )
    market_contradicted = bool(
        (rlm_flag and rlm_side is not None and rlm_side != model_side)
        or (book_dis and book_side is not None and book_side != model_side)
        or freeze
    )

    covered = home_covered_pred
    covered_with_confirm = bool(covered) if market_confirmed and covered is not None else None
    covered_with_contra = bool(covered) if market_contradicted and covered is not None else None

    if str(row.get("game_quality_status", "")).upper() == "SKIP":
        clv_vs_consensus = None
        clv_vs_pinnacle = None
        clv_vs_dk = None
        clv_vs_open = None
        beat_consensus = None
        beat_pinnacle = None
        clv_direction = None

    return {
        "spread_error": spread_error,
        "abs_spread_error": spread_error,
        "home_covered_pred": home_covered_pred,
        "graded": home_covered_pred is not None,
        "clv_vs_consensus": clv_vs_consensus,
        "clv_vs_pinnacle": clv_vs_pinnacle,
        "clv_vs_dk": clv_vs_dk,
        "clv_vs_open": clv_vs_open,
        "beat_consensus": beat_consensus,
        "beat_pinnacle": beat_pinnacle,
        "clv_direction": clv_direction,
        "market_confirmed": market_confirmed,
        "market_contradicted": market_contradicted,
        "had_steam": steam,
        "had_rlm": rlm_flag,
        "had_book_disagree": book_dis,
        "had_line_freeze": freeze,
        "line_movement_total": line_move,
        "covered_with_market_confirm": covered_with_confirm,
        "covered_with_market_contra": covered_with_contra,
    }



def main() -> None:
    _src = next((p for p in _CANDIDATES if p.exists() and p.stat().st_size > 0), None)
    if _src is None:
        raise FileNotFoundError("No prediction input found for grading")
    if _src.name != "predictions_with_context.csv":
        log.warning(
            "[GRADING] Falling back to %s — predictions_with_context.csv missing or empty. "
            "CLV fields will be null. Run enrichment/predictions_with_context.py first.",
            _src,
        )

    df = pd.read_csv(_src, low_memory=False)
    # Alias normalization: legacy files may ship event_id; standardize joins/exports on game_id.
    if "event_id" in df.columns and "game_id" not in df.columns:
        df = df.rename(columns={"event_id": "game_id"})
    if "game_id" in df.columns:
        df["game_id"] = df["game_id"].map(normalize_game_id)
    if df.empty:
        pd.DataFrame().to_csv(OUTPUT_PATH, index=False)
        log.info("Input %s empty; wrote empty graded output.", _src)
        return

    graded = df.apply(grade_row, axis=1, result_type="expand")
    out = pd.concat([df, graded], axis=1)
    out.to_csv(OUTPUT_PATH, index=False)
    log.info("Wrote %s rows to %s", len(out), OUTPUT_PATH)


if __name__ == "__main__":
    main()
