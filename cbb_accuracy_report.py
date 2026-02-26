#!/usr/bin/env python3
"""Generate per-model ATS accuracy report with dynamic/backtest/blended weights."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

DATA_DIR = Path("data")
RESULTS_LOG = DATA_DIR / "results_log.csv"
DYNAMIC_WEIGHTS = DATA_DIR / "dynamic_model_weights.json"
BACKTEST_WEIGHTS = DATA_DIR / "backtest_optimized_weights.json"
OUT_PATH = DATA_DIR / "model_accuracy_summary.csv"
GRADED_PREDS = DATA_DIR / "predictions_graded.csv"

MODEL_IDS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]


def _safe_float(value, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_backtest_weights() -> Dict[str, float]:
    out = {mid: round(1 / len(MODEL_IDS), 4) for mid in MODEL_IDS}
    if not BACKTEST_WEIGHTS.exists() or BACKTEST_WEIGHTS.stat().st_size <= 2:
        return out

    payload = json.loads(BACKTEST_WEIGHTS.read_text())
    if isinstance(payload, dict):
        for mid in MODEL_IDS:
            key = f"{mid}_weight"
            if key in payload:
                out[mid] = _safe_float(payload[key], out[mid])
    total = sum(out.values())
    return {k: round(v / total, 4) for k, v in out.items()} if total > 0 else out


def _load_dynamic_payload() -> dict:
    if not DYNAMIC_WEIGHTS.exists() or DYNAMIC_WEIGHTS.stat().st_size <= 2:
        return {}
    return json.loads(DYNAMIC_WEIGHTS.read_text())


def build_mc_calibration_curve(df: pd.DataFrame) -> None:
    if "mc_cover_pct_bucket" not in df.columns:
        return
    # Filter to games that have been graded
    graded = df[df.get("graded") == True].copy()
    if graded.empty:
        return

    curve = graded.groupby("mc_cover_pct_bucket").agg(
        predicted_prob=("mc_cover_pct", "mean"),
        actual_cover_rate=("home_covered_pred", "mean"),
        n_games=("game_id", "count"),
        calib_error=("mc_cover_calib_error", "mean"),
    ).reset_index()

    curve_path = DATA_DIR / "mc_calibration_curve.csv"
    curve.to_csv(curve_path, index=False)
    print(f"MC calibration curve written: {len(curve)} buckets to {curve_path}")


def _ats_pct(df: pd.DataFrame, spread_col: str, days: int) -> float:
    recent = df.tail(days).copy()
    recent = recent.dropna(subset=[spread_col, "actual_margin", "spread_line"])
    if recent.empty:
        return 0.0
    covered = (
        (pd.to_numeric(recent[spread_col], errors="coerce") > 0)
        == (pd.to_numeric(recent["actual_margin"], errors="coerce") > pd.to_numeric(recent["spread_line"], errors="coerce"))
    )
    return round(float(covered.mean()) * 100, 1)


def main() -> None:
    if not RESULTS_LOG.exists() or RESULTS_LOG.stat().st_size <= 2:
        print("results_log.csv not found or empty; nothing to report")
        return

    results = pd.read_csv(RESULTS_LOG)
    if "spread_line" not in results.columns and "market_spread" in results.columns:
        results["spread_line"] = results["market_spread"]

    backtest_weights = _load_backtest_weights()
    dynamic_payload = _load_dynamic_payload()
    dynamic_weights = dynamic_payload.get("dynamic_weights", {}) if isinstance(dynamic_payload, dict) else {}
    blended_weights = dynamic_payload.get("blended_weights", {}) if isinstance(dynamic_payload, dict) else {}

    rows = []
    for mid in MODEL_IDS:
        spread_col = f"{mid}_spread"
        if spread_col not in results.columns:
            continue
        rows.append(
            {
                "model_id": mid,
                "l14_ats_pct": _ats_pct(results, spread_col, 14),
                "l30_ats_pct": _ats_pct(results, spread_col, 30),
                "dynamic_weight": round(_safe_float(dynamic_weights.get(mid), 1 / 7), 4),
                "backtest_weight": round(_safe_float(backtest_weights.get(mid), 1 / 7), 4),
                "blended_weight": round(_safe_float(blended_weights.get(mid), _safe_float(backtest_weights.get(mid), 1 / 7)), 4),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} model rows to {OUT_PATH} @ {datetime.utcnow().isoformat()}Z")

    # MC calibration curve
    if GRADED_PREDS.exists():
        graded_df = pd.read_csv(GRADED_PREDS)
        build_mc_calibration_curve(graded_df)


if __name__ == "__main__":
    main()
