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
OUT_PATH = DATA_DIR / "cbb_accuracy_report.csv"
GRADED_PREDICTIONS = DATA_DIR / "predictions_graded.csv"

MODEL_IDS = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
MODEL_SPREAD_FALLBACKS = {
    "m1": ["m1_spread", "fourfactors_spread", "ens_fourfactors_spread"],
    "m2": ["m2_spread", "adjefficiency_spread", "ens_adjefficiency_spread"],
    "m3": ["m3_spread", "pythagorean_spread", "ens_pythagorean_spread"],
    "m4": ["m4_spread", "momentum_spread", "ens_momentum_spread"],
    "m5": ["m5_spread", "situational_spread", "ens_situational_spread"],
    "m6": ["m6_spread", "cagerankings_spread", "ens_cagerankings_spread"],
    "m7": ["m7_spread", "regressedeff_spread", "ens_regressedeff_spread"],
}


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


def _hydrate_model_spreads(results: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    if GRADED_PREDICTIONS.exists() and GRADED_PREDICTIONS.stat().st_size > 2:
        graded = pd.read_csv(GRADED_PREDICTIONS)
        join_col = "event_id" if "event_id" in out.columns and "event_id" in graded.columns else None
        if join_col is None and "game_id" in out.columns and "game_id" in graded.columns:
            join_col = "game_id"
        if join_col:
            extra_cols = {join_col}
            for mid in MODEL_IDS:
                extra_cols.update([c for c in MODEL_SPREAD_FALLBACKS[mid] if c in graded.columns])
            graded = graded[list(extra_cols)].drop_duplicates(subset=[join_col], keep="last")
            out = out.merge(graded, on=join_col, how="left", suffixes=("", "_graded"))

    for mid in MODEL_IDS:
        target = f"{mid}_spread"
        if target not in out.columns:
            out[target] = pd.NA
        for candidate in MODEL_SPREAD_FALLBACKS[mid]:
            if candidate == target:
                continue
            if candidate in out.columns:
                out[target] = out[target].fillna(out[candidate])
            graded_col = f"{candidate}_graded"
            if graded_col in out.columns:
                out[target] = out[target].fillna(out[graded_col])
    return out


def main() -> None:
    if not RESULTS_LOG.exists() or RESULTS_LOG.stat().st_size <= 2:
        print("results_log.csv not found or empty; nothing to report")
        return

    results = pd.read_csv(RESULTS_LOG)
    if "spread_line" not in results.columns and "market_spread" in results.columns:
        results["spread_line"] = results["market_spread"]

    results = _hydrate_model_spreads(results)

    backtest_weights = _load_backtest_weights()
    dynamic_payload = _load_dynamic_payload()
    dynamic_weights = dynamic_payload.get("dynamic_weights", {}) if isinstance(dynamic_payload, dict) else {}
    blended_weights = dynamic_payload.get("blended_weights", {}) if isinstance(dynamic_payload, dict) else {}

    rows = []
    for mid in MODEL_IDS:
        spread_col = f"{mid}_spread"
        valid_rows = int(results[[spread_col, "actual_margin", "spread_line"]].dropna().shape[0])
        verification_status = "verified" if valid_rows > 0 else "rejected"
        rows.append(
            {
                "model_id": mid,
                "l14_ats_pct": _ats_pct(results, spread_col, 14),
                "l30_ats_pct": _ats_pct(results, spread_col, 30),
                "graded_games": valid_rows,
                "verification_status": verification_status,
                "dynamic_weight": round(_safe_float(dynamic_weights.get(mid), 1 / 7), 4),
                "backtest_weight": round(_safe_float(backtest_weights.get(mid), 1 / 7), 4),
                "blended_weight": round(_safe_float(blended_weights.get(mid), _safe_float(backtest_weights.get(mid), 1 / 7)), 4),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} model rows to {OUT_PATH} @ {datetime.utcnow().isoformat()}Z")


if __name__ == "__main__":
    main()
