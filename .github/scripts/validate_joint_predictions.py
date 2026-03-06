#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _show_rows(df: pd.DataFrame, mask: pd.Series, label: str, limit: int = 20) -> None:
    cols = [
        c
        for c in [
            "event_id",
            "game_id",
            "home_team",
            "away_team",
            "model_status",
            "pred_margin",
            "pred_margin_final",
            "pred_margin_raw",
            "allocation_pct",
            "spread_signal_raw",
        ]
        if c in df.columns
    ]
    bad = df.loc[mask, cols].copy()
    if bad.empty:
        return
    print(f"[ERROR] {label} (showing up to {limit} rows)")
    print(bad.head(limit).to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate predictions_joint_latest spread guardrails.")
    parser.add_argument("--input", default="data/predictions_joint_latest.csv")
    parser.add_argument("--debug-json", default="debug/pred_margin_validation.json")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Missing predictions file: {input_path}")
        return 1

    df = pd.read_csv(input_path, low_memory=False)
    if df.empty:
        print(f"[ERROR] Empty predictions file: {input_path}")
        return 1

    status = df.get("model_status", pd.Series("", index=df.index)).astype(str)
    blocked = status.str.startswith("BLOCKED")
    margin_final_col = "pred_margin_final" if "pred_margin_final" in df.columns else "pred_margin"
    margin_final = _to_num(df.get(margin_final_col, pd.Series(index=df.index, dtype=float)))
    margin_raw = _to_num(df.get("pred_margin_raw", pd.Series(index=df.index, dtype=float)))

    violations: list[str] = []

    too_large = margin_final.abs() > 60
    if too_large.any():
        violations.append("rows_with_abs_pred_margin_final_gt_60")
        _show_rows(df, too_large, "abs(pred_margin_final) > 60")

    active_numeric = margin_final[~blocked & margin_final.notna()]
    pct_gt_35 = float((active_numeric.abs() > 35).mean() * 100.0) if not active_numeric.empty else 0.0
    if pct_gt_35 > 5.0:
        violations.append("pct_rows_abs_pred_margin_final_gt_35_exceeds_5pct")
        mask = (~blocked) & margin_final.notna() & (margin_final.abs() > 35)
        _show_rows(df, mask, f"non-blocked rows with abs(pred_margin_final)>35: {pct_gt_35:.2f}%")

    blocked_numeric = blocked & (margin_final.notna() | margin_raw.notna())
    if blocked_numeric.any():
        violations.append("blocked_rows_have_numeric_pred_margin")
        _show_rows(df, blocked_numeric, "blocked rows with numeric pred_margin")

    constant_values = {-70.0, -44.0, 44.0, 70.0}
    constant_mask = (~blocked) & margin_final.isin(constant_values)
    if constant_mask.any():
        violations.append("non_blocked_constant_pred_margin_values_detected")
        _show_rows(df, constant_mask, "non-blocked rows with suspicious constants (-70/-44/44/70)")

    payload = {
        "input": str(input_path),
        "rows": int(len(df)),
        "non_blocked_rows": int((~blocked).sum()),
        "margin_final_column": margin_final_col,
        "pct_non_blocked_abs_margin_gt_35": pct_gt_35,
        "counts": {
            "abs_margin_gt_60": int(too_large.sum()),
            "blocked_with_numeric_margin": int(blocked_numeric.sum()),
            "non_blocked_suspicious_constants": int(constant_mask.sum()),
        },
        "violations": violations,
    }

    debug_path = Path(args.debug_json)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if violations:
        print(f"[ERROR] joint prediction guardrails failed: {violations}")
        print(f"[INFO] debug report written: {debug_path.resolve()}")
        return 1

    print(
        "[OK] joint prediction guardrails passed | "
        f"rows={len(df)} non_blocked={int((~blocked).sum())} pct_abs_gt_35={pct_gt_35:.2f}%"
    )
    print(f"[INFO] debug report written: {debug_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
