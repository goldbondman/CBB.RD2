#!/usr/bin/env python3
"""Validate market line coverage and predictions-with-context line attachment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path, low_memory=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--debug-dir", default="debug")
    parser.add_argument("--min-coverage-ratio", type=float, default=0.6)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    debug_dir = Path(args.debug_dir)

    errors: list[str] = []

    merge_report_path = debug_dir / "merge_report.json"
    if merge_report_path.exists():
        report = json.loads(merge_report_path.read_text(encoding="utf-8"))
        slate_games = int(report.get("slate_games", 0))
        effective_slate_games = int(report.get("effective_slate_games", slate_games))
        filtered_final_games = int(report.get("filtered_final_games", 0))
        merged_games = int(report.get("merged_games", 0))
        coverage_basis = effective_slate_games if effective_slate_games > 0 else slate_games
        required = int(round(coverage_basis * args.min_coverage_ratio))
        print(
            "[INFO] merge_report "
            f"slate_games={slate_games} "
            f"effective_slate_games={effective_slate_games} "
            f"filtered_final_games={filtered_final_games} "
            f"merged_games={merged_games} required={required}"
        )
        if coverage_basis >= 10 and merged_games < required:
            errors.append(
                "merge_report coverage failed: "
                f"merged_games={merged_games} < required={required} "
                f"(coverage_basis_games={coverage_basis}, slate_games={slate_games})"
            )
    else:
        print(f"[WARN] merge report not found: {merge_report_path}")

    latest_path = data_dir / "market_lines_latest.csv"
    latest_df = _load_csv(latest_path)
    print(f"[INFO] market_lines_latest rows={len(latest_df)}")

    ctx_path = data_dir / "predictions_with_context.csv"
    ctx_df = _load_csv(ctx_path)
    print(f"[INFO] predictions_with_context rows={len(ctx_df)}")

    required_cols = ["event_id", "line_status", "line_missing_reason", "line_source_used", "spread_line", "total_line"]
    missing_cols = [c for c in required_cols if c not in ctx_df.columns]
    if missing_cols:
        errors.append(f"predictions_with_context missing required columns: {missing_cols}")
    else:
        spread_non_null = pd.to_numeric(ctx_df["spread_line"], errors="coerce").notna()
        total_non_null = pd.to_numeric(ctx_df["total_line"], errors="coerce").notna()
        missing_lines = ~(spread_non_null & total_non_null)
        missing_reason = ctx_df["line_missing_reason"].astype("string").str.strip()
        missing_reason_ok = missing_reason.notna() & (missing_reason != "")
        bad_missing = int((missing_lines & ~missing_reason_ok).sum())

        print(
            "[INFO] context line coverage "
            f"spread_non_null={int(spread_non_null.sum())}/{len(ctx_df)} "
            f"total_non_null={int(total_non_null.sum())}/{len(ctx_df)} "
            f"missing_with_reason={int((missing_lines & missing_reason_ok).sum())}/{int(missing_lines.sum())}"
        )
        if bad_missing > 0:
            errors.append(f"{bad_missing} rows missing spread/total without line_missing_reason")

    if errors:
        for err in errors:
            print(f"[ERROR] {err}")
        return 1

    print("[OK] market/context integrity validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
