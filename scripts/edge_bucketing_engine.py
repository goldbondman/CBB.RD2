#!/usr/bin/env python3
"""Build spread/total edge buckets from discovered predictions vs market vs actual data."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from edge_overconfidence_common import build_dataset_spec, discover_input_table, load_normalized_frame


EDGE_BINS = [0.0, 2.0, 4.0, 6.0, 8.0, float("inf")]
EDGE_LABELS = ["0-2", "2-4", "4-6", "6-8", "8+"]
SIGN_NOTE = "Spread edge uses home-team perspective line convention; edge = model_line - market_line."

OUTPUT_COLUMNS = [
    "market_type",
    "bucket",
    "games_count",
    "avg_abs_edge",
    "avg_abs_error",
    "direction_hit_rate",
    "avg_confidence",
    "median_confidence",
    "edge_definition",
    "sign_convention_note",
    "input_source_file",
    "model_line_column",
    "market_line_column",
    "actual_column",
    "generated_at_utc",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_blocked_summary(path: Path, *, reason: str, missing_files: list[str], missing_columns: dict[str, list[str]]) -> None:
    lines = [
        "# Exec Summary: edge_bucketing_engine",
        "",
        "- status: `BLOCKED`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- reason: {reason}",
    ]
    if missing_files:
        lines.append("- missing files:")
        for item in missing_files:
            lines.append(f"  - `{item}`")
    if missing_columns:
        lines.append("- missing columns:")
        for file_name, cols in missing_columns.items():
            lines.append(f"  - `{file_name}`: `{', '.join(cols)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _market_bucket_df(df: pd.DataFrame, *, market_type: str, model_col: str, market_col: str, actual_col: str) -> pd.DataFrame:
    work = df[[model_col, market_col, actual_col, "confidence"]].copy()
    work = work.dropna(subset=[model_col, market_col, actual_col])
    if work.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    work["edge"] = work[model_col] - work[market_col]
    work["abs_edge"] = work["edge"].abs()
    work["abs_error"] = (work[model_col] - work[actual_col]).abs()
    work["actual_vs_market"] = work[actual_col] - work[market_col]
    work["direction_hit"] = (work["edge"] * work["actual_vs_market"]) > 0
    work["bucket"] = pd.cut(work["abs_edge"], bins=EDGE_BINS, labels=EDGE_LABELS, right=False, include_lowest=True)

    agg = (
        work.groupby("bucket", dropna=False, observed=False)
        .agg(
            games_count=("edge", "size"),
            avg_abs_edge=("abs_edge", "mean"),
            avg_abs_error=("abs_error", "mean"),
            direction_hit_rate=("direction_hit", "mean"),
            avg_confidence=("confidence", "mean"),
            median_confidence=("confidence", "median"),
        )
        .reset_index()
    )
    agg["market_type"] = market_type
    agg["edge_definition"] = "edge = model_line - market_line"
    agg["sign_convention_note"] = SIGN_NOTE if market_type == "spread" else "Total edge uses over/under line convention."
    agg["model_line_column"] = model_col
    agg["market_line_column"] = market_col
    agg["actual_column"] = actual_col
    agg["generated_at_utc"] = _utc_now()
    return agg


def run_edge_bucketing_engine(
    *,
    data_dir: Path,
    output_csv: Path,
    output_summary_md: Path,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    discovered, inspected = discover_input_table(data_dir)
    if not discovered:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_summary_md,
            reason="No usable predictions/results file discovered.",
            missing_files=[str(data_dir / c) for c in ["results_log_graded.csv", "results_log.csv", "predictions_graded.csv"]],
            missing_columns={},
        )
        return 1

    spec, missing_columns = build_dataset_spec(discovered)
    if spec is None:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_summary_md,
            reason=f"Input discovered ({discovered.name}) but required columns are missing.",
            missing_files=[],
            missing_columns=missing_columns,
        )
        return 1

    base = load_normalized_frame(spec)
    spread = _market_bucket_df(base, market_type="spread", model_col="model_spread", market_col="market_spread", actual_col="actual_margin")
    total = _market_bucket_df(base, market_type="total", model_col="model_total", market_col="market_total", actual_col="actual_total")

    report = pd.concat([spread, total], ignore_index=True)
    if report.empty:
        report = pd.DataFrame(columns=OUTPUT_COLUMNS)
        report.to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_summary_md,
            reason=f"Discovered input {discovered.name} but no numeric rows were eligible for edge computation.",
            missing_files=[],
            missing_columns={},
        )
        return 1

    report["input_source_file"] = str(discovered)
    report = report[OUTPUT_COLUMNS]
    report.to_csv(output_csv, index=False)

    summary_lines = [
        "# Exec Summary: edge_bucketing_engine",
        "",
        "- status: `OK`",
        f"- input_source_file: `{discovered}`",
        f"- rows: `{len(report)}`",
        "- edge_definition: `edge = model_line - market_line`",
        f"- spread_sign_convention: {SIGN_NOTE}",
        "- inspected_candidates:",
    ]
    for item in inspected:
        summary_lines.append(f"  - `{item}`")
    output_summary_md.parent.mkdir(parents=True, exist_ok=True)
    output_summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/analytics/edge_buckets.csv"))
    parser.add_argument("--output-summary-md", type=Path, default=Path("data/analytics/edge_buckets_exec_summary.md"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_summary_md = (
        args.output_summary_md if args.output_summary_md.is_absolute() else repo_root / args.output_summary_md
    )

    rc = run_edge_bucketing_engine(data_dir=data_dir, output_csv=output_csv, output_summary_md=output_summary_md)
    print(json.dumps({"output_csv": str(output_csv), "output_summary_md": str(output_summary_md), "exit_code": rc}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
