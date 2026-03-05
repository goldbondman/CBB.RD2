#!/usr/bin/env python3
"""Flag overconfidence by quantile-calibrated confidence/edge/error thresholds."""

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


OUTPUT_COLUMNS = [
    "event_id",
    "game_id",
    "game_datetime_utc",
    "market_type",
    "model_line",
    "market_line",
    "actual_value",
    "edge",
    "abs_edge",
    "abs_error",
    "confidence",
    "confidence_threshold_q80",
    "edge_threshold_q80",
    "error_threshold_q80",
    "is_high_confidence",
    "is_high_edge",
    "is_high_error",
    "overconfidence_flag",
    "flag_reason",
    "edge_definition",
    "sign_convention_note",
    "input_source_file",
    "generated_at_utc",
]

SPREAD_SIGN_NOTE = (
    "Spread line convention assumes home-team perspective; negative means home favored. "
    "Edge is computed as model_line - market_line."
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _write_blocked_summary(path: Path, *, reason: str, missing_files: list[str], missing_columns: dict[str, list[str]]) -> None:
    lines = [
        "# Exec Summary: overconfidence_detector",
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


def _build_market_rows(df: pd.DataFrame, market_type: str) -> pd.DataFrame:
    if market_type == "spread":
        model_col, market_col, actual_col = "model_spread", "market_spread", "actual_margin"
        sign_note = SPREAD_SIGN_NOTE
    else:
        model_col, market_col, actual_col = "model_total", "market_total", "actual_total"
        sign_note = "Total edge is computed as model_line - market_line using totals/O-U convention."

    work = df[["event_id", "game_id", "game_datetime_utc", model_col, market_col, actual_col, "confidence"]].copy()
    work = work.dropna(subset=[model_col, market_col, actual_col])
    if work.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    work = work.rename(columns={model_col: "model_line", market_col: "market_line", actual_col: "actual_value"})
    work["edge"] = work["model_line"] - work["market_line"]
    work["abs_edge"] = work["edge"].abs()
    work["abs_error"] = (work["model_line"] - work["actual_value"]).abs()
    work["market_type"] = market_type

    conf_thr = work["confidence"].quantile(0.80) if work["confidence"].notna().any() else float("nan")
    edge_thr = work["abs_edge"].quantile(0.80)
    err_thr = work["abs_error"].quantile(0.80)

    work["confidence_threshold_q80"] = conf_thr
    work["edge_threshold_q80"] = edge_thr
    work["error_threshold_q80"] = err_thr
    work["is_high_confidence"] = work["confidence"] >= conf_thr if pd.notna(conf_thr) else False
    work["is_high_edge"] = work["abs_edge"] >= edge_thr
    work["is_high_error"] = work["abs_error"] >= err_thr
    work["overconfidence_flag"] = (work["is_high_confidence"] | work["is_high_edge"]) & work["is_high_error"]
    work["flag_reason"] = work.apply(
        lambda r: (
            "high_conf_and_high_error"
            if r["is_high_confidence"] and r["is_high_error"]
            else ("high_edge_and_high_error" if r["is_high_edge"] and r["is_high_error"] else "not_flagged")
        ),
        axis=1,
    )
    work["edge_definition"] = "edge = model_line - market_line"
    work["sign_convention_note"] = sign_note
    work["generated_at_utc"] = _utc_now()
    return work


def run_overconfidence_detector(*, data_dir: Path, output_csv: Path, output_md: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    discovered, inspected = discover_input_table(data_dir)

    if not discovered:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            reason="No usable predictions/results file discovered.",
            missing_files=[str(data_dir / c) for c in ["results_log_graded.csv", "results_log.csv", "predictions_graded.csv"]],
            missing_columns={},
        )
        return 1

    spec, missing_columns = build_dataset_spec(discovered)
    if spec is None:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            reason=f"Input discovered ({discovered.name}) but required columns are missing.",
            missing_files=[],
            missing_columns=missing_columns,
        )
        return 1

    base = load_normalized_frame(spec)
    spread_rows = _build_market_rows(base, "spread")
    total_rows = _build_market_rows(base, "total")
    report = pd.concat([spread_rows, total_rows], ignore_index=True)

    if report.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            reason=f"Discovered input {discovered.name} but no numeric rows were eligible for overconfidence checks.",
            missing_files=[],
            missing_columns={},
        )
        return 1

    report["input_source_file"] = str(discovered)
    report = report[OUTPUT_COLUMNS]
    report.to_csv(output_csv, index=False)

    flagged = int(report["overconfidence_flag"].sum())
    lines = [
        "# Exec Summary: overconfidence_detector",
        "",
        "- status: `OK`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- input_source_file: `{discovered}`",
        f"- rows: `{len(report)}`",
        f"- flagged_rows: `{flagged}`",
        "- edge_definition: `edge = model_line - market_line`",
        f"- spread_sign_convention: {SPREAD_SIGN_NOTE}",
        "- quantile_thresholds: `q80 for confidence, abs(edge), abs(error) per market type`",
        "- inspected_candidates:",
    ]
    for item in inspected:
        lines.append(f"  - `{item}`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/analytics/overconfidence_report.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/analytics/overconfidence_exec_summary.md"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md

    rc = run_overconfidence_detector(data_dir=data_dir, output_csv=output_csv, output_md=output_md)
    print(json.dumps({"output_csv": str(output_csv), "output_md": str(output_md), "exit_code": rc}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
