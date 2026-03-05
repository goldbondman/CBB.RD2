#!/usr/bin/env python3
"""Scan discovered feature tables and report per-column null rates."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "source_file",
    "column_name",
    "total_rows_in_file",
    "scanned_rows",
    "null_count",
    "non_null_count",
    "null_rate_pct",
    "dtype",
    "sample_limited",
    "scanned_at_utc",
]

INCLUDE_TOKENS = ("feature", "metric", "weighted")
EXCLUDE_TOKENS = (
    "report",
    "audit",
    "history",
    "summary",
    "ranking",
    "prediction",
    "result",
    "market",
    "backtest",
    "weight",
    "pick",
    "alert",
    "lineage",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _is_feature_candidate_name(name: str) -> bool:
    low = name.lower()
    return any(token in low for token in INCLUDE_TOKENS) and not any(token in low for token in EXCLUDE_TOKENS)


def _git_tracked_feature_candidates(repo_root: Path) -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files", "data/*.csv", "data/csv/*.csv"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    return [Path(line) for line in lines if _is_feature_candidate_name(Path(line).name)]


def discover_feature_tables(repo_root: Path, data_dir: Path) -> tuple[list[Path], list[Path]]:
    candidates: list[Path] = []
    for folder in (data_dir, data_dir / "csv"):
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.csv")):
            if _is_feature_candidate_name(path.name):
                candidates.append(path)

    unique_existing = sorted({path.resolve() for path in candidates})
    tracked = _git_tracked_feature_candidates(repo_root)
    missing_tracked = [repo_root / rel for rel in tracked if not (repo_root / rel).exists()]
    return [Path(p) for p in unique_existing], missing_tracked


def _write_blocked_summary(
    output_md: Path,
    *,
    missing_files: list[str],
    missing_columns: dict[str, list[str]],
    note: str,
) -> None:
    lines = [
        "# Exec Summary: feature_null_scanner",
        "",
        "- status: `BLOCKED`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- note: {note}",
    ]
    if missing_files:
        lines.append("- missing files:")
        for item in missing_files:
            lines.append(f"  - `{item}`")
    if missing_columns:
        lines.append("- missing columns:")
        for file_name, cols in missing_columns.items():
            lines.append(f"  - `{file_name}`: `{', '.join(cols)}`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ok_summary(output_md: Path, report_csv: Path, report_df: pd.DataFrame) -> None:
    key_cols = ["source_file", "column_name", "null_rate_pct"]
    null_rates = {}
    for col in key_cols:
        if col in report_df.columns:
            null_rates[col] = round(float(report_df[col].isna().mean()) * 100.0, 2)

    lines = [
        "# Exec Summary: feature_null_scanner",
        "",
        "- status: `OK`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- report_csv: `{report_csv}`",
        f"- rows: `{len(report_df)}`",
        f"- columns: `{len(report_df.columns)}`",
    ]
    if null_rates:
        lines.append("- report key column null rates:")
        for col, rate in null_rates.items():
            lines.append(f"  - `{col}`: `{rate}%`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_feature_null_scanner(
    *,
    repo_root: Path,
    data_dir: Path,
    output_csv: Path,
    output_md: Path,
    sample_limit: int | None,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scanned_at = _utc_now()

    feature_tables, missing_tracked = discover_feature_tables(repo_root, data_dir)
    rows: list[dict[str, object]] = []
    missing_columns: dict[str, list[str]] = {}

    if not feature_tables:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty.to_csv(output_csv, index=False)
        missing_files = [str(path) for path in missing_tracked]
        if not missing_files:
            missing_files = [
                str(data_dir / "*feature*.csv"),
                str(data_dir / "*metric*.csv"),
                str(data_dir / "*weighted*.csv"),
                str(data_dir / "csv" / "*feature*.csv"),
                str(data_dir / "csv" / "*metric*.csv"),
                str(data_dir / "csv" / "*weighted*.csv"),
            ]
        _write_blocked_summary(
            output_md,
            missing_files=missing_files,
            missing_columns={},
            note="No feature tables discovered for scanning.",
        )
        return 1

    for table_path in feature_tables:
        try:
            full_df = pd.read_csv(table_path, dtype=str, low_memory=False)
        except Exception:
            missing_columns[str(table_path)] = ["<unreadable_csv>"]
            continue

        total_rows = int(len(full_df))
        if sample_limit is not None and sample_limit >= 0:
            scan_df = full_df.head(sample_limit)
            sample_limited = total_rows > len(scan_df)
        else:
            scan_df = full_df
            sample_limited = False

        for col in scan_df.columns:
            series = scan_df[col]
            null_count = int(series.isna().sum())
            scanned_rows = int(len(scan_df))
            non_null_count = scanned_rows - null_count
            null_rate = round((null_count / scanned_rows) * 100.0, 4) if scanned_rows else 0.0
            rows.append(
                {
                    "source_file": str(table_path),
                    "column_name": str(col),
                    "total_rows_in_file": total_rows,
                    "scanned_rows": scanned_rows,
                    "null_count": null_count,
                    "non_null_count": non_null_count,
                    "null_rate_pct": null_rate,
                    "dtype": str(series.dtype),
                    "sample_limited": bool(sample_limited),
                    "scanned_at_utc": scanned_at,
                }
            )

    report_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    report_df.to_csv(output_csv, index=False)

    if missing_columns and report_df.empty:
        _write_blocked_summary(
            output_md,
            missing_files=[],
            missing_columns=missing_columns,
            note="Discovered feature files could not be read.",
        )
        return 1

    _write_ok_summary(output_md, output_csv, report_df)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/quality/feature_null_report.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/quality/feature_null_exec_summary.md"))
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional row limit per input table")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md

    rc = run_feature_null_scanner(
        repo_root=repo_root,
        data_dir=data_dir,
        output_csv=output_csv,
        output_md=output_md,
        sample_limit=args.sample_limit,
    )
    print(json.dumps({"output_csv": str(output_csv), "output_md": str(output_md), "exit_code": rc}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
