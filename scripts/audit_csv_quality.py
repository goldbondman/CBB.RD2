#!/usr/bin/env python3
"""Audit CSV quality under data/ and emit markdown + CSV reports."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path("data")
OUTPUT_MD = DATA_DIR / "dq_report_v2.md"
OUTPUT_CSV = DATA_DIR / "dq_report_v2.csv"
SKIP_PATH_TOKENS = ("archive", "backups", "old", "tmp")
MAX_FILE_SIZE_BYTES = 250 * 1024 * 1024
DEFAULT_SMALL_ROW_THRESHOLD = 50
PROGRESS_EVERY = 25


def _safe_pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round((numerator / denominator) * 100.0, 2)


def _is_numeric_series(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)


def _stringify_constant(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)


def _iter_csv_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.csv"):
        normalized = str(path).lower()
        if any(token in normalized for token in SKIP_PATH_TOKENS):
            continue
        files.append(path)
    return sorted(files)




def _df_to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "None"
    headers = [str(col) for col in df.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in df.itertuples(index=False, name=None):
        vals = []
        for value in row:
            text = "" if pd.isna(value) else str(value)
            vals.append(text.replace("|", "\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)

def audit_csvs() -> None:
    small_row_threshold = int(os.getenv("SMALL_ROW_THRESHOLD", str(DEFAULT_SMALL_ROW_THRESHOLD)))

    csv_files = _iter_csv_files(DATA_DIR)

    skipped_large: list[tuple[str, int]] = []
    failed_files: list[tuple[str, str]] = []

    summary_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []

    processed_count = 0
    for idx, file_path in enumerate(csv_files, start=1):
        try:
            file_size = file_path.stat().st_size
        except OSError as exc:
            failed_files.append((str(file_path), f"stat error: {exc}"))
            continue

        if file_size > MAX_FILE_SIZE_BYTES:
            skipped_large.append((str(file_path), file_size))
            print(f"[SKIP] Large file ({round(file_size / (1024 * 1024), 2)} MB): {file_path}")
            continue

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as exc:  # noqa: BLE001 - intentional broad capture for report continuity
            failed_files.append((str(file_path), str(exc)))
            continue

        row_count = len(df)
        col_count = len(df.columns)

        cols_with_nulls = 0
        all_null_cols = 0
        all_zero_cols = 0
        constant_cols = 0

        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            null_count = int(series.isna().sum())
            null_pct = _safe_pct(null_count, row_count)
            unique_count = int(series.nunique(dropna=True))

            constant_flag = bool(unique_count == 1 and row_count > 0)
            constant_value = ""
            if constant_flag:
                non_null_values = series.dropna()
                if not non_null_values.empty:
                    constant_value = _stringify_constant(non_null_values.iloc[0])

            all_null_flag = bool(row_count > 0 and null_count == row_count)

            zero_count = 0
            zero_pct = 0.0
            all_zero_flag = False
            if _is_numeric_series(series):
                numeric_series = pd.to_numeric(series, errors="coerce")
                zero_count = int(numeric_series.eq(0).sum())
                zero_pct = _safe_pct(zero_count, row_count)
                all_zero_flag = bool(row_count > 0 and zero_count == row_count)

            if null_count > 0:
                cols_with_nulls += 1
            if all_null_flag:
                all_null_cols += 1
            if all_zero_flag:
                all_zero_cols += 1
            if constant_flag:
                constant_cols += 1

            column_rows.append(
                {
                    "file_path": str(file_path),
                    "row_count": row_count,
                    "col_count": col_count,
                    "column": col,
                    "dtype": dtype,
                    "null_count": null_count,
                    "null_pct": null_pct,
                    "zero_count": zero_count,
                    "zero_pct": zero_pct,
                    "unique_count": unique_count,
                    "constant_flag": constant_flag,
                    "constant_value": constant_value,
                    "all_null_flag": all_null_flag,
                    "all_zero_flag": all_zero_flag,
                }
            )

        suspicious_small_rows = row_count < small_row_threshold
        suspicious_constant_columns = bool(row_count > 0 and col_count > 0 and (constant_cols / col_count) >= 0.30)

        summary_rows.append(
            {
                "file_path": str(file_path),
                "row_count": row_count,
                "col_count": col_count,
                "cols_with_nulls": cols_with_nulls,
                "all_null_cols": all_null_cols,
                "all_zero_cols": all_zero_cols,
                "constant_cols": constant_cols,
                "small_rows_flag": suspicious_small_rows,
                "suspicious_constant_columns": suspicious_constant_columns,
            }
        )

        processed_count += 1
        if idx % PROGRESS_EVERY == 0:
            print(f"Processed {idx}/{len(csv_files)} discovered files...")

    col_df = pd.DataFrame(column_rows)
    summary_df = pd.DataFrame(summary_rows)

    if not col_df.empty:
        col_df.to_csv(OUTPUT_CSV, index=False)
    else:
        pd.DataFrame(
            columns=[
                "file_path",
                "row_count",
                "col_count",
                "column",
                "dtype",
                "null_count",
                "null_pct",
                "zero_count",
                "zero_pct",
                "unique_count",
                "constant_flag",
                "constant_value",
                "all_null_flag",
                "all_zero_flag",
            ]
        ).to_csv(OUTPUT_CSV, index=False)

    report_lines: list[str] = []
    report_lines.append("# CSV Data Quality Report v2")
    report_lines.append("")
    report_lines.append(f"- Scan root: `{DATA_DIR}`")
    report_lines.append(f"- Small row threshold: `{small_row_threshold}`")
    report_lines.append(f"- Discovered CSV files: `{len(csv_files)}`")
    report_lines.append(f"- Processed CSV files: `{processed_count}`")
    report_lines.append(f"- Skipped large files (>250MB): `{len(skipped_large)}`")
    report_lines.append(f"- Failed CSV reads: `{len(failed_files)}`")
    report_lines.append("")

    report_lines.append("## Summary Table")
    report_lines.append("")

    if not summary_df.empty:
        summary_md = summary_df[
            [
                "file_path",
                "row_count",
                "col_count",
                "cols_with_nulls",
                "all_null_cols",
                "all_zero_cols",
                "constant_cols",
                "small_rows_flag",
            ]
        ].sort_values("file_path")
        report_lines.append(_df_to_markdown_table(summary_md))
    else:
        report_lines.append("No CSV files were processed.")
    report_lines.append("")

    report_lines.append("## Top Issues")
    report_lines.append("")

    if not summary_df.empty:
        small_rows_files = summary_df[summary_df["small_rows_flag"]]
        many_all_null = summary_df[summary_df["all_null_cols"] > 0].sort_values("all_null_cols", ascending=False).head(20)
        many_constant = summary_df[summary_df["constant_cols"] > 0].sort_values("constant_cols", ascending=False).head(20)
        many_all_zero = summary_df[summary_df["all_zero_cols"] > 0].sort_values("all_zero_cols", ascending=False).head(20)

        report_lines.append(f"### Files with row_count < {small_row_threshold}")
        report_lines.append("")
        report_lines.append(_df_to_markdown_table(small_rows_files[["file_path", "row_count"]]))
        report_lines.append("")

        report_lines.append("### Files with many all-null columns")
        report_lines.append("")
        report_lines.append(_df_to_markdown_table(many_all_null[["file_path", "all_null_cols", "col_count"]]))
        report_lines.append("")

        report_lines.append("### Files with many constant columns")
        report_lines.append("")
        report_lines.append(_df_to_markdown_table(many_constant[["file_path", "constant_cols", "col_count"]]))
        report_lines.append("")

        report_lines.append("### Files with many all-zero columns")
        report_lines.append("")
        report_lines.append(_df_to_markdown_table(many_all_zero[["file_path", "all_zero_cols", "col_count"]]))
        report_lines.append("")

        flagged_files = summary_df[
            (summary_df["small_rows_flag"])
            | (summary_df["all_null_cols"] > 0)
            | (summary_df["all_zero_cols"] > 0)
            | (summary_df["constant_cols"] > 0)
        ]["file_path"].tolist()

        for file_path in flagged_files:
            report_lines.append(f"## Column Detail: `{file_path}`")
            report_lines.append("")
            file_cols = col_df[col_df["file_path"] == file_path]

            null_top = file_cols.sort_values("null_pct", ascending=False).head(25)
            report_lines.append("### Top 25 columns by null_pct")
            report_lines.append("")
            report_lines.append(
                _df_to_markdown_table(null_top[["column", "dtype", "null_count", "null_pct", "all_null_flag"]])
            )
            report_lines.append("")

            zero_top = file_cols[file_cols["zero_count"] > 0].sort_values("zero_pct", ascending=False).head(25)
            report_lines.append("### Top 25 columns by zero_pct (numeric)")
            report_lines.append("")
            report_lines.append(
                _df_to_markdown_table(zero_top[["column", "dtype", "zero_count", "zero_pct", "all_zero_flag"]])
            )
            report_lines.append("")

            constant_cols_df = file_cols[file_cols["constant_flag"]].head(50)
            report_lines.append("### Constant columns (max 50)")
            report_lines.append("")
            report_lines.append(
                _df_to_markdown_table(constant_cols_df[["column", "dtype", "constant_value"]])
            )
            report_lines.append("")

    if skipped_large:
        report_lines.append("## Skipped Large Files")
        report_lines.append("")
        report_lines.append("| file_path | size_mb |")
        report_lines.append("|---|---:|")
        for file_path, size in skipped_large:
            report_lines.append(f"| `{file_path}` | {round(size / (1024 * 1024), 2)} |")
        report_lines.append("")

    if failed_files:
        report_lines.append("## Failed CSV Reads")
        report_lines.append("")
        report_lines.append("| file_path | error |")
        report_lines.append("|---|---|")
        for file_path, error in failed_files:
            report_lines.append(f"| `{file_path}` | `{error}` |")
        report_lines.append("")

    OUTPUT_MD.write_text("\n".join(report_lines), encoding="utf-8")

    print(
        "Final summary: "
        f"discovered={len(csv_files)}, processed={processed_count}, "
        f"skipped_large={len(skipped_large)}, failed={len(failed_files)}"
    )
    print(f"Wrote markdown report: {OUTPUT_MD}")
    print(f"Wrote CSV report: {OUTPUT_CSV}")


if __name__ == "__main__":
    audit_csvs()
