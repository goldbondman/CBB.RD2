#!/usr/bin/env python3
"""Audit CSV row counts and render a markdown quality report.

Pattern matching for expectation rules uses shell-style globs (fnmatch),
matched against repo-relative POSIX paths (for example, ``data/*.csv``).
The first matching rule wins.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - fallback when yaml isn't installed
    yaml = None


DEFAULT_SMALL_ROW_THRESHOLD = 50
DEFAULT_EXPECTATIONS_PATH = Path("data/dq_expectations.yml")


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def parse_expectations_fallback(path: Path) -> dict:
    """Parse a minimal subset of YAML used by dq_expectations.yml."""
    raw = path.read_text(encoding="utf-8")
    out: dict = {"defaults": {}, "files": []}

    threshold_match = re.search(r"^\s*small_row_threshold:\s*(\d+)\s*$", raw, flags=re.MULTILINE)
    if threshold_match:
        out["defaults"]["small_row_threshold"] = int(threshold_match.group(1))

    file_chunks = re.split(r"^\s*-\s+pattern:\s*", raw, flags=re.MULTILINE)[1:]
    for chunk in file_chunks:
        lines = chunk.splitlines()
        if not lines:
            continue
        pattern = _strip_quotes(lines[0])
        min_rows_match = re.search(r"^\s*min_rows:\s*(-?\d+)\s*$", chunk, flags=re.MULTILINE)
        note_match = re.search(r"^\s*note:\s*(.+?)\s*$", chunk, flags=re.MULTILINE)

        expected: dict = {}
        if min_rows_match:
            expected["min_rows"] = int(min_rows_match.group(1))
        if note_match:
            expected["note"] = _strip_quotes(note_match.group(1))

        out["files"].append({"pattern": pattern, "expected": expected})

    return out


@dataclass
class ExpectationRule:
    pattern: str
    min_rows: int | None
    note: str | None


@dataclass
class FileAudit:
    path: str
    row_count: int
    suspicious_small_rows: bool
    expected_min_rows: int | None
    below_expected_min_rows: bool
    expectation_note: str | None
    small_rows_status: str


def load_expectations(path: Path) -> tuple[int, list[ExpectationRule]]:
    """Return (small_row_threshold, ordered_expectation_rules)."""
    if not path.exists() or path.stat().st_size == 0:
        return DEFAULT_SMALL_ROW_THRESHOLD, []

    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = parse_expectations_fallback(path)

    defaults = raw.get("defaults") if isinstance(raw, dict) else {}
    small_row_threshold = DEFAULT_SMALL_ROW_THRESHOLD
    if isinstance(defaults, dict):
        value = defaults.get("small_row_threshold")
        if isinstance(value, int):
            small_row_threshold = value

    rules: list[ExpectationRule] = []
    files = raw.get("files") if isinstance(raw, dict) else []
    if not isinstance(files, list):
        return small_row_threshold, rules

    for entry in files:
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("pattern")
        expected = entry.get("expected")
        if not isinstance(pattern, str):
            continue
        min_rows = None
        note = None
        if isinstance(expected, dict):
            min_value = expected.get("min_rows")
            if isinstance(min_value, int):
                min_rows = min_value
            note_value = expected.get("note")
            if isinstance(note_value, str):
                note = note_value
        rules.append(ExpectationRule(pattern=pattern, min_rows=min_rows, note=note))

    return small_row_threshold, rules


def first_matching_rule(path: str, rules: list[ExpectationRule]) -> ExpectationRule | None:
    for rule in rules:
        if fnmatch(path, rule.pattern):
            return rule
    return None


def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(pd.read_csv(path, low_memory=False))
    except Exception:
        return 0


def audit_csvs(data_dir: Path, expectations_path: Path) -> tuple[list[FileAudit], int]:
    small_row_threshold, rules = load_expectations(expectations_path)
    audits: list[FileAudit] = []

    for csv_path in sorted(data_dir.rglob("*.csv")):
        rel_path = csv_path.as_posix()
        row_count = count_rows(csv_path)
        suspicious_small_rows = row_count < small_row_threshold

        matched = first_matching_rule(rel_path, rules)
        expected_min_rows = matched.min_rows if matched else None
        note = matched.note if matched else None
        below_expected_min_rows = (
            expected_min_rows is not None and row_count < expected_min_rows
        )

        small_rows_status = "suspicious"
        if suspicious_small_rows:
            if expected_min_rows is not None and row_count >= expected_min_rows:
                small_rows_status = "small-but-expected"
        else:
            small_rows_status = "ok"

        audits.append(
            FileAudit(
                path=rel_path,
                row_count=row_count,
                suspicious_small_rows=suspicious_small_rows,
                expected_min_rows=expected_min_rows,
                below_expected_min_rows=below_expected_min_rows,
                expectation_note=note,
                small_rows_status=small_rows_status,
            )
        )

    return audits, small_row_threshold


def render_markdown(audits: list[FileAudit], small_row_threshold: int) -> str:
    lines = [
        "# CSV Quality Audit",
        "",
        f"- Small row threshold: `{small_row_threshold}`",
        "- Expectation pattern matching: first-match-wins shell-style glob (`fnmatch`) on repo-relative POSIX file paths.",
        "",
        "## File Summary",
        "",
        "| File | Row Count | suspicious_small_rows | expected_min_rows | below_expected_min_rows | small_rows_status | expectation_note |",
        "|---|---:|---|---:|---|---|---|",
    ]

    for item in audits:
        expected = "" if item.expected_min_rows is None else str(item.expected_min_rows)
        note = "" if item.expectation_note is None else item.expectation_note.replace("|", "\\|")
        lines.append(
            f"| {item.path} | {item.row_count} | {str(item.suspicious_small_rows).lower()} | {expected} | {str(item.below_expected_min_rows).lower()} | {item.small_rows_status} | {note} |"
        )

    below_expected = [a for a in audits if a.below_expected_min_rows]
    no_rule = [a for a in audits if a.expected_min_rows is None]

    lines.extend([
        "",
        "## Row Count Expectations",
        "",
        "### Files below expected minimum",
    ])

    if below_expected:
        for item in below_expected:
            lines.append(
                f"- `{item.path}`: {item.row_count} rows (expected min {item.expected_min_rows})"
            )
    else:
        lines.append("- None")

    lines.extend(["", "### Files with no expectation rule"])
    if no_rule:
        for item in no_rule:
            lines.append(f"- `{item.path}`")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to recursively scan for CSV files.",
    )
    parser.add_argument(
        "--expectations",
        type=Path,
        default=DEFAULT_EXPECTATIONS_PATH,
        help="Path to row-count expectations YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output markdown path. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audits, threshold = audit_csvs(args.data_dir, args.expectations)
    markdown = render_markdown(audits, threshold)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
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
