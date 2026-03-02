#!/usr/bin/env python3
"""Audit CSV quality outputs and generate markdown reports."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

TARGET_STAT_COLUMNS = [
    "ortg",
    "drtg",
    "net_rtg",
    "pace",
    "efg_pct",
    "ts_pct",
    "barthag",
    "luck_score",
]

DEFAULT_VALUE_THRESHOLDS: Dict[str, float] = {
    "barthag": 0.5,
    "ens_total": 144.2,
}
DEFAULT_MATCH_RATIO = 0.95


@dataclass
class AdvancedStatResult:
    file_name: str
    column: str
    non_null_count: int
    mean: float | None
    std: float | None
    min_value: float | None
    max_value: float | None
    unique_count: int
    dead_constant_or_default: bool
    reasons: List[str]


def _to_optional_float(value: float) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _calc_advanced_stats(df: pd.DataFrame, file_name: str) -> List[AdvancedStatResult]:
    results: List[AdvancedStatResult] = []
    candidate_columns = list(dict.fromkeys([*TARGET_STAT_COLUMNS, *DEFAULT_VALUE_THRESHOLDS.keys()]))
    columns_to_check = [c for c in candidate_columns if c in df.columns]

    for col in columns_to_check:
        numeric = pd.to_numeric(df[col], errors="coerce")
        non_null = numeric.dropna()

        non_null_count = int(non_null.shape[0])
        mean = _to_optional_float(non_null.mean()) if non_null_count else None
        std = _to_optional_float(non_null.std()) if non_null_count else None
        min_value = _to_optional_float(non_null.min()) if non_null_count else None
        max_value = _to_optional_float(non_null.max()) if non_null_count else None
        unique_count = int(non_null.nunique(dropna=True)) if non_null_count else 0

        reasons: List[str] = []
        if non_null_count > 0 and unique_count <= 1:
            reasons.append("unique_count<=1")
        if non_null_count > 0 and std == 0:
            reasons.append("std==0")

        default_value = DEFAULT_VALUE_THRESHOLDS.get(col)
        if non_null_count > 0 and default_value is not None:
            default_ratio = float((non_null == default_value).mean())
            if default_ratio >= DEFAULT_MATCH_RATIO:
                reasons.append(
                    f"default_value_ratio>={DEFAULT_MATCH_RATIO:.0%}"
                    f" (value={default_value}, ratio={default_ratio:.1%})"
                )

        results.append(
            AdvancedStatResult(
                file_name=file_name,
                column=col,
                non_null_count=non_null_count,
                mean=mean,
                std=std,
                min_value=min_value,
                max_value=max_value,
                unique_count=unique_count,
                dead_constant_or_default=bool(reasons),
                reasons=reasons,
            )
        )

    return results


def _fmt_float(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def _iter_csv_files(data_dir: Path) -> Iterable[Path]:
    return sorted(data_dir.glob("*.csv"))


def build_report(data_dir: Path) -> str:
    lines: List[str] = ["# CSV Quality Audit", ""]
    all_results: List[AdvancedStatResult] = []
    errors: List[str] = []

    for csv_path in _iter_csv_files(data_dir):
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:
            errors.append(f"- `{csv_path.name}`: failed to read ({exc})")
            continue
        all_results.extend(_calc_advanced_stats(df, csv_path.name))

    lines.append("## Advanced Stats Health")
    lines.append("")

    flagged = [r for r in all_results if r.dead_constant_or_default]
    if not flagged:
        lines.append("No advanced stat columns were flagged as dead_constant_or_default.")
    else:
        lines.append("Flagged `(file, column)` pairs:")
        lines.append("")
        lines.append("| File | Column | non_null_count | unique_count | min | max | mean | std | Reasons |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in flagged:
            lines.append(
                "| {file} | {col} | {nn} | {uc} | {minv} | {maxv} | {mean} | {std} | {reasons} |".format(
                    file=row.file_name,
                    col=row.column,
                    nn=row.non_null_count,
                    uc=row.unique_count,
                    minv=_fmt_float(row.min_value),
                    maxv=_fmt_float(row.max_value),
                    mean=_fmt_float(row.mean),
                    std=_fmt_float(row.std),
                    reasons=", ".join(row.reasons),
                )
            )

    if errors:
        lines.extend(["", "## Read Errors", "", *errors])
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
    parser = argparse.ArgumentParser(description="Audit CSV quality and stat sanity")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing CSV files")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/reports/csv_quality_audit.md"),
        help="Markdown report output path",
    )
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
    report = build_report(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[OK] wrote {args.output}")
    audits, threshold = audit_csvs(args.data_dir, args.expectations)
    markdown = render_markdown(audits, threshold)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    return 0


"""Audit CSV quality under data/ and emit markdown + CSV reports."""

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


def _likely_date_column(df: pd.DataFrame) -> str | None:
    preferred = [
        "data_date",
        "game_date",
        "date",
        "game_datetime_utc",
        "datetime",
        "created_at",
        "updated_at",
    ]
    lowered = {c.lower(): c for c in df.columns}
    for name in preferred:
        if name in lowered:
            return lowered[name]
    for col in df.columns:
        lower = col.lower()
        if "date" in lower or "time" in lower:
            return col
    return None


def _format_guess(confidence: str, message: str) -> str:
    return f"- **{confidence}**: {message}"


def _root_cause_guesses(file_path: str, df: pd.DataFrame, file_cols: pd.DataFrame) -> list[str]:
    guesses: list[str] = []
    row_count = len(df)
    col_count = len(df.columns)

    date_col = _likely_date_column(df)
    if row_count < 20 and date_col and date_col in df.columns:
        parsed = pd.to_datetime(df[date_col], errors="coerce", utc=True)
        distinct_days = parsed.dt.date.nunique(dropna=True)
        if distinct_days in (1, 2):
            guesses.append(
                _format_guess(
                    "High",
                    "Pipeline likely only ran for a couple of days or historical rows were overwritten (very low row count with 1-2 distinct dates).",
                )
            )

    core_id_cols = [
        c for c in df.columns if c.lower() in {"id", "game_id", "team_id", "home_team_id", "away_team_id", "event_id"}
    ]
    core_ids_exist = any(df[c].notna().any() for c in core_id_cols)
    if col_count > 0:
        all_null_cols = int(file_cols["all_null_flag"].sum()) if not file_cols.empty else 0
        all_null_ratio = all_null_cols / col_count
        if core_ids_exist and all_null_ratio >= 0.3:
            guesses.append(
                _format_guess(
                    "High" if all_null_ratio >= 0.5 else "Med",
                    "Join likely failed or an upstream feature build step was skipped (many columns are 100% null while core IDs are present).",
                )
            )

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        zero_all = [c for c in numeric_cols if pd.to_numeric(df[c], errors="coerce").fillna(0).eq(0).all()]
        if zero_all and date_col and date_col in df.columns:
            parsed = pd.to_datetime(df[date_col], errors="coerce", utc=True)
            valid = parsed.notna()
            if valid.any() and parsed.dt.date.nunique(dropna=True) > 1:
                latest_day = parsed[valid].dt.date.max()
                latest_mask = parsed.dt.date == latest_day
                older_mask = valid & (~latest_mask)
                regressions = 0
                for col in zero_all:
                    latest_zero = pd.to_numeric(df.loc[latest_mask, col], errors="coerce").fillna(0).eq(0).all()
                    older_non_zero = pd.to_numeric(df.loc[older_mask, col], errors="coerce").fillna(0).ne(0).any()
                    if latest_zero and older_non_zero:
                        regressions += 1
                if regressions:
                    guesses.append(
                        _format_guess(
                            "Med",
                            "`fillna(0)` may be masking missing source fields (latest rows are all-zero in numeric columns that were non-zero historically).",
                        )
                    )

    lower_name = Path(file_path).name.lower()
    if "pred" in lower_name:
        data_date_col = next((c for c in df.columns if c.lower() in {"data_date", "game_date", "date"}), None)
        pred_time_col = next(
            (c for c in df.columns if c.lower() in {"created_at", "predicted_at", "generated_at", "run_at"}),
            None,
        )
        schedule_time_col = next(
            (c for c in df.columns if c.lower() in {"schedule_updated_at", "schedule_pulled_at", "pulled_at_utc", "updated_at"}),
            None,
        )
        if data_date_col and pred_time_col and schedule_time_col:
            data_max = pd.to_datetime(df[data_date_col], errors="coerce", utc=True).max()
            pred_max = pd.to_datetime(df[pred_time_col], errors="coerce", utc=True).max()
            sched_max = pd.to_datetime(df[schedule_time_col], errors="coerce", utc=True).max()
            if pd.notna(data_max) and pd.notna(pred_max) and pd.notna(sched_max):
                stale = (data_max - pred_max).days >= 1
                schedule_fresh = abs((data_max - sched_max).days) <= 1
                if stale and schedule_fresh:
                    guesses.append(
                        _format_guess(
                            "High",
                            "Prediction action is likely not scheduled or is failing (predictions stale by data date while schedule data appears fresh).",
                        )
                    )

    if not guesses:
        guesses.append(_format_guess("Low", "No strong automated root-cause signal; inspect upstream job logs and recent pipeline changes."))

    return guesses[:3]




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
            file_df = pd.read_csv(file_path, low_memory=False)

            report_lines.append("### Explain likely root cause")
            report_lines.append("")
            report_lines.extend(_root_cause_guesses(file_path, file_df, file_cols))
            report_lines.append("")

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
