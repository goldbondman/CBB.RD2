#!/usr/bin/env python3
"""Fast CSV quality audit with advanced stat sanity checks."""

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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"[OK] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
