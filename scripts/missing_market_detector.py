#!/usr/bin/env python3
"""Detect rows in discovered game/prediction tables without market line coverage."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OUTPUT_COLUMNS = [
    "base_source_file",
    "market_source_file",
    "base_id_column",
    "market_id_column",
    "base_id",
    "base_row_index",
    "market_id_match",
    "missing_market",
    "reason",
    "game_datetime_utc",
    "scanned_at_utc",
]

ID_CANDIDATES = ("game_id", "event_id")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_header(path: Path) -> list[str]:
    try:
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _row_count(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
    except Exception:
        return 0


def _pick_id_column(columns: list[str]) -> str | None:
    for col in ID_CANDIDATES:
        if col in columns:
            return col
    return None


def _scan_csvs(data_dir: Path) -> list[Path]:
    files: list[Path] = []
    for folder in (data_dir, data_dir / "csv"):
        if not folder.exists():
            continue
        files.extend(sorted(folder.glob("*.csv")))
    return sorted({path.resolve() for path in files})


def _choose_market_file(files: list[Path]) -> Path | None:
    scored: list[tuple[int, int, str, Path]] = []
    for path in files:
        low = path.name.lower()
        if "market" not in low:
            continue
        cols = _read_header(path)
        id_col = _pick_id_column(cols)
        if not id_col:
            continue
        rows = _row_count(path)
        score = 2 if "market_lines.csv" == low else 1
        scored.append((score, rows, path.name, path))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][3]


def _choose_base_file(files: list[Path]) -> Path | None:
    scored: list[tuple[int, int, str, Path]] = []
    for path in files:
        low = path.name.lower()
        if "market" in low:
            continue
        cols = _read_header(path)
        id_col = _pick_id_column(cols)
        if not id_col:
            continue
        rows = _row_count(path)
        name_score = 0
        if "prediction" in low:
            name_score += 3
        if "games" in low:
            name_score += 2
        if "schedule" in low:
            name_score += 1
        scored.append((name_score, rows, path.name, path))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][3]


def _write_blocked_summary(
    output_md: Path,
    *,
    note: str,
    missing_files: list[str],
    missing_columns: dict[str, list[str]],
) -> None:
    lines = [
        "# Exec Summary: missing_market_detector",
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
    key_cols = ["base_id", "missing_market", "market_id_match"]
    rates = {}
    for col in key_cols:
        if col in report_df.columns:
            rates[col] = round(float(report_df[col].isna().mean()) * 100.0, 2)

    missing_count = int(report_df["missing_market"].sum()) if "missing_market" in report_df.columns else 0
    lines = [
        "# Exec Summary: missing_market_detector",
        "",
        "- status: `OK`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- report_csv: `{report_csv}`",
        f"- rows: `{len(report_df)}`",
        f"- columns: `{len(report_df.columns)}`",
        f"- missing_market_rows: `{missing_count}`",
    ]
    if rates:
        lines.append("- report key column null rates:")
        for col, rate in rates.items():
            lines.append(f"  - `{col}`: `{rate}%`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_missing_market_detector(
    *,
    repo_root: Path,
    data_dir: Path,
    output_csv: Path,
    output_md: Path,
    sample_limit: int | None,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    scanned_at = _utc_now()
    files = _scan_csvs(data_dir)

    base_file = _choose_base_file(files)
    market_file = _choose_market_file(files)

    if not base_file or not market_file:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        missing = []
        if not base_file:
            missing.append(str(data_dir / "*predictions*.csv"))
            missing.append(str(data_dir / "games.csv"))
        if not market_file:
            missing.append(str(data_dir / "*market*.csv"))
        _write_blocked_summary(
            output_md,
            note="Required base or market table could not be discovered.",
            missing_files=missing,
            missing_columns={},
        )
        return 1

    base_df = pd.read_csv(base_file, dtype=str, low_memory=False)
    market_df = pd.read_csv(market_file, dtype=str, low_memory=False)
    base_id_col = _pick_id_column(list(base_df.columns))
    market_id_col = _pick_id_column(list(market_df.columns))

    missing_columns: dict[str, list[str]] = {}
    if base_id_col is None:
        missing_columns[str(base_file)] = list(ID_CANDIDATES)
    if market_id_col is None:
        missing_columns[str(market_file)] = list(ID_CANDIDATES)
    if missing_columns:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            note="Discovered files are missing required id columns.",
            missing_files=[],
            missing_columns=missing_columns,
        )
        return 1

    if base_df.empty or market_df.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        missing_files = []
        if base_df.empty:
            missing_files.append(str(base_file))
        if market_df.empty:
            missing_files.append(str(market_file))
        _write_blocked_summary(
            output_md,
            note="Required base or market input has zero rows.",
            missing_files=missing_files,
            missing_columns={},
        )
        return 1

    if sample_limit is not None and sample_limit >= 0:
        base_df = base_df.head(sample_limit)

    market_ids = set(market_df[market_id_col].fillna("").astype(str).str.strip())
    rows: list[dict[str, object]] = []
    dt_col = "game_datetime_utc" if "game_datetime_utc" in base_df.columns else ""

    for idx, row in base_df.reset_index(drop=True).iterrows():
        base_id = str(row.get(base_id_col, "") or "").strip()
        if not base_id:
            reason = "missing_base_id"
            match = False
        elif base_id not in market_ids:
            reason = "no_market_match"
            match = False
        else:
            reason = "matched"
            match = True

        rows.append(
            {
                "base_source_file": str(base_file),
                "market_source_file": str(market_file),
                "base_id_column": base_id_col,
                "market_id_column": market_id_col,
                "base_id": base_id,
                "base_row_index": int(idx),
                "market_id_match": bool(match),
                "missing_market": bool(not match),
                "reason": reason,
                "game_datetime_utc": str(row.get(dt_col, "")) if dt_col else "",
                "scanned_at_utc": scanned_at,
            }
        )

    report_df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    report_df.to_csv(output_csv, index=False)
    _write_ok_summary(output_md, output_csv, report_df)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/quality/missing_market_report.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/quality/missing_market_exec_summary.md"))
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional row limit on base rows")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md

    rc = run_missing_market_detector(
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
