#!/usr/bin/env python3
"""Write a compact execution summary markdown from a CSV dataframe."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _null_rates(df: pd.DataFrame, key_cols: list[str]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for col in key_cols:
        if col in df.columns:
            rates[col] = round(float(df[col].isna().mean()) * 100.0, 2)
    return rates


def write_exec_summary(
    df: pd.DataFrame,
    output_md: Path,
    *,
    module_name: str,
    source_csv: Path,
    key_cols: list[str] | None = None,
) -> Path:
    keys = key_cols or []
    rates = _null_rates(df, keys)

    lines = [
        f"# Exec Summary: {module_name}",
        "",
        f"- source_csv: `{source_csv}`",
        f"- rows: `{len(df)}`",
        f"- columns: `{len(df.columns)}`",
    ]

    if rates:
        lines.append("- key column null rates:")
        for col, rate in rates.items():
            lines.append(f"  - `{col}`: `{rate}%`")
    else:
        lines.append("- key column null rates: `n/a`")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_md


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--module-name", required=True, type=str)
    parser.add_argument("--key-cols", default="", type=str, help="Comma-separated key columns")
    args = parser.parse_args()

    input_csv = args.input_csv
    if not input_csv.exists():
        raise SystemExit(f"[ERROR] input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv, low_memory=False)
    key_cols = [c.strip() for c in args.key_cols.split(",") if c.strip()]
    out = write_exec_summary(
        df,
        args.output_md,
        module_name=args.module_name,
        source_csv=input_csv,
        key_cols=key_cols,
    )
    print(f"[OK] wrote exec summary: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
