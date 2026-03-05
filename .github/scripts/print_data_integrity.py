#!/usr/bin/env python3
"""Print compact CSV integrity stats and optionally write JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DATE_COLUMNS = [
    "game_datetime_utc",
    "captured_at_utc",
    "pulled_at_utc",
    "updated_at",
    "game_date",
]

KEY_COLUMNS = ["game_id", "event_id", "team_id", "game_datetime_utc"]


def summarize_csv(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    out: dict[str, Any] = {
        "file": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }

    for col in DATE_COLUMNS:
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
            if not ts.empty:
                out["min_date_utc"] = ts.min().isoformat()
                out["max_date_utc"] = ts.max().isoformat()
            break

    null_rates: dict[str, float] = {}
    for col in KEY_COLUMNS:
        if col in df.columns:
            null_rates[col] = round(float(df[col].isna().mean()) * 100.0, 2)
    if null_rates:
        out["null_rates_pct"] = null_rates
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--label", default="integrity")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    for rel in args.files:
        if not rel.exists():
            raise SystemExit(f"[ERROR] missing integrity input: {rel}")
        summaries.append(summarize_csv(rel))

    payload = {"label": args.label, "files": summaries}

    print(f"[INTEGRITY] {args.label}")
    for item in summaries:
        line = f"- {item['file']}: rows={item['rows']}, cols={item['columns']}"
        if "min_date_utc" in item and "max_date_utc" in item:
            line += f", date_range={item['min_date_utc']}..{item['max_date_utc']}"
        if "null_rates_pct" in item:
            rates = ", ".join(f"{k}={v}%" for k, v in item["null_rates_pct"].items())
            line += f", null_rates={rates}"
        print(line)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[OK] wrote integrity report: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
