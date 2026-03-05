#!/usr/bin/env python3
"""Local smoke check for analytics input readiness."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def check_csv(path: Path, min_rows: int, required_cols: list[str]) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"{path} is missing"]

    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return [f"{path} is empty"]
        rows = sum(1 for _ in reader)

    missing_cols = [col for col in required_cols if col not in header]
    if missing_cols:
        errors.append(f"{path} missing columns: {missing_cols}")
    if rows < min_rows:
        errors.append(f"{path} has {rows} rows; expected >= {min_rows}")
    if not errors:
        print(f"[OK] {path} rows={rows} cols={len(header)}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    checks: list[tuple[Path, int, list[str]]] = [
        (data_dir / "games.csv", 1, ["game_id"]),
        (data_dir / "team_game_weighted.csv", 1, ["team_id"]),
        (data_dir / "market_lines.csv", 1, ["event_id"]),
    ]

    errors: list[str] = []
    for path, min_rows, required_cols in checks:
        errors.extend(check_csv(path, min_rows, required_cols))

    primary_predictions = data_dir / "predictions_with_context.csv"
    fallback_predictions = data_dir / "predictions_combined_latest.csv"
    selected_predictions = primary_predictions if primary_predictions.exists() else fallback_predictions
    errors.extend(check_csv(selected_predictions, 0, ["game_id"]))

    if errors:
        print("[BLOCKED] analytics smoke checks failed:")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("[OK] analytics smoke checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
