#!/usr/bin/env python3
"""Standalone runner for player matchup overlay predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.player_matchup_overlay import build_player_overlay_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run standalone player matchup overlay model")
    parser.add_argument(
        "--input",
        default="data/player_overlay_input.csv",
        help="Input team-perspective CSV with opponent-prefixed rotation columns",
    )
    parser.add_argument(
        "--output",
        default="data/player_overlay_predictions.csv",
        help="Output CSV path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Overlay input not found: {input_path}")

    df = pd.read_csv(input_path, low_memory=False)
    preds = build_player_overlay_predictions(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output_path, index=False)
    print(f"[OK] wrote {len(preds)} rows -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
