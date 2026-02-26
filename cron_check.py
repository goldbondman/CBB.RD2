#!/usr/bin/env python3
"""Manual status report for the three-tier cron pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from espn_config import SEASON_ACTIVE

PIPELINE_LOG = Path("data/pipeline_errors.log")
GRADED_PATH = Path("data/predictions_graded.csv")
WEIGHTS_PATH = Path("data/active_weights.json")
DAILY_STEPS = [
    "espn_pipeline.py",
    "cbb_market_lines.py --mode closing",
    "espn_prediction_runner.py",
    "cbb_results_tracker.py",
    "cbb_grade_predictions.py",
    "cbb_accuracy_report.py",
    "cbb_season_summaries.py",
]


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _load_log() -> pd.DataFrame:
    if not PIPELINE_LOG.exists() or PIPELINE_LOG.stat().st_size == 0:
        return pd.DataFrame(columns=["timestamp", "tier", "step", "status", "detail"])

    rows = []
    for line in PIPELINE_LOG.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t", 4)
        if len(parts) < 5:
            continue
        rows.append(parts)

    return pd.DataFrame(rows, columns=["timestamp", "tier", "step", "status", "detail"])


def _load_active_weights() -> dict:
    if not WEIGHTS_PATH.exists():
        return {}
    try:
        return json.loads(WEIGHTS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _graded_since(graded: pd.DataFrame, deployed_at: str) -> int:
    if graded.empty or "game_datetime_utc" not in graded.columns:
        return 0

    g = graded.copy()
    g["game_datetime_utc"] = pd.to_datetime(g["game_datetime_utc"], utc=True, errors="coerce")
    cutoff = pd.to_datetime(deployed_at, utc=True, errors="coerce")
    if pd.isna(cutoff):
        cutoff = pd.Timestamp("2000-01-01", tz="UTC")
    return int(g[g["game_datetime_utc"] > cutoff].shape[0])


def main() -> None:
    now = datetime.now(timezone.utc)
    logs = _load_log()

    print("=== CBB Cron Pipeline Check ===")
    print(f"Generated at (UTC): {now.isoformat()}")
    print()

    print("Last successful DAILY run per step:")
    if logs.empty:
        print("  No pipeline log entries found.")
    else:
        for step in DAILY_STEPS:
            mask = (
                (logs["tier"] == "DAILY")
                & (logs["step"] == step)
                & (logs["status"] == "SUCCESS")
            )
            subset = logs.loc[mask]
            if subset.empty:
                print(f"  - {step}: never")
                continue
            ts = subset.iloc[-1]["timestamp"]
            print(f"  - {step}: {ts}")

    print()
    weights = _load_active_weights()
    deployed_at = str(weights.get("deployed_at", "2000-01-01"))
    improvement = weights.get("improvement_pp", "n/a")
    print("Current active_weights.json:")
    print(f"  deployed_at: {deployed_at}")
    print(f"  improvement_pp: {improvement}")

    key_names = [
        "market_weight",
        "recent_form_weight",
        "efficiency_weight",
        "home_court_advantage",
        "injury_weight",
    ]
    printed = False
    for key in key_names:
        if key in weights:
            print(f"  {key}: {weights[key]}")
            printed = True

    if not printed:
        numeric_items = [
            (k, v) for k, v in weights.items() if isinstance(v, (int, float)) and k != "improvement_pp"
        ]
        for key, value in numeric_items[:5]:
            print(f"  {key}: {value}")

    deployed_dt = _parse_ts(deployed_at) or datetime(2000, 1, 1, tzinfo=timezone.utc)
    days_since = (now - deployed_dt).days
    print()
    print(f"Days since last weight update: {days_since}")

    graded = pd.read_csv(GRADED_PATH) if GRADED_PATH.exists() else pd.DataFrame()
    graded_since = _graded_since(graded, deployed_at)
    print(f"Graded games since last weight update: {graded_since}")
    print(f"Biweekly 200-game threshold met: {'Y' if graded_since >= 200 else 'N'} ({graded_since}/200)")

    print(f"SEASON_ACTIVE: {SEASON_ACTIVE}")


if __name__ == "__main__":
    main()
