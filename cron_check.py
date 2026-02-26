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
DATA_DIR = Path("data")
STEP_RUNS_PATH = DATA_DIR / "pipeline_step_runs.csv"
ACTIVE_WEIGHTS_PATH = DATA_DIR / "active_weights.json"
PREDICTIONS_GRADED_PATH = DATA_DIR / "predictions_graded.csv"


def _to_utc_timestamp(value: object) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def _format_ts(ts: pd.Timestamp | None) -> str:
    if ts is None:
        return "n/a"
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_last_success_per_step() -> dict[str, str]:
    if not STEP_RUNS_PATH.exists():
        return {}

    frame = pd.read_csv(STEP_RUNS_PATH)
    required = {"step", "status", "run_timestamp_utc"}
    if not required.issubset(frame.columns):
        return {}

    frame = frame[frame["status"].astype(str).str.lower() == "success"].copy()
    if frame.empty:
        return {}

    frame["run_ts"] = pd.to_datetime(frame["run_timestamp_utc"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["run_ts"]).sort_values("run_ts")
    latest = frame.groupby("step", as_index=False).tail(1)
    return {row["step"]: _format_ts(row["run_ts"]) for _, row in latest.iterrows()}


def _load_active_weights_summary() -> tuple[str, pd.Timestamp | None]:
    if not ACTIVE_WEIGHTS_PATH.exists():
        return "active_weights.json not found", None

    payload = json.loads(ACTIVE_WEIGHTS_PATH.read_text())
    summary_parts = []
    for key in ["model_version", "strategy", "notes"]:
        if key in payload and payload[key] not in (None, ""):
            summary_parts.append(f"{key}={payload[key]}")

    if "weights" in payload and isinstance(payload["weights"], dict):
        summary_parts.append(f"weights_keys={len(payload['weights'])}")

    last_update_raw = (
        payload.get("last_weight_update")
        or payload.get("updated_at")
        or payload.get("deployed_at")
    )
    last_update = _to_utc_timestamp(last_update_raw)

    if last_update is not None:
        summary_parts.append(f"last_weight_update={_format_ts(last_update)}")
    elif last_update_raw is not None:
        summary_parts.append(f"last_weight_update=invalid({last_update_raw})")
    else:
        summary_parts.append("last_weight_update=missing")

    return ", ".join(summary_parts), last_update


def _graded_since(last_update: pd.Timestamp | None) -> int:
    if last_update is None or not PREDICTIONS_GRADED_PATH.exists():
        return 0

    frame = pd.read_csv(PREDICTIONS_GRADED_PATH)
    required = {"graded", "game_datetime_utc"}
    if not required.issubset(frame.columns):
        return 0

    frame["graded_bool"] = frame["graded"].astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])
    frame["game_datetime_utc"] = pd.to_datetime(frame["game_datetime_utc"], utc=True, errors="coerce")
    return int(((frame["graded_bool"]) & (frame["game_datetime_utc"] > last_update)).sum())


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
    print(f"cron_check generated_at_utc: {now.strftime('%Y-%m-%dT%H:%M:%SZ')}")

    print("\nLast successful run time per step:")
    last_success = _load_last_success_per_step()
    if not last_success:
        print("  - No successful step history found")
    else:
        for step in sorted(last_success):
            print(f"  - {step}: {last_success[step]}")

    summary, last_update = _load_active_weights_summary()
    print("\nCurrent active_weights.json summary:")
    print(f"  - {summary}")

    if last_update is None:
        days_since = "n/a"
    else:
        days_since = str((now - last_update.to_pydatetime()).days)

    graded_count = _graded_since(last_update)
    threshold_met = graded_count >= 200

    print("\nDays since last weight update:")
    print(f"  - {days_since}")

    print("\nGraded games since last weight update:")
    print(f"  - {graded_count}")

    print("\nBiweekly threshold status (200 graded games):")
    print(f"  - {'met' if threshold_met else 'not met'}")


if __name__ == "__main__":
    main()
