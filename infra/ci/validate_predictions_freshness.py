#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GAME_TIME_COLUMNS = ("game_datetime_utc", "game_datetime", "start_time", "commence_time", "game_time", "date", "game_date")
TIMESTAMP_COLUMNS = ("generated_at_utc", "generated_at", "predicted_at_utc", "created_at")


def _parse_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _select_source(data_dir: Path) -> tuple[Path, pd.DataFrame]:
    candidates = [data_dir / "predictions_latest.csv", data_dir / "predictions_mc_latest.csv"]
    viable: list[tuple[pd.Timestamp, Path, pd.DataFrame]] = []

    for path in candidates:
        if not path.exists() or path.stat().st_size < 10:
            continue
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue

        max_ts = pd.NaT
        for col in TIMESTAMP_COLUMNS:
            if col in df.columns:
                parsed = _parse_utc(df[col])
                if parsed.notna().any():
                    max_ts = parsed.max()
                    break
        if pd.isna(max_ts):
            max_ts = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC")

        viable.append((max_ts, path, df))

    if not viable:
        raise RuntimeError("No predictions source found (predictions_latest.csv/predictions_mc_latest.csv missing or empty)")

    viable.sort(key=lambda row: row[0])
    selected_ts, selected_path, selected_df = viable[-1]
    print(f"[INFO] selected_predictions_source={selected_path}")
    print(f"[INFO] selected_predictions_timestamp_utc={selected_ts.isoformat()}")
    return selected_path, selected_df


def validate_predictions_freshness(data_dir: Path, max_age_hours: float) -> None:
    now_utc = datetime.now(timezone.utc)
    max_age = pd.Timedelta(hours=max_age_hours)
    source_path, source_df = _select_source(data_dir)

    max_prediction_ts = pd.NaT
    used_timestamp_col = None
    for col in TIMESTAMP_COLUMNS:
        if col in source_df.columns:
            parsed = _parse_utc(source_df[col])
            if parsed.notna().any():
                max_prediction_ts = parsed.max()
                used_timestamp_col = col
                break
    if pd.isna(max_prediction_ts):
        max_prediction_ts = pd.Timestamp(source_path.stat().st_mtime, unit="s", tz="UTC")
        used_timestamp_col = "file_mtime"

    age = pd.Timestamp(now_utc) - max_prediction_ts
    if age > max_age:
        raise RuntimeError(
            f"Predictions artifact is stale: age={age} max_allowed={max_age}; "
            f"source={source_path} timestamp_col={used_timestamp_col}"
        )

    game_time_col = next((c for c in GAME_TIME_COLUMNS if c in source_df.columns), None)
    if game_time_col is None:
        raise RuntimeError(f"Predictions file has no game time column. Expected one of {GAME_TIME_COLUMNS}")

    game_times = _parse_utc(source_df[game_time_col])
    if game_times.notna().sum() == 0:
        raise RuntimeError(f"Predictions file has no parseable game times in column '{game_time_col}'")

    max_game_time = game_times.max()
    if max_game_time < pd.Timestamp(now_utc):
        raise RuntimeError(
            f"Predictions schedule is stale: max_game_time_utc={max_game_time.isoformat()} now_utc={now_utc.isoformat()}"
        )

    print(f"[INFO] now_utc={now_utc.isoformat()}")
    print(f"[INFO] freshness_threshold_hours={max_age_hours}")
    print(f"[INFO] freshness_timestamp_col={used_timestamp_col}")
    print(f"[INFO] max_prediction_timestamp_utc={max_prediction_ts.isoformat()}")
    print(f"[INFO] game_time_column={game_time_col}")
    print(f"[INFO] max_game_time_utc={max_game_time.isoformat()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate predictions artifact freshness and forward-looking schedule")
    parser.add_argument("--data-dir", default="data", help="Directory containing predictions files")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=float(os.getenv("PREDICTIONS_FRESHNESS_MAX_HOURS", "6")),
        help="Maximum allowed age for predictions artifact",
    )
    args = parser.parse_args()
    validate_predictions_freshness(Path(args.data_dir), args.max_age_hours)


if __name__ == "__main__":
    main()
