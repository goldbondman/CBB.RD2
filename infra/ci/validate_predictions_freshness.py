#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

GAME_TIME_COLUMNS = (
    "game_datetime_utc",
    "game_datetime",
    "start_time",
    "commence_time",
    "game_time",
    "date",
    "game_date",
)
RUN_TIMESTAMP_COLUMNS = (
    "generated_at_utc",
    "run_time_utc",
    "model_run_utc",
    "created_at_utc",
    "pipeline_run_utc",
)
PREDICTED_AT_COLUMN = "predicted_at_utc"
ARTIFACT_MARKER_FILE = ".artifact_marker.txt"


def _parse_utc(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="datetime64[ns, UTC]")
    normalized = series.astype(str).str.replace("Z", "+00:00", regex=False)
    return pd.to_datetime(normalized, errors="coerce", utc=True)


def _artifact_marker_status(data_dir: Path) -> str:
    marker_path = data_dir / ARTIFACT_MARKER_FILE
    if not marker_path.exists():
        return "not_found"
    marker = marker_path.read_text(encoding="utf-8").strip()
    return marker or "present"


def _candidate_run_timestamp(df: pd.DataFrame, path: Path) -> tuple[pd.Timestamp, str, str | None]:
    for col in RUN_TIMESTAMP_COLUMNS:
        if col in df.columns:
            parsed = _parse_utc(df[col])
            if parsed.notna().any():
                return parsed.max(), col, None

    warning = None
    if PREDICTED_AT_COLUMN in df.columns:
        warning = (
            f"[WARN] {path}: run timestamp column missing. Column '{PREDICTED_AT_COLUMN}' is present but is not used for freshness; "
            "falling back to file mtime."
        )

    return pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC"), "file_mtime", warning


def _select_source(data_dir: Path) -> tuple[Path, pd.DataFrame, pd.Timestamp, str, str | None]:
    candidates = [data_dir / "predictions_latest.csv", data_dir / "predictions_mc_latest.csv"]
    viable: list[tuple[pd.Timestamp, Path, pd.DataFrame, str, str | None]] = []

    for path in candidates:
        if not path.exists() or path.stat().st_size < 10:
            continue
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            continue

        max_ts, source, warning = _candidate_run_timestamp(df, path)
        viable.append((max_ts, path, df, source, warning))

    if not viable:
        raise RuntimeError("No predictions source found (predictions_latest.csv/predictions_mc_latest.csv missing or empty)")

    viable.sort(key=lambda row: row[0])
    selected_ts, selected_path, selected_df, timestamp_source, warning = viable[-1]
    print(f"[INFO] selected_predictions_source={selected_path}")
    print(f"[INFO] selected_predictions_timestamp_utc={selected_ts.isoformat()}")
    print(f"[INFO] selected_predictions_timestamp_source={timestamp_source}")
    if warning:
        print(warning)
    return selected_path, selected_df, selected_ts, timestamp_source, warning


def _max_game_time(df: pd.DataFrame) -> tuple[str | None, pd.Series | None, pd.Timestamp | None]:
    game_time_col = next((c for c in GAME_TIME_COLUMNS if c in df.columns), None)
    if game_time_col is None:
        return None, None, None

    parsed = _parse_utc(df[game_time_col])
    if parsed.notna().sum() == 0:
        return game_time_col, parsed, None
    return game_time_col, parsed, parsed.max()


def validate_predictions_freshness(data_dir: Path, max_age_hours: float) -> None:
    now_utc = datetime.now(timezone.utc)
    max_age = pd.Timedelta(hours=max_age_hours)
    marker_status = _artifact_marker_status(data_dir)

    source_path, source_df, max_prediction_ts, used_timestamp_col, warning = _select_source(data_dir)
    source_mtime = pd.Timestamp(source_path.stat().st_mtime, unit="s", tz="UTC")
    rowcount = len(source_df)

    age = pd.Timestamp(now_utc) - max_prediction_ts
    if age > max_age:
        game_time_col, _, max_game_time = _max_game_time(source_df)
        raise RuntimeError(
            "Predictions freshness gate failed: "
            f"source={source_path}; artifact_marker={marker_status}; timestamp_source={used_timestamp_col}; "
            f"selected_timestamp_utc={max_prediction_ts.isoformat()}; age={age}; max_allowed={max_age}; "
            f"file_mtime_utc={source_mtime.isoformat()}; rowcount={rowcount}; "
            f"max_game_time_utc={max_game_time.isoformat() if max_game_time is not None else 'n/a'}; "
            f"game_time_column={game_time_col or 'missing'}"
        )

    game_time_col, game_times, max_game_time = _max_game_time(source_df)
    if game_time_col is None:
        raise RuntimeError(f"Predictions schedule gate failed: no game time column. Expected one of {GAME_TIME_COLUMNS}")

    if max_game_time is None:
        parse_fail_ratio = 1.0
        raise RuntimeError(
            "Predictions schedule parse gate failed: "
            f"column={game_time_col}; parse_fail_ratio={parse_fail_ratio:.2%}; max_allowed=30.00%; "
            f"source={source_path}"
        )

    parse_successes = int(game_times.notna().sum())
    parse_fail_ratio = 1 - (parse_successes / max(rowcount, 1))
    if parse_fail_ratio > 0.30:
        raise RuntimeError(
            "Predictions schedule parse gate failed: "
            f"column={game_time_col}; parse_fail_ratio={parse_fail_ratio:.2%}; max_allowed=30.00%; "
            f"source={source_path}"
        )

    window_start_utc = pd.Timestamp(now_utc)
    if max_game_time < window_start_utc:
        raise RuntimeError(
            "Predictions forward-looking schedule gate failed: "
            f"max_game_time_utc={max_game_time.isoformat()}; window_start_utc={window_start_utc.isoformat()}; "
            f"source={source_path}; rowcount={rowcount}; game_time_column={game_time_col}"
        )

    print(f"[INFO] now_utc={now_utc.isoformat()}")
    print(f"[INFO] artifact_marker={marker_status}")
    print(f"[INFO] freshness_threshold_hours={max_age_hours}")
    print(f"[INFO] freshness_timestamp_source={used_timestamp_col}")
    print(f"[INFO] max_prediction_timestamp_utc={max_prediction_ts.isoformat()}")
    print(f"[INFO] source_file_mtime_utc={source_mtime.isoformat()}")
    print(f"[INFO] source_rowcount={rowcount}")
    print(f"[INFO] game_time_column={game_time_col}")
    print(f"[INFO] game_time_parse_fail_ratio={parse_fail_ratio:.2%}")
    print(f"[INFO] max_game_time_utc={max_game_time.isoformat()}")
    if warning:
        print(warning)


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
