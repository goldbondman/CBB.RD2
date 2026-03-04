#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

TIMESTAMP_COLUMNS = ["generated_at_utc", "run_time_utc", "model_run_utc", "generated_at", "predicted_at_utc", "timestamp"]
GAME_TIME_COLUMNS = [
    "game_time_utc",
    "game_datetime_utc",
    "commence_time",
    "start_time",
    "date",
    "game_time",
    "start_time_utc",
    "commence_time_utc",
    "scheduled_utc",
    "scheduled",
    "tipoff_utc",
    "tipoff",
]


@dataclass
class ValidationSummary:
    selected_file: str
    total_rows: int
    timestamp_source_used: str
    generated_at_utc: datetime
    freshness_age_hours: float
    time_column_used: str
    non_null_rate: float
    parseable_rate: float
    parse_fail_rate: float
    window_start_utc: datetime
    max_game_time_utc: Optional[datetime]
    result: str


def _load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size < 10:
        return None
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return None
    return None if df.empty else df


def _fallback_select_prediction_source(
    predictions_latest_path: Path,
    predictions_mc_latest_path: Path,
) -> tuple[pd.DataFrame, str, Path]:
    candidates: list[tuple[str, Path]] = [
        ("predictions_latest", predictions_latest_path),
        ("predictions_mc_latest", predictions_mc_latest_path),
    ]
    best_df: Optional[pd.DataFrame] = None
    best_label: Optional[str] = None
    best_path: Optional[Path] = None
    best_ts: Optional[pd.Timestamp] = None

    for label, path in candidates:
        df = _load_csv(path)
        if df is None:
            continue

        max_ts: Optional[pd.Timestamp] = None
        for col in ["game_time_utc", "game_datetime_utc", "generated_at_utc", "generated_at"]:
            if col in df.columns:
                ts = pd.to_datetime(df[col], errors="coerce", utc=True)
                if ts.notna().any():
                    max_ts = ts.max()
                    break

        if best_df is None:
            best_df, best_label, best_path, best_ts = df, label, path, max_ts
            continue
        if best_ts is None and max_ts is not None:
            best_df, best_label, best_path, best_ts = df, label, path, max_ts
            continue
        if max_ts is not None and best_ts is not None and max_ts > best_ts:
            best_df, best_label, best_path, best_ts = df, label, path, max_ts
            continue
        if max_ts == best_ts and len(df) > len(best_df):
            best_df, best_label, best_path, best_ts = df, label, path, max_ts

    if best_df is None or best_label is None or best_path is None:
        raise RuntimeError("No valid predictions source found in data/predictions_latest.csv or data/predictions_mc_latest.csv")
    return best_df, best_label, best_path


def select_prediction_source(
    predictions_latest_path: Path = Path("data/predictions_latest.csv"),
    predictions_mc_latest_path: Path = Path("data/predictions_mc_latest.csv"),
) -> tuple[pd.DataFrame, str, Path]:
    label_to_path = {
        "predictions_latest": predictions_latest_path,
        "predictions_mc_latest": predictions_mc_latest_path,
    }
    try:
        from build_derived_csvs import _select_prediction_source

        selected_df, selected_label = _select_prediction_source()
        if selected_df is None or selected_label is None:
            raise RuntimeError("build_derived_csvs selector returned no usable predictions source")
        selected_path = label_to_path.get(selected_label)
        if selected_path is None:
            raise RuntimeError(f"Unknown prediction source label from build_derived_csvs selector: {selected_label}")
        return selected_df, selected_label, selected_path
    except Exception:
        return _fallback_select_prediction_source(predictions_latest_path, predictions_mc_latest_path)


def _parse_generated_at(df: pd.DataFrame) -> tuple[datetime, str]:
    for col in TIMESTAMP_COLUMNS:
        if col not in df.columns:
            continue
        parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
        if parsed.notna().any():
            return parsed.max().to_pydatetime().astimezone(timezone.utc), col

    selected_df_path = df.attrs.get("_selected_file_path")
    if not selected_df_path:
        raise RuntimeError("Internal error: selected file path unavailable for mtime fallback")
    print("[WARN] No generated-at column found; falling back to file mtime (UTC).")
    mtime = datetime.fromtimestamp(Path(selected_df_path).stat().st_mtime, tz=timezone.utc)
    return mtime, "mtime"


def _parse_aware_utc(raw: object) -> Optional[datetime]:
    if raw is None or pd.isna(raw):
        return None
    value = str(raw).strip()
    if not value:
        return None
    value = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _detect_time_like_columns(df: pd.DataFrame) -> list[str]:
    needles = ("time", "date", "start", "utc")
    return [c for c in df.columns if any(n in c.lower() for n in needles)]


def _column_rates(df: pd.DataFrame, col: str) -> tuple[float, float, list[datetime], list[object]]:
    total_rows = len(df)
    if total_rows == 0:
        return 0.0, 0.0, [], []
    non_null = df[col].notna() & (df[col].astype(str).str.strip() != "")
    non_null_count = int(non_null.sum())
    non_null_rate = non_null_count / total_rows
    parsed_values: list[datetime] = []
    raw_non_null = df.loc[non_null, col]
    for raw in raw_non_null:
        parsed = _parse_aware_utc(raw)
        if parsed is not None:
            parsed_values.append(parsed)
    parseable_rate = (len(parsed_values) / non_null_count) if non_null_count else 0.0
    sample_raw = raw_non_null.head(5).tolist()
    return non_null_rate, parseable_rate, parsed_values, sample_raw


def _select_best_time_column(df: pd.DataFrame) -> tuple[Optional[str], dict[str, dict[str, object]]]:
    diagnostics: dict[str, dict[str, object]] = {}
    for col in GAME_TIME_COLUMNS:
        if col not in df.columns:
            continue
        non_null_rate, parseable_rate, parsed_values, sample_raw = _column_rates(df, col)
        diagnostics[col] = {
            "non_null_rate": non_null_rate,
            "parseable_rate": parseable_rate,
            "parsed_values": parsed_values,
            "sample_raw": sample_raw,
        }

    if not diagnostics:
        return None, diagnostics

    ranked = sorted(
        diagnostics.items(),
        key=lambda item: (item[1]["parseable_rate"], item[1]["non_null_rate"]),
        reverse=True,
    )
    best_col, best_stats = ranked[0]
    if best_stats["parseable_rate"] >= 0.70 and best_stats["non_null_rate"] >= 0.50:
        return best_col, diagnostics
    return None, diagnostics


def _print_missing_time_diagnostics(df: pd.DataFrame, diagnostics: dict[str, dict[str, object]]) -> None:
    print("[ERROR] Unable to identify a usable game time column.")
    print(f"[ERROR] Columns found ({len(df.columns)}): {list(df.columns)}")
    time_like_cols = _detect_time_like_columns(df)
    print(f"[ERROR] Detected time-like columns: {time_like_cols}")

    ranked = sorted(
        ((col, diagnostics[col]) for col in diagnostics),
        key=lambda item: (item[1]["parseable_rate"], item[1]["non_null_rate"]),
        reverse=True,
    )
    top_cols = ranked[:5]
    for col, stats in top_cols:
        print(
            "[ERROR]",
            f"column={col}",
            f"non_null_rate={stats['non_null_rate']:.3f}",
            f"parseable_rate={stats['parseable_rate']:.3f}",
            f"sample_values={stats['sample_raw']}",
        )


def _print_summary(summary: ValidationSummary) -> None:
    print("\n=== VALIDATION SUMMARY ===")
    print(f"selected_file: {summary.selected_file}")
    print(f"total_rows: {summary.total_rows}")
    print(f"timestamp_source_used: {summary.timestamp_source_used}")
    print(f"generated_at_utc: {summary.generated_at_utc.isoformat()}")
    print(f"freshness_age_hours: {summary.freshness_age_hours:.3f}")
    print(f"time_column_used: {summary.time_column_used}")
    print(f"non_null_rate: {summary.non_null_rate:.3f}")
    print(f"parseable_rate: {summary.parseable_rate:.3f}")
    print(f"window_start_utc: {summary.window_start_utc.isoformat()}")
    print(f"max_game_time_utc: {summary.max_game_time_utc.isoformat() if summary.max_game_time_utc else 'None'}")
    print(f"result: {summary.result}")
    print("=== END VALIDATION SUMMARY ===\n")


def validate(
    df: pd.DataFrame,
    *,
    selected_file: str,
    now_utc: datetime,
    timezone_local: str,
    max_hours: float,
) -> ValidationSummary:
    if now_utc.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware")

    total_rows = len(df)
    tz = ZoneInfo(timezone_local)
    window_start_utc = now_utc.astimezone(tz).astimezone(timezone.utc)
    failures: list[str] = []

    generated_at_utc, timestamp_source = _parse_generated_at(df)
    freshness_age_hours = (now_utc - generated_at_utc).total_seconds() / 3600.0

    if total_rows < 1:
        failures.append("selected predictions file is empty")
    if freshness_age_hours > max_hours:
        failures.append(f"stale predictions: freshness_age_hours={freshness_age_hours:.3f} exceeds max_hours={max_hours:.3f}")

    time_column_used, diagnostics = _select_best_time_column(df)

    non_null_rate = 0.0
    parseable_rate = 0.0
    parse_fail_rate = 1.0 if total_rows else 0.0
    max_game_time_utc: Optional[datetime] = None

    if not time_column_used:
        failures.append(f"No game time column found. Checked {GAME_TIME_COLUMNS}")
        _print_missing_time_diagnostics(df, diagnostics)
    else:
        selected_stats = diagnostics[time_column_used]
        non_null_rate = float(selected_stats["non_null_rate"])
        parseable_rate = float(selected_stats["parseable_rate"])
        parsed_values = selected_stats["parsed_values"]

        non_null_count = int((df[time_column_used].notna() & (df[time_column_used].astype(str).str.strip() != "")).sum())
        parse_fail_count = non_null_count - len(parsed_values)
        parse_fail_rate = (parse_fail_count / non_null_count) if non_null_count else 1.0

        if parsed_values:
            max_game_time_utc = max(parsed_values)

        if parse_fail_rate > 0.30:
            failures.append(f"parse_fail_rate={parse_fail_rate:.3f} exceeds threshold=0.300")
        if max_game_time_utc is None:
            failures.append("no parsable game start timestamps available")
        elif max_game_time_utc < window_start_utc:
            failures.append(
                f"not forward-looking: max_game_time_utc={max_game_time_utc.isoformat()} < window_start_utc={window_start_utc.isoformat()}"
            )

    result = "FAIL" if failures else "PASS"
    summary = ValidationSummary(
        selected_file=selected_file,
        total_rows=total_rows,
        timestamp_source_used=timestamp_source,
        generated_at_utc=generated_at_utc,
        freshness_age_hours=freshness_age_hours,
        time_column_used=time_column_used or "",
        non_null_rate=non_null_rate,
        parseable_rate=parseable_rate,
        parse_fail_rate=parse_fail_rate,
        window_start_utc=window_start_utc,
        max_game_time_utc=max_game_time_utc,
        result=result,
    )
    _print_summary(summary)

    if failures:
        raise RuntimeError("Predictions freshness validation failed: " + " | ".join(failures))
    return summary


def validate_file(
    file_path: Path,
    *,
    now_utc: datetime,
    timezone_local: str,
    max_hours: float,
) -> ValidationSummary:
    df = _load_csv(file_path)
    if df is None:
        raise RuntimeError(f"Selected predictions file missing or empty: {file_path}")
    df.attrs["_selected_file_path"] = str(file_path)
    return validate(
        df,
        selected_file=str(file_path),
        now_utc=now_utc,
        timezone_local=timezone_local,
        max_hours=max_hours,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate predictions freshness and forward-looking schedule windows")
    parser.add_argument("--data-dir", default="data", help="Directory containing predictions_latest.csv and predictions_mc_latest.csv")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=float(os.getenv("PREDICTIONS_FRESHNESS_MAX_HOURS", "6")),
        help="Maximum allowed age in hours of predictions generation timestamp",
    )
    args = parser.parse_args()

    max_hours = float(args.max_age_hours)
    timezone_local = os.getenv("TIMEZONE_LOCAL", "America/Los_Angeles")
    now_utc = datetime.now(timezone.utc)
    data_dir = Path(args.data_dir)

    try:
        df, _label, path = select_prediction_source(
            predictions_latest_path=data_dir / "predictions_latest.csv",
            predictions_mc_latest_path=data_dir / "predictions_mc_latest.csv",
        )
        df.attrs["_selected_file_path"] = str(path)
        validate(
            df,
            selected_file=str(path),
            now_utc=now_utc,
            timezone_local=timezone_local,
            max_hours=max_hours,
        )
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
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
