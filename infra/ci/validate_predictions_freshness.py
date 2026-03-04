#!/usr/bin/env python3
# home/away splits
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

TIMESTAMP_COLUMNS = [
    "generated_at_utc",
    "run_time_utc",
    "model_run_utc",
    "created_at_utc",
    "pipeline_run_utc",
    "generated_at",
    "timestamp",
]
GAME_TIME_COLUMNS = [
    "game_time_utc",
    "game_datetime_utc",
    "game_datetime",
    "start_time_utc",
    "commence_time_utc",
    "scheduled_utc",
    "scheduled",
    "tipoff_utc",
    "tipoff",
    "commence_time",
    "start_time",
    "date",
    "game_time",
    "game_date",
]
PREDICTED_AT_COLUMN = "predicted_at_utc"
ARTIFACT_MARKER_FILE = ".artifact_marker.txt"
TIME_LIKE_TOKENS = ("time", "date", "start", "utc", "commence", "tip", "sched")


@dataclass
class ValidationSummary:
    selected_file: str
    total_rows: int
    timestamp_source_used: str
    generated_at_utc: datetime
    freshness_age_hours: float
    time_column_used: str
    time_non_null_rate: float
    time_parseable_rate: float
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


def _artifact_marker_status(data_dir: Path) -> str:
    marker_path = data_dir / ARTIFACT_MARKER_FILE
    if not marker_path.exists():
        return "not_found"
    marker = marker_path.read_text(encoding="utf-8").strip()
    return marker or "present"


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


def _parse_utc_strict_series(series: pd.Series) -> tuple[pd.Series, list[object]]:
    parsed_values: list[pd.Timestamp] = []
    failing_values: list[object] = []

    for raw in series.tolist():
        parsed_dt = _parse_aware_utc(raw)
        if parsed_dt is None:
            parsed_values.append(pd.NaT)
            if raw is not None and not pd.isna(raw) and str(raw).strip():
                failing_values.append(raw)
        else:
            parsed_values.append(pd.Timestamp(parsed_dt))

    return pd.Series(parsed_values, index=series.index, dtype="datetime64[ns, UTC]"), failing_values


def _candidate_run_timestamp(df: pd.DataFrame, path: Path) -> tuple[pd.Timestamp, str, str | None]:
    for col in TIMESTAMP_COLUMNS:
        if col not in df.columns:
            continue
        parsed, _ = _parse_utc_strict_series(df[col])
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
        df = _load_csv(path)
        if df is None:
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


def _fallback_select_prediction_source(
    predictions_latest_path: Path,
    predictions_mc_latest_path: Path,
) -> tuple[pd.DataFrame, str, Path]:
    local_data_dir = predictions_latest_path.parent
    source_path, source_df, _, _, _ = _select_source(local_data_dir)
    label = "predictions_latest" if source_path.name == "predictions_latest.csv" else "predictions_mc_latest"
    return source_df, label, source_path


def _time_like_columns(columns: list[str]) -> list[str]:
    lowered = []
    for col in columns:
        name = str(col).lower()
        if any(token in name for token in TIME_LIKE_TOKENS):
            lowered.append(col)
    return lowered


def _column_priority(name: str) -> int:
    if name == "game_time_utc":
        return 0
    if name == "game_datetime_utc":
        return 1
    return 2


def _select_best_game_time_column(df: pd.DataFrame) -> tuple[str | None, pd.Series | None, dict[str, dict[str, float]], list[object]]:
    present_candidates = [c for c in GAME_TIME_COLUMNS if c in df.columns]
    if not present_candidates:
        return None, None, {}, []

    total_rows = len(df)
    diagnostics: dict[str, dict[str, float]] = {}
    best_col: str | None = None
    best_score: tuple[float, float, int] | None = None
    best_parsed: pd.Series | None = None
    best_failing_values: list[object] = []

    for col in present_candidates:
        series = df[col]
        non_null = int(series.notna().sum())
        parsed, failing_values = _parse_utc_strict_series(series)
        parseable = int(parsed.notna().sum())

        non_null_rate = (non_null / total_rows) if total_rows else 0.0
        parseable_rate = (parseable / non_null) if non_null else 0.0
        diagnostics[col] = {
            "non_null_count": float(non_null),
            "parseable_count": float(parseable),
            "non_null_rate": non_null_rate,
            "parseable_rate": parseable_rate,
        }

        score = (parseable_rate, non_null_rate, -_column_priority(col))
        if best_score is None or score > best_score:
            best_score = score
            best_col = col
            best_parsed = parsed
            best_failing_values = failing_values

    return best_col, best_parsed, diagnostics, best_failing_values


def _parse_generated_at(df: pd.DataFrame) -> tuple[datetime, str]:
    selected_df_path = df.attrs.get("_selected_file_path")
    for col in TIMESTAMP_COLUMNS:
        if col not in df.columns:
            continue
        parsed, _ = _parse_utc_strict_series(df[col])
        if parsed.notna().any():
            return parsed.max().to_pydatetime().astimezone(timezone.utc), col

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

    time_column_used = ""
    time_non_null_rate = 0.0
    time_parseable_rate = 0.0
    parse_fail_rate = 1.0 if total_rows else 0.0
    max_game_time_utc: Optional[datetime] = None

    if total_rows < 1:
        failures.append("selected predictions file is empty")
    if freshness_age_hours > max_hours:
        failures.append(f"stale predictions: freshness_age_hours={freshness_age_hours:.3f} exceeds max_hours={max_hours:.3f}")

    if total_rows:
        game_id_col = next((c for c in ["game_id", "event_id", "id"] if c in df.columns), None)
        time_column_used, parsed_times, diagnostics, failing_values = _select_best_game_time_column(df)

        if not time_column_used or parsed_times is None:
            columns_found = list(df.columns)
            time_like = _time_like_columns(columns_found)
            failures.append(f"No game time column found. Checked {GAME_TIME_COLUMNS}")
            print(f"[ERROR] Columns found in {selected_file}: {columns_found}")
            print(f"[ERROR] Time-like columns in {selected_file}: {time_like}")
        else:
            stats = diagnostics[time_column_used]
            non_null_count = int(stats["non_null_count"])
            parseable_count = int(stats["parseable_count"])
            time_non_null_rate = stats["non_null_rate"]
            time_parseable_rate = stats["parseable_rate"]
            parse_fail_rate = 1.0 - time_parseable_rate if non_null_count else 1.0

            if parseable_count:
                max_game_time_utc = parsed_times.max().to_pydatetime().astimezone(timezone.utc)

            if non_null_count == 0 or parseable_count == 0:
                raw_non_null = [v for v in df[time_column_used].tolist() if v is not None and not pd.isna(v)][:10]
                print("[ERROR] Selected game time column has no parseable values.")
                print(f"[ERROR] chosen_candidate={time_column_used}")
                print(f"[ERROR] non_null_rate={time_non_null_rate:.3f}")
                print(f"[ERROR] first_10_non_null_values={raw_non_null}")
                print(f"[ERROR] first_20_failing_values={failing_values[:20]}")

            parse_failures: list[tuple[int, str, object]] = []
            for idx, raw in df[time_column_used].items():
                if raw is None or pd.isna(raw) or not str(raw).strip():
                    continue
                if _parse_aware_utc(raw) is None:
                    gid = str(df.at[idx, game_id_col]) if game_id_col else "n/a"
                    parse_failures.append((int(idx), gid, raw))

            if parse_failures:
                print(f"[WARN] chosen_candidate={time_column_used}")
                print(f"[WARN] non_null_rate={time_non_null_rate:.3f}")
                raw_non_null = [v for v in df[time_column_used].tolist() if v is not None and not pd.isna(v)][:10]
                print(f"[WARN] first_10_non_null_values={raw_non_null}")
                print("[WARN] Time parse failures (first 20):")
                for row_idx, gid, raw in parse_failures[:20]:
                    print(f"  - row={row_idx}, game_id={gid}, raw_time={raw!r}")
                omitted = len(parse_failures) - 20
                if omitted > 0:
                    print(f"  ... omitted {omitted} additional parse failures")

            if parse_fail_rate > 0.30:
                failures.append(f"parse_fail_rate={parse_fail_rate:.3f} exceeds threshold=0.300")
            if max_game_time_utc is None:
                failures.append("no parsable game start timestamps available")
            elif max_game_time_utc < window_start_utc:
                failures.append(
                    f"not forward-looking: max_game_time_utc={max_game_time_utc.isoformat()} < window_start_utc={window_start_utc.isoformat()}"
                )
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

    upcoming_games_count = 0
    if time_column_used and max_game_time_utc is not None:
        selected_stats = diagnostics.get(time_column_used, {})
        parsed_values = selected_stats.get("parsed_values", [])
        upcoming_games_count = sum(1 for dt in parsed_values if dt >= window_start_utc)

    if freshness_age_hours <= max_hours and upcoming_games_count == 0:
        failures = [f for f in failures if not f.startswith("not forward-looking:")]

    if not failures and upcoming_games_count == 0:
        result = "SKIP"
    else:
        result = "FAIL" if failures else "PASS"

    print(
        "[INFO] gate_decision="
        f"{result} "
        f"(freshness_ok={freshness_age_hours <= max_hours}, "
        f"upcoming_games={upcoming_games_count}, "
        f"fails={len(failures)})"
    )
    summary = ValidationSummary(
        selected_file=selected_file,
        total_rows=total_rows,
        timestamp_source_used=timestamp_source,
        generated_at_utc=generated_at_utc,
        freshness_age_hours=freshness_age_hours,
        time_non_null_rate=time_non_null_rate,
        time_parseable_rate=time_parseable_rate,
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


def _print_summary(summary: ValidationSummary) -> None:
    print("\n=== VALIDATION SUMMARY ===")
    print(f"selected_file: {summary.selected_file}")
    print(f"total_rows: {summary.total_rows}")
    print(f"timestamp_source_used: {summary.timestamp_source_used}")
    print(f"generated_at_utc: {summary.generated_at_utc.isoformat()}")
    print(f"freshness_age_hours: {summary.freshness_age_hours:.3f}")
    print(f"time_column_used: {summary.time_column_used}")
    print(f"time_non_null_rate: {summary.time_non_null_rate:.3f}")
    print(f"time_parseable_rate: {summary.time_parseable_rate:.3f}")
    print(f"parse_fail_rate: {summary.parse_fail_rate:.3f}")
    print(f"window_start_utc: {summary.window_start_utc.isoformat()}")
    print(f"max_game_time_utc: {summary.max_game_time_utc.isoformat() if summary.max_game_time_utc else 'None'}")
    print(f"result: {summary.result}")
    print("=== END VALIDATION SUMMARY ===\n")


def select_prediction_source(
    predictions_latest_path: Path = Path("data/predictions_latest.csv"),
    predictions_mc_latest_path: Path = Path("data/predictions_mc_latest.csv"),
) -> tuple[pd.DataFrame, str, Path]:
    """Use build_derived_csvs selector directly when possible, else local fallback."""
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
        local_data_dir = predictions_latest_path.parent
        source_path, source_df, _, _, _ = _select_source(local_data_dir)
        label = "predictions_latest" if source_path.name == "predictions_latest.csv" else "predictions_mc_latest"
        return source_df, label, source_path


def validate_predictions_freshness(data_dir: Path, max_age_hours: float) -> ValidationSummary:
    now_utc = datetime.now(timezone.utc)
    marker_status = _artifact_marker_status(data_dir)

    source_path, source_df, _, used_timestamp_col, warning = _select_source(data_dir)
    source_df.attrs["_selected_file_path"] = str(source_path)

    try:
        summary = validate(
            source_df,
            selected_file=str(source_path),
            now_utc=now_utc,
            timezone_local=os.getenv("TIMEZONE_LOCAL", "America/Los_Angeles"),
            max_hours=max_age_hours,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "stale predictions" in message:
            raise RuntimeError(f"Predictions freshness gate failed: {message}") from exc
        if "not forward-looking" in message:
            raise RuntimeError(f"Predictions forward-looking schedule gate failed: {message}") from exc
        if "parse_fail_rate" in message or "no parsable game start timestamps available" in message:
            raise RuntimeError(f"Predictions schedule parse gate failed: {message}") from exc
        raise

    print(f"[INFO] now_utc={now_utc.isoformat()}")
    print(f"[INFO] artifact_marker={marker_status}")
    print(f"[INFO] freshness_threshold_hours={max_age_hours}")
    print(f"[INFO] freshness_timestamp_source={used_timestamp_col}")
    if warning:
        print(warning)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate predictions artifact freshness and forward-looking schedule")
    parser.add_argument("--data-dir", default="data", help="Directory containing predictions files")
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=float(os.getenv("PREDICTIONS_FRESHNESS_MAX_HOURS", "6")),
        help="Maximum allowed age for predictions artifact",
    )
    args = parser.parse_args()

    try:
        validate_predictions_freshness(Path(args.data_dir), args.max_age_hours)
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
