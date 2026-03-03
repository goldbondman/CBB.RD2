#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd

TIMESTAMP_COLUMNS = ["generated_at_utc", "run_time_utc", "model_run_utc", "timestamp"]
GAME_TIME_COLUMNS = ["game_time_utc", "commence_time", "start_time", "date", "game_time"]


@dataclass
class ValidationSummary:
    selected_file: str
    total_rows: int
    timestamp_source_used: str
    generated_at_utc: datetime
    freshness_age_hours: float
    time_column_used: str
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
    """Local mirror of build_derived_csvs._select_prediction_source() semantics."""
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
        if "game_datetime_utc" in df.columns:
            ts = pd.to_datetime(df["game_datetime_utc"], errors="coerce", utc=True)
            if ts.notna().any():
                max_ts = ts.max()
        if max_ts is None and "generated_at" in df.columns:
            ts = pd.to_datetime(df["generated_at"], errors="coerce", utc=True)
            if ts.notna().any():
                max_ts = ts.max()

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
    """Use build_derived_csvs selector directly when possible, else local fallback."""
    label_to_path = {
        "predictions_latest": predictions_latest_path,
        "predictions_mc_latest": predictions_mc_latest_path,
    }
    try:
        from build_derived_csvs import _select_prediction_source  # local import to avoid heavy import at module load

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

    time_column_used = next((col for col in GAME_TIME_COLUMNS if col in df.columns), "")
    parse_fail_rate = 1.0 if total_rows else 0.0
    max_game_time_utc: Optional[datetime] = None

    if total_rows < 1:
        failures.append("selected predictions file is empty")
    if freshness_age_hours > max_hours:
        failures.append(f"stale predictions: freshness_age_hours={freshness_age_hours:.3f} exceeds max_hours={max_hours:.3f}")

    parse_failures: list[tuple[int, str, object]] = []
    if not time_column_used:
        failures.append(f"No game time column found. Checked {GAME_TIME_COLUMNS}")
    else:
        game_id_col = next((c for c in ["game_id", "event_id", "id"] if c in df.columns), None)
        for idx, raw in df[time_column_used].items():
            parsed = _parse_aware_utc(raw)
            if parsed is None:
                gid = str(df.at[idx, game_id_col]) if game_id_col else "n/a"
                parse_failures.append((int(idx), gid, raw))
                continue
            if max_game_time_utc is None or parsed > max_game_time_utc:
                max_game_time_utc = parsed

        parse_fail_rate = len(parse_failures) / total_rows if total_rows else 1.0
        if parse_failures:
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

    result = "FAIL" if failures else "PASS"
    summary = ValidationSummary(
        selected_file=selected_file,
        total_rows=total_rows,
        timestamp_source_used=timestamp_source,
        generated_at_utc=generated_at_utc,
        freshness_age_hours=freshness_age_hours,
        time_column_used=time_column_used,
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
    print(f"parse_fail_rate: {summary.parse_fail_rate:.3f}")
    print(f"window_start_utc: {summary.window_start_utc.isoformat()}")
    print(f"max_game_time_utc: {summary.max_game_time_utc.isoformat() if summary.max_game_time_utc else 'None'}")
    print(f"result: {summary.result}")
    print("=== END VALIDATION SUMMARY ===\n")


def main() -> int:
    max_hours = float(os.getenv("PREDICTIONS_FRESHNESS_MAX_HOURS", "6"))
    timezone_local = os.getenv("TIMEZONE_LOCAL", "America/Los_Angeles")
    now_utc = datetime.now(timezone.utc)

    try:
        df, _label, path = select_prediction_source()
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
