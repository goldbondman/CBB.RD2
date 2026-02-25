from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MIN_GAMES = 20
MIN_ABS_ERROR = 1.5
CORRECTION_FACTOR = 0.5
MAX_CORRECTION = 3.0
OUTPUT_COLUMNS = [
    "dimension",
    "group",
    "n_games",
    "mean_signed_error",
    "correction",
    "actionable",
    "first_half_error",
    "second_half_error",
    "direction_consistent",
    "last_updated",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _empty_bias_table(last_updated: str) -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS).assign(last_updated=last_updated).iloc[0:0]


def _detect_signed_error_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("signed_error", "spread_error", "prediction_error", "error"):
        if candidate in df.columns:
            return candidate
    return None


def _bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def _build_spread_bucket(df: pd.DataFrame) -> pd.Series:
    spread_col = next((c for c in ("closing_line", "spread_line", "line") if c in df.columns), None)
    if spread_col is None:
        spread_size = pd.Series(np.nan, index=df.index)
    else:
        spread_size = pd.to_numeric(df[spread_col], errors="coerce").abs()

    buckets = pd.cut(
        spread_size,
        bins=[-np.inf, 3, 6, 10, np.inf],
        labels=["0-3", "3-6", "6-10", "10+"],
        right=False,
    )
    return buckets.astype(str).replace("nan", "UNKNOWN")


def _build_favorite_side(df: pd.DataFrame) -> pd.Series:
    spread_col = next((c for c in ("closing_line", "spread_line", "line") if c in df.columns), None)
    if spread_col is None:
        return pd.Series("UNKNOWN", index=df.index)

    line = pd.to_numeric(df[spread_col], errors="coerce")
    side = np.where(line < 0, "home_fav", np.where(line > 0, "away_fav", "UNKNOWN"))
    return pd.Series(side, index=df.index)


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    err_col = _detect_signed_error_column(work)
    if err_col is None:
        raise ValueError(
            "model_accuracy_report.csv missing signed error column. Expected one of: "
            "signed_error, spread_error, prediction_error, error"
        )

    work["signed_error"] = pd.to_numeric(work[err_col], errors="coerce")
    work = work.dropna(subset=["signed_error"]).copy()

    if "game_datetime_utc" in work.columns:
        work["game_datetime_utc"] = pd.to_datetime(work["game_datetime_utc"], errors="coerce", utc=True)
    else:
        work["game_datetime_utc"] = pd.NaT

    if work["game_datetime_utc"].notna().any():
        work = work.sort_values("game_datetime_utc")
    else:
        work = work.reset_index(drop=True)

    work["conference_tier"] = work.get("conference_tier", pd.Series("UNKNOWN", index=work.index)).fillna("UNKNOWN").astype(str).str.upper()
    work["spread_bucket"] = _build_spread_bucket(work)
    work["game_tier"] = work.get("game_tier", pd.Series("UNKNOWN", index=work.index)).fillna("UNKNOWN").astype(str).str.upper()

    if "home_momentum_tier" in work.columns:
        work["momentum_tier"] = work["home_momentum_tier"].fillna("UNKNOWN").astype(str).str.upper()
    else:
        work["momentum_tier"] = "UNKNOWN"

    work["favorite_side"] = _build_favorite_side(work)
    return work


def _direction_consistent(first_half_error: float, second_half_error: float) -> bool:
    if pd.isna(first_half_error) or pd.isna(second_half_error):
        return False
    if first_half_error == 0 or second_half_error == 0:
        return False
    return np.sign(first_half_error) == np.sign(second_half_error)


def _summarize_dimension(df: pd.DataFrame, dimension: str, last_updated: str) -> list[dict]:
    rows: list[dict] = []

    if dimension not in df.columns:
        return rows

    if df["game_datetime_utc"].notna().any():
        cutoff = df["game_datetime_utc"].median()
        first_mask = df["game_datetime_utc"].le(cutoff)
        second_mask = df["game_datetime_utc"].gt(cutoff)
    else:
        midpoint = max(len(df) // 2, 1)
        first_mask = pd.Series(False, index=df.index)
        first_mask.iloc[:midpoint] = True
        second_mask = ~first_mask

    for group_name, group_df in df.groupby(dimension, dropna=False):
        n_games = int(len(group_df))
        mean_error = float(group_df["signed_error"].mean()) if n_games else np.nan

        first_half = group_df[first_mask.loc[group_df.index]]
        second_half = group_df[second_mask.loc[group_df.index]]
        first_half_error = float(first_half["signed_error"].mean()) if len(first_half) else np.nan
        second_half_error = float(second_half["signed_error"].mean()) if len(second_half) else np.nan

        consistent = _direction_consistent(first_half_error, second_half_error)
        actionable = (
            n_games >= MIN_GAMES
            and abs(mean_error) >= MIN_ABS_ERROR
            and consistent
        )

        correction = float(np.clip(-mean_error * CORRECTION_FACTOR, -MAX_CORRECTION, MAX_CORRECTION)) if actionable else 0.0

        rows.append(
            {
                "dimension": dimension,
                "group": str(group_name),
                "n_games": n_games,
                "mean_signed_error": round(mean_error, 4),
                "correction": round(correction, 4),
                "actionable": bool(actionable),
                "first_half_error": round(first_half_error, 4) if not pd.isna(first_half_error) else np.nan,
                "second_half_error": round(second_half_error, 4) if not pd.isna(second_half_error) else np.nan,
                "direction_consistent": bool(consistent),
                "last_updated": last_updated,
            }
        )

    return rows


def run_bias_detection(
    accuracy_report_path: Path = Path("data/model_accuracy_report.csv"),
    output_path: Path = Path("data/model_bias_table.csv"),
) -> pd.DataFrame:
    last_updated = _now_iso()

    if not accuracy_report_path.exists():
        log.warning(
            "Insufficient sample for bias detection (n=0). Need 20+ graded games. Run backtester grading first."
        )
        result = _empty_bias_table(last_updated)
        result.to_csv(output_path, index=False)
        return result

    raw_df = pd.read_csv(accuracy_report_path)
    if "graded" in raw_df.columns:
        raw_df = raw_df[_bool_series(raw_df["graded"])].copy()

    prepared = _prepare_frame(raw_df)
    n_rows = len(prepared)

    if n_rows < MIN_GAMES:
        log.warning(
            "Insufficient sample for bias detection (n=%s). Need 20+ graded games. Run backtester grading first.",
            n_rows,
        )
        result = _empty_bias_table(last_updated)
        result.to_csv(output_path, index=False)
        return result

    dimensions = [
        "conference_tier",
        "spread_bucket",
        "game_tier",
        "momentum_tier",
        "favorite_side",
    ]

    records: list[dict] = []
    for dim in dimensions:
        records.extend(_summarize_dimension(prepared, dim, last_updated))

    result = pd.DataFrame(records, columns=OUTPUT_COLUMNS)
    result = result.sort_values(["dimension", "group"]).reset_index(drop=True)
    result.to_csv(output_path, index=False)

    actionable_n = int(result["actionable"].sum()) if not result.empty else 0
    log.info("Bias detection complete: %s rows (%s actionable)", len(result), actionable_n)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_bias_detection()
