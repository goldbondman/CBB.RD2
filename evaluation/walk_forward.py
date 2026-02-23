#!/usr/bin/env python3
"""Walk-forward validation for graded prediction performance."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path("data")
INPUT_CANDIDATES = [
    DATA_DIR / "results_log_graded.csv",
    DATA_DIR / "predictions_graded.csv",
]
OUTPUT_PATH = DATA_DIR / "walk_forward_results.csv"
RESULT_COLUMNS = ["fold", "train_n", "test_n", "test_week_start", "ats_pct", "mae", "roi_units"]


def _resolve_input_path() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists() and path.stat().st_size > 0:
            return path
    return INPUT_CANDIDATES[0]


def walk_forward_validation(
    df: pd.DataFrame,
    min_train_weeks: int = 4,
    test_window_weeks: int = 1,
) -> pd.DataFrame:
    """
    Returns per-fold ATS accuracy, MAE, and ROI.
    Mean fold test scores are a more honest out-of-sample estimate.
    """
    working = df.copy()
    working["game_datetime_utc"] = pd.to_datetime(working["game_datetime_utc"], errors="coerce", utc=True)
    working = working.dropna(subset=["game_datetime_utc"]).sort_values("game_datetime_utc")

    if "ats_correct" in working.columns:
        working["home_covered_pred"] = pd.to_numeric(working["ats_correct"], errors="coerce")
    if "spread_error" in working.columns:
        working["abs_spread_error"] = pd.to_numeric(working["spread_error"], errors="coerce").abs()

    required = ["home_covered_pred", "abs_spread_error"]
    missing = [col for col in required if col not in working.columns]
    if missing:
        raise ValueError(f"Missing required columns for walk-forward: {missing}")

    working["week"] = working["game_datetime_utc"].dt.to_period("W")
    weeks = sorted(working["week"].dropna().unique())

    results = []
    for i in range(min_train_weeks, len(weeks) - test_window_weeks + 1):
        train_weeks = weeks[:i]
        test_weeks = weeks[i : i + test_window_weeks]

        train = working[working["week"].isin(train_weeks)]
        test = working[working["week"].isin(test_weeks)]

        test = test.dropna(subset=["home_covered_pred", "abs_spread_error"])
        if len(test) < 5:
            continue

        ats_test = test["home_covered_pred"].mean() * 100
        mae_test = test["abs_spread_error"].mean()
        test_roi = test["ats_roi"].sum() if "ats_roi" in test.columns else None

        results.append(
            {
                "fold": i,
                "train_n": len(train),
                "test_n": len(test),
                "test_week_start": str(test_weeks[0]),
                "ats_pct": round(float(ats_test), 1),
                "mae": round(float(mae_test), 2),
                "roi_units": round(float(test_roi), 3) if test_roi is not None else None,
            }
        )

    results_df = pd.DataFrame(results, columns=RESULT_COLUMNS)

    if len(results_df) >= 4:
        from scipy import stats

        slope, _, _, p_val, _ = stats.linregress(results_df["fold"], results_df["ats_pct"])
        if slope < -0.5 and p_val < 0.1:
            log.warning(
                "⚠️ EDGE DECAY DETECTED: ATS%% declining %.2fpp/fold (p=%.3f). "
                "Model may be overfitting to early-season patterns.",
                slope,
                p_val,
            )

    return results_df


def main() -> None:
    input_path = _resolve_input_path()
    if not input_path.exists():
        raise FileNotFoundError(
            "No graded predictions file found. Expected one of: "
            + ", ".join(str(path) for path in INPUT_CANDIDATES)
        )

    try:
        df = pd.read_csv(input_path, low_memory=False)
    except pd.errors.EmptyDataError:
        log.info("No data available in %s; writing empty walk-forward output.", input_path)
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(OUTPUT_PATH, index=False)
        return

    results_df = walk_forward_validation(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    if results_df.empty:
        log.info("No eligible walk-forward folds generated.")
        return

    mean_ats = results_df["ats_pct"].mean()
    mean_mae = results_df["mae"].mean()
    log.info("Walk-forward folds: %s", len(results_df))
    log.info("Mean ATS%%: %.1f | Mean MAE: %.2f", mean_ats, mean_mae)
    for _, row in results_df.iterrows():
        log.info(
            "fold=%s week=%s ATS=%.1f%% MAE=%.2f",
            int(row["fold"]),
            row["test_week_start"],
            row["ats_pct"],
            row["mae"],
        )


if __name__ == "__main__":
    main()
