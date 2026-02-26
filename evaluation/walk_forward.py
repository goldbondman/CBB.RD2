#!/usr/bin/env python3
"""Walk-forward validation for graded prediction performance."""

from __future__ import annotations

import logging
import argparse
import json
from pathlib import Path

import pandas as pd

from cbb_prediction_model import ModelConfig

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path("data")
INPUT_CANDIDATES = [
    DATA_DIR / "results_log_graded.csv",
    DATA_DIR / "predictions_graded.csv",
]
OUTPUT_PATH = DATA_DIR / "walk_forward_results.csv"
RESULT_COLUMNS = ["fold", "train_n", "test_n", "test_week_start", "ats_pct", "mae", "roi_units"]

CLV_RESULT_COLUMNS = RESULT_COLUMNS + ["clv_mean"]


def _resolve_input_path() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists() and path.stat().st_size > 0:
            return path
    return INPUT_CANDIDATES[0]


def walk_forward_validation(
    df: pd.DataFrame,
    min_train_weeks: int = 4,
    test_window_weeks: int = 1,
    model_config: ModelConfig | None = None,
) -> pd.DataFrame:
    """
    Returns per-fold ATS accuracy, MAE, and ROI.
    Mean fold test scores are a more honest out-of-sample estimate.
    """
    working = df.copy()
    working["game_datetime_utc"] = pd.to_datetime(working["game_datetime_utc"], errors="coerce", utc=True)
    working = working.dropna(subset=["game_datetime_utc"]).sort_values("game_datetime_utc")

    config = model_config or ModelConfig()

    if "ats_correct" in working.columns:
        working["home_covered_pred"] = pd.to_numeric(working["ats_correct"], errors="coerce")
    if "spread_error" in working.columns:
        working["abs_spread_error"] = pd.to_numeric(working["spread_error"], errors="coerce").abs()

    model_cols = [
        "model1_schedule_pred",
        "model2_four_factors_pred",
        "model3_bidirectional_pred",
        "model4_ats_pred",
        "model5_situational_pred",
    ]
    available_model_cols = [col for col in model_cols if col in working.columns]
    close_col_candidates = ["home_spread_close", "closing_line", "spread_line_close"]
    close_col = next((col for col in close_col_candidates if col in working.columns), None)
    if available_model_cols and close_col:
        weights = {
            "model1_schedule_pred": float(getattr(config, "w_schedule", 0.25)),
            "model2_four_factors_pred": float(getattr(config, "w_four_factors", 0.20)),
            "model3_bidirectional_pred": float(getattr(config, "w_bidirectional", 0.25)),
            "model4_ats_pred": float(getattr(config, "w_ats", 0.15)),
            "model5_situational_pred": float(getattr(config, "w_situational", 0.10)),
        }
        weight_sum = sum(weights[col] for col in available_model_cols)
        if weight_sum > 0:
            weighted_pred = sum(
                pd.to_numeric(working[col], errors="coerce") * (weights[col] / weight_sum)
                for col in available_model_cols
            )
            working["clv_eval"] = weighted_pred - pd.to_numeric(working[close_col], errors="coerce")
    elif "clv_vs_close" in working.columns:
        working["clv_eval"] = pd.to_numeric(working["clv_vs_close"], errors="coerce")
    elif "clv_vs_consensus" in working.columns:
        working["clv_eval"] = pd.to_numeric(working["clv_vs_consensus"], errors="coerce")

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
                "clv_mean": round(float(test["clv_eval"].mean()), 4) if "clv_eval" in test.columns else None,
            }
        )

    results_df = pd.DataFrame(results, columns=CLV_RESULT_COLUMNS)

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


def validate_candidate_weights(candidate_path: str | Path, graded_path: str | Path) -> dict:
    candidate_path = Path(candidate_path)
    graded_path = Path(graded_path)

    if candidate_path.is_dir():
        candidate_file = candidate_path / "candidate_weights.json"
    else:
        candidate_file = candidate_path
    if not candidate_file.exists():
        raise FileNotFoundError(f"Candidate weights file not found: {candidate_file}")
    if not graded_path.exists():
        raise FileNotFoundError(f"Graded predictions file not found: {graded_path}")

    candidate_payload = json.loads(candidate_file.read_text())
    raw_weights = candidate_payload.get("weights", candidate_payload)

    default_config = ModelConfig()
    candidate_config = ModelConfig()
    for key, value in raw_weights.items():
        if hasattr(candidate_config, key):
            setattr(candidate_config, key, value)

    try:
        graded_df = pd.read_csv(graded_path, low_memory=False)
    except pd.errors.EmptyDataError:
        return {
            "candidate_better": False,
            "default_clv_mean": 0.0,
            "candidate_clv_mean": 0.0,
            "improvement_pp": 0.0,
            "folds_candidate_won": 0,
            "folds_total": 0,
            "recommendation": "INSUFFICIENT_DATA",
        }

    default_results = walk_forward_validation(graded_df, model_config=default_config)
    candidate_results = walk_forward_validation(graded_df, model_config=candidate_config)

    merged = default_results[["fold", "clv_mean"]].merge(
        candidate_results[["fold", "clv_mean"]], on="fold", suffixes=("_default", "_candidate")
    )
    comparable = merged.dropna(subset=["clv_mean_default", "clv_mean_candidate"])

    folds_total = int(len(comparable))
    folds_candidate_won = int((comparable["clv_mean_candidate"] > comparable["clv_mean_default"]).sum()) if folds_total else 0
    default_clv_mean = float(comparable["clv_mean_default"].mean()) if folds_total else 0.0
    candidate_clv_mean = float(comparable["clv_mean_candidate"].mean()) if folds_total else 0.0
    improvement_pp = candidate_clv_mean - default_clv_mean
    candidate_better = bool(candidate_clv_mean > default_clv_mean) if folds_total else False

    if folds_total < 4:
        recommendation = "INSUFFICIENT_DATA"
    else:
        win_rate = folds_candidate_won / folds_total if folds_total else 0.0
        should_deploy = (
            candidate_better
            and improvement_pp > 0.002
            and win_rate >= 0.60
        )
        recommendation = "DEPLOY" if should_deploy else "REJECT"

    return {
        "candidate_better": candidate_better,
        "default_clv_mean": round(default_clv_mean, 4),
        "candidate_clv_mean": round(candidate_clv_mean, 4),
        "improvement_pp": round(improvement_pp, 4),
        "folds_candidate_won": folds_candidate_won,
        "folds_total": folds_total,
        "recommendation": recommendation,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Path to graded predictions CSV")
    parser.add_argument(
        "--validate-candidate",
        type=Path,
        default=None,
        help="Path to candidate_weights.json (or folder containing it)",
    )
    args = parser.parse_args()

    input_path = args.input if args.input is not None else _resolve_input_path()
    if not input_path.exists():
        raise FileNotFoundError(
            "No graded predictions file found. Expected one of: "
            + ", ".join(str(path) for path in INPUT_CANDIDATES)
        )

    try:
        df = pd.read_csv(input_path, low_memory=False)
    except pd.errors.EmptyDataError:
        log.info("No data available in %s; writing empty walk-forward output.", input_path)
        df = pd.DataFrame()

    results_df = walk_forward_validation(df) if not df.empty else pd.DataFrame(columns=CLV_RESULT_COLUMNS)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_PATH, index=False)

    if results_df.empty:
        log.info("No eligible walk-forward folds generated.")
    else:
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

    if args.validate_candidate:
        validation = validate_candidate_weights(args.validate_candidate, input_path)
        log.info("Candidate validation: %s", json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
