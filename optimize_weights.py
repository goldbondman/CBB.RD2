import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from config.model_version import compute_model_version, save_version_to_history

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")

MIN_SAMPLE = 50
ROLLING_DAYS = 60
WEIGHT_KEYS = [
    "w_schedule",
    "w_four_factors",
    "w_bidirectional",
    "w_ats",
    "w_situational",
]
DEFAULT_WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.10]

MODEL_PRED_COLS = {
    "w_schedule": "model1_schedule_pred",
    "w_four_factors": "model2_four_factors_pred",
    "w_bidirectional": "model3_bidirectional_pred",
    "w_ats": "model4_ats_pred",
    "w_situational": "model5_situational_pred",
}


def load_graded(path: Path, rolling_days: int) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"event_id": str})
    if "game_datetime_utc" in df.columns:
        df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], errors="coerce", utc=True)

    graded = df[df.get("graded", False) == True].copy()

    if rolling_days and "game_datetime_utc" in graded.columns:
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=rolling_days)
        graded = graded[graded["game_datetime_utc"] >= cutoff]

    for col in ["home_covered_pred", "actual_margin", "spread_error", *MODEL_PRED_COLS.values()]:
        if col in graded.columns:
            graded[col] = pd.to_numeric(graded[col], errors="coerce")

    return graded


def check_model_cols_available(df: pd.DataFrame) -> list[str]:
    available = [col for col in MODEL_PRED_COLS.values() if col in df.columns]
    if not available:
        log.warning(
            "No individual model prediction columns found in graded data. "
            "To enable weight optimization, ensure predictions include model1..model5 columns. "
            "Optimization will be skipped this run."
        )
    return available


def objective_ats(weights: np.ndarray, df: pd.DataFrame, available_cols: list[str]) -> float:
    pred_cols = [MODEL_PRED_COLS[k] for k in WEIGHT_KEYS if MODEL_PRED_COLS[k] in available_cols]
    if not pred_cols or "actual_margin" not in df.columns:
        return 0.5

    weighted_pred = sum(
        weights[i] * df[MODEL_PRED_COLS[WEIGHT_KEYS[i]]].fillna(0.0)
        for i in range(len(WEIGHT_KEYS))
        if MODEL_PRED_COLS[WEIGHT_KEYS[i]] in available_cols
    )

    valid = df["actual_margin"].notna()
    if valid.sum() == 0:
        return 0.5

    actual = df.loc[valid, "actual_margin"]
    pred_aligned = weighted_pred.loc[valid]
    correct = ((pred_aligned > 0) == (actual > 0)).mean()
    return 1.0 - float(correct)


def optimize_weights(df: pd.DataFrame, min_sample: int) -> dict:
    available_cols = check_model_cols_available(df)

    if not available_cols or len(df) < min_sample:
        log.info(
            "Insufficient data (%s rows, need %s) or missing model columns — using default weights",
            len(df),
            min_sample,
        )
        return dict(zip(WEIGHT_KEYS, DEFAULT_WEIGHTS))

    log.info("Optimizing weights on %s graded predictions", len(df))
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.05, 0.60)] * len(WEIGHT_KEYS)
    x0 = np.array(DEFAULT_WEIGHTS)

    result = minimize(
        objective_ats,
        x0,
        args=(df, available_cols),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 500, "ftol": 1e-6},
    )

    if not result.success:
        log.warning("Weight optimization did not fully converge: %s", result.message)

    optimized = result.x / result.x.sum()
    weights = dict(zip(WEIGHT_KEYS, [round(float(w), 4) for w in optimized]))

    default_miss = objective_ats(np.array(DEFAULT_WEIGHTS), df, available_cols)
    optimized_miss = float(result.fun)
    improvement = round((default_miss - optimized_miss) * 100, 2)

    log.info("Optimized weights: %s", weights)
    log.info(
        "ATS accuracy: default=%.1f%% → optimized=%.1f%% (%+.2fpp improvement)",
        (1 - default_miss) * 100,
        (1 - optimized_miss) * 100,
        improvement,
    )

    if improvement < 0.5:
        log.info("Improvement < 0.5pp — default weights are near-optimal. Writing optimized weights anyway.")

    return weights


def run_conference_tier_optimization(df: pd.DataFrame, min_sample: int) -> dict:
    tier_weights = {"default": dict(zip(WEIGHT_KEYS, DEFAULT_WEIGHTS))}

    if "game_tier" not in df.columns:
        log.info("No game_tier column — skipping tier-specific optimization")
        return tier_weights

    for tier in ["HIGH", "MID", "LOW"]:
        tier_df = df[df["game_tier"] == tier]
        log.info("Tier %s: %s graded predictions", tier, len(tier_df))
        if len(tier_df) >= min_sample:
            tier_weights[tier] = optimize_weights(tier_df, min_sample)
        else:
            log.info("Tier %s: only %s rows (need %s) — using default weights", tier, len(tier_df), min_sample)
            tier_weights[tier] = dict(zip(WEIGHT_KEYS, DEFAULT_WEIGHTS))

    return tier_weights


def write_weights(weights: dict, tier_weights: dict, output_path: Path, df: pd.DataFrame, rolling_days: int, min_sample: int) -> None:
    output = {
        "weights": weights,
        "weights_by_tier": tier_weights,
        "optimized_at_utc": pd.Timestamp.utcnow().isoformat(),
        "training_sample_n": len(df),
        "rolling_window_days": rolling_days,
        "min_sample_used": min_sample,
        "note": (
            "Weights auto-tuned weekly. Model reads this file at prediction time. "
            "If file is missing, defaults are used."
        ),
    }
    data_dir = output_path.parent
    current_version = compute_model_version(data_dir)
    current_version["superseded_by"] = "pending"
    current_version["superseded_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_version_to_history(current_version, data_dir / "model_version_history.json")
    log.info(
        "Archived current model version: %s before weight update",
        current_version["model_version_hash"],
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2))
    log.info("model_weights.json → %s", output_path)

    new_version = compute_model_version(data_dir)
    save_version_to_history(new_version, data_dir / "model_version_history.json")
    log.info("New model version: %s", new_version["model_version_hash"])


def main() -> None:
    try:
        from espn_config import OUT_PREDICTIONS_GRADED as graded_path, OUT_MODEL_WEIGHTS as weights_path
    except ImportError:
        graded_path = Path("data/predictions_graded.csv")
        weights_path = Path("data/model_weights.json")

    parser = argparse.ArgumentParser()
    parser.add_argument("--window", type=int, default=ROLLING_DAYS)
    parser.add_argument("--min-sample", type=int, default=MIN_SAMPLE)
    args = parser.parse_args()

    graded_path = Path(graded_path)
    weights_path = Path(weights_path)

    if not graded_path.exists():
        log.warning("predictions_graded.csv not found — skipping")
        return

    df = load_graded(graded_path, args.window)
    if len(df) < 5:
        log.warning("Only %s graded predictions — skipping optimization", len(df))
        return

    weights = optimize_weights(df, args.min_sample)
    tier_weights = run_conference_tier_optimization(df, args.min_sample)
    write_weights(weights, tier_weights, weights_path, df, args.window, args.min_sample)


if __name__ == "__main__":
    main()
