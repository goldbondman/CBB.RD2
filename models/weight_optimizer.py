#!/usr/bin/env python3
"""Optimize ensemble sub-model weights from graded backtest results."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


LOG = logging.getLogger("weight_optimizer")

WEIGHT_KEYS: Sequence[str] = (
    "FourFactors",
    "AdjEfficiency",
    "Pythagorean",
    "Momentum",
    "ATSIntelligence",
    "Situational",
    "CAGERankings",
    "RegressedEff",
)

SPREAD_COLS: Sequence[str] = (
    "ens_fourfactors_spread",
    "ens_adjefficiency_spread",
    "ens_pythagorean_spread",
    "ens_momentum_spread",
    "ens_atsintelligence_spread",
    "ens_situational_spread",
    "ens_cagerankings_spread",
    "ens_regressedeff_spread",
)

TOTAL_COLS: Sequence[str] = (
    "ens_fourfactors_total",
    "ens_adjefficiency_total",
    "ens_pythagorean_total",
    "ens_momentum_total",
    "ens_atsintelligence_total",
    "ens_situational_total",
    "ens_cagerankings_total",
    "ens_regressedeff_total",
)

MIN_WEIGHT = 0.05
MIN_GAMES = 30
TRAIN_RATIO = 0.7
IMPROVEMENT_THRESHOLD = 0.1


@dataclass
class OptimizationResult:
    weights: np.ndarray
    train_mae: float
    validation_mae: float
    equal_validation_mae: float
    improved: bool


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _weighted_average(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return features @ weights


def _objective(weights: np.ndarray, features: np.ndarray, target: np.ndarray) -> float:
    pred = _weighted_average(features, weights)
    return _mae(target, pred)


def _optimize_train(features: np.ndarray, target: np.ndarray) -> np.ndarray:
    n_models = features.shape[1]
    x0 = np.full(n_models, 1.0 / n_models)
    bounds = [(MIN_WEIGHT, 1.0) for _ in range(n_models)]
    constraints = ({"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)},)

    result = minimize(
        fun=_objective,
        x0=x0,
        args=(features, target),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise RuntimeError(f"SLSQP failed: {result.message}")

    return np.asarray(result.x, dtype=float)


def _split_train_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    train_size = int(np.floor(n_rows * TRAIN_RATIO))
    train_size = max(1, min(train_size, n_rows - 1))
    return df.iloc[:train_size].copy(), df.iloc[train_size:].copy()


def _run_target_optimization(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
) -> OptimizationResult:
    train_df, validation_df = _split_train_validation(df)

    train_x = train_df.loc[:, feature_cols].to_numpy(dtype=float)
    train_y = train_df.loc[:, target_col].to_numpy(dtype=float)
    validation_x = validation_df.loc[:, feature_cols].to_numpy(dtype=float)
    validation_y = validation_df.loc[:, target_col].to_numpy(dtype=float)

    equal_weights = np.full(len(feature_cols), 1.0 / len(feature_cols))
    equal_validation_pred = _weighted_average(validation_x, equal_weights)
    equal_validation_mae = _mae(validation_y, equal_validation_pred)

    optimized_weights = _optimize_train(train_x, train_y)

    train_pred = _weighted_average(train_x, optimized_weights)
    validation_pred = _weighted_average(validation_x, optimized_weights)

    train_mae = _mae(train_y, train_pred)
    validation_mae = _mae(validation_y, validation_pred)

    improvement = equal_validation_mae - validation_mae
    improved = improvement >= IMPROVEMENT_THRESHOLD

    if not improved:
        LOG.warning(
            "Optimization did not improve over baseline — keeping equal weights. Need more graded games. "
            "(baseline=%.4f, optimized=%.4f)",
            equal_validation_mae,
            validation_mae,
        )

    final_weights = optimized_weights if improved else equal_weights
    final_validation_mae = validation_mae if improved else equal_validation_mae

    return OptimizationResult(
        weights=final_weights,
        train_mae=train_mae,
        validation_mae=final_validation_mae,
        equal_validation_mae=equal_validation_mae,
        improved=improved,
    )


def _round_weight_mapping(weights: Iterable[float]) -> Dict[str, float]:
    rounded = [round(float(w), 4) for w in weights]
    return dict(zip(WEIGHT_KEYS, rounded))


def _load_and_validate_input(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Prerequisite missing: {path} not found. "
            "Run grading first to create data/model_accuracy_report.csv with >= 30 gradeable games."
        )

    df = pd.read_csv(path)

    required_cols = set(SPREAD_COLS) | set(TOTAL_COLS) | {"game_datetime_utc", "actual_margin"}
    if "actual_total" not in df.columns:
        required_cols.update({"home_score", "away_score"})

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    if "actual_total" not in df.columns:
        df["actual_total"] = pd.to_numeric(df["home_score"], errors="coerce") + pd.to_numeric(
            df["away_score"], errors="coerce"
        )

    for numeric_col in list(SPREAD_COLS) + list(TOTAL_COLS) + ["actual_margin", "actual_total"]:
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")

    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")

    gradeable_df = df.dropna(
        subset=["game_datetime_utc", "actual_margin", "actual_total", *SPREAD_COLS, *TOTAL_COLS]
    ).copy()

    if len(gradeable_df) < MIN_GAMES:
        raise RuntimeError(
            f"Prerequisite not met: need at least {MIN_GAMES} gradeable games in {path}, "
            f"found {len(gradeable_df)}."
        )

    gradeable_df = gradeable_df.sort_values("game_datetime_utc", kind="mergesort").reset_index(drop=True)
    return gradeable_df


def _log_weight_summary(label: str, weights: Dict[str, float]) -> None:
    LOG.info("%s optimized weights (desc):", label)
    for key, value in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        LOG.info("  %-16s %.4f", key, value)


def _warn_if_floor_hit(label: str, weights: Dict[str, float]) -> None:
    floor_hits = [name for name, value in weights.items() if abs(value - MIN_WEIGHT) < 1e-8]
    if floor_hits:
        LOG.warning(
            "%s weights hit the %.2f floor for: %s. Sub-model(s) may be weak candidates for replacement.",
            label,
            MIN_WEIGHT,
            ", ".join(floor_hits),
        )


def run(input_path: Path, output_path: Path) -> None:
    df = _load_and_validate_input(input_path)
    train_df, validation_df = _split_train_validation(df)

    LOG.info("N games used for optimization: %s", len(df))

    spread_result = _run_target_optimization(df, SPREAD_COLS, "actual_margin")
    total_result = _run_target_optimization(df, TOTAL_COLS, "actual_total")

    spread_weights = _round_weight_mapping(spread_result.weights)
    total_weights = _round_weight_mapping(total_result.weights)

    _log_weight_summary("Spread", spread_weights)
    _log_weight_summary("Total", total_weights)

    spread_improvement = spread_result.equal_validation_mae - spread_result.validation_mae
    LOG.info(
        "Spread train MAE: %.4f | validation MAE: %.4f | equal-weight validation MAE: %.4f | improvement: %.4f",
        spread_result.train_mae,
        spread_result.validation_mae,
        spread_result.equal_validation_mae,
        spread_improvement,
    )

    total_improvement = total_result.equal_validation_mae - total_result.validation_mae
    LOG.info(
        "Total train MAE: %.4f | validation MAE: %.4f | equal-weight validation MAE: %.4f | improvement: %.4f",
        total_result.train_mae,
        total_result.validation_mae,
        total_result.equal_validation_mae,
        total_improvement,
    )

    if spread_result.validation_mae - spread_result.train_mae > 1.0:
        LOG.warning("Spread validation MAE exceeds train MAE by more than 1.0 (possible overfitting).")
    if total_result.validation_mae - total_result.train_mae > 1.0:
        LOG.warning("Total validation MAE exceeds train MAE by more than 1.0 (possible overfitting).")

    _warn_if_floor_hit("Spread", spread_weights)
    _warn_if_floor_hit("Total", total_weights)

    payload = {
        "weights": spread_weights,
        "total_weights": total_weights,
        "metadata": {
            "n_train_games": len(train_df),
            "n_validation_games": len(validation_df),
            "train_mae": round(spread_result.train_mae, 4),
            "validation_mae": round(spread_result.validation_mae, 4),
            "equal_weight_validation_mae": round(spread_result.equal_validation_mae, 4),
            "improvement_vs_equal": round(spread_improvement, 4),
            "optimization_date": datetime.now(timezone.utc).date().isoformat(),
            "minimum_weight_floor": MIN_WEIGHT,
            "total_train_mae": round(total_result.train_mae, 4),
            "total_validation_mae": round(total_result.validation_mae, 4),
            "total_equal_weight_validation_mae": round(total_result.equal_validation_mae, 4),
            "total_improvement_vs_equal": round(total_improvement, 4),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    LOG.info("Wrote optimized weights to %s", output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/model_accuracy_report.csv"),
        help="Path to graded backtest report CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/backtest_optimized_weights.json"),
        help="Output JSON path for optimized ensemble weights",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    try:
        run(args.input, args.output)
    except Exception as exc:  # noqa: BLE001 - intentionally surfacing clear CLI failure messages
        LOG.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
