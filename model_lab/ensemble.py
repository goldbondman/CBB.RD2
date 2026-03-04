from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import ModelLabConfig
from .evaluators import evaluate_predictions
from .splits import Fold


@dataclass
class EnsembleRunResult:
    weights_by_market: dict[str, dict[str, float]]
    fold_results: pd.DataFrame
    blocked_reasons: list[str]


def _enforce_weight_cap(weights: np.ndarray, max_weight: float) -> np.ndarray:
    w = np.maximum(np.asarray(weights, dtype=float), 0.0)
    n = len(w)
    if n == 0:
        return w

    if max_weight * n < 1.0:
        max_weight = 1.0 / n

    if w.sum() <= 0:
        w = np.ones(n, dtype=float) / n
    else:
        w = w / w.sum()

    free = np.ones(n, dtype=bool)
    while True:
        over = (w > max_weight + 1e-12) & free
        if not over.any():
            break
        excess = float((w[over] - max_weight).sum())
        w[over] = max_weight
        free[over] = False

        if free.sum() == 0:
            break
        free_sum = float(w[free].sum())
        if free_sum <= 0:
            w[free] = 1.0 / free.sum()
        else:
            w[free] = w[free] + excess * (w[free] / free_sum)

    w = np.maximum(w, 0.0)
    if w.sum() <= 0:
        return np.ones(n, dtype=float) / n
    return w / w.sum()


def _market_score(metrics: dict[str, float], market: str) -> float:
    if market in {"spread", "total"}:
        roi = metrics.get("roi")
        if pd.notna(roi):
            return float(roi)
        hit = metrics.get("hit_rate")
        if pd.notna(hit):
            return float(hit)
        mae = metrics.get("mae")
        if pd.notna(mae):
            return float(-mae)
        return float("nan")

    if market == "ml":
        brier = metrics.get("brier")
        if pd.notna(brier):
            return float(-brier)
        return float("nan")

    return float("nan")


def _evaluate_market(
    market: str,
    y_true: pd.Series,
    y_pred: pd.Series,
    frame: pd.DataFrame,
    config: ModelLabConfig,
) -> dict[str, float]:
    if market == "spread":
        return evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            market_line=frame["spread_line"],
            odds=None,
            market="spread",
            default_odds=config.default_odds,
            line_open=frame.get("spread_open"),
            line_close=frame.get("spread_close"),
        )

    if market == "total":
        return evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            market_line=frame["total_line"],
            odds=None,
            market="total",
            default_odds=config.default_odds,
            line_open=frame.get("total_open"),
            line_close=frame.get("total_close"),
        )

    if market == "ml":
        return evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            market_line=None,
            odds={"home_ml": frame.get("home_ml"), "away_ml": frame.get("away_ml")},
            market="ml",
            default_odds=config.default_odds,
        )

    raise ValueError(f"Unsupported market: {market}")


def _objective(
    weights: np.ndarray,
    market: str,
    x: np.ndarray,
    y_true: pd.Series,
    frame: pd.DataFrame,
    config: ModelLabConfig,
) -> float:
    w = _enforce_weight_cap(weights, config.max_weight)
    pred = pd.Series(x @ w, index=frame.index)
    metrics = _evaluate_market(market, y_true=y_true, y_pred=pred, frame=frame, config=config)
    score = _market_score(metrics, market)
    if pd.isna(score):
        return 1e6
    return -score


def optimize_weights(
    market: str,
    train_frame: pd.DataFrame,
    model_columns: list[str],
    config: ModelLabConfig,
) -> np.ndarray:
    x = train_frame[model_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)

    if market == "spread":
        y_true = pd.to_numeric(train_frame["actual_margin"], errors="coerce")
    elif market == "total":
        y_true = pd.to_numeric(train_frame["actual_total"], errors="coerce")
    else:
        y_true = pd.to_numeric(train_frame["home_won"], errors="coerce")

    n_models = len(model_columns)
    init = np.ones(n_models, dtype=float) / n_models

    bounds = [(0.0, config.max_weight) for _ in range(n_models)]
    constraints = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]

    try:
        res = minimize(
            _objective,
            init,
            args=(market, x, y_true, train_frame, config),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-8},
        )
        if res.success and np.isfinite(res.x).all():
            return _enforce_weight_cap(res.x, config.max_weight)
    except Exception:
        pass

    return _enforce_weight_cap(init, config.max_weight)


def build_market_dataset(predictions_by_model: dict[str, pd.DataFrame], market: str) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    model_columns: list[str] = []

    for model_name, df in predictions_by_model.items():
        if df.empty:
            continue

        keep = [
            "season_id",
            "game_id",
            "event_id",
            "game_datetime_utc",
            "game_date",
            "home_team_id",
            "away_team_id",
            "neutral_site",
            "actual_margin",
            "actual_total",
            "home_won",
            "spread_line",
            "spread_open",
            "spread_close",
            "total_line",
            "total_open",
            "total_close",
            "home_ml",
            "away_ml",
            "pred_spread",
            "pred_total",
        ]

        local = df.copy()
        for col in keep:
            if col not in local.columns:
                local[col] = pd.NA

        col_name = f"model_{model_name}"
        if market == "spread":
            local[col_name] = pd.to_numeric(local["pred_spread"], errors="coerce")
        elif market == "total":
            local[col_name] = pd.to_numeric(local["pred_total"], errors="coerce")
        else:
            spread = pd.to_numeric(local["pred_spread"], errors="coerce")
            local[col_name] = 1.0 / (1.0 + np.exp(spread / 6.0))

        local = local[[c for c in keep if c in local.columns] + [col_name]].copy()
        model_columns.append(col_name)

        if merged is None:
            merged = local
        else:
            merged = merged.merge(
                local[["game_id", col_name]],
                on="game_id",
                how="inner",
            )

    if merged is None or merged.empty:
        return pd.DataFrame()

    merged = merged.drop_duplicates(subset=["game_id"], keep="last")
    merged = merged.sort_values(["season_id", "game_datetime_utc", "game_id"], na_position="last")

    if market == "spread":
        merged = merged[merged["actual_margin"].notna()].copy()
    elif market == "total":
        merged = merged[merged["actual_total"].notna()].copy()
    else:
        merged = merged[merged["home_won"].notna()].copy()

    keep_cols = [
        "season_id",
        "game_id",
        "event_id",
        "game_datetime_utc",
        "game_date",
        "home_team_id",
        "away_team_id",
        "neutral_site",
        "actual_margin",
        "actual_total",
        "home_won",
        "spread_line",
        "spread_open",
        "spread_close",
        "total_line",
        "total_open",
        "total_close",
        "home_ml",
        "away_ml",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    model_cols = [c for c in merged.columns if c.startswith("model_")]
    return merged[keep_cols + model_cols].reset_index(drop=True)


def evaluate_ensemble(
    market: str,
    dataset: pd.DataFrame,
    folds: list[Fold],
    config: ModelLabConfig,
) -> EnsembleRunResult:
    blocked: list[str] = []
    if dataset.empty:
        blocked.append(f"empty_dataset:{market}")
        return EnsembleRunResult({}, pd.DataFrame(), blocked)
    if not folds:
        blocked.append(f"no_folds:{market}")
        return EnsembleRunResult({}, pd.DataFrame(), blocked)

    model_columns = [c for c in dataset.columns if c.startswith("model_")]
    if len(model_columns) < 2:
        blocked.append(f"insufficient_models:{market}")
        return EnsembleRunResult({}, pd.DataFrame(), blocked)

    fold_rows: list[dict[str, Any]] = []
    learned_weights: list[tuple[np.ndarray, int]] = []

    for fold in folds:
        train_idx = [i for i in fold.train_index if i in dataset.index]
        test_idx = [i for i in fold.test_index if i in dataset.index]

        train_df = dataset.loc[train_idx].copy()
        test_df = dataset.loc[test_idx].copy()
        if train_df.empty or test_df.empty:
            blocked.append(f"empty_fold:{market}:{fold.fold_id}")
            continue

        weights = optimize_weights(market, train_df, model_columns, config)
        learned_weights.append((weights, len(train_df)))

        test_pred = pd.Series(
            test_df[model_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float) @ weights,
            index=test_df.index,
        )

        if market == "spread":
            y_true = pd.to_numeric(test_df["actual_margin"], errors="coerce")
        elif market == "total":
            y_true = pd.to_numeric(test_df["actual_total"], errors="coerce")
        else:
            y_true = pd.to_numeric(test_df["home_won"], errors="coerce")

        metrics = _evaluate_market(market, y_true=y_true, y_pred=test_pred, frame=test_df, config=config)

        fold_row = {
            "market": market,
            "fold_id": fold.fold_id,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "hit_rate": metrics.get("hit_rate"),
            "roi": metrics.get("roi"),
            "clv_mean": metrics.get("clv_mean"),
            "mae": metrics.get("mae"),
            "brier": metrics.get("brier"),
            "calibration_ece": metrics.get("calibration_ece"),
        }
        for col, w in zip(model_columns, weights):
            fold_row[f"weight_{col}"] = float(w)
        fold_rows.append(fold_row)

    fold_df = pd.DataFrame(fold_rows)
    if not learned_weights:
        blocked.append(f"no_successful_folds:{market}")
        return EnsembleRunResult({}, fold_df, sorted(set(blocked)))

    total_weight = float(sum(n for _, n in learned_weights))
    avg = np.zeros(len(model_columns), dtype=float)
    for w, n in learned_weights:
        avg += w * (n / total_weight)
    avg = _enforce_weight_cap(avg, config.max_weight)

    weights_by_market = {
        market: {model_name.replace("model_", ""): float(w) for model_name, w in zip(model_columns, avg)}
    }

    return EnsembleRunResult(weights_by_market, fold_df, sorted(set(blocked)))
