from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from .evaluators import evaluate_predictions
from .splits import Fold


@dataclass
class FeatureTestResult:
    feature_scorecard: pd.DataFrame
    feature_stability: pd.DataFrame
    blocked_reasons: list[str]


EXCLUDE_FEATURE_COLS = {
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
    "spread_open",
    "spread_close",
    "spread_line",
    "total_open",
    "total_close",
    "total_line",
    "home_ml",
    "away_ml",
}


def _safe_fold_slice(df: pd.DataFrame, idx: list[int]) -> pd.DataFrame:
    present = [i for i in idx if i in df.index]
    return df.loc[present].copy()


def _feature_columns(df: pd.DataFrame, max_features: int | None = None) -> list[str]:
    features = []
    for col in df.columns:
        if col in EXCLUDE_FEATURE_COLS:
            continue
        if col.endswith("_source"):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() < 20:
            continue
        features.append(col)

    if max_features is not None and max_features > 0:
        return sorted(features)[:max_features]
    return sorted(features)


def _fit_predict(
    market: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    x_train = train_df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_test = test_df[features].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    if market == "spread":
        y_train = pd.to_numeric(train_df["actual_margin"], errors="coerce")
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)
        pred_margin = pd.Series(model.predict(x_test), index=x_test.index)
        # Convert predicted margin to spread convention.
        return -pred_margin

    if market == "total":
        y_train = pd.to_numeric(train_df["actual_total"], errors="coerce")
        model = Ridge(alpha=1.0)
        model.fit(x_train, y_train)
        return pd.Series(model.predict(x_test), index=x_test.index)

    if market == "ml":
        y_train = pd.to_numeric(train_df["home_won"], errors="coerce").fillna(0).astype(int)
        model = LogisticRegression(max_iter=500)
        model.fit(x_train, y_train)
        prob = model.predict_proba(x_test)[:, 1]
        return pd.Series(prob, index=x_test.index)

    raise ValueError(f"Unsupported market: {market}")


def _score_objective(metrics: dict[str, float], market: str) -> float:
    if market in {"spread", "total"}:
        if pd.notna(metrics.get("hit_rate")):
            return float(metrics["hit_rate"])
        if pd.notna(metrics.get("mae")):
            return float(-metrics["mae"])
        return float("nan")
    if market == "ml":
        if pd.notna(metrics.get("brier")):
            return float(-metrics["brier"])
        return float("nan")
    return float("nan")


def _evaluate_fold(
    market: str,
    test_df: pd.DataFrame,
    y_pred: pd.Series,
) -> dict[str, float]:
    if market == "spread":
        return evaluate_predictions(
            y_true=test_df["actual_margin"],
            y_pred=y_pred,
            market_line=test_df["spread_line"],
            odds=None,
            market="spread",
            line_open=test_df.get("spread_open"),
            line_close=test_df.get("spread_close"),
        )
    if market == "total":
        return evaluate_predictions(
            y_true=test_df["actual_total"],
            y_pred=y_pred,
            market_line=test_df["total_line"],
            odds=None,
            market="total",
            line_open=test_df.get("total_open"),
            line_close=test_df.get("total_close"),
        )
    if market == "ml":
        return evaluate_predictions(
            y_true=test_df["home_won"],
            y_pred=y_pred,
            market_line=None,
            odds={"home_ml": test_df.get("home_ml"), "away_ml": test_df.get("away_ml")},
            market="ml",
        )
    raise ValueError(f"Unsupported market: {market}")


def _univariate_value(market: str, train_df: pd.DataFrame, feature: str) -> float:
    x = pd.to_numeric(train_df[feature], errors="coerce")
    if market == "spread":
        y = pd.to_numeric(train_df["actual_margin"], errors="coerce")
    elif market == "total":
        y = pd.to_numeric(train_df["actual_total"], errors="coerce")
    else:
        y = pd.to_numeric(train_df["home_won"], errors="coerce")

    mask = x.notna() & y.notna()
    if mask.sum() < 20:
        return float("nan")
    return float(x[mask].corr(y[mask], method="spearman"))


def _feature_groups(features: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for feature in features:
        token = feature.split("_", 1)[0]
        groups.setdefault(token, []).append(feature)
    return groups


def _stability_score(values: list[float]) -> tuple[float, float]:
    vals = np.array([v for v in values if pd.notna(v)], dtype=float)
    if vals.size == 0:
        return float("nan"), float("nan")

    signs = np.sign(vals)
    signs = signs[signs != 0]
    if signs.size == 0:
        sign_consistency = 0.0
    else:
        sign_consistency = float(abs(signs.mean()))

    magnitudes = np.abs(vals)
    mean_mag = float(magnitudes.mean())
    std_mag = float(magnitudes.std(ddof=0))
    cv = std_mag / (mean_mag + 1e-9)
    stability = max(0.0, 1.0 - cv) * sign_consistency
    return sign_consistency, float(stability * 100.0)


def run_feature_tests(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    market: str,
    random_seed: int = 42,
    max_features: int | None = None,
) -> FeatureTestResult:
    blocked: list[str] = []
    if frame.empty:
        blocked.append("frame_empty")
        return FeatureTestResult(pd.DataFrame(), pd.DataFrame(), blocked)
    if not folds:
        blocked.append("no_folds")
        return FeatureTestResult(pd.DataFrame(), pd.DataFrame(), blocked)

    features = _feature_columns(frame, max_features=max_features)
    if not features:
        blocked.append("no_eligible_features")
        return FeatureTestResult(pd.DataFrame(), pd.DataFrame(), blocked)

    groups = _feature_groups(features)
    rng = np.random.default_rng(random_seed)

    rows: list[dict[str, Any]] = []

    for fold in folds:
        train_df = _safe_fold_slice(frame, fold.train_index)
        test_df = _safe_fold_slice(frame, fold.test_index)
        if train_df.empty or test_df.empty:
            blocked.append(f"empty_fold:{fold.fold_id}")
            continue

        try:
            base_pred = _fit_predict(market, train_df, test_df, features)
            base_metrics = _evaluate_fold(market, test_df, base_pred)
            base_score = _score_objective(base_metrics, market)
        except Exception:
            blocked.append(f"baseline_failed:{fold.fold_id}")
            continue

        for feature in features:
            uni = _univariate_value(market, train_df, feature)

            # Permutation importance.
            perm_test = test_df.copy()
            perm_test[feature] = rng.permutation(perm_test[feature].to_numpy())
            try:
                perm_pred = _fit_predict(market, train_df, perm_test, features)
                perm_metrics = _evaluate_fold(market, perm_test, perm_pred)
                perm_score = _score_objective(perm_metrics, market)
                perm_delta = base_score - perm_score
            except Exception:
                perm_delta = float("nan")

            # Drop-one single feature.
            drop_features = [f for f in features if f != feature]
            try:
                drop_pred = _fit_predict(market, train_df, test_df, drop_features)
                drop_metrics = _evaluate_fold(market, test_df, drop_pred)
                drop_score = _score_objective(drop_metrics, market)
                drop_single_delta = base_score - drop_score
            except Exception:
                drop_single_delta = float("nan")

            # Drop-one group.
            group = feature.split("_", 1)[0]
            group_features = [f for f in features if f not in groups.get(group, [])]
            try:
                group_pred = _fit_predict(market, train_df, test_df, group_features)
                group_metrics = _evaluate_fold(market, test_df, group_pred)
                group_score = _score_objective(group_metrics, market)
                drop_group_delta = base_score - group_score
            except Exception:
                drop_group_delta = float("nan")

            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "market": market,
                    "feature_name": feature,
                    "feature_group": group,
                    "train_size": fold.train_size,
                    "test_size": fold.test_size,
                    "univariate_value": uni,
                    "permutation_delta": perm_delta,
                    "drop_one_single_delta": drop_single_delta,
                    "drop_one_group_delta": drop_group_delta,
                    "baseline_score": base_score,
                }
            )

    stability_df = pd.DataFrame(rows)
    if stability_df.empty:
        blocked.append("feature_tests_no_rows")
        return FeatureTestResult(pd.DataFrame(), stability_df, sorted(set(blocked)))

    score_rows = []
    for feature, group in stability_df.groupby("feature_name", dropna=False):
        perm_values = group["permutation_delta"].dropna().tolist()
        sign_consistency, stability_score = _stability_score(perm_values)
        score_rows.append(
            {
                "market": market,
                "feature_name": feature,
                "feature_group": group["feature_group"].iloc[0],
                "n_folds": int(group["fold_id"].nunique()),
                "univariate_mean": float(group["univariate_value"].mean(skipna=True)),
                "univariate_std": float(group["univariate_value"].std(skipna=True, ddof=0)),
                "permutation_delta_mean": float(group["permutation_delta"].mean(skipna=True)),
                "permutation_delta_std": float(group["permutation_delta"].std(skipna=True, ddof=0)),
                "drop_one_single_mean": float(group["drop_one_single_delta"].mean(skipna=True)),
                "drop_one_single_std": float(group["drop_one_single_delta"].std(skipna=True, ddof=0)),
                "drop_one_group_mean": float(group["drop_one_group_delta"].mean(skipna=True)),
                "drop_one_group_std": float(group["drop_one_group_delta"].std(skipna=True, ddof=0)),
                "sign_consistency": sign_consistency,
                "stability_score": stability_score,
            }
        )

    scorecard_df = pd.DataFrame(score_rows).sort_values(
        ["stability_score", "permutation_delta_mean"], ascending=[False, False]
    )

    return FeatureTestResult(
        feature_scorecard=scorecard_df.reset_index(drop=True),
        feature_stability=stability_df.reset_index(drop=True),
        blocked_reasons=sorted(set(blocked)),
    )
