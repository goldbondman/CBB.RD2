from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from .config import ModelLabConfig
from .evaluators import evaluate_predictions
from .splits import Fold


WINDOW_CONFIGS: dict[str, tuple[int, ...]] = {
    "W_4_8": (4, 8),
    "W_4_12": (4, 12),
    "W_4_8_12": (4, 8, 12),
    "W_5_10": (5, 10),
    "W_6_11": (6, 11),
    "W_7_12": (7, 12),
}

WINDOW_TOKEN_RE = re.compile(r"L(\d+)")

EXCLUDE_FEATURE_COLS = {
    "season",
    "season_id",
    "team_id",
    "team_id_A",
    "team_id_B",
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

COMMON_KEEP_COLS = [
    "season_id",
    "game_id",
    "event_id",
    "game_datetime_utc",
    "game_date",
    "home_team_id",
    "away_team_id",
    "neutral_site",
]

MARKET_KEEP_COLS = {
    "spread": ["actual_margin", "spread_line", "spread_open", "spread_close"],
    "total": ["actual_total", "total_line", "total_open", "total_close"],
    "ml": ["home_won", "home_ml", "away_ml"],
}


@dataclass
class WindowGridResult:
    window_grid_scorecard: pd.DataFrame
    window_contract: dict[str, dict[str, Any]]
    blocked_reasons: list[str]


def _feature_columns(df: pd.DataFrame) -> list[str]:
    features: list[str] = []
    for col in df.columns:
        if col in EXCLUDE_FEATURE_COLS:
            continue
        if col.endswith("_source"):
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() < 20:
            continue
        features.append(col)
    return sorted(features)


def _window_ids_for_feature(feature_name: str) -> set[int]:
    out: set[int] = set()
    for token in WINDOW_TOKEN_RE.findall(str(feature_name)):
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def _window_feature_columns(features: list[str], window_ids: tuple[int, ...]) -> list[str]:
    allowed = set(window_ids)
    selected: list[str] = []
    for feature in features:
        ids = _window_ids_for_feature(feature)
        if not ids:
            continue
        if ids.issubset(allowed):
            selected.append(feature)
    return selected


def _available_window_ids(features: list[str]) -> set[int]:
    out: set[int] = set()
    for feature in features:
        out.update(_window_ids_for_feature(feature))
    return out


def _build_window_frame(frame: pd.DataFrame, market: str, features: list[str]) -> pd.DataFrame:
    keep_cols = [col for col in COMMON_KEEP_COLS + MARKET_KEEP_COLS.get(market, []) if col in frame.columns]
    keep_cols = keep_cols + [col for col in features if col in frame.columns]
    if not keep_cols:
        return pd.DataFrame(index=frame.index)
    return frame[keep_cols].copy()


def _safe_fold_slice(df: pd.DataFrame, idx: list[int]) -> pd.DataFrame:
    present = [i for i in idx if i in df.index]
    return df.loc[present].copy()


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


def _evaluate_fold(
    market: str,
    test_df: pd.DataFrame,
    y_pred: pd.Series,
    default_odds: int,
) -> dict[str, float]:
    if market == "spread":
        return evaluate_predictions(
            y_true=test_df["actual_margin"],
            y_pred=y_pred,
            market_line=test_df["spread_line"],
            odds=None,
            market="spread",
            default_odds=default_odds,
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
            default_odds=default_odds,
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
            default_odds=default_odds,
        )
    raise ValueError(f"Unsupported market: {market}")


def _aggregate_fold_metrics(rows: list[dict[str, Any]], market: str) -> dict[str, float]:
    if not rows:
        return {
            "n_folds": 0.0,
            "n_games": 0.0,
            "hit_rate": float("nan"),
            "roi": float("nan"),
            "roi_fold_var": float("nan"),
            "hit_rate_fold_var": float("nan"),
            "mae": float("nan"),
            "brier": float("nan"),
            "score": float("nan"),
        }

    df = pd.DataFrame(rows)
    games = pd.to_numeric(df.get("graded_n", 0), errors="coerce").fillna(0.0)

    def weighted(col: str) -> float:
        vals = pd.to_numeric(df.get(col), errors="coerce")
        mask = vals.notna() & (games > 0)
        if mask.sum() == 0:
            return float("nan")
        return float((vals[mask] * games[mask]).sum() / games[mask].sum())

    agg = {
        "n_folds": float(df["fold_id"].nunique()),
        "n_games": float(games.sum()),
        "hit_rate": weighted("hit_rate"),
        "roi": weighted("roi"),
        "roi_fold_var": float(pd.to_numeric(df.get("roi"), errors="coerce").dropna().var(ddof=0))
        if pd.to_numeric(df.get("roi"), errors="coerce").notna().any()
        else float("nan"),
        "hit_rate_fold_var": float(pd.to_numeric(df.get("hit_rate"), errors="coerce").dropna().var(ddof=0))
        if pd.to_numeric(df.get("hit_rate"), errors="coerce").notna().any()
        else float("nan"),
        "mae": weighted("mae"),
        "brier": weighted("brier"),
    }
    if market == "ml":
        agg["score"] = float(-agg["brier"]) if pd.notna(agg["brier"]) else float("nan")
    elif pd.notna(agg["hit_rate"]):
        agg["score"] = float(agg["hit_rate"])
    elif pd.notna(agg["mae"]):
        agg["score"] = float(-agg["mae"])
    else:
        agg["score"] = float("nan")
    return agg


def evaluate_window_grid(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    market: str,
    config: ModelLabConfig,
) -> WindowGridResult:
    blocked: list[str] = []
    all_features = _feature_columns(frame)
    available_window_ids = _available_window_ids(all_features)

    window_contract: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for window_name, window_ids in WINDOW_CONFIGS.items():
        missing_ids = [window for window in window_ids if window not in available_window_ids]
        window_contract[window_name] = {
            "window_ids": list(window_ids),
            "window_identifiers": [f"L{window}" for window in window_ids],
            "required_suffixes": [f"_L{window}" for window in window_ids],
            "missing_window_identifiers": [f"L{window}" for window in missing_ids],
        }

        if missing_ids:
            blocked.append(
                f"window_grid_blocked:{market}:{window_name}:missing_window_identifiers="
                f"{','.join(f'L{window}' for window in missing_ids)}"
            )
            rows.append(
                {
                    "market": market,
                    "window_config": window_name,
                    "window_ids": "/".join(str(window) for window in window_ids),
                    "window_identifiers": "/".join(f"L{window}" for window in window_ids),
                    "status": "BLOCKED",
                    "blocked_reason": "missing_window_identifiers",
                    "missing_window_identifiers": ",".join(f"L{window}" for window in missing_ids),
                    "feature_count": 0,
                    "n_folds": 0.0,
                    "n_games": 0.0,
                    "hit_rate": float("nan"),
                    "roi": float("nan"),
                    "roi_fold_var": float("nan"),
                    "hit_rate_fold_var": float("nan"),
                    "mae": float("nan"),
                    "brier": float("nan"),
                    "score": float("nan"),
                }
            )
            continue

        selected_features = _window_feature_columns(all_features, window_ids)
        if not selected_features:
            blocked.append(f"window_grid_blocked:{market}:{window_name}:no_matching_window_features")
            rows.append(
                {
                    "market": market,
                    "window_config": window_name,
                    "window_ids": "/".join(str(window) for window in window_ids),
                    "window_identifiers": "/".join(f"L{window}" for window in window_ids),
                    "status": "BLOCKED",
                    "blocked_reason": "no_matching_window_features",
                    "missing_window_identifiers": "",
                    "feature_count": 0,
                    "n_folds": 0.0,
                    "n_games": 0.0,
                    "hit_rate": float("nan"),
                    "roi": float("nan"),
                    "roi_fold_var": float("nan"),
                    "hit_rate_fold_var": float("nan"),
                    "mae": float("nan"),
                    "brier": float("nan"),
                    "score": float("nan"),
                }
            )
            continue

        window_frame = _build_window_frame(frame, market, selected_features)
        fold_rows: list[dict[str, Any]] = []
        for fold in folds:
            train_df = _safe_fold_slice(window_frame, fold.train_index)
            test_df = _safe_fold_slice(window_frame, fold.test_index)
            if train_df.empty or test_df.empty:
                continue
            try:
                pred = _fit_predict(market, train_df, test_df, selected_features)
                metrics = _evaluate_fold(market, test_df, pred, config.default_odds)
                metrics["fold_id"] = fold.fold_id
                fold_rows.append(metrics)
            except Exception:
                continue

        agg = _aggregate_fold_metrics(fold_rows, market)
        rows.append(
            {
                "market": market,
                "window_config": window_name,
                "window_ids": "/".join(str(window) for window in window_ids),
                "window_identifiers": "/".join(f"L{window}" for window in window_ids),
                "status": "OK",
                "blocked_reason": "",
                "missing_window_identifiers": "",
                "feature_count": int(len(selected_features)),
                "n_folds": agg["n_folds"],
                "n_games": agg["n_games"],
                "hit_rate": agg["hit_rate"],
                "roi": agg["roi"],
                "roi_fold_var": agg["roi_fold_var"],
                "hit_rate_fold_var": agg["hit_rate_fold_var"],
                "mae": agg["mae"],
                "brier": agg["brier"],
                "score": agg["score"],
            }
        )

    scorecard = pd.DataFrame(rows)
    if not scorecard.empty:
        scorecard = scorecard.sort_values(["status", "score"], ascending=[True, False]).reset_index(drop=True)

    return WindowGridResult(
        window_grid_scorecard=scorecard,
        window_contract=window_contract,
        blocked_reasons=sorted(set(blocked)),
    )
