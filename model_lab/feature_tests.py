from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass
import re
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
    window_grid_scorecard: pd.DataFrame
    feature_set_report: str
    selected_feature_sets: dict[str, list[str]]
    location_aware_variants: dict[str, dict[str, Any]]
    window_contract: dict[str, dict[str, Any]]
    selector_config: dict[str, float | int]
    blocked_reasons: list[str]


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

WINDOW_GRID_CONTRACT: dict[str, tuple[int, ...]] = {
    "W_4_8": (4, 8),
    "W_4_12": (4, 12),
    "W_4_8_12": (4, 8, 12),
    "W_5_10": (5, 10),
    "W_6_11": (6, 11),
    "W_7_12": (7, 12),
}

WINDOW_TOKEN_RE = re.compile(r"L(\d+)")


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


def _window_ids_for_feature(feature_name: str) -> set[int]:
    out: set[int] = set()
    for token in WINDOW_TOKEN_RE.findall(str(feature_name)):
        try:
            out.add(int(token))
        except ValueError:
            continue
    return out


def _available_window_ids(features: list[str]) -> set[int]:
    out: set[int] = set()
    for feature in features:
        out.update(_window_ids_for_feature(feature))
    return out


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
    return sign_consistency, float(stability)


def _cluster_features(
    ordered_candidates: list[str],
    corr: pd.DataFrame,
    correlation_max: float,
) -> list[list[str]]:
    if not ordered_candidates:
        return []
    if corr.empty:
        return [[feature] for feature in ordered_candidates]

    components: list[list[str]] = []
    seen: set[str] = set()
    valid = set(ordered_candidates)

    for feature in ordered_candidates:
        if feature in seen:
            continue
        stack = [feature]
        component: list[str] = []
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            if node not in corr.index:
                continue
            neighbors = corr.loc[node]
            for neighbor, value in neighbors.items():
                if neighbor == node or neighbor in seen or neighbor not in valid:
                    continue
                if pd.notna(value) and float(value) > float(correlation_max):
                    stack.append(str(neighbor))
        components.append(component)

    return components


def _apply_selector_rules(
    scorecard_df: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    stability_min: float,
    sign_consistency_min: float,
    permutation_delta_min: float,
    ablation_delta_min: float,
    correlation_max: float,
    cap_conservative: int,
    cap_balanced: int,
    cap_aggressive: int,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    scorecard = scorecard_df.copy()
    if scorecard.empty:
        return scorecard, {"conservative": [], "balanced": [], "aggressive": []}

    for col in (
        "stability_score",
        "sign_consistency",
        "permutation_delta_mean",
        "drop_one_single_mean",
        "drop_one_group_mean",
        "univariate_mean",
    ):
        if col not in scorecard.columns:
            scorecard[col] = np.nan
        scorecard[col] = pd.to_numeric(scorecard[col], errors="coerce")

    scorecard["ablation_impact_mean"] = (
        pd.concat([scorecard["drop_one_single_mean"], scorecard["drop_one_group_mean"]], axis=1)
        .max(axis=1, skipna=True)
        .astype(float)
    )

    perm_rank = scorecard["permutation_delta_mean"].rank(pct=True, method="average").fillna(0.0)
    ablation_rank = scorecard["ablation_impact_mean"].rank(pct=True, method="average").fillna(0.0)
    stability_rank = scorecard["stability_score"].rank(pct=True, method="average").fillna(0.0)
    sign_rank = scorecard["sign_consistency"].rank(pct=True, method="average").fillna(0.0)
    scorecard["selector_rank_score"] = (
        (0.40 * perm_rank) + (0.25 * ablation_rank) + (0.20 * stability_rank) + (0.15 * sign_rank)
    ).astype(float)

    passes_stability = scorecard["stability_score"] >= float(stability_min)
    passes_sign = scorecard["sign_consistency"] >= float(sign_consistency_min)
    passes_perm = scorecard["permutation_delta_mean"] >= float(permutation_delta_min)
    passes_ablation = scorecard["ablation_impact_mean"] >= float(ablation_delta_min)
    passes_signal = passes_perm | passes_ablation
    scorecard["passes_selector_thresholds"] = (passes_stability & passes_sign & passes_signal).fillna(False)

    threshold_fail_reasons: dict[str, str] = {}
    for row in scorecard.itertuples(index=False):
        feature = str(row.feature_name)
        reasons: list[str] = []
        if not (pd.notna(row.stability_score) and float(row.stability_score) >= float(stability_min)):
            reasons.append(f"stability<{stability_min:.2f}")
        if not (pd.notna(row.sign_consistency) and float(row.sign_consistency) >= float(sign_consistency_min)):
            reasons.append(f"sign_consistency<{sign_consistency_min:.2f}")
        perm_ok = pd.notna(row.permutation_delta_mean) and float(row.permutation_delta_mean) >= float(permutation_delta_min)
        abl_ok = pd.notna(row.ablation_impact_mean) and float(row.ablation_impact_mean) >= float(ablation_delta_min)
        if not (perm_ok or abl_ok):
            reasons.append(f"signal<{permutation_delta_min:.3f}|{ablation_delta_min:.3f}")
        threshold_fail_reasons[feature] = "; ".join(reasons)

    candidates = scorecard.loc[scorecard["passes_selector_thresholds"]].copy()
    candidates = candidates.sort_values(
        [
            "selector_rank_score",
            "permutation_delta_mean",
            "ablation_impact_mean",
            "stability_score",
            "sign_consistency",
            "univariate_mean",
        ],
        ascending=[False, False, False, False, False, False],
    )

    candidate_features = candidates["feature_name"].astype(str).tolist()
    corr_anchor_for: dict[str, str] = {}
    corr_cluster_for: dict[str, str] = {}
    corr_pruned_value: dict[str, float] = {}
    kept_features: list[str] = []

    corr = pd.DataFrame()
    if candidate_features:
        corr_input = frame[candidate_features].apply(pd.to_numeric, errors="coerce")
        corr = corr_input.corr(method="spearman").abs()

    components = _cluster_features(candidate_features, corr, float(correlation_max))
    anchor_rows: list[tuple[str, float, float]] = []
    for idx, component in enumerate(components, start=1):
        component_df = candidates[candidates["feature_name"].isin(component)].copy()
        component_df = component_df.sort_values(
            ["selector_rank_score", "permutation_delta_mean", "ablation_impact_mean"],
            ascending=[False, False, False],
        )
        if component_df.empty:
            continue
        anchor = str(component_df.iloc[0]["feature_name"])
        cluster_id = f"C{idx:03d}"
        for feature in component:
            corr_anchor_for[feature] = anchor
            corr_cluster_for[feature] = cluster_id
            if feature == anchor:
                continue
            if corr.empty or feature not in corr.index or anchor not in corr.columns:
                corr_pruned_value[feature] = float("nan")
            else:
                corr_pruned_value[feature] = float(corr.loc[feature, anchor])
        anchor_rows.append(
            (
                anchor,
                float(component_df.iloc[0]["selector_rank_score"]),
                float(component_df.iloc[0]["permutation_delta_mean"]),
            )
        )

    anchor_rows = sorted(anchor_rows, key=lambda row: (row[1], row[2]), reverse=True)
    kept_features = [row[0] for row in anchor_rows]

    conservative = kept_features[: max(0, int(cap_conservative))]
    balanced = kept_features[: max(0, int(cap_balanced))]
    aggressive = kept_features[: max(0, int(cap_aggressive))]

    selected_sets = {
        "conservative": conservative,
        "balanced": balanced,
        "aggressive": aggressive,
    }

    scorecard["correlation_cluster_id"] = scorecard["feature_name"].map(corr_cluster_for).fillna("")
    scorecard["correlation_anchor_feature"] = scorecard["feature_name"].map(corr_anchor_for).fillna("")
    scorecard["correlation_with_anchor"] = scorecard["feature_name"].map(corr_pruned_value)
    scorecard["is_cluster_anchor"] = scorecard["feature_name"] == scorecard["correlation_anchor_feature"]
    scorecard["selected_conservative"] = scorecard["feature_name"].isin(conservative)
    scorecard["selected_balanced"] = scorecard["feature_name"].isin(balanced)
    scorecard["selected_aggressive"] = scorecard["feature_name"].isin(aggressive)

    exclusion_reasons: list[str] = []
    for row in scorecard.itertuples(index=False):
        feature = str(row.feature_name)
        if feature in aggressive:
            exclusion_reasons.append("")
            continue
        if feature in corr_anchor_for and feature != corr_anchor_for[feature]:
            anchor = corr_anchor_for[feature]
            corr_value = corr_pruned_value.get(feature, float("nan"))
            exclusion_reasons.append(
                f"corr>{correlation_max:.2f} with {anchor} ({corr_value:.3f})"
            )
            continue
        exclusion_reasons.append(threshold_fail_reasons.get(feature, "not_selected"))
    scorecard["exclusion_reason"] = exclusion_reasons
    scorecard["selection_status"] = np.where(scorecard["selected_aggressive"], "selected", "removed")

    scorecard = scorecard.sort_values(
        [
            "selected_balanced",
            "selected_aggressive",
            "passes_selector_thresholds",
            "selector_rank_score",
            "permutation_delta_mean",
            "stability_score",
        ],
        ascending=[False, False, False, False, False, False],
    ).reset_index(drop=True)
    return scorecard, selected_sets


def _filter_features_for_window_config(features: list[str], window_ids: tuple[int, ...]) -> list[str]:
    allowed = set(window_ids)
    filtered: list[str] = []
    for feature in features:
        ids = _window_ids_for_feature(feature)
        if not ids:
            filtered.append(feature)
            continue
        if ids.issubset(allowed):
            filtered.append(feature)
    return filtered


def _aggregate_eval_rows(eval_rows: list[dict[str, Any]], market: str) -> dict[str, float]:
    if not eval_rows:
        return {
            "n_folds": 0.0,
            "n_games": 0.0,
            "hit_rate": float("nan"),
            "roi": float("nan"),
            "mae": float("nan"),
            "brier": float("nan"),
            "score": float("nan"),
        }

    df = pd.DataFrame(eval_rows)
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


def _evaluate_feature_set(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    market: str,
    features: list[str],
) -> dict[str, float]:
    if not features:
        return {
            "n_folds": 0.0,
            "n_games": 0.0,
            "hit_rate": float("nan"),
            "roi": float("nan"),
            "mae": float("nan"),
            "brier": float("nan"),
            "score": float("nan"),
        }

    eval_rows: list[dict[str, Any]] = []
    for fold in folds:
        train_df = _safe_fold_slice(frame, fold.train_index)
        test_df = _safe_fold_slice(frame, fold.test_index)
        if train_df.empty or test_df.empty:
            continue
        try:
            pred = _fit_predict(market, train_df, test_df, features)
            metrics = _evaluate_fold(market, test_df, pred)
            metrics["fold_id"] = fold.fold_id
            eval_rows.append(metrics)
        except Exception:
            continue

    return _aggregate_eval_rows(eval_rows, market)


def _build_window_contract(features: list[str]) -> dict[str, dict[str, Any]]:
    available_ids = _available_window_ids(features)
    contract: dict[str, dict[str, Any]] = {}
    for key, windows in WINDOW_GRID_CONTRACT.items():
        missing = [window for window in windows if window not in available_ids]
        contract[key] = {
            "window_ids": list(windows),
            "window_identifiers": [f"L{window}" for window in windows],
            "required_suffixes": [f"_L{window}" for window in windows],
            "missing_window_identifiers": [f"L{window}" for window in missing],
        }
    return contract


def _build_window_grid_scorecard(
    *,
    frame: pd.DataFrame,
    folds: list[Fold],
    market: str,
    features: list[str],
    scorecard_df: pd.DataFrame,
    stability_min: float,
    sign_consistency_min: float,
    permutation_delta_min: float,
    ablation_delta_min: float,
    correlation_max: float,
    cap_conservative: int,
    cap_balanced: int,
    cap_aggressive: int,
) -> tuple[pd.DataFrame, list[str]]:
    blocked: list[str] = []
    rows: list[dict[str, Any]] = []
    available_ids = _available_window_ids(features)

    for window_key, window_ids in WINDOW_GRID_CONTRACT.items():
        missing = [window for window in window_ids if window not in available_ids]
        if missing:
            blocked.append(
                f"window_grid_blocked:{window_key}:missing_window_identifiers={','.join(f'L{window}' for window in missing)}"
            )
            rows.append(
                {
                    "market": market,
                    "window_config": window_key,
                    "window_ids": "/".join(str(window) for window in window_ids),
                    "window_identifiers": "/".join(f"L{window}" for window in window_ids),
                    "status": "BLOCKED",
                    "blocked_reason": "missing_window_identifiers",
                    "missing_window_identifiers": ",".join(f"L{window}" for window in missing),
                    "feature_pool_n": 0,
                    "selected_conservative_n": 0,
                    "selected_balanced_n": 0,
                    "selected_aggressive_n": 0,
                    "eval_n_folds": 0.0,
                    "eval_n_games": 0.0,
                    "eval_hit_rate": float("nan"),
                    "eval_roi": float("nan"),
                    "eval_mae": float("nan"),
                    "eval_brier": float("nan"),
                    "eval_score": float("nan"),
                }
            )
            continue

        filtered_features = _filter_features_for_window_config(features, window_ids)
        filtered_scorecard = scorecard_df[scorecard_df["feature_name"].isin(filtered_features)].copy()
        _, selected = _apply_selector_rules(
            filtered_scorecard,
            frame,
            stability_min=stability_min,
            sign_consistency_min=sign_consistency_min,
            permutation_delta_min=permutation_delta_min,
            ablation_delta_min=ablation_delta_min,
            correlation_max=correlation_max,
            cap_conservative=cap_conservative,
            cap_balanced=cap_balanced,
            cap_aggressive=cap_aggressive,
        )
        balanced = selected.get("balanced", [])
        eval_metrics = _evaluate_feature_set(frame, folds, market=market, features=balanced)
        rows.append(
            {
                "market": market,
                "window_config": window_key,
                "window_ids": "/".join(str(window) for window in window_ids),
                "window_identifiers": "/".join(f"L{window}" for window in window_ids),
                "status": "OK",
                "blocked_reason": "",
                "missing_window_identifiers": "",
                "feature_pool_n": int(len(filtered_features)),
                "selected_conservative_n": int(len(selected.get("conservative", []))),
                "selected_balanced_n": int(len(selected.get("balanced", []))),
                "selected_aggressive_n": int(len(selected.get("aggressive", []))),
                "eval_n_folds": float(eval_metrics["n_folds"]),
                "eval_n_games": float(eval_metrics["n_games"]),
                "eval_hit_rate": float(eval_metrics["hit_rate"]),
                "eval_roi": float(eval_metrics["roi"]),
                "eval_mae": float(eval_metrics["mae"]),
                "eval_brier": float(eval_metrics["brier"]),
                "eval_score": float(eval_metrics["score"]),
            }
        )

    grid = pd.DataFrame(rows)
    if not grid.empty:
        grid = grid.sort_values(["status", "eval_score", "selected_balanced_n"], ascending=[True, False, False]).reset_index(drop=True)
    return grid, sorted(set(blocked))


def _build_location_aware_variants(
    frame: pd.DataFrame,
    selected_sets: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    cols = set(frame.columns)
    variants: dict[str, dict[str, Any]] = {}

    for profile, features in selected_sets.items():
        missing: list[str] = []
        selected_split_cols: list[str] = []
        for feature in features:
            if feature.startswith("home_") or feature.startswith("away_"):
                selected_split_cols.append(feature)
                continue
            home_col = f"home_{feature}"
            away_col = f"away_{feature}"
            if home_col in cols and away_col in cols:
                selected_split_cols.extend([home_col, away_col])
                continue
            if home_col not in cols:
                missing.append(home_col)
            if away_col not in cols:
                missing.append(away_col)

        selected_split_cols = sorted(set(selected_split_cols))
        missing = sorted(set(missing))
        if not features:
            status = "BLOCKED"
            reason = "no_base_features_selected"
        elif missing:
            status = "BLOCKED"
            reason = "missing_home_away_split_columns"
        else:
            status = "ACTIVE"
            reason = ""

        variants[profile] = {
            "status": status,
            "reason": reason,
            "base_features": list(features),
            "location_features": selected_split_cols,
            "missing_columns": missing,
        }

    return variants


def _render_feature_set_report(
    *,
    market: str,
    scorecard_df: pd.DataFrame,
    window_grid_df: pd.DataFrame,
    selected_sets: dict[str, list[str]],
    selector_config: dict[str, float | int],
    window_contract: dict[str, dict[str, Any]],
    location_aware_variants: dict[str, dict[str, Any]],
) -> str:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines: list[str] = [
        f"# Feature Set Report ({market})",
        "",
        f"Generated (UTC): `{generated_at}`",
        "",
        "## Selector Rules (AUTO_V2)",
        f"- StabilityScore >= `{float(selector_config['stability_min']):.2f}`",
        f"- SignConsistency >= `{float(selector_config['sign_consistency_min']):.2f}`",
        f"- Permutation delta >= `{float(selector_config['permutation_delta_min']):.4f}`",
        f"- Ablation impact >= `{float(selector_config['ablation_delta_min']):.4f}`",
        f"- Correlation pruning/cluster threshold: `>{float(selector_config['correlation_max']):.2f}`",
        (
            "- Size caps: "
            f"conservative={int(selector_config['cap_conservative'])}, "
            f"balanced={int(selector_config['cap_balanced'])}, "
            f"aggressive={int(selector_config['cap_aggressive'])}"
        ),
        "",
        "## Window Config Contract",
    ]

    for key in WINDOW_GRID_CONTRACT:
        cfg = window_contract.get(key, {})
        lines.append(
            f"- `{key}` -> identifiers `{','.join(cfg.get('window_identifiers', []))}`; "
            f"suffixes `{','.join(cfg.get('required_suffixes', []))}`"
        )
        missing = cfg.get("missing_window_identifiers", [])
        if missing:
            lines.append(f"  missing: `{','.join(missing)}`")

    lines.append("")
    lines.append("## Selected Feature Sets")
    for profile in ("conservative", "balanced", "aggressive"):
        features = selected_sets.get(profile, [])
        lines.append(f"- `{profile}` ({len(features)}): {', '.join(features) if features else 'none'}")

    lines.append("")
    lines.append("## Location-Aware Variant")
    for profile in ("conservative", "balanced", "aggressive"):
        payload = location_aware_variants.get(profile, {})
        lines.append(f"- `{profile}`: `{payload.get('status', 'BLOCKED')}`")
        if payload.get("status") == "BLOCKED":
            missing = payload.get("missing_columns", [])
            if missing:
                lines.append(f"  missing columns: `{', '.join(missing)}`")
            else:
                lines.append(f"  reason: `{payload.get('reason', 'unknown')}`")
        else:
            location_features = payload.get("location_features", [])
            lines.append(f"  location split columns used: `{', '.join(location_features)}`")

    lines.append("")
    lines.append("## Window Grid Summary")
    if window_grid_df.empty:
        lines.append("- no rows")
    else:
        for row in window_grid_df.itertuples(index=False):
            lines.append(
                f"- `{row.window_config}` status={row.status}, "
                f"missing=`{row.missing_window_identifiers}`, "
                f"balanced_n={int(row.selected_balanced_n)}, score={row.eval_score}"
            )

    lines.append("")
    lines.append("## Top Balanced Features")
    if scorecard_df.empty:
        lines.append("- scorecard empty")
    else:
        top = scorecard_df[scorecard_df["selected_balanced"].astype(bool)].head(20)
        if top.empty:
            lines.append("- no balanced features selected")
        else:
            for row in top.itertuples(index=False):
                lines.append(
                    f"- `{row.feature_name}` "
                    f"(perm={row.permutation_delta_mean:.4f}, stab={row.stability_score:.2f}, sign={row.sign_consistency:.2f})"
                )
    return "\n".join(lines) + "\n"


def build_generated_feature_set_payloads(
    *,
    market: str,
    selected_sets: dict[str, list[str]],
    selector_config: dict[str, float | int],
    window_contract: dict[str, dict[str, Any]],
    window_grid_scorecard: pd.DataFrame,
    location_aware_variants: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    best_window = ""
    if not window_grid_scorecard.empty:
        ok_rows = window_grid_scorecard[window_grid_scorecard["status"] == "OK"].copy()
        if not ok_rows.empty:
            ok_rows = ok_rows.sort_values(["eval_score", "selected_balanced_n"], ascending=[False, False])
            best_window = str(ok_rows.iloc[0]["window_config"])

    payloads: dict[str, dict[str, Any]] = {}
    for profile in ("conservative", "balanced", "aggressive"):
        payloads[profile] = {
            "selector_version": "AUTO_V2",
            "market": market,
            "profile": profile,
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "window_grid_best": best_window,
            "window_contract": window_contract,
            "selector_rules": selector_config,
            "features": selected_sets.get(profile, []),
            "feature_count": int(len(selected_sets.get(profile, []))),
            "location_aware_variant": location_aware_variants.get(profile, {}),
        }
    return payloads


def run_feature_tests(
    frame: pd.DataFrame,
    folds: list[Fold],
    *,
    market: str,
    random_seed: int = 42,
    max_features: int | None = None,
    stability_min: float = 0.60,
    sign_consistency_min: float = 0.70,
    permutation_delta_min: float = 0.001,
    ablation_delta_min: float = 0.001,
    correlation_max: float = 0.85,
    cap_conservative: int = 12,
    cap_balanced: int = 18,
    cap_aggressive: int = 25,
) -> FeatureTestResult:
    blocked: list[str] = []
    empty_sets = {"conservative": [], "balanced": [], "aggressive": []}
    selector_config: dict[str, float | int] = {
        "stability_min": float(stability_min),
        "sign_consistency_min": float(sign_consistency_min),
        "permutation_delta_min": float(permutation_delta_min),
        "ablation_delta_min": float(ablation_delta_min),
        "correlation_max": float(correlation_max),
        "cap_conservative": int(cap_conservative),
        "cap_balanced": int(cap_balanced),
        "cap_aggressive": int(cap_aggressive),
    }
    if frame.empty:
        blocked.append("frame_empty")
        return FeatureTestResult(
            feature_scorecard=pd.DataFrame(),
            feature_stability=pd.DataFrame(),
            window_grid_scorecard=pd.DataFrame(),
            feature_set_report="",
            selected_feature_sets=empty_sets,
            location_aware_variants={},
            window_contract={},
            selector_config=selector_config,
            blocked_reasons=blocked,
        )
    if not folds:
        blocked.append("no_folds")
        return FeatureTestResult(
            feature_scorecard=pd.DataFrame(),
            feature_stability=pd.DataFrame(),
            window_grid_scorecard=pd.DataFrame(),
            feature_set_report="",
            selected_feature_sets=empty_sets,
            location_aware_variants={},
            window_contract={},
            selector_config=selector_config,
            blocked_reasons=blocked,
        )

    features = _feature_columns(frame, max_features=max_features)
    if not features:
        blocked.append("no_eligible_features")
        return FeatureTestResult(
            feature_scorecard=pd.DataFrame(),
            feature_stability=pd.DataFrame(),
            window_grid_scorecard=pd.DataFrame(),
            feature_set_report="",
            selected_feature_sets=empty_sets,
            location_aware_variants={},
            window_contract={},
            selector_config=selector_config,
            blocked_reasons=blocked,
        )

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
        return FeatureTestResult(
            feature_scorecard=pd.DataFrame(),
            feature_stability=stability_df,
            window_grid_scorecard=pd.DataFrame(),
            feature_set_report="",
            selected_feature_sets=empty_sets,
            location_aware_variants={},
            window_contract={},
            selector_config=selector_config,
            blocked_reasons=sorted(set(blocked)),
        )

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

    scorecard_df = pd.DataFrame(score_rows)
    scorecard_df, selected_sets = _apply_selector_rules(
        scorecard_df,
        frame,
        stability_min=stability_min,
        sign_consistency_min=sign_consistency_min,
        permutation_delta_min=permutation_delta_min,
        ablation_delta_min=ablation_delta_min,
        correlation_max=correlation_max,
        cap_conservative=cap_conservative,
        cap_balanced=cap_balanced,
        cap_aggressive=cap_aggressive,
    )

    window_contract = _build_window_contract(features)
    window_grid_scorecard, window_grid_blocked = _build_window_grid_scorecard(
        frame=frame,
        folds=folds,
        market=market,
        features=features,
        scorecard_df=scorecard_df,
        stability_min=stability_min,
        sign_consistency_min=sign_consistency_min,
        permutation_delta_min=permutation_delta_min,
        ablation_delta_min=ablation_delta_min,
        correlation_max=correlation_max,
        cap_conservative=cap_conservative,
        cap_balanced=cap_balanced,
        cap_aggressive=cap_aggressive,
    )
    blocked.extend(window_grid_blocked)

    location_aware_variants = _build_location_aware_variants(frame, selected_sets)
    for profile in ("conservative", "balanced", "aggressive"):
        payload = location_aware_variants.get(profile, {})
        if payload.get("status") != "BLOCKED":
            continue
        missing_cols = payload.get("missing_columns", [])
        if missing_cols:
            blocked.append(
                f"location_aware_blocked:{profile}:missing_columns={','.join(missing_cols)}"
            )
        else:
            blocked.append(f"location_aware_blocked:{profile}:{payload.get('reason', 'unknown')}")

    feature_set_report = _render_feature_set_report(
        market=market,
        scorecard_df=scorecard_df,
        window_grid_df=window_grid_scorecard,
        selected_sets=selected_sets,
        selector_config=selector_config,
        window_contract=window_contract,
        location_aware_variants=location_aware_variants,
    )

    return FeatureTestResult(
        feature_scorecard=scorecard_df.reset_index(drop=True),
        feature_stability=stability_df.reset_index(drop=True),
        window_grid_scorecard=window_grid_scorecard.reset_index(drop=True),
        feature_set_report=feature_set_report,
        selected_feature_sets=selected_sets,
        location_aware_variants=location_aware_variants,
        window_contract=window_contract,
        selector_config=selector_config,
        blocked_reasons=sorted(set(blocked)),
    )
