from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ModelLabConfig
from .splits import build_rolling_folds


DEFAULT_MARKETS = ["spread", "total", "ml"]
DEFAULT_TIERS = ["conservative", "balanced", "aggressive"]


@dataclass
class FeatureSelectorResult:
    run_id: str
    feature_set_report_path: Path
    generated_paths: dict[str, str]
    selected_feature_sets: dict[str, dict[str, list[str]]]
    location_aware_variants: dict[str, dict[str, dict[str, Any]]]
    blocked_reasons: list[str]


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _candidate_table(
    scorecard: pd.DataFrame,
    stability: pd.DataFrame,
    *,
    market: str,
    config: ModelLabConfig,
) -> pd.DataFrame:
    if "market" in scorecard.columns:
        scored = scorecard.loc[scorecard["market"].astype(str) == market].copy()
    else:
        scored = scorecard.copy()
    if scored.empty:
        return scored

    for col in (
        "stability_score",
        "sign_consistency",
        "permutation_delta_mean",
        "drop_one_single_mean",
        "drop_one_group_mean",
        "univariate_mean",
    ):
        if col not in scored.columns:
            scored[col] = np.nan
        scored[col] = _to_num(scored[col])

    if "feature_name" not in scored.columns:
        scored["feature_name"] = ""

    if "feature_group" not in scored.columns:
        scored["feature_group"] = scored["feature_name"].astype(str).str.split("_", n=1).str[0]

    if "roi_impact_mean" in scored.columns:
        scored["roi_impact_mean"] = _to_num(scored["roi_impact_mean"])
    else:
        scored["roi_impact_mean"] = scored["permutation_delta_mean"]

    if (
        not stability.empty
        and "feature_name" in stability.columns
        and scored["permutation_delta_mean"].isna().all()
    ):
        if "market" in stability.columns:
            stab_market = stability.loc[stability["market"].astype(str) == market].copy()
        else:
            stab_market = stability.copy()
        perm_from_stability = (
            stab_market.groupby("feature_name", dropna=False)["permutation_delta"]
            .mean()
        )
        scored["permutation_delta_mean"] = scored["feature_name"].map(perm_from_stability)

    scored["importance_delta"] = pd.concat(
        [
            scored["permutation_delta_mean"],
            scored["drop_one_single_mean"],
            scored["drop_one_group_mean"],
        ],
        axis=1,
    ).max(axis=1, skipna=True)

    min_importance = float(max(config.selector_permutation_delta_min, config.selector_ablation_delta_min))
    pass_mask = (
        (scored["stability_score"] >= float(config.selector_stability_min))
        & (scored["sign_consistency"] >= float(config.selector_sign_consistency_min))
        & (scored["importance_delta"] >= min_importance)
    )
    scored["passes_hard_filters"] = pass_mask.fillna(False)
    filtered = scored.loc[scored["passes_hard_filters"]].copy()
    if filtered.empty:
        return filtered

    filtered = filtered.sort_values(
        ["stability_score", "roi_impact_mean", "importance_delta", "univariate_mean"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return filtered


def _connected_components(features: list[str], corr: pd.DataFrame, threshold: float) -> list[list[str]]:
    if not features:
        return []
    if corr.empty:
        return [[feature] for feature in features]

    valid = set(features)
    seen: set[str] = set()
    components: list[list[str]] = []

    for feature in features:
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
                if pd.notna(value) and float(value) > threshold:
                    stack.append(str(neighbor))
        components.append(component)

    return components


def _training_indices(frame: pd.DataFrame, config: ModelLabConfig) -> tuple[list[int], list[str]]:
    folds, blocked = build_rolling_folds(frame, config)
    if not folds:
        return [], blocked
    train_idx: set[int] = set()
    for fold in folds:
        train_idx.update([idx for idx in fold.train_index if idx in frame.index])
    return sorted(train_idx), blocked


def _training_correlation(frame: pd.DataFrame, features: list[str], train_idx: list[int]) -> pd.DataFrame:
    if not features or not train_idx:
        return pd.DataFrame()
    train_df = frame.loc[train_idx, features].copy()
    train_df = train_df.apply(pd.to_numeric, errors="coerce")
    return train_df.corr(method="spearman").abs()


def _prune_correlated(
    candidates: pd.DataFrame,
    corr: pd.DataFrame,
    corr_threshold: float,
) -> tuple[list[str], dict[str, str], dict[str, float], dict[str, str]]:
    if candidates.empty:
        return [], {}, {}, {}

    ranked_features = candidates["feature_name"].astype(str).tolist()
    components = _connected_components(ranked_features, corr, corr_threshold)

    selected: list[str] = []
    removed_with: dict[str, str] = {}
    removed_corr: dict[str, float] = {}
    cluster_by_feature: dict[str, str] = {}

    for idx, component in enumerate(components, start=1):
        cluster_id = f"C{idx:03d}"
        subset = candidates[candidates["feature_name"].isin(component)].copy()
        subset = subset.sort_values(
            ["stability_score", "roi_impact_mean", "importance_delta"],
            ascending=[False, False, False],
        )
        if subset.empty:
            continue
        anchor = str(subset.iloc[0]["feature_name"])
        selected.append(anchor)
        cluster_by_feature[anchor] = cluster_id
        for feature in subset["feature_name"].astype(str).tolist()[1:]:
            cluster_by_feature[feature] = cluster_id
            removed_with[feature] = anchor
            if not corr.empty and feature in corr.index and anchor in corr.columns:
                removed_corr[feature] = float(corr.loc[feature, anchor])
            else:
                removed_corr[feature] = float("nan")

    selected = sorted(
        selected,
        key=lambda feature: (
            float(candidates.loc[candidates["feature_name"] == feature, "stability_score"].iloc[0]),
            float(candidates.loc[candidates["feature_name"] == feature, "roi_impact_mean"].iloc[0]),
            float(candidates.loc[candidates["feature_name"] == feature, "importance_delta"].iloc[0]),
        ),
        reverse=True,
    )
    return selected, removed_with, removed_corr, cluster_by_feature


def _tiered_feature_sets(selected_features: list[str], config: ModelLabConfig) -> dict[str, list[str]]:
    return {
        "conservative": selected_features[: int(max(0, config.feature_cap_conservative))],
        "balanced": selected_features[: int(max(0, config.feature_cap_balanced))],
        "aggressive": selected_features[: int(max(0, config.feature_cap_aggressive))],
    }


def _location_aware_variant(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    if not features:
        return {
            "status": "BLOCKED",
            "reason": "no_selected_features",
            "location_features": [],
            "missing_columns": [],
            "neutral_policy": "",
        }

    cols = set(frame.columns)
    missing: list[str] = []
    location_features: list[str] = []

    for feature in features:
        base_name = str(feature)
        if base_name.startswith("home_"):
            base_name = base_name[len("home_") :]
        elif base_name.startswith("away_"):
            base_name = base_name[len("away_") :]

        home_col = f"home_{base_name}"
        away_col = f"away_{base_name}"
        has_home = home_col in cols
        has_away = away_col in cols
        if has_home and has_away:
            location_features.extend([home_col, away_col])
        else:
            if not has_home:
                missing.append(home_col)
            if not has_away:
                missing.append(away_col)

    missing = sorted(set(missing))
    location_features = sorted(set(location_features))
    if missing:
        return {
            "status": "BLOCKED",
            "reason": "missing_home_away_split_columns",
            "location_features": location_features,
            "missing_columns": missing,
            "neutral_policy": "",
        }

    neutral_policy = (
        "neutral_site=1 -> average(home_*,away_*); otherwise home_away/home baseline"
        if "neutral_site" in frame.columns
        else "neutral_site flag unavailable"
    )
    return {
        "status": "ACTIVE",
        "reason": "",
        "location_features": location_features,
        "missing_columns": [],
        "neutral_policy": neutral_policy,
    }


def build_location_aware_frame(frame: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    variant = _location_aware_variant(frame, features)
    if variant["status"] != "ACTIVE":
        return frame.copy(), [], list(variant["missing_columns"])

    out = frame.copy()
    used_cols: list[str] = []
    has_home_away = "home_away" in out.columns
    home_away = out.get("home_away", pd.Series("home", index=out.index)).astype(str).str.lower()
    neutral = pd.to_numeric(out.get("neutral_site", pd.Series(0, index=out.index)), errors="coerce").fillna(0)

    for feature in features:
        base_name = str(feature)
        if base_name.startswith("home_"):
            base_name = base_name[len("home_") :]
        elif base_name.startswith("away_"):
            base_name = base_name[len("away_") :]

        home_col = f"home_{base_name}"
        away_col = f"away_{base_name}"
        if home_col not in out.columns or away_col not in out.columns:
            continue

        home_vals = pd.to_numeric(out[home_col], errors="coerce")
        away_vals = pd.to_numeric(out[away_col], errors="coerce")
        if has_home_away:
            loc_vals = np.where(home_away == "away", away_vals, home_vals)
        else:
            loc_vals = home_vals
        loc_vals = pd.Series(loc_vals, index=out.index, dtype=float)
        loc_vals = loc_vals.where(neutral != 1, (home_vals + away_vals) / 2.0)
        loc_col = f"loc_{base_name}"
        out[loc_col] = loc_vals
        used_cols.append(loc_col)

    return out, sorted(set(used_cols)), []


def _payload(
    *,
    market: str,
    tier: str,
    features: list[str],
    location_variant: dict[str, Any],
    blocked_reasons: list[str],
    config: ModelLabConfig,
    run_id: str,
) -> dict[str, Any]:
    return {
        "selector_version": "AUTO_V2",
        "run_id": run_id,
        "market": market,
        "tier": tier,
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "selection_rules": {
            "min_stability_score": float(config.selector_stability_min),
            "min_sign_consistency": float(config.selector_sign_consistency_min),
            "min_importance_delta": float(max(config.selector_permutation_delta_min, config.selector_ablation_delta_min)),
            "correlation_threshold": float(config.selector_correlation_max),
            "cap_conservative": int(config.feature_cap_conservative),
            "cap_balanced": int(config.feature_cap_balanced),
            "cap_aggressive": int(config.feature_cap_aggressive),
        },
        "features": features,
        "feature_count": int(len(features)),
        "location_aware": location_variant,
        "blocked_reasons": blocked_reasons,
    }


def _report_header(run_id: str, config: ModelLabConfig) -> list[str]:
    lines = [
        "# Model Lab Feature Selector (AUTO_V2)",
        "",
        f"Run ID: `{run_id}`",
        f"Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        "",
        "## Selection Criteria",
        f"- min StabilityScore: `{float(config.selector_stability_min):.2f}`",
        f"- min SignConsistency: `{float(config.selector_sign_consistency_min):.2f}`",
        f"- min importance delta: `{float(max(config.selector_permutation_delta_min, config.selector_ablation_delta_min)):.4f}`",
        f"- correlation threshold: `>{float(config.selector_correlation_max):.2f}`",
        (
            "- tier caps: "
            f"conservative={int(config.feature_cap_conservative)}, "
            f"balanced={int(config.feature_cap_balanced)}, "
            f"aggressive={int(config.feature_cap_aggressive)}"
        ),
        "",
    ]
    return lines


def select_features_from_run(
    run_dir: Path,
    config: ModelLabConfig,
    *,
    markets: list[str] | None = None,
) -> FeatureSelectorResult:
    run_id = run_dir.name
    target_markets = markets if markets else list(DEFAULT_MARKETS)
    blocked: list[str] = []

    scorecard_path = run_dir / "feature_scorecard.csv"
    stability_path = run_dir / "feature_stability.csv"
    scorecard = _read_csv_optional(scorecard_path)
    stability = _read_csv_optional(stability_path)

    if scorecard.empty:
        blocked.append("select_features_missing:feature_scorecard.csv")
    if stability.empty:
        blocked.append("select_features_missing:feature_stability.csv")

    output_dir = (config.repo_root / "feature_sets" / "generated").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_feature_sets: dict[str, dict[str, list[str]]] = {}
    location_aware_variants: dict[str, dict[str, dict[str, Any]]] = {}
    generated_paths: dict[str, str] = {}
    report_lines = _report_header(run_id, config)

    for market in target_markets:
        report_lines.append(f"## Market `{market}`")
        selected_feature_sets[market] = {tier: [] for tier in DEFAULT_TIERS}
        location_aware_variants[market] = {}
        market_blocked: list[str] = []

        if scorecard.empty:
            market_blocked.append("scorecard_empty")
        candidates = _candidate_table(scorecard, stability, market=market, config=config) if not scorecard.empty else pd.DataFrame()
        if candidates.empty:
            market_blocked.append("no_candidates_after_hard_filters")

        frame_path = run_dir / f"{market}_frame.csv"
        frame = _read_csv_optional(frame_path)
        if frame.empty:
            market_blocked.append(f"missing_frame:{frame_path.name}")

        corr = pd.DataFrame()
        selected_anchor_features: list[str] = []
        removed_with: dict[str, str] = {}
        removed_corr: dict[str, float] = {}
        cluster_by_feature: dict[str, str] = {}
        if not frame.empty and not candidates.empty:
            train_idx, fold_blocked = _training_indices(frame, config)
            market_blocked.extend(fold_blocked)
            if not train_idx:
                market_blocked.append("no_training_rows_from_rolling_folds")
            else:
                candidate_features = [feature for feature in candidates["feature_name"].astype(str).tolist() if feature in frame.columns]
                missing_candidate_cols = sorted(set(candidates["feature_name"].astype(str).tolist()) - set(candidate_features))
                if missing_candidate_cols:
                    market_blocked.append(
                        f"candidate_columns_missing_in_frame:{','.join(missing_candidate_cols)}"
                    )
                corr = _training_correlation(frame, candidate_features, train_idx)
                selected_anchor_features, removed_with, removed_corr, cluster_by_feature = _prune_correlated(
                    candidates[candidates["feature_name"].isin(candidate_features)].copy(),
                    corr,
                    float(config.selector_correlation_max),
                )
                if not selected_anchor_features:
                    market_blocked.append("all_candidates_pruned_or_unavailable")

        tiers = _tiered_feature_sets(selected_anchor_features, config)
        selected_feature_sets[market] = tiers

        report_lines.append(f"- candidate rows after hard filters: `{len(candidates)}`")
        report_lines.append(f"- selected anchors after correlation pruning: `{len(selected_anchor_features)}`")
        if removed_with:
            report_lines.append("- pruned by correlation clusters:")
            for feature in sorted(removed_with):
                anchor = removed_with[feature]
                corr_val = removed_corr.get(feature, float("nan"))
                cluster_id = cluster_by_feature.get(feature, "")
                report_lines.append(
                    f"  - `{feature}` removed vs `{anchor}` (cluster `{cluster_id}`, corr `{corr_val:.3f}`)"
                )

        for tier in DEFAULT_TIERS:
            tier_features = list(tiers.get(tier, []))
            variant = _location_aware_variant(frame, tier_features) if not frame.empty else {
                "status": "BLOCKED",
                "reason": "missing_market_frame",
                "location_features": [],
                "missing_columns": [f"home_{feature}" for feature in tier_features] + [f"away_{feature}" for feature in tier_features],
                "neutral_policy": "",
            }
            location_aware_variants[market][tier] = variant
            if variant.get("status") == "BLOCKED" and variant.get("reason") == "missing_home_away_split_columns":
                market_blocked.append(
                    f"location_aware_blocked:{tier}:missing_columns={','.join(variant.get('missing_columns', []))}"
                )

            payload = _payload(
                market=market,
                tier=tier,
                features=tier_features,
                location_variant=variant,
                blocked_reasons=sorted(set(market_blocked)),
                config=config,
                run_id=run_id,
            )
            json_path = output_dir / f"{market}_AUTO_V2_{tier}.json"
            json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            generated_paths[f"feature_set_{market}_{tier}"] = str(json_path.resolve())

            report_lines.append(f"- `{tier}` features ({len(tier_features)}): {', '.join(tier_features) if tier_features else 'none'}")
            if variant.get("status") == "ACTIVE":
                report_lines.append(
                    f"  location-aware ACTIVE; split columns used: {', '.join(variant.get('location_features', []))}"
                )
                report_lines.append(f"  neutral behavior: {variant.get('neutral_policy', '')}")
            else:
                report_lines.append(
                    f"  location-aware BLOCKED ({variant.get('reason', 'unknown')}); "
                    f"missing: {', '.join(variant.get('missing_columns', []))}"
                )

        if market_blocked:
            blocked.extend([f"{market}:{reason}" for reason in sorted(set(market_blocked))])
        report_lines.append("")

    report_path = run_dir / "feature_set_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return FeatureSelectorResult(
        run_id=run_id,
        feature_set_report_path=report_path,
        generated_paths=generated_paths,
        selected_feature_sets=selected_feature_sets,
        location_aware_variants=location_aware_variants,
        blocked_reasons=sorted(set(blocked)),
    )
