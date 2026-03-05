from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


WINDOW_TOKENS = ("L4", "L7", "L10", "L12", "season")


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _model_score(row: pd.Series) -> float:
    market = str(row.get("market", ""))
    if market == "ml":
        for val in (row.get("roi"), row.get("hit_rate"),):
            if pd.notna(val):
                return float(val)
        brier = row.get("brier")
        if pd.notna(brier):
            return float(-brier)
        return float("nan")

    for val in (row.get("roi"), row.get("hit_rate")):
        if pd.notna(val):
            return float(val)
    mae = row.get("mae")
    if pd.notna(mae):
        return float(-mae)
    return float("nan")


def _window_combo(name: str) -> str:
    text = str(name)
    tokens: list[str] = []
    for token in WINDOW_TOKENS:
        pattern = rf"(?:^|_){re.escape(token)}(?:_|$)"
        if re.search(pattern, text):
            tokens.append(token)
    if not tokens:
        return "none"
    ordered = [tok for tok in WINDOW_TOKENS if tok in tokens]
    return "/".join(ordered)


def generate_exec_summary(run_dir: Path) -> tuple[Path, list[str]]:
    blocked: list[str] = []
    model_path = run_dir / "model_scorecard.csv"
    feature_path = run_dir / "feature_scorecard.csv"

    model_df = _read_csv_optional(model_path)
    feature_df = _read_csv_optional(feature_path)

    lines: list[str] = []
    lines.append("# Executive Summary")
    lines.append("")
    lines.append("## MODEL PERFORMANCE SUMMARY")

    top_model_name = ""
    if model_df.empty:
        blocked.append("exec_summary:model_scorecard_missing_or_empty")
        lines.append("- BLOCKED: `model_scorecard.csv` is missing or empty.")
    else:
        work = model_df.copy()
        for col in ("roi", "hit_rate", "mae", "brier", "n_folds", "n_games"):
            if col in work.columns:
                work[col] = _to_num(work[col])
            else:
                work[col] = np.nan

        work["score"] = work.apply(_model_score, axis=1)
        scored = work[work["score"].notna()].copy()
        if scored.empty:
            blocked.append("exec_summary:model_scores_unavailable")
            lines.append("- BLOCKED: model rows exist but no scorable metrics were available.")
        else:
            overall = scored.groupby("model_name", dropna=False)["score"].mean().sort_values(ascending=False)
            top_overall = overall.head(3)
            if not top_overall.empty:
                top_model_name = str(top_overall.index[0])
            lines.append("- Top performing models across rolling folds:")
            for model_name, value in top_overall.items():
                lines.append(f"  - {model_name}: composite score {value:.4f}")

            stability = (
                scored.groupby("model_name", dropna=False)[["n_folds", "n_games"]]
                .mean()
                .sort_values(["n_folds", "n_games"], ascending=[False, False])
            )
            lines.append("- Stability ranking (fold/game coverage proxy):")
            for model_name, row in stability.head(3).iterrows():
                lines.append(f"  - {model_name}: avg folds {row['n_folds']:.2f}, avg games {row['n_games']:.1f}")

            lines.append("- ROI / hit rate highlights:")
            for market in ("spread", "total", "ml"):
                market_df = scored[scored["market"] == market].copy()
                if market_df.empty:
                    lines.append(f"  - {market}: no scored rows.")
                    continue
                roi_rows = market_df[market_df["roi"].notna()]
                if not roi_rows.empty:
                    best = roi_rows.sort_values("roi", ascending=False).iloc[0]
                    lines.append(f"  - {market}: best ROI {best['roi']:.4f} by {best['model_name']}")
                    continue
                hit_rows = market_df[market_df["hit_rate"].notna()]
                if not hit_rows.empty:
                    best = hit_rows.sort_values("hit_rate", ascending=False).iloc[0]
                    lines.append(f"  - {market}: best hit rate {best['hit_rate']:.4f} by {best['model_name']}")
                    continue
                lines.append(f"  - {market}: ROI/hit rate unavailable.")

    lines.append("")
    lines.append("## WINDOW TEST RESULTS")
    best_window_combo = ""
    if feature_df.empty:
        blocked.append("exec_summary:feature_scorecard_missing_or_empty")
        lines.append("- BLOCKED: `feature_scorecard.csv` is missing or empty.")
    else:
        feature_work = feature_df.copy()
        for col in ("permutation_delta_mean", "stability_score"):
            if col in feature_work.columns:
                feature_work[col] = _to_num(feature_work[col])
            else:
                feature_work[col] = np.nan
        feature_work["window_combo"] = feature_work["feature_name"].astype(str).map(_window_combo)
        combo_stats = (
            feature_work[feature_work["window_combo"] != "none"]
            .groupby("window_combo", dropna=False)[["permutation_delta_mean", "stability_score"]]
            .mean()
            .sort_values(["permutation_delta_mean", "stability_score"], ascending=[False, False])
        )
        combo_stats = combo_stats[combo_stats["permutation_delta_mean"].notna()]
        if combo_stats.empty:
            blocked.append("exec_summary:window_combo_unavailable")
            lines.append("- BLOCKED: no finite window-signal metrics were available in `feature_scorecard.csv`.")
        else:
            best_window_combo = str(combo_stats.index[0])
            strong_alt = str(combo_stats.index[1]) if len(combo_stats) > 1 else best_window_combo
            weak = str(combo_stats.index[-1])
            lines.append(f"- Best: {best_window_combo}")
            lines.append(f"- Strong alternative: {strong_alt}")
            lines.append(f"- Weak: {weak}")

    lines.append("")
    lines.append("## FEATURE SIGNAL SUMMARY")
    if feature_df.empty:
        lines.append("- BLOCKED: feature signal aggregates unavailable.")
    else:
        feature_work = feature_df.copy()
        for col in ("permutation_delta_mean", "stability_score"):
            feature_work[col] = _to_num(feature_work.get(col, pd.Series(np.nan, index=feature_work.index)))

        if "feature_group" not in feature_work.columns:
            blocked.append("exec_summary:feature_group_missing")
            lines.append("- BLOCKED: `feature_group` column missing from `feature_scorecard.csv`.")
        else:
            group_stats = (
                feature_work.groupby("feature_group", dropna=False)[["permutation_delta_mean", "stability_score"]]
                .mean()
                .sort_values(["permutation_delta_mean", "stability_score"], ascending=[False, False])
            )
            group_stats = group_stats[group_stats["permutation_delta_mean"].notna()]
            if group_stats.empty:
                blocked.append("exec_summary:feature_group_signal_unavailable")
                lines.append("- BLOCKED: no finite feature-group signal values were available.")
            else:
                top_group = group_stats.head(1)
                mid_group = group_stats.iloc[1:2] if len(group_stats) > 1 else top_group
                low_group = group_stats.tail(1)
                for group_name, row in top_group.iterrows():
                    lines.append(
                        f"- {group_name} features showed the strongest signal "
                        f"(permutation {row['permutation_delta_mean']:.4f}, stability {row['stability_score']:.2f})."
                    )
                for group_name, row in mid_group.iterrows():
                    lines.append(
                        f"- {group_name} features had moderate signal "
                        f"(permutation {row['permutation_delta_mean']:.4f}, stability {row['stability_score']:.2f})."
                    )
                for group_name, row in low_group.iterrows():
                    lines.append(
                        f"- {group_name} features were least stable/signal-rich "
                        f"(permutation {row['permutation_delta_mean']:.4f}, stability {row['stability_score']:.2f})."
                    )

    lines.append("")
    lines.append("## FEATURES KEPT")
    balanced_selected: list[str] = []
    if feature_df.empty:
        lines.append("- BLOCKED: selected feature list unavailable.")
    else:
        feature_work = feature_df.copy()
        for col in ("permutation_delta_mean", "stability_score", "sign_consistency"):
            if col in feature_work.columns:
                feature_work[col] = _to_num(feature_work[col])
            else:
                feature_work[col] = np.nan

        selected_col = "selected_balanced" if "selected_balanced" in feature_work.columns else None
        if selected_col is None:
            blocked.append("exec_summary:selected_balanced_missing")
            selected_rows = feature_work.sort_values(
                ["permutation_delta_mean", "stability_score"], ascending=[False, False]
            ).head(18)
        else:
            selected_rows = feature_work[feature_work[selected_col].astype(bool)].copy()
        if selected_rows.empty:
            lines.append("- No features passed selection for the Balanced set.")
        else:
            balanced_selected = selected_rows["feature_name"].astype(str).tolist()
            for _, row in selected_rows.head(12).iterrows():
                lines.append(
                    f"- Feature: {row['feature_name']}  "
                    f"Reason: permutation {row['permutation_delta_mean']:.4f}, "
                    f"stability {row['stability_score']:.2f}, "
                    f"sign consistency {row['sign_consistency']:.2f}."
                )

    lines.append("")
    lines.append("## FEATURES REMOVED")
    if feature_df.empty:
        lines.append("- BLOCKED: removed feature list unavailable.")
    else:
        feature_work = feature_df.copy()
        selected_aggressive = (
            feature_work["selected_aggressive"].astype(bool)
            if "selected_aggressive" in feature_work.columns
            else pd.Series(False, index=feature_work.index)
        )
        removed = feature_work.loc[~selected_aggressive].copy()
        if removed.empty:
            lines.append("- None (all scored features remained in the Aggressive set).")
        else:
            if "exclusion_reason" not in removed.columns:
                removed["exclusion_reason"] = "not_selected"
            for _, row in removed.head(12).iterrows():
                lines.append(f"- {row['feature_name']}: {row['exclusion_reason']}")

    lines.append("")
    lines.append("## LOCATION EFFECT INSIGHT")
    if feature_df.empty:
        lines.append("- BLOCKED: location impact cannot be computed without feature scorecard.")
    else:
        feature_work = feature_df.copy()
        feature_work["permutation_delta_mean"] = _to_num(feature_work.get("permutation_delta_mean", pd.Series(np.nan, index=feature_work.index)))
        names = feature_work["feature_name"].astype(str)
        home_mask = names.str.startswith("home_")
        away_mask = names.str.startswith("away_")
        overall_mask = ~home_mask & ~away_mask
        if home_mask.sum() == 0 and away_mask.sum() == 0:
            blocked.append("exec_summary:location_split_features_missing")
            lines.append("- BLOCKED: no `home_*`/`away_*` features were present in `feature_scorecard.csv`.")
        else:
            home_mean = float(feature_work.loc[home_mask, "permutation_delta_mean"].mean(skipna=True))
            away_mean = float(feature_work.loc[away_mask, "permutation_delta_mean"].mean(skipna=True))
            overall_mean = float(feature_work.loc[overall_mask, "permutation_delta_mean"].mean(skipna=True))
            if pd.isna(home_mean) and pd.isna(away_mean) and pd.isna(overall_mean):
                blocked.append("exec_summary:location_signal_unavailable")
                lines.append("- BLOCKED: location split features exist but no finite permutation metrics were available.")
            else:
                lines.append(
                    f"- Home split mean permutation delta: {home_mean:.4f}; "
                    f"away split: {away_mean:.4f}; overall: {overall_mean:.4f}."
                )
                if pd.notna(home_mean) and pd.notna(overall_mean) and home_mean > overall_mean:
                    lines.append("- Home split features improved signal versus overall baselines.")
                elif pd.notna(home_mean) and pd.notna(overall_mean):
                    lines.append("- Home split features were not stronger than overall baselines in this run.")
                if pd.notna(away_mean) and pd.notna(home_mean) and away_mean < home_mean:
                    lines.append("- Away split features contributed less signal than home split features.")

    lines.append("")
    lines.append("## NEXT STEPS")
    lines.append("- Use the Balanced feature set (18 cap) as the default candidate pool for the next iteration.")
    if best_window_combo:
        lines.append(f"- Prioritize window pattern `{best_window_combo}` for the next fold tests.")
    else:
        lines.append("- Add/retain window-tagged features so window sensitivity can be compared directly.")
    if top_model_name:
        lines.append(f"- Use `{top_model_name}` as a baseline benchmark model in the next run.")
    lines.append("- Re-run `score-models`, `feature-signal`, and `ensemble` on the same run_id before retraining.")
    lines.append("- Investigate high-correlation removals and consider grouping correlated features upstream.")

    out_path = run_dir / "EXEC_SUMMARY.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path, sorted(set(blocked))
