#!/usr/bin/env python3
"""Build consolidated backtest training data for optimizer input."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

DATA_DIR = Path("data")
DEFAULT_OUTPUT = DATA_DIR / "backtest_training_data.csv"


def normalize_game_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.lstrip("0")
    return text or "0"


def to_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def maybe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, low_memory=False)


def calc_delta(home_val: Any, away_val: Any) -> Optional[float]:
    if pd.isna(home_val) or pd.isna(away_val):
        return None
    return float(home_val) - float(away_val)


def choose_market_row(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["capture_type"] = work.get("capture_type", "").astype(str).str.lower()
    timestamp_col = "captured_at_utc" if "captured_at_utc" in work.columns else "pulled_at_utc"
    if timestamp_col in work.columns:
        work["_capture_ts"] = pd.to_datetime(work[timestamp_col], errors="coerce", utc=True)
    else:
        work["_capture_ts"] = pd.NaT
    work["_is_closing"] = (work["capture_type"] == "closing").astype(int)
    work = work.sort_values(["game_id", "_is_closing", "_capture_ts"], ascending=[True, False, False])
    return work.drop_duplicates(subset=["game_id"], keep="first")


def resolve_col(frame: pd.DataFrame, preferred: str, fallback: Optional[str] = None) -> pd.Series:
    if preferred in frame.columns:
        return frame[preferred]
    if fallback and fallback in frame.columns:
        return frame[fallback]
    return pd.Series([pd.NA] * len(frame), index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated optimizer training dataset.")
    parser.add_argument("--season", type=int, help="Optional season filter, e.g. 2026")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    results_path = DATA_DIR / "results_log.csv"
    team_path = DATA_DIR / "team_game_weighted.csv"
    rotation_path = DATA_DIR / "rotation_features.csv"
    avail_path = DATA_DIR / "player_availability_features.csv"
    market_path = DATA_DIR / "market_lines.csv"
    pred_path = DATA_DIR / "predictions_history.csv"
    situational_path = DATA_DIR / "situational_features.csv"

    results = pd.read_csv(results_path, low_memory=False)
    results["game_id"] = results.get("game_id", results.get("event_id")).map(normalize_game_id)

    to_numeric(
        results,
        [
            "home_score_actual",
            "away_score_actual",
            "spread_line",
            "pred_spread",
            "pred_total",
            "actual_margin",
            "actual_total",
            "home_team_id",
            "away_team_id",
            "game_date",
        ],
    )

    results["actual_margin"] = results["actual_margin"].where(
        results["actual_margin"].notna(), results["home_score_actual"] - results["away_score_actual"]
    )
    results["actual_total"] = results["actual_total"].where(
        results["actual_total"].notna(), results["home_score_actual"] + results["away_score_actual"]
    )

    spread = results["spread_line"]
    margin = results["actual_margin"]
    results["home_covered_ats"] = pd.NA
    results.loc[(margin - spread) > 0, "home_covered_ats"] = 1
    results.loc[(margin - spread) < 0, "home_covered_ats"] = 0

    total_line = resolve_col(results, "market_total", "pred_total")
    results["covered_over"] = pd.NA
    diff_total = results["actual_total"] - pd.to_numeric(total_line, errors="coerce")
    results.loc[diff_total > 0, "covered_over"] = 1
    results.loc[diff_total < 0, "covered_over"] = 0

    results = results[results["home_covered_ats"].notna()].copy()

    if args.season is not None:
        date_text = results.get("game_date", pd.Series(dtype="float64")).astype("Int64").astype(str)
        results = results[date_text.str[:4] == str(args.season)].copy()

    results["season"] = results.get("game_date", pd.Series(dtype="float64")).astype("Int64").astype(str).str[:4]
    results["game_date"] = results.get("game_date", pd.Series(dtype="float64")).astype("Int64").astype(str)

    team = pd.read_csv(team_path, low_memory=False)
    team["game_id"] = team.get("game_id", team.get("event_id")).map(normalize_game_id)
    to_numeric(team, ["team_id"])

    team_cols = {
        "net_rtg_l5": "net_rtg_delta_l5",
        "net_rtg_l10": "net_rtg_delta_l10",
        "adj_ortg": "adj_ortg_delta",
        "adj_drtg": "adj_drtg_delta",
        "cage_em_diff": "cage_em_diff",
        "cage_t_diff": "cage_t_diff",
        "cage_o_diff": "cage_o_diff",
        "cage_d_diff": "cage_d_diff",
        "efg_pct_l10": "efg_delta_l10",
        "tov_pct_l5": "to_rate_delta_l5",
        "orb_pct_l10": "orb_delta_l10",
        "ftr_l5": "ftrate_delta_l5",
        "pace_l5": "pace_delta_l5",
        "rest_days": "rest_delta",
        "fatigue_index": "travel_fatigue_delta",
    }

    home_team = team[["game_id", "team_id", *[c for c in team_cols if c in team.columns]]].rename(
        columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in team_cols if c in team.columns}}
    )
    away_team = team[["game_id", "team_id", *[c for c in team_cols if c in team.columns]]].rename(
        columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in team_cols if c in team.columns}}
    )

    merged = results.merge(home_team, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_team, on=["game_id", "away_team_id"], how="left")

    for source_col, out_col in team_cols.items():
        home_col = f"home_{source_col}"
        away_col = f"away_{source_col}"
        if home_col in merged.columns and away_col in merged.columns:
            merged[out_col] = [calc_delta(h, a) for h, a in zip(merged[home_col], merged[away_col])]
        else:
            merged[out_col] = pd.NA

    if "cage_em_diff" in results.columns:
        merged["cage_em_diff"] = pd.to_numeric(results["cage_em_diff"], errors="coerce")

    merged["home_field"] = 1

    rotation = maybe_read_csv(rotation_path)
    rot_map = {
        "rot_efg_l5": "rot_efg_delta",
        "to_swing": "rot_to_swing_diff",
        "exec_tax": "exec_tax_diff",
        "three_pt_fragility": "three_pt_fragility_diff",
        "rot_minshare_sd": "rot_minshare_sd_diff",
        "top2_pused_share": "top2_pused_share_diff",
        "closer_ft_pct": "closer_ft_pct_delta",
    }
    if rotation is not None:
        rotation["game_id"] = rotation.get("game_id", rotation.get("event_id")).map(normalize_game_id)
        to_numeric(rotation, ["team_id", *[c for c in rot_map if c in rotation.columns]])
        home_rot = rotation[["game_id", "team_id", *[c for c in rot_map if c in rotation.columns]]].rename(
            columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in rot_map if c in rotation.columns}}
        )
        away_rot = rotation[["game_id", "team_id", *[c for c in rot_map if c in rotation.columns]]].rename(
            columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in rot_map if c in rotation.columns}}
        )
        merged = merged.merge(home_rot, on=["game_id", "home_team_id"], how="left")
        merged = merged.merge(away_rot, on=["game_id", "away_team_id"], how="left")
        for source_col, out_col in rot_map.items():
            hcol = f"home_{source_col}"
            acol = f"away_{source_col}"
            merged[out_col] = [calc_delta(h, a) for h, a in zip(merged.get(hcol, pd.Series([pd.NA] * len(merged))), merged.get(acol, pd.Series([pd.NA] * len(merged))))]
    else:
        for out_col in rot_map.values():
            merged[out_col] = pd.NA

    availability = maybe_read_csv(avail_path)
    avail_map = {
        "star_availability": "star_availability_delta",
        "minutes_available": "minutes_available_delta",
        "lineup_continuity": "lineup_continuity_delta",
        "usage_gini": "usage_gini_delta",
    }
    if availability is not None:
        availability["game_id"] = availability.get("game_id", availability.get("event_id")).map(normalize_game_id)
        to_numeric(availability, ["team_id", *[c for c in avail_map if c in availability.columns]])
        home_av = availability[["game_id", "team_id", *[c for c in avail_map if c in availability.columns], *[c for c in ["new_starter_flag"] if c in availability.columns]]].rename(
            columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in avail_map if c in availability.columns}, "new_starter_flag": "new_starter_flag_home"}
        )
        away_av = availability[["game_id", "team_id", *[c for c in avail_map if c in availability.columns], *[c for c in ["new_starter_flag"] if c in availability.columns]]].rename(
            columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in avail_map if c in availability.columns}, "new_starter_flag": "new_starter_flag_away"}
        )
        merged = merged.merge(home_av, on=["game_id", "home_team_id"], how="left")
        merged = merged.merge(away_av, on=["game_id", "away_team_id"], how="left")
        for source_col, out_col in avail_map.items():
            hcol = f"home_{source_col}"
            acol = f"away_{source_col}"
            merged[out_col] = [calc_delta(h, a) for h, a in zip(merged.get(hcol, pd.Series([pd.NA] * len(merged))), merged.get(acol, pd.Series([pd.NA] * len(merged))))]
        if "new_starter_flag_home" not in merged.columns:
            merged["new_starter_flag_home"] = pd.NA
        if "new_starter_flag_away" not in merged.columns:
            merged["new_starter_flag_away"] = pd.NA
    else:
        for out_col in avail_map.values():
            merged[out_col] = pd.NA
        merged["new_starter_flag_home"] = pd.NA
        merged["new_starter_flag_away"] = pd.NA

    market = pd.read_csv(market_path, low_memory=False)
    market["game_id"] = market.get("game_id", market.get("event_id")).map(normalize_game_id)
    market = choose_market_row(market)
    to_numeric(market, ["home_spread_open", "home_spread_current", "total_current", "total_open"])
    market_subset = market[["game_id", "home_spread_open", "home_spread_current", "total_current", "total_open"]].rename(
        columns={
            "home_spread_open": "opening_spread",
            "home_spread_current": "closing_spread",
            "total_current": "total_line",
            "total_open": "total_opening_line",
        }
    )
    merged = merged.merge(market_subset, on="game_id", how="left")
    merged["total_line"] = merged["total_line"].where(merged["total_line"].notna(), merged["total_opening_line"])

    predictions = maybe_read_csv(pred_path)
    if predictions is not None:
        predictions["game_id"] = predictions.get("game_id", predictions.get("event_id")).map(normalize_game_id)
        pred_subset = predictions[[c for c in ["game_id", "pred_spread", "pred_total"] if c in predictions.columns]].drop_duplicates("game_id", keep="last")
        merged = merged.merge(pred_subset, on="game_id", how="left", suffixes=("", "_predhist"))
        merged["pred_spread"] = resolve_col(merged, "pred_spread", "pred_spread_predhist")
        merged["pred_total"] = resolve_col(merged, "pred_total", "pred_total_predhist")

    situational = maybe_read_csv(situational_path)
    if situational is not None:
        situational["game_id"] = situational.get("game_id", situational.get("event_id")).map(normalize_game_id)
        to_numeric(situational, ["team_id", "situational_edge_score", "rest_delta"])
        keep_cols = [
            "game_id", "team_id", "lookahead_flag", "letdown_flag", "bounce_back_flag", "revenge_flag",
            "revenge_margin", "bubble_pressure_flag", "must_win_flag", "fatigue_flag", "extended_rest_flag",
            "is_rivalry_game", "is_neutral_site", "is_conference_game", "situational_edge_score", "rest_delta",
        ]
        for col in keep_cols:
            if col not in situational.columns:
                situational[col] = pd.NA
        situational = situational[keep_cols]

        home_sit = situational.rename(columns={
            "team_id": "home_team_id",
            "lookahead_flag": "home_lookahead_flag",
            "letdown_flag": "home_letdown_flag",
            "bounce_back_flag": "home_bounce_back_flag",
            "revenge_flag": "home_revenge_flag",
            "revenge_margin": "home_revenge_margin",
            "bubble_pressure_flag": "home_bubble_pressure_flag",
            "must_win_flag": "home_must_win_flag",
            "fatigue_flag": "home_fatigue_flag",
            "extended_rest_flag": "home_extended_rest_flag",
            "situational_edge_score": "home_situational_edge_score",
        })
        away_sit = situational.rename(columns={
            "team_id": "away_team_id",
            "lookahead_flag": "away_lookahead_flag",
            "letdown_flag": "away_letdown_flag",
            "bounce_back_flag": "away_bounce_back_flag",
            "revenge_flag": "away_revenge_flag",
            "bubble_pressure_flag": "away_bubble_pressure_flag",
            "fatigue_flag": "away_fatigue_flag",
            "extended_rest_flag": "away_extended_rest_flag",
            "situational_edge_score": "away_situational_edge_score",
        })
        merged = merged.merge(home_sit, on=["game_id", "home_team_id"], how="left")
        merged = merged.merge(away_sit, on=["game_id", "away_team_id"], how="left", suffixes=("", "_awaydup"))
        merged["situational_edge_delta"] = merged["home_situational_edge_score"] - merged["away_situational_edge_score"]
        merged["rest_delta"] = resolve_col(merged, "rest_delta", "rest_delta_awaydup")

    to_numeric(merged, ["pred_spread", "pred_total", "closing_spread", "opening_spread", "total_line"])
    merged["clv_delta"] = merged["closing_spread"] - merged["pred_spread"]

    merged["data_completeness_tier"] = "team_only"
    has_tier3 = merged["rot_efg_delta"].notna()
    has_tier4 = merged["star_availability_delta"].notna()
    has_market = merged["closing_spread"].notna()
    merged.loc[has_market & has_tier3 & has_tier4, "data_completeness_tier"] = "full"
    merged.loc[~has_market, "data_completeness_tier"] = "no_market"
    merged["created_at"] = datetime.now(timezone.utc).isoformat()

    out_cols = [
        "game_id", "game_date", "season", "home_team_id", "away_team_id",
        "net_rtg_delta_l5", "net_rtg_delta_l10", "adj_ortg_delta", "adj_drtg_delta",
        "cage_em_diff", "cage_t_diff", "cage_o_diff", "cage_d_diff", "efg_delta_l10",
        "to_rate_delta_l5", "orb_delta_l10", "ftrate_delta_l5", "pace_delta_l5",
        "home_field", "rest_delta", "travel_fatigue_delta",
        "rot_efg_delta", "rot_to_swing_diff", "exec_tax_diff", "three_pt_fragility_diff",
        "rot_minshare_sd_diff", "top2_pused_share_diff", "closer_ft_pct_delta",
        "star_availability_delta", "minutes_available_delta", "lineup_continuity_delta",
        "usage_gini_delta", "new_starter_flag_home", "new_starter_flag_away",
        "opening_spread", "closing_spread", "total_line", "clv_delta",
        "home_lookahead_flag", "home_letdown_flag", "home_bounce_back_flag", "home_revenge_flag", "home_revenge_margin",
        "away_lookahead_flag", "away_letdown_flag", "away_bounce_back_flag", "away_revenge_flag",
        "home_bubble_pressure_flag", "away_bubble_pressure_flag", "home_must_win_flag",
        "home_fatigue_flag", "away_fatigue_flag", "home_extended_rest_flag", "away_extended_rest_flag",
        "is_rivalry_game", "is_neutral_site", "is_conference_game", "situational_edge_delta",
        "actual_margin", "home_covered_ats", "actual_total", "covered_over", "pred_spread", "pred_total",
        "data_completeness_tier", "created_at",
    ]

    for col in out_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    out_df = merged[out_cols].copy()

    total_rows = len(out_df)
    rows_tier12 = out_df["net_rtg_delta_l10"].notna().sum()
    rows_tier3 = out_df["rot_efg_delta"].notna().sum()
    rows_tier4 = out_df["star_availability_delta"].notna().sum()
    rows_market = out_df["closing_spread"].notna().sum()
    rows_pred = out_df["pred_spread"].notna().sum()

    tier1_core = [
        "net_rtg_delta_l5", "net_rtg_delta_l10", "adj_ortg_delta", "adj_drtg_delta", "efg_delta_l10",
        "to_rate_delta_l5", "orb_delta_l10", "ftrate_delta_l5", "pace_delta_l5", "rest_delta", "travel_fatigue_delta",
    ]
    usable_tier1 = out_df[out_df["home_covered_ats"].notna() & out_df[tier1_core].notna().all(axis=1)].shape[0]

    full_cols = tier1_core + [
        "rot_efg_delta", "rot_to_swing_diff", "exec_tax_diff", "three_pt_fragility_diff",
        "rot_minshare_sd_diff", "top2_pused_share_diff", "star_availability_delta", "minutes_available_delta",
        "lineup_continuity_delta", "usage_gini_delta", "closing_spread", "total_line",
    ]
    usable_full = out_df[out_df["home_covered_ats"].notna() & out_df[full_cols].notna().all(axis=1)].shape[0]

    print(f"Total rows (graded games): {total_rows}")
    print(f"Rows with full Tier 1+2 (net_rtg_delta non-null): {rows_tier12}")
    print(f"Rows with Tier 3 (rot_efg_delta non-null): {rows_tier3}")
    print(f"Rows with Tier 4 (star_availability_delta non-null): {rows_tier4}")
    print(f"Rows with market data (closing_spread non-null): {rows_market}")
    print(f"Rows with pred_spread non-null: {rows_pred}")
    print(f"Rows usable for Tier-1-only optimization (Tier 1+2 + outcome): {usable_tier1}")
    print(f"Rows usable for full optimization (all tiers + outcome): {usable_full}")

    if usable_tier1 < 50:
        print("[WARNING] Insufficient graded games for optimization.\nNeed 50+ rows with Tier 1 features and ATS outcomes.\nFix grading chain before running optimizer.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Wrote training data to {output_path}")


if __name__ == "__main__":
    main()
