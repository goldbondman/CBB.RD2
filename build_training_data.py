#!/usr/bin/env python3
"""Build consolidated backtest training data from full game history.

Base table: team_game_weighted.csv (~4,600 post-game rows after pivoting home/away).
Replaces results_log.csv (9 rows) as the base.

Pipeline:
  1. team_game_weighted.csv  → pivot home/away → ~4,600 wide rows
  2. player_game_logs.csv    → aggregate per (event_id, team_id) → join
  3. market_lines.csv        → closing/opening spread+total  → join
  4. cbb_rankings.csv        → cage_em/t/o/d diff            → join
  5. rotation/availability/situational/luck feature CSVs      → join
  6. team_ats_profile.csv    → ATS profile per team           → join
  7. team_pretournament_snapshot.csv → player/injury cols     → join
  8. predictions_history.csv → pred_spread/pred_total        → optional join
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
DEFAULT_OUTPUT = DATA_DIR / "backtest_training_data.csv"

# ── Game-level columns (identical in both home/away rows for the same event) ─
GAME_LEVEL_COLS = [
    "event_id", "game_datetime_utc", "game_datetime_pst",
    "venue", "neutral_site", "completed", "state", "is_ot", "num_ot",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "home_conference", "away_conference", "home_rank", "away_rank",
    "spread", "over_under", "home_ml", "away_ml",
    "home_h1", "away_h1", "home_h2", "away_h2",
    "home_ot1", "away_ot1", "home_ot2", "away_ot2", "home_ot3", "away_ot3",
]

# Team-specific box score + record columns (get home_/away_ prefix after pivot)
TEAM_BOX_COLS = [
    # Records at time of game
    "wins", "losses", "home_wins", "home_losses", "away_wins", "away_losses",
    "conf_wins", "conf_losses", "win_pct",
    # Box score for this game
    "points_for", "points_against", "margin",
    "h1_pts", "h2_pts",
    "fgm", "fga", "tpm", "tpa", "ftm", "fta",
    "orb", "drb", "reb", "ast", "stl", "blk", "tov", "pf",
    # ATS result for this game
    "cover", "cover_margin",
]

# Team-specific rolling metric columns (get home_/away_ prefix after pivot)
TEAM_METRIC_COLS = [
    # Efficiency rolling
    "net_rtg_l5", "net_rtg_l10",
    "adj_ortg", "adj_drtg", "adj_net_rtg",
    "efg_pct_l5", "efg_pct_l10",
    "tov_pct_l5", "orb_pct_l10", "ftr_l5",
    "pace_l5", "ortg_l5", "drtg_l5", "ortg_l10", "drtg_l10",
    "margin_l5", "margin_l10",
    # ATS rolling
    "cover_l10", "cover_rate_l10", "cover_rate_season", "ats_margin_l10",
    # Context
    "rest_days", "fatigue_index",
    "momentum_score", "form_rating", "luck_score",
    # Opponent-quality context
    "opp_avg_net_rtg_l5", "opp_avg_net_rtg_l10",
    # Home/away splits
    "ha_net_rtg_l10", "ha_efg_pct_l10",
    # Streaks
    "win_streak", "cover_streak",
    # Weighted quality rolling (key offensive/defensive weighted metrics)
    "net_rtg_wtd_qual_l5", "net_rtg_wtd_qual_l10",
    "ortg_wtd_off_l5", "ortg_wtd_off_l10",
    "drtg_wtd_def_l5", "drtg_wtd_def_l10",
    "efg_pct_wtd_off_l5", "efg_pct_wtd_off_l10",
    # Performance vs expected
    "perf_vs_exp_ortg", "perf_vs_exp_net",
    "perf_vs_exp_ortg_l5", "perf_vs_exp_net_l5",
]

# Delta map: team_game_weighted col → output delta column name
DELTA_MAP = {
    "net_rtg_l5":    "net_rtg_delta_l5",
    "net_rtg_l10":   "net_rtg_delta_l10",
    "adj_ortg":      "adj_ortg_delta",
    "adj_drtg":      "adj_drtg_delta",
    "adj_net_rtg":   "adj_net_rtg_delta",
    "efg_pct_l10":   "efg_delta_l10",
    "tov_pct_l5":    "to_rate_delta_l5",
    "orb_pct_l10":   "orb_delta_l10",
    "ftr_l5":        "ftrate_delta_l5",
    "pace_l5":       "pace_delta_l5",
    "rest_days":     "rest_delta",
    "fatigue_index": "travel_fatigue_delta",
}

# Metadata columns to exclude from team-specific pivoting
_META_SKIP = frozenset({
    "home_away", "source", "parse_version", "pulled_at_utc",
    "pipeline_run_id", "_improved", "game_id", "conference_name",
    "conf_id",  # mostly redundant with conference
})


# ── Utility functions ─────────────────────────────────────────────────────────

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


def choose_market_row(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["capture_type"] = work.get("capture_type", pd.Series("", index=work.index)).astype(str).str.lower()
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


def calc_delta(home_val: Any, away_val: Any) -> Optional[float]:
    if pd.isna(home_val) or pd.isna(away_val):
        return None
    return float(home_val) - float(away_val)


# ── Core builders ─────────────────────────────────────────────────────────────

def aggregate_player_stats(player_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player_game_logs per (event_id, team_id) → per-game team player stats.

    Returns DataFrame with columns:
        event_id, team_id,
        pg_top_scorer_pts, pg_top_scorer_efg,
        pg_bench_pts, pg_bench_pts_share,
        pg_starters_count, pg_total_min, pg_bench_min_share,
        pg_starters_fgm, pg_starters_fga
    """
    if player_df is None or player_df.empty:
        return pd.DataFrame(columns=["event_id", "team_id"])

    df = player_df.copy()
    for c in ["pts", "fgm", "fga", "tpm", "min"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "starter" in df.columns:
        df["_is_starter"] = df["starter"].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        df["_is_starter"] = False

    # Exclude DNP players
    if "did_not_play" in df.columns:
        df = df[~df["did_not_play"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()

    rows = []
    for (event_id, team_id), grp in df.groupby(["event_id", "team_id"]):
        total_pts = float(grp["pts"].sum())
        starter_mask = grp["_is_starter"]
        bench_mask = ~starter_mask

        bench_pts = float(grp.loc[bench_mask, "pts"].sum())
        bench_pts_share = bench_pts / total_pts if total_pts > 0 else np.nan

        starters_count = int(starter_mask.sum())
        total_min = float(grp["min"].sum()) if "min" in grp.columns else np.nan
        bench_min = float(grp.loc[bench_mask, "min"].sum()) if "min" in grp.columns else 0.0
        bench_min_share = bench_min / total_min if total_min and total_min > 0 else np.nan

        starters_fgm = float(grp.loc[starter_mask, "fgm"].sum()) if "fgm" in grp.columns else np.nan
        starters_fga = float(grp.loc[starter_mask, "fga"].sum()) if "fga" in grp.columns else np.nan

        if len(grp) > 0 and "pts" in grp.columns:
            top_idx = grp["pts"].idxmax()
            top_pts = float(grp.loc[top_idx, "pts"])
            top_fgm = float(grp.loc[top_idx, "fgm"]) if "fgm" in grp.columns else np.nan
            top_fga = float(grp.loc[top_idx, "fga"]) if "fga" in grp.columns else np.nan
            top_tpm = float(grp.loc[top_idx, "tpm"]) if "tpm" in grp.columns else 0.0
            top_efg = (top_fgm + 0.5 * top_tpm) / top_fga if top_fga and top_fga > 0 else np.nan
        else:
            top_pts = np.nan
            top_efg = np.nan

        rows.append({
            "event_id":           event_id,
            "team_id":            team_id,
            "pg_top_scorer_pts":  round(top_pts, 1) if not (isinstance(top_pts, float) and np.isnan(top_pts)) else np.nan,
            "pg_top_scorer_efg":  round(top_efg, 3) if not (isinstance(top_efg, float) and np.isnan(top_efg)) else np.nan,
            "pg_bench_pts":       round(bench_pts, 1),
            "pg_bench_pts_share": round(bench_pts_share, 3) if not (isinstance(bench_pts_share, float) and np.isnan(bench_pts_share)) else np.nan,
            "pg_starters_count":  starters_count,
            "pg_total_min":       round(total_min, 1) if total_min and not (isinstance(total_min, float) and np.isnan(total_min)) else np.nan,
            "pg_bench_min_share": round(bench_min_share, 3) if not (isinstance(bench_min_share, float) and np.isnan(bench_min_share)) else np.nan,
            "pg_starters_fgm":    starters_fgm,
            "pg_starters_fga":    starters_fga,
        })

    return pd.DataFrame(rows)


def build_base_table(weighted_path: Path) -> pd.DataFrame:
    """Load team_game_weighted.csv, filter to completed games, pivot home/away to wide.

    Returns one row per game (~4,600 rows) with home_/away_ prefixed team columns.
    Game-level columns (spread, over_under, team names, etc.) come from home rows.
    """
    df = pd.read_csv(weighted_path, low_memory=False)
    df = df[df["state"] == "post"].copy()
    df["game_id"] = df["event_id"].map(normalize_game_id)

    # Dedup: keep latest pull per (event_id, home_away)
    if "pulled_at_utc" in df.columns:
        df = df.sort_values("pulled_at_utc").drop_duplicates(["event_id", "home_away"], keep="last")
    else:
        df = df.drop_duplicates(["event_id", "home_away"], keep="last")

    avail_game = [c for c in GAME_LEVEL_COLS if c in df.columns]
    all_team_cols = TEAM_BOX_COLS + TEAM_METRIC_COLS
    avail_team = [
        c for c in all_team_cols
        if c in df.columns and c not in _META_SKIP and c not in GAME_LEVEL_COLS
    ]
    # Remove exact duplicates (a col could appear in both lists)
    seen = set()
    avail_team_deduped = []
    for c in avail_team:
        if c not in seen:
            seen.add(c)
            avail_team_deduped.append(c)
    avail_team = avail_team_deduped

    home = df[df["home_away"] == "home"].copy()
    away = df[df["home_away"] == "away"].copy()

    # Home side: all game cols + prefixed team-specific cols
    home_wide = home[[*avail_game, "game_id", *avail_team]].rename(
        columns={c: f"home_{c}" for c in avail_team}
    )

    # Away side: just event_id + prefixed team-specific cols
    away_wide = away[["event_id", *avail_team]].rename(
        columns={c: f"away_{c}" for c in avail_team}
    )

    wide = home_wide.merge(away_wide, on="event_id", how="inner")
    return wide


# ── Feature join helpers ──────────────────────────────────────────────────────

def _join_team_feature(
    merged: pd.DataFrame,
    feat_path: Path,
    feat_cols: list[str],
    col_renames: Optional[dict] = None,
    require: bool = True,
) -> pd.DataFrame:
    """Generic join of a per-(game_id, team_id) feature file for home+away.

    feat_cols: columns from the feature file to include (besides game_id/team_id).
    col_renames: {old_name: new_name} applied BEFORE pivoting to home_/away_.
    """
    if not feat_path.exists() or feat_path.stat().st_size == 0:
        if require:
            raise FileNotFoundError(f"Required input missing: {feat_path}")
        return merged

    feat = pd.read_csv(feat_path, low_memory=False)
    gid_col = "game_id" if "game_id" in feat.columns else "event_id"
    feat["game_id"] = feat[gid_col].map(normalize_game_id)
    to_numeric(feat, ["team_id"])

    if col_renames:
        feat = feat.rename(columns=col_renames)

    avail = [c for c in feat_cols if c in feat.columns]
    if not avail:
        return merged

    to_numeric(feat, avail)
    to_numeric(merged, ["home_team_id", "away_team_id"])

    home_feat = feat[["game_id", "team_id", *avail]].rename(
        columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in avail}}
    )
    away_feat = feat[["game_id", "team_id", *avail]].rename(
        columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in avail}}
    )
    merged = merged.merge(home_feat, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_feat, on=["game_id", "away_team_id"], how="left")
    return merged


def join_rankings(merged: pd.DataFrame, rankings_path: Path) -> pd.DataFrame:
    """Join current cage rankings by team_id to compute game-level differentials."""
    cage_cols = ["cage_em_diff", "cage_t_diff", "cage_o_diff", "cage_d_diff"]
    if not rankings_path.exists() or rankings_path.stat().st_size == 0:
        for c in cage_cols:
            merged[c] = pd.NA
        return merged

    rank = pd.read_csv(rankings_path, low_memory=False)
    to_numeric(rank, ["team_id", "cage_em", "cage_o", "cage_d", "cage_t"])
    to_numeric(merged, ["home_team_id", "away_team_id"])

    rank_sub = rank[["team_id"] + [c for c in ["cage_em", "cage_o", "cage_d", "cage_t"] if c in rank.columns]]

    merged = merged.merge(
        rank_sub.rename(columns={c: f"home_{c}" for c in rank_sub.columns if c != "team_id"}),
        left_on="home_team_id", right_on="team_id", how="left",
    ).drop(columns=["team_id"], errors="ignore")

    merged = merged.merge(
        rank_sub.rename(columns={c: f"away_{c}" for c in rank_sub.columns if c != "team_id"}),
        left_on="away_team_id", right_on="team_id", how="left",
    ).drop(columns=["team_id"], errors="ignore")

    for metric in ["em", "t", "o", "d"]:
        hc, ac, dc = f"home_cage_{metric}", f"away_cage_{metric}", f"cage_{metric}_diff"
        if hc in merged.columns and ac in merged.columns:
            merged[dc] = pd.to_numeric(merged[hc], errors="coerce") - pd.to_numeric(merged[ac], errors="coerce")
        else:
            merged[dc] = pd.NA

    return merged


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build consolidated optimizer training dataset.")
    parser.add_argument("--season", type=int, help="Optional season filter, e.g. 2026")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    weighted_path    = DATA_DIR / "team_game_weighted.csv"
    player_path      = DATA_DIR / "player_game_logs.csv"
    market_path      = DATA_DIR / "market_lines.csv"
    rankings_path    = DATA_DIR / "cbb_rankings.csv"
    rotation_path    = DATA_DIR / "rotation_features.csv"
    avail_path       = DATA_DIR / "player_availability_features.csv"
    situational_path = DATA_DIR / "situational_features.csv"
    luck_path        = DATA_DIR / "luck_regression_features.csv"
    ats_path         = DATA_DIR / "team_ats_profile.csv"
    snap_path        = DATA_DIR / "team_pretournament_snapshot.csv"
    pred_path        = DATA_DIR / "predictions_history.csv"
    results_path     = DATA_DIR / "results_log.csv"

    # ── 1. Build base table from team_game_weighted (~4,600 post-game rows) ──
    print(f"Loading base table from {weighted_path}...")
    merged = build_base_table(weighted_path)
    print(f"  Base table: {len(merged)} rows")

    # ── 2. Compute actual outcomes ───────────────────────────────────────────
    to_numeric(merged, ["home_points_for", "away_points_for", "spread", "over_under",
                        "home_team_id", "away_team_id"])

    merged["actual_margin"] = merged["home_points_for"] - merged["away_points_for"]
    merged["actual_total"]  = merged["home_points_for"] + merged["away_points_for"]
    # Backward-compat aliases
    merged["home_score_actual"] = merged["home_points_for"]
    merged["away_score_actual"] = merged["away_points_for"]

    # Derive game_date and season from datetime
    if "game_datetime_utc" in merged.columns:
        merged["game_date"] = (
            pd.to_datetime(merged["game_datetime_utc"], errors="coerce", utc=True)
            .dt.strftime("%Y%m%d")
        )
        merged["season"] = merged["game_date"].str[:4]

    # ── 3. Season filter ─────────────────────────────────────────────────────
    if args.season is not None:
        merged = merged[merged["season"] == str(args.season)].copy()
        print(f"  After season={args.season} filter: {len(merged)} rows")

    # ── 4. Aggregate player stats per game ───────────────────────────────────
    if player_path.exists() and player_path.stat().st_size > 0:
        print(f"Aggregating player stats from {player_path}...")
        player_df = pd.read_csv(player_path, low_memory=False)
        player_agg = aggregate_player_stats(player_df)
        if not player_agg.empty:
            player_agg["game_id"] = player_agg["event_id"].map(normalize_game_id)
            to_numeric(player_agg, ["team_id"])
            pg_cols = [c for c in player_agg.columns if c not in ["event_id", "team_id", "game_id"]]
            home_pg = player_agg[["game_id", "team_id", *pg_cols]].rename(
                columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in pg_cols}}
            )
            away_pg = player_agg[["game_id", "team_id", *pg_cols]].rename(
                columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in pg_cols}}
            )
            merged = merged.merge(home_pg, on=["game_id", "home_team_id"], how="left")
            merged = merged.merge(away_pg, on=["game_id", "away_team_id"], how="left")
            print(f"  Player agg rows: {len(player_agg)}")

    # ── 5. Join market lines (closing > opening > ESPN spread) ───────────────
    if not market_path.exists() or market_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required input missing: {market_path}")
    print(f"Joining market lines from {market_path}...")
    market = pd.read_csv(market_path, low_memory=False)
    mkt_id_col = "game_id" if "game_id" in market.columns else "event_id"
    market["game_id"] = market[mkt_id_col].map(normalize_game_id)
    market = choose_market_row(market)
    to_numeric(market, ["home_spread_open", "home_spread_current", "total_current", "total_open"])
    mkt_sub = market[
        ["game_id"]
        + [c for c in ["home_spread_open", "home_spread_current", "total_current", "total_open"] if c in market.columns]
    ].rename(columns={
        "home_spread_open":    "opening_spread",
        "home_spread_current": "closing_spread",
        "total_current":       "total_line",
        "total_open":          "total_opening_line",
    })
    merged = merged.merge(mkt_sub, on="game_id", how="left")
    # Fill total from opening if closing null
    if "total_line" in merged.columns and "total_opening_line" in merged.columns:
        merged["total_line"] = merged["total_line"].where(merged["total_line"].notna(), merged["total_opening_line"])

    # Best available spread: closing market → opening market → ESPN spread
    espn_spread = pd.to_numeric(merged.get("spread", pd.Series([pd.NA] * len(merged))), errors="coerce")
    espn_total  = pd.to_numeric(merged.get("over_under", pd.Series([pd.NA] * len(merged))), errors="coerce")
    merged["espn_spread"] = espn_spread
    merged["espn_total"]  = espn_total

    to_numeric(merged, ["closing_spread", "opening_spread", "total_line"])
    spread_line = resolve_col(merged, "closing_spread", "opening_spread")
    spread_line = spread_line.where(spread_line.notna(), espn_spread)
    merged["spread_line"] = spread_line

    total_line = resolve_col(merged, "total_line")
    total_line = total_line.where(total_line.notna(), espn_total)
    merged["total_line"] = total_line

    # Compute ATS / OU from best available line
    to_numeric(merged, ["actual_margin", "actual_total", "spread_line", "total_line"])
    merged["home_covered_ats"] = pd.NA
    ats_diff = merged["actual_margin"] - merged["spread_line"]
    merged.loc[ats_diff.notna() & (ats_diff > 0), "home_covered_ats"] = 1
    merged.loc[ats_diff.notna() & (ats_diff < 0), "home_covered_ats"] = 0

    merged["covered_over"] = pd.NA
    ou_diff = merged["actual_total"] - merged["total_line"]
    merged.loc[ou_diff.notna() & (ou_diff > 0), "covered_over"] = 1
    merged.loc[ou_diff.notna() & (ou_diff < 0), "covered_over"] = 0

    # ── 6. Join cage rankings differentials ─────────────────────────────────
    print(f"Joining rankings from {rankings_path}...")
    merged = join_rankings(merged, rankings_path)

    # ── 7. Compute deltas (home − away) for rolling metrics ──────────────────
    for src_col, delta_col in DELTA_MAP.items():
        hc, ac = f"home_{src_col}", f"away_{src_col}"
        if hc in merged.columns and ac in merged.columns:
            merged[delta_col] = [
                calc_delta(h, a)
                for h, a in zip(
                    pd.to_numeric(merged[hc], errors="coerce"),
                    pd.to_numeric(merged[ac], errors="coerce"),
                )
            ]
        else:
            merged[delta_col] = pd.NA

    merged["home_field"] = 1

    # ── 8. Rotation features ─────────────────────────────────────────────────
    if not rotation_path.exists() or rotation_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required input missing: {rotation_path}")
    print(f"Joining rotation features from {rotation_path}...")
    rotation = pd.read_csv(rotation_path, low_memory=False)
    gid_col = "game_id" if "game_id" in rotation.columns else "event_id"
    rotation["game_id"] = rotation[gid_col].map(normalize_game_id)
    to_numeric(rotation, ["team_id"])

    rot_map = {
        "rot_efg_l5":         "rot_efg_delta",
        "to_swing":           "rot_to_swing_diff",
        "exec_tax":           "exec_tax_diff",
        "three_pt_fragility": "three_pt_fragility_diff",
        "rot_minshare_sd":    "rot_minshare_sd_diff",
        "top2_pused_share":   "top2_pused_share_diff",
        "closer_ft_pct":      "closer_ft_pct_delta",
    }
    rot_extra_cols = [
        "rot_size", "rot_efg_l10", "rot_to_rate_l5", "rot_to_rate_l10",
        "rot_ftrate_l5", "rot_3par_l10", "rot_stocks_per40_l10",
        "rot_pf_per40_l5", "rot_minshare_sd", "top2_pused_share", "closer_ft_pct",
    ]
    rot_all_cols = list(set(list(rot_map.keys()) + rot_extra_cols))
    rot_avail = [c for c in rot_all_cols if c in rotation.columns]

    to_numeric(rotation, rot_avail)
    home_rot = rotation[["game_id", "team_id", *rot_avail]].rename(
        columns={"team_id": "home_team_id", **{c: f"home_{c}" for c in rot_avail}}
    )
    away_rot = rotation[["game_id", "team_id", *rot_avail]].rename(
        columns={"team_id": "away_team_id", **{c: f"away_{c}" for c in rot_avail}}
    )
    merged = merged.merge(home_rot, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_rot, on=["game_id", "away_team_id"], how="left")

    for src_col, delta_col in rot_map.items():
        hc, ac = f"home_{src_col}", f"away_{src_col}"
        h_ser = merged.get(hc, pd.Series([pd.NA] * len(merged)))
        a_ser = merged.get(ac, pd.Series([pd.NA] * len(merged)))
        merged[delta_col] = [calc_delta(h, a) for h, a in zip(h_ser, a_ser)]

    # ── 9. Availability features ──────────────────────────────────────────────
    if not avail_path.exists() or avail_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required input missing: {avail_path}")
    print(f"Joining availability features from {avail_path}...")
    availability = pd.read_csv(avail_path, low_memory=False)
    gid_col = "game_id" if "game_id" in availability.columns else "event_id"
    availability["game_id"] = availability[gid_col].map(normalize_game_id)
    to_numeric(availability, ["team_id"])

    # Rename to match expected avail_map keys
    avail_renames = {
        "star_availability_score": "star_availability",
        "minutes_available_pct":   "minutes_available",
        "lineup_continuity_l3":    "lineup_continuity",
    }
    availability = availability.rename(columns={k: v for k, v in avail_renames.items() if k in availability.columns})

    avail_map = {
        "star_availability":   "star_availability_delta",
        "minutes_available":   "minutes_available_delta",
        "lineup_continuity":   "lineup_continuity_delta",
        "usage_gini":          "usage_gini_delta",
    }
    avail_extra = ["new_starter_flag", "injury_impact_delta", "top1_usage_share"]
    avail_all = list(set(list(avail_map.keys()) + avail_extra))
    avail_avail = [c for c in avail_all if c in availability.columns]

    to_numeric(availability, [c for c in avail_avail if c != "new_starter_flag"])
    home_av = availability[["game_id", "team_id", *avail_avail]].rename(
        columns={
            "team_id": "home_team_id",
            "new_starter_flag": "new_starter_flag_home",
            **{c: f"home_{c}" for c in avail_avail if c != "new_starter_flag"},
        }
    )
    away_av = availability[["game_id", "team_id", *avail_avail]].rename(
        columns={
            "team_id": "away_team_id",
            "new_starter_flag": "new_starter_flag_away",
            **{c: f"away_{c}" for c in avail_avail if c != "new_starter_flag"},
        }
    )
    merged = merged.merge(home_av, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_av, on=["game_id", "away_team_id"], how="left")

    for src_col, delta_col in avail_map.items():
        hc, ac = f"home_{src_col}", f"away_{src_col}"
        h_ser = merged.get(hc, pd.Series([pd.NA] * len(merged)))
        a_ser = merged.get(ac, pd.Series([pd.NA] * len(merged)))
        merged[delta_col] = [calc_delta(h, a) for h, a in zip(h_ser, a_ser)]

    for flag_col in ["new_starter_flag_home", "new_starter_flag_away"]:
        if flag_col not in merged.columns:
            merged[flag_col] = pd.NA

    # ── 10. Situational features ──────────────────────────────────────────────
    if not situational_path.exists() or situational_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required input missing: {situational_path}")
    print(f"Joining situational features from {situational_path}...")
    situational = pd.read_csv(situational_path, low_memory=False)
    gid_col = "game_id" if "game_id" in situational.columns else "event_id"
    situational["game_id"] = situational[gid_col].map(normalize_game_id)
    to_numeric(situational, ["team_id", "situational_edge_score", "rest_delta"])

    sit_keep = [
        "game_id", "team_id",
        "lookahead_flag", "letdown_flag", "bounce_back_flag", "revenge_flag",
        "revenge_margin", "bubble_pressure_flag", "must_win_flag", "fatigue_flag",
        "extended_rest_flag", "is_rivalry_game", "is_neutral_site", "is_conference_game",
        "situational_edge_score", "rest_delta",
    ]
    for col in sit_keep:
        if col not in situational.columns:
            situational[col] = pd.NA
    sit_sub = situational[[c for c in sit_keep if c in situational.columns]]

    home_sit = sit_sub.rename(columns={
        "team_id":               "home_team_id",
        "lookahead_flag":        "home_lookahead_flag",
        "letdown_flag":          "home_letdown_flag",
        "bounce_back_flag":      "home_bounce_back_flag",
        "revenge_flag":          "home_revenge_flag",
        "revenge_margin":        "home_revenge_margin",
        "bubble_pressure_flag":  "home_bubble_pressure_flag",
        "must_win_flag":         "home_must_win_flag",
        "fatigue_flag":          "home_fatigue_flag",
        "extended_rest_flag":    "home_extended_rest_flag",
        "situational_edge_score": "home_situational_edge_score",
    })
    away_sit = sit_sub.rename(columns={
        "team_id":               "away_team_id",
        "lookahead_flag":        "away_lookahead_flag",
        "letdown_flag":          "away_letdown_flag",
        "bounce_back_flag":      "away_bounce_back_flag",
        "revenge_flag":          "away_revenge_flag",
        "bubble_pressure_flag":  "away_bubble_pressure_flag",
        "fatigue_flag":          "away_fatigue_flag",
        "extended_rest_flag":    "away_extended_rest_flag",
        "situational_edge_score": "away_situational_edge_score",
    })
    merged = merged.merge(home_sit, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_sit, on=["game_id", "away_team_id"], how="left", suffixes=("", "_awaydup"))

    # Compute situational delta
    merged["situational_edge_delta"] = (
        pd.to_numeric(merged.get("home_situational_edge_score"), errors="coerce")
        - pd.to_numeric(merged.get("away_situational_edge_score"), errors="coerce")
    )
    # Keep the game-level rest_delta (from home row; away version in _awaydup)
    if "rest_delta" not in merged.columns:
        merged["rest_delta"] = pd.NA
    # Override DELTA_MAP rest_delta if situational has it
    sit_rest = pd.to_numeric(merged.get("rest_delta"), errors="coerce")
    calc_rest = pd.to_numeric(merged.get(DELTA_MAP["rest_days"], pd.Series(dtype=float)), errors="coerce")
    merged["rest_delta"] = sit_rest.where(sit_rest.notna(), calc_rest)

    # Shared game-level situational cols (same for both teams)
    for gc in ["is_rivalry_game", "is_neutral_site", "is_conference_game"]:
        if gc not in merged.columns:
            merged[gc] = merged.get(f"{gc}_awaydup", pd.NA)

    # ── 11. Luck regression features ─────────────────────────────────────────
    if not luck_path.exists() or luck_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required input missing: {luck_path}")
    print(f"Joining luck features from {luck_path}...")
    luck = pd.read_csv(luck_path, low_memory=False)
    gid_col = "game_id" if "game_id" in luck.columns else "event_id"
    luck["game_id"] = luck[gid_col].map(normalize_game_id)
    to_numeric(luck, ["team_id"])

    luck_map = {
        "luck_score":           "luck_score_delta",
        "luck_score_l10":       "luck_score_l10_delta",
        "three_pt_luck_l5":     "three_pt_luck_delta",
        "opp_three_pt_luck_l5": "opp_three_pt_luck_delta",
        "close_game_luck_l20":  "close_game_luck_delta",
        "net_rtg_trend":        "net_rtg_trend_delta",
        "efg_luck_l5":          "efg_luck_delta",
        "composite_luck_score": "composite_luck_delta",
    }
    luck_extra = ["regression_candidate_flag"]
    luck_all = list(set(list(luck_map.keys()) + luck_extra))
    luck_avail = [c for c in luck_all if c in luck.columns]

    to_numeric(luck, [c for c in luck_avail if c not in ["regression_candidate_flag"]])
    home_luck = luck[["game_id", "team_id", *luck_avail]].rename(
        columns={
            "team_id": "home_team_id",
            "regression_candidate_flag": "home_regression_flag",
            **{c: f"home_{c}" for c in luck_avail if c != "regression_candidate_flag"},
        }
    )
    away_luck = luck[["game_id", "team_id", *luck_avail]].rename(
        columns={
            "team_id": "away_team_id",
            "regression_candidate_flag": "away_regression_flag",
            **{c: f"away_{c}" for c in luck_avail if c != "regression_candidate_flag"},
        }
    )
    merged = merged.merge(home_luck, on=["game_id", "home_team_id"], how="left")
    merged = merged.merge(away_luck, on=["game_id", "away_team_id"], how="left")

    for src_col, delta_col in luck_map.items():
        hc, ac = f"home_{src_col}", f"away_{src_col}"
        h_ser = merged.get(hc, pd.Series([pd.NA] * len(merged)))
        a_ser = merged.get(ac, pd.Series([pd.NA] * len(merged)))
        merged[delta_col] = [calc_delta(h, a) for h, a in zip(h_ser, a_ser)]

    for flag_col in ["home_regression_flag", "away_regression_flag"]:
        if flag_col not in merged.columns:
            merged[flag_col] = pd.NA

    # ── 12. ATS profile ───────────────────────────────────────────────────────
    if not ats_path.exists() or ats_path.stat().st_size == 0:
        raise FileNotFoundError(f"Required input missing: {ats_path}")
    print(f"Joining ATS profile from {ats_path}...")
    ats = pd.read_csv(ats_path, low_memory=False)
    to_numeric(ats, ["team_id"])
    ats_cols = ["team_id", "cover_rate_season", "cover_rate_l10", "ats_margin_l10",
                "cover_rate_l5", "favorite_cover_rate", "underdog_cover_rate"]
    ats_sub = ats[[c for c in ats_cols if c in ats.columns]]
    ats_metric_cols = [c for c in ats_sub.columns if c != "team_id"]

    home_ats = ats_sub.rename(columns=lambda c: f"home_ats_{c}" if c != "team_id" else c)
    away_ats = ats_sub.rename(columns=lambda c: f"away_ats_{c}" if c != "team_id" else c)
    merged = merged.merge(home_ats, left_on="home_team_id", right_on="team_id", how="left").drop(columns=["team_id"], errors="ignore")
    merged = merged.merge(away_ats, left_on="away_team_id", right_on="team_id", how="left").drop(columns=["team_id"], errors="ignore")

    # ── 13. Snapshot player & injury cols ────────────────────────────────────
    if snap_path.exists() and snap_path.stat().st_size > 0:
        print(f"Joining snapshot player cols from {snap_path}...")
        snap = pd.read_csv(snap_path, low_memory=False)
        to_numeric(snap, ["team_id"])
        snap_player_cols = [
            "team_id", "t_top_scorer_efg_l5", "t_bench_pts_share_l5",
            "t_star_reliance_risk", "t_team_injury_burden", "t_n_injured_starters_l3",
        ]
        snap_sub = snap[[c for c in snap_player_cols if c in snap.columns]]
        snap_metric_cols = [c for c in snap_sub.columns if c != "team_id"]
        home_snap = snap_sub.rename(columns=lambda c: f"home_{c}" if c != "team_id" else c)
        away_snap = snap_sub.rename(columns=lambda c: f"away_{c}" if c != "team_id" else c)
        merged = merged.merge(home_snap, left_on="home_team_id", right_on="team_id", how="left").drop(columns=["team_id"], errors="ignore")
        merged = merged.merge(away_snap, left_on="away_team_id", right_on="team_id", how="left").drop(columns=["team_id"], errors="ignore")

    # ── 14. Predictions history (optional) ───────────────────────────────────
    for phist_path in [pred_path, results_path]:
        if phist_path.exists() and phist_path.stat().st_size > 0:
            pred_df = pd.read_csv(phist_path, low_memory=False)
            if "pred_spread" not in pred_df.columns and "pred_total" not in pred_df.columns:
                continue
            gid_col = "game_id" if "game_id" in pred_df.columns else "event_id"
            pred_df["game_id"] = pred_df[gid_col].map(normalize_game_id)
            pred_cols = [c for c in ["game_id", "pred_spread", "pred_total"] if c in pred_df.columns]
            pred_sub = pred_df[pred_cols].drop_duplicates("game_id", keep="last")
            merged = merged.merge(pred_sub, on="game_id", how="left", suffixes=("", "_predhist"))
            merged["pred_spread"] = resolve_col(merged, "pred_spread", "pred_spread_predhist")
            merged["pred_total"]  = resolve_col(merged, "pred_total",  "pred_total_predhist")
            break  # use first available

    # ── 15. CLV delta ─────────────────────────────────────────────────────────
    to_numeric(merged, ["closing_spread", "pred_spread"])
    merged["clv_delta"] = (
        pd.to_numeric(merged.get("closing_spread"), errors="coerce")
        - pd.to_numeric(merged.get("pred_spread"), errors="coerce")
    )

    # ── 16. Data completeness tier ────────────────────────────────────────────
    tier1_core = [
        "net_rtg_delta_l5", "net_rtg_delta_l10", "adj_ortg_delta", "adj_drtg_delta",
        "efg_delta_l10", "to_rate_delta_l5", "orb_delta_l10",
        "ftrate_delta_l5", "pace_delta_l5", "rest_delta", "travel_fatigue_delta",
    ]
    merged["data_completeness_tier"] = "team_only"
    has_tier3  = merged.get("rot_efg_delta", pd.Series(dtype=float)).notna()
    has_tier4  = merged.get("star_availability_delta", pd.Series(dtype=float)).notna()
    has_market = merged.get("closing_spread", pd.Series(dtype=float)).notna()
    merged.loc[has_market & has_tier3 & has_tier4, "data_completeness_tier"] = "full"
    merged.loc[~has_market, "data_completeness_tier"] = "no_market"

    merged["created_at"] = datetime.now(timezone.utc).isoformat()

    # ── 17. Build final output column list ───────────────────────────────────
    out_cols = [
        # Identifiers
        "game_id", "event_id", "game_date", "season", "game_datetime_utc",
        # Teams
        "home_team_id", "away_team_id", "home_team", "away_team",
        "home_conference", "away_conference", "home_rank", "away_rank",
        # Game context
        "neutral_site", "is_ot", "num_ot",
        # Home box scores (actual game result)
        "home_score_actual", "away_score_actual",
        "home_points_for", "home_points_against",
        "home_h1_pts", "home_h2_pts",
        "home_fgm", "home_fga", "home_tpm", "home_tpa",
        "home_ftm", "home_fta", "home_orb", "home_drb",
        "home_ast", "home_stl", "home_blk", "home_tov", "home_pf",
        # Away box scores
        "away_points_for", "away_points_against",
        "away_h1_pts", "away_h2_pts",
        "away_fgm", "away_fga", "away_tpm", "away_tpa",
        "away_ftm", "away_fta", "away_orb", "away_drb",
        "away_ast", "away_stl", "away_blk", "away_tov", "away_pf",
        # Half scores (game-level)
        "home_h1", "away_h1", "home_h2", "away_h2",
        # Records at time of game
        "home_wins", "home_losses", "home_win_pct",
        "home_home_wins", "home_home_losses",
        "home_away_wins", "home_away_losses",
        "home_conf_wins", "home_conf_losses",
        "away_wins", "away_losses", "away_win_pct",
        "away_home_wins", "away_home_losses",
        "away_away_wins", "away_away_losses",
        "away_conf_wins", "away_conf_losses",
        # Per-team rolling features (home)
        "home_net_rtg_l5", "home_net_rtg_l10",
        "home_adj_ortg", "home_adj_drtg", "home_adj_net_rtg",
        "home_efg_pct_l5", "home_efg_pct_l10",
        "home_tov_pct_l5", "home_orb_pct_l10", "home_ftr_l5",
        "home_pace_l5", "home_ortg_l5", "home_drtg_l5",
        "home_margin_l5", "home_margin_l10",
        "home_cover_l10", "home_cover_rate_l10", "home_cover_rate_season",
        "home_ats_margin_l10", "home_cover_margin",
        "home_rest_days", "home_fatigue_index",
        "home_momentum_score", "home_form_rating", "home_luck_score",
        "home_win_streak", "home_cover_streak",
        # Per-team rolling features (away)
        "away_net_rtg_l5", "away_net_rtg_l10",
        "away_adj_ortg", "away_adj_drtg", "away_adj_net_rtg",
        "away_efg_pct_l5", "away_efg_pct_l10",
        "away_tov_pct_l5", "away_orb_pct_l10", "away_ftr_l5",
        "away_pace_l5", "away_ortg_l5", "away_drtg_l5",
        "away_margin_l5", "away_margin_l10",
        "away_cover_l10", "away_cover_rate_l10", "away_cover_rate_season",
        "away_ats_margin_l10", "away_cover_margin",
        "away_rest_days", "away_fatigue_index",
        "away_momentum_score", "away_form_rating", "away_luck_score",
        "away_win_streak", "away_cover_streak",
        # Delta features (home minus away)
        "net_rtg_delta_l5", "net_rtg_delta_l10",
        "adj_ortg_delta", "adj_drtg_delta", "adj_net_rtg_delta",
        "efg_delta_l10", "to_rate_delta_l5", "orb_delta_l10",
        "ftrate_delta_l5", "pace_delta_l5",
        "home_field", "rest_delta", "travel_fatigue_delta",
        # Cage rankings differentials
        "cage_em_diff", "cage_t_diff", "cage_o_diff", "cage_d_diff",
        # Per-game player aggregates (home)
        "home_pg_top_scorer_pts", "home_pg_top_scorer_efg",
        "home_pg_bench_pts", "home_pg_bench_pts_share",
        "home_pg_starters_count", "home_pg_bench_min_share",
        "home_pg_starters_fgm", "home_pg_starters_fga",
        # Per-game player aggregates (away)
        "away_pg_top_scorer_pts", "away_pg_top_scorer_efg",
        "away_pg_bench_pts", "away_pg_bench_pts_share",
        "away_pg_starters_count", "away_pg_bench_min_share",
        "away_pg_starters_fgm", "away_pg_starters_fga",
        # Rotation features (deltas)
        "rot_efg_delta", "rot_to_swing_diff", "exec_tax_diff",
        "three_pt_fragility_diff", "rot_minshare_sd_diff",
        "top2_pused_share_diff", "closer_ft_pct_delta",
        # Rotation raw (home)
        "home_rot_size", "home_rot_efg_l5", "home_rot_efg_l10",
        "home_rot_to_rate_l5", "home_closer_ft_pct",
        # Rotation raw (away)
        "away_rot_size", "away_rot_efg_l5", "away_rot_efg_l10",
        "away_rot_to_rate_l5", "away_closer_ft_pct",
        # Availability (deltas)
        "star_availability_delta", "minutes_available_delta",
        "lineup_continuity_delta", "usage_gini_delta",
        "new_starter_flag_home", "new_starter_flag_away",
        # ATS profile
        "home_ats_cover_rate_season", "home_ats_cover_rate_l10",
        "home_ats_ats_margin_l10", "home_ats_cover_rate_l5",
        "home_ats_favorite_cover_rate", "home_ats_underdog_cover_rate",
        "away_ats_cover_rate_season", "away_ats_cover_rate_l10",
        "away_ats_ats_margin_l10", "away_ats_cover_rate_l5",
        "away_ats_favorite_cover_rate", "away_ats_underdog_cover_rate",
        # Snapshot player cols
        "home_t_top_scorer_efg_l5", "home_t_bench_pts_share_l5",
        "home_t_star_reliance_risk", "home_t_team_injury_burden",
        "home_t_n_injured_starters_l3",
        "away_t_top_scorer_efg_l5", "away_t_bench_pts_share_l5",
        "away_t_star_reliance_risk", "away_t_team_injury_burden",
        "away_t_n_injured_starters_l3",
        # Market & predictions
        "espn_spread", "espn_total",
        "opening_spread", "closing_spread", "total_line", "spread_line",
        "clv_delta", "pred_spread", "pred_total",
        # Situational
        "home_lookahead_flag", "home_letdown_flag", "home_bounce_back_flag",
        "home_revenge_flag", "home_revenge_margin",
        "away_lookahead_flag", "away_letdown_flag", "away_bounce_back_flag",
        "away_revenge_flag",
        "home_bubble_pressure_flag", "away_bubble_pressure_flag",
        "home_must_win_flag", "home_fatigue_flag", "away_fatigue_flag",
        "home_extended_rest_flag", "away_extended_rest_flag",
        "is_rivalry_game", "is_neutral_site", "is_conference_game",
        "situational_edge_delta",
        # Outcomes
        "actual_margin", "home_covered_ats",
        "actual_total", "covered_over",
        # Luck
        "luck_score_delta", "luck_score_l10_delta",
        "three_pt_luck_delta", "opp_three_pt_luck_delta",
        "close_game_luck_delta", "net_rtg_trend_delta",
        "efg_luck_delta", "composite_luck_delta",
        "home_regression_flag", "away_regression_flag",
        # Metadata
        "data_completeness_tier", "created_at",
    ]

    # Fill missing output columns with NA
    for col in out_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    out_df = merged[out_cols].copy()

    # ── 18. Summary stats ─────────────────────────────────────────────────────
    total_rows   = len(out_df)
    rows_tier12  = out_df["net_rtg_delta_l10"].notna().sum()
    rows_tier3   = out_df["rot_efg_delta"].notna().sum()
    rows_tier4   = out_df["star_availability_delta"].notna().sum()
    rows_market  = out_df["closing_spread"].notna().sum()
    rows_spread  = out_df["spread_line"].notna().sum()
    rows_graded  = out_df["home_covered_ats"].notna().sum()
    rows_pred    = out_df["pred_spread"].notna().sum()

    tier1_core_cols = [
        "net_rtg_delta_l5", "net_rtg_delta_l10", "adj_ortg_delta", "adj_drtg_delta",
        "efg_delta_l10", "to_rate_delta_l5", "orb_delta_l10",
        "ftrate_delta_l5", "pace_delta_l5", "rest_delta", "travel_fatigue_delta",
    ]
    full_feat_cols = tier1_core_cols + [
        "rot_efg_delta", "rot_minshare_sd_diff", "top2_pused_share_diff",
        "star_availability_delta", "lineup_continuity_delta",
        "closing_spread", "total_line",
    ]
    usable_tier1 = out_df[
        out_df["home_covered_ats"].notna()
        & out_df[tier1_core_cols].notna().all(axis=1)
    ].shape[0]
    usable_full = out_df[
        out_df["home_covered_ats"].notna()
        & out_df[full_feat_cols].notna().all(axis=1)
    ].shape[0]

    print(f"\nTraining data summary:")
    print(f"  Total rows (all post games):         {total_rows}")
    print(f"  Rows with Tier 1+2 (net_rtg non-null): {rows_tier12}")
    print(f"  Rows with Tier 3 (rot_efg non-null):    {rows_tier3}")
    print(f"  Rows with Tier 4 (avail non-null):      {rows_tier4}")
    print(f"  Rows with closing market spread:        {rows_market}")
    print(f"  Rows with any spread (market|ESPN):     {rows_spread}")
    print(f"  Rows ATS-graded (home_covered_ats):     {rows_graded}")
    print(f"  Rows with pred_spread:                  {rows_pred}")
    print(f"  Usable for Tier-1 optimization:         {usable_tier1}")
    print(f"  Usable for full optimization:           {usable_full}")
    print(f"  Output columns:                         {len(out_cols)}")

    if usable_tier1 < 50:
        print("[WARNING] Fewer than 50 graded rows with Tier 1 features.")
        print("          Optimizer will have limited training data.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"\nWrote training data to {output_path}")


if __name__ == "__main__":
    main()
