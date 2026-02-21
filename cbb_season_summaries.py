"""
cbb_season_summaries.py — Season-Aggregated Summary Builder

Reads existing per-game CSVs and writes two season-level summary files:
  data/player_season_summary.csv — one row per player, season aggregates
  data/team_season_summary.csv   — one row per team, season aggregates

Idempotent: always rebuilds from scratch on each run.
First-run safe: gracefully skips or returns empty output if input files
are missing.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path("data")

PLAYER_LOGS_CSV   = DATA_DIR / "player_game_logs.csv"
PLAYER_METRICS_CSV = DATA_DIR / "player_game_metrics.csv"
TEAM_WEIGHTED_CSV  = DATA_DIR / "team_game_weighted.csv"
RESULTS_LOG_CSV    = DATA_DIR / "results_log.csv"

PLAYER_SUMMARY_CSV = DATA_DIR / "player_season_summary.csv"
TEAM_SUMMARY_CSV   = DATA_DIR / "team_season_summary.csv"


def _safe_read(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV if it exists and is non-empty; otherwise return empty DataFrame."""
    if not path.exists() or path.stat().st_size == 0:
        log.warning(f"Input file not found or empty: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        log.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def build_player_season_summary(output_path: Path = PLAYER_SUMMARY_CSV) -> pd.DataFrame:
    """
    Build player_season_summary.csv from player_game_logs.csv +
    player_game_metrics.csv.

    Returns the summary DataFrame (empty if inputs are missing).
    """
    logs    = _safe_read(PLAYER_LOGS_CSV)
    metrics = _safe_read(PLAYER_METRICS_CSV)

    if logs.empty and metrics.empty:
        log.info("No player data available — writing empty player_season_summary.csv")
        empty = pd.DataFrame(columns=[
            "athlete_id", "player", "team_id", "team",
            "games_played", "avg_min", "avg_pts", "avg_reb", "avg_ast",
            "avg_stl", "avg_blk", "avg_tov", "ts_pct", "usage_pct",
            "season_pts_share", "updated_at",
        ])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output_path, index=False)
        return empty

    # Prefer metrics file when available; fall back to logs
    base = metrics if not metrics.empty else logs

    # Numeric stat columns we want to aggregate
    stat_cols = {
        "min":       "avg_min",
        "pts":       "avg_pts",
        "reb":       "avg_reb",
        "ast":       "avg_ast",
        "stl":       "avg_stl",
        "blk":       "avg_blk",
        "tov":       "avg_tov",
    }

    id_col   = "athlete_id" if "athlete_id" in base.columns else None
    name_col = "player"     if "player"     in base.columns else None
    tid_col  = "team_id"    if "team_id"    in base.columns else None
    team_col = "team"       if "team"       in base.columns else None

    group_keys = [c for c in [id_col, name_col, tid_col, team_col] if c]
    if not group_keys:
        log.warning("player_game_logs/metrics missing identity columns — skipping player summary")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()

    # Convert stat columns to numeric
    for col in stat_cols:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    grouped = base.groupby(group_keys, as_index=False)

    # games_played = row count per player
    gp = base.groupby(group_keys).size().reset_index(name="games_played")
    means_dict: dict[str, object] = {}
    for src, dst in stat_cols.items():
        if src in base.columns:
            means_dict[src] = "mean"
    if means_dict:
        means = grouped.agg(means_dict).rename(columns={s: d for s, d in stat_cols.items()
                                                         if s in base.columns})
        summary = gp.merge(means, on=group_keys, how="left")
    else:
        summary = gp

    # ── TS% = pts / (2*(fga + 0.44*fta)) ────────────────────────────────────
    for col in ["fga", "fta", "pts"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    if all(c in base.columns for c in ["pts", "fga", "fta"]):
        ts_df = base.groupby(group_keys)[["pts", "fga", "fta"]].sum().reset_index()
        ts_df["ts_pct"] = (ts_df["pts"] /
                           (2 * (ts_df["fga"] + 0.44 * ts_df["fta"]))).round(4)
        ts_df["ts_pct"] = ts_df["ts_pct"].clip(0, 1)
        summary = summary.merge(ts_df[group_keys + ["ts_pct"]], on=group_keys, how="left")
    else:
        summary["ts_pct"] = None

    # ── usage_pct heuristic = (player pts share within team+game) ─────────────
    summary["usage_pct"] = None

    # ── season_pts_share = player total pts / team total pts ──────────────────
    if "pts" in base.columns and tid_col:
        player_pts = base.groupby(group_keys)["pts"].sum().reset_index(name="_player_pts")
        team_pts   = base.groupby([tid_col])["pts"].sum().reset_index(name="_team_pts")
        summary = summary.merge(player_pts, on=group_keys, how="left")
        summary = summary.merge(team_pts, on=[tid_col], how="left")
        summary["season_pts_share"] = (
            summary["_player_pts"] / summary["_team_pts"].replace(0, pd.NA)
        ).round(4)
        summary = summary.drop(columns=["_player_pts", "_team_pts"])
    else:
        summary["season_pts_share"] = None

    summary["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Reorder to canonical columns (drop extras)
    final_cols = [
        "athlete_id", "player", "team_id", "team",
        "games_played", "avg_min", "avg_pts", "avg_reb", "avg_ast",
        "avg_stl", "avg_blk", "avg_tov", "ts_pct", "usage_pct",
        "season_pts_share", "updated_at",
    ]
    for c in final_cols:
        if c not in summary.columns:
            summary[c] = None
    summary = summary[final_cols]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    log.info(f"player_season_summary.csv → {output_path}  ({len(summary):,} rows)")
    return summary


def build_team_season_summary(output_path: Path = TEAM_SUMMARY_CSV) -> pd.DataFrame:
    """
    Build team_season_summary.csv from team_game_weighted.csv +
    results_log.csv (optional, for ATS/O-U columns).

    Returns the summary DataFrame (empty if inputs are missing).
    """
    weighted = _safe_read(TEAM_WEIGHTED_CSV)
    results  = _safe_read(RESULTS_LOG_CSV)

    empty_cols = [
        "team_id", "team", "conference",
        "wins", "losses",
        "home_wins", "home_losses",
        "away_wins", "away_losses",
        "neutral_wins", "neutral_losses",
        "avg_margin", "avg_ortg", "avg_drtg", "avg_pace",
        "ats_wins", "ats_losses",
        "ou_over", "ou_under",
        "avg_pred_spread_error",
        "updated_at",
    ]

    if weighted.empty:
        log.info("No team_game_weighted data — writing empty team_season_summary.csv")
        empty = pd.DataFrame(columns=empty_cols)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output_path, index=False)
        return empty

    # ── Build win/loss records from weighted CSV ───────────────────────────────
    tid_col  = "team_id"   if "team_id"   in weighted.columns else None
    team_col = "team"      if "team"      in weighted.columns else None
    conf_col = "conference" if "conference" in weighted.columns else None
    ha_col   = "home_away"  if "home_away"  in weighted.columns else None

    if tid_col is None and team_col is None:
        log.warning("team_game_weighted missing team_id/team columns — skipping team summary")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()

    group_keys = [c for c in [tid_col, team_col] if c]
    df = weighted.copy()

    # Determine if team won: actual_margin > 0 means home team won.
    # team_game_weighted is one row per team per game; we need to know the team's perspective.
    if "actual_margin" in df.columns and ha_col:
        df["actual_margin"] = pd.to_numeric(df["actual_margin"], errors="coerce")
        df["_won"] = (
            ((df[ha_col] == "home") & (df["actual_margin"] > 0)) |
            ((df[ha_col] == "away") & (df["actual_margin"] < 0))
        ).astype(int)
    elif "actual_margin" in df.columns:
        df["actual_margin"] = pd.to_numeric(df["actual_margin"], errors="coerce")
        df["_won"] = (df["actual_margin"] > 0).astype(int)
    else:
        df["_won"] = None

    # Games played
    gp = df.groupby(group_keys).size().reset_index(name="_gp")

    agg_map: dict[str, object] = {}
    if "_won" in df.columns and df["_won"].notna().any():
        agg_map["_won"] = "sum"
    for col in ["actual_margin", "adj_off_rtg", "adj_def_rtg", "pace"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            agg_map[col] = "mean"

    if agg_map:
        sums = df.groupby(group_keys).agg(agg_map).reset_index()
    else:
        sums = df[group_keys].drop_duplicates()

    summary = gp.merge(sums, on=group_keys, how="left")

    summary["wins"]   = summary.get("_won", pd.Series(dtype=float)).fillna(0).astype(int)
    summary["losses"] = summary["_gp"] - summary["wins"]

    # ── Home / away / neutral splits ──────────────────────────────────────────
    for venue, label in [("home", "home"), ("away", "away"), ("neutral", "neutral")]:
        if ha_col:
            sub = df[df[ha_col] == venue]
        else:
            sub = pd.DataFrame()

        if not sub.empty and "_won" in sub.columns:
            vw = sub.groupby(group_keys)["_won"].agg(["sum", "count"]).reset_index()
            vw.columns = list(group_keys) + [f"{label}_wins", f"{label}_gp"]
            vw[f"{label}_losses"] = vw[f"{label}_gp"] - vw[f"{label}_wins"]
            summary = summary.merge(
                vw[group_keys + [f"{label}_wins", f"{label}_losses"]],
                on=group_keys, how="left",
            )
        else:
            summary[f"{label}_wins"]   = None
            summary[f"{label}_losses"] = None

    # ── Rating columns ─────────────────────────────────────────────────────────
    col_map = {
        "actual_margin": "avg_margin",
        "adj_off_rtg":   "avg_ortg",
        "adj_def_rtg":   "avg_drtg",
        "pace":          "avg_pace",
    }
    for src, dst in col_map.items():
        if src in summary.columns:
            summary[dst] = summary[src].round(2)
        else:
            summary[dst] = None

    # conference column
    if conf_col:
        conf_df = df.groupby(group_keys)[conf_col].first().reset_index()
        summary = summary.merge(conf_df, on=group_keys, how="left")
        summary = summary.rename(columns={conf_col: "conference"})
    else:
        summary["conference"] = None

    # ── ATS / O-U from results_log ────────────────────────────────────────────
    summary["ats_wins"]            = None
    summary["ats_losses"]          = None
    summary["ou_over"]             = None
    summary["ou_under"]            = None
    summary["avg_pred_spread_error"] = None

    if not results.empty and tid_col:
        for side in ["home", "away"]:
            tid_field = f"{side}_team_id"
            ats_field = "primary_ats_correct"
            ou_field  = "primary_ou_correct"
            err_field = "primary_margin_error"

            if tid_field not in results.columns:
                continue

            for col in [ats_field, ou_field, err_field]:
                if col in results.columns:
                    results[col] = pd.to_numeric(results[col], errors="coerce")

            sub = results.rename(columns={tid_field: tid_col})
            if team_col and f"{side}_team" in results.columns:
                sub = sub.rename(columns={f"{side}_team": team_col})

            agg2: dict[str, object] = {}
            if ats_field in sub.columns:
                agg2[ats_field] = "sum"
                agg2["_ats_n"]  = (ats_field, "count")
            if ou_field in sub.columns:
                agg2[ou_field]   = "sum"
            if err_field in sub.columns:
                agg2[err_field] = "mean"

            if agg2:
                sub_keys = [c for c in group_keys if c in sub.columns]
                if sub_keys:
                    r_agg = (
                        sub.groupby(sub_keys)
                        .agg(**{k: v for k, v in agg2.items() if k != "_ats_n"})
                        .reset_index()
                    )
                    if ats_field in r_agg.columns:
                        r_agg["ats_wins_part"]   = r_agg[ats_field].fillna(0).astype(int)
                        r_agg["ats_games_part"]  = sub.groupby(sub_keys)[ats_field].count().values
                    if ou_field in r_agg.columns:
                        r_agg["ou_over_part"]  = r_agg[ou_field].fillna(0).astype(int)
                    if err_field in r_agg.columns:
                        r_agg["err_part"] = r_agg[err_field]

                    summary = summary.merge(
                        r_agg[sub_keys + [c for c in
                              ["ats_wins_part", "ats_games_part", "ou_over_part", "err_part"]
                              if c in r_agg.columns]],
                        on=sub_keys, how="left",
                    )

        # Combine home/away partial columns if both sides contributed
        for base_col, part_col in [("ats_wins", "ats_wins_part"), ("ou_over", "ou_over_part")]:
            if part_col in summary.columns:
                summary[base_col] = summary.get(part_col, pd.Series()).fillna(0).astype(int)

        if "ats_games_part" in summary.columns and "ats_wins" in summary.columns:
            summary["ats_losses"] = (
                summary["ats_games_part"].fillna(0).astype(int) - summary["ats_wins"]
            )

        if "ou_over_part" in summary.columns:
            summary["ou_under"] = None  # we don't separately track O/U losses here

        if "err_part" in summary.columns:
            summary["avg_pred_spread_error"] = summary["err_part"].round(3)

        # Drop scratch columns
        drop_cols = [c for c in summary.columns if c.endswith("_part")]
        summary = summary.drop(columns=drop_cols, errors="ignore")

    summary["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Reorder to canonical columns
    final_cols = [
        "team_id", "team", "conference",
        "wins", "losses",
        "home_wins", "home_losses",
        "away_wins", "away_losses",
        "neutral_wins", "neutral_losses",
        "avg_margin", "avg_ortg", "avg_drtg", "avg_pace",
        "ats_wins", "ats_losses",
        "ou_over", "ou_under",
        "avg_pred_spread_error",
        "updated_at",
    ]
    for c in final_cols:
        if c not in summary.columns:
            summary[c] = None
    summary = summary[final_cols]

    # Drop internal scratch columns that may have leaked through
    summary = summary.drop(
        columns=[c for c in summary.columns if c.startswith("_")],
        errors="ignore",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    log.info(f"team_season_summary.csv → {output_path}  ({len(summary):,} rows)")
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    build_player_season_summary()
    build_team_season_summary()
    log.info("Season summaries complete.")


if __name__ == "__main__":
    main()
