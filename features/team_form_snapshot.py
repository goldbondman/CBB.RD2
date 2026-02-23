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
TEAM_METRICS_CSV   = DATA_DIR / "team_game_metrics.csv"
GAMES_CSV          = DATA_DIR / "games.csv"
RANKINGS_CSV       = DATA_DIR / "cbb_rankings.csv"
TEAM_SOS_CSV       = DATA_DIR / "team_game_sos.csv"

PLAYER_SUMMARY_CSV     = DATA_DIR / "player_season_summary.csv"
TEAM_SUMMARY_CSV       = DATA_DIR / "team_season_summary.csv"
TEAM_ATS_PROFILE_CSV   = DATA_DIR / "team_ats_profile.csv"
TEAM_LUCK_CSV          = DATA_DIR / "team_luck_regression.csv"
TEAM_SITUATIONAL_CSV   = DATA_DIR / "team_situational.csv"
TEAM_RESUME_CSV        = DATA_DIR / "team_resume.csv"
TEAM_MATCHUP_CSV       = DATA_DIR / "team_matchup_history.csv"
CONF_SUMMARY_CSV       = DATA_DIR / "conference_daily_summary.csv"


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


def build_team_ats_profile(output_path: Path = TEAM_ATS_PROFILE_CSV) -> pd.DataFrame:
    """
    Build team_ats_profile.csv — one row per team, ATS/O-U snapshot.
    """
    df = _safe_read(TEAM_METRICS_CSV)
    if df.empty:
        log.info("No team_game_metrics data — skipping team_ats_profile")
        return pd.DataFrame()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    id_cols = [c for c in ["team_id", "team", "conference"] if c in df.columns]
    if "cover" in df.columns:
        df["cover"] = pd.to_numeric(df["cover"], errors="coerce")
    if "cover_margin" in df.columns:
        df["cover_margin"] = pd.to_numeric(df["cover_margin"], errors="coerce")
    if "ou_result" in df.columns:
        df["ou_result"] = pd.to_numeric(df["ou_result"], errors="coerce")

    rows = []
    for keys, grp in df.groupby(id_cols):
        grp = grp.sort_values("game_datetime_utc", na_position="last") if "game_datetime_utc" in grp.columns else grp
        record: dict = {}
        for k, v in zip(id_cols, keys if isinstance(keys, tuple) else (keys,)):
            record[k] = v

        cover_s = grp["cover"] if "cover" in grp.columns else pd.Series(dtype=float)
        cm_s    = grp["cover_margin"] if "cover_margin" in grp.columns else pd.Series(dtype=float)

        record["games_with_spread"] = int(cover_s.notna().sum())
        record["cover_rate_season"] = round(float(cover_s.mean()), 4) if cover_s.notna().any() else None
        record["cover_rate_l10"]    = round(float(cover_s.tail(10).mean()), 4) if cover_s.notna().any() else None
        record["cover_rate_l5"]     = round(float(cover_s.tail(5).mean()), 4) if cover_s.notna().any() else None
        record["ats_margin_season"] = round(float(cm_s.mean()), 4) if cm_s.notna().any() else None
        record["ats_margin_l10"]    = round(float(cm_s.tail(10).mean()), 4) if cm_s.notna().any() else None
        record["ats_margin_l5"]     = round(float(cm_s.tail(5).mean()), 4) if cm_s.notna().any() else None

        if "home_away" in grp.columns:
            home_cover = grp.loc[grp["home_away"] == "home", "cover"] if "cover" in grp.columns else pd.Series(dtype=float)
            away_cover = grp.loc[grp["home_away"] == "away", "cover"] if "cover" in grp.columns else pd.Series(dtype=float)
            record["home_cover_rate"] = round(float(home_cover.mean()), 4) if home_cover.notna().any() else None
            record["away_cover_rate"] = round(float(away_cover.mean()), 4) if away_cover.notna().any() else None
        else:
            record["home_cover_rate"] = None
            record["away_cover_rate"] = None

        # cover_streak: use existing column from last row, or compute
        if "cover_streak" in grp.columns:
            record["cover_streak"] = grp["cover_streak"].iloc[-1] if len(grp) > 0 else 0
        else:
            streak = 0
            if "cover" in grp.columns:
                for v in reversed(cover_s.dropna().tolist()):
                    cv = int(v)
                    if streak == 0:
                        streak = 1 if cv else -1
                    elif cv and streak > 0:
                        streak += 1
                    elif not cv and streak < 0:
                        streak -= 1
                    else:
                        break
            record["cover_streak"] = streak

        ou_s = grp["ou_result"] if "ou_result" in grp.columns else pd.Series(dtype=float)
        record["ou_over_rate_season"] = round(float(ou_s.mean()), 4) if ou_s.notna().any() else None
        record["ou_over_rate_l10"]    = round(float(ou_s.tail(10).mean()), 4) if ou_s.notna().any() else None

        record["updated_at"] = now
        rows.append(record)

    out_cols = [
        "team_id", "team", "conference",
        "cover_rate_season", "cover_rate_l10", "cover_rate_l5",
        "ats_margin_season", "ats_margin_l10", "ats_margin_l5",
        "home_cover_rate", "away_cover_rate",
        "cover_streak", "games_with_spread",
        "ou_over_rate_season", "ou_over_rate_l10", "updated_at",
    ]
    summary = pd.DataFrame(rows)
    for c in out_cols:
        if c not in summary.columns:
            summary[c] = None
    summary = summary[[c for c in out_cols if c in summary.columns]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    log.info(f"team_ats_profile.csv → {output_path}  ({len(summary):,} rows)")
    return summary


def build_team_luck_regression(output_path: Path = TEAM_LUCK_CSV) -> pd.DataFrame:
    """
    Build team_luck_regression.csv — one row per team, luck/regression snapshot.
    """
    df = _safe_read(TEAM_METRICS_CSV)
    if df.empty:
        log.info("No team_game_metrics data — skipping team_luck_regression")
        return pd.DataFrame()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    id_cols = [c for c in ["team_id", "team", "conference"] if c in df.columns]

    for col in ["win", "pythagorean_win_pct", "luck_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    rows = []
    for keys, grp in df.groupby(id_cols):
        grp = grp.sort_values("game_datetime_utc", na_position="last") if "game_datetime_utc" in grp.columns else grp
        record: dict = {}
        for k, v in zip(id_cols, keys if isinstance(keys, tuple) else (keys,)):
            record[k] = v

        gp = len(grp)
        record["games_played"] = gp

        win_s = grp["win"] if "win" in grp.columns else pd.Series(dtype=float)
        record["actual_win_pct"] = round(float(win_s.mean()), 4) if win_s.notna().any() else 0.0

        pyth_s = grp["pythagorean_win_pct"] if "pythagorean_win_pct" in grp.columns else pd.Series(dtype=float)
        record["pythagorean_win_pct"] = round(float(pyth_s.mean()), 4) if pyth_s.notna().any() else 0.0

        record["luck_score"] = round(record["actual_win_pct"] - record["pythagorean_win_pct"], 4)

        luck_col = grp["luck_score"] if "luck_score" in grp.columns else pd.Series(dtype=float)
        record["luck_trend_l5"] = round(float(luck_col.tail(5).mean()), 4) if luck_col.notna().any() else 0.0

        record["regression_risk_flag"] = int(record["luck_score"] > 0.08 and gp >= 10)
        record["expected_future_win_pct"] = record["pythagorean_win_pct"]
        record["updated_at"] = now
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)

    if "luck_score" in summary.columns:
        summary["luck_rank"] = summary["luck_score"].rank(method="min", ascending=False).astype("Int64")
        summary["luck_percentile"] = (summary["luck_score"].rank(pct=True) * 100).round(1)
    else:
        summary["luck_rank"] = None
        summary["luck_percentile"] = None

    out_cols = [
        "team_id", "team", "conference",
        "games_played", "actual_win_pct", "pythagorean_win_pct",
        "luck_score", "luck_rank", "luck_percentile",
        "regression_risk_flag", "expected_future_win_pct",
        "luck_trend_l5", "updated_at",
    ]
    for c in out_cols:
        if c not in summary.columns:
            summary[c] = None
    summary = summary[[c for c in out_cols if c in summary.columns]]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    log.info(f"team_luck_regression.csv → {output_path}  ({len(summary):,} rows)")
    return summary


def build_team_situational(output_path: Path = TEAM_SITUATIONAL_CSV) -> pd.DataFrame:
    """
    Build team_situational.csv — one row per team per game, situational columns.
    """
    df = _safe_read(TEAM_METRICS_CSV)
    if df.empty:
        log.info("No team_game_metrics data — skipping team_situational")
        return pd.DataFrame()

    sit_cols = [
        "event_id", "team_id", "game_datetime_utc", "home_away",
        "rest_days", "games_l7", "games_l14",
        "win_streak", "lose_streak", "cover_streak",
        "close_win_pct_season", "close_game_win_pct",
        "blowout_flag", "close_game_flag",
        "margin", "margin_capped",
    ]
    present = [c for c in sit_cols if c in df.columns]
    out = df[present].copy()

    # scoring_consistency_l5: std of margin over last 5 games (rolling, shift(1), min_periods=3)
    if "margin" in df.columns and "team_id" in df.columns:
        df_sorted = df.copy()
        if "game_datetime_utc" in df_sorted.columns:
            df_sorted = df_sorted.sort_values(["team_id", "game_datetime_utc"])
        df_sorted["margin"] = pd.to_numeric(df_sorted["margin"], errors="coerce")
        out = out.copy()
        out.index = df_sorted.index
        out["scoring_consistency_l5"] = df_sorted.groupby("team_id")["margin"].transform(
            lambda s: s.shift(1).rolling(5, min_periods=3).std().round(2)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    log.info(f"team_situational.csv → {output_path}  ({len(out):,} rows)")
    return out


def build_team_resume(output_path: Path = TEAM_RESUME_CSV) -> pd.DataFrame:
    """
    Build team_resume.csv — one row per team, resume/quad record snapshot.
    """
    games    = _safe_read(GAMES_CSV)
    metrics  = _safe_read(TEAM_METRICS_CSV)
    sos      = _safe_read(TEAM_SOS_CSV)
    rankings = _safe_read(RANKINGS_CSV)

    if games.empty and metrics.empty:
        log.info("No games/metrics data — skipping team_resume")
        return pd.DataFrame()

    base = metrics if not metrics.empty else games
    id_cols = [c for c in ["team_id", "team", "conference"] if c in base.columns]
    if not id_cols:
        log.warning("build_team_resume: missing identity columns")
        return pd.DataFrame()

    # Merge SOS opponent net rating if available
    if not sos.empty and "opp_avg_net_rtg_season" in sos.columns:
        sos_cols = [c for c in ["event_id", "team_id", "opp_avg_net_rtg_season"] if c in sos.columns]
        merge_on = [c for c in ["event_id", "team_id"] if c in base.columns and c in sos.columns]
        if merge_on:
            base = base.merge(sos[sos_cols].drop_duplicates(subset=merge_on),
                              on=merge_on, how="left", suffixes=("", "_sos"))
            if "opp_avg_net_rtg_season_sos" in base.columns and "opp_avg_net_rtg_season" not in base.columns:
                base = base.rename(columns={"opp_avg_net_rtg_season_sos": "opp_avg_net_rtg_season"})

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for col in ["win", "opp_avg_net_rtg_season"]:
        if col in base.columns:
            base[col] = pd.to_numeric(base[col], errors="coerce")

    # WAB lookup
    wab_map: dict = {}
    if not rankings.empty and "team_id" in rankings.columns and "wab" in rankings.columns:
        wab_map = rankings.set_index("team_id")["wab"].to_dict()

    rows = []
    for keys, grp in base.groupby(id_cols):
        record: dict = {}
        for k, v in zip(id_cols, keys if isinstance(keys, tuple) else (keys,)):
            record[k] = v

        win_s = grp["win"] if "win" in grp.columns else pd.Series(dtype=float)
        ha_s  = grp["home_away"] if "home_away" in grp.columns else pd.Series("", index=grp.index)
        conf_s = grp.get("is_conf_game", None)
        opp_net = grp["opp_avg_net_rtg_season"] if "opp_avg_net_rtg_season" in grp.columns else pd.Series(dtype=float)

        record["wins"]   = int(win_s.sum() if win_s.notna().any() else 0)
        record["losses"] = int((1 - win_s).sum() if win_s.notna().any() else 0)

        record["home_wins"]    = int(win_s[ha_s == "home"].sum()) if (ha_s == "home").any() else 0
        record["home_losses"]  = int((1 - win_s[ha_s == "home"]).sum()) if (ha_s == "home").any() else 0
        record["away_wins"]    = int(win_s[ha_s == "away"].sum()) if (ha_s == "away").any() else 0
        record["away_losses"]  = int((1 - win_s[ha_s == "away"]).sum()) if (ha_s == "away").any() else 0
        record["neutral_wins"] = int(win_s[ha_s == "neutral"].sum()) if (ha_s == "neutral").any() else 0

        # Note: standard NCAA quad definitions use NET rank + home/away/neutral splits.
        # Here we use opp_avg_net_rtg_season as a proxy (Q1 proxy: >5.0, Q2: 0–5)
        # since per-game NET rank data is not available in this pipeline.
        q1_mask = opp_net > 5.0
        q2_mask = (opp_net >= 0.0) & (opp_net <= 5.0)
        q1_w = int(win_s[q1_mask].sum()) if q1_mask.any() else 0
        q1_l = int((1 - win_s[q1_mask]).sum()) if q1_mask.any() else 0
        q2_w = int(win_s[q2_mask].sum()) if q2_mask.any() else 0
        q2_l = int((1 - win_s[q2_mask]).sum()) if q2_mask.any() else 0
        record["q1_wins"]    = q1_w
        record["q1_losses"]  = q1_l
        record["q2_wins"]    = q2_w
        record["q2_losses"]  = q2_l
        record["road_wins"]  = record["away_wins"]
        record["q1_win_pct"] = round(q1_w / (q1_w + q1_l), 4) if (q1_w + q1_l) > 0 else 0.0

        tid = record.get("team_id")
        record["wab"] = float(wab_map.get(tid, 0.0)) if tid else 0.0

        # resume_score
        if not rankings.empty and "team_id" in rankings.columns and "resume_score" in rankings.columns:
            rs_map = rankings.set_index("team_id")["resume_score"].to_dict()
            record["resume_score"] = float(rs_map.get(tid, 0.0)) if tid else 0.0
        else:
            record["resume_score"] = round(
                q1_w * 2.5 + q2_w * 1.0 - q1_l * 1.5 - q2_l * 0.5 + record["road_wins"] * 1.0,
                2,
            )

        record["updated_at"] = now
        rows.append(record)

    if not rows:
        return pd.DataFrame()

    summary = pd.DataFrame(rows)
    if "opp_avg_net_rtg_season" in base.columns:
        sos_latest = (
            base.sort_values("game_datetime_utc", na_position="last")
            .groupby(id_cols)["opp_avg_net_rtg_season"]
            .last()
            .reset_index(name="_sos")
        )
        summary = summary.merge(sos_latest, on=id_cols, how="left")
        summary["sos_rank"] = summary["_sos"].rank(method="min", ascending=False).astype("Int64")
        summary = summary.drop(columns=["_sos"], errors="ignore")
    else:
        summary["sos_rank"] = None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    log.info(f"team_resume.csv → {output_path}  ({len(summary):,} rows)")
    return summary


def build_team_matchup_history(output_path: Path = TEAM_MATCHUP_CSV) -> pd.DataFrame:
    """
    Build team_matchup_history.csv — one row per (team, opponent) pair this season.
    """
    games   = _safe_read(GAMES_CSV)
    metrics = _safe_read(TEAM_METRICS_CSV)

    if games.empty and metrics.empty:
        log.info("No games/metrics data — skipping team_matchup_history")
        return pd.DataFrame()

    # Build home/away pairs from games.csv
    if not games.empty and all(c in games.columns for c in ["home_team_id", "away_team_id", "event_id"]):
        pairs = games[["event_id", "home_team_id", "away_team_id"]].copy()
        pairs = pairs.rename(columns={"home_team_id": "team_id_a", "away_team_id": "team_id_b"})
    else:
        log.info("build_team_matchup_history: insufficient games columns — skipping")
        return pd.DataFrame()

    if metrics.empty:
        return pd.DataFrame()

    # metrics columns we want per team-game
    metric_cols = [c for c in ["win", "margin", "ortg", "drtg", "pace", "cover", "game_datetime_utc"]
                   if c in metrics.columns]

    for col in ["win", "margin", "ortg", "drtg", "pace", "cover"]:
        if col in metrics.columns:
            metrics[col] = pd.to_numeric(metrics[col], errors="coerce")

    rows = []
    for _, pair in pairs.iterrows():
        eid = pair["event_id"]
        ta, tb = pair["team_id_a"], pair["team_id_b"]
        for team_id, opp_id in [(ta, tb), (tb, ta)]:
            team_games = metrics[(metrics["team_id"] == team_id) & (metrics["event_id"] == eid)]
            if team_games.empty:
                continue
            row: dict = {"team_id": team_id, "opponent_id": opp_id}
            for col in metric_cols:
                row[col] = team_games[col].iloc[0] if col in team_games.columns else None
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    detail = pd.DataFrame(rows)
    grp_cols = ["team_id", "opponent_id"]
    agg_map: dict = {}
    if "win" in detail.columns:
        agg_map["win"] = "sum"
    if "margin" in detail.columns:
        agg_map["margin"] = "mean"
    for col in ["ortg", "drtg", "pace", "cover"]:
        if col in detail.columns:
            agg_map[col] = "mean"
    if "game_datetime_utc" in detail.columns:
        agg_map["game_datetime_utc"] = ["count", "max"]

    if not agg_map:
        return pd.DataFrame()

    summary = detail.groupby(grp_cols).agg(agg_map)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns]
    summary = summary.reset_index()

    renames = {
        "win_sum": "h2h_wins", "margin_mean": "h2h_avg_margin",
        "ortg_mean": "h2h_avg_ortg", "drtg_mean": "h2h_avg_drtg",
        "pace_mean": "h2h_avg_pace", "cover_mean": "h2h_cover_rate",
        "game_datetime_utc_count": "meetings_this_season",
        "game_datetime_utc_max": "last_meeting_date",
    }
    summary = summary.rename(columns={k: v for k, v in renames.items() if k in summary.columns})

    if "h2h_wins" in summary.columns and "meetings_this_season" in summary.columns:
        summary["h2h_losses"] = summary["meetings_this_season"] - summary["h2h_wins"].fillna(0).astype(int)

    summary["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    log.info(f"team_matchup_history.csv → {output_path}  ({len(summary):,} rows)")
    return summary


def build_conference_summary(output_path: Path = CONF_SUMMARY_CSV) -> pd.DataFrame:
    """
    Build/append conference_daily_summary.csv — append-only, one batch per run.
    """
    weighted = _safe_read(TEAM_WEIGHTED_CSV)
    sos      = _safe_read(TEAM_SOS_CSV)

    if weighted.empty:
        log.info("No team_game_weighted data — skipping conference_summary")
        return pd.DataFrame()

    # Get latest row per team (most recent game) from weighted
    if "game_datetime_utc" in weighted.columns:
        weighted["game_datetime_utc"] = pd.to_datetime(weighted["game_datetime_utc"], utc=True, errors="coerce")
        latest = (weighted.sort_values("game_datetime_utc", na_position="last")
                  .groupby("team_id", as_index=False)
                  .last())
    else:
        latest = weighted.groupby("team_id", as_index=False).last()

    if not sos.empty and "opp_avg_net_rtg_season" in sos.columns and "team_id" in sos.columns:
        sos_latest = (sos.sort_values("game_datetime_utc", na_position="last")
                      if "game_datetime_utc" in sos.columns else sos)
        sos_snap = sos_latest.groupby("team_id")["opp_avg_net_rtg_season"].last().reset_index()
        latest = latest.merge(sos_snap, on="team_id", how="left", suffixes=("", "_sos"))
        if "opp_avg_net_rtg_season_sos" in latest.columns and "opp_avg_net_rtg_season" not in latest.columns:
            latest = latest.rename(columns={"opp_avg_net_rtg_season_sos": "opp_avg_net_rtg_season"})

    if "conference" not in latest.columns:
        log.warning("build_conference_summary: 'conference' column missing in weighted CSV")
        return pd.DataFrame()

    for col in ["net_rtg", "adj_ortg", "adj_drtg", "opp_avg_net_rtg_season"]:
        if col in latest.columns:
            latest[col] = pd.to_numeric(latest[col], errors="coerce")

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for conf, grp in latest.groupby("conference"):
        record: dict = {"conference": conf, "snapshot_at": now_str}
        record["n_teams"] = len(grp)
        for col, key in [("net_rtg", "avg_net_rtg"), ("adj_ortg", "avg_adj_ortg"),
                         ("adj_drtg", "avg_adj_drtg"), ("opp_avg_net_rtg_season", "avg_sos")]:
            if col in grp.columns:
                record[key] = round(float(grp[col].mean()), 3)
            else:
                record[key] = None
        team_col = "team" if "team" in grp.columns else None
        net_col  = "net_rtg" if "net_rtg" in grp.columns else None
        if team_col and net_col:
            record["best_net_rtg_team"]  = grp.loc[grp[net_col].idxmax(), team_col] if grp[net_col].notna().any() else None
            record["worst_net_rtg_team"] = grp.loc[grp[net_col].idxmin(), team_col] if grp[net_col].notna().any() else None
        else:
            record["best_net_rtg_team"]  = None
            record["worst_net_rtg_team"] = None
        rows.append(record)

    new_df = pd.DataFrame(rows)

    # Append-only: read existing rows and append
    if output_path.exists() and output_path.stat().st_size > 0:
        try:
            existing = pd.read_csv(output_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception as exc:
            log.warning(f"Could not read existing conference_daily_summary.csv: {exc}")
            combined = new_df
    else:
        combined = new_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    log.info(f"conference_daily_summary.csv → {output_path}  ({len(new_df):,} new rows, "
             f"{len(combined):,} total rows)")
    return new_df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    build_player_season_summary()
    build_team_season_summary()
    build_team_ats_profile()
    build_team_luck_regression()
    build_team_situational()
    build_team_resume()
    build_team_matchup_history()
    build_conference_summary()
    log.info("Season summaries complete.")


if __name__ == "__main__":
    main()
