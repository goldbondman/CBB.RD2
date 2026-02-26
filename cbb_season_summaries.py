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

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from config.logging_config import get_logger
from espn_config import conference_id_to_name

log = get_logger(__name__)

DATA_DIR = Path("data")
CSV_DATA_DIR = DATA_DIR / "csv"

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


def _safe_read_with_fallback(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV from data/, then fallback to data/csv/ if needed."""
    df = _safe_read(path, **kwargs)
    if not df.empty:
        return df
    fallback = CSV_DATA_DIR / path.name
    if fallback == path:
        return df
    return _safe_read(fallback, **kwargs)


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
        ts_df["ts_pct"] = pd.to_numeric((ts_df["pts"] /
                           (2 * (ts_df["fga"] + 0.44 * ts_df["fta"]))), errors="coerce").round(4)
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
        summary["season_pts_share"] = pd.to_numeric((
            summary["_player_pts"] / summary["_team_pts"].replace(0, pd.NA)
        ), errors="coerce").round(4)
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
    weighted = _safe_read_with_fallback(TEAM_WEIGHTED_CSV, low_memory=False)
    results = _safe_read_with_fallback(DATA_DIR / "results_log_graded.csv", low_memory=False)
    if results.empty:
        results = _safe_read_with_fallback(RESULTS_LOG_CSV, low_memory=False)

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
        "avg_clv", "clv_positive_rate", "avg_clv_edge_games", "clv_sample_size", "clv_grade",
        "updated_at",
    ]

    if weighted.empty:
        log.info("No team_game_weighted data — writing empty team_season_summary.csv")
        empty = pd.DataFrame(columns=empty_cols)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        empty.to_csv(output_path, index=False)
        return empty

    # ── Build records/averages from weighted CSV ───────────────────────────────
    tid_col = "team_id" if "team_id" in weighted.columns else None
    team_col = "team" if "team" in weighted.columns else None
    conf_col = "conference" if "conference" in weighted.columns else None
    ha_col = "home_away" if "home_away" in weighted.columns else None

    if tid_col is None and team_col is None:
        log.warning("team_game_weighted missing team_id/team columns — skipping team summary")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(output_path, index=False)
        return pd.DataFrame()

    group_keys = [c for c in [tid_col, team_col] if c]
    df = weighted.copy()

    for col in ["win", "margin", "ortg", "drtg", "pace", "neutral_site", "cover", "event_id", "over_under", "points_for", "completed"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "win" not in df.columns:
        log.warning("team_game_weighted missing 'win' column; cannot build team records")
        df["win"] = pd.NA

    # Restrict aggregates to played games; upcoming fixtures should not poison means/records
    played = df.copy()
    if "completed" in played.columns:
        played["completed"] = played["completed"].astype(str).str.lower().map({"true": 1, "false": 0})
        played = played[played["completed"] == 1]
    elif "win" in played.columns:
        played = played[played["win"].notna()]

    # Games played
    gp = played.groupby(group_keys).size().reset_index(name="_gp")

    summary = gp.copy()

    wl = played.groupby(group_keys).agg(
        wins=("win", lambda x: (x == 1).sum()),
        losses=("win", lambda x: (x == 0).sum()),
    ).reset_index()
    summary = summary.merge(wl, on=group_keys, how="left")

    # ── Home / away / neutral splits ──────────────────────────────────────────
    for venue, label in [("home", "home"), ("away", "away"), ("neutral", "neutral")]:
        if ha_col:
            sub = played[played[ha_col] == venue]
        else:
            sub = pd.DataFrame()

        if not sub.empty and "win" in sub.columns:
            vw = sub.groupby(group_keys).agg(
                **{
                    f"{label}_wins": ("win", lambda x: (x == 1).sum()),
                    f"{label}_losses": ("win", lambda x: (x == 0).sum()),
                }
            ).reset_index()
            summary = summary.merge(
                vw[group_keys + [f"{label}_wins", f"{label}_losses"]],
                on=group_keys, how="left",
            )
        else:
            summary[f"{label}_wins"]   = None
            summary[f"{label}_losses"] = None

    # ── Rating columns ─────────────────────────────────────────────────────────
    mean_cols = [c for c in ["margin", "ortg", "drtg", "pace"] if c in played.columns]
    if mean_cols:
        means = played.groupby(group_keys)[mean_cols].mean().reset_index()
        summary = summary.merge(means, on=group_keys, how="left")
    col_map = {"margin": "avg_margin", "ortg": "avg_ortg", "drtg": "avg_drtg", "pace": "avg_pace"}
    for src, dst in col_map.items():
        summary[dst] = summary[src].round(3) if src in summary.columns else None

    for c in ["home_wins", "home_losses", "away_wins", "away_losses", "neutral_wins", "neutral_losses"]:
        if c in summary.columns:
            summary[c] = summary[c].fillna(0).astype(int)

    # conference column
    if conf_col:
        conf_df = df.groupby(group_keys)[conf_col].first().reset_index()
        summary = summary.merge(conf_df, on=group_keys, how="left")
        summary = summary.rename(columns={conf_col: "conference"})
    else:
        summary["conference"] = None

    # ── ATS / O-U from results_log ────────────────────────────────────────────
    # Use a normalized team merge key so summary/results dtypes don't block joins.
    summary["_team_merge_key"] = summary["team_id"].astype(str).str.strip()

    ats_source = results if (not results.empty and "team_id" in results.columns) else played
    if not ats_source.empty and "team_id" in ats_source.columns:
        ats_col = "cover" if "cover" in ats_source.columns else ("ats_correct" if "ats_correct" in ats_source.columns else None)
        ou_col = "ou_correct" if "ou_correct" in ats_source.columns else None
        err_col = next((c for c in ["spread_error", "absolute_error", "pred_error", "primary_margin_error"] if c in ats_source.columns), None)

        ats_source = ats_source.copy()
        ats_source["_team_merge_key"] = ats_source["team_id"].astype(str).str.strip()

        if ats_col:
            ats_source[ats_col] = pd.to_numeric(ats_source[ats_col], errors="coerce")
            ats = ats_source[ats_source[ats_col].notna()].groupby("_team_merge_key").agg(
                ats_wins=(ats_col, lambda x: (x == 1).sum()),
                ats_losses=(ats_col, lambda x: (x == 0).sum()),
            ).reset_index()
            summary = summary.merge(ats, on="_team_merge_key", how="left")

        if ou_col:
            ats_source[ou_col] = pd.to_numeric(ats_source[ou_col], errors="coerce")
            ou = ats_source[ats_source[ou_col].notna()].groupby("_team_merge_key").agg(
                ou_over=(ou_col, lambda x: (x == 1).sum()),
                ou_under=(ou_col, lambda x: (x == 0).sum()),
            ).reset_index()
            summary = summary.merge(ou, on="_team_merge_key", how="left")
        elif {"event_id", "over_under", "points_for"}.issubset(played.columns):
            event_totals = played.groupby("event_id").agg(total_pts=("points_for", "sum"), ou_line=("over_under", "first")).reset_index()
            event_totals["went_over"] = (event_totals["total_pts"] > event_totals["ou_line"]).astype(float)
            event_teams = played[["event_id", "team_id"]].copy()
            event_teams["_team_merge_key"] = event_teams["team_id"].astype(str).str.strip()
            event_teams = event_teams.merge(event_totals[["event_id", "went_over"]], on="event_id", how="left")
            ou = event_teams[event_teams["went_over"].notna()].groupby("_team_merge_key").agg(
                ou_over=("went_over", lambda x: (x == 1).sum()),
                ou_under=("went_over", lambda x: (x == 0).sum()),
            ).reset_index()
            summary = summary.merge(ou, on="_team_merge_key", how="left")

        if err_col:
            ats_source[err_col] = pd.to_numeric(ats_source[err_col], errors="coerce")
            err = ats_source.groupby("_team_merge_key")[err_col].mean().reset_index().rename(columns={err_col: "avg_pred_spread_error"})
            summary = summary.merge(err, on="_team_merge_key", how="left")

    # CLV metrics
    pred_col = next((c for c in ["pred_spread", "ensemble_spread", "prediction", "model_spread"] if c in results.columns), None)
    close_col = next((c for c in ["spread", "closing_spread", "closing_line", "vegas_line", "spread_line", "closing_spread_line"] if c in results.columns), None)
    edge_col = next((c for c in ["edge_flag", "is_alpha"] if c in results.columns), None)
    if pred_col and close_col and "team_id" in results.columns:
        clv_df = results[["team_id", pred_col, close_col] + ([edge_col] if edge_col else [])].copy()
        clv_df["_team_merge_key"] = clv_df["team_id"].astype(str).str.strip()
        clv_df[pred_col] = pd.to_numeric(clv_df[pred_col], errors="coerce")
        clv_df[close_col] = pd.to_numeric(clv_df[close_col], errors="coerce")
        clv_df["clv"] = clv_df[pred_col] - clv_df[close_col]
        clv_df = clv_df[clv_df["clv"].notna()]
        if not clv_df.empty:
            clv_agg = clv_df.groupby("_team_merge_key").agg(
                avg_clv=("clv", "mean"),
                clv_positive_rate=("clv", lambda x: (x > 0).mean()),
                clv_sample_size=("clv", "count"),
            ).reset_index()
            if edge_col:
                edge_games = clv_df[pd.to_numeric(clv_df[edge_col], errors="coerce") == 1]
                edge_agg = edge_games.groupby("_team_merge_key")["clv"].mean().reset_index().rename(columns={"clv": "avg_clv_edge_games"})
                clv_agg = clv_agg.merge(edge_agg, on="_team_merge_key", how="left")
            else:
                clv_agg["avg_clv_edge_games"] = None

            def _clv_grade(row: pd.Series) -> str:
                if row["clv_sample_size"] < 10:
                    return "INSUFFICIENT_SAMPLE"
                if row["avg_clv"] > 2 and row["clv_positive_rate"] > 0.58:
                    return "A"
                if row["avg_clv"] > 1 and row["clv_positive_rate"] > 0.54:
                    return "B"
                if row["avg_clv"] > 0 and row["clv_positive_rate"] > 0.50:
                    return "C"
                if row["avg_clv"] > -1:
                    return "D"
                return "F"

            clv_agg["clv_grade"] = clv_agg.apply(_clv_grade, axis=1)
            summary = summary.merge(clv_agg, on="_team_merge_key", how="left")

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
        "avg_clv", "clv_positive_rate", "avg_clv_edge_games", "clv_sample_size", "clv_grade",
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
    df = _safe_read_with_fallback(TEAM_METRICS_CSV)
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
    unmapped_conf_ids: set[str] = set()
    for keys, grp in df.groupby(id_cols):
        grp = grp.sort_values("game_datetime_utc", na_position="last") if "game_datetime_utc" in grp.columns else grp
        record: dict = {}
        for k, v in zip(id_cols, keys if isinstance(keys, tuple) else (keys,)):
            record[k] = v

        cover_s = grp["cover"] if "cover" in grp.columns else pd.Series(dtype=float)
        cm_s = grp["cover_margin"] if "cover_margin" in grp.columns else pd.Series(dtype=float)
        spread_s = pd.to_numeric(grp["spread"], errors="coerce") if "spread" in grp.columns else pd.Series(dtype=float)

        cover_s = pd.to_numeric(cover_s, errors="coerce")
        cm_s = pd.to_numeric(cm_s, errors="coerce")
        has_cover_mask = cover_s.notna()
        graded_cover = cover_s[has_cover_mask]
        graded_cm = cm_s[has_cover_mask]

        record["conference_name"] = conference_id_to_name(record.get("conference"))
        if record.get("conference") not in (None, "", record["conference_name"]) and record["conference_name"] == str(record.get("conference")):
            unmapped_conf_ids.add(str(record.get("conference")))

        record["games_with_spread"] = int(spread_s.notna().sum())
        record["games_with_cover_result"] = int(graded_cover.notna().sum())

        record["cover_rate_season"] = round(float(graded_cover.mean()), 4) if graded_cover.notna().any() else None
        record["cover_rate_l10"] = round(float(graded_cover.tail(10).mean()), 4) if graded_cover.notna().any() else None
        record["cover_rate_l5"] = round(float(graded_cover.tail(5).mean()), 4) if graded_cover.notna().any() else None

        record["ats_margin_season"] = round(float(graded_cm.mean()), 4) if graded_cm.notna().any() else None
        record["ats_margin_l10"] = round(float(graded_cm.tail(10).mean()), 4) if graded_cm.notna().any() else None
        record["ats_margin_l5"] = round(float(graded_cm.tail(5).mean()), 4) if graded_cm.notna().any() else None

        record["ats_wins"] = int((graded_cover == 1).sum())
        record["ats_losses"] = int((graded_cover == 0).sum())
        if "ats_push" in grp.columns:
            ats_push = pd.to_numeric(grp["ats_push"], errors="coerce")
            record["ats_pushes"] = int(ats_push.fillna(0).astype(int).sum())
        else:
            record["ats_pushes"] = 0

        if spread_s.notna().any():
            fav_cover = cover_s[(spread_s < 0) & has_cover_mask]
            dog_cover = cover_s[(spread_s > 0) & has_cover_mask]
            record["favorite_cover_rate"] = round(float(fav_cover.mean()), 4) if fav_cover.notna().any() else None
            record["underdog_cover_rate"] = round(float(dog_cover.mean()), 4) if dog_cover.notna().any() else None
            record["games_as_favorite"] = int(fav_cover.notna().sum())
            record["games_as_underdog"] = int(dog_cover.notna().sum())
        else:
            record["favorite_cover_rate"] = None
            record["underdog_cover_rate"] = None
            record["games_as_favorite"] = 0
            record["games_as_underdog"] = 0

        if graded_cm.notna().any():
            record["avg_cover_margin_when_covered"] = round(float(graded_cm[graded_cover == 1].mean()), 4) if (graded_cover == 1).any() else None
            record["avg_loss_margin_when_failed"] = round(float(graded_cm[graded_cover == 0].mean()), 4) if (graded_cover == 0).any() else None
        else:
            record["avg_cover_margin_when_covered"] = None
            record["avg_loss_margin_when_failed"] = None

        if graded_cover.notna().sum() >= 5:
            recent_rate = graded_cover.tail(5).mean()
            season_rate = graded_cover.mean()
            record["ats_regression_risk"] = int(recent_rate > 0.70 and season_rate < 0.55)
        else:
            record["ats_regression_risk"] = None

        if "home_away" in grp.columns:
            home_cover = pd.to_numeric(grp.loc[grp["home_away"] == "home", "cover"], errors="coerce") if "cover" in grp.columns else pd.Series(dtype=float)
            away_cover = pd.to_numeric(grp.loc[grp["home_away"] == "away", "cover"], errors="coerce") if "cover" in grp.columns else pd.Series(dtype=float)
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
        "team_id", "team", "conference", "conference_name",
        "cover_rate_season", "cover_rate_l10", "cover_rate_l5",
        "ats_margin_season", "ats_margin_l10", "ats_margin_l5",
        "ats_wins", "ats_losses", "ats_pushes",
        "favorite_cover_rate", "underdog_cover_rate", "games_as_favorite", "games_as_underdog",
        "avg_cover_margin_when_covered", "avg_loss_margin_when_failed", "ats_regression_risk",
        "home_cover_rate", "away_cover_rate",
        "cover_streak", "games_with_spread", "games_with_cover_result",
        "ou_over_rate_season", "ou_over_rate_l10", "updated_at",
    ]
    summary = pd.DataFrame(rows)
    for c in out_cols:
        if c not in summary.columns:
            summary[c] = None
    summary = summary[[c for c in out_cols if c in summary.columns]]

    if unmapped_conf_ids:
        log.warning(f"build_team_ats_profile: unmapped conference ids {sorted(unmapped_conf_ids)}")

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
    all_games = _safe_read_with_fallback(TEAM_WEIGHTED_CSV)
    if all_games.empty:
        all_games = _safe_read(TEAM_METRICS_CSV)
    if all_games.empty:
        log.info("No weighted/metrics data — skipping team_situational")
        return pd.DataFrame()

    required_cols = ["event_id", "team_id", "opponent_id", "game_datetime_utc", "home_away"]
    missing = [c for c in required_cols if c not in all_games.columns]
    if missing:
        log.warning("build_team_situational: missing required columns: %s", missing)
        return pd.DataFrame()

    df = all_games.copy()
    df["team_id"] = df["team_id"].astype(str).str.strip()
    df["opponent_id"] = df["opponent_id"].astype(str).str.strip()
    df["event_id"] = df["event_id"].astype(str).str.strip()
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["game_datetime_utc", "team_id", "event_id"]).copy()

    # ── Attempt to load pre-computed metrics ──
    metrics_df = _safe_read_with_fallback(TEAM_METRICS_CSV)
    precomputed_cols = [
        "rest_days", "games_l7", "games_l14", "win_streak",
        "cover_streak", "close_win_pct_season", "close_game_win_pct"
    ]
    if not metrics_df.empty and all(c in metrics_df.columns for c in precomputed_cols):
        metrics_subset = metrics_df[["event_id", "team_id"] + precomputed_cols].copy()
        metrics_subset["event_id"] = metrics_subset["event_id"].astype(str).str.strip()
        metrics_subset["team_id"] = metrics_subset["team_id"].astype(str).str.strip()
        df = df.merge(metrics_subset, on=["event_id", "team_id"], how="left")
        log.info("Using pre-computed rolling columns from team_game_metrics.csv")
        use_precomputed = True
    else:
        log.info("Pre-computed rolling columns not found or incomplete; falling back to re-computation")
        use_precomputed = False

    # Attach spread line if weighted input is missing it.
    if "spread" not in df.columns and "spread_line" not in df.columns:
        games = _safe_read_with_fallback(GAMES_CSV)
        if not games.empty and "event_id" in games.columns:
            games["event_id"] = games["event_id"].astype(str).str.strip()
            spread_candidates = [c for c in ["spread_line", "spread"] if c in games.columns]
            if spread_candidates:
                gcols = ["event_id", *spread_candidates]
                df = df.merge(games[gcols].drop_duplicates(subset=["event_id"]), on="event_id", how="left")
    spread_col = "spread_line" if "spread_line" in df.columns else "spread" if "spread" in df.columns else None

    team_name_col = "team" if "team" in df.columns else "team_name" if "team_name" in df.columns else None
    team_map = {}
    if team_name_col:
        team_map = (df[["team_id", team_name_col]]
                    .dropna(subset=["team_id", team_name_col])
                    .drop_duplicates(subset=["team_id"])
                    .set_index("team_id")[team_name_col]
                    .to_dict())

    def _season_start(dt: pd.Timestamp) -> pd.Timestamp:
        year = dt.year if dt.month >= 8 else dt.year - 1
        return pd.Timestamp(f"{year}-08-01", tz="UTC")

    def _extract_outcome(row: pd.Series) -> int | None:
        pf = pd.to_numeric(row.get("points_for"), errors="coerce")
        pa = pd.to_numeric(row.get("points_against"), errors="coerce")
        if pd.notna(pf) and pd.notna(pa):
            return 1 if pf > pa else -1
        w = row.get("wins", row.get("win", None))
        wv = pd.to_numeric(w, errors="coerce")
        if pd.notna(wv):
            return 1 if float(wv) >= 0.5 else -1
        return None

    def _compute_streak(values: list[int]) -> int:
        if not values:
            return 0
        streak = 0
        last = values[-1]
        for v in reversed(values):
            if v == last:
                streak += v
            else:
                break
        return int(streak)

    team_histories = {
        str(tid): grp.sort_values("game_datetime_utc").reset_index(drop=True)
        for tid, grp in df.groupby("team_id", dropna=False)
    }

    rows: list[dict] = []
    spread_available = 0
    for _, row in df.sort_values(["team_id", "game_datetime_utc", "event_id"]).iterrows():
        tid = str(row["team_id"]).strip()
        game_dt = row["game_datetime_utc"]
        hist = team_histories.get(tid, pd.DataFrame())
        prior = hist[hist["game_datetime_utc"] < game_dt]

        # ── Check for pre-computed columns ──
        if use_precomputed and pd.notna(row.get("rest_days")):
            rest_days = float(row["rest_days"])
            games_l7 = int(row["games_l7"])
            games_l14 = int(row["games_l14"])
            win_streak = int(row["win_streak"])
            cover_streak = int(row["cover_streak"])
            close_win_pct_season = row["close_win_pct_season"]
            close_game_win_pct = row["close_game_win_pct"]
        else:
            # rest/games window
            if prior.empty:
                rest_days = 7.0
            else:
                last_game_dt = prior["game_datetime_utc"].iloc[-1]
                rest_days = min(round((game_dt - last_game_dt).total_seconds() / 86400, 1), 14.0)
            cutoff_7 = game_dt - pd.Timedelta(days=7)
            cutoff_14 = game_dt - pd.Timedelta(days=14)
            games_l7 = int((prior["game_datetime_utc"] >= cutoff_7).sum())
            games_l14 = int((prior["game_datetime_utc"] >= cutoff_14).sum())

            # win streak
            outcomes = [o for o in prior.apply(_extract_outcome, axis=1).tolist() if o is not None]
            win_streak = _compute_streak(outcomes)

            # cover streak
            cover_outcomes: list[int] = []
            if spread_col:
                for _, g in prior.iterrows():
                    spread_val = pd.to_numeric(g.get(spread_col), errors="coerce")
                    pf = pd.to_numeric(g.get("points_for"), errors="coerce")
                    pa = pd.to_numeric(g.get("points_against"), errors="coerce")
                    if pd.isna(spread_val) or pd.isna(pf) or pd.isna(pa):
                        continue
                    if str(g.get("home_away", "")).strip().lower() == "home":
                        covered = (pf - pa) > float(spread_val)
                    else:
                        covered = (pa - pf) > -float(spread_val)
                    cover_outcomes.append(1 if covered else -1)
                spread_available += int(prior[spread_col].notna().any())
            cover_streak = _compute_streak(cover_outcomes)

            # close game percentages
            season_prior = prior[prior["game_datetime_utc"] >= _season_start(game_dt)]
            season_margin = (pd.to_numeric(season_prior.get("points_for"), errors="coerce") -
                             pd.to_numeric(season_prior.get("points_against"), errors="coerce"))
            close_games = season_prior[season_margin.abs() <= 5]
            if len(close_games) < 3:
                close_win_pct_season = None
            else:
                close_wins = (
                    pd.to_numeric(close_games.get("points_for"), errors="coerce") >
                    pd.to_numeric(close_games.get("points_against"), errors="coerce")
                ).sum()
                close_win_pct_season = round(float(close_wins / len(close_games)), 3)

            recent_10 = prior.tail(10)
            recent_margin = (pd.to_numeric(recent_10.get("points_for"), errors="coerce") -
                             pd.to_numeric(recent_10.get("points_against"), errors="coerce"))
            close_l10 = recent_10[recent_margin.abs() <= 5]
            if len(close_l10) < 2:
                close_game_win_pct = None
            else:
                close_l10_wins = (
                    pd.to_numeric(close_l10.get("points_for"), errors="coerce") >
                    pd.to_numeric(close_l10.get("points_against"), errors="coerce")
                ).sum()
                close_game_win_pct = round(float(close_l10_wins / len(close_l10)), 3)

        recent_5 = prior.tail(5)
        if len(recent_5) < 3:
            scoring_consistency_l5 = None
        else:
            margins = (pd.to_numeric(recent_5.get("points_for"), errors="coerce") -
                       pd.to_numeric(recent_5.get("points_against"), errors="coerce")).dropna()
            scoring_consistency_l5 = round(float(margins.std(ddof=0)), 2) if len(margins) >= 3 else None

        pf_now = pd.to_numeric(row.get("points_for"), errors="coerce")
        pa_now = pd.to_numeric(row.get("points_against"), errors="coerce")
        if pd.isna(pf_now) or pd.isna(pa_now) or (float(pf_now) == 0.0 and float(pa_now) == 0.0):
            blowout_flag = None
            close_game_flag = None
            margin = None
            margin_capped = None
        else:
            margin = float(pf_now - pa_now)
            blowout_flag = 1 if abs(margin) >= 15 else 0
            close_game_flag = 1 if abs(margin) <= 5 else 0
            margin_capped = float(max(-15.0, min(15.0, margin)))

        fatigue_index = round((games_l7 * 0.6) + (games_l14 * 0.3) + max(0.0, (3.0 - rest_days) * 0.5), 2)

        rows.append({
            "event_id": str(row.get("event_id", "")).strip(),
            "team_id": tid,
            "team_name": team_map.get(tid, tid),
            "opponent_id": str(row.get("opponent_id", "")).strip(),
            "opponent_name": team_map.get(str(row.get("opponent_id", "")).strip(), str(row.get("opponent_id", "")).strip()),
            "game_datetime_utc": game_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "home_away": row.get("home_away"),
            "rest_days": rest_days,
            "games_l7": games_l7,
            "games_l14": games_l14,
            "fatigue_index": fatigue_index,
            "win_streak": win_streak,
            "cover_streak": cover_streak,
            "close_win_pct_season": close_win_pct_season,
            "close_game_win_pct": close_game_win_pct,
            "scoring_consistency_l5": scoring_consistency_l5,
            "blowout_flag": blowout_flag,
            "close_game_flag": close_game_flag,
            "margin": margin,
            "margin_capped": margin_capped,
        })

    out = pd.DataFrame(rows)
    output_cols = [
        "event_id", "team_id", "team_name", "opponent_id", "opponent_name",
        "game_datetime_utc", "home_away", "rest_days", "games_l7", "games_l14",
        "fatigue_index", "win_streak", "cover_streak", "close_win_pct_season",
        "close_game_win_pct", "scoring_consistency_l5", "blowout_flag",
        "close_game_flag", "margin", "margin_capped",
    ]
    out = out.reindex(columns=output_cols)

    if spread_col and len(df):
        spread_rate = float(df[spread_col].notna().mean())
        if spread_rate < 0.10:
            log.warning("Spread data coverage low for team_situational: %.1f%%", spread_rate * 100)
    elif not spread_col:
        log.warning("Spread data column missing; cover_streak defaults to 0")

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
