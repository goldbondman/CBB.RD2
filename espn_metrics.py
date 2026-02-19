"""
ESPN CBB Pipeline — Advanced Metrics
Per-game stats and rolling team averages from box score data.
All functions pure (DataFrame in, DataFrame out). No I/O.

Per-game metrics:
  efg_pct, ts_pct, fg_pct, three_pct, ft_pct
  three_par, ftr
  orb_pct, drb_pct, tov_pct
  poss, ortg, drtg, net_rtg, pace
  h1_margin, h2_margin, h1_ortg, h2_ortg (half splits)
  pythagorean_win_pct, luck_score
  win, cover, cover_margin, ats_result
  blowout_flag, close_game_flag, garbage_margin

Rolling windows (L5, L10) — all leak-free via shift(1):
  All per-game metrics above
  points_for, points_against, margin
  Shooting variance: efg_std_l10, three_pct_std_l10
  Schedule: rest_days, games_l7, games_l14
  Streaks: win_streak, lose_streak, cover_streak
  Home/away splits: ha_ortg_l10, ha_drtg_l10, ha_net_rtg_l10
  ATS rolling: cover_rate_l10, ats_margin_l10
  Close game: close_win_pct_season
"""

import logging
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
REGULATION_MINUTES  = 40.0
LEAGUE_AVG_PACE     = 70.0
PYTHAGOREAN_EXP     = 11.5   # standard CBB exponent
BLOWOUT_THRESHOLD   = 15     # margin >= this = blowout
CLOSE_GAME_THRESHOLD = 5     # margin <= this = close game
CAP_MARGIN          = 15     # cap margin for Pythagorean/rolling to reduce garbage time noise
ROLLING_WINDOWS     = [5, 10]

ROLLING_METRICS = [
    "points_for", "points_against", "margin", "margin_capped",
    "efg_pct", "ts_pct", "three_par", "ftr",
    "fg_pct", "three_pct", "ft_pct",
    "orb_pct", "drb_pct", "tov_pct",
    "ortg", "drtg", "net_rtg", "poss", "pace",
    "h1_pts", "h2_pts", "h1_pts_against", "h2_pts_against",
    "h1_margin", "h2_margin",
    "win", "cover", "cover_margin",
    "stl", "blk", "ast",
]


# ── Safe math ─────────────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return (pd.to_numeric(df[name], errors="coerce")
            if name in df.columns
            else pd.Series(np.nan, index=df.index))


# ── Per-game metrics ──────────────────────────────────────────────────────────

def add_per_game_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    fgm = _col(df, "fgm"); fga = _col(df, "fga")
    tpm = _col(df, "tpm"); tpa = _col(df, "tpa")
    ftm = _col(df, "ftm"); fta = _col(df, "fta")
    orb = _col(df, "orb"); drb = _col(df, "drb"); reb = _col(df, "reb")
    tov = _col(df, "tov")
    pts     = _col(df, "points_for")
    opp_pts = _col(df, "points_against")
    spread  = _col(df, "spread")
    h1_pts  = _col(df, "h1_pts")
    h2_pts  = _col(df, "h2_pts")
    h1_opp  = _col(df, "h1_pts_against")
    h2_opp  = _col(df, "h2_pts_against")

    fga_s = fga.replace(0, np.nan)

    # ── Shooting ──
    df["efg_pct"]    = ((fgm + 0.5 * tpm) / fga_s * 100).round(1)
    df["ts_pct"]     = (pts / (2 * (fga + 0.44 * fta)).replace(0, np.nan) * 100).round(1)
    df["fg_pct"]     = (fgm / fga_s * 100).round(1)
    df["three_pct"]  = (tpm / tpa.replace(0, np.nan) * 100).round(1)
    df["ft_pct"]     = (ftm / fta.replace(0, np.nan) * 100).round(1)
    df["three_par"]  = (tpa / fga_s * 100).round(1)
    df["ftr"]        = (fta / fga_s * 100).round(1)

    # ── Rebounding ──
    orb_c = orb.fillna(reb - drb)
    drb_c = drb.fillna(reb - orb)
    df["orb_pct"] = (orb_c / reb.replace(0, np.nan) * 100).round(1)
    df["drb_pct"] = (drb_c / reb.replace(0, np.nan) * 100).round(1)

    # ── Possessions ──
    poss = fga - orb_c + tov + 0.44 * fta
    df["poss"] = poss.round(1)

    # ── Efficiency ──
    poss_s = poss.replace(0, np.nan)
    df["ortg"]    = (pts    / poss_s * 100).round(1)
    df["drtg"]    = (opp_pts / poss_s * 100).round(1)
    df["net_rtg"] = (df["ortg"] - df["drtg"]).round(1)
    df["tov_pct"] = (tov / poss_s * 100).round(1)

    # ── Pace ──
    df["pace"] = (poss / REGULATION_MINUTES * 40).round(1)
    # Note: adj_pace computed in espn_sos.py after opponent join

    # ── Half splits ──
    df["h1_margin"] = (h1_pts - h1_opp).round(1)
    df["h2_margin"] = (h2_pts - h2_opp).round(1)
    # Half ORTG (rough — uses team poss estimate, halved)
    half_poss = poss_s / 2
    df["h1_ortg"] = (h1_pts  / half_poss * 100).round(1)
    df["h2_ortg"] = (h2_pts  / half_poss * 100).round(1)
    df["h1_drtg"] = (h1_opp / half_poss * 100).round(1)
    df["h2_drtg"] = (h2_opp / half_poss * 100).round(1)

    # ── Game outcome flags ──
    margin = _col(df, "margin")
    df["win"]             = (margin > 0).astype(float).where(margin.notna())
    df["close_game_flag"] = (margin.abs() <= CLOSE_GAME_THRESHOLD).astype(float).where(margin.notna())
    df["blowout_flag"]    = (margin.abs() >= BLOWOUT_THRESHOLD).astype(float).where(margin.notna())

    # Margin capped — reduces blowout inflation in rolling metrics
    df["margin_capped"] = margin.clip(-CAP_MARGIN, CAP_MARGIN)

    # ── ATS (vs spread) ──
    # Positive spread = team is favorite (spread given as negative for favorite)
    # cover = team margin > -spread (i.e. margin + spread > 0)
    # ESPN spread convention: negative = home favorite (e.g. -5.5 means home -5.5)
    home_away = df.get("home_away", pd.Series("", index=df.index))
    team_spread = spread.where(home_away == "home", -spread)  # flip for away team
    df["cover_margin"] = (margin + team_spread).round(1)
    df["cover"]        = (df["cover_margin"] > 0).astype(float).where(
                          margin.notna() & team_spread.notna())
    df["ats_push"]     = (df["cover_margin"] == 0).astype(float).where(
                          margin.notna() & team_spread.notna())

    # ── Pythagorean win % ──
    exp = PYTHAGOREAN_EXP
    pts_pow     = pts    ** exp
    opp_pts_pow = opp_pts ** exp
    denom = (pts_pow + opp_pts_pow).replace(0, np.nan)
    df["pythagorean_win_pct"] = (pts_pow / denom).round(3)

    return df


def add_luck_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Luck score = actual win% - Pythagorean win%.
    Positive = team is over-performing their scoring margin (due to regress).
    Negative = team is under-performing (due to improve).
    Computed as season-to-date using expanding mean, leak-free.
    """
    if "win" not in df.columns or "pythagorean_win_pct" not in df.columns:
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["team_id", "_sort_dt"])

    df["actual_win_pct_season"] = df.groupby("team_id")["win"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean().round(3)
    )
    df["pyth_win_pct_season"] = df.groupby("team_id")["pythagorean_win_pct"].transform(
        lambda s: s.shift(1).expanding(min_periods=3).mean().round(3)
    )
    df["luck_score"] = (df["actual_win_pct_season"] - df["pyth_win_pct_season"]).round(3)

    df = df.drop(columns=["_sort_dt"], errors="ignore")
    return df


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rest days, schedule density, win/loss/cover streaks.
    All computed per team, leak-free.
    """
    if "team_id" not in df.columns or "game_datetime_utc" not in df.columns:
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["team_id", "_sort_dt"])

    # ── Rest days ──
    df["rest_days"] = (
        df.groupby("team_id")["_sort_dt"]
        .transform(lambda s: s.diff().dt.days)
    ).fillna(3).clip(0, 21)   # cap at 21, fill first game with 3 (neutral)

    # ── Games played in last 7 / 14 days ──
    def _games_in_window(group: pd.Series, days: int) -> pd.Series:
        result = []
        dates = group.values
        for i, dt in enumerate(dates):
            cutoff = dt - pd.Timedelta(days=days)
            # Count games strictly before this game within the window
            count = sum(1 for d in dates[:i] if d >= cutoff)
            result.append(count)
        return pd.Series(result, index=group.index)

    df["games_l7"]  = df.groupby("team_id")["_sort_dt"].transform(
        lambda s: _games_in_window(s, 7))
    df["games_l14"] = df.groupby("team_id")["_sort_dt"].transform(
        lambda s: _games_in_window(s, 14))

    # ── Win/loss streak ──
    def _streak(series: pd.Series) -> pd.Series:
        """Current streak length (+N = win streak, -N = loss streak) using prior games."""
        result = []
        streak = 0
        for val in series.shift(1):
            if pd.isna(val):
                result.append(0)
                continue
            w = int(val)
            if streak == 0:
                streak = 1 if w else -1
            elif w and streak > 0:
                streak += 1
            elif not w and streak < 0:
                streak -= 1
            else:
                streak = 1 if w else -1
            result.append(streak)
        return pd.Series(result, index=series.index)

    if "win" in df.columns:
        df["win_streak"] = df.groupby("team_id")["win"].transform(_streak)

    if "cover" in df.columns:
        df["cover_streak"] = df.groupby("team_id")["cover"].transform(_streak)

    df = df.drop(columns=["_sort_dt"], errors="ignore")
    return df


def add_rolling_metrics(df: pd.DataFrame,
                        windows: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Rolling averages per metric in ROLLING_METRICS, grouped by team_id.
    Uses shift(1) for leak-free pregame features.
    """
    if "team_id" not in df.columns or "game_datetime_utc" not in df.columns:
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["team_id", "_sort_dt"])

    present = [m for m in ROLLING_METRICS if m in df.columns]

    for w in windows:
        for m in present:
            df[f"{m}_l{w}"] = df.groupby("team_id")[m].transform(
                lambda s, ww=w: s.shift(1).rolling(ww, min_periods=1).mean().round(2)
            )

    # ── Shooting variance (10-game rolling std) ──
    for metric, col_name in [("efg_pct", "efg_std_l10"),
                              ("three_pct", "three_pct_std_l10"),
                              ("net_rtg", "net_rtg_std_l10")]:
        if metric in df.columns:
            df[col_name] = df.groupby("team_id")[metric].transform(
                lambda s: s.shift(1).rolling(10, min_periods=3).std().round(2)
            )

    # ── ATS rolling rates ──
    if "cover" in df.columns:
        df["cover_rate_l10"] = df.groupby("team_id")["cover"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean().round(3)
        )
        df["cover_rate_season"] = df.groupby("team_id")["cover"].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean().round(3)
        )

    if "cover_margin" in df.columns:
        df["ats_margin_l10"] = df.groupby("team_id")["cover_margin"].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean().round(2)
        )

    # ── Close game win % ──
    if "win" in df.columns and "close_game_flag" in df.columns:
        # Win % in close games only — season rolling
        close_wins = df["win"].where(df["close_game_flag"] == 1)
        df["close_win_pct_season"] = df.groupby("team_id")[df.columns[
            df.columns.get_loc("win")]].transform(
            lambda s: s.shift(1).expanding(min_periods=2).mean().round(3)
        )
        # Simpler approach
        df["close_game_win_pct"] = df.groupby("team_id").apply(
            lambda g: g["win"].where(g["close_game_flag"] == 1)
            .shift(1).expanding(min_periods=2).mean()
        ).reset_index(level=0, drop=True).round(3)

    df = df.drop(columns=["_sort_dt"], errors="ignore")
    return df


def add_home_away_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Separate rolling efficiency metrics split by home vs away.
    Gives context on how a team performs at home vs on the road.
    """
    if "home_away" not in df.columns:
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["team_id", "_sort_dt"])

    ha_metrics = ["ortg", "drtg", "net_rtg", "efg_pct", "tov_pct", "pace"]
    present    = [m for m in ha_metrics if m in df.columns]

    for m in present:
        df[f"ha_{m}_l10"] = df.groupby(["team_id", "home_away"])[m].transform(
            lambda s: s.shift(1).rolling(10, min_periods=3).mean().round(2)
        )

    # Home/away net rating differential — how much better/worse on road
    if "net_rtg" in df.columns:
        home_rtg = df[df["home_away"] == "home"].groupby("team_id")["net_rtg"].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean()
        )
        away_rtg = df[df["home_away"] == "away"].groupby("team_id")["net_rtg"].transform(
            lambda s: s.shift(1).expanding(min_periods=3).mean()
        )
        df.loc[df["home_away"] == "home", "home_net_rtg_season"] = home_rtg
        df.loc[df["home_away"] == "away", "away_net_rtg_season"] = away_rtg

    df = df.drop(columns=["_sort_dt"], errors="ignore")
    return df


def add_true_rebound_pcts(df: pd.DataFrame) -> pd.DataFrame:
    """True ORB%/DRB% using opponent boards. Only runs if opp columns present."""
    if not all(c in df.columns for c in ["opp_orb", "opp_drb"]):
        return df
    df = df.copy()
    orb = pd.to_numeric(df.get("orb", np.nan), errors="coerce")
    drb = pd.to_numeric(df.get("drb", np.nan), errors="coerce")
    opp_orb = pd.to_numeric(df["opp_orb"], errors="coerce")
    opp_drb = pd.to_numeric(df["opp_drb"], errors="coerce")
    df["true_orb_pct"] = (orb / (orb + opp_drb).replace(0, np.nan) * 100).round(1)
    df["true_drb_pct"] = (drb / (drb + opp_orb).replace(0, np.nan) * 100).round(1)
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def derive_records(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive cumulative wins/losses/record from game results.

    ESPN's summary API competitor objects do NOT carry a records[] field —
    that only exists on the scoreboard competitor. Rather than patching the
    parser, we compute all record columns here from the margin/win column
    that is already in the dataset.

    Columns added (all reflect the team's record ENTERING this game — leak-free):
      wins, losses, record              overall (e.g. "15-8")
      home_wins, home_losses, home_record
      away_wins, away_losses, away_record
      conf_wins, conf_losses, conf_record
      conf_rank                         rank within conference by win% entering game
      win_pct, home_win_pct, away_win_pct, conf_win_pct
    """
    if df.empty:
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df.get("game_datetime_utc", ""), utc=True,
                                     errors="coerce")
    df = df.sort_values(["team_id", "_sort_dt"])

    win_col = "win"
    if win_col not in df.columns:
        log.warning("derive_records: 'win' column missing — skipping record derivation")
        df = df.drop(columns=["_sort_dt"], errors="ignore")
        return df

    win = pd.to_numeric(df[win_col], errors="coerce").fillna(0)
    loss = 1 - win  # completed games only; NaN margin → win=NaN → treat as 0

    # Only count completed games (margin is not NaN)
    completed = pd.to_numeric(df.get("margin", pd.Series(np.nan, index=df.index)),
                               errors="coerce").notna().astype(float)
    win_c  = win  * completed
    loss_c = loss * completed

    # ── Overall record ────────────────────────────────────────────────────────
    df["wins"]   = df.groupby("team_id")[win_col].transform(
        lambda s: pd.to_numeric(s, errors="coerce").fillna(0)
                    .mul(completed.reindex(s.index, fill_value=0))
                    .shift(1).expanding().sum().fillna(0).astype(int)
    )
    df["losses"] = df.groupby("team_id")["win"].transform(
        lambda s: (1 - pd.to_numeric(s, errors="coerce").fillna(0))
                    .mul(completed.reindex(s.index, fill_value=0))
                    .shift(1).expanding().sum().fillna(0).astype(int)
    )
    df["win_pct"] = (
        df["wins"] / (df["wins"] + df["losses"]).replace(0, np.nan)
    ).round(3)
    df["record"] = df["wins"].astype(int).astype(str) + "-" + df["losses"].astype(int).astype(str)

    # ── Home / away splits ─────────────────────────────────────────────────────
    home_away = df.get("home_away", pd.Series("", index=df.index)).fillna("")

    for side, label in [("home", "home"), ("away", "away")]:
        mask = (home_away == side).astype(float)
        w_side = win_c  * mask
        l_side = loss_c * mask

        df[f"{label}_wins"] = df.groupby("team_id")["win"].transform(
            lambda s, m=mask: (
                pd.to_numeric(s, errors="coerce").fillna(0)
                .mul(completed.reindex(s.index, fill_value=0))
                .mul(m.reindex(s.index, fill_value=0))
                .shift(1).expanding().sum().fillna(0).astype(int)
            )
        )
        df[f"{label}_losses"] = df.groupby("team_id")["win"].transform(
            lambda s, m=mask: (
                (1 - pd.to_numeric(s, errors="coerce").fillna(0))
                .mul(completed.reindex(s.index, fill_value=0))
                .mul(m.reindex(s.index, fill_value=0))
                .shift(1).expanding().sum().fillna(0).astype(int)
            )
        )
        df[f"{label}_win_pct"] = (
            df[f"{label}_wins"] /
            (df[f"{label}_wins"] + df[f"{label}_losses"]).replace(0, np.nan)
        ).round(3)
        df[f"{label}_record"] = (
            df[f"{label}_wins"].astype(int).astype(str) + "-" +
            df[f"{label}_losses"].astype(int).astype(str)
        )

    # ── Conference record ─────────────────────────────────────────────────────
    # opp_conference must match team's conference for conf game
    opp_conf = df.get("opp_conference", pd.Series("", index=df.index)).fillna("")
    team_conf = df.get("conference", pd.Series("", index=df.index)).fillna("")
    # A game is a conf game if both teams share a non-empty conference
    conf_game = (
        (team_conf != "") & (opp_conf != "") & (team_conf == opp_conf)
    ).astype(float)

    df["conf_wins"] = df.groupby("team_id")["win"].transform(
        lambda s: (
            pd.to_numeric(s, errors="coerce").fillna(0)
            .mul(completed.reindex(s.index, fill_value=0))
            .mul(conf_game.reindex(s.index, fill_value=0))
            .shift(1).expanding().sum().fillna(0).astype(int)
        )
    )
    df["conf_losses"] = df.groupby("team_id")["win"].transform(
        lambda s: (
            (1 - pd.to_numeric(s, errors="coerce").fillna(0))
            .mul(completed.reindex(s.index, fill_value=0))
            .mul(conf_game.reindex(s.index, fill_value=0))
            .shift(1).expanding().sum().fillna(0).astype(int)
        )
    )
    df["conf_win_pct"] = (
        df["conf_wins"] /
        (df["conf_wins"] + df["conf_losses"]).replace(0, np.nan)
    ).round(3)
    df["conf_record"] = (
        df["conf_wins"].astype(int).astype(str) + "-" +
        df["conf_losses"].astype(int).astype(str)
    )

    # ── Conference rank (within conference, by win_pct at time of game) ───────
    if "conference" in df.columns:
        # Rank each team within its conference based on overall win_pct entering game
        # Ties broken by wins (more wins = better rank)
        df["conf_rank"] = (
            df.groupby(["conference", "_sort_dt"])["win_pct"]
            .rank(method="min", ascending=False, na_option="bottom")
            .astype("Int64")
        )
    else:
        df["conf_rank"] = pd.NA

    df = df.drop(columns=["_sort_dt"], errors="ignore")
    log.info("derive_records: computed wins/losses/record/conf_rank for all teams")
    return df


def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full metrics pipeline. Call from espn_pipeline.py after team logs are written.
    Input: team_game_logs DataFrame (full historical, not just current run).
    Output: same DataFrame with all derived columns appended.
    """
    log.info(f"Computing metrics for {len(df)} team-game rows")
    df = add_per_game_metrics(df)
    df = derive_records(df)          # wins/losses/record/conf_rank from game results
    df = add_luck_score(df)
    df = add_schedule_features(df)
    df = add_rolling_metrics(df, windows=ROLLING_WINDOWS)
    df = add_home_away_splits(df)
    df = add_true_rebound_pcts(df)
    log.info("Metrics complete")
    return df
