"""
ESPN CBB Pipeline: Strength of Schedule and Opponent-Adjusted Metrics
Requires team_game_metrics.csv (output of espn_metrics.py) as input.

Metrics computed:

STRENGTH OF SCHEDULE (3 windows per metric):
  opp_avg_ortg         Average ORTG of opponents faced
  opp_avg_drtg         Average DRTG of opponents faced
  opp_avg_net_rtg      Average NetRTG of opponents faced (primary SOS number)
  opp_avg_efg_pct      Average eFG% of opponents faced
  opp_avg_pace         Average pace of opponents faced

  Each computed for:
    _season   Full season (all prior games)
    _l5       Last 5 games
    _l10      Last 10 games

OPPONENT-CONTEXT METRICS (performance vs. what opponents allow):
  efg_vs_opp_allow      Team eFG% minus avg eFG% opponents allow to others
  orb_vs_opp_allow      Team ORB% minus avg ORB% opponents allow to others
  drb_vs_opp_allow      Team DRB% minus avg DRB% opponents allow to others
  reb_vs_opp_allow      Team REB% (total) vs opponent context
  tov_vs_opp_force      Team TOV% minus avg TOV% opponents force on others
  ftr_vs_opp_allow      Team FTR minus avg FTR opponents allow to others

  Each computed for:
    _season, _l5, _l10

ADJUSTED RATINGS:
  adj_ortg              ORTG adjusted for opponent defensive strength (drtg)
  adj_drtg              DRTG adjusted for opponent offensive strength (ortg)
  adj_net_rtg           adj_ortg - adj_drtg

ADJUSTED PACE:
  adj_pace              Pace normalized by opponent tempo context, scaled to league average pace
"""

import logging
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# League average constants (D1-ish baselines; update each season if desired)
LEAGUE_AVG_ORTG = 103.0
LEAGUE_AVG_DRTG = 103.0
LEAGUE_AVG_EFG = 50.5
LEAGUE_AVG_ORB = 28.0
LEAGUE_AVG_DRB = 72.0
LEAGUE_AVG_TOV = 18.0
LEAGUE_AVG_FTR = 28.0

# Pace is computed dynamically from the dataset, with a safe fallback
LEAGUE_AVG_PACE_FALLBACK = 70.0

SOS_WINDOWS = [5, 10]  # rolling windows; season is always included


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Leak-free rolling mean: shift(1) so game N uses only prior games."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _expanding_mean(series: pd.Series) -> pd.Series:
    """Leak-free expanding (season-to-date) mean using only prior games."""
    return series.shift(1).expanding(min_periods=1).mean()


def _get_league_avg_pace(df: pd.DataFrame) -> float:
    """
    Compute league average pace from the dataset.
    This is used as a scaling constant, not a per-team feature.

    Returns a positive float, otherwise falls back to LEAGUE_AVG_PACE_FALLBACK.
    """
    pace = pd.to_numeric(df.get("pace", np.nan), errors="coerce")

    # Prefer a robust central tendency to damp outliers
    val = float(np.nanmedian(pace.values))
    if np.isfinite(val) and val > 0:
        return val

    # Fallback to mean if median is unusable
    val = float(np.nanmean(pace.values))
    if np.isfinite(val) and val > 0:
        return val

    return float(LEAGUE_AVG_PACE_FALLBACK)


# ── Step 1: Build opponent season-to-date stat lookup ─────────────────────────

def _build_opponent_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game row, look up the opponent's rolling stats
    (season, L5, L10) as of that game date.

    Strategy:
      - Sort by game_datetime_utc
      - For each team compute expanding/rolling means with shift(1)
      - Join back to the original df on (opponent_id, game_datetime_utc)

    Returns a DataFrame indexed the same as df with opp_* columns added.
    """
    required = {"team_id", "game_datetime_utc", "opponent_id"}
    missing = required - set(df.columns)
    if missing:
        log.warning(f"_build_opponent_lookup: missing columns {missing} , skipping SOS")
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["team_id", "_sort_dt"])

    # Metrics we want to pull from the opponent side
    opp_src_metrics = [
        "ortg", "drtg", "net_rtg", "efg_pct", "pace",
        # What opponents ALLOW/FORCE (for context metrics)
        "efg_pct",
        "orb_pct", "drb_pct", "tov_pct", "ftr",
    ]
    opp_src_metrics = list(dict.fromkeys(opp_src_metrics))
    present_metrics = [m for m in opp_src_metrics if m in df.columns]

    team_stats = df[["team_id", "_sort_dt"] + present_metrics].copy()

    rolled = team_stats.groupby("team_id")[present_metrics].transform(
        lambda s: _expanding_mean(s)
    )
    rolled.columns = [f"_opp_{c}_season" for c in present_metrics]
    team_stats = pd.concat([team_stats, rolled], axis=1)

    for w in SOS_WINDOWS:
        r = team_stats.groupby("team_id")[present_metrics].transform(
            lambda s, ww=w: _rolling_mean(s, ww)
        )
        r.columns = [f"_opp_{c}_l{w}" for c in present_metrics]
        team_stats = pd.concat([team_stats, r], axis=1)

    stat_cols = [c for c in team_stats.columns if c.startswith("_opp_")]
    lookup = team_stats[["team_id", "_sort_dt"] + stat_cols].copy()
    lookup = lookup.rename(columns={"team_id": "opponent_id", "_sort_dt": "_opp_dt"})

    df = df.merge(
        lookup,
        how="left",
        left_on=["opponent_id", "_sort_dt"],
        right_on=["opponent_id", "_opp_dt"],
    )
    df = df.drop(columns=["_opp_dt"], errors="ignore")

    return df


# ── Step 2: Build "what opponents allow/force" lookup ─────────────────────────

def _build_allowed_forced_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, compute rolling averages of what THEIR OPPONENTS
    allowed/forced to OTHER opponents, giving us an opponent-context baseline.

    Example:
      Team A shot eFG%=55% vs Team B, but Team B typically allows 52%
      to opponents, so Team A is +3% above what Team B usually allows.

    We do this by:
      1. For each game row, find the opponent (Team B)
      2. Look at Team B's defensive view (what offenses did vs Team B)
      3. Average that prior to this game date

    Concretely: for each row (Team A vs Team B), the "opp_allows_efg" is the
    average eFG% that teams other than Team A shot against Team B, prior to
    this game date.
    """
    required = {"team_id", "opponent_id", "game_datetime_utc"}
    if not required.issubset(df.columns):
        log.warning("_build_allowed_forced_lookup: missing required columns , skipping")
        return df

    df = df.copy()
    if "_sort_dt" not in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")

    allow_metrics = {
        "efg_pct": "allows_efg",
        "orb_pct": "allows_orb",
        "drb_pct": "allows_drb",
        "tov_pct": "forces_tov",
        "ftr": "allows_ftr",
    }

    present = {k: v for k, v in allow_metrics.items() if k in df.columns}
    if not present:
        return df

    defense_view = df[["opponent_id", "_sort_dt", "team_id"] + list(present.keys())].copy()
    defense_view = defense_view.rename(columns={"opponent_id": "def_team_id", "team_id": "off_team_id"})
    defense_view = defense_view.sort_values(["def_team_id", "_sort_dt"])

    for src_col, label in present.items():
        defense_view[f"_allow_{label}_season"] = defense_view.groupby("def_team_id")[src_col].transform(
            _expanding_mean
        )
        for w in SOS_WINDOWS:
            defense_view[f"_allow_{label}_l{w}"] = defense_view.groupby("def_team_id")[src_col].transform(
                lambda s, ww=w: _rolling_mean(s, ww)
            )

    allow_cols = [c for c in defense_view.columns if c.startswith("_allow_")]
    lookup = defense_view[["def_team_id", "_sort_dt"] + allow_cols].copy()
    lookup = lookup.rename(columns={"def_team_id": "opponent_id", "_sort_dt": "_allow_dt"})

    df = df.merge(
        lookup,
        how="left",
        left_on=["opponent_id", "_sort_dt"],
        right_on=["opponent_id", "_allow_dt"],
    )
    df = df.drop(columns=["_allow_dt"], errors="ignore")

    return df


# ── Step 3: Compute SOS and context metrics ───────────────────────────────────

def _add_sos_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename internal _opp_* columns to clean opp_* SOS columns.
    """
    rename_map = {}
    for suffix in ["season"] + [f"l{w}" for w in SOS_WINDOWS]:
        for metric, label in [
            ("ortg", "opp_avg_ortg"),
            ("drtg", "opp_avg_drtg"),
            ("net_rtg", "opp_avg_net_rtg"),
            ("efg_pct", "opp_avg_efg"),
            ("pace", "opp_avg_pace"),
        ]:
            src = f"_opp_{metric}_{suffix}"
            dst = f"{label}_{suffix}"
            if src in df.columns:
                rename_map[src] = dst

    df = df.rename(columns=rename_map)
    return df


def _add_context_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each window, compute team metric minus opponent's allowed/forced baseline.
    Positive means team performed above what that opponent typically allows/forces.
    Negative means below.
    """
    context_pairs = [
        ("efg_pct", "allows_efg", "efg_vs_opp"),
        ("orb_pct", "allows_orb", "orb_vs_opp"),
        ("drb_pct", "allows_drb", "drb_vs_opp"),
        ("tov_pct", "forces_tov", "tov_vs_opp"),
        ("ftr", "allows_ftr", "ftr_vs_opp"),
    ]

    for suffix in ["season"] + [f"l{w}" for w in SOS_WINDOWS]:
        for team_col, allow_base, out_base in context_pairs:
            allow_col = f"_allow_{allow_base}_{suffix}"
            out_col = f"{out_base}_{suffix}"
            if team_col in df.columns and allow_col in df.columns:
                team_vals = pd.to_numeric(df[team_col], errors="coerce")
                allow_vals = pd.to_numeric(df[allow_col], errors="coerce")
                df[out_col] = (team_vals - allow_vals).round(2)

    drop_cols = [c for c in df.columns if c.startswith("_allow_") or c.startswith("_opp_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def _add_adjusted_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    True adjusted pace using opponent pace data.

    adj_pace = (team_pace / avg(team_pace, opp_avg_pace_season)) * league_avg_pace

    Accounts for opponent tempo:
      A team that plays fast against slow opponents is genuinely faster than one
      that looks fast mainly because opponents push pace.

    league_avg_pace is computed dynamically from the dataset (median pace),
    with a safe fallback.
    """
    if "pace" not in df.columns or "opp_avg_pace_season" not in df.columns:
        return df

    league_avg_pace = _get_league_avg_pace(df)

    team_pace = pd.to_numeric(df["pace"], errors="coerce")
    opp_pace = pd.to_numeric(df["opp_avg_pace_season"], errors="coerce")
    avg_pace = (team_pace + opp_pace) / 2

    df["adj_pace"] = (team_pace / avg_pace.replace(0, np.nan) * league_avg_pace).round(1)
    return df


def _add_adjusted_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute opponent-adjusted efficiency ratings.

    adj_ortg = ortg - (opp_avg_drtg_season - LEAGUE_AVG_DRTG)
      How well did you score, accounting for how tough the defense was?

    adj_drtg = drtg + (opp_avg_ortg_season - LEAGUE_AVG_ORTG)
      How well did you defend, accounting for how tough the offense was?

    adj_net_rtg = adj_ortg - adj_drtg
    """
    if "ortg" not in df.columns:
        return df

    ortg = pd.to_numeric(df.get("ortg", np.nan), errors="coerce")
    drtg = pd.to_numeric(df.get("drtg", np.nan), errors="coerce")
    opp_drtg = pd.to_numeric(df.get("opp_avg_drtg_season", np.nan), errors="coerce")
    opp_ortg = pd.to_numeric(df.get("opp_avg_ortg_season", np.nan), errors="coerce")

    df["adj_ortg"] = (ortg - (opp_drtg - LEAGUE_AVG_DRTG)).round(1)
    df["adj_drtg"] = (drtg + (opp_ortg - LEAGUE_AVG_ORTG)).round(1)
    df["adj_net_rtg"] = (df["adj_ortg"] - df["adj_drtg"]).round(1)

    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_sos_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full SOS pipeline. Input should be the output of compute_all_metrics()
    (team_game_metrics.csv with per-game advanced stats).

    Requires columns:
      team_id, opponent_id, game_datetime_utc,
      ortg, drtg, net_rtg, efg_pct, pace,
      orb_pct, drb_pct, tov_pct, ftr

    Returns df with SOS, context, and adjusted rating columns appended.
    Output is written to team_game_sos.csv by the pipeline.
    """
    if df.empty:
        log.warning("compute_sos_metrics: empty DataFrame , skipping")
        return df

    missing = {"team_id", "opponent_id", "game_datetime_utc"} - set(df.columns)
    if missing:
        log.warning(f"compute_sos_metrics: missing required columns {missing} , skipping")
        return df

    log.info(f"Computing SOS metrics for {len(df)} rows")

    df = _build_opponent_lookup(df)
    df = _build_allowed_forced_lookup(df)
    df = _add_sos_columns(df)
    df = _add_context_metrics(df)
    df = _add_adjusted_pace(df)
    df = _add_adjusted_ratings(df)

    df = df.drop(columns=["_sort_dt"], errors="ignore")

    log.info("SOS metrics complete")
    return df
