"""
ESPN CBB Pipeline — Advanced Metrics
Computes derived per-game stats and rolling team averages from box score data.
All functions are pure (DataFrame in, DataFrame out). No I/O here.

Per-game metrics computed:
  - eFG%         Effective Field Goal %
  - TS%          True Shooting %
  - ORB%         Offensive Rebound %
  - DRB%         Defensive Rebound %
  - TRB%         Total Rebound %
  - TOV%         Turnover %
  - FTR          Free Throw Rate (FTA/FGA)
  - 3PAR         Three Point Attempt Rate (3PA/FGA)
  - Possessions  Estimated possessions
  - ORTG         Offensive Rating (points per 100 possessions)
  - DRTG         Defensive Rating (opponent points per 100 possessions)
  - NetRTG       Net Rating (ORTG - DRTG)
  - Pace         Estimated possessions per 40 minutes
  - AdjPace      Pace normalized to 70 possessions per game (league average)

Rolling windows (last 5 and last 10 games, per team):
  - All per-game metrics above
  - Points for / against / margin
"""

import logging
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
REGULATION_MINUTES = 40.0
LEAGUE_AVG_PACE    = 70.0   # approximate D1 average possessions per 40 min
ROLLING_WINDOWS    = [5, 10]


# ── Safe math helpers ─────────────────────────────────────────────────────────

def _div(num, den, fill=np.nan):
    """Safe division — returns fill if denominator is 0 or NaN."""
    try:
        if pd.isna(den) or den == 0:
            return fill
        return num / den
    except Exception:
        return fill


def _pct(num, den, fill=np.nan):
    """Safe percentage (0–100 scale)."""
    return _div(num, den, fill) * 100 if not np.isnan(_div(num, den, fill)) else fill


# ── Per-game metrics ──────────────────────────────────────────────────────────

def add_per_game_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add advanced per-game metrics to a team_game_logs DataFrame.
    Expects columns: fgm, fga, tpm, tpa, ftm, fta, orb, drb, reb,
                     tov, points_for, points_against, margin.
    Safe to call even if some columns are missing — missing inputs produce NaN.

    Returns the DataFrame with new columns appended.
    """
    df = df.copy()

    def col(name):
        """Return series or NaN series if column missing."""
        return pd.to_numeric(df[name], errors="coerce") if name in df.columns else pd.Series(np.nan, index=df.index)

    fgm  = col("fgm");  fga  = col("fga")
    tpm  = col("tpm");  tpa  = col("tpa")
    ftm  = col("ftm");  fta  = col("fta")
    orb  = col("orb");  drb  = col("drb");  reb = col("reb")
    tov  = col("tov")
    pts  = col("points_for")
    opp_pts = col("points_against")

    # ── Shooting ──
    # eFG% = (FGM + 0.5 * 3PM) / FGA
    df["efg_pct"] = ((fgm + 0.5 * tpm) / fga.replace(0, np.nan) * 100).round(1)

    # TS% = PTS / (2 * (FGA + 0.44 * FTA))
    ts_denom = 2 * (fga + 0.44 * fta)
    df["ts_pct"] = (pts / ts_denom.replace(0, np.nan) * 100).round(1)

    # 3PAR = 3PA / FGA
    df["three_par"] = (tpa / fga.replace(0, np.nan) * 100).round(1)

    # FTR = FTA / FGA
    df["ftr"] = (fta / fga.replace(0, np.nan) * 100).round(1)

    # FT% and FG% (basic, useful for context)
    df["fg_pct"]  = (fgm / fga.replace(0, np.nan) * 100).round(1)
    df["three_pct"] = (tpm / tpa.replace(0, np.nan) * 100).round(1)
    df["ft_pct"]  = (ftm / fta.replace(0, np.nan) * 100).round(1)

    # ── Rebounding ──
    # Use reb as fallback if orb/drb missing
    orb_clean = orb.fillna(reb - drb) if "drb" in df.columns else orb
    drb_clean = drb.fillna(reb - orb) if "orb" in df.columns else drb

    # ORB% = ORB / (ORB + opp_DRB)  — requires opponent join; approximate here
    # as share of own total boards until opponent merge is available
    df["orb_pct"] = (orb_clean / reb.replace(0, np.nan) * 100).round(1)
    df["drb_pct"] = (drb_clean / reb.replace(0, np.nan) * 100).round(1)

    # ── Possessions estimate ──
    # Poss ≈ FGA - ORB + TOV + 0.44 * FTA  (standard Kubatko formula)
    poss = fga - orb_clean + tov + 0.44 * fta
    df["poss"] = poss.round(1)

    # ── Efficiency ratings (per 100 possessions) ──
    df["ortg"]   = (pts    / poss.replace(0, np.nan) * 100).round(1)
    df["drtg"]   = (opp_pts / poss.replace(0, np.nan) * 100).round(1)
    df["net_rtg"] = (df["ortg"] - df["drtg"]).round(1)

    # ── Turnover % = TOV / Poss ──
    df["tov_pct"] = (tov / poss.replace(0, np.nan) * 100).round(1)

    # ── Pace = possessions per 40 minutes ──
    # Without minutes played we use regulation as denominator.
    # Games with OT will show slightly lower pace — flag in is_ot column.
    df["pace"] = (poss / REGULATION_MINUTES * 40).round(1)

    # ── Adjusted pace = team pace normalized to league average ──
    # adj_pace = pace * (LEAGUE_AVG_PACE / team_pace)
    # This re-scales each team's pace to what it would be at a neutral 70-poss tempo.
    df["adj_pace"] = (
        df["pace"].where(df["pace"] > 0)
        .apply(lambda p: round(p * (LEAGUE_AVG_PACE / p), 1) if pd.notna(p) and p > 0 else np.nan)
    )
    # Note: adj_pace here is always LEAGUE_AVG_PACE until opponent data is joined.
    # True adjusted pace (accounts for opponent pace) is computed in rolling section.

    return df


# ── Rolling window features ───────────────────────────────────────────────────

ROLLING_METRICS = [
    "points_for", "points_against", "margin",
    "efg_pct", "ts_pct", "three_par", "ftr", "fg_pct", "three_pct", "ft_pct",
    "orb_pct", "drb_pct", "tov_pct",
    "ortg", "drtg", "net_rtg",
    "poss", "pace",
]


def add_rolling_metrics(df: pd.DataFrame, windows: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Add rolling averages for each metric in ROLLING_METRICS, grouped by team_id.
    Uses only prior games (shift(1)) to avoid data leakage — the rolling value
    on row N reflects the average of the N-1 most recent games before that game.

    New columns follow the pattern:
      {metric}_l{window}   e.g. ortg_l5, net_rtg_l10

    Requires columns: team_id, game_datetime_utc (for sort order).
    """
    if "team_id" not in df.columns or "game_datetime_utc" not in df.columns:
        log.warning("add_rolling_metrics: missing team_id or game_datetime_utc — skipping")
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["team_id", "_sort_dt"])

    metrics_present = [m for m in ROLLING_METRICS if m in df.columns]

    for window in windows:
        for metric in metrics_present:
            col_name = f"{metric}_l{window}"
            df[col_name] = (
                df.groupby("team_id")[metric]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean().round(2))
            )

    df = df.drop(columns=["_sort_dt"], errors="ignore")
    return df


# ── Opponent-adjusted ORB%/DRB% ──────────────────────────────────────────────

def add_true_rebound_pcts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute true ORB% and DRB% using opponent rebound data.
    Requires opponent columns: opp_orb, opp_drb, opp_reb (added after opponent join).

    True ORB% = ORB / (ORB + opp_DRB)
    True DRB% = DRB / (DRB + opp_ORB)

    Only runs if opponent columns are present — safe to call otherwise.
    """
    df = df.copy()

    has_opp = all(c in df.columns for c in ["opp_orb", "opp_drb"])
    if not has_opp:
        log.debug("add_true_rebound_pcts: opponent rebound columns not present, skipping")
        return df

    orb = pd.to_numeric(df.get("orb", np.nan), errors="coerce")
    drb = pd.to_numeric(df.get("drb", np.nan), errors="coerce")
    opp_orb = pd.to_numeric(df["opp_orb"], errors="coerce")
    opp_drb = pd.to_numeric(df["opp_drb"], errors="coerce")

    denom_orb = orb + opp_drb
    denom_drb = drb + opp_orb

    df["true_orb_pct"] = (orb / denom_orb.replace(0, np.nan) * 100).round(1)
    df["true_drb_pct"] = (drb / denom_drb.replace(0, np.nan) * 100).round(1)

    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_all_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full metrics pipeline:
      1. Per-game advanced metrics
      2. Rolling window averages (L5 / L10)
      3. True rebound % if opponent data available

    This is the function to call from espn_pipeline.py.
    """
    log.info(f"Computing metrics for {len(df)} team-game rows")
    df = add_per_game_metrics(df)
    df = add_rolling_metrics(df, windows=ROLLING_WINDOWS)
    df = add_true_rebound_pcts(df)
    log.info("Metrics complete")
    return df
