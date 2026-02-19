"""
ESPN CBB Pipeline — Opponent-Weighted Rolling Metrics
Computes L5/L10 rolling averages where each game is weighted by the
quality of the opponent faced, derived from SOS efficiency metrics.

Philosophy (from the quants):
  Equal-weight rolling averages treat a 110 ORTG against a 95-DRTG defense
  the same as a 110 ORTG against a 115-DRTG defense. That's wrong.
  Weighting by opponent quality surfaces true performance signal.

Three weight schemes per metric:
  _wtd_off_l{N}   Weighted by opponent DRTG (how hard was it to score?)
  _wtd_def_l{N}   Weighted by opponent ORTG (how hard was it to stop them?)
  _wtd_qual_l{N}  Weighted by opponent NetRTG (overall opponent quality)

Weight formula:
  w_off  = opp_DRTG / LEAGUE_AVG_DRTG
    → opponent with 90 DRTG (stingy D) gives w=0.87 → game down-weighted
    → opponent with 115 DRTG (bad D)   gives w=1.11 → game up-weighted
    Wait — we want the inverse: scoring well against a TOUGH defense
    (low DRTG = hard to score against) should be weighted MORE.

  Correct:
    w_off  = LEAGUE_AVG_DRTG / opp_DRTG
      (scoring against a 90 DRTG team gets w=103/90=1.14 — harder D, higher weight)
    w_def  = opp_ORTG / LEAGUE_AVG_ORTG
      (defending a 115 ORTG offense gets w=115/103=1.12 — better O, higher weight)
    w_qual = (opp_NetRTG + OFFSET) / OFFSET
      (opponent with +8 NetRTG gets higher weight than opponent with -5)

All weights are normalized within each rolling window so they sum to 1,
making the weighted average directly comparable to the unweighted one.

Output: team_game_weighted.csv
"""

import logging
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
LEAGUE_AVG_ORTG  = 103.0
LEAGUE_AVG_DRTG  = 103.0
NET_RTG_OFFSET   = 20.0   # shift NetRTG into positive range for weighting
                           # (worst team ~-15, best ~+30; offset keeps all > 0)
ROLLING_WINDOWS  = [5, 10]
MIN_PERIODS      = 2       # minimum games before reporting weighted values

# Metrics weighted by OPPONENT DEFENSE quality (offensive performance metrics)
OFF_METRICS = [
    "ortg", "efg_pct", "ts_pct", "fg_pct", "three_pct",
    "h1_ortg", "h2_ortg", "points_for",
    "fgm", "fga", "tpm", "tpa", "ftm", "fta",
    "ast", "three_par",
]

# Metrics weighted by OPPONENT OFFENSE quality (defensive performance metrics)
DEF_METRICS = [
    "drtg", "h1_drtg", "h2_drtg", "points_against",
    "stl", "blk", "tov_pct",
]

# Metrics weighted by OVERALL OPPONENT QUALITY (two-way / general)
QUAL_METRICS = [
    "net_rtg", "margin", "margin_capped",
    "orb_pct", "drb_pct", "poss", "pace",
    "h1_margin", "h2_margin", "cover_margin",
    "win", "cover",
]


# ── Core weighted rolling function ───────────────────────────────────────────

def _weighted_rolling(
    values: pd.Series,
    weights: pd.Series,
    window: int,
    min_periods: int = MIN_PERIODS,
) -> pd.Series:
    """
    Compute leak-free weighted rolling mean.
    Uses shift(1) so game N uses only games 1..N-1.

    For each position i, takes the last `window` observations from
    values[i-window:i] and weights[i-window:i], normalizes weights,
    returns sum(v * w_norm).
    """
    vals = values.shift(1).values
    wts  = weights.shift(1).values
    result = np.full(len(vals), np.nan)

    for i in range(len(vals)):
        start = max(0, i - window)
        v = vals[start:i]
        w = wts[start:i]

        # Drop NaN pairs
        mask = ~(np.isnan(v) | np.isnan(w))
        v, w = v[mask], w[mask]

        if len(v) < min_periods:
            continue

        w_sum = w.sum()
        if w_sum == 0:
            result[i] = np.nanmean(v)
        else:
            result[i] = np.dot(v, w / w_sum)

    return pd.Series(result, index=values.index).round(2)


# ── Weight construction ───────────────────────────────────────────────────────

def _build_weights(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build three weight columns per row from opponent efficiency data.
    Expects opp_avg_drtg_season, opp_avg_ortg_season, opp_avg_net_rtg_season
    from espn_sos.py output. Falls back to 1.0 if not available.
    """
    df = df.copy()

    opp_drtg    = pd.to_numeric(df.get("opp_avg_drtg_season",
                                       LEAGUE_AVG_DRTG), errors="coerce")
    opp_ortg    = pd.to_numeric(df.get("opp_avg_ortg_season",
                                       LEAGUE_AVG_ORTG), errors="coerce")
    opp_net_rtg = pd.to_numeric(df.get("opp_avg_net_rtg_season",
                                       0.0), errors="coerce")

    # w_off: higher when opponent has LOWER DRTG (tougher D to score against)
    df["_w_off"]  = (LEAGUE_AVG_DRTG / opp_drtg.replace(0, np.nan)
                    ).clip(0.5, 2.0).fillna(1.0)

    # w_def: higher when opponent has HIGHER ORTG (tougher O to defend)
    df["_w_def"]  = (opp_ortg / LEAGUE_AVG_ORTG
                    ).clip(0.5, 2.0).fillna(1.0)

    # w_qual: higher when opponent has better NetRTG overall
    df["_w_qual"] = ((opp_net_rtg + NET_RTG_OFFSET) / NET_RTG_OFFSET
                    ).clip(0.25, 3.0).fillna(1.0)

    return df


# ── Per-game performance-vs-expectation columns ───────────────────────────────

def add_performance_vs_expectation(df: pd.DataFrame) -> pd.DataFrame:
    """
    How much did a team EXCEED or FALL SHORT of what the opponent typically
    allows/forces, on a per-game basis?

    perf_vs_exp_ortg   = ortg - opp_avg_drtg_season
    perf_vs_exp_drtg   = drtg - opp_avg_ortg_season  (negative = good defense)
    perf_vs_exp_net    = net_rtg - opp_avg_net_rtg_season

    These are the game-level "did you beat expectations" columns.
    Rolling averages of these give a clean quality-adjusted trend signal.
    """
    df = df.copy()

    if "ortg" in df.columns and "opp_avg_drtg_season" in df.columns:
        df["perf_vs_exp_ortg"] = (
            pd.to_numeric(df["ortg"], errors="coerce") -
            pd.to_numeric(df["opp_avg_drtg_season"], errors="coerce")
        ).round(1)

    if "drtg" in df.columns and "opp_avg_ortg_season" in df.columns:
        df["perf_vs_exp_drtg"] = (
            pd.to_numeric(df["drtg"], errors="coerce") -
            pd.to_numeric(df["opp_avg_ortg_season"], errors="coerce")
        ).round(1)
        # Flip sign so positive = good defense
        df["perf_vs_exp_def"] = -df["perf_vs_exp_drtg"]

    if "net_rtg" in df.columns and "opp_avg_net_rtg_season" in df.columns:
        df["perf_vs_exp_net"] = (
            pd.to_numeric(df["net_rtg"], errors="coerce") -
            pd.to_numeric(df["opp_avg_net_rtg_season"], errors="coerce")
        ).round(1)

    return df


# ── Weighted rolling computation ──────────────────────────────────────────────

def add_weighted_rolling(df: pd.DataFrame,
                          windows: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Add opponent-quality-weighted rolling averages for all metric groups.

    New columns follow the pattern:
      {metric}_wtd_off_l{N}    e.g. ortg_wtd_off_l5
      {metric}_wtd_def_l{N}    e.g. drtg_wtd_def_l10
      {metric}_wtd_qual_l{N}   e.g. net_rtg_wtd_qual_l5

    Also adds rolling means of perf_vs_exp columns:
      perf_vs_exp_ortg_l{N}
      perf_vs_exp_def_l{N}
      perf_vs_exp_net_l{N}
    """
    if "team_id" not in df.columns or "game_datetime_utc" not in df.columns:
        log.warning("add_weighted_rolling: missing required columns")
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True,
                                    errors="coerce")
    df = df.sort_values(["team_id", "_sort_dt"])

    # ── Weight columns ──
    df = _build_weights(df)

    metric_groups = [
        (OFF_METRICS,  "_w_off",  "wtd_off"),
        (DEF_METRICS,  "_w_def",  "wtd_def"),
        (QUAL_METRICS, "_w_qual", "wtd_qual"),
    ]

    for window in windows:
        for metric_list, weight_col, suffix in metric_groups:
            present = [m for m in metric_list if m in df.columns]
            for metric in present:
                col_name = f"{metric}_{suffix}_l{window}"
                df[col_name] = df.groupby("team_id", group_keys=False).apply(
                    lambda g, m=metric, w=weight_col, ww=window: _weighted_rolling(
                        pd.to_numeric(g[m], errors="coerce"),
                        pd.to_numeric(g[w], errors="coerce"),
                        ww,
                    )
                )

        # Rolling averages of perf_vs_exp columns (unweighted — already adjusted)
        for pve_col in ["perf_vs_exp_ortg", "perf_vs_exp_def", "perf_vs_exp_net"]:
            if pve_col in df.columns:
                df[f"{pve_col}_l{window}"] = df.groupby("team_id")[pve_col].transform(
                    lambda s: s.shift(1).rolling(window, min_periods=MIN_PERIODS)
                    .mean().round(2)
                )

    # ── Weighted cover rate ──
    # Cover weighted by opponent quality — covering vs ranked opponents counts more
    if "cover" in df.columns:
        for window in windows:
            df[f"cover_wtd_qual_rate_l{window}"] = df.groupby(
                "team_id", group_keys=False
            ).apply(
                lambda g, ww=window: _weighted_rolling(
                    pd.to_numeric(g["cover"], errors="coerce"),
                    pd.to_numeric(g["_w_qual"], errors="coerce"),
                    ww,
                )
            )

    # ── Momentum score ──
    # Composite: recent weighted net rating trend (L5 vs L10 weighted)
    # Positive = improving momentum, negative = declining
    if ("net_rtg_wtd_qual_l5" in df.columns and
            "net_rtg_wtd_qual_l10" in df.columns):
        df["momentum_score"] = (
            df["net_rtg_wtd_qual_l5"] - df["net_rtg_wtd_qual_l10"]
        ).round(2)

    # ── Schedule-adjusted form rating ──
    # Combines weighted net rtg + performance vs expectation
    # High score = performing well AND doing it against good opponents
    if ("net_rtg_wtd_qual_l5" in df.columns and
            "perf_vs_exp_net_l5" in df.columns):
        net_w = pd.to_numeric(df["net_rtg_wtd_qual_l5"], errors="coerce")
        pve   = pd.to_numeric(df["perf_vs_exp_net_l5"], errors="coerce")
        df["form_rating"] = (0.6 * net_w + 0.4 * pve).round(2)

    # Drop internal weight columns
    df = df.drop(columns=["_w_off", "_w_def", "_w_qual", "_sort_dt"],
                 errors="ignore")

    log.info(f"Weighted rolling metrics complete — "
             f"{len([c for c in df.columns if 'wtd' in c])} weighted columns added")
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_weighted_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full weighted metrics pipeline.
    Input:  team_game_sos.csv DataFrame (needs SOS opp_avg_* columns)
    Output: same DataFrame with weighted rolling columns appended.
    Written to team_game_weighted.csv by the pipeline.
    """
    if df.empty:
        log.warning("compute_weighted_metrics: empty DataFrame")
        return df

    log.info(f"Computing weighted metrics for {len(df)} team-game rows")
    df = add_performance_vs_expectation(df)
    df = add_weighted_rolling(df, windows=ROLLING_WINDOWS)
    log.info("Weighted metrics complete")
    return df
