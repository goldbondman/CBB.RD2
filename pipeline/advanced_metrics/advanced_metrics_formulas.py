"""Centralized formulas for Advanced Metrics Codex.

All functions are deterministic and side-effect free.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def safe_zscore(value: float, mean: float, std: float) -> float:
    """Compute z-score with safe zero-std handling.

    If std is 0 or non-finite, returns 0.0.
    """
    if not np.isfinite(std) or std == 0:
        return 0.0
    if not np.isfinite(value) or not np.isfinite(mean):
        return 0.0
    return float((value - mean) / std)


def zscore_series(series: pd.Series, by: pd.Series | None = None) -> pd.Series:
    """Vectorized z-score for a pandas Series.

    Uses population std (ddof=0). If std=0, output is 0 for non-null rows.
    """
    values = pd.to_numeric(series, errors="coerce")
    if by is None:
        mean = values.mean()
        std = values.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            out = pd.Series(0.0, index=values.index, dtype=float)
            return out.where(values.notna(), np.nan)
        return (values - mean) / std

    out = pd.Series(np.nan, index=values.index, dtype=float)
    groups = by.reindex(values.index)
    for _, idx in groups.groupby(groups).groups.items():
        sub = values.loc[idx]
        mean = sub.mean()
        std = sub.std(ddof=0)
        if not np.isfinite(std) or std == 0:
            z = pd.Series(0.0, index=sub.index, dtype=float)
            z = z.where(sub.notna(), np.nan)
        else:
            z = (sub - mean) / std
        out.loc[sub.index] = z
    return out


def rolling_mean(series: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    """Rolling mean helper."""
    values = pd.to_numeric(series, errors="coerce")
    return values.rolling(window=window, min_periods=min_periods).mean()


def rolling_std(series: pd.Series, window: int, min_periods: int = 2) -> pd.Series:
    """Rolling population std helper (ddof=0)."""
    values = pd.to_numeric(series, errors="coerce")
    return values.rolling(window=window, min_periods=min_periods).std(ddof=0)


def expanding_mean(series: pd.Series, min_periods: int = 1) -> pd.Series:
    """Expanding mean helper."""
    values = pd.to_numeric(series, errors="coerce")
    return values.expanding(min_periods=min_periods).mean()


def mean_abs(values: Iterable[float]) -> float:
    """Mean absolute value helper."""
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(np.abs(arr)))


def odi_star(e_fg_odi: float, to_odi: float, orb_odi: float, ftr_odi: float) -> float:
    """ODI* = 0.4*eFG_ODI + 0.25*TO_ODI + 0.2*ORB_ODI + 0.15*FTR_ODI."""
    return 0.4 * e_fg_odi + 0.25 * to_odi + 0.2 * orb_odi + 0.15 * ftr_odi


def pei(orb_off_adj: float, orb_def_adj: float, tov_def_adj: float, tov_off_adj: float) -> float:
    """PEI = (ORB_off_adj - ORB_def_adj) + (TOV_def_adj - TOV_off_adj)."""
    return (orb_off_adj - orb_def_adj) + (tov_def_adj - tov_off_adj)


def posw(pei_value: float, projected_pace: float) -> float:
    """POSW = PEI * projected_pace / 100."""
    return pei_value * projected_pace / 100.0


def svi(z_efg: float, z_three_pa_rate: float, z_ft_pts_per_fga: float) -> float:
    """SVI = z(eFG%) + 0.5*z(3PA_rate) + 0.5*z(FT_pts/FGA)."""
    return z_efg + 0.5 * z_three_pa_rate + 0.5 * z_ft_pts_per_fga


def ics(returning_minutes: float, rotation_stability: float, star_continuity: float) -> float:
    """ICS = 0.4*returning_minutes + 0.3*rotation_stability + 0.3*star_continuity."""
    return 0.4 * returning_minutes + 0.3 * rotation_stability + 0.3 * star_continuity


def bsi(bench_minutes: float, bench_ts_rel: float, bench_reb_rel: float) -> float:
    """BSI = bench_minutes * (0.6*bench_TS_rel + 0.4*bench_reb_rel)."""
    return bench_minutes * (0.6 * bench_ts_rel + 0.4 * bench_reb_rel)


def pxp(ics_value: float, bsi_value: float, clutch_pct: float) -> float:
    """PXP = 0.4*ICS + 0.3*BSI + 0.3*clutch_pct."""
    return 0.4 * ics_value + 0.3 * bsi_value + 0.3 * clutch_pct


def lns(minutes_share: pd.Series, player_tois: pd.Series) -> float:
    """LNS = sum(minutes_share_i * player_TOIS_i)."""
    ms = pd.to_numeric(minutes_share, errors="coerce")
    tois = pd.to_numeric(player_tois, errors="coerce")
    return float((ms * tois).sum(min_count=1))


def sie(star_ts: float, team_ts: float, star_usage: float) -> float:
    """SIE = (star_TS% - team_TS%) * star_usage."""
    return (star_ts - team_ts) * star_usage


def usef(sie_value: float, weighted_ts_top_usage_cluster: float) -> float:
    """USEF = SIE + weighted_TS_top_usage_cluster."""
    return sie_value + weighted_ts_top_usage_cluster


def dpc(bsi_value: float, best_rotation_netrtg: float, worst_rotation_netrtg: float) -> float:
    """DPC = BSI + (best_rotation_netrtg - worst_rotation_netrtg)."""
    return bsi_value + (best_rotation_netrtg - worst_rotation_netrtg)


def fii(dpc_value: float, backup_quality_key_positions: float, ffc_inverted: float) -> float:
    """FII = DPC * backup_quality_key_positions + FFC_inverted."""
    return dpc_value * backup_quality_key_positions + ffc_inverted


def sme(
    star_three_pa_rate: float,
    star_rim_rate: float,
    star_midrange_rate: float,
    star_post_ups: float,
    opp3p_allowed: float,
    opp_rim_fg_allowed: float,
    opp_mid_fg_allowed: float,
    opp_post_ups_allowed: float,
) -> float:
    """SME = sum(star_shot_profile_i * opp_def_weakness_profile_i)."""
    return (
        star_three_pa_rate * opp3p_allowed
        + star_rim_rate * opp_rim_fg_allowed
        + star_midrange_rate * opp_mid_fg_allowed
        + star_post_ups * opp_post_ups_allowed
    )


def sch(
    pace_a: float,
    pace_b: float,
    three_pa_rate_a: float,
    three_pa_rate_b: float,
    ftr_a: float,
    ftr_b: float,
    size_a: float,
    size_b: float,
) -> float:
    """SCH = |paceA-paceB| + |3PA_rateA-3PA_rateB| + |FTrA-FTrB| + |sizeA-sizeB|."""
    return (
        abs(pace_a - pace_b)
        + abs(three_pa_rate_a - three_pa_rate_b)
        + abs(ftr_a - ftr_b)
        + abs(size_a - size_b)
    )


def vol(netrtg_adj_last8: pd.Series) -> float:
    """VOL = std(netrtg_adj, last 8)."""
    arr = pd.to_numeric(netrtg_adj_last8, errors="coerce").dropna()
    if arr.empty:
        return float("nan")
    return float(arr.std(ddof=0))


def tc(z_mean_abs_pace_delta_last12: float) -> float:
    """TC = -z(mean(|game_pace - team_avg_pace|) last 12)."""
    return -z_mean_abs_pace_delta_last12


def wl(z_ft_ppp_diff: float, z_ftr_diff: float) -> float:
    """WL = z(FT_PPP - opp_FT_PPP) + 0.5*z(FTr - opp_FTr)."""
    return z_ft_ppp_diff + 0.5 * z_ftr_diff


def rfd(days_rest_home: float, days_rest_away: float, back2back_penalty: float) -> float:
    """RFD = days_rest_home - days_rest_away + 0.5*back2back_penalty."""
    return (days_rest_home - days_rest_away) + 0.5 * back2back_penalty


def gsr(tournament_stage: float, elimination_risk: float, spread_magnitude: float) -> float:
    """GSR = 0.4*tournament_stage + 0.3*elimination_risk + 0.3*spread_magnitude."""
    return 0.4 * tournament_stage + 0.3 * elimination_risk + 0.3 * spread_magnitude


def alt(elevation_home: float, elevation_away: float, cross_country_miles: float) -> float:
    """ALT = elevation_home - elevation_away + cross_country_miles/1000."""
    return (elevation_home - elevation_away) + (cross_country_miles / 1000.0)

