"""Shared box-score derivations used by the advanced metric library."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def add_shared_derivations(team_game_df: pd.DataFrame) -> pd.DataFrame:
    """Add shared team-game derivation columns from box score inputs only."""
    df = team_game_df.copy()

    # Team box score
    fga = _to_num(df["fga"])
    fgm = _to_num(df["fgm"])
    tpa = _to_num(df["tpa"])
    tpm = _to_num(df["tpm"])
    fta = _to_num(df["fta"])
    ftm = _to_num(df["ftm"])
    orb = _to_num(df["orb"])
    drb = _to_num(df["drb"])
    tov = _to_num(df["tov"])
    pts_for = _to_num(df["points_for"])

    # Opponent mirrored box score (same row)
    opp_fga = _to_num(df["opp_fga"])
    opp_fgm = _to_num(df["opp_fgm"])
    opp_tpa = _to_num(df["opp_tpa"])
    opp_tpm = _to_num(df["opp_tpm"])
    opp_fta = _to_num(df["opp_fta"])
    opp_ftm = _to_num(df["opp_ftm"])
    opp_orb = _to_num(df["opp_orb"])
    opp_drb = _to_num(df["opp_drb"])
    opp_tov = _to_num(df["opp_tov"])
    pts_against = _to_num(df["points_against"])

    # Possessions
    poss = fga - orb + tov + (0.44 * fta)
    opp_poss = opp_fga - opp_orb + opp_tov + (0.44 * opp_fta)

    # Core efficiency
    off_eff = _safe_div(pts_for * 100.0, poss)
    def_eff = _safe_div(pts_against * 100.0, opp_poss)
    net_rtg = off_eff - def_eff

    # Shooting / profile
    efg = _safe_div(fgm + (0.5 * tpm), fga)
    three_pa_rate = _safe_div(tpa, fga)
    ftr = _safe_div(fta, fga)
    ft_pts_per_fga = _safe_div(ftm, fga)
    ft_pts_per_poss = _safe_div(ftm, poss)

    # Rebounding / ball security
    orb_pct = _safe_div(orb, orb + opp_drb)
    drb_pct = _safe_div(drb, drb + opp_orb)
    tov_pct = _safe_div(tov, poss)

    df["poss"] = poss
    df["OffEff"] = off_eff
    df["DefEff"] = def_eff
    df["NetRtg"] = net_rtg
    df["eFG"] = efg
    df["3PA_rate"] = three_pa_rate
    df["FTr"] = ftr
    df["FT_pts_per_FGA"] = ft_pts_per_fga
    df["FT_pts_per_poss"] = ft_pts_per_poss
    df["ORB%"] = orb_pct
    df["DRB%"] = drb_pct
    df["TOV%"] = tov_pct

    # Internal helper columns for opponent-side references.
    df["_opp_eFG"] = _safe_div(opp_fgm + (0.5 * opp_tpm), opp_fga)
    df["_opp_FTr"] = _safe_div(opp_fta, opp_fga)
    df["_opp_ORB%"] = _safe_div(opp_orb, opp_orb + drb)
    df["_opp_TOV%"] = _safe_div(opp_tov, opp_poss)

    return df

