"""Feature compute functions for the registry-driven Feature Engine."""

from __future__ import annotations

import pandas as pd

from .rolling_window_layer import within_season_zscore


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def f_WL(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"WL": (_to_num(df["points_for"]) > _to_num(df["points_against"])).astype(float)})


def f_ANE(df: pd.DataFrame) -> pd.DataFrame:
    out = _to_num(df["NetRtg"]) - _to_num(df["opp_pre_NetRtg_season"]).fillna(0.0)
    return pd.DataFrame({"ANE": out})


def f_SVI(df: pd.DataFrame) -> pd.DataFrame:
    wl = _to_num(df["WL"]).fillna(0.0)
    opp_strength = _to_num(df["opp_pre_NetRtg_season"]).fillna(0.0)
    return pd.DataFrame({"SVI": wl * (1.0 + (opp_strength / 30.0))})


def f_PEQ(df: pd.DataFrame) -> pd.DataFrame:
    out = _to_num(df["OffEff"]) * (1.0 - _to_num(df["TOV%"]).fillna(0.0))
    return pd.DataFrame({"PEQ": out})


def f_POSW(df: pd.DataFrame) -> pd.DataFrame:
    orb = _to_num(df["ORB%"]).fillna(0.0)
    drb = _to_num(df["DRB%"]).fillna(0.0)
    tov = _to_num(df["TOV%"]).fillna(0.0)
    out = (orb + drb + (1.0 - tov)) / 3.0
    return pd.DataFrame({"POSW": out})


def f_ODI(df: pd.DataFrame) -> pd.DataFrame:
    efg_edge = _to_num(df["eFG"]) - _to_num(df["opp_eFG"])
    tov_edge = _to_num(df["opp_TOV%"]) - _to_num(df["TOV%"])
    orb_edge = _to_num(df["ORB%"]) - _to_num(df["opp_ORB%"])
    ftr_edge = _to_num(df["FTr"]) - _to_num(df["opp_FTr"])
    out = (0.40 * efg_edge) + (0.25 * tov_edge) + (0.20 * orb_edge) + (0.15 * ftr_edge)
    return pd.DataFrame({"ODI": out})


def f_factor_ODIs(df: pd.DataFrame) -> pd.DataFrame:
    efg_odi = _to_num(df["eFG"]) - _to_num(df["opp_eFG"])
    to_odi = _to_num(df["opp_TOV%"]) - _to_num(df["TOV%"])
    orb_odi = _to_num(df["ORB%"]) - _to_num(df["opp_ORB%"])
    ftr_odi = _to_num(df["FTr"]) - _to_num(df["opp_FTr"])
    return pd.DataFrame(
        {
            "eFG_ODI": efg_odi,
            "TO_ODI": to_odi,
            "ORB_ODI": orb_odi,
            "FTR_ODI": ftr_odi,
        }
    )


def f_TC(df: pd.DataFrame) -> pd.DataFrame:
    out = _to_num(df["poss"]) - _to_num(df["opp_pre_poss_season"])
    return pd.DataFrame({"TC": out})


def f_TIN(df: pd.DataFrame) -> pd.DataFrame:
    out = (_to_num(df["poss"]) - _to_num(df["pre_poss_season"])).abs()
    return pd.DataFrame({"TIN": out})


def f_VOL(df: pd.DataFrame) -> pd.DataFrame:
    out = (_to_num(df["NetRtg"]) - _to_num(df["pre_NetRtg_season"])).abs()
    return pd.DataFrame({"VOL": out})


def f_DPC(df: pd.DataFrame) -> pd.DataFrame:
    bm = _to_num(df["bench_minutes_share"]).fillna(0.0)
    rb = _to_num(df["REB_rate_bench"]).fillna(0.0)
    tsb = _to_num(df["TS_bench"]).fillna(0.0)
    out = (0.40 * bm) + (0.30 * rb) + (0.30 * tsb)
    return pd.DataFrame({"DPC": out})


def f_FFC(df: pd.DataFrame) -> pd.DataFrame:
    efg = _to_num(df["eFG"]).fillna(0.0)
    tov = _to_num(df["TOV%"]).fillna(0.0)
    orb = _to_num(df["ORB%"]).fillna(0.0)
    ftr = _to_num(df["FTr"]).fillna(0.0)
    out = (0.40 * efg) - (0.25 * tov) + (0.20 * orb) + (0.15 * ftr)
    return pd.DataFrame({"FFC": out})


def f_PXP(df: pd.DataFrame) -> pd.DataFrame:
    out = _to_num(df["NetRtg"]) - _to_num(df["pre_NetRtg_season"])
    return pd.DataFrame({"PXP": out})


def f_SCI(df: pd.DataFrame) -> pd.DataFrame:
    z = within_season_zscore(
        df,
        value_column="ODI",
        season_column="season",
        date_column="game_datetime_utc",
        group_columns=("team_id", "season"),
    )
    return pd.DataFrame({"SCI": z.abs()})


def f_ODI_A(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"ODI_A": _to_num(df["ODI_A"])})


def f_ODI_B(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"ODI_B": _to_num(df["ODI_B"])})


def f_PEI_matchup(df: pd.DataFrame) -> pd.DataFrame:
    out = _to_num(df["PEQ_A"]) - _to_num(df["PEQ_B"])
    return pd.DataFrame({"PEI_matchup": out})


def f_POSW_matchup(df: pd.DataFrame) -> pd.DataFrame:
    out = _to_num(df["POSW_A"]) - _to_num(df["POSW_B"])
    return pd.DataFrame({"POSW": out})


def f_odi_diff(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"ODI_diff": _to_num(df["ODI_A"]) - _to_num(df["ODI_B"])})


def f_odi_sum(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"ODI_sum": _to_num(df["ODI_A"]) + _to_num(df["ODI_B"])})


def f_MTI(df: pd.DataFrame) -> pd.DataFrame:
    out = (_to_num(df["ANE_A"]).abs() + _to_num(df["ANE_B"]).abs()) / 2.0
    return pd.DataFrame({"MTI": out})


def f_SCI_matchup(df: pd.DataFrame) -> pd.DataFrame:
    out = (_to_num(df["SCI_A"]) + _to_num(df["SCI_B"])) / 2.0
    return pd.DataFrame({"SCI": out})


def f_factor_ODIs_AB(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eFG_ODI_A": _to_num(df["eFG_ODI_A"]),
            "eFG_ODI_B": _to_num(df["eFG_ODI_B"]),
            "TO_ODI_A": _to_num(df["TO_ODI_A"]),
            "TO_ODI_B": _to_num(df["TO_ODI_B"]),
            "ORB_ODI_A": _to_num(df["ORB_ODI_A"]),
            "ORB_ODI_B": _to_num(df["ORB_ODI_B"]),
            "FTR_ODI_A": _to_num(df["FTR_ODI_A"]),
            "FTR_ODI_B": _to_num(df["FTR_ODI_B"]),
        }
    )


def f_factor_ODIs_diffs_sums(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "eFG_ODI_diff": _to_num(df["eFG_ODI_A"]) - _to_num(df["eFG_ODI_B"]),
            "eFG_ODI_sum": _to_num(df["eFG_ODI_A"]) + _to_num(df["eFG_ODI_B"]),
            "TO_ODI_diff": _to_num(df["TO_ODI_A"]) - _to_num(df["TO_ODI_B"]),
            "TO_ODI_sum": _to_num(df["TO_ODI_A"]) + _to_num(df["TO_ODI_B"]),
            "ORB_ODI_diff": _to_num(df["ORB_ODI_A"]) - _to_num(df["ORB_ODI_B"]),
            "ORB_ODI_sum": _to_num(df["ORB_ODI_A"]) + _to_num(df["ORB_ODI_B"]),
            "FTR_ODI_diff": _to_num(df["FTR_ODI_A"]) - _to_num(df["FTR_ODI_B"]),
            "FTR_ODI_sum": _to_num(df["FTR_ODI_A"]) + _to_num(df["FTR_ODI_B"]),
        }
    )
