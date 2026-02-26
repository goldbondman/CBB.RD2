from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlayerOverlayConfig:
    """Configuration for player-boxscore-derived matchup overlay adjustments."""

    # mapping from interaction terms to pts/100
    k_ft_mean: float = 9.0
    k_to_mean: float = 16.0
    k_exec_mean: float = 7.0

    # pace handling
    default_pace: float = 68.5
    pace_cap: float = 80.0

    # variance mapping
    k_total_std_3pfrag: float = 3.0
    k_total_std_ft: float = 0.9
    k_total_std_to: float = 2.2
    k_spread_std_exec: float = 1.7
    k_spread_std_to: float = 1.4

    # output caps
    max_total_adj_pts: float = 10.0
    max_spread_adj_pts: float = 8.0
    max_total_std_adj: float = 8.0
    max_spread_std_adj: float = 6.0
    eps: float = 1e-9


CORE_FEATURE_COLS = [
    "rot_efg_l5",
    "rot_to_rate_l5",
    "rot_ftrate_l5",
    "rot_3par_l10",
    "rot_stocks_per40_l10",
    "rot_pf_per40_l5",
    "rot_minshare_sd",
    "rot_3p_pct_sd",
    "rot_to_rate_sd",
    "top2_pused_share",
    "top2_to_rate",
    "closer_ft_pct",
]


def _clip_series(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lo, hi)


def _is_home_series(df: pd.DataFrame) -> pd.Series:
    if "is_home" not in df.columns:
        return pd.Series(False, index=df.index)
    raw = df["is_home"]
    if raw.dtype == bool:
        return raw.fillna(False)
    return raw.astype(str).str.strip().str.lower().isin({"1", "true", "t", "yes", "y", "home"})


def _ensure_columns(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = 0.0
    return out


def _compute_terms(df: pd.DataFrame, cfg: PlayerOverlayConfig) -> pd.DataFrame:
    data = _ensure_columns(df, CORE_FEATURE_COLS + [f"opp_{c}" for c in CORE_FEATURE_COLS])

    top2_share = _clip_series(data["top2_pused_share"], 0.0, 1.0)
    top2_to = _clip_series(data["top2_to_rate"], 0.0, 0.6)
    opp_stocks = _clip_series(data["opp_rot_stocks_per40_l10"], 0.0, 20.0)
    min_sd = _clip_series(data["rot_minshare_sd"], 0.0, 0.5)
    opp_min_sd = _clip_series(data["opp_rot_minshare_sd"], 0.0, 0.5)

    ftrate = _clip_series(data["rot_ftrate_l5"], 0.0, 1.5)
    opp_pf = _clip_series(data["opp_rot_pf_per40_l5"], 0.0, 15.0)

    par3 = _clip_series(data["rot_3par_l10"], 0.0, 1.2)
    pct3_sd = _clip_series(data["rot_3p_pct_sd"], 0.0, 0.3)

    to_sd = _clip_series(data["rot_to_rate_sd"], 0.0, 0.3)

    to_swing = (top2_share * top2_to) * opp_stocks * (1.0 + 0.5 * min_sd)
    ft_volume = ftrate * opp_pf
    exec_tax = min_sd * opp_stocks
    three_pt_fragility = par3 * pct3_sd * (1.0 + 0.3 * opp_stocks) * (1.0 + 0.2 * opp_min_sd)

    to_swing = to_swing.clip(0.0, 4.0)
    ft_volume = ft_volume.clip(0.0, 6.0)
    exec_tax = exec_tax.clip(0.0, 6.0)
    three_pt_fragility = three_pt_fragility.clip(0.0, 3.0)

    off_eff_adj_100 = cfg.k_ft_mean * ft_volume - cfg.k_to_mean * to_swing - cfg.k_exec_mean * exec_tax
    off_eff_adj_100 = off_eff_adj_100.clip(-15.0, 15.0)

    out = data.copy()
    out["to_swing"] = to_swing
    out["ft_volume"] = ft_volume
    out["exec_tax"] = exec_tax
    out["three_pt_fragility"] = three_pt_fragility
    out["to_sd"] = to_sd
    out["off_eff_adj_100"] = off_eff_adj_100
    return out


def build_player_overlay_predictions(
    upcoming_team_perspective_df: pd.DataFrame,
    cfg: PlayerOverlayConfig | None = None,
) -> pd.DataFrame:
    """Build independent game-level overlay predictions from team-perspective rows.

    Sign convention for spread output:
      + predicted_spread_adjustment_pts_home favors home team.
    """
    if cfg is None:
        cfg = PlayerOverlayConfig()

    if upcoming_team_perspective_df is None or upcoming_team_perspective_df.empty:
        return pd.DataFrame()

    df = _compute_terms(upcoming_team_perspective_df, cfg)
    df["is_home"] = _is_home_series(df)

    if "expected_pace" in df.columns:
        pace_raw = pd.to_numeric(df["expected_pace"], errors="coerce")
        pace = pace_raw.fillna(cfg.default_pace).clip(40.0, cfg.pace_cap)
    else:
        pace = pd.Series(cfg.default_pace, index=df.index)
    df["pace_used"] = pace

    df["off_adj_pts"] = df["off_eff_adj_100"] * (df["pace_used"] / 100.0)

    game_rows = []
    for game_id, grp in df.groupby("game_id", dropna=False):
        home = grp[grp["is_home"]]
        away = grp[~grp["is_home"]]

        if home.empty and len(grp) >= 1:
            home = grp.iloc[[0]]
        if away.empty:
            away = grp.loc[~grp.index.isin(home.index)]
            if away.empty and len(grp) >= 2:
                away = grp.iloc[[1]]

        home_row = home.iloc[0]
        away_row = away.iloc[0] if not away.empty else home_row

        pace_used = float(np.clip(home_row.get("pace_used", cfg.default_pace), 40.0, cfg.pace_cap))
        home_off_adj_pts = float(home_row.get("off_eff_adj_100", 0.0) * (pace_used / 100.0))
        away_off_adj_pts = float(away_row.get("off_eff_adj_100", 0.0) * (pace_used / 100.0))

        total_adj = float(np.clip(home_off_adj_pts + away_off_adj_pts, -cfg.max_total_adj_pts, cfg.max_total_adj_pts))
        spread_adj_home = float(np.clip(home_off_adj_pts - away_off_adj_pts, -cfg.max_spread_adj_pts, cfg.max_spread_adj_pts))

        total_std = (
            cfg.k_total_std_3pfrag * (float(home_row.get("three_pt_fragility", 0.0)) + float(away_row.get("three_pt_fragility", 0.0)))
            + cfg.k_total_std_ft * (float(home_row.get("ft_volume", 0.0)) + float(away_row.get("ft_volume", 0.0)))
            + cfg.k_total_std_to * (float(home_row.get("to_sd", 0.0)) + float(away_row.get("to_sd", 0.0)))
        )
        total_std = float(np.clip(total_std, 0.0, cfg.max_total_std_adj))

        spread_std = (
            cfg.k_spread_std_exec * (float(home_row.get("exec_tax", 0.0)) + float(away_row.get("exec_tax", 0.0)))
            + cfg.k_spread_std_to * (float(home_row.get("to_sd", 0.0)) + float(away_row.get("to_sd", 0.0)))
        )
        spread_std = float(np.clip(spread_std, 0.0, cfg.max_spread_std_adj))

        confidence_adjustment = -float(np.clip((0.65 * total_std + 0.35 * spread_std) / 10.0, 0.0, 1.0))

        season = home_row.get("season", away_row.get("season", np.nan))
        game_date = home_row.get("game_date", away_row.get("game_date", np.nan))

        game_rows.append(
            {
                "season": season,
                "game_id": game_id,
                "game_date": game_date,
                "pace_used": pace_used,
                "predicted_total_adjustment_pts": total_adj,
                "predicted_spread_adjustment_pts_home": spread_adj_home,
                "total_std_adjustment": total_std,
                "spread_std_adjustment": spread_std,
                "confidence_adjustment": confidence_adjustment,
                "home_team_id": home_row.get("team_id", np.nan),
                "away_team_id": away_row.get("team_id", np.nan),
                "home_to_swing": float(home_row.get("to_swing", 0.0)),
                "away_to_swing": float(away_row.get("to_swing", 0.0)),
                "home_ft_volume": float(home_row.get("ft_volume", 0.0)),
                "away_ft_volume": float(away_row.get("ft_volume", 0.0)),
                "home_exec_tax": float(home_row.get("exec_tax", 0.0)),
                "away_exec_tax": float(away_row.get("exec_tax", 0.0)),
                "home_three_pt_fragility": float(home_row.get("three_pt_fragility", 0.0)),
                "away_three_pt_fragility": float(away_row.get("three_pt_fragility", 0.0)),
                "home_off_eff_adj_100": float(home_row.get("off_eff_adj_100", 0.0)),
                "away_off_eff_adj_100": float(away_row.get("off_eff_adj_100", 0.0)),
            }
        )

    games_df = pd.DataFrame(game_rows)
    out = df.merge(games_df, on="game_id", how="left", suffixes=("", "_game"))

    out["predicted_spread_adjustment_pts_team"] = np.where(
        out["is_home"],
        out["predicted_spread_adjustment_pts_home"],
        -out["predicted_spread_adjustment_pts_home"],
    )
    out["predicted_spread_adjustment_pts_home"] = np.where(
        out["is_home"],
        out["predicted_spread_adjustment_pts_home"],
        np.nan,
    )
    return out
