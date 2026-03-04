#!/usr/bin/env python3
"""
backtesting/compute_metrics.py
===============================
Metric Library — Step 1 of the CBB backtesting pipeline.

Computes all 13 named metrics + Lx rolling variants from existing
team_game_metrics.csv and team_game_weighted.csv data files.

Outputs:
  data/team_game_metrics_advanced.csv  — per-team-per-game with all metrics
  data/matchup_metrics.csv             — per-game matchup-level _diff features

Metric definitions
------------------
ANE  Adjusted Net Efficiency         = net_rtg rolling (raw off-def efficiency)
SVI  Schedule-adj Victory Index      = quality-weighted win rate rolling
PEQ  Performance Efficiency Quotient = adj_net_rtg rolling (SOS-adjusted)
WL   Win-Loss Rate                   = win% rolling
DPC  Defensive Performance Composite = inverted drtg rolling
FFC  Four Factors Composite          = Oliver-weighted four factors rolling
PXP  Performance vs Expected         = win − pythagorean_win_pct rolling
ODI  Opponent-adj Differential Index = perf_vs_exp_net rolling
eFG_ODI  efg vs opp baseline         = efg_vs_opp (from weighted CSV)
TO_ODI   tov vs opp (inverted)       = −tov_vs_opp (lower TOV = better)
ORB_ODI  orb vs opp                  = orb_vs_opp
FTR_ODI  ftr vs opp                  = ftr_vs_opp
MTI  Matchup Tension Index           = avg abs adj_net_rtg both teams (game-level)
SCI  Signal Confidence Index         = abs(cover_rate_l10 − 0.5) per team
"""

from __future__ import annotations

import pathlib
import warnings
from typing import List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DATA = pathlib.Path("data")
OUT_ADVANCED = DATA / "team_game_metrics_advanced.csv"
OUT_MATCHUP  = DATA / "matchup_metrics.csv"

# Lx windows to compute
LX_WINDOWS = [4, 7, 10, 12]

# League mean baselines (from observed data)
EFG_MEAN  = 51.6   # efg_pct percentage scale
TOV_MEAN  = 17.1   # tov_pct percentage scale
ORB_MEAN  = 30.2   # orb_pct percentage scale
FTR_MEAN  = 35.1   # ftr percentage scale
DRTG_MEAN = 109.4  # defensive rating


# ─────────────────────────────────────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────────────────────────────────────

def load_sources() -> tuple[pd.DataFrame, pd.DataFrame]:
    tgm_path = DATA / "team_game_metrics.csv"
    tgw_path = DATA / "team_game_weighted.csv"
    for p in (tgm_path, tgw_path):
        if not p.exists():
            raise FileNotFoundError(f"Required source not found: {p}")

    tgm = pd.read_csv(tgm_path, low_memory=False)
    tgw = pd.read_csv(tgw_path, low_memory=False)

    print(f"[load] team_game_metrics:  {len(tgm):,} rows, {len(tgm.columns)} cols")
    print(f"[load] team_game_weighted: {len(tgw):,} rows, {len(tgw.columns)} cols")

    # Confirm join uniqueness
    tgm_dupes = tgm.duplicated(subset=["event_id", "team_id"]).sum()
    tgw_dupes = tgw.duplicated(subset=["event_id", "team_id"]).sum()
    if tgm_dupes or tgw_dupes:
        print(f"[warn] Duplicate (event_id, team_id) rows: tgm={tgm_dupes}, tgw={tgw_dupes}")

    return tgm, tgw


# ─────────────────────────────────────────────────────────────────────────────
# Merge sources
# ─────────────────────────────────────────────────────────────────────────────

def merge_sources(tgm: pd.DataFrame, tgw: pd.DataFrame) -> pd.DataFrame:
    """
    Merge tgm + tgw on (event_id, team_id).
    Pull only the columns we need from tgw to avoid column explosion.
    """
    tgw_cols = [
        "event_id", "team_id",
        "adj_net_rtg", "adj_ortg", "adj_drtg",
        "perf_vs_exp_net", "perf_vs_exp_ortg", "perf_vs_exp_def",
        "net_rtg_wtd_qual_l5", "net_rtg_wtd_qual_l10",
        "efg_vs_opp_l10", "tov_vs_opp_l10", "orb_vs_opp_l10", "ftr_vs_opp_l10",
        "efg_vs_opp_season", "tov_vs_opp_season", "orb_vs_opp_season", "ftr_vs_opp_season",
        "momentum_score", "form_rating",
    ]
    tgw_use = tgw[[c for c in tgw_cols if c in tgw.columns]].copy()

    df = tgm.merge(tgw_use, on=["event_id", "team_id"], how="left")

    # Parse datetime
    df["game_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")

    # Opponent adj_net_rtg lookup (for SVI quality weight)
    if "opponent_id" in df.columns and "adj_net_rtg" in df.columns:
        adj_lookup = (
            df[["event_id", "team_id", "adj_net_rtg"]]
            .rename(columns={"team_id": "opponent_id", "adj_net_rtg": "opp_adj_net_rtg"})
        )
        df = df.merge(adj_lookup, on=["event_id", "opponent_id"], how="left")
    else:
        df["opp_adj_net_rtg"] = np.nan

    df = df.copy()  # defragment
    print(f"[merge] Combined shape: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Per-game base values
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_game_bases(df: pd.DataFrame) -> pd.DataFrame:
    """Compute scalar per-game base value for each metric (before rolling)."""

    # ── ANE: raw net efficiency
    df["ane_g"] = df["net_rtg"].astype(float)

    # ── SVI: schedule-adjusted victory index
    # quality_factor: normalised opp adj_net_rtg → [0, 2] scale
    #   opp_adj_net_rtg clipped to [-30, 30] → quality [0, 2]
    opp_q = df["opp_adj_net_rtg"].clip(-30, 30).fillna(0)
    df["svi_g"] = df["win"].astype(float) * (1 + opp_q / 30)
    # range: 0 (loss vs anyone) → ~2 (win vs adj_net=+30 team)

    # ── PEQ: performance efficiency quotient = SOS-adj net rating per game
    df["peq_g"] = df["adj_net_rtg"].astype(float)

    # ── WL: raw win indicator
    df["wl_g"] = df["win"].astype(float)

    # ── DPC: defensive performance composite (inverted drtg, centered)
    # higher = better defense; unit = pts/100 above/below league avg
    df["dpc_g"] = DRTG_MEAN - df["drtg"].astype(float)

    # ── FFC: four factors composite (Oliver weights, percentage-point units, centered)
    df["ffc_g"] = (
        0.40 * (df["efg_pct"].astype(float) - EFG_MEAN)
        + 0.25 * (TOV_MEAN - df["tov_pct"].astype(float))   # lower TOV = better
        + 0.20 * (df["orb_pct"].astype(float) - ORB_MEAN)
        + 0.15 * (df["ftr"].astype(float).clip(upper=60.0) - FTR_MEAN)
    )
    # range: ~-15 to +15 pts (composite advantage vs average)

    # ── PXP: performance vs pythagorean expectation
    df["pxp_g"] = (
        df["win"].astype(float) - df["pythagorean_win_pct"].astype(float)
    )
    # range: -1 to +1 per game; rolling avg: -0.5 to +0.5

    # ── ODI: opponent-adjusted differential (perf vs expected net)
    df["odi_g"] = df["perf_vs_exp_net"].astype(float)
    # range: very noisy per game (±100); rolling avg ±20

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Rolling Lx windows
# ─────────────────────────────────────────────────────────────────────────────

BASE_METRICS = {
    "ANE": "ane_g",
    "SVI": "svi_g",
    "PEQ": "peq_g",
    "WL":  "wl_g",
    "DPC": "dpc_g",
    "FFC": "ffc_g",
    "PXP": "pxp_g",
    "ODI": "odi_g",
}


def compute_rolling_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each team, sort by game_dt and compute rolling(L).mean() for each metric.
    Also compute:
      - {metric}_season  = expanding mean (season-to-date)
      - ANE_trend_short  = ANE_L4 - ANE_L10
      - ANE_trend_med    = ANE_L7 - ANE_L12
      - DPC_L10_std      = rolling(10).std(dpc_g)
      - DPC_trend        = DPC_L4 - DPC_L10
    """
    df = df.sort_values(["team_id", "game_dt"]).copy()

    grp = df.groupby("team_id", sort=False)

    for metric_name, base_col in BASE_METRICS.items():
        if base_col not in df.columns:
            print(f"[warn] Base column '{base_col}' missing — skipping {metric_name}")
            continue

        series = grp[base_col]

        # Season (expanding mean)
        col_season = f"{metric_name}_season"
        df[col_season] = series.transform(lambda s: s.expanding(min_periods=1).mean())

        # Rolling Lx windows
        for L in LX_WINDOWS:
            col_lx = f"{metric_name}_L{L}"
            df[col_lx] = series.transform(
                lambda s, _L=L: s.rolling(_L, min_periods=max(1, _L // 2)).mean()
            )

    # ── ANE trend signals
    if "ANE_L4" in df.columns and "ANE_L10" in df.columns:
        df["ANE_trend_short"] = df["ANE_L4"] - df["ANE_L10"]
    if "ANE_L7" in df.columns and "ANE_L12" in df.columns:
        df["ANE_trend_med"]   = df["ANE_L7"] - df["ANE_L12"]

    # ── DPC std and trend
    if "dpc_g" in df.columns:
        dpc_grp = df.groupby("team_id", sort=False)["dpc_g"]
        df["DPC_L10_std"] = dpc_grp.transform(
            lambda s: s.rolling(10, min_periods=5).std()
        )
        if "DPC_L4" in df.columns and "DPC_L10" in df.columns:
            df["DPC_trend"] = df["DPC_L4"] - df["DPC_L10"]

    # ── SCI: signal confidence index = |win_rate_l10 - 0.5|
    # cover_rate_l10 is present but 100% null in current data; use win_l10 (0-1 rate)
    cr = df.get("cover_rate_l10")
    if cr is not None and cr.notna().any():
        df["SCI"] = (cr - 0.5).abs()
    elif "win_l10" in df.columns:
        df["SCI"] = (df["win_l10"] - 0.5).abs()  # win_l10 is already 0-1 rate
    else:
        df["SCI"] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Four-factor opponent-adjusted (eFG/TO/ORB/FTR ODI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_four_factor_odi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map existing vs_opp columns from team_game_weighted to the plan's names.
    Direction 'A': higher = Team A advantage.
    TO_ODI is inverted (lower turnover rate vs what opponent forces = good).
    """
    # Season-level (from tgw)
    if "efg_vs_opp_season" in df.columns:
        df["eFG_ODI"] = df["efg_vs_opp_season"].astype(float)
    elif "efg_pct" in df.columns:
        df["eFG_ODI"] = df["efg_pct"] - EFG_MEAN

    if "tov_vs_opp_season" in df.columns:
        # tov_vs_opp = team_tov − opp_avg_forced; negative = team turns it over less = good
        # Invert so higher = better (direction A)
        df["TO_ODI"] = -df["tov_vs_opp_season"].astype(float)
    elif "tov_pct" in df.columns:
        df["TO_ODI"] = TOV_MEAN - df["tov_pct"]  # higher = less turnovers than avg

    if "orb_vs_opp_season" in df.columns:
        df["ORB_ODI"] = df["orb_vs_opp_season"].astype(float)
    elif "orb_pct" in df.columns:
        df["ORB_ODI"] = df["orb_pct"] - ORB_MEAN

    if "ftr_vs_opp_season" in df.columns:
        df["FTR_ODI"] = df["ftr_vs_opp_season"].astype(float)
    elif "ftr" in df.columns:
        df["FTR_ODI"] = df["ftr"] - FTR_MEAN

    # L10 variants (primary use in backtester)
    if "efg_vs_opp_l10" in df.columns:
        df["eFG_ODI_L10"] = df["efg_vs_opp_l10"].astype(float)
    if "tov_vs_opp_l10" in df.columns:
        df["TO_ODI_L10"] = -df["tov_vs_opp_l10"].astype(float)
    if "orb_vs_opp_l10" in df.columns:
        df["ORB_ODI_L10"] = df["orb_vs_opp_l10"].astype(float)
    if "ftr_vs_opp_l10" in df.columns:
        df["FTR_ODI_L10"] = df["ftr_vs_opp_l10"].astype(float)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Build matchup_metrics.csv (game-level, home=A, away=B)
# ─────────────────────────────────────────────────────────────────────────────

DIFF_METRICS: List[str] = [
    "ANE_season", "ANE_L4", "ANE_L7", "ANE_L10", "ANE_L12",
    "ANE_trend_short", "ANE_trend_med",
    "SVI_season", "SVI_L4", "SVI_L7", "SVI_L10", "SVI_L12",
    "PEQ_season", "PEQ_L4", "PEQ_L7", "PEQ_L10", "PEQ_L12",
    "WL_season",  "WL_L4",  "WL_L7",  "WL_L10",  "WL_L12",
    "DPC_season", "DPC_L4", "DPC_L7", "DPC_L10", "DPC_L12",
    "DPC_trend",
    "FFC_season", "FFC_L4", "FFC_L7", "FFC_L10", "FFC_L12",
    "PXP_season", "PXP_L4", "PXP_L7", "PXP_L10", "PXP_L12",
    "ODI_season", "ODI_L4", "ODI_L7", "ODI_L10", "ODI_L12",
    "eFG_ODI", "TO_ODI", "ORB_ODI", "FTR_ODI",
    "eFG_ODI_L10", "TO_ODI_L10", "ORB_ODI_L10", "FTR_ODI_L10",
    "SCI",
]

SUM_METRICS: List[str] = ["DPC_season", "DPC_L10", "ODI_season", "ODI_L10"]


def build_matchup_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot per-team-per-game rows into one row per game.
    home_away='home' → Team A, 'away' → Team B.
    Compute _diff (A - B) and _sum (A + B) columns.
    Add MTI (game-level) and rest flags.
    """
    if "home_away" not in df.columns:
        raise ValueError("home_away column required for matchup pivot")

    home = df[df["home_away"] == "home"].copy()
    away = df[df["home_away"] == "away"].copy()

    # Shared game-level columns (take from home row)
    game_cols = [
        "event_id", "game_datetime_utc", "game_dt",
        "home_team", "away_team", "home_team_id", "away_team_id",
        "spread", "over_under", "neutral_site",
        "completed", "win",   # home win = 1, home loss = 0 → actual outcome
        "margin", "cover", "ats_push",
    ]
    game_cols = [c for c in game_cols if c in home.columns]

    matchup = home[game_cols].copy()
    matchup = matchup.rename(columns={"win": "home_win", "margin": "home_margin",
                                       "cover": "home_cover"})

    # Index both sides for fast column extraction
    home_idx = home.set_index("event_id")
    away_idx = away.set_index("event_id")

    available_diff = [m for m in DIFF_METRICS if m in df.columns]
    available_sum  = [m for m in SUM_METRICS  if m in df.columns]

    for metric in available_diff:
        h_vals = home_idx[metric].reindex(matchup["event_id"]).values
        a_vals = away_idx[metric].reindex(matchup["event_id"]).values
        matchup[f"{metric}_diff"] = h_vals - a_vals

    for metric in available_sum:
        h_vals = home_idx[metric].reindex(matchup["event_id"]).values
        a_vals = away_idx[metric].reindex(matchup["event_id"]).values
        matchup[f"{metric}_sum"] = h_vals + a_vals

    # MTI: Matchup Tension Index = avg of abs(adj_net_rtg) for both teams
    if "adj_net_rtg" in df.columns:
        h_anr = home_idx["adj_net_rtg"].reindex(matchup["event_id"]).values
        a_anr = away_idx["adj_net_rtg"].reindex(matchup["event_id"]).values
        matchup["MTI"] = (np.abs(h_anr) + np.abs(a_anr)) / 2.0

    # Rest flags (from rest_days per team)
    if "rest_days" in df.columns:
        h_rest = home_idx["rest_days"].reindex(matchup["event_id"]).values
        a_rest = away_idx["rest_days"].reindex(matchup["event_id"]).values
        matchup["rest_days_A"] = h_rest
        matchup["rest_days_B"] = a_rest
        matchup["short_rest_A"]    = (h_rest <= 4).astype(int)
        matchup["extended_rest_A"] = (h_rest >= 7).astype(int)
        matchup["short_rest_B"]    = (a_rest <= 4).astype(int)
        matchup["extended_rest_B"] = (a_rest >= 7).astype(int)
        matchup["rest_advantage"]    = ((h_rest >= 7) & (a_rest <= 4)).astype(int)
        matchup["rest_disadvantage"] = ((h_rest <= 4) & (a_rest >= 7)).astype(int)

    # SCI per team (magnitude filter — use home team's SCI as primary)
    if "SCI" in df.columns:
        matchup["SCI_A"] = home_idx["SCI"].reindex(matchup["event_id"]).values
        matchup["SCI_B"] = away_idx["SCI"].reindex(matchup["event_id"]).values
        matchup["SCI"]   = matchup[["SCI_A", "SCI_B"]].mean(axis=1)

    # Sort by date
    matchup = matchup.sort_values("game_dt").reset_index(drop=True)

    return matchup


# ─────────────────────────────────────────────────────────────────────────────
# Summary / schema validation
# ─────────────────────────────────────────────────────────────────────────────

def print_schema_summary(df: pd.DataFrame, label: str) -> None:
    new_metric_cols = [c for c in df.columns if any(
        c.startswith(m) for m in list(BASE_METRICS.keys()) + ["eFG_ODI", "TO_ODI", "ORB_ODI", "FTR_ODI", "MTI", "SCI"]
    )]
    null_rates = df[new_metric_cols].isna().mean().round(3)
    high_null = null_rates[null_rates > 0.3]

    print(f"\n[schema] {label}")
    print(f"  Rows: {len(df):,}  |  Metric cols added: {len(new_metric_cols)}")
    if len(high_null):
        print(f"  High-null cols (>30% missing):")
        for col, rate in high_null.items():
            print(f"    {col}: {rate:.1%}")
    else:
        print("  All metric cols have <30% null rate — OK")

    # Sample one team's metrics to sanity-check
    if "team" in df.columns:
        sample_team = df[df["team"].str.contains("Duke", na=False)].sort_values("game_dt").tail(3)
        if not sample_team.empty and "ANE_L10" in df.columns:
            print(f"\n  Sample (Duke, last 3 games):")
            show = ["game_dt", "ANE_L10", "SVI_L10", "WL_L10", "DPC_L10", "FFC_L10", "PXP_L10", "ODI_L10"]
            show = [c for c in show if c in sample_team.columns]
            print(sample_team[show].to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run(write_files: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full metric computation pipeline.
    Returns (team_level_df, matchup_level_df).
    """
    print("=" * 60)
    print("METRIC LIBRARY — compute_metrics.py")
    print("=" * 60)

    # Step 1: Load
    tgm, tgw = load_sources()

    # Step 2: Merge
    df = merge_sources(tgm, tgw)

    # Step 3: Per-game base values
    df = compute_per_game_bases(df)

    # Step 4: Rolling Lx windows
    df = compute_rolling_windows(df)

    # Step 5: Four-factor opponent-adjusted ODIs
    df = compute_four_factor_odi(df)

    # Step 6: Schema summary
    print_schema_summary(df, "team_game_metrics_advanced.csv")

    # Step 7: Build matchup-level table
    matchup = build_matchup_metrics(df)
    print(f"\n[matchup] matchup_metrics.csv: {len(matchup):,} rows, {len(matchup.columns)} cols")

    # Confirm outcome column
    if "home_win" in matchup.columns:
        home_win_rate = matchup["home_win"].mean()
        print(f"  Home win rate: {home_win_rate:.3f}  (expected ~0.56 for CBB)")
        null_outcomes = matchup["home_win"].isna().sum()
        print(f"  Null outcomes: {null_outcomes}")

    # Step 8: Write
    if write_files:
        df.to_csv(OUT_ADVANCED, index=False)
        matchup.to_csv(OUT_MATCHUP, index=False)
        print(f"\n[write] {OUT_ADVANCED} — {len(df):,} rows")
        print(f"[write] {OUT_MATCHUP} — {len(matchup):,} rows")

    print("\n[done] Metric library complete.")
    return df, matchup


if __name__ == "__main__":
    run()
