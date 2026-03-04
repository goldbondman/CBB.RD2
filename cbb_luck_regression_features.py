#!/usr/bin/env python3
"""Build per-team per-game luck and mean-reversion features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
TEAM_LOGS_PATH = DATA_DIR / "team_game_logs.csv"
TEAM_WEIGHTED_PATH = DATA_DIR / "team_game_weighted.csv"
GAMES_PATH = DATA_DIR / "games.csv"
OUTPUT_PATH = DATA_DIR / "luck_regression_features.csv"
PYTH_EXPONENT = 11.5


def normalize_game_id(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    text = text.str.lstrip("0")
    return text.mask(text == "", "0")


def to_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def rolling_prior(group: pd.DataFrame, value_col: str, window: int, min_periods: int) -> pd.Series:
    return group[value_col].shift(1).rolling(window=window, min_periods=min_periods).mean()


def zscore_prior_global(df: pd.DataFrame, col: str) -> pd.Series:
    base = pd.to_numeric(df[col], errors="coerce")
    prior = base.shift(1)
    mean = prior.expanding(min_periods=2).mean()
    std = prior.expanding(min_periods=2).std(ddof=0)
    z = (base - mean) / std.replace(0, np.nan)
    return z


def build_features(logs: pd.DataFrame, weighted: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    logs = logs.copy()
    weighted = weighted.copy()
    games = games.copy()

    logs["game_id"] = normalize_game_id(logs.get("game_id", logs.get("event_id")))
    weighted["game_id"] = normalize_game_id(weighted.get("game_id", weighted.get("event_id")))
    games["game_id"] = normalize_game_id(games.get("game_id", games.get("event_id")))

    for frame in [logs, weighted]:
        frame["game_datetime_utc"] = pd.to_datetime(frame["game_datetime_utc"], utc=True, errors="coerce")
        to_numeric(frame, ["team_id", "points_for", "points_against", "win", "tpm", "tpa", "ftm", "fta", "tov", "fga", "fgm"])

    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True, errors="coerce")
    games["season"] = pd.to_numeric(games.get("date", pd.NA).astype(str).str[:4], errors="coerce")

    base_cols = [
        "game_id",
        "team_id",
        "game_datetime_utc",
        "points_for",
        "points_against",
        "win",
        "tpm",
        "tpa",
        "ftm",
        "fta",
        "tov",
        "fga",
        "fgm",
    ]
    base = logs[[c for c in base_cols if c in logs.columns]].copy()

    weighted_cols = ["game_id", "team_id", "net_rtg", "efg_pct", "tov_pct", "ft_pct"]
    weighted_sub = weighted[[c for c in weighted_cols if c in weighted.columns]].copy()
    base = base.merge(weighted_sub, on=["game_id", "team_id"], how="left", suffixes=("", "_weighted"))

    base = base.merge(games[["game_id", "season"]], on="game_id", how="left")

    base["is_win"] = (base["points_for"] > base["points_against"]).astype(float)
    base["margin"] = base["points_for"] - base["points_against"]
    base["close_game"] = (base["margin"].abs() <= 5).astype(float)
    base["close_win"] = np.where(base["close_game"] == 1, base["is_win"], np.nan)

    base["team_3p_pct"] = np.where(base["tpa"] > 0, base["tpm"] / base["tpa"], np.nan)
    base["ft_pct"] = np.where(base["fta"] > 0, base["ftm"] / base["fta"], np.nan)
    base["efg_pct"] = np.where(base["fga"] > 0, (base["fgm"] + 0.5 * base["tpm"]) / base["fga"], base.get("efg_pct"))
    base["to_rate"] = np.where((base["fga"] + 0.44 * base["fta"] + base["tov"]) > 0, base["tov"] / (base["fga"] + 0.44 * base["fta"] + base["tov"]), np.nan)

    # Opponent 3P% via self-join on game_id (spec requirement).
    opp = logs[["game_id", "team_id", "tpm", "tpa"]].rename(
        columns={"team_id": "opp_team_id", "tpm": "opp_tpm", "tpa": "opp_tpa"}
    )
    pair = logs[["game_id", "team_id"]].merge(opp, on="game_id", how="left")
    pair = pair[pair["team_id"] != pair["opp_team_id"]].drop_duplicates(["game_id", "team_id"])
    pair["opp_3p_pct"] = np.where(pair["opp_tpa"] > 0, pair["opp_tpm"] / pair["opp_tpa"], np.nan)
    base = base.merge(pair[["game_id", "team_id", "opp_3p_pct"]], on=["game_id", "team_id"], how="left")

    valid_pyth = (base["points_for"] > 0) & (base["points_against"] > 0)
    base["pyth_win_prob"] = np.where(
        valid_pyth,
        (base["points_for"] ** PYTH_EXPONENT) / ((base["points_for"] ** PYTH_EXPONENT) + (base["points_against"] ** PYTH_EXPONENT)),
        np.nan,
    )

    base = base.sort_values(["team_id", "game_datetime_utc", "game_id"]).reset_index(drop=True)

    group = base.groupby("team_id", group_keys=False)
    prior_games = group.cumcount()
    base["games_played_prior"] = prior_games
    base["wins_prior"] = group["is_win"].cumsum() - base["is_win"]
    base["pyth_expected_wins"] = group["pyth_win_prob"].cumsum().shift(1)

    with np.errstate(invalid="ignore", divide="ignore"):
        base["pyth_actual_win_pct"] = np.where(base["games_played_prior"] > 0, base["wins_prior"] / base["games_played_prior"], np.nan)
        pyth_exp_pct = np.where(base["games_played_prior"] > 0, base["pyth_expected_wins"] / base["games_played_prior"], np.nan)
    base["luck_score"] = base["pyth_actual_win_pct"] - pyth_exp_pct

    base["actual_win_l10"] = group["is_win"].shift(1).rolling(10, min_periods=5).mean()
    base["pyth_win_l10"] = group["pyth_win_prob"].shift(1).rolling(10, min_periods=5).mean()
    base["luck_score_l10"] = base["actual_win_l10"] - base["pyth_win_l10"]

    base["team_3p_pct_season_avg"] = group["team_3p_pct"].shift(1).expanding(min_periods=5).mean().reset_index(level=0, drop=True)
    base["team_3p_pct_l5"] = group["team_3p_pct"].shift(1).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    base["three_pt_luck_l5"] = (base["team_3p_pct_l5"] - base["team_3p_pct_season_avg"]).clip(-0.15, 0.15)

    base["opp_3p_pct_season_avg"] = group["opp_3p_pct"].shift(1).expanding(min_periods=5).mean().reset_index(level=0, drop=True)
    base["opp_3p_pct_l5"] = group["opp_3p_pct"].shift(1).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    base["opp_three_pt_luck_l5"] = base["opp_3p_pct_l5"] - base["opp_3p_pct_season_avg"]

    close_games_prior = group["close_game"].shift(1).rolling(20, min_periods=10).sum().reset_index(level=0, drop=True)
    close_wins_prior = group["close_win"].shift(1).rolling(20, min_periods=10).sum().reset_index(level=0, drop=True)
    base["close_game_record_l20"] = np.where(close_games_prior >= 5, close_wins_prior / close_games_prior, np.nan)
    base["close_game_expected_rate"] = 0.5
    base["close_game_luck_l20"] = base["close_game_record_l20"] - 0.5

    base["net_rtg_season_avg"] = group["net_rtg"].shift(1).expanding(min_periods=5).mean().reset_index(level=0, drop=True)
    base["net_rtg_l5"] = group["net_rtg"].shift(1).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    base["net_rtg_trend"] = base["net_rtg_l5"] - base["net_rtg_season_avg"]

    base["efg_pct_season_avg"] = group["efg_pct"].shift(1).expanding(min_periods=5).mean().reset_index(level=0, drop=True)
    base["efg_pct_l5"] = group["efg_pct"].shift(1).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    base["efg_luck_l5"] = base["efg_pct_l5"] - base["efg_pct_season_avg"]

    base["to_rate_season_avg"] = group["to_rate"].shift(1).expanding(min_periods=5).mean().reset_index(level=0, drop=True)
    base["to_rate_l5"] = group["to_rate"].shift(1).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    base["to_rate_luck_l5"] = base["to_rate_l5"] - base["to_rate_season_avg"]

    base["ft_pct_season_avg"] = group["ft_pct"].shift(1).expanding(min_periods=5).mean().reset_index(level=0, drop=True)
    base["ft_pct_l5"] = group["ft_pct"].shift(1).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
    base["ft_luck_l5"] = base["ft_pct_l5"] - base["ft_pct_season_avg"]

    base = base.sort_values(["game_datetime_utc", "game_id", "team_id"]).reset_index(drop=True)
    for signal in ["luck_score", "three_pt_luck_l5", "opp_three_pt_luck_l5", "close_game_luck_l20", "ft_luck_l5"]:
        base[f"{signal}_normalized"] = zscore_prior_global(base, signal)

    base["composite_luck_score"] = (
        0.30 * base["luck_score_normalized"]
        + 0.25 * base["three_pt_luck_l5_normalized"]
        + 0.20 * base["opp_three_pt_luck_l5_normalized"]
        + 0.15 * base["close_game_luck_l20_normalized"]
        + 0.10 * base["ft_luck_l5_normalized"]
    )
    base["regression_candidate_flag"] = np.select(
        [base["composite_luck_score"] > 1.5, base["composite_luck_score"] < -1.5],
        [1, -1],
        default=0,
    )

    season_games = base["games_played_prior"] >= 5
    for col in [
        "luck_score",
        "team_3p_pct_season_avg",
        "opp_3p_pct_season_avg",
        "net_rtg_season_avg",
        "efg_pct_season_avg",
        "to_rate_season_avg",
        "ft_pct_season_avg",
    ]:
        base.loc[~season_games, col] = np.nan

    out_cols = [
        "game_id",
        "team_id",
        "game_datetime_utc",
        "season",
        "pyth_expected_wins",
        "pyth_actual_win_pct",
        "luck_score",
        "luck_score_l10",
        "team_3p_pct_season_avg",
        "team_3p_pct_l5",
        "three_pt_luck_l5",
        "opp_3p_pct_season_avg",
        "opp_3p_pct_l5",
        "opp_three_pt_luck_l5",
        "close_game_record_l20",
        "close_game_expected_rate",
        "close_game_luck_l20",
        "net_rtg_season_avg",
        "net_rtg_l5",
        "net_rtg_trend",
        "efg_pct_season_avg",
        "efg_pct_l5",
        "efg_luck_l5",
        "to_rate_season_avg",
        "to_rate_l5",
        "to_rate_luck_l5",
        "ft_pct_season_avg",
        "ft_pct_l5",
        "ft_luck_l5",
        "composite_luck_score",
        "regression_candidate_flag",
    ]
    return base[out_cols].copy()


def print_summary(df: pd.DataFrame) -> None:
    print(f"Total rows written: {len(df)}")
    luck = pd.to_numeric(df["luck_score"], errors="coerce")
    print(f"luck_score distribution: mean={luck.mean():.4f}, std={luck.std():.4f}")
    reg_pct = pd.to_numeric(df["regression_candidate_flag"], errors="coerce").abs().eq(1).mean() * 100
    print(f"regression_candidate_flag rate: {reg_pct:.2f}%")
    rolling_cols = [
        "luck_score",
        "luck_score_l10",
        "three_pt_luck_l5",
        "opp_three_pt_luck_l5",
        "close_game_luck_l20",
        "net_rtg_trend",
        "efg_luck_l5",
        "to_rate_luck_l5",
        "ft_luck_l5",
        "composite_luck_score",
    ]
    print("Non-null rates:")
    print((1 - df[rolling_cols].isna().mean()).round(3).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Build luck/regression features per team-game.")
    parser.add_argument("--season", type=int, help="Optional season filter for output.")
    parser.add_argument("--days-back", type=int, help="Optional trailing day filter for output.")
    args = parser.parse_args()

    logs = pd.read_csv(TEAM_LOGS_PATH, low_memory=False)
    weighted = pd.read_csv(TEAM_WEIGHTED_PATH, low_memory=False)
    games = pd.read_csv(GAMES_PATH, low_memory=False)

    features = build_features(logs=logs, weighted=weighted, games=games)

    if args.season is not None:
        features = features[pd.to_numeric(features["season"], errors="coerce") == args.season].copy()

    if args.days_back is not None and not features.empty:
        max_dt = pd.to_datetime(features["game_datetime_utc"], utc=True, errors="coerce").max()
        cutoff = max_dt - pd.Timedelta(days=args.days_back)
        features = features[pd.to_datetime(features["game_datetime_utc"], utc=True, errors="coerce") >= cutoff].copy()

    features = features.sort_values(["game_datetime_utc", "game_id", "team_id"]).reset_index(drop=True)
    features.to_csv(OUTPUT_PATH, index=False)

    print_summary(features)


if __name__ == "__main__":
    main()
# home/away splits
