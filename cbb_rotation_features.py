#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
LOGS_PATH = DATA_DIR / "player_game_logs.csv"
METRICS_PATH = DATA_DIR / "player_game_metrics.csv"
OUT_PATH = DATA_DIR / "rotation_features.csv"

ROLLING_COLS = [
    "rot_efg_l5",
    "rot_efg_l10",
    "rot_to_rate_l5",
    "rot_to_rate_l10",
    "rot_ftrate_l5",
    "rot_3par_l10",
    "rot_stocks_per40_l10",
    "rot_pf_per40_l5",
]


def normalize_game_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return s.str.lstrip("0").replace("", "0")


def derive_season(dt: pd.Series) -> pd.Series:
    d = pd.to_datetime(dt, utc=True, errors="coerce")
    return np.where(d.dt.month >= 8, d.dt.year + 1, d.dt.year)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace(0, np.nan)
    return num / den


def load_inputs() -> pd.DataFrame:
    logs = pd.read_csv(LOGS_PATH, low_memory=False)
    metrics = pd.read_csv(METRICS_PATH, low_memory=False)

    logs = logs.rename(columns={"event_id": "game_id"})
    metrics = metrics.rename(columns={"event_id": "game_id"})

    merge_cols = [
        "game_id",
        "team_id",
        "athlete_id",
        "usage_rate",
        "efg_pct",
        "three_pct",
        "ft_pct",
    ]
    m = metrics[[c for c in merge_cols if c in metrics.columns]].copy()

    df = logs.merge(m, on=["game_id", "team_id", "athlete_id"], how="left", suffixes=("", "_m"))
    for col in ["min", "fgm", "fga", "tpm", "tpa", "fta", "ftm", "tov", "stl", "blk", "pf", "pts", "usage_rate"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["game_id"] = normalize_game_id(df["game_id"])
    df["team_id"] = df["team_id"].astype(str)
    df["athlete_id"] = df["athlete_id"].astype(str)
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df[df["game_datetime_utc"].notna()].copy()
    df["season"] = derive_season(df["game_datetime_utc"])

    # player game metrics from boxscore
    df["efg"] = _safe_div(df["fgm"] + 0.5 * df["tpm"], df["fga"])
    poss = df["fga"] + 0.44 * df["fta"] + df["tov"]
    df["to_rate"] = _safe_div(df["tov"], poss)
    df["ftrate"] = _safe_div(df["fta"], df["fga"])
    df["three_par"] = _safe_div(df["tpa"], df["fga"])
    df["stocks_per40"] = _safe_div(df["stl"] + df["blk"], df["min"]) * 40.0
    df["pf_per40"] = _safe_div(df["pf"], df["min"]) * 40.0
    df["three_pct_game"] = _safe_div(df["tpm"], df["tpa"])

    usage = pd.to_numeric(df.get("usage_rate"), errors="coerce")
    df["usage_frac"] = usage.where(usage <= 1, usage / 100.0).clip(lower=0, upper=1)
    return df


def add_player_rolling_std(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["athlete_id", "game_datetime_utc", "game_id"]).copy()
    g = df.groupby("athlete_id", group_keys=False)
    df["three_pct_std_l5_prior"] = g["three_pct_game"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).std())
    df["to_rate_std_l5_prior"] = g["to_rate"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).std())
    return df


def build_team_game_rotation(df: pd.DataFrame) -> pd.DataFrame:
    work = df[df["min"].fillna(0) > 0].copy()
    work = work.sort_values(["game_id", "team_id", "min", "athlete_id"], ascending=[True, True, False, True])
    work["rot_rank"] = work.groupby(["game_id", "team_id"]).cumcount() + 1
    rot = work[work["rot_rank"] <= 7].copy()

    rot_tot_min = rot.groupby(["game_id", "team_id"])["min"].transform("sum")
    rot["rot_weight"] = _safe_div(rot["min"], rot_tot_min)

    top2u = rot.sort_values(["game_id", "team_id", "usage_frac"], ascending=[True, True, False]).copy()
    top2u["u_rank"] = top2u.groupby(["game_id", "team_id"]).cumcount() + 1
    top2u = top2u[top2u["u_rank"] <= 2]

    top2_share = top2u.groupby(["game_id", "team_id"])["usage_frac"].sum().rename("top2_pused_share")

    top2u["weighted_to"] = top2u["to_rate"] * top2u["usage_frac"]
    top2_agg_df = top2u.groupby(["game_id", "team_id"], as_index=False).agg(
        usage_sum=("usage_frac", "sum"),
        weighted_to_sum=("weighted_to", "sum"),
    )
    top2_agg_df["top2_to_rate"] = _safe_div(top2_agg_df["weighted_to_sum"], top2_agg_df["usage_sum"])
    top2_agg = top2_agg_df.set_index(["game_id", "team_id"])["top2_to_rate"]

    rot_base = rot.groupby(["game_id", "team_id"], as_index=False).agg(
        game_datetime_utc=("game_datetime_utc", "first"),
        season=("season", "first"),
        rot_size=("athlete_id", "count"),
    )

    rot_base["rot_size"] = rot_base["rot_size"].clip(upper=7)

    team_minutes = df[df["min"].fillna(0) > 0].groupby(["game_id", "team_id"]) ["min"].sum().rename("team_min")
    min_share_sd = rot.join(team_minutes, on=["game_id", "team_id"])
    min_share_sd["min_share"] = _safe_div(min_share_sd["min"], min_share_sd["team_min"])
    min_share_sd = min_share_sd.groupby(["game_id", "team_id"])["min_share"].std().rename("rot_minshare_sd")

    sd_features = rot.groupby(["game_id", "team_id"]).agg(
        rot_3p_pct_sd=("three_pct_std_l5_prior", "mean"),
        rot_to_rate_sd=("to_rate_std_l5_prior", "mean"),
    )

    # game-level rotation-weighted metrics (for rolling team features)
    rot["w_efg"] = rot["rot_weight"] * rot["efg"].fillna(0)
    rot["w_to_rate"] = rot["rot_weight"] * rot["to_rate"].fillna(0)
    rot["w_ftrate"] = rot["rot_weight"] * rot["ftrate"].fillna(0)
    rot["w_3par"] = rot["rot_weight"] * rot["three_par"].fillna(0)
    rot["w_stocks40"] = rot["rot_weight"] * rot["stocks_per40"].fillna(0)
    rot["w_pf40"] = rot["rot_weight"] * rot["pf_per40"].fillna(0)
    game_weighted = rot.groupby(["game_id", "team_id"], as_index=False).agg(
        gm_rot_efg=("w_efg", "sum"),
        gm_rot_to_rate=("w_to_rate", "sum"),
        gm_rot_ftrate=("w_ftrate", "sum"),
        gm_rot_3par=("w_3par", "sum"),
        gm_rot_stocks_per40=("w_stocks40", "sum"),
        gm_rot_pf_per40=("w_pf40", "sum"),
    )

    out = rot_base.merge(top2_share.reset_index(), on=["game_id", "team_id"], how="left")
    out = out.merge(top2_agg.reset_index(), on=["game_id", "team_id"], how="left")
    out = out.merge(min_share_sd.reset_index(), on=["game_id", "team_id"], how="left")
    out = out.merge(sd_features.reset_index(), on=["game_id", "team_id"], how="left")
    out = out.merge(game_weighted, on=["game_id", "team_id"], how="left")
    return out


def add_team_rolling_features(team_df: pd.DataFrame) -> pd.DataFrame:
    team_df = team_df.sort_values(["team_id", "game_datetime_utc", "game_id"]).copy()
    grp = team_df.groupby("team_id", group_keys=False)

    team_df["rot_efg_l5"] = grp["gm_rot_efg"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    team_df["rot_efg_l10"] = grp["gm_rot_efg"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    team_df["rot_to_rate_l5"] = grp["gm_rot_to_rate"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    team_df["rot_to_rate_l10"] = grp["gm_rot_to_rate"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    team_df["rot_ftrate_l5"] = grp["gm_rot_ftrate"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    team_df["rot_3par_l10"] = grp["gm_rot_3par"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    team_df["rot_stocks_per40_l10"] = grp["gm_rot_stocks_per40"].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    team_df["rot_pf_per40_l5"] = grp["gm_rot_pf_per40"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())

    return team_df


def add_closer_ft_pct(df_players: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    team_scores = (
        df_players.groupby(["game_id", "team_id"], as_index=False)
        .agg(game_datetime_utc=("game_datetime_utc", "first"), team_pts=("pts", "sum"))
    )
    opp = team_scores.rename(columns={"team_id": "opp_team_id", "team_pts": "opp_pts"})
    team_scores = team_scores.merge(opp[["game_id", "opp_team_id", "opp_pts"]], on="game_id", how="left")
    team_scores = team_scores[team_scores["team_id"] != team_scores["opp_team_id"]].copy()
    team_scores["margin_abs"] = (team_scores["team_pts"] - team_scores["opp_pts"]).abs()

    player_ft = df_players[["game_id", "team_id", "athlete_id", "game_datetime_utc", "usage_frac", "ftm", "fta"]].copy()
    player_ft = player_ft.merge(team_scores[["game_id", "team_id", "margin_abs"]], on=["game_id", "team_id"], how="left")

    out_vals = []
    for team_id, grp in player_ft.sort_values("game_datetime_utc").groupby("team_id"):
        rows = grp.sort_values(["game_datetime_utc", "game_id"])
        game_order = rows[["game_id", "game_datetime_utc"]].drop_duplicates().reset_index(drop=True)
        for _, gm in game_order.iterrows():
            gid = gm["game_id"]
            hist_games = game_order[game_order["game_datetime_utc"] < gm["game_datetime_utc"]].tail(10)["game_id"]
            hist = rows[rows["game_id"].isin(hist_games)]
            close = hist[hist["margin_abs"] <= 5]

            candidate_pool = close if not close.empty else hist
            if candidate_pool.empty:
                out_vals.append((gid, team_id, np.nan))
                continue

            usage_by_player = candidate_pool.groupby("athlete_id")["usage_frac"].mean().sort_values(ascending=False)
            if usage_by_player.empty:
                out_vals.append((gid, team_id, np.nan))
                continue

            closer_id = usage_by_player.index[0]
            closer_rows = candidate_pool[candidate_pool["athlete_id"] == closer_id]
            fta = closer_rows["fta"].sum(skipna=True)
            ftm = closer_rows["ftm"].sum(skipna=True)
            closer_ft = np.nan if fta <= 0 else (ftm / fta)
            out_vals.append((gid, team_id, closer_ft))

    closer_df = pd.DataFrame(out_vals, columns=["game_id", "team_id", "closer_ft_pct"])
    return team_df.merge(closer_df, on=["game_id", "team_id"], how="left")


def add_opponent_columns(team_df: pd.DataFrame) -> pd.DataFrame:
    id_cols = ["game_id", "team_id", "game_datetime_utc", "season"]
    feat_cols = [c for c in team_df.columns if c not in id_cols and not c.startswith("gm_rot_")]

    opp = team_df[["game_id", "team_id"] + feat_cols].copy()
    opp = opp.rename(columns={"team_id": "opp_team_id", **{c: f"opp_{c}" for c in feat_cols}})

    merged = team_df.merge(opp, on="game_id", how="left")
    merged = merged[merged["team_id"] != merged["opp_team_id"]].drop(columns=["opp_team_id"])
    return merged


def build_rotation_features(season: int | None = None, days_back: int | None = None) -> pd.DataFrame:
    players = load_inputs()
    players = add_player_rolling_std(players)

    team = build_team_game_rotation(players)
    team = add_team_rolling_features(team)
    team = add_closer_ft_pct(players, team)

    keep_cols = [
        "game_id",
        "team_id",
        "game_datetime_utc",
        "season",
        "rot_size",
        "top2_pused_share",
        "top2_to_rate",
        "rot_efg_l5",
        "rot_efg_l10",
        "rot_to_rate_l5",
        "rot_to_rate_l10",
        "rot_ftrate_l5",
        "rot_3par_l10",
        "rot_stocks_per40_l10",
        "rot_pf_per40_l5",
        "rot_minshare_sd",
        "rot_3p_pct_sd",
        "rot_to_rate_sd",
        "closer_ft_pct",
    ]
    out = team[keep_cols].copy()
    out = add_opponent_columns(out)

    if season is not None:
        out = out[out["season"] == season].copy()

    if days_back is not None and not out.empty:
        max_dt = out["game_datetime_utc"].max()
        cutoff = max_dt - pd.Timedelta(days=days_back)
        out = out[out["game_datetime_utc"] >= cutoff].copy()

    out = out.sort_values(["game_datetime_utc", "game_id", "team_id"]).reset_index(drop=True)
    return out


def print_summary(df: pd.DataFrame) -> None:
    print(f"Total rows written: {len(df):,}")
    print("Non-null rates for rolling columns:")
    for c in ROLLING_COLS:
        rate = 1.0 - df[c].isna().mean() if c in df.columns and len(df) else 0.0
        flag = " <60%" if rate < 0.60 else ""
        print(f"  {c}: {rate:.1%}{flag}")

    sample_cols = ["game_id", "team_id", "rot_efg_l5", "rot_to_rate_l5", "top2_pused_share", "closer_ft_pct"]
    print("Sample rows:")
    if df.empty:
        print("  [empty]")
    else:
        print(df[sample_cols].head(3).to_string(index=False))

    print(f"Count of games where closer_ft_pct is None: {df['closer_ft_pct'].isna().sum():,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build team rotation features from player game logs/metrics.")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--days-back", type=int, default=None)
    args = parser.parse_args()

    out = build_rotation_features(season=args.season, days_back=args.days_back)
    out.to_csv(OUT_PATH, index=False)
    print_summary(out)


if __name__ == "__main__":
    main()
