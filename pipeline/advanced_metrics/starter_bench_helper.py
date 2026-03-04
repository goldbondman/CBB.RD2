"""Centralized starter/bench feature helper."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _to_bool(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "yes", "y", "t"})


def _ts_from_totals(points: float, fga: float, fta: float) -> float:
    den = 2.0 * (fga + (0.44 * fta))
    if den <= 0:
        return np.nan
    return float(points) / float(den)


def compute_starter_bench_features(player_game_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-(event_id, team_id) starter/bench splits.

    Starter logic:
    - If a `starter` flag exists and is populated, use it.
    - Otherwise use top-5 players by `min` for that game/team.
    """
    output_cols = [
        "event_id",
        "team_id",
        "bench_minutes_share",
        "TS_bench",
        "TS_starters",
        "REB_rate_bench",
        "REB_rate_starters",
    ]
    if player_game_df is None or player_game_df.empty:
        return pd.DataFrame(columns=output_cols)

    required = {"event_id", "team_id", "min", "pts", "fga", "fta", "reb"}
    missing = required - set(player_game_df.columns)
    if missing:
        raise ValueError(f"starter/bench helper missing required columns: {sorted(missing)}")

    df = player_game_df.copy()
    for col in ["min", "pts", "fga", "fta", "reb"]:
        df[col] = _to_num(df[col]).fillna(0.0)

    if "did_not_play" in df.columns:
        dnp = _to_bool(df["did_not_play"])
        df = df.loc[~dnp].copy()

    rows: list[dict] = []
    for (event_id, team_id), grp in df.groupby(["event_id", "team_id"], sort=False):
        grp = grp.copy()
        if grp.empty:
            continue

        has_starter_signal = "starter" in grp.columns and grp["starter"].notna().any()
        if has_starter_signal:
            starter_mask = _to_bool(grp["starter"])
        else:
            top_idx = grp.sort_values(["min"], ascending=False).head(5).index
            starter_mask = grp.index.isin(top_idx)
            starter_mask = pd.Series(starter_mask, index=grp.index)

        bench_mask = ~starter_mask

        total_min = float(grp["min"].sum())
        bench_min = float(grp.loc[bench_mask, "min"].sum())
        bench_minutes_share = (bench_min / total_min) if total_min > 0 else np.nan

        st_pts = float(grp.loc[starter_mask, "pts"].sum())
        st_fga = float(grp.loc[starter_mask, "fga"].sum())
        st_fta = float(grp.loc[starter_mask, "fta"].sum())
        bn_pts = float(grp.loc[bench_mask, "pts"].sum())
        bn_fga = float(grp.loc[bench_mask, "fga"].sum())
        bn_fta = float(grp.loc[bench_mask, "fta"].sum())

        ts_starters = _ts_from_totals(st_pts, st_fga, st_fta)
        ts_bench = _ts_from_totals(bn_pts, bn_fga, bn_fta)

        team_reb = float(grp["reb"].sum())
        st_reb = float(grp.loc[starter_mask, "reb"].sum())
        bn_reb = float(grp.loc[bench_mask, "reb"].sum())
        reb_rate_starters = (st_reb / team_reb) if team_reb > 0 else np.nan
        reb_rate_bench = (bn_reb / team_reb) if team_reb > 0 else np.nan

        rows.append(
            {
                "event_id": event_id,
                "team_id": team_id,
                "bench_minutes_share": bench_minutes_share,
                "TS_bench": ts_bench,
                "TS_starters": ts_starters,
                "REB_rate_bench": reb_rate_bench,
                "REB_rate_starters": reb_rate_starters,
            }
        )

    return pd.DataFrame(rows, columns=output_cols)

