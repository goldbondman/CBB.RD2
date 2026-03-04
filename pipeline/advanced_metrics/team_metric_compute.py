"""Team metric computation layer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .metric_registry import TEAM_METRIC_REGISTRY, team_metric_names
from .rolling_window_layer import add_metric_rollups
from .shared_derivations import add_shared_derivations
from .starter_bench_helper import compute_starter_bench_features


def _derive_season(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    season = dt.dt.year.where(dt.dt.month < 10, dt.dt.year + 1)
    return season.astype("Int64")


def _attach_team_pregame_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["team_id", "season", "game_datetime_utc", "event_id"]).reset_index(drop=True)
    g = out.groupby(["team_id", "season"], sort=False)
    out["pre_NetRtg_season"] = g["NetRtg"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding(min_periods=1).mean()
    )
    out["pre_poss_season"] = g["poss"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding(min_periods=1).mean()
    )
    return out


def _attach_opponent_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    opp_lookup = out[
        [
            "event_id",
            "team_id",
            "eFG",
            "TOV%",
            "ORB%",
            "FTr",
            "pre_NetRtg_season",
            "pre_poss_season",
        ]
    ].rename(
        columns={
            "team_id": "opponent_id",
            "eFG": "opp_eFG",
            "TOV%": "opp_TOV%",
            "ORB%": "opp_ORB%",
            "FTr": "opp_FTr",
            "pre_NetRtg_season": "opp_pre_NetRtg_season",
            "pre_poss_season": "opp_pre_poss_season",
        }
    )
    out = out.merge(opp_lookup, on=["event_id", "opponent_id"], how="left")
    return out


def compute_team_game_metrics(
    team_game_df: pd.DataFrame,
    player_game_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute team-game metrics from team/player box score tables."""
    required = {
        "event_id",
        "game_datetime_utc",
        "team_id",
        "opponent_id",
        "home_away",
        "points_for",
        "points_against",
        "fgm",
        "fga",
        "tpm",
        "tpa",
        "ftm",
        "fta",
        "orb",
        "drb",
        "tov",
        "opp_fgm",
        "opp_fga",
        "opp_tpm",
        "opp_tpa",
        "opp_ftm",
        "opp_fta",
        "opp_orb",
        "opp_drb",
        "opp_tov",
    }
    missing = sorted(required - set(team_game_df.columns))
    if missing:
        raise ValueError(f"team metric compute missing required columns: {missing}")

    out = team_game_df.copy()
    out["game_datetime_utc"] = pd.to_datetime(out["game_datetime_utc"], utc=True, errors="coerce")
    out["season"] = _derive_season(out["game_datetime_utc"])

    out = add_shared_derivations(out)
    out = _attach_team_pregame_baselines(out)
    out = _attach_opponent_context(out)

    if player_game_df is not None and not player_game_df.empty:
        sb = compute_starter_bench_features(player_game_df)
        out = out.merge(sb, on=["event_id", "team_id"], how="left")
    else:
        for col in [
            "bench_minutes_share",
            "TS_bench",
            "TS_starters",
            "REB_rate_bench",
            "REB_rate_starters",
        ]:
            out[col] = np.nan

    for metric_name, definition in TEAM_METRIC_REGISTRY.items():
        needed = set(definition.required_inputs + definition.derived_inputs)
        metric_missing = needed - set(out.columns)
        if metric_missing:
            out[metric_name] = np.nan
            continue
        out[metric_name] = definition.compute_fn(out)

    out = add_metric_rollups(
        out,
        metric_columns=team_metric_names(),
        group_columns=("team_id", "season"),
        date_column="game_datetime_utc",
    )

    base_cols = [
        "event_id",
        "game_datetime_utc",
        "season",
        "team_id",
        "opponent_id",
        "home_away",
        "points_for",
        "points_against",
    ]
    derivation_cols = [
        "poss",
        "OffEff",
        "DefEff",
        "NetRtg",
        "eFG",
        "3PA_rate",
        "FTr",
        "FT_pts_per_FGA",
        "FT_pts_per_poss",
        "ORB%",
        "DRB%",
        "TOV%",
        "bench_minutes_share",
        "TS_bench",
        "TS_starters",
        "REB_rate_bench",
        "REB_rate_starters",
    ]

    metric_cols = team_metric_names()
    roll_cols: list[str] = []
    for metric in metric_cols:
        roll_cols.extend(
            [
                f"{metric}_season",
                f"{metric}_L4",
                f"{metric}_L7",
                f"{metric}_L10",
                f"{metric}_L12",
                f"{metric}_L10_std",
                f"{metric}_trend_L4_L10",
                f"{metric}_trend_L10_season",
            ]
        )

    selected = [c for c in base_cols + derivation_cols + metric_cols + roll_cols if c in out.columns]
    return out[selected].sort_values(["game_datetime_utc", "event_id", "team_id"]).reset_index(drop=True)


def generate_metric_tables(
    *,
    team_game_logs_path: str | Path,
    player_game_logs_path: str | Path,
    team_game_metrics_out: str | Path,
    matchup_metrics_out: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate team_game_metrics and matchup_metrics tables and write CSVs."""
    from .matchup_metric_compute import compute_matchup_metrics

    team_df = pd.read_csv(team_game_logs_path, low_memory=False)
    player_df = pd.read_csv(player_game_logs_path, low_memory=False)

    team_metrics = compute_team_game_metrics(team_df, player_df)
    matchup_metrics = compute_matchup_metrics(team_metrics)

    Path(team_game_metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(matchup_metrics_out).parent.mkdir(parents=True, exist_ok=True)
    team_metrics.to_csv(team_game_metrics_out, index=False)
    matchup_metrics.to_csv(matchup_metrics_out, index=False)
    return team_metrics, matchup_metrics

