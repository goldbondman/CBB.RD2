"""Matchup metric computation layer."""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

from .metric_registry import MATCHUP_METRIC_REGISTRY, matchup_metric_names
from .rolling_window_layer import add_metric_rollups

_SIDE_COL_PATTERN = re.compile(r"^(?P<base>.+)_(?P<side>A|B)(?P<suffix>(?:_.+)?)$")


def _add_location_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        match = _SIDE_COL_PATTERN.match(col)
        if not match:
            continue
        base = match.group("base")
        if base in {"team_id"}:
            continue
        side = match.group("side")
        suffix = match.group("suffix") or ""
        prefix = "home" if side == "A" else "away"
        alias = f"{prefix}_{base}{suffix}"
        if alias not in out.columns:
            out[alias] = out[col]
    return out


def compute_matchup_metrics(team_game_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Build one-row-per-event matchup metrics from team-game metrics."""
    required = {"event_id", "game_datetime_utc", "season", "team_id", "home_away"}
    missing = sorted(required - set(team_game_metrics_df.columns))
    if missing:
        raise ValueError(f"matchup metric compute missing required columns: {missing}")

    team_df = team_game_metrics_df.copy()
    team_df["game_datetime_utc"] = pd.to_datetime(team_df["game_datetime_utc"], utc=True, errors="coerce")

    home = team_df[team_df["home_away"].astype(str).str.lower() == "home"].copy()
    away = team_df[team_df["home_away"].astype(str).str.lower() == "away"].copy()
    if home.empty or away.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "game_datetime_utc",
                "season",
                "home_team_id",
                "away_team_id",
                "PEI_matchup",
                "ODI_diff",
                "ODI_sum",
                "MTI",
                "SCI",
            ]
        )

    keep_cols = [
        "event_id",
        "game_datetime_utc",
        "season",
        "team_id",
        "ANE",
        "SVI",
        "PEQ",
        "POSW",
        "WL",
        "ODI",
        "TC",
        "TIN",
        "VOL",
        "DPC",
        "FFC",
        "PXP",
        "SCI",
    ]
    keep_cols = [c for c in keep_cols if c in team_df.columns]

    home = home[keep_cols].rename(columns={"team_id": "home_team_id"})
    away = away[keep_cols].rename(columns={"team_id": "away_team_id"})
    home = home.rename(columns={c: f"{c}_home" for c in keep_cols if c not in {"event_id", "game_datetime_utc", "season", "home_team_id"}})
    away = away.rename(columns={c: f"{c}_away" for c in keep_cols if c not in {"event_id", "game_datetime_utc", "season", "away_team_id"}})

    merged = home.merge(
        away,
        on=["event_id", "game_datetime_utc", "season"],
        how="inner",
    )

    for metric_name, definition in MATCHUP_METRIC_REGISTRY.items():
        needed = set(definition.required_inputs + definition.derived_inputs)
        missing_inputs = needed - set(merged.columns)
        if missing_inputs:
            merged[metric_name] = np.nan
            continue
        merged[metric_name] = definition.compute_fn(merged)

    # Leak-free matchup rollups by home team-season.
    merged = add_metric_rollups(
        merged,
        metric_columns=matchup_metric_names(),
        group_columns=("home_team_id", "season"),
        date_column="game_datetime_utc",
    )

    base_cols = ["event_id", "game_datetime_utc", "season", "home_team_id", "away_team_id"]
    metric_cols = matchup_metric_names()
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

    selected = [c for c in base_cols + metric_cols + roll_cols if c in merged.columns]
    result = merged[selected].sort_values(["game_datetime_utc", "event_id"]).reset_index(drop=True)
    return _add_location_aliases(result)

