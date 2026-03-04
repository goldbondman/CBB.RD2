"""Leak-free rolling window and within-season z-score utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd

WINDOWS = (4, 7, 10, 12)


def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def add_leak_free_windows(
    df: pd.DataFrame,
    metric_columns: list[str],
    *,
    group_columns: tuple[str, ...],
    season_column: str = "season",
    date_column: str = "game_datetime_utc",
    event_column: str = "event_id",
) -> pd.DataFrame:
    """Add leak-free rolling columns for every metric.

    For each metric, creates:
    - season mean
    - L4, L7, L10, L12 means
    - L10 std
    - trend_L4_L10
    - trend_L10_season
    """
    out = df.copy()
    sort_cols = [*group_columns, season_column, date_column, event_column]
    sort_cols = [c for c in sort_cols if c in out.columns]
    out["_row_id"] = np.arange(len(out))
    out = out.sort_values(sort_cols).reset_index(drop=True)

    grouping = [col for col in group_columns if col in out.columns]
    if season_column not in grouping and season_column in out.columns:
        grouping = [*grouping, season_column]
    if not grouping:
        raise ValueError("add_leak_free_windows requires at least one valid grouping column")
    grp = out.groupby(grouping, sort=False)

    for metric in metric_columns:
        if metric not in out.columns:
            out[metric] = np.nan
        out[metric] = _to_num(out[metric])

        out[f"{metric}_season"] = grp[metric].transform(
            lambda s: _to_num(s).shift(1).expanding(min_periods=1).mean()
        )
        for window in WINDOWS:
            out[f"{metric}_L{window}"] = grp[metric].transform(
                lambda s, w=window: _to_num(s).shift(1).rolling(w, min_periods=1).mean()
            )
        out[f"{metric}_L10_std"] = grp[metric].transform(
            lambda s: _to_num(s).shift(1).rolling(10, min_periods=3).std()
        )
        out[f"{metric}_trend_L4_L10"] = out[f"{metric}_L4"] - out[f"{metric}_L10"]
        out[f"{metric}_trend_L10_season"] = out[f"{metric}_L10"] - out[f"{metric}_season"]

    out = out.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)
    return out


def add_metric_rollups(
    df: pd.DataFrame,
    metric_columns: list[str],
    *,
    group_columns: tuple[str, ...],
    date_column: str = "game_datetime_utc",
) -> pd.DataFrame:
    """Backward-compatible alias used by older modules."""
    return add_leak_free_windows(
        df,
        metric_columns,
        group_columns=group_columns,
        date_column=date_column,
    )


def within_season_zscore(
    df: pd.DataFrame,
    value_column: str,
    *,
    season_column: str = "season",
    date_column: str = "game_datetime_utc",
    event_column: str = "event_id",
    group_columns: tuple[str, ...] | None = None,
    min_periods: int = 5,
) -> pd.Series:
    """Leak-free z-score computed only from prior games in the same season."""
    work = df.copy()
    work["_row_id"] = np.arange(len(work))
    sort_cols = [season_column, date_column, event_column]
    if group_columns:
        sort_cols = [*group_columns, date_column, event_column]
    sort_cols = [c for c in sort_cols if c in work.columns]
    work = work.sort_values(sort_cols).reset_index(drop=True)

    if group_columns:
        grouping = [col for col in group_columns if col in work.columns]
        if not grouping:
            grouping = [season_column] if season_column in work.columns else []
        if not grouping:
            work["_z"] = np.nan
            work = work.sort_values("_row_id")
            return work["_z"].reset_index(drop=True)
        grp = work.groupby(grouping, sort=False)[value_column]
    else:
        grp = work.groupby(season_column, sort=False)[value_column]

    prior_mean = grp.transform(lambda s: _to_num(s).shift(1).expanding(min_periods=min_periods).mean())
    prior_std = grp.transform(lambda s: _to_num(s).shift(1).expanding(min_periods=min_periods).std())
    work["_z"] = (_to_num(work[value_column]) - prior_mean) / prior_std.replace(0, np.nan)

    work = work.sort_values("_row_id")
    return work["_z"].reset_index(drop=True)
