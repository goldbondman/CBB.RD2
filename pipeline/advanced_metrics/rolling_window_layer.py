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


def _location_rollup_stats(
    metric: pd.Series,
    locations: pd.Series,
    *,
    location_value: str,
) -> dict[str, pd.Series]:
    """Compute leak-free rollups using only prior games at one location."""
    loc_mask = locations.astype(str).str.lower() == str(location_value).lower()
    idx = metric.index
    out_empty = pd.Series(np.nan, index=idx, dtype="float64")
    if loc_mask.sum() == 0:
        return {
            "season": out_empty.copy(),
            "L4": out_empty.copy(),
            "L7": out_empty.copy(),
            "L10": out_empty.copy(),
            "L12": out_empty.copy(),
            "L10_std": out_empty.copy(),
            "trend_L4_L10": out_empty.copy(),
            "trend_L10_season": out_empty.copy(),
        }

    loc_values = _to_num(metric[loc_mask]).reset_index(drop=True)
    shifted = loc_values.shift(1)

    pre_season = shifted.expanding(min_periods=1).mean()
    pre_l4 = shifted.rolling(4, min_periods=1).mean()
    pre_l7 = shifted.rolling(7, min_periods=1).mean()
    pre_l10 = shifted.rolling(10, min_periods=1).mean()
    pre_l12 = shifted.rolling(12, min_periods=1).mean()
    pre_l10_std = shifted.rolling(10, min_periods=3).std()
    pre_trend_l4_l10 = pre_l4 - pre_l10
    pre_trend_l10_season = pre_l10 - pre_season

    post_season = loc_values.expanding(min_periods=1).mean()
    post_l4 = loc_values.rolling(4, min_periods=1).mean()
    post_l7 = loc_values.rolling(7, min_periods=1).mean()
    post_l10 = loc_values.rolling(10, min_periods=1).mean()
    post_l12 = loc_values.rolling(12, min_periods=1).mean()
    post_l10_std = loc_values.rolling(10, min_periods=3).std()
    post_trend_l4_l10 = post_l4 - post_l10
    post_trend_l10_season = post_l10 - post_season

    def _align_loc_values(pre_values: pd.Series, post_values: pd.Series) -> pd.Series:
        pre_aligned = pd.Series(np.nan, index=idx, dtype="float64")
        post_aligned = pd.Series(np.nan, index=idx, dtype="float64")
        pre_aligned.loc[loc_mask] = pre_values.to_numpy(dtype=float)
        post_aligned.loc[loc_mask] = post_values.to_numpy(dtype=float)

        # For non-location rows, carry forward latest location history including the last location game.
        result = post_aligned.ffill()
        # For location rows themselves, keep strict pregame history (exclude current row).
        result.loc[loc_mask] = pre_aligned.loc[loc_mask]
        return result

    return {
        "season": _align_loc_values(pre_season, post_season),
        "L4": _align_loc_values(pre_l4, post_l4),
        "L7": _align_loc_values(pre_l7, post_l7),
        "L10": _align_loc_values(pre_l10, post_l10),
        "L12": _align_loc_values(pre_l12, post_l12),
        "L10_std": _align_loc_values(pre_l10_std, post_l10_std),
        "trend_L4_L10": _align_loc_values(pre_trend_l4_l10, post_trend_l4_l10),
        "trend_L10_season": _align_loc_values(pre_trend_l10_season, post_trend_l10_season),
    }


def add_location_split_windows(
    df: pd.DataFrame,
    metric_columns: list[str],
    *,
    group_columns: tuple[str, ...],
    location_column: str = "home_away",
    season_column: str = "season",
    date_column: str = "game_datetime_utc",
    event_column: str = "event_id",
    home_value: str = "home",
    away_value: str = "away",
) -> pd.DataFrame:
    """Add leak-free location-aware rollups (home_* and away_*) for each metric."""
    out = df.copy()
    if location_column not in out.columns:
        return out

    sort_cols = [*group_columns, season_column, date_column, event_column]
    sort_cols = [c for c in sort_cols if c in out.columns]
    out["_row_id"] = np.arange(len(out))
    out = out.sort_values(sort_cols).reset_index(drop=True)

    grouping = [col for col in group_columns if col in out.columns]
    if season_column not in grouping and season_column in out.columns:
        grouping = [*grouping, season_column]
    if not grouping:
        out = out.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)
        return out

    locations = out[location_column].astype(str).str.lower()
    grouped = out.groupby(grouping, sort=False)

    for metric in metric_columns:
        if metric not in out.columns:
            out[metric] = np.nan
        out[metric] = _to_num(out[metric])

        for prefix, loc_value in (("home", home_value), ("away", away_value)):
            for suffix in (
                "season",
                "L4",
                "L7",
                "L10",
                "L12",
                "L10_std",
                "trend_L4_L10",
                "trend_L10_season",
            ):
                out[f"{prefix}_{metric}_{suffix}"] = np.nan

            for _, group in grouped:
                group_idx = group.index
                metric_group = out.loc[group_idx, metric]
                location_group = locations.loc[group_idx]
                stats = _location_rollup_stats(
                    metric_group,
                    location_group,
                    location_value=loc_value,
                )
                for suffix, values in stats.items():
                    out.loc[group_idx, f"{prefix}_{metric}_{suffix}"] = values.to_numpy(dtype=float)

    out = out.sort_values("_row_id").drop(columns=["_row_id"]).reset_index(drop=True)
    return out


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
