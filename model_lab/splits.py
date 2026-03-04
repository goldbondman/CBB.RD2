from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from .config import ModelLabConfig


@dataclass
class Fold:
    fold_id: str
    mode: str
    train_start_utc: str | None
    train_end_utc: str | None
    test_start_utc: str | None
    test_end_utc: str | None
    train_size: int
    test_size: int
    train_seasons: list[int]
    test_seasons: list[int]
    train_index: list[int]
    test_index: list[int]

    def to_manifest(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("train_index", None)
        payload.pop("test_index", None)
        return payload


def _dt_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if "game_datetime_utc" not in df.columns or df.empty:
        return None, None
    dt = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    if dt.notna().sum() == 0:
        return None, None
    return dt.min().isoformat(), dt.max().isoformat()


def _season_rolling_folds(df: pd.DataFrame, config: ModelLabConfig) -> list[Fold]:
    if "season_id" not in df.columns:
        return []

    season = pd.to_numeric(df["season_id"], errors="coerce").astype("Int64")
    seasons = sorted(int(v) for v in season.dropna().unique().tolist())
    if len(seasons) < 2:
        return []

    folds: list[Fold] = []
    for idx, test_season in enumerate(seasons[1:], start=1):
        train_mask = season < test_season
        test_mask = season == test_season

        train_idx = df.index[train_mask].tolist()
        test_idx = df.index[test_mask].tolist()
        if len(train_idx) < config.min_train_games:
            continue
        if len(test_idx) < config.min_test_games:
            continue

        train_df = df.loc[train_idx]
        test_df = df.loc[test_idx]
        train_start, train_end = _dt_bounds(train_df)
        test_start, test_end = _dt_bounds(test_df)

        fold = Fold(
            fold_id=f"season_{test_season}",
            mode="season",
            train_start_utc=train_start,
            train_end_utc=train_end,
            test_start_utc=test_start,
            test_end_utc=test_end,
            train_size=len(train_idx),
            test_size=len(test_idx),
            train_seasons=sorted(int(v) for v in season[train_mask].dropna().unique().tolist()),
            test_seasons=[int(test_season)],
            train_index=train_idx,
            test_index=test_idx,
        )
        folds.append(fold)

    return folds


def _date_rolling_folds(df: pd.DataFrame, config: ModelLabConfig) -> list[Fold]:
    if "game_datetime_utc" not in df.columns:
        return []

    dt = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    valid = df[dt.notna()].copy()
    if valid.empty:
        return []

    valid["_dt"] = pd.to_datetime(valid["game_datetime_utc"], utc=True, errors="coerce")
    valid = valid.sort_values("_dt").reset_index()

    min_train = config.min_train_games
    test_window = timedelta(days=config.date_test_window_days)
    step = timedelta(days=config.date_step_days)

    folds: list[Fold] = []
    if len(valid) < (min_train + config.min_test_games):
        return folds

    start_anchor = valid.loc[min_train - 1, "_dt"]
    if pd.isna(start_anchor):
        return folds

    test_start = start_anchor + timedelta(days=1)
    last_date = valid["_dt"].max()
    fold_num = 1

    while test_start <= last_date:
        test_end = test_start + test_window

        train_mask = valid["_dt"] < test_start
        test_mask = (valid["_dt"] >= test_start) & (valid["_dt"] < test_end)

        train_idx = valid.loc[train_mask, "index"].tolist()
        test_idx = valid.loc[test_mask, "index"].tolist()

        if len(train_idx) >= config.min_train_games and len(test_idx) >= config.min_test_games:
            train_df = df.loc[train_idx]
            test_df = df.loc[test_idx]
            train_start_s, train_end_s = _dt_bounds(train_df)
            test_start_s, test_end_s = _dt_bounds(test_df)

            season_train = (
                pd.to_numeric(train_df.get("season_id"), errors="coerce").dropna().astype(int).unique().tolist()
                if "season_id" in train_df.columns
                else []
            )
            season_test = (
                pd.to_numeric(test_df.get("season_id"), errors="coerce").dropna().astype(int).unique().tolist()
                if "season_id" in test_df.columns
                else []
            )

            folds.append(
                Fold(
                    fold_id=f"date_{fold_num:02d}",
                    mode="date",
                    train_start_utc=train_start_s,
                    train_end_utc=train_end_s,
                    test_start_utc=test_start_s,
                    test_end_utc=test_end_s,
                    train_size=len(train_idx),
                    test_size=len(test_idx),
                    train_seasons=sorted(season_train),
                    test_seasons=sorted(season_test),
                    train_index=train_idx,
                    test_index=test_idx,
                )
            )
            fold_num += 1

        test_start = test_start + step

    return folds


def build_rolling_folds(df: pd.DataFrame, config: ModelLabConfig) -> tuple[list[Fold], list[str]]:
    blocked: list[str] = []
    folds = _season_rolling_folds(df, config)
    if folds:
        return folds, blocked

    blocked.append("season_folds_unavailable_or_insufficient")
    folds = _date_rolling_folds(df, config)
    if folds:
        return folds, blocked

    blocked.append("date_folds_unavailable_or_insufficient")
    return [], blocked
