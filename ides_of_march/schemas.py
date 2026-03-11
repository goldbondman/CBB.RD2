from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


PREDICTIONS_REQUIRED = [
    "run_id",
    "model_version",
    "game_id",
    "game_date_pst",
    "game_start_time_pst",
    "game_start_datetime_pst",
    "game_start_datetime_utc",
    "team_a",
    "team_b",
    "model_spread_team_a",
    "market_spread_team_a",
    "team_a_win_probability",
    "team_a_cover_probability",
    "team_a_win_probability_mc",
    "team_a_cover_probability_mc",
    "spread_confidence",
    "agreement_bucket",
    "final_bet_flag",
]

BET_RECS_REQUIRED = [
    "run_id",
    "model_version",
    "game_id",
    "game_date_pst",
    "game_start_time_pst",
    "game_start_datetime_pst",
    "team_a",
    "team_b",
    "bet_type",
    "bet_side",
    "market_line",
    "model_line",
    "edge",
    "confidence",
    "mc_win_probability",
    "mc_cover_probability",
]

AGREEMENT_REQUIRED = [
    "run_id",
    "model_version",
    "agreement_bucket",
    "sample_size",
    "straight_up_win_pct",
    "ats_win_pct",
    "avg_spread_edge",
]

VARIANT_REQUIRED = [
    "run_id",
    "model_version",
    "variant_name",
    "games_tested",
    "spread_mae",
    "winner_accuracy",
    "ats_win_pct_all",
]


@dataclass
class SchemaCheck:
    ok: bool
    missing_columns: list[str]


def _missing(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return sorted(set(required) - set(df.columns))


def validate_predictions(df: pd.DataFrame) -> SchemaCheck:
    missing = _missing(df, PREDICTIONS_REQUIRED)
    return SchemaCheck(ok=not missing, missing_columns=missing)


def validate_bet_recs(df: pd.DataFrame) -> SchemaCheck:
    missing = _missing(df, BET_RECS_REQUIRED)
    return SchemaCheck(ok=not missing, missing_columns=missing)


def validate_agreement(df: pd.DataFrame) -> SchemaCheck:
    missing = _missing(df, AGREEMENT_REQUIRED)
    return SchemaCheck(ok=not missing, missing_columns=missing)


def validate_backtest_summary(df: pd.DataFrame) -> SchemaCheck:
    missing = _missing(df, VARIANT_REQUIRED)
    return SchemaCheck(ok=not missing, missing_columns=missing)
