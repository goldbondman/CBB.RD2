from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


PREDICTIONS_REQUIRED = [
    "game_id",
    "event_id",
    "game_datetime_utc",
    "home_team",
    "away_team",
    "home_team_id",
    "away_team_id",
    "projected_spread",
    "market_spread",
    "projected_margin_home",
    "win_prob_home",
    "ats_cover_prob_home",
    "confidence_score",
    "agreement_bucket",
    "bet_recommendation",
]

BET_RECS_REQUIRED = [
    "game_id",
    "event_id",
    "game_datetime_utc",
    "home_team",
    "away_team",
    "market_spread",
    "projected_spread",
    "edge_home",
    "win_prob_home",
    "ats_cover_prob_home",
    "confidence_score",
    "bet_recommendation",
    "line_source_used",
]

AGREEMENT_REQUIRED = [
    "agreement_bucket",
    "sample_size",
    "su_accuracy",
    "ats_accuracy",
    "avg_edge",
    "confidence_mean",
]

SITUATIONAL_RULEBOOK_REQUIRED = [
    "rule_id",
    "description",
    "sample_size",
    "raw_ats_rate",
    "shrunk_ats_rate",
    "effect",
    "accepted",
]

VARIANT_REQUIRED = [
    "variant_id",
    "sample_size",
    "spread_mae",
    "winner_accuracy",
    "ats_accuracy",
    "calibration_brier",
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


def validate_rulebook(df: pd.DataFrame) -> SchemaCheck:
    missing = _missing(df, SITUATIONAL_RULEBOOK_REQUIRED)
    return SchemaCheck(ok=not missing, missing_columns=missing)


def validate_variant_scorecard(df: pd.DataFrame) -> SchemaCheck:
    missing = _missing(df, VARIANT_REQUIRED)
    return SchemaCheck(ok=not missing, missing_columns=missing)
