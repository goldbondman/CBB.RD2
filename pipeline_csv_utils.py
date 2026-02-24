from __future__ import annotations

import pathlib

import pandas as pd

from config.logging_config import get_logger
from config.schemas import CRITICAL_OUTPUT_SCHEMAS, validate_and_write
from espn_config import PIPELINE_RUN_ID

logger = get_logger(__name__)

PRIMARY_KEYS_MAP: dict[str, list[str]] = {
    'team_game_metrics.csv': ['event_id', 'team_id'],
    'team_game_logs.csv': ['event_id', 'team_id'],
    'team_game_weighted.csv': ['event_id', 'team_id'],
    'player_game_metrics.csv': ['event_id', 'athlete_id'],
    'player_game_logs.csv': ['event_id', 'athlete_id'],
    'games.csv': ['game_id'],
    'cbb_rankings.csv': ['team_id'],
    'results_log.csv': ['event_id'],
    'results_log_graded.csv': ['event_id'],
    'predictions_latest.csv': ['event_id'],
    'predictions_combined_latest.csv': ['event_id'],
}


PIPELINE_NUMERIC_COLS: list[str] = [
    'pred_spread', 'pred_total', 'predicted_spread', 'predicted_total',
    'spread', 'over_under', 'home_spread_open', 'home_spread_current',
    'line_movement', 'home_ml', 'away_ml', 'home_win_prob', 'away_win_prob',
    'home_score', 'away_score', 'actual_margin', 'actual_total', 'home_won',
    'win', 'wins', 'losses', 'confidence', 'model_confidence',
]


def normalize_numeric_dtypes(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Coerce known numeric pipeline columns to nullable numeric dtypes."""
    if df is None or df.empty:
        return df

    out = df.copy()
    cols = numeric_cols or PIPELINE_NUMERIC_COLS
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors='coerce').astype('Float64')
COLUMN_ALIASES: dict[str, list[str]] = {
    'event_id': ['game_id', 'eventId', 'gameId'],
    'pred_spread': ['predicted_spread', 'prediction_spread'],
    'home_ml': ['home_money_line', 'home_moneyline'],
    'away_ml': ['away_money_line', 'away_moneyline'],
    'cover': ['ats_cover', 'cover_result'],
    'cover_margin': ['ats_margin', 'margin_vs_spread'],
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize known aliases to canonical pipeline column names."""
    out = df.copy()
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out = out.rename(columns={alias: canonical})
                break
    return out


def dedupe_by_primary_key(df: pd.DataFrame, path: str | pathlib.Path) -> pd.DataFrame:
    """Drop duplicates for known CSVs using deterministic keep='last'."""
    fname = pathlib.Path(path).name
    pk = PRIMARY_KEYS_MAP.get(fname)
    if not pk:
        return df

    available_pk = [c for c in pk if c in df.columns]
    if not available_pk:
        logger.warning(
            "No primary key columns available for dedupe in %s; expected %s",
            fname,
            pk,
        )
        return df
    if available_pk != pk:
        logger.warning(
            "Partial primary key for dedupe in %s: using %s (expected %s)",
            fname,
            available_pk,
            pk,
        )

    before = len(df)
    deduped = df.drop_duplicates(subset=available_pk, keep='last')
    removed = before - len(deduped)
    if removed > 0:
        logger.warning("Deduplicated %s rows in %s on %s", removed, fname, available_pk)
    return deduped


def safe_write_csv(
    df: pd.DataFrame,
    path: str | pathlib.Path,
    index: bool = False,
    label: str | None = None,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Apply deterministic dedupe, stamp run_id, validate schema, and persist atomically."""
    p = pathlib.Path(path)
    out = dedupe_by_primary_key(df, p)

    if out.empty and not allow_empty:
        logger.warning("%s is empty; skipping write to %s", label or p.stem, p)
        return out

    if 'pipeline_run_id' not in out.columns:
        out = out.copy()
        out['pipeline_run_id'] = PIPELINE_RUN_ID

    schema = CRITICAL_OUTPUT_SCHEMAS.get(p.stem)
    if schema:
        validate_and_write(out, p, schema, label or p.stem, index=index)
        return out

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    out.to_csv(tmp, index=index)
    tmp.replace(p)
    logger.info("✓ %s → %s (%s rows)", label or p.stem, p, len(out))
    return out
