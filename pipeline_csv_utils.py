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


def dedupe_by_primary_key(df: pd.DataFrame, path: str | pathlib.Path) -> pd.DataFrame:
    """Drop duplicates for known CSVs using deterministic keep='last'."""
    fname = pathlib.Path(path).name
    pk = PRIMARY_KEYS_MAP.get(fname)
    if not pk:
        return df

    available_pk = [c for c in pk if c in df.columns]
    if not available_pk:
        logger.warning("Primary key columns %s not present in %s; skipping dedupe", pk, fname)
        return df

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
