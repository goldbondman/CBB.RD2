from __future__ import annotations

import logging
import pathlib
import pandas as pd

logger = logging.getLogger(__name__)

PRIMARY_KEYS_MAP: dict[str, list[str]] = {
    'team_game_metrics.csv': ['event_id', 'team_id'],
    'team_game_logs.csv': ['event_id', 'team_id'],
    'team_game_weighted.csv': ['event_id', 'team_id'],
    'player_game_metrics.csv': ['event_id', 'athlete_id'],
    'player_game_logs.csv': ['event_id', 'athlete_id'],
    'games.csv': ['game_id'],
    'cbb_rankings.csv': ['team_id'],
    'results_log.csv': ['game_id'],
    'results_log_graded.csv': ['game_id'],
    'predictions_latest.csv': ['game_id'],
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


def safe_write_csv(df: pd.DataFrame, path: str | pathlib.Path, *, index: bool = False) -> pd.DataFrame:
    """Apply deterministic dedupe before write and persist atomically."""
    p = pathlib.Path(path)
    out = dedupe_by_primary_key(df, p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    out.to_csv(tmp, index=index)
    tmp.replace(p)
    return out
