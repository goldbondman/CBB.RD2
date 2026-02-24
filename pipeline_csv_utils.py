from __future__ import annotations

import pathlib

import pandas as pd

from config.logging_config import get_logger
from config.schemas import CRITICAL_OUTPUT_SCHEMAS, validate_and_write
from espn_config import PIPELINE_RUN_ID

logger = get_logger(__name__)

ESPN_CONFERENCE_MAP: dict[str, str] = {
    "1": "Atlantic Coast Conference",
    "2": "Big East",
    "3": "Big Ten",
    "4": "Big 12",
    "5": "Conference USA",
    "7": "Pac-12",
    "8": "Southeastern Conference",
    "9": "Atlantic 10",
    "10": "Western Athletic Conference",
    "11": "Missouri Valley Conference",
    "12": "Mountain West",
    "13": "Metro Atlantic Athletic Conference",
    "14": "Mid-American Conference",
    "15": "Big Sky Conference",
    "16": "Southern Conference",
    "17": "Big South",
    "18": "Ohio Valley Conference",
    "19": "Sun Belt Conference",
    "20": "Southland Conference",
    "21": "Northeast Conference",
    "22": "Independent",
    "23": "Patriot League",
    "24": "Colonial Athletic Association",
    "25": "Horizon League",
    "26": "Southwestern Athletic Conference",
    "27": "Mid-Eastern Athletic Conference",
    "28": "Summit League",
    "29": "America East Conference",
    "30": "Atlantic Sun Conference",
    "31": "Big Ten",
    "32": "American Athletic Conference",
    "33": "Conference USA",
    "34": "Ivy League",
    "35": "West Coast Conference",
    "36": "Western Athletic Conference",
    "37": "Big West Conference",
    "44": "Independents",
    "45": "Horizon League",
    "46": "Missouri Valley Conference",
    "49": "Big Ten",
    "50": "Mid-American Conference",
    "62": "Atlantic Coast Conference",
}


def add_conference_name(df: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical conference_name values for all conference-like columns."""
    if df.empty:
        return df

    out = df.copy()

    def _translate(value: object) -> str:
        if value is None or pd.isna(value):
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return ESPN_CONFERENCE_MAP.get(text, text)

    for col in ("conference", "home_conference", "away_conference"):
        if col in out.columns:
            out[col] = out[col].apply(_translate)

    if "conference_name" in out.columns:
        out["conference_name"] = out["conference_name"].apply(_translate)
    elif "conference" in out.columns:
        out["conference_name"] = out["conference"].apply(_translate)
    elif "home_conference" in out.columns:
        out["conference_name"] = out["home_conference"].apply(_translate)

    return out

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
    out = add_conference_name(out)

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
