"""
cbb_pipeline_logger.py — Pipeline Run Audit Logger

Appends a row to data/pipeline_run_log.csv at the end of each
espn_pipeline.py run. The log is append-only and never truncated.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

LOG_PATH = Path("data") / "pipeline_run_log.csv"

COLUMNS = [
    "run_timestamp_utc",
    "trigger",
    "days_back",
    "dates_fetched",
    "games_found",
    "games_parsed",
    "games_failed",
    "team_rows_written",
    "player_rows_written",
    "parse_version",
    "dry_run",
    "status",
]


def log_pipeline_run(
    *,
    trigger: str = "scheduled",
    days_back: int = 0,
    dates_fetched: int = 0,
    games_found: int = 0,
    games_parsed: int = 0,
    games_failed: int = 0,
    team_rows_written: int = 0,
    player_rows_written: int = 0,
    parse_version: str = "",
    dry_run: bool = False,
    status: str = "ok",
    output_path: Path = LOG_PATH,
) -> None:
    """Append one audit row to pipeline_run_log.csv."""
    row = {
        "run_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trigger": trigger,
        "days_back": days_back,
        "dates_fetched": dates_fetched,
        "games_found": games_found,
        "games_parsed": games_parsed,
        "games_failed": games_failed,
        "team_rows_written": team_rows_written,
        "player_rows_written": player_rows_written,
        "parse_version": parse_version,
        "dry_run": dry_run,
        "status": status,
    }
    new_df = pd.DataFrame([row], columns=COLUMNS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(output_path, index=False)
    log.info(f"Pipeline run logged → {output_path}")
