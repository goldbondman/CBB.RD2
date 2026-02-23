"""
ESPN CBB Pipeline — Configuration
All constants and environment variables in one place.
"""

import os
from zoneinfo import ZoneInfo
from pathlib import Path

import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent.resolve()
CSV_DIR   = BASE_DIR / "data"
JSON_DIR  = BASE_DIR / "data" / "raw_json"

CSV_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)

# Output CSVs
OUT_GAMES       = CSV_DIR / "games.csv"
OUT_TEAM_LOGS   = CSV_DIR / "team_game_logs.csv"
OUT_PLAYER_LOGS = CSV_DIR / "player_game_logs.csv"
OUT_METRICS     = CSV_DIR / "team_game_metrics.csv"
OUT_SOS              = CSV_DIR / "team_game_sos.csv"
OUT_PLAYER_PROXY     = CSV_DIR / "player_injury_proxy.csv"
OUT_TEAM_INJURY      = CSV_DIR / "team_injury_impact.csv"
OUT_WEIGHTED             = CSV_DIR / "team_game_weighted.csv"
OUT_PLAYER_METRICS       = CSV_DIR / "player_game_metrics.csv"
OUT_TOURNAMENT_METRICS   = CSV_DIR / "team_tournament_metrics.csv"
OUT_TOURNAMENT_SNAPSHOT  = CSV_DIR / "team_pretournament_snapshot.csv"
OUT_RANKINGS             = CSV_DIR / "cbb_rankings.csv"
OUT_RANKINGS_CONF        = CSV_DIR / "cbb_rankings_by_conference.csv"
OUT_PREDICTIONS_PRIMARY  = CSV_DIR / "predictions_primary.csv"
OUT_PREDICTIONS_LATEST   = CSV_DIR / "predictions_latest.csv"
OUT_PREDICTIONS_COMBINED = CSV_DIR / "predictions_combined_latest.csv"
OUT_DIVERGENCE_LATEST    = CSV_DIR / "predictions_divergence_latest.csv"

# Rolling window splits
OUT_ROLLING_L5       = CSV_DIR / "team_rolling_l5.csv"
OUT_ROLLING_L10      = CSV_DIR / "team_rolling_l10.csv"
OUT_WEIGHTED_ROLLING = CSV_DIR / "team_weighted_rolling.csv"

# Half splits
OUT_HALFSPLITS = CSV_DIR / "team_game_halfsplits.csv"

# Focused snapshot tables
OUT_ATS_PROFILE        = CSV_DIR / "team_ats_profile.csv"
OUT_LUCK_REGRESSION    = CSV_DIR / "team_luck_regression.csv"
OUT_SITUATIONAL        = CSV_DIR / "team_situational.csv"
OUT_RESUME             = CSV_DIR / "team_resume.csv"
OUT_MATCHUP_HISTORY    = CSV_DIR / "team_matchup_history.csv"
OUT_CONFERENCE_SUMMARY = CSV_DIR / "conference_daily_summary.csv"

# Travel
OUT_TRAVEL_FATIGUE = CSV_DIR / "team_travel_fatigue.csv"
OUT_VENUE_GEOCODES = CSV_DIR / "venue_geocodes.csv"

# Player splits
OUT_PLAYER_ROLLING_L5  = CSV_DIR / "player_rolling_l5.csv"
OUT_PLAYER_ROLE_SPLITS = CSV_DIR / "player_role_splits.csv"

# ── ESPN API ─────────────────────────────────────────────────────────────────
ESPN_SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/scoreboard"
    "?dates={date}&groups=50&limit=1000"
)
ESPN_SUMMARY_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/mens-college-basketball/summary?event={event_id}"
)

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept":     "application/json,text/plain,*/*",
}

# ── HTTP Retry ────────────────────────────────────────────────────────────────
REQUEST_TIMEOUT    = int(os.getenv("ESPN_TIMEOUT",        "25"))
MAX_RETRIES        = int(os.getenv("ESPN_MAX_RETRIES",     "3"))
RETRY_INITIAL_DELAY = float(os.getenv("ESPN_RETRY_DELAY", "1.0"))
RETRY_BACKOFF      = float(os.getenv("ESPN_RETRY_BACKOFF", "2.0"))

# ── Run window ────────────────────────────────────────────────────────────────
# How many days back to fetch on each run.
# Override via env var for backfill runs; default is 3 for daily cron.
DAYS_BACK = int(os.getenv("DAYS_BACK", "3"))

# Timezone used to determine "today" (PST keeps us safe for late-night games)
TZ = ZoneInfo("America/Los_Angeles")

# ── Checkpoint ───────────────────────────────────────────────────────────────
# /tmp ensures this is NEVER accidentally committed to the repo.
CHECKPOINT_FILE = os.getenv("CHECKPOINT_FILE", "/tmp/espn_cbb_checkpoint.json")

# ── Pipeline metadata ─────────────────────────────────────────────────────────
SOURCE      = "espn"
PARSE_VERSION = "v1.0.0"
DRY_RUN     = os.getenv("DRY_RUN", "0").strip().lower() in ("1", "true", "yes")
PIPELINE_RUN_ID = os.environ.get(
    "GITHUB_RUN_ID",
    pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Seconds to sleep between summary fetches to avoid hammering ESPN.
FETCH_SLEEP = float(os.getenv("FETCH_SLEEP", "0.15"))


# ── League averages (single source of truth for all modules) ─────────────
LEAGUE_AVG_ORTG   = 103.0
LEAGUE_AVG_DRTG   = 103.0
LEAGUE_AVG_PACE   = 70.0
LEAGUE_AVG_EFG    = 50.5
LEAGUE_AVG_TOV    = 18.0
LEAGUE_AVG_FTR    = 28.0
LEAGUE_AVG_ORB    = 30.0
LEAGUE_AVG_DRB    = 70.0
PYTHAGOREAN_EXP   = 11.5
DEFAULT_HCA       = 3.2
