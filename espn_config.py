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
DATA_DIR  = CSV_DIR
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
OUT_PREDICTIONS_GRADED   = CSV_DIR / "predictions_graded.csv"
OUT_MODEL_WEIGHTS        = CSV_DIR / "model_weights.json"
OUT_CONFIDENCE_CALIBRATION = CSV_DIR / "confidence_calibration.json"
OUT_DIVERGENCE_LATEST    = CSV_DIR / "predictions_divergence_latest.csv"
OUT_PREDICTIONS_GRADED   = CSV_DIR / "predictions_graded.csv"
OUT_BIAS_TABLE           = CSV_DIR / "model_bias_table.csv"
OUT_BIAS_REPORT          = CSV_DIR / "bias_report.json"
OUT_BIAS_HISTORY         = CSV_DIR / "bias_history.csv"

# ── New output files ─────────────────────────────────────
OUT_PREDICTIONS_GRADED     = CSV_DIR / "predictions_graded.csv"
OUT_PREDICTIONS_CONTEXT    = CSV_DIR / "predictions_with_context.csv"
OUT_MARKET_LINES           = DATA_DIR / "market_lines.csv"
OUT_FORM_SNAPSHOT          = CSV_DIR / "team_form_snapshot.csv"
OUT_EDGE_HISTORY           = CSV_DIR / "edge_history.csv"

# Accuracy outputs
OUT_ACCURACY_WEEKLY        = CSV_DIR / "model_accuracy_weekly.csv"
OUT_ACCURACY_BY_CONF       = CSV_DIR / "model_accuracy_by_conf.csv"
OUT_MODEL_VERSION_HISTORY = CSV_DIR / "model_version_history.json"
OUT_ACCURACY_BY_VERSION   = CSV_DIR / "accuracy_by_version.csv"
OUT_MODEL_CALIBRATION      = CSV_DIR / "model_calibration.csv"
OUT_ACCURACY_SUMMARY       = CSV_DIR / "model_accuracy_summary.csv"
OUT_HCA_ANALYSIS           = CSV_DIR / "hca_analysis.csv"

# Self-improvement outputs
OUT_BIAS_TABLE             = CSV_DIR / "model_bias_table.csv"
OUT_BIAS_REPORT            = CSV_DIR / "bias_report.json"
OUT_BIAS_HISTORY           = CSV_DIR / "bias_history.csv"
OUT_MODEL_WEIGHTS          = CSV_DIR / "model_weights.json"
OUT_CONFIDENCE_CALIBRATION = CSV_DIR / "confidence_calibration.json"

# Rankings history
OUT_RANKINGS_HISTORY       = CSV_DIR / "rankings_history.csv"

# Results
OUT_RESULTS_LOG            = CSV_DIR / "results_log.csv"

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

ESPN_CONFERENCE_MAP = {
    "1": "America East Conference",
    "2": "Atlantic Coast Conference",
    "3": "Atlantic 10",
    "4": "Big East",
    "5": "Big Sky Conference",
    "6": "Big South Conference",
    "7": "Big Ten",
    "8": "Big 12",
    "9": "Big West Conference",
    "10": "Colonial Athletic Association",
    "11": "Conference USA",
    "12": "Ivy League",
    "13": "Metro Atlantic Athletic Conference",
    "14": "Mid-American Conference",
    "15": "Mid-Eastern Athletic Conference",
    "16": "Mid-Eastern Athletic Conference",
    "17": "Big South Conference",
    "18": "Missouri Valley Conference",
    "19": "Northeast Conference",
    "20": "Ohio Valley Conference",
    "21": "Northeast Conference",
    "22": "Patriot League",
    "23": "Southeastern Conference",
    "24": "Southern Conference",
    "25": "Southland Conference",
    "26": "Southwestern Athletic Conference",
    "27": "Sun Belt Conference",
    "28": "Summit League",
    "29": "West Coast Conference",
    "30": "Western Athletic Conference",
    "31": "Big Ten",
    "32": "American Athletic Conference",
    "33": "Conference USA",
    "34": "Ivy League",
    "35": "West Coast Conference",
    "36": "Western Athletic Conference",
    "37": "Big West Conference",
    "44": "Mountain West",
    "45": "Horizon League",
    "46": "ASUN Conference",
    "49": "Summit League",
    "50": "Mid-American Conference",
    "62": "American Athletic Conference",
}


def conference_id_to_name(conference_id) -> str:
    """Convert ESPN numeric conference ID to human-readable name."""
    if conference_id is None or pd.isna(conference_id):
        return ""
    cid = str(conference_id).strip()
    return ESPN_CONFERENCE_MAP.get(cid, cid)


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
SEASON_ACTIVE: bool = True

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


CONFERENCE_TIERS = {
    "HIGH": {
        "Atlantic Coast", "Big Ten", "Big 12", "Southeastern", "Big East",
    },
    "MID": {
        "American Athletic", "Mountain West", "Atlantic 10",
        "Missouri Valley", "West Coast", "Conference USA",
        "Sun Belt", "Mid-American", "Colonial Athletic", "ASUN",
    },
    "LOW": {
        "Big South", "Horizon", "Ivy League",
        "Metro Atlantic", "Northeast",
        "Ohio Valley", "Patriot", "Southern Conference", "Southland",
        "Southwestern Athletic", "Mid-Eastern Athletic", "Big Sky", "Western Athletic",
        "America East", "Summit League",
    },
}


def get_conference_tier(conference: str) -> str:
    """Return HIGH / MID / LOW / UNKNOWN for a conference string."""
    if not conference or pd.isna(conference):
        return "UNKNOWN"

    conf_clean = str(conference).strip()
    for tier, conf_set in CONFERENCE_TIERS.items():
        for conf_name in conf_set:
            if conf_name.lower() in conf_clean.lower():
                return tier
    return "UNKNOWN"


def get_game_tier(home_conference: str, away_conference: str) -> str:
    """Classify game as pure conference tier or cross-tier matchup."""
    tier_h = get_conference_tier(home_conference)
    tier_a = get_conference_tier(away_conference)

    if tier_h == tier_a:
        return tier_h

    tiers = tuple(sorted([tier_h, tier_a]))
    cross_map = {
        ("HIGH", "LOW"): "CROSS_HIGH_LOW",
        ("HIGH", "MID"): "CROSS_HIGH_MID",
        ("LOW", "MID"): "CROSS_MID_LOW",
    }
    return cross_map.get(tiers, "UNKNOWN")
