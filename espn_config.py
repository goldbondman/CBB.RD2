"""
ESPN CBB Pipeline — Configuration
All constants and environment variables in one place.
"""

import os
from zoneinfo import ZoneInfo
from pathlib import Path

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
OUT_SOS         = CSV_DIR / "team_game_sos.csv"

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

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Seconds to sleep between summary fetches to avoid hammering ESPN.
FETCH_SLEEP = float(os.getenv("FETCH_SLEEP", "0.15"))
