# ESPN CBB Data Pipeline — Requirements

## Purpose
Fetch, parse, and store men's college basketball game data and box scores from the ESPN API. This is the data foundation of the project — if this pipeline is broken, everything downstream is broken.

## Data Requirements

### What we fetch
| Output File | Source | Granularity | Key Fields |
|---|---|---|---|
| `data/games.csv` | ESPN Scoreboard API | One row per game | game_id, date, teams, scores, status, odds |
| `data/team_game_logs.csv` | ESPN Summary API | One row per team per game | box score stats, points, margin |
| `data/player_game_logs.csv` | ESPN Summary API | One row per player per game | individual stats, minutes, shooting splits |

### Data completeness standard
- **90% minimum** completion rate: every completed game in `games.csv` must have corresponding rows in `team_game_logs.csv`
- Games marked `completed=False` are allowed to be missing from logs (not yet played)
- The validation step in the workflow enforces this and will fail the run if breached

### Season coverage
- **Daily runs**: last 3 days (catches same-day games + any late updates from prior 2 days)
- **Backfill runs**: triggered manually via `workflow_dispatch` with `days_back=120` to cover full season (Nov 1 – present)
- Date window always includes today + tomorrow to avoid PST/UTC boundary misses on late-night games

---

## Data Sources

### ESPN API (primary and only source)
- **Scoreboard**: `https://site.api.espn.com/.../scoreboard?dates={YYYYMMDD}&groups=50&limit=1000`
  - Returns all D1 games for a given date
  - `groups=50` filters to men's college basketball
  - `limit=1000` ensures we don't miss games on busy days
- **Summary**: `https://site.api.espn.com/.../summary?event={event_id}`
  - Returns full box score, player stats, odds, venue for a single game
- **Auth**: None required (public API)
- **Rate limit**: Not officially documented; pipeline sleeps 150ms between summary fetches

---

## Pipeline Architecture

### Files
| File | Responsibility |
|---|---|
| `espn_config.py` | All constants and env vars. Single source of truth for configuration. |
| `espn_client.py` | HTTP fetch with retry/backoff. No business logic. |
| `espn_parsers.py` | ESPN JSON → flat Python dicts. No I/O, pure functions. |
| `espn_pipeline.py` | Orchestration: fetch dates → parse → write CSVs. Entry point. |
| `.github/workflows/update-espn-cbb.yml` | GitHub Actions cron + manual trigger. |

### Flow
```
Scoreboard API (per date)
    └── parse_scoreboard_event()  →  games.csv
            │
            └── game_ids
                    │
                    ▼
            Summary API (per game)
                └── parse_summary()
                        ├── summary_to_team_rows()  →  team_game_logs.csv
                        └── player rows             →  player_game_logs.csv
```

---

## Reliability Requirements

### Retry logic
- HTTP requests: up to 3 attempts with exponential backoff (1s → 2s → 4s)
- Failed games: automatically retried once after all games are processed, with a 2s delay before retry pass

### Checkpoint/resume
- Checkpoint saved to `/tmp/espn_cbb_checkpoint.json` every 25 games
- If the pipeline crashes mid-run, the next run resumes from where it left off
- Checkpoint is in `/tmp` and **never committed to the repo** — a stale committed checkpoint was the primary bug in the previous pipeline version

### CSV persistence
- Previous run's CSVs are downloaded from GitHub Actions artifact at the **start** of each run
- `_append_dedupe_write` merges new data with existing, always keeping the best row per game:
  - Prefers `completed=True` over `completed=False` (stale pending rows get replaced)
  - Tie-breaks on `pulled_at_utc` (most recent wins)
- Writes are atomic: temp file written first, then renamed to final path

---

## Configuration

All config lives in `espn_config.py`. Key env vars:

| Variable | Default | Description |
|---|---|---|
| `DAYS_BACK` | `3` | How many days back to fetch on each run |
| `DRY_RUN` | `0` | Set to `1` to run without writing any files |
| `ESPN_TIMEOUT` | `25` | HTTP request timeout in seconds |
| `ESPN_MAX_RETRIES` | `3` | Max retry attempts per request |
| `FETCH_SLEEP` | `0.15` | Seconds between summary fetches |
| `CHECKPOINT_FILE` | `/tmp/espn_cbb_checkpoint.json` | Must stay in `/tmp` |

---

## GitHub Actions Workflow

### Schedule
- Runs daily at **10:00 AM UTC** (2:00 AM PST / 3:00 AM PDT)
- Chosen to capture all games from the prior day, which typically end by ~midnight ET

### Manual backfill
Trigger via `workflow_dispatch` in the GitHub Actions UI. Set `days_back` to:
- `3` — normal daily catch-up
- `120` — full season backfill (Nov 1 through today)

### Steps (in order)
1. Checkout repo
2. **Download previous CSVs** from prior artifact into `data/` ← critical for persistence
3. Set up Python 3.11
4. Install `pandas requests`
5. Run `espn_pipeline.py`
6. Validate outputs (completion rate check)
7. Show file sizes
8. Upload CSVs as artifact `espn-cbb-csvs` (retained 7 days)

---

## Known Limitations

- ESPN's public API has no SLA and can return incomplete box scores for very recently completed games. Re-running with the same date range 24 hours later will fill these in via the dedupe logic.
- Odds data (`spread`, `over_under`, `home_ml`, `away_ml`) is best-effort — ESPN doesn't always populate these fields, especially for early-season or low-profile games.
- Player `starter` status is sometimes absent from the ESPN response; stored as `None` when missing.
- ESPN does not provide official play-by-play in this pipeline (summary endpoint only).

---

## How to Run Locally

```bash
# Install dependencies
pip install pandas requests

# Normal 3-day run
python espn_pipeline.py

# Full season backfill
DAYS_BACK=120 python espn_pipeline.py

# Dry run (no files written)
DRY_RUN=1 DAYS_BACK=3 python espn_pipeline.py
```

Output files will be written to `data/` relative to the script location.
