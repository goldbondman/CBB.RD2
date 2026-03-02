# Market lines data model

## Files

- `data/market_lines_snapshots.csv` (append-only source of truth)
  - every capture run appends rows; history is never deleted.
- `data/market_lines_latest.csv` (derived view)
  - rebuilt from snapshots, keeping the max `captured_at_utc` per `(game_id, book, market_type)`.
- `data/market_lines_closing.csv` (derived view)
  - rebuilt from snapshots per `(game_id, book, market_type)` using:
    1. last snapshot strictly before `games.game_datetime_utc` when a start time exists.
    2. if game is completed and start time is missing, use last snapshot before `final_score_timestamp_utc` if present.
    3. otherwise fallback to last snapshot overall.

## Required snapshot fields

The ingestion schema in `ingestion/market_lines.py` ensures these fields exist in snapshots:

- `captured_at_utc` (required)
- `book` (required)
- `market_type` (`spread` / `total` / `ml`)
- `home_spread_current`, `total_current`, `home_ml`, `away_ml` (as available)
- `source` (e.g., ESPN/DK/Pinnacle)

## Build step

Either command can rebuild the derived views:

- `python -m ingestion.market_lines --build-views-only`
- `python scripts/build_market_line_views.py`

## Line movement features

`cbb_line_movement_features.py` now reads from `data/market_lines_snapshots.csv` (with fallback to legacy `data/market_lines.csv` if snapshots are absent), so movement metrics are computed from historical captures and not latest-only rows.
# Market data ingestion behavior

## `data/market_lines.csv` write path and append-only semantics

The single source-of-truth writer is `ingestion.market_lines.append_market_rows`.

Behavior:

1. Load existing `data/market_lines.csv` when present.
2. Load newly fetched in-memory rows.
3. Concatenate logically (existing + new), but only append rows that are not already present by key.
4. Dedupe key selection (in priority order):
   - `(game_id, book, captured_at_utc)` when all columns exist.
   - `(game_id, book, market_type, line_type, snapshot_ts)` when available.
   - `(game_id, book, home_spread_current, total_current, home_ml, away_ml, run_id)` when available.
   - Fallback for current schema: `(game_id, capture_type, captured_at_utc)`.
5. Sort by `captured_at_utc` when present.
6. Write back atomically via temp-file + rename.

This preserves append-only history while preventing duplicate snapshots across reruns.
# Market data ingestion

## Backfill market line snapshots

Use `scripts/backfill_market_lines.py` to backfill `data/market_lines_snapshots.csv` for a date range without deleting existing rows.

### Notes

- The script reuses the same normalization path as `ingestion.market_lines`.
- Dedupe uses the same key shape as the market pipeline: `(event_id, capture_type, capture_hour)`.
- Historical snapshot APIs are not available from these sources, so backfill captures **current lines as observed when the script runs**.

### Usage examples

```bash
# Default: last 100 days
python scripts/backfill_market_lines.py

# Explicit date range
python scripts/backfill_market_lines.py --start_date 2026-01-01 --end_date 2026-02-01

# Last 30 days for Action-only enrichment (no direct book pulls)
python scripts/backfill_market_lines.py --days 30 --books ""

# Backfill with only DraftKings matching enabled
python scripts/backfill_market_lines.py --days 14 --books draftkings

# Dry run (print summary but do not write)
python scripts/backfill_market_lines.py --days 7 --dry_run
```

### Manual workflow convenience example

If you are running a manual backfill workflow (for example via `workflow_dispatch`), use:

- `days_back = 100`
- `append = true`
- `target_file = data/market_lines_master.csv`
- `mode = pregame` (or `morning`)

### Safety notes

- Backfill is idempotent because master append applies dedupe keys before writing.
- Do **not** run two backfills concurrently (they can race on the same append target).

### Output summary

The script prints:

- days processed
- games found
- snapshots fetched
- new rows added
- duplicates skipped


## Market Lines CLI hardening (date inputs)

- `python -m ingestion.market_lines --mode pregame` now succeeds with omitted date inputs (defaults to UTC today).
- `start_date`, `end_date`, and `days_back` treat blank values as unset (`None`).
- `--days-back` is validated as `>= 1` only when it is explicitly provided.
- `data/market_lines_master.csv` is the durable append-only in-repo store and is created automatically when missing.

### Safe backfill examples

```bash
# Omitted date inputs (scheduled/manual defaults)
python -m ingestion.market_lines --mode pregame

# Rolling window backfill
python -m ingestion.market_lines --mode pregame --days-back 100

# Explicit date range backfill
python -m ingestion.market_lines --mode pregame --start-date 2026-01-01 --end-date 2026-01-31
```
