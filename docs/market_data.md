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

### Output summary

The script prints:

- days processed
- games found
- snapshots fetched
- new rows added
- duplicates skipped
