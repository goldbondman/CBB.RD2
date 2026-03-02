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
