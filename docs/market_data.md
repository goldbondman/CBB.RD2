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
