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
