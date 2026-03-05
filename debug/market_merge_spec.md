# Market Merge Spec (Backtest + Training)

## Canonical ID
- Primary key: `game_id` / `event_id` normalized to digit-only string.
- Normalization: trim, drop `.0`, strip non-digits.

## Training data merge (`build_training_data.py`)
- Base table key: `game_id` from `team_game_weighted.csv` home rows (`state=post`).
- Market source precedence:
  1. `data/market_lines_closing.csv`
  2. `data/market_lines_latest.csv`
  3. `data/market_lines.csv`
  4. `data/odds_snapshot.csv`
- Row selection per game: sort by `is_closing desc`, `source_rank asc`, `capture_ts desc`, then keep first.
- Market fields:
  - `opening_spread` <- first available from `opening_spread|home_spread_open|spread_open`
  - `closing_spread` <- `closing_spread|home_spread_current|home_spread|spread_line|market_spread|close_home_spread`
  - `total_line` <- `total_line|total_current|market_total|close_total|over_under|total`
- Fallback fields from weighted game row:
  - `espn_spread` <- `spread`
  - `espn_total` <- `over_under`
- Final lines:
  - `spread_line = coalesce(closing_spread, opening_spread, espn_spread)`
  - `total_line = coalesce(total_line, espn_total)`

## Backtest results merge (`cbb_backtester.py`)
- Market source precedence identical to training path.
- For each game:
  - `home_market_spread = coalesce(closing_spread, opening_spread, espn_spread)`
  - `market_total = coalesce(market_total, espn_total)`
  - `spread_line = home_market_spread`
  - `total_line = market_total`
- Prediction aliases:
  - `pred_spread = ens_spread`
  - `pred_total = ens_total`
- CLV definition:
  - `clv_delta = closing_spread - pred_spread`

## Join diagnostics
- `debug/backtest_data_audit.json` logs row counts, source files, join hit counts, null rates, and sample unmatched rows.
- `debug/backtest_context_validation.json` logs validation rates and threshold checks.
