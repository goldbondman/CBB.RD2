# Predictions CSV Schema

| Column | Description | Type |
|---|---|---|
| `run_date` | YYYY-MM-DD representing the pipeline execution date. | string |
| `game_date` | YYYY-MM-DD representing the actual game day. | string |
| `start_time_local` | ISO8601 local start time (PST context). | datetime |
| `league` | League ID or name (e.g., "NCAAM"). | string |
| `game_id` | Canonical game/event identifier. | string |
| `home_team` | Name of the home side. | string |
| `away_team` | Name of the away side. | string |
| `projected_spread` | Model's final projected point spread (negative = home favored). | float |
| `market_spread` | Consensus market spread at time of run. | float |
| `spread_delta` | `market_spread - projected_spread` (vulnerability edge). | float |
| `projected_total` | Model's final projected game total. | float |
| `market_total` | Consensus market total at time of run. | float |
| `total_delta` | `projected_total - market_total`. | float |
| `win_prob_home` | Model's estimated win probability for the home team. | float |
| `data_integrity_status` | Status: `Verified`, `Partial`, or `Conflict`. | string |
| `sources` | Compact string of input sources used (e.g., "ESPN\|Pinnacle"). | string |
| `generated_at` | UTC timestamp of generation. | datetime |
