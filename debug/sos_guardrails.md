# SOS Size Guardrails

## Expected Grain

- `team_game_metrics.csv` and `team_game_sos.csv` should remain at team-game grain.
- Expected row count is approximately `2 x number_of_games` (one row per team per game).
- SOS merges should enrich columns only and should not materially increase row count.

## Blow-up Indicators

- Post-merge row count multiplier above `1.25x` relative to pre-merge input.
- Duplicate join keys in lookup tables for:
  - `_build_opponent_lookup`: `(opponent_id, event_id)` or `(opponent_id, _opp_dt)`
  - `_build_allowed_forced_lookup`: `(opponent_id, event_id)` or `(opponent_id, _allow_dt)`
- Null-heavy join keys (`_sort_dt`/`event_id`) that cause many-to-many joins.
- Output file size above `SOS_MAX_OUTPUT_BYTES` (default `2GB`).

## Debug Artifacts

- `debug/sos_size_audit.json`: stage-level row counts, duplication reports, join key dtypes/null rates, join hit rates.
- `debug/sos_duplicate_input_samples.csv`: sample duplicate team-game keys found in SOS input.
- `debug/sos_merge_violation_samples.csv`: duplicate lookup-key samples when merge validation fails.

## Operational Guidance

- If `sos_size_audit.json` shows high duplication on lookup keys, inspect upstream ingestion for duplicate `(team_id, event_id)` rows.
- Prefer `event_id`/`game_id` joins over coarse timestamps when available.
- Treat any row multiplier > `1.25x` as a data integrity failure, not as a normal condition.
