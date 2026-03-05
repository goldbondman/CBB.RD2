# SOS Blow-up Root Cause Report

## Historical failure evidence (GitHub Actions run `22736368169`)

- Failing step: `Run dependency-aware perplexity stages via contract runner`.
- `espn_pipeline.py` called `compute_sos_metrics(df_metrics_out)`.
- Stack trace terminated at:
  - `espn_sos.py`, `compute_sos_metrics` -> `_build_allowed_forced_lookup`
  - `_build_allowed_forced_lookup` merge call
- Error:
  - `numpy._core._exceptions._ArrayMemoryError: Unable to allocate 81.4 GiB for an array with shape (50, 218558826)`

Interpretation: merge cardinality exploded (many-to-many behavior), causing row-index expansion to hundreds of millions.

## Local instrumented verification

- Input (`team_game_logs.csv` -> `compute_all_metrics`) was `10,796` rows.
- `compute_sos_metrics` audit (`debug/sos_size_audit.json`) now records:
  - join key selected: `event_id` (not coarse timestamp)
  - `_build_opponent_lookup` multiplier: `1.0`
  - `_build_allowed_forced_lookup` multiplier: `1.0`
  - final multiplier: `1.0`

## Synthetic duplicate stress check

- Injected `500` duplicate team-event rows into SOS input.
- Result after dedupe and guarded merges:
  - input rows: `11,296`
  - output rows: `10,796`
  - output grain remained one row per team-event.
- Duplicate sample emitted to `debug/sos_duplicate_input_samples.csv`.

## Conclusion

Primary explosion source was the SOS merge path (historically `_build_allowed_forced_lookup` merge).  
Fixes now enforce:
- deterministic dedupe at team-game grain
- lookup-key uniqueness before merge
- `many_to_one` merge validation
- row-growth and file-size guardrails with explicit failure messages.
