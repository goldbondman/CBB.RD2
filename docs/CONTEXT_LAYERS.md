# Context Layers

This document defines context-layer pipeline order, module dependencies, and downstream consumers.

## Pipeline Order

1. Core feature builders (`games.csv`, `team_game_weighted.csv`, market lines, predictions inputs).
2. `context_layers/run_all.py` builds `data/context/context_overlay_latest.csv`.
3. Prediction and analytics modules consume `context_overlay_latest.csv`:
   - predictions workflow keeps overlay as dependency artifact.
   - analytics workflow runs gates and segment explorer with overlay-enriched fields.
4. Gate and segment outputs:
   - `data/gates/gate_results.csv`
   - `data/gates/gate_rules_best.csv`
   - `data/analytics/segment_performance.csv`

## Module Dependencies

- `context_layers/run_all.py` reads optional layer outputs when present:
  - `data/tournaments/fatigue_flags.csv`
  - `data/tournaments/neutral_adjustments.csv`
  - `data/tournaments/leverage_flags.csv`
  - `data/tournaments/familiarity_score.csv`
  - `data/ncaa/upset_profile.csv`
  - `data/totals/tempo_control_score.csv`
  - `data/halves/first_half_pace.csv`
  - `data/totals/second_half_total_factors.csv`
- If a module file is missing, overlay columns remain null and status is recorded in `data/context/context_overlay_exec_summary.md`.

## Downstream Consumption

- Predictions:
  - workflow now enforces context overlay build before prediction generation.
  - overlay is uploaded with rolling predictions artifacts.
- Gates (`scripts/gate_builder.py`):
  - core gates: min edge, volatility block, low-total block, sharp confirmation, overconfidence block.
  - context-aware behavior via config (`config/gate_builder_context.yml`) and optional context/auxiliary files.
- Segment explorer (`scripts/segment_performance_explorer.py`):
  - existing groups retained.
  - new context groups:
    - `season_phase` (`regular`, `conf_tournament`, `ncaa`)
    - `ncaa_round`
    - `fatigue_tier`
    - `neutral_scope`
    - `fh_fast_start_tier`
    - `sh_total_bias_tier`

## Consumer Mapping

- Predictions consume overlay for game-type/tier context.
- Gates consume overlay-adjacent tiers to filter candidate plays.
- Segment analytics consumes overlay tiers to evaluate contextual performance slices.
