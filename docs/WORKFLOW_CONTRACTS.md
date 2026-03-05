# Workflow Contracts

## Workflow Paths
- `.github/workflows/cbb_predictions_rolling.yml`
- `.github/workflows/cbb_analytics.yml`
- `.github/workflows/player_overlay.yml`

## Discovery Summary

### cbb_predictions_rolling.yml
- Job: `predict`
- Python/script commands:
  - `scripts/audit_csv_quality.py`
  - `scripts/ci_data_quality_gate.py`
  - `cbb_travel_fatigue.py`
  - `scripts/validate_data_infrastructure.py`
  - `espn_prediction_runner.py`
  - `cbb_ensemble.py`
  - `cbb_monte_carlo.py --input data/predictions_combined_latest.csv --n-sims 5000`
  - `cbb_player_matchup.py --games data/predictions_combined_latest.csv`
  - `infra/ci/validate_predictions_freshness.py`
  - `build_derived_csvs.py`
- Required upstream artifacts/files:
  - `data/games.csv`
  - `data/team_game_weighted.csv`
- Core outputs:
  - `data/predictions_primary.csv`
  - `data/predictions_latest.csv`
  - `data/predictions_combined_latest.csv`
  - `data/predictions_history.csv`
- Optional outputs:
  - `data/ensemble_predictions_latest.csv`
  - `data/predictions_mc_latest.csv`
  - derived files from `build_derived_csvs.py`

### cbb_analytics.yml
- Job: `analytics`
- Python/script commands:
  - `ingestion/historical_backfill.py`
  - `cbb_backtester.py`
  - `scripts/validate_backtest_market_data.py`
  - `evaluation/predictions_graded.py`
  - `build_training_data.py`
  - `cbb_results_tracker.py`
  - `cbb_season_summaries.py`
  - `cbb_accuracy_report.py`
  - `scripts/validate_data_infrastructure.py`
- Required upstream artifacts/files:
  - `data/market_lines.csv` (from market lines artifact or local history)
  - `data/team_game_weighted.csv` (from ESPN data artifact)
  - predictions artifact (`data/predictions_combined_latest.csv` or `data/predictions_with_context.csv`)
- Core outputs:
  - `data/results_log.csv`
  - `data/results_summary.csv`
- Optional outputs:
  - `data/backtest_results_latest.csv`
  - `data/backtest_training_data.csv`
  - `data/team_season_summary.csv`
  - `data/market_lines_closing.csv`

### player_overlay.yml
- Job: `overlay-tests`
- Python/script commands:
  - `pytest -q tests/test_player_matchup_overlay.py`
  - `scripts/run_player_overlay.py --input ... --output ...` (manual/standalone path)
- Required input for standalone overlay:
  - `data/player_overlay_input.csv` (or explicit `--input` path)
- Core output:
  - `data/player_overlay_predictions.csv`

## Downstream Dependency Highlights
- `data/predictions_combined_latest.csv` is consumed by:
  - `cbb_analytics.yml`
  - `cbb_user_deliverables.yml`
  - `market_lines.yml`
  - `cbb_results_tracker.py`
  - `pipeline/integrity.py`
  - `model_lab/module_dag.yaml`
- `data/backtest_results_latest.csv` is consumed by:
  - `cbb_analytics.yml`
  - `scripts/validate_backtest_market_data.py`
  - `model_lab/config.py`
- `data/player_overlay_predictions.csv` is produced/consumed only by:
  - `player_overlay.yml`
  - `scripts/run_player_overlay.py`

## Contract Table

| workflow | jobs | critical steps | required inputs | contract outputs | blocking checks |
|---|---|---|---|---|---|
| `cbb_predictions_rolling` | `predict` | download ESPN artifact, run primary, merge combined, append history | `data/games.csv`, `data/team_game_weighted.csv` | `predictions_primary.csv`, `predictions_latest.csv`, `predictions_combined_latest.csv`, `predictions_history.csv` | fail if required upstream files missing/empty; fail if combined output missing; block optional enrichers when prerequisites missing |
| `cbb_analytics` | `analytics` | resolve/download artifacts, merge staged data, run grading/tracker/reporting | `data/market_lines.csv`, `data/team_game_weighted.csv`, predictions rolling output | `results_log.csv`, `results_summary.csv` (plus optional backtest outputs) | fail if required artifacts missing; fail if required inputs missing/empty; freshness gate for predictions inputs |
| `player_overlay` | `overlay-tests`, `overlay-run` | unit tests, optional overlay run, output validation | `data/player_overlay_input.csv` | `player_overlay_predictions.csv` | fail if overlay input missing for overlay run; fail if output missing/empty/required columns missing |

## Contract Manifests
- `.github/pipeline_contracts/cbb_predictions_rolling.contract.json`
- `.github/pipeline_contracts/cbb_analytics.contract.json`
- `.github/pipeline_contracts/player_overlay.contract.json`
