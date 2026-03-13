# Repo Audit

- Generated at: `2026-03-12T15:48:07.367698+00:00`

## Entrypoints

- Workflow count: `25`
- Script entrypoints: `50`
- Module entrypoints: `122`
- Notebook count: `0`

## Artifacts

- Tracked artifacts: `10`

| Artifact | Producers | Consumers |
|---|---:|---:|
| `data/cbb_picks_today.csv` | 1 | 0 |
| `data/cbb_pure_trend_picks_today.csv` | 1 | 0 |
| `data/cbb_trend_picks_today.csv` | 1 | 1 |
| `data/games.csv` | 0 | 2 |
| `data/internal/core_metrics.csv` | 0 | 1 |
| `data/internal/matchup_features.csv` | 0 | 2 |
| `data/line_movement_features.csv` | 0 | 1 |
| `data/market_lines_latest_by_game.csv` | 0 | 1 |
| `data/predictions_combined_latest.csv` | 0 | 1 |
| `data/team_game_metrics.csv` | 0 | 1 |

## Dependency Hotspots

| Module | Inbound Refs | File |
|---|---:|---|
| `espn_config` | 22 | `espn_config.py` |
| `pipeline_csv_utils` | 18 | `pipeline_csv_utils.py` |
| `config.logging_config` | 12 | `config\logging_config.py` |
| `model_lab.config` | 10 | `model_lab\config.py` |
| `ingestion.market_lines` | 9 | `ingestion\market_lines.py` |
| `ides_of_march.utils` | 8 | `ides_of_march\utils.py` |
| `pipeline.advanced_metrics.feature_registry` | 7 | `pipeline\advanced_metrics\feature_registry.py` |
| `pipeline.market_canonical` | 7 | `pipeline\market_canonical.py` |
| `model_lab.splits` | 6 | `model_lab\splits.py` |
| `build_pes_columns` | 5 | `build_pes_columns.py` |
| `cbb_config` | 5 | `cbb_config.py` |
| `backtesting.direction_map` | 4 | `backtesting\direction_map.py` |
| `build_derived_csvs` | 4 | `build_derived_csvs.py` |
| `cbb_prediction_model` | 4 | `cbb_prediction_model.py` |
| `ides_of_march.config` | 4 | `ides_of_march\config.py` |

## Retire Candidates

- `audit_duplicates.py`: unreferenced_uninvoked
- `audit_feature_leakage.py`: unreferenced_uninvoked
- `audit_timezones.py`: unreferenced_uninvoked
- `backtesting\lx_days.py`: unreferenced_uninvoked
- `build_schema_registry.py`: unreferenced_uninvoked
- `check_artifact.py`: unreferenced_uninvoked
- `check_schema_drift.py`: unreferenced_uninvoked
- `compare_teams.py`: unreferenced_uninvoked
- `ides_of_march\__main__.py`: unreferenced_uninvoked
- `ides_of_march\safety.py`: unreferenced_uninvoked
- `ides_of_march\schemas.py`: unreferenced_uninvoked
- `pipeline\__main__.py`: unreferenced_uninvoked
- `pipeline\advanced_metrics\advanced_metrics_formulas.py`: unreferenced_uninvoked
- `pipeline\advanced_metrics\feature_sets.py`: unreferenced_uninvoked
- `pipeline\advanced_metrics\team_metric_compute.py`: unreferenced_uninvoked
- `pipeline\advanced_metrics\validation_layer.py`: unreferenced_uninvoked
- `query_runs.py`: unreferenced_uninvoked
- `scripts\edge_overconfidence_common.py`: unreferenced_uninvoked
- `scripts\smoke_market_lines_run_capture.py`: unreferenced_uninvoked
- `signals_library.py`: unreferenced_uninvoked

## Quarantine Candidates

- `feature_builders_overlap`: audit_feature_leakage.py, cbb_line_movement_features.py, cbb_luck_regression_features.py, cbb_rotation_features.py, cbb_situational_features.py, evaluation/feature_audit.py, model_lab/feature_selector.py, model_lab/feature_tests.py, pipeline/advanced_metrics/__init__.py, pipeline/advanced_metrics/advanced_metrics_formulas.py, pipeline/advanced_metrics/audit_logger.py, pipeline/advanced_metrics/build_advanced_metrics.py
- `model_runners_overlap`: .github/scripts/validate_joint_predictions.py, backtesting/__init__.py, backtesting/combos.py, backtesting/compute_metrics.py, backtesting/direction_map.py, backtesting/engine.py, backtesting/git_ops.py, backtesting/lx_days.py, backtesting/reporting.py, backtesting/signal_evaluator.py, backtesting/stale.py, backtesting/thresholds.py
- `market_lines_overlap`: ingestion/historical_backfill.py, ingestion/market_lines.py, scripts/backfill_market_lines.py, scripts/smoke_argparse_market_lines.py, scripts/smoke_cli_market_lines.py, scripts/smoke_market_lines_run_capture.py, scripts/smoke_normalization_market_lines.py, scripts/test_market_lines_append_only.py
