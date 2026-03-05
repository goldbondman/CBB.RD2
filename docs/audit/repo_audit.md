# Repo Audit

- Generated at: `2026-03-05T05:56:04.569869+00:00`

## Entrypoints

- Workflow count: `8`
- Script entrypoints: `36`
- Module entrypoints: `68`
- Notebook count: `0`

## Artifacts

- Tracked artifacts: `4`

| Artifact | Producers | Consumers |
|---|---:|---:|
| `data/games.csv` | 0 | 2 |
| `data/line_movement_features.csv` | 0 | 1 |
| `data/predictions_combined_latest.csv` | 0 | 1 |
| `data/team_game_metrics.csv` | 0 | 1 |

## Dependency Hotspots

| Module | Inbound Refs | File |
|---|---:|---|
| `espn_config` | 22 | `espn_config.py` |
| `pipeline_csv_utils` | 18 | `pipeline_csv_utils.py` |
| `config.logging_config` | 10 | `config\logging_config.py` |
| `model_lab.config` | 10 | `model_lab\config.py` |
| `ingestion.market_lines` | 8 | `ingestion\market_lines.py` |
| `pipeline.advanced_metrics.feature_registry` | 7 | `pipeline\advanced_metrics\feature_registry.py` |
| `model_lab.splits` | 6 | `model_lab\splits.py` |
| `cbb_config` | 5 | `cbb_config.py` |
| `build_derived_csvs` | 4 | `build_derived_csvs.py` |
| `model_lab.evaluators` | 4 | `model_lab\evaluators.py` |
| `pipeline.advanced_metrics.rolling_window_layer` | 4 | `pipeline\advanced_metrics\rolling_window_layer.py` |
| `backtesting.direction_map` | 3 | `backtesting\direction_map.py` |
| `cbb_backtester` | 3 | `cbb_backtester.py` |
| `cbb_ensemble` | 3 | `cbb_ensemble.py` |
| `cbb_output_schemas` | 3 | `cbb_output_schemas.py` |

## Retire Candidates

- `audit_duplicates.py`: unreferenced_uninvoked
- `audit_feature_leakage.py`: unreferenced_uninvoked
- `audit_timezones.py`: unreferenced_uninvoked
- `backtesting\lx_days.py`: unreferenced_uninvoked
- `build_schema_registry.py`: unreferenced_uninvoked
- `check_artifact.py`: unreferenced_uninvoked
- `check_schema_drift.py`: unreferenced_uninvoked
- `compare_teams.py`: unreferenced_uninvoked
- `pipeline\__main__.py`: unreferenced_uninvoked
- `pipeline\advanced_metrics\feature_sets.py`: unreferenced_uninvoked
- `pipeline\advanced_metrics\team_metric_compute.py`: unreferenced_uninvoked
- `pipeline\advanced_metrics\validation_layer.py`: unreferenced_uninvoked
- `query_runs.py`: unreferenced_uninvoked
- `scripts\smoke_market_lines_run_capture.py`: unreferenced_uninvoked
- `signals_library.py`: unreferenced_uninvoked
- `tests\test_backtest_outputs.py`: unreferenced_uninvoked
- `tests\test_build_backtest_csvs.py`: unreferenced_uninvoked
- `tests\test_build_derived_csvs.py`: unreferenced_uninvoked
- `tests\test_cbb_accuracy_report.py`: unreferenced_uninvoked
- `tests\test_cbb_backtester_market_lines.py`: unreferenced_uninvoked

## Quarantine Candidates

- `feature_builders_overlap`: audit_feature_leakage.py, cbb_line_movement_features.py, cbb_luck_regression_features.py, cbb_rotation_features.py, cbb_situational_features.py, evaluation/feature_audit.py, model_lab/feature_selector.py, model_lab/feature_tests.py, pipeline/advanced_metrics/__init__.py, pipeline/advanced_metrics/audit_logger.py, pipeline/advanced_metrics/cache_layer.py, pipeline/advanced_metrics/compute_runner.py
- `model_runners_overlap`: backtesting/__init__.py, backtesting/combos.py, backtesting/compute_metrics.py, backtesting/direction_map.py, backtesting/engine.py, backtesting/git_ops.py, backtesting/lx_days.py, backtesting/reporting.py, backtesting/stale.py, backtesting/thresholds.py, build_backtest_csvs.py, cbb_backtester.py
- `market_lines_overlap`: ingestion/historical_backfill.py, ingestion/market_lines.py, scripts/backfill_market_lines.py, scripts/smoke_argparse_market_lines.py, scripts/smoke_cli_market_lines.py, scripts/smoke_market_lines_run_capture.py, scripts/smoke_normalization_market_lines.py, scripts/test_market_lines_append_only.py
