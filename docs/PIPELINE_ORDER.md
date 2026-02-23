# Pipeline Execution Order

| Script | Reads | Writes | Triggered By |
|--------|-------|--------|--------------|
| `ingestion/espn_pipeline.py` | ESPN API endpoints, existing `data/csv/*.csv` | `data/csv/games.csv`, `data/csv/team_game_metrics.csv`, derived ESPN CSVs | `update_espn_cbb.yml`, `cbb_predictions_rolling.yml`, `cbb_backtest_tracker.yml` |
| `ingestion/cbb_results_tracker.py` | `data/csv/predictions_combined_latest.csv`, `data/csv/games.csv` | `data/results_log.csv`, grading outputs | `cbb_predictions_rolling.yml`, `Makefile grade` |
| `features/espn_metrics.py` | `data/csv/team_game_logs.csv` | rolling metrics CSVs | `update_espn_cbb.yml`, `Makefile features` |
| `features/espn_weighted_metrics.py` | `data/csv/team_game_metrics.csv`, SOS metrics | `data/csv/team_game_weighted.csv` | `update_espn_cbb.yml`, `Makefile features` |
| `features/espn_rankings.py` | weighted metrics, tournament snapshot | `data/csv/cbb_rankings.csv`, conference rankings | `update_espn_cbb.yml`, `Makefile features` |
| `features/team_form_snapshot.py` | `data/csv/team_game_metrics.csv` | `data/csv/team_form_snapshot.csv` | `cbb_backtest_tracker.yml`, `Makefile features` |
| `models/espn_prediction_runner.py` | rankings + metrics + optional odds | `data/csv/predictions_latest.csv`, combined predictions | `update_espn_cbb.yml`, `cbb_predictions_rolling.yml`, `Makefile predict` |
| `enrichment/predictions_with_context.py` | latest predictions + context CSVs | context-enriched prediction CSV | `Makefile predict` |
| `enrichment/edge_history.py` | enriched predictions and/or graded outputs | `data/csv/edge_history.csv` | `Makefile predict`, `Makefile grade` |
| `evaluation/model_accuracy_weekly.py` | graded predictions history | weekly model-accuracy summary CSV | `Makefile grade` |
| `evaluation/bias_detector.py` | predictions/results history | bias report artifact(s) | `Makefile tune` |
| `evaluation/optimize_weights.py` | backtest inputs + bias outputs | updated `config/model_weights.json` | `Makefile tune` |
| `evaluation/calibrate_confidence.py` | predictions + outcomes + model weights | `config/confidence_calibration.json` | `Makefile tune` |
