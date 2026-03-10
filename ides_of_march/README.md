# IDES.OF.MARCH

Standalone layered CBB prediction system.

## CLI

- `python -m ides_of_march.cli audit --strict`
- `python -m ides_of_march.cli predict --mc-mode confidence_filter --hours-ahead 48`
- `python -m ides_of_march.cli backtest --start-date 20250101 --end-date 20260301`

## Outputs

IDES outputs are physically separated by category under `data/`:

- `data/actionable/`: `bet_recommendations.csv`, `watchlist_games.csv`, `no_bet_explanations.csv`, `daily_card_summary.csv`
- `data/reports/`: `game_predictions_master.csv`, `agreement_analysis_results.csv`, `backtest_model_summary.csv`
- `data/plumbing/`: schedule, boxscore, rolling, matchup, totals, context, situational, and monte-carlo plumbing CSVs
- `data/contracts/`: `csv_contract_registry.csv`, `schema_rules.json`
- `data/logs/`: `pipeline_run_log.csv`, `run_manifest.json`
