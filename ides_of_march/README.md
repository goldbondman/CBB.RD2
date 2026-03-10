# IDES.OF.MARCH

Standalone layered CBB prediction system.

## CLI

- `python -m ides_of_march.cli audit --strict`
- `python -m ides_of_march.cli predict --mc-mode confidence_filter --hours-ahead 48`
- `python -m ides_of_march.cli backtest --start-date 20250101 --end-date 20260301`

## Outputs

All outputs are isolated under `data/ides_of_march/`:

- `predictions_latest.csv`
- `bet_recs.csv`
- `agreement_bucket_report.csv`
- `situational_rulebook.csv`
- `backtest_variant_scorecard.csv`
- `run_manifest.json`
