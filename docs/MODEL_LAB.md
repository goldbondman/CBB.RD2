# Model Lab

## Purpose
Model Lab evaluates feature signal versus noise and scores existing model outputs using rolling forward splits only.

The implementation is isolated to `model_lab/` and writes run artifacts under:

`data/model_lab_runs/<run_id>/`

## Data Build
`python -m model_lab.cli build-frames --run-id <run_id>`

The frame builder (`model_lab/data_builder.py`) does the following:

1. Loads Feature Engine matchup data from:
- `data/matchup_metrics.csv`

2. Loads labels (priority order):
- `data/results_log_graded.csv`
- `data/results_log.csv`
- `data/backtest_training_data.csv`
- `data/backtest_results_latest.csv`

3. Loads market lines (priority order):
- `data/market_lines_closing.csv`
- `data/market_lines.csv`
- `data/games.csv`

4. Normalizes join keys safely:
- canonical `game_id`/`event_id`
- UTC datetime parsing
- `season_id` derivation from season or datetime

5. Builds frames:
- `spread_frame.csv` (rows with `actual_margin`)
- `total_frame.csv` (rows with `actual_total`)
- `ml_frame.csv` (rows with `home_won`)

The frame build writes row counts and NaN report into `run_manifest.json`.

## Rolling Splits
`model_lab/splits.py` implements forward-only folds:

1. Preferred mode: season rolling (`season_id`)
- train: all seasons before test season
- test: one forward season at a time

2. Fallback mode: date rolling (`game_datetime_utc`)
- contiguous forward windows
- train always strictly before test

No random split is used.

## Model Scoring
`python -m model_lab.cli score-models --run-id <run_id>`

This command:

1. Loads standardized model predictions using `model_lab/model_wrappers.py`.
2. Builds market datasets (`spread`, `total`, `ml`).
3. Applies rolling forward folds.
4. Scores each model with `model_lab/evaluators.py` and `model_lab/metrics.py`:
- hit rate
- ROI (defaults to -110 when odds are missing)
- CLV (when open/close lines exist)
- MAE (spread/total)
- Brier and calibration ECE (ML)

Output:
- `model_scorecard.csv`

## Feature Signal Tests
`python -m model_lab.cli feature-signal --run-id <run_id> --market spread`

Runs `model_lab/feature_tests.py`:

1. Univariate signal per feature (Spearman correlation).
2. Permutation importance against a baseline fold model.
3. Drop-one ablation:
- single feature removal
- feature-group removal (group by prefix)
4. Feature Stability Score across folds.

Outputs:
- `feature_scorecard.csv`
- `feature_stability.csv`

## Ensemble Optimization
`python -m model_lab.cli ensemble --run-id <run_id> --markets spread total ml --max-weight 0.5`

Runs `model_lab/ensemble.py`:

1. Optimizes weights on training folds only.
2. Enforces constraints:
- weights >= 0
- sum(weights) = 1
- max weight <= configured cap (default 0.5)
3. Evaluates optimized ensemble on forward test folds.

Output:
- `ensemble_weights.json`

## Output Interpretation
All command runs update:
- `run_manifest.json`

Key fields in the manifest:
- `git_sha`
- timestamps (`created_at_utc`, `updated_at_utc`)
- built frame paths and row counts
- folds used by market
- blocked reasons
- NaN report
- artifact paths

Primary artifacts in each run directory:
- `model_scorecard.csv`
- `feature_scorecard.csv`
- `feature_stability.csv`
- `ensemble_weights.json`
- `run_manifest.json`
