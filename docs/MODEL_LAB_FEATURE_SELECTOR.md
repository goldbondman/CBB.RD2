# Model Lab Feature Selector (Option B / AUTO_V2)

This document defines the semi-automatic feature selector and window-grid evaluator used by `model_lab`.

## Scope

- Inputs are run artifacts under `data/model_lab_runs/<run_id>/`.
- Evaluation is rolling-forward only via `model_lab.splits.build_rolling_folds`.
- No random train/test split is used.

## Selection Criteria

The selector reads:

- `feature_scorecard.csv`
- `feature_stability.csv`

Per market (`spread`, `total`, `ml`), candidate features are ranked and filtered with hard guards:

- `StabilityScore >= selector_stability_min`
- `SignConsistency >= selector_sign_consistency_min`
- `ImportanceDelta >= max(selector_permutation_delta_min, selector_ablation_delta_min)`

`ImportanceDelta` is the max of:

- `permutation_delta_mean`
- `drop_one_single_mean`
- `drop_one_group_mean`

Ranking order after hard filters:

1. `stability_score` (desc)
2. `roi_impact_mean` (desc; fallback to permutation delta when unavailable)
3. `importance_delta` (desc)
4. `univariate_mean` (desc)

## Correlation Pruning

Correlation pruning is applied on training rows only:

1. Build rolling folds for the market frame.
2. Union all fold `train_index` rows.
3. Compute absolute Spearman correlation on candidate feature columns.
4. Build correlation clusters where pairwise absolute correlation exceeds `selector_correlation_max`.
5. Keep one anchor per cluster:
   - highest `stability_score`, then
   - highest `roi_impact_mean`, then
   - highest `importance_delta`.

Tier caps are then applied:

- conservative: `feature_cap_conservative`
- balanced: `feature_cap_balanced`
- aggressive: `feature_cap_aggressive`

## Window Grid Testing

`model_lab/window_grid.py` evaluates fixed window combos:

- `W_4_8`
- `W_4_12`
- `W_4_8_12`
- `W_5_10`
- `W_6_11`
- `W_7_12`

Contract mapping:

- Each config maps to required window identifiers: `L4`, `L8`, `L12`, etc.
- Features are selected when their parsed window tokens (`L\d+`) are a subset of the config window set.
- Configs are BLOCKED when any required window identifier is missing from available features.

For each market/config, the evaluator:

1. Builds a model frame using only matching windowed features plus required market label/line columns.
2. Runs existing rolling folds only.
3. Scores by fold and aggregates weighted metrics.
4. Writes `window_grid_scorecard.csv`.

## Location-Aware Behavior (Home/Away/Neutral)

Selector output includes a location-aware variant check per market/tier.

Rules:

1. For each selected feature, expected split columns are `home_<base>` and `away_<base>`.
2. If all required split columns exist, location-aware status is `ACTIVE`.
3. If any required split column is missing, status is `BLOCKED` and report includes exact missing column names.

When active, location-aware frame behavior is:

- home games: prefer `home_*`
- away games: prefer `away_*` (using `home_away` flag when present)
- neutral games (`neutral_site=1`): use average of `home_*` and `away_*`

If split columns do not exist, the location-aware variant is not silently downgraded; it is explicitly blocked and reported.

## Outputs

Run directory artifacts:

- `data/model_lab_runs/<run_id>/window_grid_scorecard.csv`
- `data/model_lab_runs/<run_id>/feature_set_report.md`

Generated feature sets:

- `feature_sets/generated/spread_AUTO_V2_conservative.json`
- `feature_sets/generated/spread_AUTO_V2_balanced.json`
- `feature_sets/generated/spread_AUTO_V2_aggressive.json`
- `feature_sets/generated/total_AUTO_V2_conservative.json`
- `feature_sets/generated/total_AUTO_V2_balanced.json`
- `feature_sets/generated/total_AUTO_V2_aggressive.json`
- `feature_sets/generated/ml_AUTO_V2_conservative.json`
- `feature_sets/generated/ml_AUTO_V2_balanced.json`
- `feature_sets/generated/ml_AUTO_V2_aggressive.json`
