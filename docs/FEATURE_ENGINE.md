# Feature Engine

## Overview
The advanced Feature Engine computes `team_game` and `matchup` metrics from canonical box score inputs only:
- `data/team_game_logs.csv`
- `data/player_game_logs.csv`

Hard rules enforced by design:
- No play-by-play usage.
- Required columns are never invented.
- Rolling windows are leak-free (`shift(1)`; prior games only).
- Z-scores are computed within-season only.
- Starter/bench logic is centralized in `pipeline/advanced_metrics/starter_bench_helper.py`.

## Module Layout
- `pipeline/advanced_metrics/feature_registry.py`
- `pipeline/advanced_metrics/feature_dag.py`
- `pipeline/advanced_metrics/integrity_gate.py`
- `pipeline/advanced_metrics/cache_layer.py`
- `pipeline/advanced_metrics/rolling_window_layer.py`
- `pipeline/advanced_metrics/metric_library.py`
- `pipeline/advanced_metrics/feature_sets.py`
- `pipeline/advanced_metrics/audit_logger.py`
- `pipeline/advanced_metrics/compute_runner.py`

## Registry Fields
Each `FeatureSpec` in `feature_registry.py` includes:
- `name`: canonical registry feature name.
- `grain`: `"team_game"` or `"matchup"`.
- `required_inputs`: required dataframe columns.
- `derived_inputs`: explicit derivation notes used by the feature.
- `dependencies`: registry feature dependencies.
- `compute_fn`: function reference from `metric_library.py`.
- `output_cols`: output column names emitted by this feature.
- `cache`:
  - `enabled`: bool
  - `key_fields`: e.g. `("season_id", "window_id", "feature_name")`
  - `version_hash`: SHA256 of compute function source + mapping payload

## How To Add A Feature
1. Add or update compute logic in `pipeline/advanced_metrics/metric_library.py`.
2. Register the feature in `pipeline/advanced_metrics/feature_registry.py` with:
   - exact `required_inputs`
   - explicit `derived_inputs`
   - `dependencies`
   - `output_cols`
3. If needed in model bundles, add the registry name to `pipeline/advanced_metrics/feature_sets.py`.
4. Run the engine via `compute_features(...)`.

## DAG and Blocking Behavior
- `feature_dag.py` validates dependencies and performs topological ordering.
- Dependency cycles raise an error and stop the run.
- `integrity_gate.py` validates `required_inputs` before compute.
- Missing required inputs mark a feature `BLOCKED` with exact missing columns.
- `BLOCKED` features are skipped, runner continues, and output cols are kept as `NaN`.

## Caching
Cache root:
- `data/cache/features/`

Partitioning:
- `grain/feature_name/season_{season_id}__window_{window_id}.csv`
- `grain/feature_name/season_{season_id}__window_{window_id}.manifest.json`

Manifest fields:
- `version_hash`
- `created_at_utc`
- `row_count`
- `schema_hash`
- `key_fields`

Invalidation:
- Cache is used only when manifest `version_hash` matches registry `version_hash`.
- Any compute function or mapping change updates `version_hash` and forces recompute.

## Rolling Window Layer
`rolling_window_layer.add_leak_free_windows` generates for each metric:
- `{metric}_season`
- `{metric}_L4`
- `{metric}_L7`
- `{metric}_L10`
- `{metric}_L12`
- `{metric}_L10_std`
- `{metric}_trend_L4_L10`
- `{metric}_trend_L10_season`

All windows use pregame-only history (`shift(1)`).

## Running Compute
Python API:
```python
from pipeline.advanced_metrics import compute_features

team_game_metrics, matchup_metrics = compute_features(
    season_id=None,
    limit_games=None,
    rebuild=False,
)
```

CLI:
```bash
py -3 -m pipeline.advanced_metrics.compute_runner --season-id 2025 --limit-games 100
```

Outputs:
- `data/team_game_metrics.csv`
- `data/matchup_metrics.csv`
- `data/feature_runs/feature_run_manifest.json`

## Interpreting BLOCKED and NaNs
- `BLOCKED` means a feature did not run because required input columns were missing.
- `NaN` in active features typically indicates insufficient prior history (for leak-free windows/z-scores) or missing source values.
- Run-level blocked features and NaN counts are recorded in `data/feature_runs/feature_run_manifest.json`.
