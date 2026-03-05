# Edge Analyzer

## What It Does

`model_lab.edge_analyzer` explains where model edge and ROI come from using rolling forward folds only.

For one market/model pair, it produces:

- `edge_bucket_report.csv`
- `segment_report.csv`
- `worst_misses.csv`
- `EDGE_EXEC_SUMMARY.md`
- `data/model_lab_runs/<run_id>/edge_analyzer/run_manifest.json`

All bucket/segment metrics are computed on **test fold rows only**.

## Required Inputs

The analyzer uses existing Model Lab inputs and run artifacts:

- model predictions for one model (`DEFAULT_MODEL_NAMES` or `ensemble`)
- market frame (`spread_frame.csv` / `total_frame.csv` / `ml_frame.csv`) from run dir, or rebuilt if missing
- rolling folds from `model_lab.splits.build_rolling_folds`

Columns required by market:

- spread: `pred_spread`, `actual_margin`, `spread_line`
- total: `pred_total`, `actual_total`, `total_line`
- ml: model probability source (`pred_spread` or `pred_conf`), `home_won`, `home_ml` + `away_ml` for implied-edge buckets

Segment columns (optional, BLOCKED if missing/insufficient):

- `VOL`, `MTI`, `SCI`
- `home_away`, `neutral_site`
- `spread_line` (spread favorite/dog)
- `home_ml`, `away_ml` (ml implied buckets)
- `total_line` (total regime)

## How To Run

```bash
python -m model_lab.cli edge-analyze --run-id <id> --market <spread|total|ml> --model <name|ensemble> [--min-n 50] [--limit N]
```

Arguments:

- `--run-id`: run directory under `data/model_lab_runs/`
- `--market`: `spread`, `total`, or `ml`
- `--model`: one base model name or `ensemble`
- `--min-n`: small-sample warning threshold (default `50`)
- `--limit`: optional row limit before fold construction

## Output Interpretation

### edge_bucket_report.csv

Absolute edge bins:

- `[0,1)`, `[1,2)`, `[2,3)`, `[3,4)`, `[4,inf)`

Per bucket:

- `n`, `graded_n`, `hit_rate`, `roi`, `avg_edge`, `avg_abs_error`, `clv_mean`
- `small_sample_flag`
- `status` (`OK` or `BLOCKED`)

### segment_report.csv

Segment families:

- `VOL` quantiles (5 bins)
- `MTI` quantiles (5 bins)
- `SCI` quantiles (5 bins)
- `location` (home/away/neutral)
- `favorite_dog`
- `total_regime` (low/med/high for total market)

If a segment cannot be computed, it is written as `BLOCKED` with exact `missing_columns`.

### worst_misses.csv

Top 20 largest absolute errors on test rows with game context, prediction vs actual, line/edge, and snapshot feature values for:

- `ODI_diff`, `PEQ`, `SVI`, `WL`, `VOL`, `TIN`, `MTI`, `SCI`, `DPC`, `PXP`

Missing snapshot fields are reported in `snapshot_missing_columns`.

### EDGE_EXEC_SUMMARY.md

One-page summary:

- top 3 ROI segments
- bottom 3 bleed segments
- sample-size warnings
- blocked segment list with exact missing columns
- 3 rule-style gating recommendations

## Leakage Guardrails

- Fold definitions come from `build_rolling_folds`.
- Analyzer materializes fold membership and keeps only rows appearing in fold `test_index`.
- Buckets, quantiles, and segment aggregates are computed on this test-only table.
- No random split is used.

## Sample-Size Warnings

- Buckets/segments are marked `small_sample_flag=true` when `n < min_n`.
- Executive summary highlights small-sample groups.
- Recommended gates are intended to be used only when sample thresholds are met.
