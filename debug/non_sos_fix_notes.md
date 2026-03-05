# Non-SOS Fix Rerun Notes

## Scope
- This patch set does **not** modify `espn_sos.py`.
- Changes cover D1 filtering, joint-model sanity diagnostics, missing-prediction coverage reporting, contract-runner error clarity, and workflow artifact stability.

## Local rerun
```bash
python .github/scripts/run_pipeline_contract.py \
  --contract .github/pipeline_contracts/espn_pipeline_contract.json \
  --job update \
  --days-back 3 \
  --manifest data/perplexity_models_manifest.json \
  --stages "espn_pipeline,rotation_features,situational_features,advanced_metrics_builder,joint_models_predictions"
```

## Expected debug outputs
- `debug/d1_filter_report.csv`
- `debug/sanity_outliers.csv`
- `debug/missing_predictions_report.csv`
- `debug/feature_scale_checks.json`
- `debug/contract_missing_outputs.json` (on contract failure)
- `debug/oversize_outputs.json` (only if a CSV exceeds size guard)
- `data/run_manifest.json` (from workflow run-manifest step)

