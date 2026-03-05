# Perplexity Artifact File Map

Scope: `CBB.RD2/.github/workflows/cbb_perplexity_models.yml` and `CBB.RD2/.github/pipeline_contracts/espn_pipeline_contract.json`.

| File | Producer stage | Producer command/script | Expected write path | Expected columns | Expected min rows |
|---|---|---|---|---|---|
| `advanced_metrics.csv` | `advanced_metrics_builder` | `python -m pipeline.advanced_metrics.build_advanced_metrics` (`pipeline/advanced_metrics/build_advanced_metrics.py`) | `data/advanced_metrics.csv` | `event_id, team_id` | `0` (`allow_empty_when_no_games=true`) |
| `predictions_joint_latest.csv` | `joint_models_predictions` | `python -m model_lab.joint_models` (`model_lab/joint_models.py`) | `data/predictions_joint_latest.csv` | `game_id, pred_total, pred_margin` | `0` (`allow_empty_when_no_games=true`) |
| `predictions_joint_snapshots.csv` | `joint_models_predictions` | `python -m model_lab.joint_models` (`model_lab/joint_models.py`) | `data/predictions_joint_snapshots.csv` | `game_id, generated_at_utc` | `0` (`allow_empty_when_no_games=true`) |

## Consumers

- `Validate required outputs` step in `cbb_perplexity_models.yml` (`.github/scripts/validate_perplexity_outputs.sh`)
- `Health summary` step in `cbb_perplexity_models.yml`
- `Upload model-output artifact bundle` step in `cbb_perplexity_models.yml`
- `Commit and push model outputs to main` step in `cbb_perplexity_models.yml`

## Near-miss filenames found in repo (not used by this workflow)

- `data/predictions_latest.csv`
- `data/predictions_mc_latest.csv`
- `data/predictions_joint.csv` (reference variants only; not the contract output)
- `data/team_game_metrics_advanced.csv` (`metrics_advanced` style output)
- `*snapshots.csv` variants such as `data/market_lines_snapshots.csv`
