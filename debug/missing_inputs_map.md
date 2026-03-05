# Missing Inputs Map

| input_name | meaning/definition | where expected (file + column) | producer stage | producer script/function | consumers |
|---|---|---|---|---|---|
| `lns_diff` | Home minus away LNS (lineup/network stability proxy) differential. | `model_lab/joint_models.py` weighted spread features (`spread_features['lns_diff']`) and model status blocks. | `joint_models_predictions` | `model_lab.joint_models.compute_spread_features` (from `adv_LNS` or fallback `form_rating/net_rtg_l10`) | `model_lab/joint_models.py` spread model weighting (`SPREAD_WEIGHTS`) |
| `odi_star_diff` | Home minus away ODI* offensive/defensive interaction differential. | `model_lab/joint_models.py` (`spread_features['odi_star_diff']`) | `advanced_metrics_builder` + `joint_models_predictions` fallback | `pipeline.advanced_metrics.build_advanced_metrics` (`ODI_star`) and `model_lab.joint_models._team_fallback_features` | `model_lab/joint_models.py` spread model weighting |
| `posw` | Pace-overlap stress weight; spread side uses differential, totals side uses summed value. | `model_lab/joint_models.py` (`spread_features['posw']`, `totals_features['posw']`) | `advanced_metrics_builder` + `joint_models_predictions` fallback | `pipeline.advanced_metrics.build_advanced_metrics` (`POSW`) and fallback in `model_lab.joint_models._team_fallback_features` | `model_lab/joint_models.py` spread + totals weights |
| `sch` | Style-clash harmonic (pace/3PA/FTr/size interaction). | `model_lab/joint_models.py` (`totals_features['sch']`) | `advanced_metrics_builder` + `joint_models_predictions` fallback | `pipeline.advanced_metrics.build_advanced_metrics` (`SCH`) and fallback `model_lab.joint_models.compute_totals_features` | `model_lab/joint_models.py` totals weights |
| `tc_diff` | Tempo-control differential (home minus away). | `model_lab/joint_models.py` (`totals_features['tc_diff']`) | `advanced_metrics_builder` + `joint_models_predictions` fallback | `pipeline.advanced_metrics.build_advanced_metrics` (`TC`) and fallback `model_lab.joint_models.compute_totals_features` | `model_lab/joint_models.py` totals weights |
| `vol_diff` | Home minus away volatility differential. | `model_lab/joint_models.py` (`spread_features['vol_diff']`) | `advanced_metrics_builder` + `joint_models_predictions` fallback | `pipeline.advanced_metrics.build_advanced_metrics` (`VOL`) and fallback in `model_lab.joint_models.compute_spread_features` | `model_lab/joint_models.py` spread weights |
| `wl` | Whistle-load signal (FT PPP and FTr pressure interaction). | `model_lab/joint_models.py` (`totals_features['wl']`) | `advanced_metrics_builder` + `joint_models_predictions` fallback | `pipeline.advanced_metrics.build_advanced_metrics` (`WL`) and fallback `model_lab.joint_models._team_fallback_features` | `model_lab/joint_models.py` totals weights |
| `team_snapshot` | Per-team latest merged state (weighted + advanced + rotation) consumed by joint model row builder. | `data/team_snapshot.csv` (required `team_id`; feature columns from weighted/advanced/rotation) | `joint_models_predictions` (first-class artifact after fix) | `model_lab.joint_models._latest_team_snapshot` + `_write_team_snapshot_artifact` | `model_lab.joint_models.run_joint_predictions` |

## Notes
- `BLOCKED_MISSING_INPUT` originates in:
  - `model_lab/joint_models.py` (`predict_game_joint`, low coverage gate; `run_joint_predictions`, team snapshot gate)
  - `pipeline/advanced_metrics/build_advanced_metrics.py` (`metric_status_*` per-row blocks).
- No orphaned tokens were found for this set after alias mapping was added (`FEATURE_ALIAS_REGISTRY` in `model_lab/joint_models.py`).
- Expected source paths used in blocked diagnostics:
  - `data/team_snapshot.csv`
  - `data/team_game_weighted.csv`
  - `data/advanced_metrics.csv`
