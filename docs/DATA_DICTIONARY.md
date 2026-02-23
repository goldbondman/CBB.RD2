# Data Dictionary

## `team_game_metrics.csv`
- **Purpose:** Canonical team-level per-game metrics used for feature generation.
- **Row definition:** One team in one game.
- **Update frequency:** Every ingestion run.
- **Key columns:** `season:int`, `game_id:str`, `team_id:str`, `opponent_id:str`, `possessions:float`, `off_rating:float`, `def_rating:float`.
- **Writers:** `ingestion/espn_pipeline.py`
- **Readers:** `features/espn_weighted_metrics.py`, `features/team_form_snapshot.py`, `models/espn_prediction_runner.py`

## `predictions_combined_latest.csv`
- **Purpose:** Unified latest prediction output used downstream for grading and exports.
- **Row definition:** One model prediction per game per model version snapshot.
- **Update frequency:** Every prediction run.
- **Key columns:** `game_id:str`, `model_version:str`, `pred_spread:float`, `pred_total:float`, `confidence:float`, `created_at:datetime`.
- **Writers:** `models/espn_prediction_runner.py`, `enrichment/predictions_with_context.py`
- **Readers:** `ingestion/cbb_results_tracker.py`, `enrichment/edge_history.py`

## `predictions_graded.csv`
- **Purpose:** Prediction outcomes with post-game correctness labels.
- **Row definition:** One graded prediction against a final result.
- **Update frequency:** Daily grading cycle.
- **Key columns:** `game_id:str`, `model_version:str`, `grade:str`, `actual_margin:float`, `edge:float`.
- **Writers:** `evaluation/predictions_graded.py`
- **Readers:** `evaluation/model_accuracy_weekly.py`, `evaluation/bias_detector.py`

## `results_log.csv`
- **Purpose:** Rolling audit log of tracked prediction results.
- **Row definition:** One tracked game result event.
- **Update frequency:** Daily results tracker run.
- **Key columns:** `date:date`, `game_id:str`, `status:str`, `result:str`, `pulled_at:datetime`.
- **Writers:** `ingestion/cbb_results_tracker.py`
- **Readers:** `evaluation/model_accuracy_weekly.py`, `evaluation/optimize_weights.py`

## `model_bias_table.csv`
- **Purpose:** Runtime model-bias calibration lookup.
- **Row definition:** One model/version bias bucket entry.
- **Update frequency:** Tune cycle.
- **Key columns:** `model_version:str`, `bias_metric:float`, `updated_at:datetime`.
- **Writers:** `evaluation/bias_detector.py`
- **Readers:** `evaluation/optimize_weights.py`, `models/espn_prediction_runner.py`

## `edge_history.csv`
- **Purpose:** Historical edge tracking for prediction lifecycle.
- **Row definition:** One edge record per game/model state transition.
- **Update frequency:** Predict and grade cycles.
- **Key columns:** `game_id:str`, `model_version:str`, `mode:str`, `edge_value:float`, `timestamp:datetime`.
- **Writers:** `enrichment/edge_history.py`
- **Readers:** `evaluation/model_accuracy_weekly.py`

## `team_form_snapshot.csv`
- **Purpose:** Team recent-form snapshot features for modeling.
- **Row definition:** One team snapshot at pipeline run time.
- **Update frequency:** Feature refresh cycle.
- **Key columns:** `team_id:str`, `season:int`, `form_window:str`, `off_trend:float`, `def_trend:float`.
- **Writers:** `features/team_form_snapshot.py`
- **Readers:** `models/espn_prediction_runner.py`

## `model_weights.json`
- **Purpose:** Ensemble/model weight configuration used at runtime.
- **Row definition:** JSON object keyed by component model name.
- **Update frequency:** Tune cycle or manual update.
- **Key fields:** `model_name -> weight:float`, optional metadata keys.
- **Writers:** `evaluation/optimize_weights.py`
- **Readers:** `models/cbb_ensemble.py`, `models/espn_prediction_runner.py`
