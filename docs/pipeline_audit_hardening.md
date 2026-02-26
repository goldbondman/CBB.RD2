# Pipeline Audit + Hardening Pass

## 1) CURRENT STATE MAP

### Entrypoints and IO

| Stage | Entrypoint(s) | Main Inputs | Main Outputs |
|---|---|---|---|
| Prediction generation | `espn_prediction_runner.py`, `cbb_ensemble.py` | `data/team_game_weighted.csv` (fallback metrics/logs), `data/games.csv`, `data/market_lines.csv` | `data/predictions_*.csv`, `data/predictions_latest.csv`, `data/ensemble_predictions_latest.csv` |
| Derived CSV builders | `build_derived_csvs.py`, `build_backtest_csvs.py` | `data/results_log.csv`, model/pred files | `data/csv/*.csv`, `data/results_log_graded.csv` |
| Backtesting | `cbb_backtester.py`, `build_backtest_csvs.py` | `data/results_log.csv`, `data/predictions_combined_latest.csv` | summary/calibration/edge CSVs |
| Evaluation/reporting | `evaluation/predictions_graded.py`, `evaluation/model_accuracy_weekly.py` | graded logs/predictions | report CSVs |
| Recommendations | `enrichment/predictions_with_context.py` | predictions + lines + context features | `data/predictions_with_context.csv`, `data/csv/bet_recs.csv` |
| Training / retraining | `optimize_weights.py`, `models/weight_optimizer.py` | historical outcomes + model splits | weight artifacts |

### Identity keys observed
- Mixed keys: `game_id` and `event_id` both appear in prediction paths.
- Team identity usually `team_id` / `home_team_id` / `away_team_id`.
- Time identity usually `game_datetime_utc`.

### Freshness and schema behavior
- Existing code often uses fallback and `*_latest.csv` implicit dependencies.
- Freshness checks are inconsistent and mostly soft warnings.
- Schema drift risk exists due to optional columns and no canonical prediction schema enforcement.

## 2) PIPELINE FLOW DIAGRAM

```text
raw ESPN pulls + enrichers
        |
        v
  data/games.csv + team_game_*.csv + market_lines.csv
        |
        +--> espn_prediction_runner.py --> predictions_YYYYMMDD.csv --> predictions_latest.csv
        |                                              |
        |                                              +--> cbb_ensemble.py --> ensemble_predictions_latest.csv
        |
        +--> enrichment/predictions_with_context.py --> predictions_with_context.csv --> bet_recs.csv
        |
        +--> cbb_results_tracker.py --> results_log.csv --> build_backtest_csvs.py --> results_log_graded.csv + backtest csvs
                                                                    |
                                                                    +--> evaluation scripts

Forks / risks:
- Predict path and backtest path each pick their own "latest" files.
- Duplicate grading/eval logic exists in tracker/backtest/evaluation scripts.
- Leakage risk where file timestamps are not compared against a strict as_of.
```

## 3) RISK REGISTER (RANKED)

1. **Leakage Risk (High)**
   - Impact: inflated backtest and false confidence.
   - Detection: fail integrity if `games.game_datetime_utc` or `market_lines.pulled_at` > `as_of`.
   - Fix: enforced integrity gate in `pipeline/integrity.py`.

2. **Identity Risk (High)**
   - Impact: wrong joins and duplicate predictions per game.
   - Detection: uniqueness check on prediction `game_id` and referential check to schedule.
   - Fix: gate check in backtest mode.

3. **Non-Determinism (Medium-High)**
   - Impact: irreproducible run outputs.
   - Detection: missing run manifest / context tags.
   - Fix: deterministic `RunContext` + `run_manifest.json`.

4. **Evaluation Methodology Drift (Medium)**
   - Impact: inconsistent ROI/accuracy reporting.
   - Detection: missing standard output files.
   - Fix: standard suite writes `evaluation.csv` + `evaluation.json`.

5. **Update Thrash (Medium)**
   - Impact: overreactive retrain/promotion behavior.
   - Detection: policy check over recent run tags.
   - Fix: 3-of-5 BAD + games + cooldown governance in `pipeline/update_policy.py`.

## 4) HARDENING PLAN (MINIMAL CHANGE)

Implemented minimal orchestration package `pipeline/` with:
- **RunContext**: `run_id`, `as_of`, `model_version`, `feature_version`, `prediction_timestamp`.
- **Integrity Gate** (fail-fast): required files/columns, uniqueness, referential checks, cutoff checks.
- **Manifest + artifacting**: writes `data/artifacts/run_manifest.json` and `data/artifacts/integrity.json`.
- **Standard evaluation**: writes `data/evaluation.csv` and `data/evaluation.json`.
- **Update governance**: `pipeline update-check` with 3-of-5 BAD, min games, cooldown, force override support.
- **Decision log**: unresolved semantic ambiguity captured in `DECISIONS_NEEDED.md`.

## 5) IMPLEMENTATION DIFFS
- Added new package: `pipeline/` (`__main__`, CLI, integrity, evaluation, run context, update policy).
- Added tests for integrity and update governance rules.
- Added decision-log file creation via audit command.

## 6) SAMPLE RUN OUTPUT
- `python -m pipeline audit` writes `data/artifacts/current_state_map.json`.
- `python -m pipeline run --mode backtest --start YYYY-MM-DD` writes:
  - `data/artifacts/integrity.json`
  - `data/artifacts/run_manifest.json`
  - `data/evaluation.csv`
  - `data/evaluation.json`

## 7) UPDATE POLICY SUMMARY
A run is BAD when >=2 degradation signals fire:
- ROI drop (`ROI_DROP_ABS=0.03` or ROI < -2%)
- CLV/Calibration drift (`CLV_SPREAD_DROP_PTS=0.25` or `BRIER_WORSEN=0.01`)
- Error drift (`SPREAD_MAE_WORSEN=0.35` or `TOTAL_MAE_WORSEN=1.25`)

Promotion eligible only if all true:
- BAD in >=3 of last 5 runs
- >=150 graded games since last update
- >=7 days since last promotion

Manual override:
- `python -m pipeline update-check --force-update --override-reason "..."`
