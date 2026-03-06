# CBB Analytics Usage Map

## Trigger conditions
- Schedule: `0 8 * * *` (daily tracker_only), `0 10 * * 1` (weekly backtest_only)
- workflow_run: after Market Lines Capture or CBB Predictions Rolling completes successfully
- workflow_dispatch: manual with run_mode, start_date, end_date
- workflow_call: callable from other workflows

## Artifact inputs
| Artifact | Source Workflow | Required | Notes |
|---|---|---|---|
| INFRA-espn-data | update_espn_cbb.yml | Optional (Tier A) | continue-on-error; fallback pipeline runs if missing |
| INFRA-market-lines | market_lines.yml | Required | Downloaded by run ID from latest successful run |
| INFRA-predictions-with-context | market_lines.yml | Required | Same run ID as INFRA-market-lines |
| INFRA-predictions-rolling | cbb_predictions_rolling.yml | Required | Contains prediction CSVs + ready marker |
| INFRA-ready-cbb-predictions-rolling | cbb_predictions_rolling.yml | **Optional** (as of fix) | Redundant marker; main artifact already contains it |

## Output artifacts
| Artifact | Contents | Retention |
|---|---|---|
| INFRA-cbb-analytics-integrity | Integrity reports | 90 days |
| INFRA-cbb-analytics-debug | Debug state, integrity reports | 30 days |
| INFRA-analytics-results | results_log.csv, backtest outputs, context overlays, gate results, segment performance | 90 days |
| INFRA-ready-cbb-analytics | Ready marker JSON | 30 days |

## Scripts/functions invoked
- `.github/scripts/download_latest_artifact.py` — artifact resolution
- `.github/scripts/check_csv.sh` — CSV validation
- `.github/scripts/data_integrity_report.py` — integrity reporting
- `espn_pipeline.py` — ESPN data fallback
- `context_layers/run_all.py` — context overlay builder
- `scripts/validate_data_infrastructure.py` — health check
- `ingestion/historical_backfill.py` — market lines backfill
- `cbb_backtester.py` — backtesting engine
- `evaluation/predictions_graded.py` — prediction grading
- `build_training_data.py` — ML training data assembly
- `cbb_results_tracker.py` — results tracking
- `cbb_season_summaries.py` — season summaries
- `cbb_accuracy_report.py` — accuracy reporting
- `scripts/gate_builder.py` — gate construction
- `scripts/segment_performance_explorer.py` — segment analysis

## Known consumers
- INFRA-ready-cbb-analytics: No downstream workflow consumers found in repo
- Committed outputs (results_log.csv, etc.): consumed by repo data files for historical tracking

## Current inferred purpose
cbb_analytics.yml is the **post-prediction analytics and results tracking workflow**. It:
1. Merges staged artifacts from upstream workflows into working data
2. Runs ESPN fallback if upstream ESPN data is unavailable
3. Executes backtesting, grading, training data generation, results tracking
4. Produces analytics reports (accuracy, gates, segments)
5. Commits curated analytics outputs back to the repo
