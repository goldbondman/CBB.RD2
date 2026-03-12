# Workflow Artifact Contracts

## INFRA-espn-data producer

- Producer workflow: `.github/workflows/update_espn_cbb.yml`
- Producing job: `update`
- Artifact name: `INFRA-espn-data`
- Core contract files (validated before upload):
  - `data/games.csv` (>= 1 row)
  - `data/team_game_weighted.csv` (>= 1 row)
  - `data/team_pretournament_snapshot.csv` (>= 1 row)

## INFRA-espn-data consumers

- `.github/workflows/cbb_predictions_rolling.yml`
  - Downloads `INFRA-espn-data` for prediction inputs.
- `.github/workflows/cbb_analytics.yml`
  - Best-effort download of `INFRA-espn-data` from `main`, then merges with other upstream artifacts.
  - If the artifact is missing or required ESPN CSVs are missing/invalid, analytics runs a local fallback:
    - `python espn_pipeline.py --days-back 3`
    - Re-validates required ESPN CSVs.
    - Re-uploads an `INFRA-espn-data` artifact from the analytics run.

## Analytics fallback behavior

- `cbb_analytics.yml` now self-heals when prior `update_espn_cbb` artifacts are unavailable.
- The workflow writes an integrity summary and a debug bundle artifact (`INFRA-cbb-analytics-debug`) with:
  - `data/` tree output (depth 2)
  - Key CSV row counts
  - Integrity report outputs

## INFRA-rolling-performance-dashboard producer

- Producer workflow: `.github/workflows/rolling_performance_dashboard.yml`
- Trigger chain:
  - `workflow_run` after successful `results-writer`
  - weekly schedule
  - manual dispatch
- Artifact name: `INFRA-rolling-performance-dashboard`
- Core outputs:
  - `data/performance_tracker.csv`
  - `data/edge_vs_actual_margin.csv`
  - `data/clv_tracker.csv`
  - `data/weekly_performance_summary_*.txt`

## Rolling tracker runtime notes

- Script entrypoint: `scripts/rolling_performance_dashboard.py`
- Required source input:
  - `data/reports/game_predictions_master.csv`
- Optional enrichment:
  - `data/plumbing/line_movement.csv` for CLV metrics
- If graded outcomes or CLV fields are unavailable, output files are still written with explicit provisional/blocked rows instead of failing silently.
