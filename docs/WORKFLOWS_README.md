# CBB Workflow Guide

## What These Workflows Do

- `cbb_predictions_rolling.yml`
  - Builds rolling game predictions from ESPN pipeline data.
  - Optionally runs ensemble, Monte Carlo, and derived CSV enrichments.
  - Produces core prediction CSVs and a ready marker.

- `cbb_analytics.yml`
  - Consumes market lines + ESPN data + predictions outputs.
  - Runs backtesting (mode-dependent), grading, results tracking, and accuracy reporting.
  - Produces analytics CSVs and a ready marker.

- `player_overlay.yml`
  - Runs overlay model unit tests.
  - In standalone mode, runs player overlay scoring on `player_overlay_input.csv`.
  - Produces overlay predictions and a ready marker.

## Required Inputs Before Each Workflow

- `cbb_predictions_rolling`
  - Required: `data/games.csv`, `data/team_game_weighted.csv`
  - Optional enrichers require:
    - travel fatigue: `data/team_game_logs.csv`, `data/venue_geocodes.csv`
    - player matchup: `data/player_game_metrics.csv`

- `cbb_analytics`
  - Required: `data/market_lines.csv`, `data/team_game_weighted.csv`
  - ESPN dependency resolution:
    - Tier A: download `INFRA-espn-data` from latest successful `update_espn_cbb.yml` run on `main`.
    - Tier B fallback: use committed `data/games.csv` + `data/team_game_weighted.csv` only if validation passes.
  - Required predictions dependency:
    - `data/predictions_with_context.csv` OR `data/predictions_combined_latest.csv`
  - Freshness gate: predictions input file must be <= 36 hours old.

- `player_overlay`
  - Required: `data/player_overlay_input.csv` (or explicit `--input` path)
  - If missing, workflow fails as BLOCKED and reports exact missing files.

## Main CSV Outputs

- Predictions workflow:
  - `data/predictions_primary.csv`
  - `data/predictions_latest.csv`
  - `data/predictions_combined_latest.csv`
  - `data/predictions_history.csv`

- Analytics workflow:
  - `data/results_log.csv`
  - `data/results_summary.csv`
  - optional: `data/backtest_results_latest.csv`, `data/backtest_training_data.csv`

- Player overlay workflow:
  - `data/player_overlay_predictions.csv`

## Integrity Reports

Each workflow writes standardized reports to:

- `data/integrity_reports/<workflow_name>/<run_id>/integrity_report.json`
- `data/integrity_reports/<workflow_name>/<run_id>/INTEGRITY_EXEC_SUMMARY.md`

Report includes:
- file path
- row count and column count
- missing required columns
- null-rate for configured key columns
- oldest/newest date when date column is present

If a required file/column check fails, status is `BLOCKED` and the workflow fails.

`cbb_analytics` blocked behavior:
- Writes `data/integrity_reports/cbb_analytics/<run_id>/INTEGRITY_EXEC_SUMMARY.md` with missing artifact/files and required remediation.
- `workflow_dispatch`: exits 0 after writing the BLOCKED report.
- `schedule`/automated events: exits 1 after writing the BLOCKED report.
- Remediation: run `Update ESPN CBB Data` successfully on `main` to publish `INFRA-espn-data`.

## Ordered Execution

- Parent orchestrator: `.github/workflows/cbb_master_nightly.yml`
  - order: predictions -> analytics -> player_overlay
  - enforced via `needs:` with reusable workflow calls (`workflow_call`)

Existing standalone triggers remain for backward compatibility.

## Local Smoke Run

Run:

```bash
python scripts/analytics_smoke.py
```

```bash
python scripts/workflow_smoke.py
```

Optional date:

```bash
python scripts/workflow_smoke.py --date 20260304
```

Live/networked prediction pull (optional):

```bash
python scripts/workflow_smoke.py --live --date 20260304
```

Smoke outputs:
- `data/smoke/<timestamp>/` (copied output CSV snapshots)
- `data/smoke/<timestamp>/integrity_reports/` (integrity report snapshots)
