# Repo Inventory Summary

Generated: 2026-03-12 (America/Los_Angeles)

## Canonical Paths
- historical_warehouse.parquet: `C:/Users/brand/OneDrive/Desktop/CBB/data/historical_warehouse.parquet`
- situational layer backtester: `C:/Users/brand/OneDrive/Desktop/CBB/CBB.RD2/situational_layer_backtesting.py`
- PES backtester: `C:/Users/brand/OneDrive/Desktop/CBB/CBB.RD2/pes_backtesting.py`
- upset candidate inputs:
  - `C:/Users/brand/OneDrive/Desktop/CBB/CBB.RD2/data/layer_backtests/layer_backtest_input.csv`
  - `C:/Users/brand/OneDrive/Desktop/CBB/CBB.RD2/data/upset_watch.csv`
- layer registry: `C:/Users/brand/OneDrive/Desktop/CBB/CBB.RD2/data/layer_registry.csv`
- game predictions master: `C:/Users/brand/OneDrive/Desktop/CBB/CBB.RD2/data/reports/game_predictions_master.csv`
- results writer: `C:/Users/brand/OneDrive/Desktop/CBB/CBB.RD2/results_writer.py`

## Schema Snapshot
- `historical_warehouse.parquet`: 393 rows, 2054 columns.
  - `market_spread` non-null rows: 0 (warehouse-only ATS grading is blocked without fallback join).
- `backtest_results_latest.csv`: contains model and realized outcomes used for fallback side-level backtesting.
- `game_predictions_master.csv`: 21 rows, 73 columns.
  - Graded outcome fields (`actual_winner`, `covered_team_a`, `covered_team_b`, `final_margin`) are not currently present.

## Existing Workflow/Command Inventory
- Existing workflow directly invoking results writer:
  - `.github/workflows/results_writer.yml` -> `python results_writer.py ...`
- New phase scripts added in `scripts/`:
  - `scripts/backtesting_activation.py`
  - `scripts/calibration_attribution.py`
  - `scripts/rolling_performance_dashboard.py`
  - shared helpers: `scripts/phase_common.py`

## Execution Commands (this run)
- Prereqs / integrity:
  - `py -3 scripts/audit_csv_quality.py`
  - `py -3 scripts/ci_data_quality_gate.py`
  - `py -3 scripts/validate_data_infrastructure.py`
  - `py -3 scripts/verify_pipeline_integrity.py`
  - `py -3 check_schema_drift.py`
- Phase 2:
  - `py -3 scripts/backtesting_activation.py --output-dir data/layer_backtests --min-sample 50 --edge-threshold 4.0 --p-threshold 0.05`
- Phase 3:
  - `py -3 scripts/calibration_attribution.py --output-dir data`
- Phase 4:
  - `py -3 scripts/rolling_performance_dashboard.py --predictions-path data/reports/game_predictions_master.csv --output-dir data --run-layer-monitor true`

## Notes
- Backtesting activation is fully wired and executable with fallback merge logic (warehouse + backtest outputs).
- Seed-specific calibration remains blocked by missing seed-linked probability/outcome artifact.
