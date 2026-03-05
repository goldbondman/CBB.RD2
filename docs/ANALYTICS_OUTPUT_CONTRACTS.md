# Analytics Output Contracts

This contract defines planned module outputs for the feature-suite foundation.

Rules:
- each module writes one CSV under `data/`
- each module writes one execution summary markdown
- summaries should be generated with `scripts/exec_summary.py`

## Planned Modules (14)

| module | csv output path | exec summary path |
|---|---|---|
| `quality_baseline_audit` | `data/quality/quality_baseline_audit.csv` | `data/analytics/exec_summaries/quality_baseline_audit_exec_summary.md` |
| `quality_missingness_scan` | `data/quality/quality_missingness_scan.csv` | `data/analytics/exec_summaries/quality_missingness_scan_exec_summary.md` |
| `market_consensus_snapshot` | `data/market/market_consensus_snapshot.csv` | `data/analytics/exec_summaries/market_consensus_snapshot_exec_summary.md` |
| `market_move_flags` | `data/market/market_move_flags.csv` | `data/analytics/exec_summaries/market_move_flags_exec_summary.md` |
| `team_strength_snapshot` | `data/teams/team_strength_snapshot.csv` | `data/analytics/exec_summaries/team_strength_snapshot_exec_summary.md` |
| `team_form_windows` | `data/teams/team_form_windows.csv` | `data/analytics/exec_summaries/team_form_windows_exec_summary.md` |
| `team_rest_travel_index` | `data/teams/team_rest_travel_index.csv` | `data/analytics/exec_summaries/team_rest_travel_index_exec_summary.md` |
| `matchup_style_edges` | `data/matchups/matchup_style_edges.csv` | `data/analytics/exec_summaries/matchup_style_edges_exec_summary.md` |
| `matchup_availability_overlay` | `data/matchups/matchup_availability_overlay.csv` | `data/analytics/exec_summaries/matchup_availability_overlay_exec_summary.md` |
| `gate_line_value_checks` | `data/gates/gate_line_value_checks.csv` | `data/analytics/exec_summaries/gate_line_value_checks_exec_summary.md` |
| `gate_prediction_drift` | `data/gates/gate_prediction_drift.csv` | `data/analytics/exec_summaries/gate_prediction_drift_exec_summary.md` |
| `prediction_calibration_monitor` | `data/predictions/prediction_calibration_monitor.csv` | `data/analytics/exec_summaries/prediction_calibration_monitor_exec_summary.md` |
| `betting_card_candidates` | `data/betting/betting_card_candidates.csv` | `data/analytics/exec_summaries/betting_card_candidates_exec_summary.md` |
| `betting_portfolio_exposure` | `data/betting/betting_portfolio_exposure.csv` | `data/analytics/exec_summaries/betting_portfolio_exposure_exec_summary.md` |

## Shared Validation

Use:

```bash
bash .github/scripts/check_csv.sh <path> <min_rows> [required_cols_csv]
```

## Shared Exec Summary

Use:

```bash
python scripts/exec_summary.py \
  --input-csv <csv_path> \
  --output-md <summary_path> \
  --module-name <module_name> \
  --key-cols <col1,col2,...>
```
