# Rolling Performance Dashboard Runbook

## Purpose
Operational tracker for rolling model performance after `results_writer.py` updates outcomes.

## Command
```bash
py -3 scripts/rolling_performance_dashboard.py \
  --predictions-path data/reports/game_predictions_master.csv \
  --output-dir data \
  --run-layer-monitor true
```

## Output Files
- `data/performance_tracker.csv`
- `data/edge_vs_actual_margin.csv`
- `data/clv_tracker.csv`
- `data/weekly_performance_summary_YYYY-MM-DD.txt`

## Dimensions Tracked
- market (`spread`, `total`, `moneyline`)
- bet type (`ats`, `ou`, `ml`)
- confidence tier
- situational layer fired
- PES tier
- seed matchup
- phase

## Windows Tracked
- `last_10`
- `last_25`
- `last_50`
- `last_100`
- `season_to_date`
- `all_time`

## Guardrails
- If graded outcomes are missing, tracker emits empty/provisional outputs instead of fabricating results.
- CLV output is marked provisional when `data/plumbing/line_movement.csv` or required CLV columns are unavailable.
- Optional layer monitor hook executes `layer_performance_monitor.py` and logs PASS/WARN without stopping tracker export.
