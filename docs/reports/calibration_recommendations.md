# Calibration and Attribution Recommendations

Generated at UTC: 2026-03-12T18:00:47Z

## Outputs
- buckets: C:\Users\brand\OneDrive\Desktop\CBB\CBB.RD2\data\calibration_buckets.csv
- report: C:\Users\brand\OneDrive\Desktop\CBB\CBB.RD2\data\calibration_report.csv
- attribution: C:\Users\brand\OneDrive\Desktop\CBB\CBB.RD2\data\performance_attribution.csv
- redundancy: C:\Users\brand\OneDrive\Desktop\CBB\CBB.RD2\data\metric_redundancy_report.csv
- phase_performance: C:\Users\brand\OneDrive\Desktop\CBB\CBB.RD2\data\phase_performance.csv
- seed_calibration: C:\Users\brand\OneDrive\Desktop\CBB\CBB.RD2\data\seed_calibration.csv
- pes_incremental_lift: C:\Users\brand\OneDrive\Desktop\CBB\CBB.RD2\data\pes_incremental_lift.csv
- edge_decay: C:\Users\brand\OneDrive\Desktop\CBB\CBB.RD2\data\edge_decay_report.csv

## Recommended Actions
1. Reweight submodels using `performance_attribution.csv` after filtering to models with stable ATS sample >= 200.
2. Hold seed-specific deployment claims until a seed-linked probability artifact exists (`seed_calibration.csv` currently BLOCKED).
3. Use `edge_decay_report.csv` to reduce stake when `move_against_model=true` and move bucket >= 1.0.
4. Re-run this report after market-line backfill expands ATS sample beyond the current constrained set.
