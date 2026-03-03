# CBB Betting Deliverables

This directory contains schemas and documentation for the "Bettor-Friendly" outputs produced by the prediction pipeline.

## Daily Deliverables

The following files are bundled into the `USER-deliverables` artifact produced by each pipeline run:

| File | Description |
|---|---|
| `bet_recs.csv` | **High Priority.** Betting recommendations where the model has a significant edge. |
| `daily_predictions.csv` | Full list of game predictions for the current/upcoming slate. |
| `matchup_preview.csv` | Detailed team-vs-team metrics and situational flags for today's games. |
| `upset_watch.csv` | Games where the underdog has a high probability of covering or winning outright. |
| `cbb_accuracy_report.csv` | Recent model performance and ROI tracking. |

## Schemas

Detailed column descriptions for these files can be found in `USER/schemas/`:

- [bets_schema.md](file:///c:/Users/brand/CBB.RD2/USER/schemas/bets_schema.md)
- [predictions_schema.md](file:///c:/Users/brand/CBB.RD2/USER/schemas/predictions_schema.md)
- [signals_summary_schema.md](file:///c:/Users/brand/CBB.RD2/USER/schemas/signals_summary_schema.md)

## Frequency

Deliverables are updated daily around **10:00 UTC** (2:00 AM PST).
