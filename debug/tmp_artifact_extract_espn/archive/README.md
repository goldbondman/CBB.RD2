# Data archive layout

This folder keeps historical snapshot CSVs out of `data/` root so the active pipeline outputs stay easy to find.

## Subfolders

- `rankings_snapshots/`: timestamped `cbb_rankings_*.csv` exports.
- `predictions_daily/`: dated `predictions_*.csv` exports (keeps `predictions_latest.csv` in `data/`).
- `ensemble_daily/`: dated `ensemble_predictions_*.csv` exports (keeps `ensemble_predictions_latest.csv` in `data/`).

## Notes

- No schema/content changes were made; files were moved only.
- Active/current files remain in `data/` for compatibility with existing pipeline code.
