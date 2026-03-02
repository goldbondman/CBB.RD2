# CSV Quality Audit

## Advanced Stats Health

Flagged `(file, column)` pairs:

| File | Column | non_null_count | unique_count | min | max | mean | std | Reasons |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ensemble_predictions_20260223.csv | ens_total | 32 | 1 | 144.2000 | 144.2000 | 144.2000 | 0.0000 | unique_count<=1, std==0, default_value_ratio>=95% (value=144.2, ratio=100.0%) |
| ensemble_predictions_20260224.csv | ens_total | 34 | 1 | 144.2000 | 144.2000 | 144.2000 | 0.0000 | unique_count<=1, std==0, default_value_ratio>=95% (value=144.2, ratio=100.0%) |
| ensemble_predictions_latest.csv | ens_total | 34 | 1 | 144.2000 | 144.2000 | 144.2000 | 0.0000 | unique_count<=1, std==0, default_value_ratio>=95% (value=144.2, ratio=100.0%) |
| predictions_history.csv | ens_total | 34 | 1 | 144.2000 | 144.2000 | 144.2000 | 0.0000 | unique_count<=1, std==0, default_value_ratio>=95% (value=144.2, ratio=100.0%) |
