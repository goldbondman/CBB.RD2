# Data Quality Audits

## CSV quality audit (`dq_report_v2`)

Run the CSV audit script from the repo root:

```bash
python scripts/audit_csv_quality.py
```

### What it does
- Recursively scans `data/` for `.csv` files.
- Skips archive-like paths (`archive`, `backups`, `old`, `tmp`).
- Skips files larger than 250MB (and reports each skip).
- Computes per-file and per-column quality metrics.
- Writes:
  - `data/dq_report_v2.md` (human-readable markdown report)
  - `data/dq_report_v2.csv` (machine-readable per-column output)

### Thresholds
- `SMALL_ROW_THRESHOLD` controls `small_rows_flag`.
- Default is `50` rows.
- Override for a run:

```bash
SMALL_ROW_THRESHOLD=100 python scripts/audit_csv_quality.py
```

### How to interpret key flags
- `small_rows_flag`: file has fewer rows than `SMALL_ROW_THRESHOLD`.
- `all_null_flag` (column): every row in that column is null.
- `all_zero_flag` (column): numeric column values are all zero.
- `constant_flag` (column): non-null values collapse to a single unique value.
- `suspicious_constant_columns` (file): at least 30% of columns are constant.

Use the report's **Top Issues** and **Column Detail** sections to identify files/columns to clean before downstream modeling.
