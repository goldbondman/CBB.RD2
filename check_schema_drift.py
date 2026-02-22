import json
import pathlib
import sys

import pandas as pd

registry = json.loads(pathlib.Path('data/schema_registry.json').read_text())
drifts = []

for path_str, expected in registry.items():
    path = pathlib.Path(path_str)
    if not path.exists():
        drifts.append({'file': path_str, 'type': 'MISSING_FILE', 'severity': 'CRITICAL', 'detail': 'File no longer exists'})
        continue
    try:
        df = pd.read_csv(path, nrows=500, low_memory=False)
    except Exception as e:
        drifts.append({'file': path_str, 'type': 'UNREADABLE', 'severity': 'CRITICAL', 'detail': str(e)})
        continue

    expected_cols = set(expected['columns'].keys())
    actual_cols = set(df.columns)

    for col in expected_cols - actual_cols:
        drifts.append({'file': path_str, 'type': 'COLUMN_DROPPED', 'severity': 'CRITICAL', 'detail': f'{col} was present, now missing'})

    for col in actual_cols - expected_cols:
        drifts.append({'file': path_str, 'type': 'COLUMN_ADDED', 'severity': 'INFO', 'detail': f'{col} is new'})

    for col in expected_cols & actual_cols:
        exp = expected['columns'][col]
        actual_null = round(df[col].isna().mean(), 4)
        exp_null = exp['null_rate']
        if actual_null - exp_null > 0.30:
            drifts.append({
                'file': path_str,
                'type': 'NULL_RATE_SPIKE',
                'severity': 'WARNING',
                'detail': f'{col}: null rate {exp_null*100:.0f}% → {actual_null*100:.0f}%',
            })

        actual_dtype = str(df[col].dtype)
        if actual_dtype != exp['dtype']:
            drifts.append({
                'file': path_str,
                'type': 'DTYPE_CHANGED',
                'severity': 'WARNING',
                'detail': f'{col}: {exp["dtype"]} → {actual_dtype}',
            })

critical = [d for d in drifts if d['severity'] == 'CRITICAL']
warnings = [d for d in drifts if d['severity'] == 'WARNING']
info = [d for d in drifts if d['severity'] == 'INFO']

print(f"Schema drift check: {len(critical)} critical, {len(warnings)} warnings, {len(info)} info")
for d in drifts:
    print(f"  [{d['severity']}] {d['file']} | {d['type']}: {d['detail']}")

pathlib.Path('data/schema_drift_report.json').write_text(json.dumps(drifts, indent=2))

if critical:
    print('\nCRITICAL drift detected — pipeline outputs have changed unexpectedly')
    sys.exit(1)
