import json
import pathlib
from datetime import datetime, timezone

import pandas as pd

registry = {}
for path in sorted(pathlib.Path('data').rglob('*.csv')):
    # Skip if in data/csv/ to avoid duplication with data/
    if 'data/csv' in str(path.as_posix()):
        continue

    try:
        df = pd.read_csv(path, nrows=500, low_memory=False)
        registry[str(path)] = {
            'columns': {
                col: {
                    'dtype': str(df[col].dtype),
                    'null_rate': round(df[col].isna().mean(), 4),
                    'sample_values': df[col].dropna().head(3).tolist(),
                    'null_rate_note': 'approximate (based on first 500 rows)' if len(df) >= 500 else 'exact'
                }
                for col in df.columns
            },
            'row_count_sample': len(df),
            'registered_at': datetime.now(timezone.utc).isoformat(),
        }
        print(f"[OK] {path}: {len(df.columns)} columns")
    except Exception as e:
        print(f"[FAIL] {path}: {e}")

# Handle numpy types in JSON serialization
def numpy_encoder(obj):
    if hasattr(obj, 'item'):
        return obj.item()
    return str(obj)

pathlib.Path('data/schema_registry.json').write_text(
    json.dumps(registry, indent=2, default=numpy_encoder)
)
print(f"\nRegistry written: {len(registry)} files")
