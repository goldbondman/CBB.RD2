import json
import pathlib
from datetime import datetime

import pandas as pd

registry = {}
for path in sorted(pathlib.Path('data').rglob('*.csv')):
    try:
        df = pd.read_csv(path, nrows=500, low_memory=False)
        registry[str(path)] = {
            'columns': {
                col: {
                    'dtype': str(df[col].dtype),
                    'null_rate': round(df[col].isna().mean(), 4),
                    'sample_values': df[col].dropna().head(3).tolist(),
                }
                for col in df.columns
            },
            'row_count_sample': len(df),
            'registered_at': datetime.utcnow().isoformat(),
        }
        print(f"[OK] {path}: {len(df.columns)} columns")
    except Exception as e:
        print(f"[FAIL] {path}: {e}")

pathlib.Path('data/schema_registry.json').write_text(json.dumps(registry, indent=2))
print(f"\nRegistry written: {len(registry)} files")
