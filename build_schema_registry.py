import json
import pathlib
from datetime import datetime, timezone

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
                    'null_rate_note': 'approximate (based on first 500 rows)' if len(df) >= 500 else 'exact'
                }
                for col in df.columns
            },
            'row_count_sample': int(min(len(df), 500)),
            'stats_are_sampled': bool(len(df) >= 500),
            'row_count_sample': min(len(df), 500),
            'stats_are_sampled': len(df) >= 500,
            'registered_at': datetime.now(timezone.utc).isoformat(),
        }
        print(f"[OK] {path}: {len(df.columns)} columns")
    except Exception as e:
        print(f"[FAIL] {path}: {e}")

pathlib.Path('data/schema_registry.json').write_text(
    json.dumps(registry, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
)
print(f"\nRegistry written: {len(registry)} files")
