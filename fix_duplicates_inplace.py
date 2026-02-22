import pathlib

import pandas as pd

from pipeline_csv_utils import dedupe_by_primary_key


def main() -> None:
    for path in sorted(pathlib.Path('data').rglob('*.csv')):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:
            print(f"[WARN] {path}: read failed ({exc})")
            continue

        deduped = dedupe_by_primary_key(df, path)
        removed = len(df) - len(deduped)
        if removed > 0:
            deduped.to_csv(path, index=False)
            print(f"[FIXED] {path}: removed {removed} duplicate rows")
        else:
            print(f"[OK] {path}: no primary-key duplicates")


if __name__ == '__main__':
    main()
