import json
import pathlib

import pandas as pd

PRIMARY_KEYS = {
    'team_game_metrics.csv': ['event_id', 'team_id'],
    'team_game_logs.csv': ['event_id', 'team_id'],
    'team_game_weighted.csv': ['event_id', 'team_id'],
    'player_game_metrics.csv': ['event_id', 'athlete_id'],
    'player_game_logs.csv': ['event_id', 'athlete_id'],
    'cbb_rankings.csv': ['team_id'],
    'predictions_latest.csv': ['game_id'],
    'results_log.csv': ['game_id'],
    'results_log_graded.csv': ['game_id'],
    'games.csv': ['game_id'],
}

issues = []
for path in sorted(pathlib.Path('data').rglob('*.csv')):
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] {path}: failed to read ({exc})")
        continue

    fname = path.name
    pk_cols = PRIMARY_KEYS.get(fname)

    if pk_cols:
        available = [c for c in pk_cols if c in df.columns]
        if available:
            dupes = df[df.duplicated(subset=available, keep=False)]
            if len(dupes) > 0:
                issues.append({
                    'file': str(path),
                    'pk_cols': available,
                    'duplicate_rows': int(len(dupes)),
                    'unique_keys_affected': int(dupes.groupby(available).ngroups),
                    'severity': 'CRITICAL',
                })
                print(f"[CRITICAL] {path}: {len(dupes)} duplicate rows on {available}")
                print("  Sample duplicate keys:")
                print(dupes[available].drop_duplicates().head(3).to_string(index=False))
            else:
                print(f"[OK] {path}: no duplicates on {available}")
        else:
            print(f"[WARN] {path}: primary key columns {pk_cols} not found in CSV")
    else:
        full_dupes = df[df.duplicated(keep=False)]
        if len(full_dupes) > 0:
            issues.append({
                'file': str(path),
                'pk_cols': 'all columns',
                'duplicate_rows': int(len(full_dupes)),
                'severity': 'WARNING',
            })
            print(f"[WARNING] {path}: {len(full_dupes)} fully-identical duplicate rows")

print(f"\nTotal files with duplicates: {len(issues)}")
pathlib.Path('data/duplicate_audit.json').write_text(json.dumps(issues, indent=2))
