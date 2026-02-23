import pandas as pd
from pathlib import Path

DEFAULT_PATHS = [
    Path("data/team_game_metrics.csv"),
    Path("data/csv/team_game_metrics.csv"),
    Path("data/team_game_weighted.csv"),
    Path("data/csv/team_game_weighted.csv"),
]


def _resolve_metrics_path() -> Path:
    for path in DEFAULT_PATHS:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "Could not find team_game_metrics.csv in expected paths: "
        + ", ".join(str(p) for p in DEFAULT_PATHS)
    )


metrics_path = _resolve_metrics_path()
metrics = pd.read_csv(metrics_path, low_memory=False)
metrics['game_datetime_utc'] = pd.to_datetime(metrics['game_datetime_utc'], utc=True, errors='coerce')
metrics = metrics.sort_values(['team_id', 'game_datetime_utc'])

leakage_issues = []

# Test: if a rolling column correlates more strongly with the CURRENT
# game's value than with the prior game's value, it likely includes
# the current game in the window
rolling_pairs = [
    ('net_rtg',  'net_rtg_l5'),
    ('ortg',     'ortg_l5'),
    ('drtg',     'drtg_l5'),
    ('efg_pct',  'efg_pct_l5'),
    ('tov_pct',  'tov_pct_l5'),
]

for team_id, group in metrics.groupby('team_id'):
    if len(group) < 6:
        continue
    group = group.sort_values('game_datetime_utc').reset_index(drop=True)

    for base_col, l5_col in rolling_pairs:
        if base_col not in group.columns or l5_col not in group.columns:
            continue

        current_corr = group[base_col].corr(group[l5_col])
        lagged_corr = group[base_col].shift(1).corr(group[l5_col])

        # If L5 correlates more with current game than with prior game,
        # the current game is likely included in the window
        if current_corr > lagged_corr + 0.15 and current_corr > 0.85:
            leakage_issues.append({
                'team_id': team_id,
                'column': l5_col,
                'current_corr': round(current_corr, 3),
                'lagged_corr': round(lagged_corr, 3),
            })

by_col = {}
for issue in leakage_issues:
    by_col.setdefault(issue['column'], []).append(issue)

if by_col:
    print(f"[FAIL] Potential feature leakage detected in {len(by_col)} columns ({metrics_path}):")
    for col, issues in sorted(by_col.items()):
        avg_curr = sum(i['current_corr'] for i in issues) / len(issues)
        avg_lag = sum(i['lagged_corr'] for i in issues) / len(issues)
        print(f"  {col}: {len(issues)} teams — avg current_corr={avg_curr:.3f} vs lagged_corr={avg_lag:.3f}")
else:
    print(f"[OK] No feature leakage detected — all rolling windows appear to use prior games only ({metrics_path})")
