import pathlib
import sys
import numpy as np
import pandas as pd

TOLERANCE = 0.5

def _candidate_paths(name: str):
    return [pathlib.Path('data') / name, pathlib.Path('data/csv') / name]


def _find_file(name: str):
    for p in _candidate_paths(name):
        if p.exists():
            return p
    return None


def _to_bool_series(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(False, index=[])
    vals = s.astype(str).str.strip().str.lower()
    return vals.isin({'1', 'true', 't', 'yes', 'y'})


def _report_bad(dataset: str, entity: str, column: str, diff: pd.Series, issues: list):
    bad = diff[diff > TOLERANCE]
    if bad.empty:
        return
    issues.append({
        'dataset': dataset,
        'entity': entity,
        'column': column,
        'bad_rows': int(bad.size),
        'max_diff': float(bad.max()),
    })


def verify_team_metrics(issues: list):
    path = _find_file('team_game_metrics.csv')
    if not path:
        print('[SKIP] team_game_metrics.csv not found in data/ or data/csv/')
        return

    df = pd.read_csv(path, low_memory=False)
    df['_dt'] = pd.to_datetime(df['game_datetime_utc'], utc=True, errors='coerce')
    df = df.sort_values(['team_id', '_dt']).reset_index(drop=True)

    cols_to_check = {
        'net_rtg': ('net_rtg_l5', 'net_rtg_l10'),
        'ortg': ('ortg_l5', 'ortg_l10'),
        'drtg': ('drtg_l5', 'drtg_l10'),
        'efg_pct': ('efg_pct_l5', 'efg_pct_l10'),
        'tov_pct': ('tov_pct_l5', 'tov_pct_l10'),
        'orb_pct': ('orb_pct_l5', 'orb_pct_l10'),
        'poss': ('poss_l5', 'poss_l10'),
    }

    for team_id, g in df.groupby('team_id', sort=False):
        for base_col, (l5_col, l10_col) in cols_to_check.items():
            if base_col not in g.columns:
                continue
            base = pd.to_numeric(g[base_col], errors='coerce')
            rec_l5 = base.shift(1).rolling(5, min_periods=1).mean().round(2)
            rec_l10 = base.shift(1).rolling(10, min_periods=1).mean().round(2)

            if l5_col in g.columns:
                got = pd.to_numeric(g[l5_col], errors='coerce')
                _report_bad('team_game_metrics.csv', str(team_id), l5_col, (got - rec_l5).abs(), issues)
            if l10_col in g.columns:
                got = pd.to_numeric(g[l10_col], errors='coerce')
                _report_bad('team_game_metrics.csv', str(team_id), l10_col, (got - rec_l10).abs(), issues)

    print(f'[OK] verified team rolling windows from {path}')


def verify_player_metrics(issues: list):
    path = _find_file('player_game_metrics.csv')
    if not path:
        print('[SKIP] player_game_metrics.csv not found in data/ or data/csv/')
        return

    df = pd.read_csv(path, low_memory=False)
    df['_dt'] = pd.to_datetime(df['game_datetime_utc'], utc=True, errors='coerce')
    df = df.sort_values(['athlete_id', '_dt']).reset_index(drop=True)

    active = (~_to_bool_series(df.get('did_not_play', pd.Series(False, index=df.index)))) & pd.to_numeric(df.get('min', 0), errors='coerce').gt(0)

    counting_stats = {
        'min', 'pts', 'fgm', 'fga', 'tpm', 'tpa', 'ftm', 'fta',
        'orb', 'drb', 'reb', 'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus'
    }

    base_metrics = [
        'min', 'pts', 'fgm', 'fga', 'tpm', 'tpa', 'ftm', 'fta',
        'orb', 'drb', 'reb', 'ast', 'stl', 'blk', 'tov', 'pf', 'plus_minus',
        'efg_pct', 'ts_pct', 'fg_pct', 'three_pct', 'ft_pct',
        'usage_rate', 'ast_tov_ratio', 'pts_per_fga', 'floor_pct'
    ]

    for base_col in base_metrics:
        if base_col not in df.columns:
            continue
        is_count = base_col in counting_stats
        series = pd.to_numeric(df[base_col], errors='coerce').fillna(0 if is_count else np.nan)
        masked_series = series.where(active)
        grouped = masked_series.groupby(df['athlete_id'])
        for window in (5, 10):
            col = f'{base_col}_l{window}'
            if col not in df.columns:
                continue
            rec = grouped.transform(
                lambda s, _w=window: s.shift(1).rolling(_w, min_periods=min(3, _w)).mean().round(2)
            )
            got = pd.to_numeric(df[col], errors='coerce')
            _report_bad('player_game_metrics.csv', 'all_players', col, (got - rec).abs(), issues)

    if 'starter' in df.columns:
        starter = _to_bool_series(df['starter'])
        for metric in ['pts', 'min', 'efg_pct', 'usage_rate']:
            if metric not in df.columns:
                continue
            for role, mask in [('starter', starter), ('bench', ~starter)]:
                col = f'{metric}_{role}_l5'
                if col not in df.columns:
                    continue
                rec = df.groupby('athlete_id')[metric].transform(
                    lambda s, _m=mask: (
                        pd.to_numeric(s, errors='coerce')
                        .where(_m.reindex(s.index, fill_value=False))
                        .shift(1)
                        .rolling(5, min_periods=2)
                        .mean()
                        .round(2)
                    )
                )
                got = pd.to_numeric(df[col], errors='coerce')
                _report_bad('player_game_metrics.csv', 'all_players', col, (got - rec).abs(), issues)

    print(f'[OK] verified player rolling windows from {path}')


def main():
    issues = []
    verify_team_metrics(issues)
    verify_player_metrics(issues)

    if issues:
        print('\n[FAIL] rolling window mismatches found:')
        for i in issues[:50]:
            print(i)
        print(f'... total issues: {len(issues)}')
        sys.exit(1)

    print('\n[PASS] all checked rolling window values are mathematically consistent.')


if __name__ == '__main__':
    main()
