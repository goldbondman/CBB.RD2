import pathlib
import pandas as pd


def resolve_path(primary: pathlib.Path) -> pathlib.Path:
    if primary.exists():
        return primary
    fallback = primary.parent / 'csv' / primary.name
    return fallback if fallback.exists() else primary


def normalize_game_id(series: pd.Series) -> pd.Series:
    """Normalize game IDs by stripping float suffixes and leading zeros."""
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.lstrip("0")
    )


def main():
    preds_path = resolve_path(pathlib.Path('data/predictions_latest.csv'))
    results_path = resolve_path(pathlib.Path('data/results_log.csv'))
    games_path = resolve_path(pathlib.Path('data/games.csv'))

    if not preds_path.exists():
        print('[SKIP] No predictions_latest.csv')
        raise SystemExit(0)

    preds = pd.read_csv(preds_path)
    preds['game_id_norm'] = normalize_game_id(preds['game_id'])

    print(f'Predictions: {len(preds)} rows ({preds_path})')

    if results_path.exists():
        results = pd.read_csv(results_path)
        if 'game_id' in results.columns:
            results['game_id_norm'] = normalize_game_id(results['game_id'])
        else:
            results['game_id_norm'] = pd.Series('', index=results.index)

        print(f'Results log: {len(results)} rows ({results_path})')

        if 'game_id' in results.columns:
            matched = preds['game_id_norm'].isin(results['game_id_norm']).sum()
            pct = matched / len(preds) * 100
            print(f"\nMatch on normalized game_id (predictions -> graded log): {matched}/{len(preds)} ({pct:.1f}%)")

        for col in ['ats_correct', 'ou_correct', 'winner_correct', 'actual_spread']:
            if col in results.columns:
                graded = results[col].notna().sum()
                pct = graded / len(results) * 100 if len(results) else 0
                print(f'Graded ({col}): {graded}/{len(results)} ({pct:.1f}%)')
    else:
        print(f'Results log: not found ({results_path})')

    if games_path.exists():
        games = pd.read_csv(games_path)
        games['game_id_norm'] = normalize_game_id(games['game_id'])

        if 'completed' in games.columns:
            completed = games[games['completed'].astype(str).str.lower().isin(['true', '1', 'yes'])].copy()
        else:
            print('[WARN] Missing "completed" column in games.csv — skipping completed check')
            completed = pd.DataFrame(columns=games.columns)

        matched_completed = preds['game_id_norm'].isin(completed['game_id_norm']).sum()
        pct_completed = matched_completed / len(preds) * 100
        print(f"\nMatch on normalized game_id (predictions -> completed games): {matched_completed}/{len(preds)} ({pct_completed:.1f}%)")

        matched_any = preds['game_id_norm'].isin(games['game_id_norm']).sum()
        pct_any = matched_any / len(preds) * 100
        print(f"Match on normalized game_id (predictions -> all scheduled games): {matched_any}/{len(preds)} ({pct_any:.1f}%)")

        pending = preds[preds['game_id_norm'].isin(games['game_id_norm']) & ~preds['game_id_norm'].isin(completed['game_id_norm'])]
        missing = preds[~preds['game_id_norm'].isin(games['game_id_norm'])]
        print(f'Pending (scheduled but not final): {len(pending)}')
        print(f'Missing (not found in games feed): {len(missing)}')

        if len(missing):
            print('\nMissing predictions (sample):')
            print(missing[['game_id', 'home_team', 'away_team']].head(10).to_string(index=False))
    else:
        print(f'Games file: not found ({games_path})')


if __name__ == "__main__":
    main()
