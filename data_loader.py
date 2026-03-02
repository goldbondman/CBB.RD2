"""
Data loader for the CBB picks-tracking application.

Provides :func:`load_app_data` and :func:`save_app_data` for reading/writing
the canonical CSV files, plus :class:`CSVDataManager` for object-oriented access.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_app_data(data_dir="./data"):
    """Load all 5 CSV files into properly typed DataFrames"""

    files = {
        'handicappers': 'handicappers.csv',
        'raw_tweets': 'raw_tweets.csv',
        'raw_picks': 'raw_picks.csv',
        'picks': 'picks.csv',
        'games': 'games.csv'
    }

    data = {}

    # Handicappers
    data['handicappers'] = pd.read_csv(
        Path(data_dir)/files['handicappers'],
        dtype={
            'handicapper_id': 'int32',
            'tier': 'category',
            'status': 'category'
        }
    )

    # Raw tweets with timestamp
    data['raw_tweets'] = pd.read_csv(
        Path(data_dir)/files['raw_tweets'],
        parse_dates=['created_at', 'ingested_at'],
        dtype={'handicapper_id': 'int32'}
    )

    # Raw picks
    data['raw_picks'] = pd.read_csv(
        Path(data_dir)/files['raw_picks'],
        dtype={
            'raw_pick_id': 'int32',
            'handicapper_id': 'int32',
            'line': 'float64',
            'units': 'float64',
            'parse_status': 'category'
        }
    )

    # Picks
    data['picks'] = pd.read_csv(
        Path(data_dir)/files['picks'],
        dtype={
            'pick_id': 'int32',
            'raw_pick_id': 'int32',
            'handicapper_id': 'int32',
            'game_id': 'int32',
            'line': 'float64',
            'units': 'float64',
            'mapping_status': 'category'
        }
    )

    # Games
    data['games'] = pd.read_csv(
        Path(data_dir)/files['games'],
        parse_dates=['date'],
        dtype={
            'game_id': 'int32',
            'closing_spread': 'float64',
            'total_line': 'float64'
        }
    )

    # Validate primary keys (no duplicates)
    for name, df in data.items():
        pks = [col for col in df.columns if col.endswith('_id')]
        if pks:
            assert df[pks[0]].duplicated().sum() == 0, f"Duplicate {pks[0]} in {name}"

    return data


def save_app_data(data, data_dir="./data"):
    """Save all DataFrames back to CSVs preserving dtypes"""
    Path(data_dir).mkdir(exist_ok=True)

    for name, df in data.items():
        df.to_csv(Path(data_dir)/f"{name}.csv", index=False)


class CSVDataManager:
    """Object-oriented wrapper around load_app_data / save_app_data."""

    def __init__(self, data_dir="./data"):
        self.data_dir = Path(data_dir)

    def load_app_data(self):
        """Load all 5 CSV files and return as a dict of DataFrames."""
        return load_app_data(self.data_dir)

    def save_app_data(self, data):
        """Persist all DataFrames back to their CSV files."""
        save_app_data(data, self.data_dir)

    def get_next_id(self, table_name: str) -> int:
        """Return the next available integer ID for the given table."""
        id_col = f"{table_name.rstrip('s')}_id"
        try:
            data = self.load_app_data()
            df = data.get(table_name)
            if df is None or df.empty or id_col not in df.columns:
                return 1
            return int(df[id_col].max()) + 1
        except Exception:
            return 1
