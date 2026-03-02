from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np


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
    """Wrapper around load_app_data/save_app_data providing a stateful interface."""

    def __init__(self, data_dir="./data"):
        self.data_dir = str(data_dir)

    def load_app_data(self):
        return load_app_data(self.data_dir)

    def save_app_data(self, data):
        save_app_data(data, self.data_dir)

    def get_next_id(self, table_name: str) -> int:
        """Return the next sequential integer ID for the given table.

        Falls back to 1 if the table is empty or the id column is absent.
        """
        try:
            data = load_app_data(self.data_dir)
        except (FileNotFoundError, AssertionError):
            return 1

        df = data.get(table_name)
        if df is None or df.empty:
            return 1

        id_cols = [c for c in df.columns if c.endswith('_id')]
        if not id_cols:
            return 1

        return int(df[id_cols[0]].max()) + 1
