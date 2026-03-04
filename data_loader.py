from __future__ import annotations

import pandas as pd
from pathlib import Path
import numpy as np
from typing import Dict, Any


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
    """Manages loading and saving of handicapper app CSV data files."""

    def __init__(self, data_dir: str = "./data/handicapper"):
        self.data_dir = Path(data_dir)

    def load_app_data(self) -> Dict[str, pd.DataFrame]:
        """Load all app data files from data_dir."""
        return load_app_data(self.data_dir)

    def append_record(self, table_name: str, record: Dict[str, Any]) -> int:
        """Append a new record to a CSV table and return its new primary key.

        Parameters
        ----------
        table_name : str
            Logical table name (e.g. ``'picks'``, ``'raw_picks'``).
        record : dict
            Column-value mapping for the new row.  The primary-key column
            (``<table_name[:-1]>_id`` or ``<table_name>_id``) must *not* be
            provided; it is assigned automatically.

        Returns
        -------
        int
            The new primary-key value that was assigned.
        """
        csv_path = self.data_dir / f"{table_name}.csv"
        pk_col = f"{table_name[:-1]}_id" if table_name.endswith("s") else f"{table_name}_id"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if pk_col in df.columns and not df.empty:
                new_id = int(df[pk_col].max()) + 1
            else:
                new_id = 1
        else:
            df = pd.DataFrame()
            new_id = 1

        record = dict(record)
        record[pk_col] = new_id
        new_row = pd.DataFrame([record])

        if df.empty:
            result = new_row
        else:
            result = pd.concat([df, new_row], ignore_index=True)

        self.data_dir.mkdir(parents=True, exist_ok=True)
        result.to_csv(csv_path, index=False)
        return new_id
