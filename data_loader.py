"""
Data loader for the CBB picks-tracking application.

Provides :func:`load_app_data` and :func:`save_app_data` for reading and
writing the canonical CSV files used by the app, plus :class:`CSVDataManager`
for an object-oriented interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np  # noqa: F401 – re-exported for downstream consumers that
                    # historically imported numpy via this module
import pandas as pd


def load_app_data(data_dir="./data") -> Dict[str, pd.DataFrame]:
    """Load all 5 CSV files into properly typed DataFrames.

    Parameters
    ----------
    data_dir:
        Directory that contains the five CSV files.  Defaults to ``./data``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``handicappers``, ``raw_tweets``, ``raw_picks``, ``picks``,
        ``games``.
    """
    files = {
        'handicappers': 'handicappers.csv',
        'raw_tweets': 'raw_tweets.csv',
        'raw_picks': 'raw_picks.csv',
        'picks': 'picks.csv',
        'games': 'games.csv',
    }

    data: Dict[str, pd.DataFrame] = {}

    # Handicappers
    data['handicappers'] = pd.read_csv(
        Path(data_dir) / files['handicappers'],
        dtype={
            'handicapper_id': 'int32',
            'tier': 'category',
            'status': 'category',
        }
    )

    # Raw tweets with timestamp
    data['raw_tweets'] = pd.read_csv(
        Path(data_dir) / files['raw_tweets'],
        parse_dates=['created_at', 'ingested_at'],
        dtype={'handicapper_id': 'int32'}
    )

    # Raw picks
    data['raw_picks'] = pd.read_csv(
        Path(data_dir) / files['raw_picks'],
        dtype={
            'raw_pick_id': 'int32',
            'handicapper_id': 'int32',
            'line': 'float64',
            'units': 'float64',
            'parse_status': 'category',
        }
    )

    # Picks
    data['picks'] = pd.read_csv(
        Path(data_dir) / files['picks'],
        dtype={
            'pick_id': 'int32',
            'raw_pick_id': 'int32',
            'handicapper_id': 'int32',
            'game_id': 'int32',
            'line': 'float64',
            'units': 'float64',
            'mapping_status': 'category',
        }
    )

    # Games
    data['games'] = pd.read_csv(
        Path(data_dir) / files['games'],
        parse_dates=['date'],
        dtype={
            'game_id': 'int32',
            'closing_spread': 'float64',
            'total_line': 'float64',
        }
    )

    # Validate primary keys (no duplicates)
    for name, df in data.items():
        pks = [col for col in df.columns if col.endswith('_id')]
        if pks:
            assert df[pks[0]].duplicated().sum() == 0, \
                f"Duplicate {pks[0]} in {name}"

    return data


def save_app_data(data: Dict[str, pd.DataFrame], data_dir="./data") -> None:
    """Save all DataFrames back to CSVs preserving dtypes."""
    Path(data_dir).mkdir(exist_ok=True)

    for name, df in data.items():
        df.to_csv(Path(data_dir) / f"{name}.csv", index=False)


class CSVDataManager:
    """Object-oriented wrapper for reading and writing the handicapper CSV files."""

    _TABLE_ID_COLS: Dict[str, str] = {
        "handicappers": "handicapper_id",
        "raw_tweets": "tweet_id",
        "raw_picks": "raw_pick_id",
        "picks": "pick_id",
        "games": "game_id",
    }

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parent / "data" / "handicapper"
        self.data_dir = Path(data_dir)

    def load_app_data(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV tables and return them as DataFrames."""
        return load_app_data(self.data_dir)

    def append_record(self, table_name: str, record: dict) -> int:
        """Append *record* to *table_name*.csv and return the new row's ID.

        Parameters
        ----------
        table_name:
            Logical table name (``handicappers``, ``raw_picks``, etc.).
        record:
            Dict of column→value pairs for the new row (must not include the
            primary-key column; it is assigned automatically).

        Returns
        -------
        int
            Auto-assigned primary-key value for the new row.
        """
        path = self.data_dir / f"{table_name}.csv"
        id_col = self._TABLE_ID_COLS.get(table_name)

        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame()

        if id_col and id_col in df.columns and not df.empty:
            new_id = int(df[id_col].max()) + 1
        else:
            new_id = 1

        new_record = dict(record)
        if id_col:
            new_record = {id_col: new_id, **new_record}

        new_df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(path, index=False)
        return new_id
