"""
Data loader for the CBB picks-tracking application.

Provides :class:`CSVDataManager` for robust loading, saving, and managing of
the five canonical CSV files used by the app, as well as module-level
:func:`load_app_data` and :func:`save_app_data` convenience wrappers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CSVDataManager
# ---------------------------------------------------------------------------

class CSVDataManager:
    """Master data loader/saver for handicapper app - CSV only."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.dtypes = self._get_dtypes()

    # ------------------------------------------------------------------
    # dtype definitions
    # ------------------------------------------------------------------

    def _get_dtypes(self) -> Dict[str, dict]:
        """Predefined dtypes for consistent loading."""
        return {
            'handicappers': {
                'handicapper_id': 'int32',
                'tier': 'category',
                'status': 'category',
            },
            'raw_tweets': {
                'handicapper_id': 'int32',
            },
            'raw_picks': {
                'raw_pick_id': 'int32',
                'handicapper_id': 'int32',
                'line': 'float64',
                'units': 'float64',
                'parse_status': 'category',
            },
            'picks': {
                'pick_id': 'int32',
                'raw_pick_id': 'int32',
                'handicapper_id': 'int32',
                'game_id': 'int32',
                'line': 'float64',
                'units': 'float64',
                'mapping_status': 'category',
            },
            'games': {
                'game_id': 'int32',
                'closing_spread': 'float64',
                'total_line': 'float64',
            },
        }

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_app_data(self) -> Dict[str, pd.DataFrame]:
        """Load all 5 CSV files with proper dtypes and validation."""
        csv_files = {
            'handicappers': 'handicappers.csv',
            'raw_tweets':   'raw_tweets.csv',
            'raw_picks':    'raw_picks.csv',
            'picks':        'picks.csv',
            'games':        'games.csv',
        }

        data: Dict[str, pd.DataFrame] = {}
        for name, filename in csv_files.items():
            filepath = self.data_dir / filename

            if not filepath.exists():
                logger.warning("%s not found - creating empty DataFrame", filename)
                data[name] = pd.DataFrame()
                continue

            try:
                dtype = self.dtypes.get(name, {})
                parse_dates: list = []
                if name == 'raw_tweets':
                    parse_dates = ['created_at', 'ingested_at']
                elif name == 'games':
                    parse_dates = ['date']

                df = pd.read_csv(filepath, dtype=dtype, parse_dates=parse_dates or False)
                data[name] = df
                logger.info("Loaded %s: %d rows", name, len(df))

            except Exception as exc:  # noqa: BLE001
                logger.error("Error loading %s: %s", filename, exc)
                data[name] = pd.DataFrame()

        self._validate_data_integrity(data)
        return data

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def _validate_data_integrity(self, data: Dict[str, pd.DataFrame]) -> None:
        """Check primary keys for duplicates (raises AssertionError on violation)."""
        pk_checks = {
            'handicappers': 'handicapper_id',
            'raw_tweets':   'tweet_id',
            'raw_picks':    'raw_pick_id',
            'picks':        'pick_id',
            'games':        'game_id',
        }
        for name, pk_col in pk_checks.items():
            if name in data and not data[name].empty and pk_col in data[name].columns:
                duplicates = data[name][pk_col].duplicated().sum()
                assert duplicates == 0, f"Duplicate {pk_col} in {name}"

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save_app_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Save all DataFrames back to CSVs preserving structure."""
        self.data_dir.mkdir(exist_ok=True)
        for name, df in data.items():
            if df.empty:
                logger.warning("Skipping empty %s", name)
                continue
            filepath = self.data_dir / f"{name}.csv"
            df.to_csv(filepath, index=False)
            logger.info("Saved %s: %d rows to %s", name, len(df), filepath)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_next_id(self, table_name: str) -> int:
        """Get the next sequential ID for new records."""
        data = self.load_app_data()
        df = data.get(table_name, pd.DataFrame())
        if df.empty:
            return 1
        id_col = table_name.rstrip('s') + '_id'
        return int(df[id_col].max()) + 1 if id_col in df.columns else 1

    def append_record(self, table_name: str, record: dict) -> int:
        """Append a single record to a CSV (auto-assigns ID if missing)."""
        data = self.load_app_data()
        df = data.get(table_name, pd.DataFrame())

        id_col = table_name.rstrip('s') + '_id'
        if id_col not in record:
            record[id_col] = self.get_next_id(table_name)

        data[table_name] = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        self.save_app_data(data)
        return record[id_col]


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

def load_app_data(data_dir: str = "./data") -> Dict[str, pd.DataFrame]:
    """Load all 5 CSV files into properly typed DataFrames.

    Parameters
    ----------
    data_dir:
        Directory containing the CSV files.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: ``handicappers``, ``raw_tweets``, ``raw_picks``, ``picks``,
        ``games``.

    Raises
    ------
    AssertionError
        If any table has duplicate primary-key values.
    """
    return CSVDataManager(data_dir).load_app_data()


def save_app_data(data: Dict[str, pd.DataFrame], data_dir: str = "./data") -> None:
    """Save DataFrames back to CSVs.

    Parameters
    ----------
    data:
        Mapping of table name → DataFrame (as returned by :func:`load_app_data`).
    data_dir:
        Destination directory (created if it does not exist).
    """
    CSVDataManager(data_dir).save_app_data(data)
