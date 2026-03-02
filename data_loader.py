"""
Data loader for the CBB picks-tracking application.

Provides :func:`load_app_data`, which reads the canonical CSV files used by
the app and returns them as a dict of ``{name: pd.DataFrame}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

# Root of the repository (this file lives at the repo root).
_REPO_ROOT = Path(__file__).parent

# Mapping of logical name → path relative to the repo root.
_APP_DATA_FILES: Dict[str, Path] = {
    "handicappers": _REPO_ROOT / "data" / "handicappers.csv",
    "raw_tweets":   _REPO_ROOT / "data" / "raw_tweets.csv",
    "raw_picks":    _REPO_ROOT / "data" / "raw_picks.csv",
    "picks":        _REPO_ROOT / "data" / "picks.csv",
    "games":        _REPO_ROOT / "data" / "app_games.csv",
}


def load_app_data() -> Dict[str, pd.DataFrame]:
    """Load all application data files and return them as DataFrames.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are logical file names (``handicappers``, ``raw_tweets``,
        ``raw_picks``, ``picks``, ``games``); values are the corresponding
        DataFrames.

    Raises
    ------
    FileNotFoundError
        If any expected data file is missing.
    """
    data: Dict[str, pd.DataFrame] = {}
    for name, path in _APP_DATA_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Required app data file not found: {path}"
            )
        data[name] = pd.read_csv(path)
    return data
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
