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
