from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import pandas as pd

DateLike = Union[str, pd.Timestamp, None]


@dataclass(frozen=True)
class TeamSplitMatchup:
    home_efg: float
    away_efg: float
    home_orb: float
    away_orb: float
    home_tov: float
    away_tov: float
    home_pace: float
    away_pace: float
    home_netrtg: float
    away_netrtg: float
    home_ftr: float
    away_ftr: float


@lru_cache(maxsize=2)
def _load_team_splits(csv_path: str = "team_splits.csv") -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, low_memory=False)
    if df.empty:
        return df

    required = {
        "team_id", "season", "date", "efg_home", "efg_away", "orb_home", "orb_away",
        "tov_home", "tov_away", "pace_home", "pace_away", "netrtg_home", "netrtg_away",
        "ftr_home", "ftr_away",
    }
    if not required.issubset(df.columns):
        return pd.DataFrame()

    df = df.copy()
    # Normalize to timezone-aware UTC so comparisons are consistent with
    # game_datetime_utc values coming from schedule feeds.
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
    df = df[df["date"].notna() & df["team_id"].notna()].copy()
    df["team_id"] = df["team_id"].astype(str)
    return df.sort_values(["team_id", "season", "date"]).reset_index(drop=True)


def _season_from_date(ts: pd.Timestamp) -> int:
    return ts.year if ts.month < 7 else ts.year + 1


def _latest_team_row(df: pd.DataFrame, team_id: str, as_of: pd.Timestamp, season: int) -> Optional[pd.Series]:
    team_rows = df[(df["team_id"] == str(team_id)) & (df["season"] == season) & (df["date"] <= as_of)]
    if team_rows.empty:
        team_rows = df[(df["team_id"] == str(team_id)) & (df["date"] <= as_of)]
    if team_rows.empty:
        team_rows = df[df["team_id"] == str(team_id)]
    if team_rows.empty:
        return None
    return team_rows.iloc[-1]


def get_team_splits(
    date: DateLike,
    home_id: Union[str, int],
    away_id: Union[str, int],
    csv_path: str = "team_splits.csv",
) -> Optional[TeamSplitMatchup]:
    """Return latest known home/away split metrics for a matchup date/team pair."""
    df = _load_team_splits(csv_path)
    if df.empty:
        return None

    as_of = pd.to_datetime(date, errors="coerce", utc=True) if date is not None else pd.NaT
    if pd.isna(as_of):
        as_of = pd.Timestamp.now(tz="UTC")
    season = _season_from_date(as_of)

    home = _latest_team_row(df, str(home_id), as_of, season)
    away = _latest_team_row(df, str(away_id), as_of, season)
    if home is None or away is None:
        return None

    try:
        return TeamSplitMatchup(
            home_efg=float(home["efg_home"]),
            away_efg=float(away["efg_away"]),
            home_orb=float(home["orb_home"]),
            away_orb=float(away["orb_away"]),
            home_tov=float(home["tov_home"]),
            away_tov=float(away["tov_away"]),
            home_pace=float(home["pace_home"]),
            away_pace=float(away["pace_away"]),
            home_netrtg=float(home["netrtg_home"]),
            away_netrtg=float(away["netrtg_away"]),
            home_ftr=float(home["ftr_home"]),
            away_ftr=float(away["ftr_away"]),
        )
    except (TypeError, ValueError, KeyError):
        return None
