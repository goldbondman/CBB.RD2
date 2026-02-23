"""
cbb_travel_fatigue.py — Team Travel Fatigue Features

Computes travel distance, rest, and schedule-density features per team per game.

Inputs:  data/games.csv, data/team_game_logs.csv, data/venue_geocodes.csv
Output:  data/team_travel_fatigue.csv
"""

from __future__ import annotations

import re
import string
from datetime import datetime, timezone
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd

from espn_config import OUT_TRAVEL_FATIGUE, OUT_VENUE_GEOCODES, OUT_GAMES, OUT_TEAM_LOGS
from config.logging_config import get_logger

log = get_logger(__name__)

DATA_DIR = Path("data")

# ── Conference-average travel distances (miles) — fallback when venue unknown ──
CONF_AVG_MILES: dict[str, float] = {
    "SEC":       350.0,
    "ACC":       280.0,
    "Big Ten":   420.0,
    "Big 12":    480.0,
    "Pac-12":    550.0,
    "Big East":  200.0,
    "AAC":       520.0,
    "Mountain West": 600.0,
    "Sun Belt":  400.0,
    "Conference USA": 450.0,
    "Atlantic 10": 250.0,
    "WCC":       600.0,
    "Missouri Valley": 300.0,
    "Horizon":   250.0,
    "Colonial":  220.0,
    "Southern":  350.0,
    "Southland": 350.0,
    "Ohio Valley": 300.0,
    "Summit":    400.0,
    "Big West":  500.0,
    "MAC":       250.0,
    "CUSA":      450.0,
    "MEAC":      350.0,
    "SWAC":      400.0,
    "NEC":       180.0,
    "Patriot":   200.0,
    "Ivy":       180.0,
    "America East": 200.0,
    "Atlantic Sun": 400.0,
    "Big South": 300.0,
    "CAA":       220.0,
    "Big Sky":   550.0,
    "WAC":       600.0,
}
DEFAULT_TRAVEL_MILES = 350.0


def _normalize_venue(name: str) -> str:
    """Lowercase, strip punctuation and extra whitespace for fuzzy matching."""
    s = name.lower()
    s = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8  # Earth radius in miles
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(a))


def _safe_read(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV if it exists and is non-empty; otherwise return empty DataFrame."""
    if not path.exists() or path.stat().st_size == 0:
        log.warning(f"Input file not found or empty: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:
        log.warning(f"Could not read {path}: {exc}")
        return pd.DataFrame()


def _build_geocode_lookup(geo_df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Build normalized-name → (lat, lon) lookup dict from venue_geocodes.csv."""
    lookup: dict[str, tuple[float, float]] = {}
    if geo_df.empty:
        return lookup
    for _, row in geo_df.iterrows():
        name = str(row.get("venue_name", ""))
        if not name:
            continue
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
        except (KeyError, ValueError, TypeError):
            continue
        lookup[_normalize_venue(name)] = (lat, lon)
    return lookup


def compute_travel_fatigue(
    games_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    geo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute travel fatigue features for each team-game row.

    Parameters
    ----------
    games_df : DataFrame from games.csv  (contains venue, home/away, datetime)
    logs_df  : DataFrame from team_game_logs.csv (team_id, event_id, home_away, conference)
    geo_df   : DataFrame from venue_geocodes.csv

    Returns
    -------
    DataFrame with travel fatigue columns (one row per team per game).
    """
    if games_df.empty or logs_df.empty:
        log.warning("compute_travel_fatigue: empty games or logs — returning empty DataFrame")
        return pd.DataFrame()

    geo_lookup = _build_geocode_lookup(geo_df)
    log.info(f"Venue geocode lookup: {len(geo_lookup)} venues")

    # Normalize team logs: ensure required columns present
    for col in ["event_id", "team_id"]:
        if col not in logs_df.columns:
            log.warning(f"compute_travel_fatigue: missing '{col}' in logs — aborting")
            return pd.DataFrame()

    # Build a per-team-game base from logs
    base_cols = [c for c in [
        "event_id", "team_id", "team", "home_away", "conference",
        "game_datetime_utc",
    ] if c in logs_df.columns]
    base = logs_df[base_cols].drop_duplicates(subset=["event_id", "team_id"]).copy()

    # Attach venue from games_df
    venue_col_candidates = ["venue", "venue_name", "location"]
    venue_col = next((c for c in venue_col_candidates if c in games_df.columns), None)
    if venue_col:
        game_venue = games_df[["event_id", venue_col]].drop_duplicates("event_id").rename(
            columns={venue_col: "venue"}
        )
        base = base.merge(game_venue, on="event_id", how="left")
    else:
        base["venue"] = None

    # Sort by team and game date
    if "game_datetime_utc" in base.columns:
        base["_dt"] = pd.to_datetime(base["game_datetime_utc"], utc=True, errors="coerce")
    else:
        base["_dt"] = pd.NaT

    base = base.sort_values(["team_id", "_dt"], na_position="last")
    base = base.reset_index(drop=True)

    rows = []
    for team_id, grp in base.groupby("team_id"):
        grp = grp.reset_index(drop=True)
        prev_venue: str | None = None
        prev_dt: pd.Timestamp | None = None
        conf = grp["conference"].iloc[0] if "conference" in grp.columns else ""

        for idx, row in grp.iterrows():
            rec: dict = {
                "event_id":           row.get("event_id"),
                "team_id":            team_id,
                "team":               row.get("team"),
                "game_datetime_utc":  row.get("game_datetime_utc"),
                "home_away":          row.get("home_away"),
                "venue":              row.get("venue"),
                "prev_game_date":     prev_dt.isoformat() if pd.notna(prev_dt) else None,
                "prev_venue":         prev_venue,
            }

            cur_dt = row.get("_dt")

            # rest_days
            if prev_dt is not None and pd.notna(cur_dt) and pd.notna(prev_dt):
                rest_days = float((cur_dt - prev_dt).days)
            else:
                rest_days = 7.0  # first game of season default
            rec["rest_days"] = rest_days

            # estimated_travel_miles
            cur_venue  = str(row.get("venue") or "")
            cur_norm   = _normalize_venue(cur_venue) if cur_venue else ""
            prev_norm  = _normalize_venue(prev_venue) if prev_venue else ""

            miles = None
            if cur_norm and prev_norm and cur_norm != prev_norm:
                cur_geo  = geo_lookup.get(cur_norm)
                prev_geo = geo_lookup.get(prev_norm)
                if cur_geo and prev_geo:
                    miles = _haversine_miles(prev_geo[0], prev_geo[1], cur_geo[0], cur_geo[1])

            if miles is None:
                # fallback: conference-average
                miles = CONF_AVG_MILES.get(str(conf), DEFAULT_TRAVEL_MILES)
                if cur_norm == prev_norm and cur_norm:
                    miles = 0.0  # same venue — no travel

            rec["estimated_travel_miles"] = round(miles, 1)
            rec["is_back_to_back"]        = bool(rest_days <= 1)
            rec["is_road_back_to_back"]   = bool(rest_days <= 1 and str(row.get("home_away", "")) == "away")

            prev_venue = cur_venue or prev_venue
            prev_dt    = cur_dt if pd.notna(cur_dt) else prev_dt
            rows.append(rec)

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    # games_in_last_7 / 14 days
    result["_dt"] = pd.to_datetime(result["game_datetime_utc"], utc=True, errors="coerce")
    result = result.sort_values(["team_id", "_dt"], na_position="last").reset_index(drop=True)

    def _count_prior_games(grp: pd.Series, days: int) -> pd.Series:
        dts = grp.values
        counts = []
        for i, dt in enumerate(dts):
            if pd.isnull(dt):
                counts.append(0)
                continue
            cutoff = dt - pd.Timedelta(days=days)
            counts.append(sum(1 for d in dts[:i] if not pd.isnull(d) and d >= cutoff))
        return pd.Series(counts, index=grp.index)

    result["games_in_last_7_days"]  = result.groupby("team_id")["_dt"].transform(
        lambda s: _count_prior_games(s, 7)
    )
    result["games_in_last_14_days"] = result.groupby("team_id")["_dt"].transform(
        lambda s: _count_prior_games(s, 14)
    )

    # fatigue_score = clip((1/max(rest,1))*0.5 + (g7/4)*0.3 + (miles/1000)*0.2, 0, 1)
    rest_s  = result["rest_days"].clip(lower=1)
    g7_s    = result["games_in_last_7_days"]
    miles_s = result["estimated_travel_miles"]
    result["fatigue_score"] = (
        (1.0 / rest_s) * 0.5
        + (g7_s / 4.0) * 0.3
        + (miles_s / 1000.0) * 0.2
    ).clip(0.0, 1.0).round(4)

    result = result.drop(columns=["_dt"], errors="ignore")

    out_cols = [
        "event_id", "team_id", "team", "game_datetime_utc",
        "home_away", "venue", "prev_game_date", "prev_venue",
        "rest_days", "estimated_travel_miles",
        "is_back_to_back", "is_road_back_to_back",
        "games_in_last_7_days", "games_in_last_14_days",
        "fatigue_score",
    ]
    for c in out_cols:
        if c not in result.columns:
            result[c] = None
    return result[[c for c in out_cols if c in result.columns]]


def main() -> None:
    
    games_df = _safe_read(OUT_GAMES)
    logs_df  = _safe_read(OUT_TEAM_LOGS)
    geo_df   = _safe_read(OUT_VENUE_GEOCODES)

    result = compute_travel_fatigue(games_df, logs_df, geo_df)

    if result.empty:
        log.warning("No travel fatigue data produced — nothing to write")
        return

    OUT_TRAVEL_FATIGUE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_TRAVEL_FATIGUE, index=False)
    log.info(f"team_travel_fatigue.csv → {OUT_TRAVEL_FATIGUE}  ({len(result):,} rows)")


if __name__ == "__main__":
    main()
