#!/usr/bin/env python3
"""
ESPN CBB Pipeline — Prediction Runner
Bridges team_game_weighted.csv → CBBPredictionModel → predictions_YYYYMMDD.csv

Reads from the same data/ directory the main pipeline writes.
Fetches scheduled games, builds GameData objects (including recursive opponent_history),
runs CBBPredictionModel.predict_game(), and writes outputs.

Usage:
    python espn_prediction_runner.py
        - Default: rolling window of the next 40 hours in America/Los_Angeles time.
    python espn_prediction_runner.py --date 20250315
        - Specific date YYYYMMDD (PST date context).
    python espn_prediction_runner.py --hours-ahead 40
        - Rolling window horizon (default: 40 hours).
    python espn_prediction_runner.py --game-type ncaa_r1
        - Tournament context override.
    python espn_prediction_runner.py --decay smooth
        - Weight decay.

Outputs written to data/:
    predictions_<label>.csv     - label is YYYYMMDD for date mode, else a timestamp label
    predictions_latest.csv      - always overwritten with most recent run

Important:
    - Leak-free history: for each predicted matchup, the cutoff is the game kickoff time in UTC.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from espn_config import (
    CSV_DIR as DATA_DIR,
    OUT_WEIGHTED  as CSV_WEIGHTED,
    OUT_METRICS   as CSV_METRICS,
    OUT_GAMES     as CSV_GAMES,
    OUT_TEAM_LOGS as CSV_LOGS,
    OUT_TOURNAMENT_SNAPSHOT as CSV_SNAPSHOT,
    TZ,
)

OUT_PREDICTIONS_LATEST = DATA_DIR / "predictions_latest.csv"
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Local imports ──────────────────────────────────────────────────────────────
try:
    from cbb_prediction_model import (
        CBBPredictionModel,
        GameData,
        ModelConfig,
    )
except ImportError as e:
    raise SystemExit(
        f"Cannot import cbb_prediction_model.py. Ensure it is in the same directory.\n{e}"
    )

try:
    from espn_tournament import (
        compute_underdog_winner_score,
        build_pretournament_snapshot,
    )
    TOURNAMENT_AVAILABLE = True
except ImportError:
    TOURNAMENT_AVAILABLE = False

try:
    from espn_client import fetch_scoreboard
    ESPN_CLIENT_AVAILABLE = True
except ImportError:
    ESPN_CLIENT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
BOX_COL_MAP = {
    "fgm": "fgm",
    "fga": "fga",
    "tpm": "tpm",
    "tpa": "tpa",
    "ftm": "ftm",
    "fta": "fta",
    "orb": "orb",
    "drb": "drb",
    "tov": "tov",
    "pf":  "pf",    # v2.1: required for foul-rate confidence adjustment
}

# Required downstream enrichment fields from latest pre-game team row
REQUIRED_TEAM_CONTEXT_FIELDS = [
    "fgm", "fga", "ftm", "fta", "tpm", "tpa",
    "orb", "drb", "reb", "tov", "ast",
    "wins", "losses", "conference",
]

OPP_HISTORY_WINDOW = 5
TEAM_GAMES_WINDOW = 10


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_team_game_data() -> pd.DataFrame:
    """
    Load the richest available team data file.
    Priority: weighted > metrics > logs.
    Ensures game_datetime_utc is parsed and sorted chronologically.
    """
    for path, label in [
        (CSV_WEIGHTED, "team_game_weighted"),
        (CSV_METRICS,  "team_game_metrics"),
        (CSV_LOGS,     "team_game_logs"),
    ]:
        if path.exists() and path.stat().st_size > 100:
            log.info(f"Loading team data from {path.name} ({path.stat().st_size:,} bytes)")
            df = pd.read_csv(path, dtype=str, low_memory=False)
            df["game_datetime_utc"] = pd.to_datetime(
                df.get("game_datetime_utc", pd.NaT), utc=True, errors="coerce"
            )

            non_numeric = {
                "team_id", "team", "opponent_id", "opponent",
                "home_away", "conference", "event_id", "game_id",
                "game_datetime_utc", "game_datetime_pst", "venue",
                "state", "source", "parse_version", "t_offensive_archetype",
            }
            for col in df.columns:
                if col not in non_numeric:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.sort_values(["team_id", "game_datetime_utc"])
            log.info(f"Loaded {len(df)} team-game rows, {df['team_id'].nunique()} teams")
            return df

    raise FileNotFoundError(
        "No team game data found. Run espn_pipeline.py first.\n"
        f"Looked for: {CSV_WEIGHTED}, {CSV_METRICS}, {CSV_LOGS}"
    )


def load_games_schedule() -> pd.DataFrame:
    """Load games.csv scoreboard data for scheduled games fallback."""
    if not CSV_GAMES.exists():
        log.warning(f"{CSV_GAMES} not found. Cannot use fallback schedule.")
        return pd.DataFrame()

    df = pd.read_csv(CSV_GAMES, dtype=str, low_memory=False)
    df["game_datetime_utc"] = pd.to_datetime(
        df.get("game_datetime_utc", pd.NaT), utc=True, errors="coerce"
    )
    return df


def load_tournament_snapshot() -> Optional[pd.DataFrame]:
    """Load pre-tournament snapshot if available (for UWS enrichment)."""
    if not TOURNAMENT_AVAILABLE:
        return None
    if CSV_SNAPSHOT.exists() and CSV_SNAPSHOT.stat().st_size > 100:
        df = pd.read_csv(CSV_SNAPSHOT, dtype=str, low_memory=False)
        for col in df.columns:
            if col not in ("team_id", "team", "t_offensive_archetype"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        log.info(f"Loaded tournament snapshot: {len(df)} teams")
        return df.set_index("team_id") if "team_id" in df.columns else None
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# GAMEDATA BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _row_to_box(row: pd.Series) -> Dict[str, float]:
    box: Dict[str, float] = {}
    for csv_col, box_key in BOX_COL_MAP.items():
        val = row.get(csv_col, 0.0)
        box[box_key] = float(val) if pd.notna(val) else 0.0
    return box


def _safe_int(value, default: int = 0) -> int:
    """Convert numeric-like values to int, treating NaN/None/empty as default."""
    if value is None or pd.isna(value):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_game_data(row: pd.Series, opp_history: List[GameData]) -> GameData:
    team_score = _safe_int(row.get("points_for", 0), default=0)
    opp_score = _safe_int(row.get("points_against", 0), default=0)
    neutral = bool(row.get("neutral_site", False))

    dt = row.get("game_datetime_utc")
    if pd.isna(dt):
        dt = datetime.now(TZ)

    team_box = _row_to_box(row)

    opp_box: Dict[str, float] = {}
    for csv_col, box_key in BOX_COL_MAP.items():
        opp_col = f"opp_{csv_col}"
        val = row.get(opp_col, 0.0)
        opp_box[box_key] = float(val) if pd.notna(val) else 0.0

    if all(v == 0.0 for v in opp_box.values()):
        avg_fga = team_box.get("fga", 58.0)
        opp_box = {
            "fgm": avg_fga * 0.44,
            "fga": avg_fga,
            "tpm": avg_fga * 0.15,
            "tpa": avg_fga * 0.36,
            "ftm": avg_fga * 0.22,
            "fta": avg_fga * 0.30,
            "orb": 8.0,
            "drb": 23.0,
            "tov": 12.0,
        }

    return GameData(
        game_id=str(row.get("event_id", "")),
        date=dt if isinstance(dt, datetime) else dt.to_pydatetime(),
        team_name=str(row.get("team", "")),
        opponent_name=str(row.get("opponent", "")),
        team_score=team_score,
        opponent_score=opp_score,
        neutral_site=neutral,
        team_box=team_box,
        opponent_box=opp_box,
        opponent_history=opp_history,
    )


def build_team_game_list(
    team_id: str,
    all_data: pd.DataFrame,
    cutoff_dt: Optional[pd.Timestamp] = None,
    max_games: int = TEAM_GAMES_WINDOW,
) -> List[GameData]:
    team_rows = all_data[all_data["team_id"].astype(str) == str(team_id)].copy()

    if cutoff_dt is not None:
        team_rows = team_rows[team_rows["game_datetime_utc"] < cutoff_dt]

    team_rows = team_rows.sort_values("game_datetime_utc")

    if team_rows.empty:
        log.warning(f"No game history found for team_id={team_id}")
        return []

    recent_rows = team_rows.tail(max_games)
    game_list: List[GameData] = []

    for _, row in recent_rows.iterrows():
        opp_id = str(row.get("opponent_id", ""))
        opp_cutoff = row["game_datetime_utc"]
        opp_history = _build_opponent_history(opp_id, all_data, opp_cutoff)
        gd = _build_game_data(row, opp_history)
        game_list.append(gd)

    return game_list


def _build_opponent_history(
    opp_id: str,
    all_data: pd.DataFrame,
    before_dt: pd.Timestamp,
) -> List[GameData]:
    if not opp_id or opp_id == "nan":
        return []

    opp_rows = all_data[
        (all_data["team_id"].astype(str) == opp_id) &
        (all_data["game_datetime_utc"] < before_dt)
    ].sort_values("game_datetime_utc").tail(OPP_HISTORY_WINDOW)

    history: List[GameData] = []
    for _, row in opp_rows.iterrows():
        gd = _build_game_data(row, opp_history=[])
        history.append(gd)

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULE DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_utc_dt(val) -> Optional[pd.Timestamp]:
    """
    ESPN comp["date"] is ISO string. games.csv may be string or empty.
    Returns UTC Timestamp or None.
    """
    if val is None:
        return None
    try:
        ts = pd.to_datetime(val, utc=True, errors="coerce")
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _build_matchup_from_scoreboard_event(event: Dict) -> Optional[Dict]:
    comps = event.get("competitions", [{}])
    comp = comps[0] if comps else {}
    completed = comp.get("status", {}).get("type", {}).get("completed", False)
    if completed:
        return None

    home = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "home"), {})
    away = next((c for c in comp.get("competitors", []) if c.get("homeAway") == "away"), {})
    if not home or not away:
        return None

    odds = (comp.get("odds") or [{}])[0]
    return {
        "game_id": str(event.get("id", "")),
        "home_team_id": str(home.get("team", {}).get("id", "")),
        "away_team_id": str(away.get("team", {}).get("id", "")),
        "home_team": home.get("team", {}).get("displayName", ""),
        "away_team": away.get("team", {}).get("displayName", ""),
        "game_datetime_utc": comp.get("date", ""),
        "neutral_site": bool(comp.get("neutralSite", False)),
        "over_under": odds.get("overUnder"),
        "spread": odds.get("spread"),
        "home_ml": odds.get("homeTeamOdds", {}).get("moneyLine"),
        "away_ml": odds.get("awayTeamOdds", {}).get("moneyLine"),
    }


def get_matchups_for_date(target_date: str) -> List[Dict]:
    """
    Gets scheduled (not completed) matchups for a given YYYYMMDD date.
    Prefers live scoreboard fetch. Falls back to games.csv.
    """
    target_dt_str = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"
    log.info(f"Looking for games on {target_dt_str}")

    if ESPN_CLIENT_AVAILABLE:
        try:
            raw = fetch_scoreboard(target_date)
            events = raw.get("events") or []
            matchups: List[Dict] = []
            for event in events:
                m = _build_matchup_from_scoreboard_event(event)
                if m:
                    matchups.append(m)
            log.info(f"Found {len(matchups)} scheduled games via ESPN scoreboard")
            return matchups
        except Exception as exc:
            log.warning(f"Live scoreboard fetch failed ({exc}). Falling back to games.csv.")

    games_df = load_games_schedule()
    if games_df.empty:
        log.error("No game schedule data available")
        return []

    target_mask = (
        games_df["game_datetime_utc"]
        .dt.date.astype(str)
        .str.startswith(target_dt_str)
    )
    incomplete = games_df["completed"].astype(str).str.lower().isin(["false", "0", ""])

    day_games = games_df[target_mask & incomplete].copy()
    if day_games.empty and "date" in games_df.columns:
        day_games = games_df[games_df["date"].astype(str) == target_date].copy()

    matchups: List[Dict] = []
    for _, row in day_games.iterrows():
        matchups.append({
            "game_id": str(row.get("game_id", "")),
            "home_team_id": str(row.get("home_team_id", "")),
            "away_team_id": str(row.get("away_team_id", "")),
            "home_team": str(row.get("home_team", "")),
            "away_team": str(row.get("away_team", "")),
            "game_datetime_utc": str(row.get("game_datetime_utc", "")),
            "neutral_site": str(row.get("neutral_site", "false")).lower() == "true",
            "over_under": row.get("over_under"),
            "spread": row.get("spread"),
            "home_ml": row.get("home_ml"),
            "away_ml": row.get("away_ml"),
        })

    log.info(f"Found {len(matchups)} scheduled games from games.csv")
    return matchups


def get_matchups_in_window(start_local: datetime, end_local: datetime) -> List[Dict]:
    """
    Rolling window matchups between [start_local, end_local] in America/Los_Angeles.
    We fetch per date and then filter by kickoff UTC timestamps.
    """
    if start_local.tzinfo is None or end_local.tzinfo is None:
        raise ValueError("start_local and end_local must be timezone-aware")

    start_utc = pd.Timestamp(start_local.astimezone(ZoneInfo("UTC")))
    end_utc = pd.Timestamp(end_local.astimezone(ZoneInfo("UTC")))

    # Unique local dates touched by the window
    dates: List[str] = []
    d = start_local.date()
    while d <= end_local.date():
        dates.append(d.strftime("%Y%m%d"))
        d = (datetime.combine(d, datetime.min.time(), tzinfo=TZ) + timedelta(days=1)).date()

    log.info(f"Rolling window local: {start_local.isoformat()} -> {end_local.isoformat()}")
    log.info(f"Rolling window UTC:   {start_utc.isoformat()} -> {end_utc.isoformat()}")
    log.info(f"Dates queried: {dates}")

    all_matchups: List[Dict] = []
    for yyyymmdd in dates:
        all_matchups.extend(get_matchups_for_date(yyyymmdd))

    # Filter to window by kickoff time
    filtered: List[Dict] = []
    for m in all_matchups:
        kick = _parse_utc_dt(m.get("game_datetime_utc"))
        if kick is None:
            continue
        if start_utc <= kick <= end_utc:
            filtered.append(m)

    # Deduplicate
    seen = set()
    uniq: List[Dict] = []
    for m in filtered:
        gid = str(m.get("game_id") or "")
        key = gid if gid else (m.get("home_team_id"), m.get("away_team_id"), m.get("game_datetime_utc"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(m)

    log.info(f"Window matchups: {len(uniq)} games")
    return uniq


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_predictions(
    matchups: List[Dict],
    all_data: pd.DataFrame,
    model: CBBPredictionModel,
    snapshot: Optional[pd.DataFrame],
    game_type: str = "regular",
    default_cutoff_dt: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Runs predictions for matchups.
    Leak-free: cutoff is per-game kickoff time if available, else default_cutoff_dt.
    """
    results: List[Dict] = []
    skipped = 0

    for matchup in matchups:
        home_id = matchup.get("home_team_id")
        away_id = matchup.get("away_team_id")
        home_name = matchup.get("home_team")
        away_name = matchup.get("away_team")
        game_id = matchup.get("game_id")
        neutral = bool(matchup.get("neutral_site", False))

        kick_utc = _parse_utc_dt(matchup.get("game_datetime_utc"))
        cutoff_dt = kick_utc if kick_utc is not None else default_cutoff_dt

        log.info(f"  Processing: {home_name} vs {away_name} (game_id={game_id})")

        home_games = build_team_game_list(str(home_id), all_data, cutoff_dt=cutoff_dt)
        away_games = build_team_game_list(str(away_id), all_data, cutoff_dt=cutoff_dt)

        if not home_games or not away_games:
            log.warning(
                f"    Skipping {home_name} vs {away_name}. Insufficient history "
                f"(home={len(home_games)}, away={len(away_games)})"
            )
            skipped += 1
            continue

        try:
            prediction = model.predict_game(
                home_games=home_games,
                away_games=away_games,
                neutral_site=neutral,
                game_type=game_type,
            )
        except Exception as exc:
            log.error(f"    Prediction failed for {game_id}: {exc}")
            skipped += 1
            continue

        spread_line = _safe_float(matchup.get("spread"))
        total_line = _safe_float(matchup.get("over_under"))

        spread_diff = (
            round(prediction["predicted_spread"] - spread_line, 2)
            if spread_line is not None else None
        )
        total_diff = (
            round(prediction["predicted_total"] - total_line, 2)
            if total_line is not None else None
        )

        pred_spread = prediction["predicted_spread"]
        if spread_line is not None and spread_diff is not None:
            edge_flag = abs(spread_diff) > 3.0
            spread_pick = f"{home_name} covers" if pred_spread < spread_line else f"{away_name} covers"
        else:
            edge_flag = False
            spread_pick = f"{home_name} -" if pred_spread < 0 else f"{away_name} -"
            spread_pick += f"{abs(pred_spread):.1f}"

        uws_result: Dict = {}
        if TOURNAMENT_AVAILABLE and snapshot is not None:
            uws_result = _compute_uws_for_matchup(matchup, snapshot, game_type=game_type)

        bd = prediction.get("breakdown", {})
        home_ctx = _latest_team_context(all_data, str(home_id), cutoff_dt)
        away_ctx = _latest_team_context(all_data, str(away_id), cutoff_dt)
        row = {
            "game_id": game_id,
            "game_datetime_utc": matchup.get("game_datetime_utc"),
            "home_team_id": home_id,
            "away_team_id": away_id,
            "home_team": home_name,
            "away_team": away_name,
            "neutral_site": neutral,

            "pred_spread": round(pred_spread, 2),
            "pred_total": round(prediction["predicted_total"], 2),
            "pred_home_score": round(prediction["predicted_total"] / 2 - pred_spread / 2, 1),
            "pred_away_score": round(prediction["predicted_total"] / 2 + pred_spread / 2, 1),
            "model_confidence": round(prediction["confidence"], 3),
            "pace_projected": round(prediction["pace"], 1),

            "home_net_eff": round(prediction["home_net_eff"], 2),
            "away_net_eff": round(prediction["away_net_eff"], 2),
            "home_off_eff_vs_exp": round(prediction["home_off_eff_vs_exp"], 2),
            "away_off_eff_vs_exp": round(prediction["away_off_eff_vs_exp"], 2),
            "eff_edge": round(bd.get("eff_edge", 0), 2),
            "composite_edge": round(bd.get("composite_edge", 0), 2),
            "hca": round(bd.get("hca", 0), 2),

            "efg_delta": round(bd.get("efg_delta", 0), 2),
            "tov_delta": round(bd.get("tov_delta", 0), 2),
            "orb_delta": round(bd.get("orb_delta", 0), 2),
            "drb_delta": round(bd.get("drb_delta", 0), 2),
            "ftr_delta": round(bd.get("ftr_delta", 0), 2),

            "spread_line": spread_line,
            "total_line": total_line,
            "home_ml": _safe_float(matchup.get("home_ml")),
            "away_ml": _safe_float(matchup.get("away_ml")),
            "spread_diff_vs_line": spread_diff,
            "total_diff_vs_line": total_diff,
            "spread_pick": spread_pick,
            "edge_flag": int(edge_flag),
            "total_direction": "OVER" if (total_diff or 0) > 2 else "UNDER" if (total_diff or 0) < -2 else "PUSH",

            "home_games_used": len(home_games),
            "away_games_used": len(away_games),

            "game_type": game_type,
            "predicted_at_utc": pd.Timestamp.now("UTC").isoformat(),
            "history_cutoff_utc": cutoff_dt.isoformat() if cutoff_dt is not None else None,

            # Required downstream team metadata
            "home_conference": home_ctx.get("conference"),
            "home_wins": home_ctx.get("wins"),
            "home_losses": home_ctx.get("losses"),
            "away_conference": away_ctx.get("conference"),
            "away_wins": away_ctx.get("wins"),
            "away_losses": away_ctx.get("losses"),

            # Required downstream box score fields (legacy uppercase contract)
            "home_FGA": home_ctx.get("fga"),
            "home_FGM": home_ctx.get("fgm"),
            "home_FTA": home_ctx.get("fta"),
            "home_FTM": home_ctx.get("ftm"),
            "home_TPA": home_ctx.get("tpa"),
            "home_TPM": home_ctx.get("tpm"),
            "home_ORB": home_ctx.get("orb"),
            "home_DRB": home_ctx.get("drb"),
            "home_RB": home_ctx.get("reb"),
            "home_TO": home_ctx.get("tov"),
            "home_AST": home_ctx.get("ast"),

            "away_FGA": away_ctx.get("fga"),
            "away_FGM": away_ctx.get("fgm"),
            "away_FTA": away_ctx.get("fta"),
            "away_FTM": away_ctx.get("ftm"),
            "away_TPA": away_ctx.get("tpa"),
            "away_TPM": away_ctx.get("tpm"),
            "away_ORB": away_ctx.get("orb"),
            "away_DRB": away_ctx.get("drb"),
            "away_RB": away_ctx.get("reb"),
            "away_TO": away_ctx.get("tov"),
            "away_AST": away_ctx.get("ast"),
        }

        row.update(uws_result)
        results.append(row)

    log.info(f"Predictions complete: {len(results)} generated, {skipped} skipped")
    return pd.DataFrame(results) if results else pd.DataFrame()


def _compute_uws_for_matchup(matchup: Dict, snapshot: pd.DataFrame, game_type: str) -> Dict:
    home_ml = _safe_float(matchup.get("home_ml"))
    away_ml = _safe_float(matchup.get("away_ml"))
    if home_ml is None or away_ml is None:
        return {}

    if home_ml < away_ml:
        fav_id = matchup.get("home_team_id")
        dog_id = matchup.get("away_team_id")
    else:
        fav_id = matchup.get("away_team_id")
        dog_id = matchup.get("home_team_id")

    fav_snap = snapshot.loc[str(fav_id)].to_dict() if str(fav_id) in snapshot.index else {}
    dog_snap = snapshot.loc[str(dog_id)].to_dict() if str(dog_id) in snapshot.index else {}
    if not fav_snap or not dog_snap:
        return {}

    try:
        uws = compute_underdog_winner_score(
            favorite_stats=fav_snap,
            underdog_stats=dog_snap,
            game_type=game_type,
        )
        return dict(uws.items())
    except Exception as exc:
        log.warning(f"UWS computation failed: {exc}")
        return {}


def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _latest_team_context(
    all_data: pd.DataFrame,
    team_id: str,
    cutoff_dt: Optional[pd.Timestamp],
) -> Dict[str, Optional[object]]:
    """Get latest available team row before cutoff for output enrichment."""
    rows = all_data[all_data["team_id"].astype(str) == str(team_id)].copy()
    if cutoff_dt is not None:
        rows = rows[rows["game_datetime_utc"] < cutoff_dt]
    rows = rows.sort_values("game_datetime_utc")

    if rows.empty:
        return {k: None for k in REQUIRED_TEAM_CONTEXT_FIELDS}

    latest = rows.iloc[-1]
    return {k: latest.get(k, None) for k in REQUIRED_TEAM_CONTEXT_FIELDS}


def _validate_prediction_output_schema(df: pd.DataFrame) -> None:
    """Integrity gate: required downstream columns must exist before write."""
    required = {
        "home_FGA", "home_FGM", "home_FTA", "home_FTM", "home_TPA", "home_TPM",
        "home_ORB", "home_DRB", "home_RB", "home_TO", "home_AST",
        "away_FGA", "away_FGM", "away_FTA", "away_FTM", "away_TPA", "away_TPM",
        "away_ORB", "away_DRB", "away_RB", "away_TO", "away_AST",
        "home_wins", "home_losses", "home_conference",
        "away_wins", "away_losses", "away_conference",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Prediction output missing required downstream fields: {missing}")


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_predictions(df: pd.DataFrame, label: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not df.empty:
        _validate_prediction_output_schema(df)
    dated_path = DATA_DIR / f"predictions_{label}.csv"
    df.to_csv(dated_path, index=False)
    df.to_csv(OUT_PREDICTIONS_LATEST, index=False)

    log.info(f"Wrote {len(df)} predictions -> {dated_path}")
    log.info("Updated predictions_latest.csv")
    return dated_path


def print_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("No predictions generated.")
        return

    print()
    print("=" * 110)
    print(f"{'MATCHUP':<44} {'PRED SPREAD':>12} {'PRED TOT':>9} {'LINE':>8} {'DIFF':>7} {'CONF':>6} {'EDGE':>5}")
    print("=" * 110)

    for _, row in df.iterrows():
        matchup = f"{row.get('home_team','')} vs {row.get('away_team','')}"[:43]
        spread = f"{row.get('pred_spread', 0.0):+.1f}"
        total = f"{row.get('pred_total', 0.0):.1f}"
        line = f"{row.get('spread_line', np.nan):+.1f}" if pd.notna(row.get("spread_line")) else "  N/A"
        diff = f"{row.get('spread_diff_vs_line', np.nan):+.1f}" if pd.notna(row.get("spread_diff_vs_line")) else "  N/A"
        conf = f"{row.get('model_confidence', 0.0):.0%}"
        edge = "⚡" if int(row.get("edge_flag", 0) or 0) == 1 else ""
        print(f"{matchup:<44} {spread:>12} {total:>9} {line:>8} {diff:>7} {conf:>6} {edge:>5}")

    print("=" * 110)

    edges = df[df.get("edge_flag", 0) == 1] if "edge_flag" in df.columns else pd.DataFrame()
    if not edges.empty:
        print(f"\n⚡ EDGE ALERTS ({len(edges)} games with spread diff > 3 pts vs line):")
        for _, row in edges.iterrows():
            print(
                f"   {row.get('home_team','')} vs {row.get('away_team','')}: "
                f"Model {row.get('pred_spread', 0.0):+.1f} vs Line {row.get('spread_line', 0.0):+.1f} "
                f"-> {row.get('spread_pick','')}"
            )

    if "uws_total" in df.columns:
        strong = df[pd.to_numeric(df["uws_total"], errors="coerce") >= 55]
        if not strong.empty:
            print(f"\n🚨 STRONG UPSET ALERTS ({len(strong)} games UWS >= 55):")
            for _, row in strong.iterrows():
                print(
                    f"   {row.get('home_team','')} vs {row.get('away_team','')}: "
                    f"UWS {row.get('uws_total', 0):.0f}/70"
                )

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run CBB predictions for scheduled games")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date YYYYMMDD (date mode). If omitted, rolling window mode is used.",
    )
    parser.add_argument(
        "--hours-ahead",
        type=int,
        default=40,
        help="Rolling window horizon in hours (default: 40). Only used if --date is omitted.",
    )
    parser.add_argument(
        "--game-type",
        type=str,
        default="regular",
        choices=["regular", "conf_tournament", "ncaa_r1", "ncaa_r2"],
        help="Tournament context for UWS multipliers",
    )
    parser.add_argument(
        "--decay",
        type=str,
        default="smooth",
        choices=["smooth", "plateau", "simple"],
        help="Game weight decay function",
    )
    parser.add_argument(
        "--min-games",
        type=int,
        default=3,
        help="Minimum games required to include a team (default: 3)",
    )
    args = parser.parse_args()

    log.info(f"{'='*70}")
    log.info(f"CBB Prediction Runner")
    log.info(f"Game type: {args.game_type} | Decay: {args.decay} | Min games: {args.min_games}")
    if args.date:
        log.info(f"Mode: date | Target: {args.date}")
    else:
        log.info(f"Mode: rolling | Hours ahead: {args.hours_ahead}")
    log.info(f"{'='*70}")

    all_data = load_team_game_data()
    snapshot = load_tournament_snapshot()

    if args.date:
        # Date mode: predict scheduled games on that date.
        target_date = args.date
        matchups = get_matchups_for_date(target_date)
        if not matchups:
            log.warning(f"No scheduled games found for {target_date}. Exiting.")
            sys.exit(0)

        # Default cutoff if kickoff missing: start of target date UTC
        default_cutoff_dt = pd.Timestamp(
            f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}",
            tz="UTC"
        )

        label = target_date
        log.info(f"Running predictions for {len(matchups)} games")
    else:
        # Rolling mode: now -> now+hours
        start_local = datetime.now(TZ)
        end_local = start_local + timedelta(hours=int(args.hours_ahead))
        matchups = get_matchups_in_window(start_local, end_local)
        if not matchups:
            log.warning("No scheduled games found in rolling window. Exiting.")
            sys.exit(0)

        default_cutoff_dt = None
        label = datetime.now(ZoneInfo("UTC")).strftime("%Y%m%dT%H%M%SZ")
        log.info(f"Running predictions for {len(matchups)} games in rolling window")

    config = ModelConfig(
        decay_type=args.decay,
        min_games_for_full_confidence=args.min_games,
    )
    model = CBBPredictionModel(config)

    results_df = run_predictions(
        matchups=matchups,
        all_data=all_data,
        model=model,
        snapshot=snapshot,
        game_type=args.game_type,
        default_cutoff_dt=default_cutoff_dt,
    )

    if results_df.empty:
        log.warning("No predictions generated. Check team data coverage.")
        sys.exit(0)

    out_path = write_predictions(results_df, label=label)
    print_summary(results_df)

    log.info(f"Done. Output: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
