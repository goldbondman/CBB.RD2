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
import json
import math
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
    OUT_RANKINGS as CSV_RANKINGS,
    TZ,
)
try:
    from espn_config import get_game_tier, conference_id_to_name
except ImportError:
    def get_game_tier(home_conf: str, away_conf: str) -> str:
        return "unknown"
    def conference_id_to_name(cid) -> str:
        return str(cid)

from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from config.logging_config import get_logger
from config.model_version import compute_model_version, save_version_to_history
from pipeline_csv_utils import add_conference_name, safe_write_csv
from pipeline_csv_utils import normalize_numeric_dtypes, safe_write_csv
from pipeline_csv_utils import normalize_column_names, safe_write_csv
from models.alpha_evaluator import evaluate_alpha

OUT_PREDICTIONS_LATEST = DATA_DIR / "predictions_latest.csv"

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

log = get_logger(__name__)

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

    Integrity fallback: if the canonical root data file is missing,
    automatically try data/csv/<filename> so pipeline runs in repos that
    keep raw artifacts under data/csv.
    """
    for path, label in [
        (CSV_WEIGHTED, "team_game_weighted"),
        (CSV_METRICS,  "team_game_metrics"),
        (CSV_LOGS,     "team_game_logs"),
    ]:
        candidate_paths = [path, DATA_DIR / "csv" / path.name]
        chosen = next(
            (p for p in candidate_paths if p.exists() and p.stat().st_size > 100),
            None,
        )
        if chosen is not None:
            log.info(
                "Loading team data from %s (%s bytes)",
                chosen,
                f"{chosen.stat().st_size:,}",
            )
            df = pd.read_csv(chosen, dtype=str, low_memory=False)
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


def load_team_context_data() -> pd.DataFrame:
    """Load season-aggregated context rows (one latest row per team)."""
    team_context_sources = [
        CSV_SNAPSHOT,
        CSV_WEIGHTED,
        DATA_DIR / "team_pretournament_snapshot.csv",
        DATA_DIR / "team_game_weighted.csv",
    ]

    for src in team_context_sources:
        if src.exists() and src.stat().st_size > 100:
            ctx_df = pd.read_csv(src, dtype=str, low_memory=False)
            if "game_datetime_utc" in ctx_df.columns:
                ctx_df["game_datetime_utc"] = pd.to_datetime(
                    ctx_df["game_datetime_utc"], utc=True, errors="coerce"
                )
                ctx_df = ctx_df.sort_values("game_datetime_utc")
            ctx_df = ctx_df.drop_duplicates("team_id", keep="last")
            log.info("Team context: %d teams from %s", len(ctx_df), src)
            return ctx_df

    log.warning("No team context source found — predictions will use defaults")
    return pd.DataFrame()


def load_games_schedule() -> pd.DataFrame:
    """Load games.csv scoreboard data for scheduled games fallback."""
    schedule_path = CSV_GAMES if CSV_GAMES.exists() else (DATA_DIR / "csv" / CSV_GAMES.name)
    if not schedule_path.exists():
        log.warning("%s not found. Cannot use fallback schedule.", CSV_GAMES)
        return pd.DataFrame()

    df = pd.read_csv(schedule_path, dtype=str, low_memory=False)
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
    context_df: pd.DataFrame,
    model: CBBPredictionModel,
    snapshot: Optional[pd.DataFrame],
    version: Dict,
    game_type: str = "regular",
    default_cutoff_dt: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Runs predictions for matchups.
    Leak-free: cutoff is per-game kickoff time if available, else default_cutoff_dt.
    """
    all_data = normalize_numeric_dtypes(all_data)
    context_df = normalize_numeric_dtypes(context_df)

    results: List[Dict] = []
    _first_call = True
    skipped = 0
    rankings_df = _load_rankings()
    schedule_df = load_games_schedule()
    team_schedule_df = build_team_schedule_index(schedule_df)

    for matchup in matchups:
        home_id = matchup.get("home_team_id")
        away_id = matchup.get("away_team_id")
        home_name = matchup.get("home_team")
        away_name = matchup.get("away_team")
        game_id = matchup.get("game_id")
        neutral = bool(matchup.get("neutral_site", False))

        kick_utc = _parse_utc_dt(matchup.get("game_datetime_utc"))
        cutoff_dt = kick_utc if kick_utc is not None else default_cutoff_dt
        if cutoff_dt is None:
            # Conservative fallback for malformed/missing kickoff datetimes:
            # never include future games relative to prediction runtime.
            cutoff_dt = pd.Timestamp.now(tz="UTC")
            log.warning(
                "    Missing kickoff for game_id=%s; using current UTC as leak-safe cutoff (%s)",
                game_id,
                cutoff_dt.isoformat(),
            )

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
            home_ctx = _latest_team_context(all_data, context_df, str(home_id), cutoff_dt)
            away_ctx = _latest_team_context(all_data, context_df, str(away_id), cutoff_dt)
            prediction = model.predict_game(
                home_games=home_games,
                away_games=away_games,
                neutral_site=neutral,
                game_type=game_type,
                home_team_profile=home_ctx,
                away_team_profile=away_ctx,
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
        home_ctx = _latest_team_context(all_data, context_df, str(home_id), cutoff_dt)
        away_ctx = _latest_team_context(all_data, context_df, str(away_id), cutoff_dt)

        _home_em = float(home_ctx.get("net_eff") or home_ctx.get("adj_net_rtg") or home_ctx.get("net_rtg") or 0.0)
        _away_em = float(away_ctx.get("net_eff") or away_ctx.get("adj_net_rtg") or away_ctx.get("net_rtg") or 0.0)
        _em_diff = _home_em - _away_em
        _hca_boost = 3.5
        home_win_prob = round(
            1.0 / (1.0 + math.exp(-(_em_diff + _hca_boost) / 10.0)), 4
        )

        game_date = str(matchup.get("game_datetime_utc") or "")
        trap_home = detect_trap_game(str(home_id), game_date, team_schedule_df, rankings_df)
        trap_away = detect_trap_game(str(away_id), game_date, team_schedule_df, rankings_df)
        revenge = detect_revenge_spot(str(home_id), str(away_id), game_date, schedule_df)

        favored_team = "home" if pred_spread < 0 else "away"
        trap_for_favorite = bool(trap_home.get("trap_game_flag") if favored_team == "home" else trap_away.get("trap_game_flag"))

        if revenge.get("revenge_flag"):
            if revenge.get("revenge_team") == "home":
                pred_spread = round(pred_spread - 0.5, 2)
            elif revenge.get("revenge_team") == "away":
                pred_spread = round(pred_spread + 0.5, 2)

        alpha = evaluate_alpha(
            pred_spread=pred_spread,
            spread_line=spread_line,
            model_confidence=prediction["confidence"],
            trap_for_favorite=trap_for_favorite,
            revenge_info=revenge,
            market_context=None,
            game_id=str(game_id),
            home_team=str(home_name),
            away_team=str(away_name),
        )
        totals_proj = model_total(home_ctx, away_ctx, log_actuals=_first_call)
        _first_call = False
        line_advisory = line_shopping_advisory(pred_spread, spread_line)

        _poss = totals_proj.get("projected_poss")

        all_home_games = all_data[
            (all_data["team_id"].astype(str) == str(home_id)) &
            (all_data["game_datetime_utc"] < cutoff_dt)
        ]
        all_away_games = all_data[
            (all_data["team_id"].astype(str) == str(away_id)) &
            (all_data["game_datetime_utc"] < cutoff_dt)
        ]

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
            "projected_total": totals_proj.get("projected_total"),
            "projected_poss": _poss,
            "pace_projected": _poss,
            "total_confidence_adj": totals_proj.get("total_confidence_adj"),
            "pred_home_score": round(prediction["predicted_total"] / 2 - pred_spread / 2, 1),
            "pred_away_score": round(prediction["predicted_total"] / 2 + pred_spread / 2, 1),
            "model_confidence": round(prediction["confidence"], 3),
            "kelly_fraction": alpha["kelly_fraction"],
            "kelly_units": alpha["kelly_units"],
            "kelly_multiplier": alpha["kelly_multiplier"],
            "market_evaluated": alpha["market_evaluated"],
            "edge_pts": alpha["edge_pts"],
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
            "total_edge": (round(totals_proj.get("projected_total", 0) - total_line, 2)
                           if total_line is not None else None),
            "line_shopping_advisory": line_advisory,
            "spread_pick": spread_pick,
            "edge_flag": int(edge_flag),
            "total_direction": "OVER" if (total_diff or 0) > 2 else "UNDER" if (total_diff or 0) < -2 else "PUSH",

            "home_games_used": len(all_home_games),
            "away_games_used": len(all_away_games),
            "home_win_prob": home_win_prob,

            "game_type": game_type,
            "predicted_at_utc": pd.Timestamp.now("UTC").isoformat(),
            "history_cutoff_utc": cutoff_dt.isoformat() if cutoff_dt is not None else None,
            "model_version_hash": version["model_version_hash"],
            "pipeline_run_id": version["pipeline_run_id"],
            "model_weights_used": json.dumps(
                version["config_snapshot"].get("weights", {})
            ),
            "active_bias_corrections": version["config_snapshot"].get(
                "active_bias_corrections", 0
            ),

            # Required downstream team metadata
            "home_conference": home_ctx.get("conference"),
            "home_wins": home_ctx.get("wins", 0),
            "home_losses": home_ctx.get("losses", 0),
            "away_conference": away_ctx.get("conference"),
            "away_wins": away_ctx.get("wins", 0),
            "away_losses": away_ctx.get("losses", 0),
            "game_tier": get_game_tier(
                home_ctx.get("conference", ""),
                away_ctx.get("conference", ""),
            ),

            "home_form_rating": home_ctx.get("form_rating"),
            "away_form_rating": away_ctx.get("form_rating"),
            "home_momentum_score": home_ctx.get("momentum_score"),
            "away_momentum_score": away_ctx.get("momentum_score"),
            "home_momentum_tier": home_ctx.get("momentum_tier"),
            "away_momentum_tier": away_ctx.get("momentum_tier"),
            "home_luck_score": home_ctx.get("luck_score"),
            "away_luck_score": away_ctx.get("luck_score"),
            "home_ha_net_rtg_l10": home_ctx.get("ha_net_rtg_l10"),
            "away_ha_net_rtg_l10": away_ctx.get("ha_net_rtg_l10"),
            "home_net_rtg_std_l10": home_ctx.get("net_rtg_std_l10"),
            "away_net_rtg_std_l10": away_ctx.get("net_rtg_std_l10"),

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

            "dead_spread_flag": int((home_ctx.get("dead_spread_flag") or 0) > 0 or (away_ctx.get("dead_spread_flag") or 0) > 0),
            "trap_game_flag": int(bool(trap_home.get("trap_game_flag") or trap_away.get("trap_game_flag"))),
            "trap_game_reason": trap_home.get("trap_game_reason") or trap_away.get("trap_game_reason") or "",
            "revenge_flag": int(bool(revenge.get("revenge_flag"))),
            "revenge_team": revenge.get("revenge_team", ""),
            "home_fatigue_index": _safe_float(home_ctx.get("fatigue_index")),
            "away_fatigue_index": _safe_float(away_ctx.get("fatigue_index")),
            "alpha_reasoning": alpha.get("alpha_reasoning", ""),
            "is_alpha": alpha.get("is_alpha", False),
            "edge_types": alpha.get("edge_types", ""),
        }

        row.update(uws_result)
        results.append(row)

    res_df = pd.DataFrame(results) if results else pd.DataFrame()
    log.info(
        "Predictions complete: %d generated, %d skipped | median home_win_prob: %.3f",
        len(res_df), skipped, res_df["home_win_prob"].median() if not res_df.empty else 0.0
    )
    return res_df


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




def _load_rankings() -> pd.DataFrame:
    if CSV_RANKINGS.exists() and CSV_RANKINGS.stat().st_size > 100:
        df = pd.read_csv(CSV_RANKINGS, dtype=str, low_memory=False)
        if "team_id" in df.columns:
            if "cage_rank" not in df.columns and "rank" in df.columns:
                df["cage_rank"] = pd.to_numeric(df["rank"], errors="coerce")
            else:
                df["cage_rank"] = pd.to_numeric(df.get("cage_rank"), errors="coerce")
            return df
    return pd.DataFrame(columns=["team_id", "cage_rank"])


def build_team_schedule_index(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(columns=["team_id", "opponent_id", "game_datetime_utc"])

    base = schedule_df.copy()
    home = pd.DataFrame({
        "team_id": base.get("home_team_id", ""),
        "opponent_id": base.get("away_team_id", ""),
        "game_datetime_utc": base.get("game_datetime_utc", ""),
    })
    away = pd.DataFrame({
        "team_id": base.get("away_team_id", ""),
        "opponent_id": base.get("home_team_id", ""),
        "game_datetime_utc": base.get("game_datetime_utc", ""),
    })
    out = pd.concat([home, away], ignore_index=True)
    out["team_id"] = out["team_id"].astype(str)
    out["opponent_id"] = out["opponent_id"].astype(str)
    out["game_datetime_utc"] = out["game_datetime_utc"].astype(str)
    out = out.sort_values(["team_id", "game_datetime_utc"]).reset_index(drop=True)
    return out


def detect_trap_game(team_id: str, game_date: str, schedule_df: pd.DataFrame, rankings_df: pd.DataFrame) -> dict:
    """Flag potential trap-game spot for a ranked team facing weak opposition."""
    if schedule_df.empty or rankings_df.empty:
        return {"trap_game_flag": False, "trap_game_reason": ""}

    team_games = schedule_df[schedule_df["team_id"].astype(str) == str(team_id)].sort_values("game_datetime_utc").reset_index(drop=True)
    mask = team_games["game_datetime_utc"].astype(str).str[:10] == str(game_date)[:10]
    game_idx = team_games[mask].index
    if len(game_idx) == 0:
        return {"trap_game_flag": False, "trap_game_reason": ""}

    pos = int(game_idx[0])
    prev_game = team_games.iloc[pos - 1] if pos > 0 else None
    next_game = team_games.iloc[pos + 1] if pos < len(team_games) - 1 else None

    rank_map = dict(zip(rankings_df["team_id"].astype(str), pd.to_numeric(rankings_df["cage_rank"], errors="coerce").fillna(999)))
    opp_rank = rank_map.get(str(team_games.iloc[pos].get("opponent_id", "")), 999)
    prev_rank = rank_map.get(str(prev_game.get("opponent_id", "")), 999) if prev_game is not None else 999
    next_rank = rank_map.get(str(next_game.get("opponent_id", "")), 999) if next_game is not None else 999
    team_rank = rank_map.get(str(team_id), 999)

    is_trap = bool(team_rank <= 40 and opp_rank > 100 and (prev_rank <= 40 or next_rank <= 40))
    return {
        "trap_game_flag": is_trap,
        "trap_game_reason": (
            f"Ranked team (#{int(team_rank)}) vs weak opp (#{int(opp_rank)}) between quality games"
            if is_trap else ""
        ),
    }


def detect_revenge_spot(home_team_id: str, away_team_id: str, game_date: str, results_df: pd.DataFrame, lookback_days: int = 45) -> dict:
    """Check if either side recently lost this same matchup and is in a revenge spot."""
    if results_df.empty:
        return {"revenge_flag": False, "revenge_team": "", "revenge_margin": None}

    cutoff = pd.Timestamp(game_date, tz="UTC") - pd.Timedelta(days=lookback_days)
    results = results_df.copy()
    results["dt"] = pd.to_datetime(results.get("game_datetime_utc"), utc=True, errors="coerce")
    recent = results[results["dt"] >= cutoff]

    home_lost = recent[(recent["home_team_id"].astype(str) == str(home_team_id)) &
                       (recent["away_team_id"].astype(str) == str(away_team_id)) &
                       (pd.to_numeric(recent["home_score"], errors="coerce") < pd.to_numeric(recent["away_score"], errors="coerce"))]
    away_lost = recent[(recent["home_team_id"].astype(str) == str(away_team_id)) &
                       (recent["away_team_id"].astype(str) == str(home_team_id)) &
                       (pd.to_numeric(recent["home_score"], errors="coerce") > pd.to_numeric(recent["away_score"], errors="coerce"))]

    if len(home_lost) > 0:
        r = home_lost.sort_values("dt").iloc[-1]
        return {"revenge_flag": True, "revenge_team": "home", "revenge_margin": int(float(r["away_score"]) - float(r["home_score"]))}
    if len(away_lost) > 0:
        r = away_lost.sort_values("dt").iloc[-1]
        return {"revenge_flag": True, "revenge_team": "away", "revenge_margin": int(float(r["home_score"]) - float(r["away_score"]))}

    return {"revenge_flag": False, "revenge_team": "", "revenge_margin": None}


def line_shopping_advisory(model_spread: float, closing_line: Optional[float]) -> str:
    """Flag near-threshold edges where shopping a half-point matters."""
    if closing_line is None:
        return ""
    edge = abs(model_spread - closing_line)
    if 2.0 <= edge <= 4.0:
        key_number = None
        for n in [1, 2, 3, 5, 6, 7]:
            if abs(closing_line) % n < 0.6:
                key_number = n
                break
        if key_number:
            return (f"LINE SHOP: edge {edge:.1f}pts — closing on key number {key_number}. "
                    f"Half point could be critical.")
    return ""


def model_total(team_a: dict, team_b: dict, log_actuals: bool = False) -> dict:
    """Dedicated totals model using pace + ortg/drtg interaction."""
    LEAGUE_AVG_PACE = 67.2
    LEAGUE_AVG_ORTG = 110.0

    def get_metric(team: dict, cols: List[str], default: float) -> float:
        for col in cols:
            val = team.get(col)
            if val is not None and pd.notna(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return default

    _home_pace = get_metric(team_a, ["pace", "adj_pace", "cage_t"], LEAGUE_AVG_PACE)
    _away_pace = get_metric(team_b, ["pace", "adj_pace", "cage_t"], LEAGUE_AVG_PACE)
    projected_poss = round((_home_pace + _away_pace) / 2, 1)

    _home_ortg = get_metric(team_a, ["ortg", "adj_ortg", "cage_o"], LEAGUE_AVG_ORTG)
    _away_ortg = get_metric(team_b, ["ortg", "adj_ortg", "cage_o"], LEAGUE_AVG_ORTG)
    projected_total = round(((_home_ortg + _away_ortg) * projected_poss) / 100, 1)

    if log_actuals:
        log.info(
            "model_total first call: home_pace=%.1f, away_pace=%.1f, home_ortg=%.1f, away_ortg=%.1f -> total=%.1f",
            _home_pace, _away_pace, _home_ortg, _away_ortg, projected_total
        )

    _home_games = _safe_int(team_a.get("games_played") or team_a.get("game_number") or 0, default=0)
    _away_games = _safe_int(team_b.get("games_played") or team_b.get("game_number") or 0, default=0)
    _min_games = min(_home_games, _away_games)
    total_confidence_adj = round(min(1.0, 0.6 + (_min_games / 25.0) * 0.4), 3)

    score_a = (_home_ortg / 100) * projected_poss
    score_b = (_away_ortg / 100) * projected_poss

    return {
        "projected_total": projected_total,
        "projected_score_a": round(score_a, 1),
        "projected_score_b": round(score_b, 1),
        "projected_poss": projected_poss,
        "total_confidence_adj": total_confidence_adj,
    }


def _safe_float(val, default: Optional[float] = None) -> Optional[float]:
    try:
        if val is None:
            return default
        out = float(val)
        return default if pd.isna(out) else out
    except (TypeError, ValueError):
        return default


def _latest_team_context(
    all_data: pd.DataFrame,
    context_df: pd.DataFrame,
    team_id: str,
    cutoff_dt: Optional[pd.Timestamp],
) -> Dict[str, Optional[object]]:
    """Get latest available team row before cutoff for output enrichment."""
    rows = pd.DataFrame()
    if not context_df.empty and "team_id" in context_df.columns:
        rows = context_df[context_df["team_id"].astype(str) == str(team_id)].copy()
        if "game_datetime_utc" in rows.columns and cutoff_dt is not None:
            rows = rows[rows["game_datetime_utc"] < cutoff_dt]

    if rows.empty:
        rows = all_data[all_data["team_id"].astype(str) == str(team_id)].copy()
        if cutoff_dt is not None:
            rows = rows[rows["game_datetime_utc"] < cutoff_dt]
        rows = rows.sort_values("game_datetime_utc")

    if rows.empty:
        return {k: None for k in REQUIRED_TEAM_CONTEXT_FIELDS}

    latest = rows.iloc[-1]
    out = latest.to_dict()

    raw_conf = out.get("conference") or out.get("conf_id", "")
    if str(raw_conf).strip().isdigit():
        out["conference"] = conference_id_to_name(str(raw_conf).strip())

    try:
        all_team_rows = all_data[
            all_data["team_id"].astype(str) == str(team_id)
        ].copy()
        if cutoff_dt is not None:
            all_team_rows = all_team_rows[
                all_team_rows["game_datetime_utc"] < cutoff_dt
            ]
        if not all_team_rows.empty:
            per_game_wins = pd.to_numeric(
                all_team_rows.get(
                    "wins",
                    all_team_rows.get("win", pd.Series(dtype=float)),
                ),
                errors="coerce",
            ).fillna(0)

            max_val = per_game_wins.max()
            if pd.isna(max_val) or max_val <= 1:
                season_wins = int(per_game_wins.sum())
                season_losses = int(len(all_team_rows) - season_wins)
            else:
                season_wins = int(per_game_wins.iloc[-1])
                loss_col = pd.to_numeric(
                    all_team_rows.get("losses", pd.Series(dtype=float)),
                    errors="coerce",
                ).fillna(0)
                season_losses = int(loss_col.iloc[-1])

            out["wins"] = season_wins
            out["losses"] = season_losses
        else:
            out["wins"] = 0
            out["losses"] = 0
    except Exception as e:  # noqa: BLE001
        log.debug("Could not compute season record for %s: %s", team_id, e)
        out.setdefault("wins", 0)
        out.setdefault("losses", 0)

    for k in REQUIRED_TEAM_CONTEXT_FIELDS:
        out.setdefault(k, None)
    return out


def _validate_prediction_output_schema(df: pd.DataFrame) -> None:
    """Integrity gate: required downstream columns must exist before write."""
    from cbb_output_schemas import validate_output
    validate_output(df, "predictions", strict=True)


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_predictions(df: pd.DataFrame, label: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_df = normalize_column_names(df)

    # Backward-compatible normalization for downstream schema + dedupe contracts.
    # Keep legacy columns (game_id/pred_spread) while adding canonical aliases
    # expected by predictions_latest validation (event_id/predicted_spread).
    if "event_id" not in out_df.columns and "game_id" in out_df.columns:
        out_df["event_id"] = out_df["game_id"]
        log.info("Normalized prediction output: added event_id from game_id")
    if "predicted_spread" not in out_df.columns and "pred_spread" in out_df.columns:
        out_df["predicted_spread"] = out_df["pred_spread"]
        log.info("Normalized prediction output: added predicted_spread from pred_spread")

    out_df = add_conference_name(out_df)

    if not out_df.empty:
        _validate_prediction_output_schema(out_df)
    dated_path = DATA_DIR / f"predictions_{label}.csv"
    safe_write_csv(out_df, dated_path, index=False, label="predictions_dated", allow_empty=True)
    safe_write_csv(out_df, OUT_PREDICTIONS_LATEST, index=False, label="predictions_latest", allow_empty=True)

    for written_path in (dated_path, OUT_PREDICTIONS_LATEST):
        if not written_path.exists():
            raise RuntimeError(f"Predictions write failed: file was not created at {written_path}")
        written_df = pd.read_csv(written_path, dtype={"event_id": str, "game_id": str}, low_memory=False)
        if written_df.empty:
            raise RuntimeError(f"Predictions write failed: file is empty at {written_path}")
        missing_cols = [c for c in ["event_id", "game_id", "pred_spread"] if c not in written_df.columns]
        if missing_cols:
            raise RuntimeError(
                f"Predictions write failed: missing required columns {missing_cols} in {written_path}"
            )

    log.info(f"Wrote {len(out_df)} predictions -> {dated_path}")
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

    version = compute_model_version(DATA_DIR)
    save_version_to_history(version, DATA_DIR / "model_version_history.json")
    log.info(
        f"Model version: {version['model_version_hash']} | "
        f"Run: {version['pipeline_run_id']}"
    )

    log.info(f"{'='*70}")
    log.info(f"CBB Prediction Runner")
    log.info(f"Game type: {args.game_type} | Decay: {args.decay} | Min games: {args.min_games}")
    if args.date:
        log.info(f"Mode: date | Target: {args.date}")
    else:
        log.info(f"Mode: rolling | Hours ahead: {args.hours_ahead}")
    log.info(f"{'='*70}")

    all_data = load_team_game_data()
    context_df = load_team_context_data()
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
        context_df=context_df,
        model=model,
        snapshot=snapshot,
        version=version,
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
