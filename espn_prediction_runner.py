#!/usr/bin/env python3
"""
ESPN CBB Pipeline — Prediction Runner
Bridges team_game_weighted.csv → CBBPredictionModel → predictions_YYYYMMDD.csv

Reads from the same data/ directory the main pipeline writes.
Fetches scheduled games, builds GameData objects (including recursive opponent_history),
runs CBBPredictionModel.predict_game_with_components(), and writes outputs.

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
import dataclasses
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
    SEASON_ACTIVE,
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
from cbb_situational import (
    detect_trap_game,
    detect_revenge_spot,
    line_shopping_advisory,
    model_total,
    build_team_schedule_index,
)

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
    from cbb_ensemble import EnsemblePredictor, EnsembleConfig, TeamProfile
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

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
    "pace", "ortg", "adj_pace", "adj_ortg", "adj_drtg", "drtg", "net_eff", "adj_net_rtg",
]

OPP_HISTORY_WINDOW = 5
TEAM_GAMES_WINDOW = 10


def apply_active_weights(config: ModelConfig) -> None:
    """Overlay deployed weights onto ModelConfig when available."""
    active_weights_path = Path("data/active_weights.json")
    if not active_weights_path.exists():
        return

    try:
        weights = json.loads(active_weights_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - keep runner resilient to malformed weight files
        log.warning("Unable to load %s: %s", active_weights_path, exc)
        return

    applied = 0
    for field in dataclasses.fields(config):
        if field.name in weights:
            setattr(config, field.name, weights[field.name])
            applied += 1

    if applied:
        log.info("Applied %d active weight override(s) from %s", applied, active_weights_path)



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
            ctx_df = normalize_numeric_dtypes(ctx_df)
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
    skipped_records: List[Dict[str, str]] = []
    rankings_df = _load_rankings()
    schedule_df = load_games_schedule()
    team_schedule_df = build_team_schedule_index(schedule_df)
    args_min_games = int(getattr(getattr(model, "config", None), "min_games_for_full_confidence", 0) or 0)
    ensemble = EnsemblePredictor(EnsembleConfig.from_optimized()) if ENSEMBLE_AVAILABLE else None

    for i, matchup in enumerate(matchups):
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
            skipped_records.append({
                "game_id": str(game_id or ""),
                "home_team": str(home_name or ""),
                "away_team": str(away_name or ""),
                "game_datetime_utc": str(matchup.get("game_datetime_utc") or ""),
                "skip_reason": "insufficient_history",
                "skip_detail": (
                    f"home={len(home_games)} games, away={len(away_games)} games, "
                    f"min={args_min_games}"
                ),
                "skipped_at_utc": pd.Timestamp.now("UTC").isoformat(),
            })
            skipped += 1
            continue

        try:
            home_ctx = _latest_team_context(all_data, context_df, str(home_id), cutoff_dt)
            away_ctx = _latest_team_context(all_data, context_df, str(away_id), cutoff_dt)
            _home_has_pace = home_ctx.get("pace") is not None or home_ctx.get("poss") is not None
            _away_has_pace = away_ctx.get("pace") is not None or away_ctx.get("poss") is not None
            if not _home_has_pace or not _away_has_pace:
                log.warning(
                    "[CTX] game_id=%s missing pace context: home_pace=%s away_pace=%s",
                    game_id,
                    home_ctx.get("pace") or home_ctx.get("poss"),
                    away_ctx.get("pace") or away_ctx.get("poss"),
                )
            prediction = model.predict_game_with_components(
                home_games=home_games,
                away_games=away_games,
                neutral_site=neutral,
                game_type=game_type,
                home_team_profile=home_ctx,
                away_team_profile=away_ctx,
            )
        except Exception as exc:
            log.error(f"    Prediction failed for {game_id}: {exc}")
            skipped_records.append({
                "game_id": str(game_id or ""),
                "home_team": str(home_name or ""),
                "away_team": str(away_name or ""),
                "game_datetime_utc": str(matchup.get("game_datetime_utc") or ""),
                "skip_reason": "prediction_exception",
                "skip_detail": str(exc),
                "skipped_at_utc": pd.Timestamp.now("UTC").isoformat(),
            })
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
            "eff_edge": round(prediction.get("model_eff_edge", 0), 2),
            "composite_edge": round(prediction.get("model_composite_edge", 0), 2),
            "hca": round(prediction.get("model_hca_applied", bd.get("hca", 0)), 2),

            "model_efg_delta": round(prediction.get("model_efg_delta", 0), 2),
            "model_tov_delta": round(prediction.get("model_tov_delta", 0), 2),
            "model_orb_delta": round(prediction.get("model_orb_delta", 0), 2),
            "model_drb_delta": round(prediction.get("model_drb_delta", 0), 2),
            "model_ftr_delta": round(prediction.get("model_ftr_delta", 0), 2),
            "model_tpar_delta": prediction.get("model_tpar_delta"),
            "model_eff_edge": round(prediction.get("model_eff_edge", 0), 2),
            "model_composite_edge": round(prediction.get("model_composite_edge", 0), 2),
            "model_raw_edge": round(prediction.get("model_raw_edge", 0), 2),
            "model_home_pace": round(prediction.get("model_home_pace", 0), 1),
            "model_away_pace": round(prediction.get("model_away_pace", 0), 1),
            "model_hca_applied": round(prediction.get("model_hca_applied", 0), 2),

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

            "home_games_used": int(all_data[(all_data["team_id"].astype(str) == str(home_id)) & (all_data["game_datetime_utc"] < cutoff_dt)].shape[0]),
            "away_games_used": int(all_data[(all_data["team_id"].astype(str) == str(away_id)) & (all_data["game_datetime_utc"] < cutoff_dt)].shape[0]),
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

        if ensemble is not None:
            home_profile = TeamProfile(
                team_id=str(home_id or ""),
                team_name=str(home_name or ""),
                conference=str(home_ctx.get("conference") or ""),
                games_before=int(len(home_games)),
                cage_em=float(home_ctx.get("adj_net_rtg") or home_ctx.get("net_eff") or 0.0),
                cage_o=float(home_ctx.get("adj_ortg") or home_ctx.get("ortg") or 0.0),
                cage_d=float(home_ctx.get("adj_drtg") or home_ctx.get("drtg") or 0.0),
                cage_t=float(home_ctx.get("adj_pace") or home_ctx.get("pace") or 0.0),
                barthag=float(home_ctx.get("barthag") or 0.5),
            )
            away_profile = TeamProfile(
                team_id=str(away_id or ""),
                team_name=str(away_name or ""),
                conference=str(away_ctx.get("conference") or ""),
                games_before=int(len(away_games)),
                cage_em=float(away_ctx.get("adj_net_rtg") or away_ctx.get("net_eff") or 0.0),
                cage_o=float(away_ctx.get("adj_ortg") or away_ctx.get("ortg") or 0.0),
                cage_d=float(away_ctx.get("adj_drtg") or away_ctx.get("drtg") or 0.0),
                cage_t=float(away_ctx.get("adj_pace") or away_ctx.get("pace") or 0.0),
                barthag=float(away_ctx.get("barthag") or 0.5),
            )
            ens_result = ensemble.predict(
                home_profile,
                away_profile,
                neutral=neutral,
                spread_line=spread_line,
                total_line=total_line,
                primary_spread=prediction["predicted_spread"],
            )
            row["ensemble_spread"] = round(ens_result.spread, 2)
            row["m8_spread"] = round(float(prediction["predicted_spread"]), 2)

        row.update(uws_result)
        results.append(row)

    res_df = pd.DataFrame(results) if results else pd.DataFrame()

    if not res_df.empty:
        null_count = res_df["pred_spread"].isna().sum()
        null_pct = (null_count / len(res_df)) * 100
        unique_totals = res_df["projected_total"].nunique()

        log.info(
            "[DIAG] run_predictions | pred_spread: %d/%d non-null (%.1f%% null) | projected_total unique: %d",
            len(res_df) - null_count, len(res_df), null_pct, unique_totals
        )

        if unique_totals <= 1 and len(res_df) > 1:
            log.warning("[DIAG] projected_total unique count is %d for %d games", unique_totals, len(res_df))

    if skipped_records:
        dq_path = DATA_DIR / "dq_skipped_games.csv"
        dq_df = pd.DataFrame(skipped_records)
        if dq_path.exists():
            existing = pd.read_csv(dq_path)
            dq_df = pd.concat([existing, dq_df], ignore_index=True)
            dq_df["skipped_at_utc"] = pd.to_datetime(dq_df["skipped_at_utc"], utc=True, errors="coerce")
            cutoff = pd.Timestamp.now("UTC") - pd.Timedelta(days=30)
            dq_df = dq_df[dq_df["skipped_at_utc"] > cutoff]

        dq_df.to_csv(dq_path, index=False)
        log.warning(
            "[DQ] %d games skipped this run — reasons: %s | written to %s",
            len(skipped_records),
            dq_df["skip_reason"].value_counts().to_dict(),
            dq_path,
        )

    log.info(f"Predictions complete: {len(results)} generated, {skipped} skipped")
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
            _gdt = pd.to_datetime(rows["game_datetime_utc"], utc=True, errors="coerce")
            _cutoff_safe = cutoff_dt if cutoff_dt.tzinfo is not None else cutoff_dt.tz_localize("UTC")
            rows = rows[_gdt < _cutoff_safe]

    if rows.empty:
        rows = all_data[all_data["team_id"].astype(str) == str(team_id)].copy()
        if cutoff_dt is not None:
            _gdt = pd.to_datetime(rows["game_datetime_utc"], utc=True, errors="coerce")
            _cutoff_safe = cutoff_dt if cutoff_dt.tzinfo is not None else cutoff_dt.tz_localize("UTC")
            rows = rows[_gdt < _cutoff_safe]
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

    _NUMERIC_CTX_FIELDS = [
        "pace", "ortg", "drtg", "poss", "net_rtg",
        "ortg_l5", "ortg_l10", "drtg_l5", "drtg_l10",
        "pace_l5", "pace_l10", "poss_l5", "poss_l10",
        "adj_pace", "adj_ortg", "adj_drtg", "adj_net_rtg",
        "net_eff", "net_rtg_l5", "net_rtg_l10",
        "efg_pct", "tov_pct", "orb_pct", "drb_pct", "ftr",
        "efg_pct_l5", "tov_pct_l5", "orb_pct_l5",
    ]
    for _nf in _NUMERIC_CTX_FIELDS:
        if _nf in out and out[_nf] is not None:
            try:
                out[_nf] = float(out[_nf])
            except (TypeError, ValueError):
                out[_nf] = None

    for k in REQUIRED_TEAM_CONTEXT_FIELDS:
        out.setdefault(k, None)
    return out




def _coalesce_pred_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee pred_spread is populated before write.
    Tries aliases in priority order — first non-null wins.
    Logs a warning if recovery was needed so it is visible in run logs.
    Never silently writes a fully-null pred_spread column.
    """
    if "pred_spread" not in df.columns or df["pred_spread"].isna().all():
        for alias in ["ens_ens_spread", "predicted_spread", "ensemble_spread"]:
            if alias in df.columns and df[alias].notna().any():
                df["pred_spread"] = df[alias]
                log.warning(
                    "[INTEGRITY] pred_spread was null — recovered from '%s' (%d rows)",
                    alias, df["pred_spread"].notna().sum()
                )
                break
        else:
            log.error(
                "[INTEGRITY] pred_spread is null and no alias found. "
                "Aliases checked: ens_ens_spread, predicted_spread, ensemble_spread. "
                "Available columns: %s",
                sorted(df.columns.tolist())
            )

    # Final gate — raise if still null after recovery attempts
    if df["pred_spread"].isna().all():
        raise RuntimeError(
            "pred_spread is fully null after all recovery attempts. "
            "Cannot write predictions file with no spread values."
        )

    null_pct = df["pred_spread"].isna().mean() * 100
    if null_pct > 10:
        log.warning(
            "[INTEGRITY] pred_spread is %.0f%% null after coalesce — "
            "check model output for games with insufficient history",
            null_pct
        )
    return df

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
    elif "game_id" not in out_df.columns and "event_id" in out_df.columns:
        out_df["game_id"] = out_df["event_id"]
        log.info("Normalized prediction output: added game_id from event_id")

    if "predicted_spread" not in out_df.columns and "pred_spread" in out_df.columns:
        out_df["predicted_spread"] = out_df["pred_spread"]
        log.info("Normalized prediction output: added predicted_spread from pred_spread")

    out_df = add_conference_name(out_df)

    out_df = _coalesce_pred_spread(out_df)

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

    if not SEASON_ACTIVE:
        log.info("Pipeline paused — SEASON_ACTIVE is False")
        sys.exit(0)

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
    active_path = Path("data/active_weights.json")
    if active_path.exists():
        weights = json.loads(active_path.read_text())
        for f in dataclasses.fields(config):
            if f.name in weights and not f.name.startswith("_"):
                setattr(config, f.name, type(getattr(config, f.name))(weights[f.name]))
        log.info("Active weights loaded (deployed %s)", weights.get("deployed_at", "unknown"))
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
