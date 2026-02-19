#!/usr/bin/env python3
"""
ESPN CBB Pipeline — Prediction Runner
Bridges team_game_weighted.csv → CBBPredictionModel → predictions_YYYYMMDD.csv

Reads from the same data/ directory the main pipeline writes.
Fetches tomorrow's scheduled games, builds GameData objects (including recursive
opponent_history), runs CBBPredictionModel.predict_game(), and writes outputs.

Usage:
    python espn_prediction_runner.py                    # tomorrow's games
    python espn_prediction_runner.py --date 20250315    # specific date
    python espn_prediction_runner.py --game-type ncaa_r1  # override game type
    python espn_prediction_runner.py --days-back 1      # days back for pipeline fetch

Outputs written to data/:
    predictions_YYYYMMDD.csv    — one row per scheduled game with spread/total/UWS
    predictions_latest.csv      — always overwritten with most recent run (easy CI reference)

Pipeline position:
    Runs AFTER espn_pipeline.py (needs team_game_weighted.csv populated).
    In GitHub Actions: separate job that depends on the update job.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ── Local imports ──────────────────────────────────────────────────────────────
# Prediction model lives alongside this file in the repo root
try:
    from cbb_prediction_model import (
        CBBPredictionModel,
        GameData,
        ModelConfig,
    )
except ImportError as e:
    raise SystemExit(
        f"Cannot import cbb_prediction_model.py — ensure it is in the same directory.\n{e}"
    )

# Tournament metrics (optional — enriches output with UWS when available)
try:
    from espn_tournament import (
        compute_underdog_winner_score,
        build_pretournament_snapshot,
    )
    TOURNAMENT_AVAILABLE = True
except ImportError:
    TOURNAMENT_AVAILABLE = False

# ESPN client for fetching scheduled games (scoreboard for tomorrow)
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

TZ = ZoneInfo("America/Los_Angeles")

# ── File paths (mirrors espn_config.py conventions) ───────────────────────────
DATA_DIR = Path("data")

CSV_WEIGHTED   = DATA_DIR / "team_game_weighted.csv"
CSV_METRICS    = DATA_DIR / "team_game_metrics.csv"
CSV_GAMES      = DATA_DIR / "games.csv"
CSV_LOGS       = DATA_DIR / "team_game_logs.csv"
CSV_SNAPSHOT   = DATA_DIR / "team_pretournament_snapshot.csv"

OUT_PREDICTIONS_LATEST = DATA_DIR / "predictions_latest.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
# Columns from weighted/metrics CSV → GameData.team_box key mapping
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
}

# How many games back to pull per team for opponent_history (recursive context)
OPP_HISTORY_WINDOW = 5
# How many of a team's own recent games to build GameData list from
TEAM_GAMES_WINDOW = 10


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_team_game_data() -> pd.DataFrame:
    """
    Load the richest available team data file.
    Priority: weighted > metrics > logs (use whatever the pipeline produced).
    Ensures game_datetime_utc is parsed and data is sorted chronologically.
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
            # Coerce all numeric columns
            for col in df.columns:
                if col not in ("team_id", "team", "opponent_id", "opponent",
                               "home_away", "conference", "event_id", "game_id",
                               "game_datetime_utc", "game_datetime_pst", "venue",
                               "state", "source", "parse_version", "t_offensive_archetype"):
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            df = df.sort_values(["team_id", "game_datetime_utc"])
            log.info(f"Loaded {len(df)} team-game rows, {df['team_id'].nunique()} teams")
            return df

    raise FileNotFoundError(
        "No team game data found. Run espn_pipeline.py first.\n"
        f"Looked for: {CSV_WEIGHTED}, {CSV_METRICS}, {CSV_LOGS}"
    )


def load_games_schedule() -> pd.DataFrame:
    """Load the games.csv scoreboard data for finding tomorrow's matchups."""
    if not CSV_GAMES.exists():
        log.warning(f"{CSV_GAMES} not found — cannot identify scheduled games")
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
    """Extract box score stats dict from a team-game log row."""
    box = {}
    for csv_col, box_key in BOX_COL_MAP.items():
        val = row.get(csv_col, 0.0)
        box[box_key] = float(val) if pd.notna(val) else 0.0
    return box


def _build_game_data(row: pd.Series, opp_history: List[GameData]) -> GameData:
    """Convert a single team-game row to a GameData object."""
    team_score = int(row.get("points_for", 0) or 0)
    opp_score  = int(row.get("points_against", 0) or 0)
    neutral    = bool(row.get("neutral_site", False))

    # Parse date
    dt = row.get("game_datetime_utc")
    if pd.isna(dt):
        dt = datetime.now(TZ)

    team_box = _row_to_box(row)

    # Build opponent box — stored as opp_* columns if available, else zeros
    opp_box = {}
    for csv_col, box_key in BOX_COL_MAP.items():
        opp_col = f"opp_{csv_col}"
        val = row.get(opp_col, 0.0)
        opp_box[box_key] = float(val) if pd.notna(val) else 0.0

    # If opponent box is all zeros (not stored), populate from mirrored perspective
    # using what opponent allowed (inferred from team_game side)
    # This is acceptable for the baseline; full production would join opponent rows
    if all(v == 0.0 for v in opp_box.values()):
        # Rough mirror: give opponent average values so possessions can be estimated
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
        game_id      = str(row.get("event_id", "")),
        date         = dt if isinstance(dt, datetime) else dt.to_pydatetime(),
        team_name    = str(row.get("team", "")),
        opponent_name= str(row.get("opponent", "")),
        team_score   = team_score,
        opponent_score = opp_score,
        neutral_site = neutral,
        team_box     = team_box,
        opponent_box = opp_box,
        opponent_history = opp_history,
    )


def build_team_game_list(
    team_id: str,
    all_data: pd.DataFrame,
    cutoff_dt: Optional[pd.Timestamp] = None,
    max_games: int = TEAM_GAMES_WINDOW,
) -> List[GameData]:
    """
    Build a list of GameData objects for a team's recent games.
    Includes recursive opponent_history for each game (the secret sauce).

    Parameters
    ----------
    team_id  : ESPN team ID
    all_data : Full team_game_weighted DataFrame (all teams, full season)
    cutoff_dt: Only include games strictly before this timestamp (leak-free)
    max_games: How many recent games to return
    """
    team_rows = all_data[all_data["team_id"].astype(str) == str(team_id)].copy()

    if cutoff_dt is not None:
        team_rows = team_rows[team_rows["game_datetime_utc"] < cutoff_dt]

    team_rows = team_rows.sort_values("game_datetime_utc")

    if team_rows.empty:
        log.warning(f"No game history found for team_id={team_id}")
        return []

    recent_rows = team_rows.tail(max_games)
    game_list   = []

    for _, row in recent_rows.iterrows():
        # Build opponent's recent history (for normalized baseline)
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
    """
    Build the opponent's recent games (for recursive baseline context).
    Deliberately shallow recursion — opponent_history entries have empty histories
    to avoid exponential blowup.
    """
    if not opp_id or opp_id == "nan":
        return []

    opp_rows = all_data[
        (all_data["team_id"].astype(str) == opp_id) &
        (all_data["game_datetime_utc"] < before_dt)
    ].sort_values("game_datetime_utc").tail(OPP_HISTORY_WINDOW)

    history = []
    for _, row in opp_rows.iterrows():
        gd = _build_game_data(row, opp_history=[])  # Shallow — no recursive nesting
        history.append(gd)

    return history


# ═══════════════════════════════════════════════════════════════════════════════
# TOMORROW'S GAMES DISCOVERY
# ═══════════════════════════════════════════════════════════════════════════════

def get_tomorrows_matchups(
    target_date: Optional[str] = None,
) -> List[Dict]:
    """
    Return a list of scheduled (not yet completed) matchups for tomorrow.
    Falls back to reading games.csv if ESPN client isn't available.

    Parameters
    ----------
    target_date : YYYYMMDD string. Defaults to tomorrow in PST.

    Returns list of dicts with keys:
        game_id, home_team_id, away_team_id, home_team, away_team,
        game_datetime_utc, neutral_site, over_under, spread
    """
    if target_date is None:
        tomorrow = datetime.now(TZ) + timedelta(days=1)
        target_date = tomorrow.strftime("%Y%m%d")

    target_dt_str = f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}"
    log.info(f"Looking for games on {target_dt_str}")

    # ── Try live scoreboard fetch first ──────────────────────────────────────
    if ESPN_CLIENT_AVAILABLE:
        try:
            raw = fetch_scoreboard(target_date)
            events = raw.get("events") or []
            matchups = []
            for event in events:
                comps = event.get("competitions", [{}])
                comp  = comps[0] if comps else {}
                completed = comp.get("status", {}).get("type", {}).get("completed", False)
                if completed:
                    continue  # Skip already completed games

                home = next(
                    (c for c in comp.get("competitors", []) if c.get("homeAway") == "home"),
                    {}
                )
                away = next(
                    (c for c in comp.get("competitors", []) if c.get("homeAway") == "away"),
                    {}
                )
                if not home or not away:
                    continue

                odds = (comp.get("odds") or [{}])[0]
                matchups.append({
                    "game_id":           str(event.get("id", "")),
                    "home_team_id":      str(home.get("team", {}).get("id", "")),
                    "away_team_id":      str(away.get("team", {}).get("id", "")),
                    "home_team":         home.get("team", {}).get("displayName", ""),
                    "away_team":         away.get("team", {}).get("displayName", ""),
                    "game_datetime_utc": comp.get("date", ""),
                    "neutral_site":      bool(comp.get("neutralSite", False)),
                    "over_under":        odds.get("overUnder"),
                    "spread":            odds.get("spread"),
                    "home_ml":           odds.get("homeTeamOdds", {}).get("moneyLine"),
                    "away_ml":           odds.get("awayTeamOdds", {}).get("moneyLine"),
                })
            log.info(f"Found {len(matchups)} scheduled games via ESPN scoreboard")
            return matchups
        except Exception as exc:
            log.warning(f"Live scoreboard fetch failed ({exc}), falling back to games.csv")

    # ── Fallback: read games.csv already written by pipeline ─────────────────
    games_df = load_games_schedule()
    if games_df.empty:
        log.error("No game schedule data available")
        return []

    target_mask = (
        games_df["game_datetime_utc"]
        .dt.date
        .astype(str)
        .str.startswith(target_dt_str)
    )
    incomplete = games_df["completed"].astype(str).str.lower().isin(["false", "0", ""])

    tomorrow_games = games_df[target_mask & incomplete].copy()
    if tomorrow_games.empty:
        # Fallback to searching by 'date' column if UTC datetime is empty
        if "date" in games_df.columns:
            tomorrow_games = games_df[
                games_df["date"].astype(str) == target_date
            ].copy()

    matchups = []
    for _, row in tomorrow_games.iterrows():
        matchups.append({
            "game_id":           str(row.get("game_id", "")),
            "home_team_id":      str(row.get("home_team_id", "")),
            "away_team_id":      str(row.get("away_team_id", "")),
            "home_team":         str(row.get("home_team", "")),
            "away_team":         str(row.get("away_team", "")),
            "game_datetime_utc": str(row.get("game_datetime_utc", "")),
            "neutral_site":      str(row.get("neutral_site", "false")).lower() == "true",
            "over_under":        row.get("over_under"),
            "spread":            row.get("spread"),
            "home_ml":           row.get("home_ml"),
            "away_ml":           row.get("away_ml"),
        })

    log.info(f"Found {len(matchups)} scheduled games from games.csv")
    return matchups


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_predictions(
    matchups: List[Dict],
    all_data: pd.DataFrame,
    model: CBBPredictionModel,
    snapshot: Optional[pd.DataFrame],
    game_type: str = "ncaa_r1",
    cutoff_dt: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Run predictions for all provided matchups.

    For each game:
      1. Build home team's recent GameData list (with opponent_history)
      2. Build away team's recent GameData list (with opponent_history)
      3. Run CBBPredictionModel.predict_game()
      4. Optionally enrich with UWS from espn_tournament.py
      5. Compare to opening line (if available)

    Returns a DataFrame with one row per matchup.
    """
    results = []
    skipped = 0

    for matchup in matchups:
        home_id   = matchup["home_team_id"]
        away_id   = matchup["away_team_id"]
        home_name = matchup["home_team"]
        away_name = matchup["away_team"]
        game_id   = matchup["game_id"]
        neutral   = matchup["neutral_site"]

        log.info(f"  Processing: {home_name} vs {away_name} (game_id={game_id})")

        # Build GameData lists
        home_games = build_team_game_list(home_id, all_data, cutoff_dt=cutoff_dt)
        away_games = build_team_game_list(away_id, all_data, cutoff_dt=cutoff_dt)

        if not home_games or not away_games:
            log.warning(f"    Skipping {home_name} vs {away_name} — insufficient history "
                        f"(home={len(home_games)}, away={len(away_games)})")
            skipped += 1
            continue

        # Run core prediction
        try:
            prediction = model.predict_game(
                home_games=home_games,
                away_games=away_games,
                neutral_site=neutral,
            )
        except Exception as exc:
            log.error(f"    Prediction failed for {game_id}: {exc}")
            skipped += 1
            continue

        # ── Line comparison ──────────────────────────────────────────────────
        spread_line = _safe_float(matchup.get("spread"))
        total_line  = _safe_float(matchup.get("over_under"))

        spread_diff = (
            round(prediction["predicted_spread"] - spread_line, 2)
            if spread_line is not None else None
        )
        total_diff = (
            round(prediction["predicted_total"] - total_line, 2)
            if total_line is not None else None
        )

        # ── Spread direction ─────────────────────────────────────────────────
        # Negative predicted spread = model likes home team
        pred_spread = prediction["predicted_spread"]
        if spread_line is not None and spread_diff is not None:
            # Model disagrees with line by > threshold → potential edge
            edge_flag = abs(spread_diff) > 3.0
            spread_pick = (
                f"{home_name} covers" if pred_spread < spread_line
                else f"{away_name} covers"
            )
        else:
            edge_flag   = False
            spread_pick = f"{home_name} -" if pred_spread < 0 else f"{away_name} -"
            spread_pick += f"{abs(pred_spread):.1f}"

        # ── UWS enrichment (if underdog is identifiable) ─────────────────────
        uws_result = {}
        if TOURNAMENT_AVAILABLE and snapshot is not None:
            uws_result = _compute_uws_for_matchup(
                matchup, snapshot, game_type=game_type
            )

        # ── Flat result row ──────────────────────────────────────────────────
        bd = prediction.get("breakdown", {})
        row = {
            "game_id":              game_id,
            "game_datetime_utc":    matchup["game_datetime_utc"],
            "home_team_id":         home_id,
            "away_team_id":         away_id,
            "home_team":            home_name,
            "away_team":            away_name,
            "neutral_site":         neutral,

            # Model outputs
            "pred_spread":          round(pred_spread, 2),
            "pred_total":           round(prediction["predicted_total"], 2),
            "pred_home_score":      round(prediction["predicted_total"] / 2 - pred_spread / 2, 1),
            "pred_away_score":      round(prediction["predicted_total"] / 2 + pred_spread / 2, 1),
            "model_confidence":     round(prediction["confidence"], 3),
            "pace_projected":       round(prediction["pace"], 1),

            # Efficiency
            "home_net_eff":         round(prediction["home_net_eff"], 2),
            "away_net_eff":         round(prediction["away_net_eff"], 2),
            "home_off_eff_vs_exp":  round(prediction["home_off_eff_vs_exp"], 2),
            "away_off_eff_vs_exp":  round(prediction["away_off_eff_vs_exp"], 2),
            "eff_edge":             round(bd.get("eff_edge", 0), 2),
            "composite_edge":       round(bd.get("composite_edge", 0), 2),
            "hca":                  round(bd.get("hca", 0), 2),

            # Four Factors deltas
            "efg_delta":            round(bd.get("efg_delta", 0), 2),
            "tov_delta":            round(bd.get("tov_delta", 0), 2),
            "orb_delta":            round(bd.get("orb_delta", 0), 2),
            "drb_delta":            round(bd.get("drb_delta", 0), 2),
            "ftr_delta":            round(bd.get("ftr_delta", 0), 2),

            # Line comparison
            "spread_line":          spread_line,
            "total_line":           total_line,
            "home_ml":              _safe_float(matchup.get("home_ml")),
            "away_ml":              _safe_float(matchup.get("away_ml")),
            "spread_diff_vs_line":  spread_diff,
            "total_diff_vs_line":   total_diff,
            "spread_pick":          spread_pick,
            "edge_flag":            int(edge_flag),
            "total_direction":      "OVER" if (total_diff or 0) > 2 else "UNDER" if (total_diff or 0) < -2 else "PUSH",

            # Data quality
            "home_games_used":      len(home_games),
            "away_games_used":      len(away_games),

            # Meta
            "game_type":            game_type,
            "predicted_at_utc":     pd.Timestamp.now("UTC").isoformat(),
        }

        # Merge UWS fields
        row.update(uws_result)
        results.append(row)

    log.info(f"Predictions complete: {len(results)} generated, {skipped} skipped")
    return pd.DataFrame(results) if results else pd.DataFrame()


def _compute_uws_for_matchup(
    matchup: Dict,
    snapshot: pd.DataFrame,
    game_type: str,
) -> Dict:
    """
    Identify favorite/underdog from money line and compute UWS.
    Returns a flat dict of uws_* prefixed columns.
    """
    home_ml = _safe_float(matchup.get("home_ml"))
    away_ml = _safe_float(matchup.get("away_ml"))

    if home_ml is None or away_ml is None:
        return {}

    # More negative moneyline = bigger favorite
    if home_ml < away_ml:
        fav_id = matchup["home_team_id"]
        dog_id = matchup["away_team_id"]
    else:
        fav_id = matchup["away_team_id"]
        dog_id = matchup["home_team_id"]

    fav_snap = snapshot.loc[fav_id].to_dict() if fav_id in snapshot.index else {}
    dog_snap = snapshot.loc[dog_id].to_dict() if dog_id in snapshot.index else {}

    if not fav_snap or not dog_snap:
        return {}

    try:
        uws = compute_underdog_winner_score(
            favorite_stats=fav_snap,
            underdog_stats=dog_snap,
            game_type=game_type,
        )
        # Prefix all UWS keys
        return {f"uws_{k}": v for k, v in uws.items()}
    except Exception as exc:
        log.warning(f"UWS computation failed: {exc}")
        return {}


def _safe_float(val) -> Optional[float]:
    """Convert to float, return None on failure."""
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT WRITERS
# ═══════════════════════════════════════════════════════════════════════════════

def write_predictions(df: pd.DataFrame, target_date: str) -> Path:
    """Write predictions to a dated CSV and overwrite predictions_latest.csv."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dated_path = DATA_DIR / f"predictions_{target_date}.csv"
    df.to_csv(dated_path, index=False)
    df.to_csv(OUT_PREDICTIONS_LATEST, index=False)

    log.info(f"Wrote {len(df)} predictions → {dated_path}")
    log.info(f"Updated predictions_latest.csv")
    return dated_path


def print_summary(df: pd.DataFrame) -> None:
    """Print a human-readable matchup table to stdout for CI logs."""
    if df.empty:
        print("No predictions generated.")
        return

    print()
    print("=" * 100)
    print(f"{'MATCHUP':<42} {'PRED SPREAD':>12} {'PRED TOT':>9} {'LINE':>8} {'DIFF':>7} {'CONF':>6} {'EDGE':>5}")
    print("=" * 100)

    for _, row in df.iterrows():
        matchup = f"{row['home_team']} vs {row['away_team']}"[:41]
        spread  = f"{row['pred_spread']:+.1f}"
        total   = f"{row['pred_total']:.1f}"
        line    = f"{row['spread_line']:+.1f}" if pd.notna(row.get('spread_line')) else "  N/A"
        diff    = f"{row['spread_diff_vs_line']:+.1f}" if pd.notna(row.get('spread_diff_vs_line')) else "  N/A"
        conf    = f"{row['model_confidence']:.0%}"
        edge    = "⚡" if row.get('edge_flag') else ""
        print(f"{matchup:<42} {spread:>12} {total:>9} {line:>8} {diff:>7} {conf:>6} {edge:>5}")

    print("=" * 100)

    # Edge alerts
    edges = df[df["edge_flag"] == 1]
    if not edges.empty:
        print(f"\n⚡ EDGE ALERTS ({len(edges)} games with spread diff >3 pts vs line):")
        for _, row in edges.iterrows():
            print(f"   {row['home_team']} vs {row['away_team']}: "
                  f"Model {row['pred_spread']:+.1f} vs Line {row['spread_line']:+.1f} "
                  f"→ {row['spread_pick']}")

    # UWS alerts
    if "uws_uws_total" in df.columns:
        strong = df[df["uws_uws_total"] >= 55]
        if not strong.empty:
            print(f"\n🚨 STRONG UPSET ALERTS ({len(strong)} games UWS ≥ 55):")
            for _, row in strong.iterrows():
                print(f"   {row['home_team']} vs {row['away_team']}: "
                      f"UWS {row['uws_uws_total']:.0f}/70 — "
                      f"{row.get('uws_uws_primary_narrative', '')}")

    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run CBB predictions for tomorrow's games"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Target date YYYYMMDD (default: tomorrow in PST)",
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

    # Resolve target date
    if args.date:
        target_date = args.date
    else:
        target_date = (datetime.now(TZ) + timedelta(days=1)).strftime("%Y%m%d")

    log.info(f"{'='*60}")
    log.info(f"CBB Prediction Runner — Target: {target_date}")
    log.info(f"Game type: {args.game_type} | Decay: {args.decay}")
    log.info(f"{'='*60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    all_data = load_team_game_data()
    snapshot = load_tournament_snapshot()

    # ── Get scheduled matchups ────────────────────────────────────────────────
    matchups = get_tomorrows_matchups(target_date=target_date)
    if not matchups:
        log.warning(f"No scheduled games found for {target_date}. Exiting.")
        sys.exit(0)

    log.info(f"Running predictions for {len(matchups)} games")

    # ── Initialize model ──────────────────────────────────────────────────────
    config = ModelConfig(
        decay_type=args.decay,
        min_games_for_full_confidence=args.min_games,
    )
    model = CBBPredictionModel(config)

    # Cutoff = start of target date (no leakage of future games into history)
    cutoff_dt = pd.Timestamp(
        f"{target_date[:4]}-{target_date[4:6]}-{target_date[6:]}",
        tz="UTC"
    )

    # ── Run predictions ───────────────────────────────────────────────────────
    results_df = run_predictions(
        matchups=matchups,
        all_data=all_data,
        model=model,
        snapshot=snapshot,
        game_type=args.game_type,
        cutoff_dt=cutoff_dt,
    )

    if results_df.empty:
        log.warning("No predictions generated — check team data coverage")
        sys.exit(0)

    # ── Write outputs ─────────────────────────────────────────────────────────
    out_path = write_predictions(results_df, target_date)
    print_summary(results_df)

    log.info(f"Done. Output: {out_path}")
    return out_path


if __name__ == "__main__":
    main()
