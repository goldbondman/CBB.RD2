"""Capture CBB market lines at opening/pregame/closing checkpoints."""

from __future__ import annotations

import argparse
import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from pipeline_csv_utils import normalize_numeric_dtypes

try:
    from espn_gap_fillers import fill_market_row_gaps
except ImportError:
    logging.getLogger(__name__).warning(
        "espn_gap_fillers not found — gap-fill enrichment disabled. "
        "Market lines will capture ESPN data only."
    )

    def fill_market_row_gaps(row: dict) -> dict:  # noqa: E306
        """No-op fallback when espn_gap_fillers is unavailable."""
        return row

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CBB-Pipeline/1.0)",
    "Accept": "application/json",
}
REQUEST_DELAY = 2.0
REQUEST_TIMEOUT = 15

STEAM_MOVE_POINTS = 2.0
STEAM_MOVE_HOURS = 2.0
BOOK_DISAGREE_POINTS = 1.5
PUBLIC_BET_THRESHOLD = 65
RLM_LINE_MOVE_MIN = 0.5

MARKET_LINES_SCHEMA_COLUMNS = [
    "event_id",
    "capture_type",
    "captured_at_utc",
    "pulled_at_utc",
    "verification_status",
    "verification_notes",
    "home_spread_open",
    "home_spread_current",
    "line_movement",
    "total_open",
    "total_current",
    "pinnacle_spread",
    "draftkings_spread",
    "home_tickets_pct",
    "away_tickets_pct",
    "home_money_pct",
    "away_money_pct",
    "steam_flag",
    "rlm_flag",
    "rlm_sharp_side",
    "rlm_note",
    "book_disagreement_flag",
    "book_spread_diff",
    "book_sharp_side",
    "book_note",
    "line_freeze_flag",
    "home_win_prob",
    "away_win_prob",
    "home_ats_wins",
    "home_ats_losses",
    "away_ats_wins",
    "away_ats_losses",
    "home_team_id",
    "away_team_id",
    "home_ml",
    "away_ml",
    "spread",
    "dk_spread",
    "total",
    "over_under",
]


<<<<<<< HEAD
def bootstrap_market_lines_schema(path: str | Path) -> Path:
    """Ensure market_lines.csv exists and has required schema columns."""
    csv_path = Path(path)
    if csv_path.is_dir():
        csv_path = csv_path / "market_lines.csv"

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if csv_path.exists() and csv_path.stat().st_size > 0:
        df = pd.read_csv(csv_path, dtype={"event_id": str}, low_memory=False)
=======
def bootstrap_market_lines_schema(data_dir: Path) -> Path:
    """Ensure market_lines.csv exists and has required schema columns."""
    market_path = data_dir / "market_lines.csv"
    market_path.parent.mkdir(parents=True, exist_ok=True)

    if market_path.exists():
        df = pd.read_csv(market_path, dtype={"event_id": str}, low_memory=False)
>>>>>>> 8cfd96c (fix(ingestion): fix duplicate function syntax error and update deprecated pandas api in market_lines.py)
    else:
        df = pd.DataFrame(columns=MARKET_LINES_SCHEMA_COLUMNS)

    for col in MARKET_LINES_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    ordered = [c for c in MARKET_LINES_SCHEMA_COLUMNS if c in df.columns]
    remainder = [c for c in df.columns if c not in ordered]
    df = df[ordered + remainder]
    df.to_csv(csv_path, index=False)
    log.info("Bootstrapped market lines schema at %s", csv_path)
    return csv_path


def fetch_action_network(game_date: date) -> list[dict]:
    date_str = game_date.strftime("%Y%m%d")
    # bookIds: 15=Pinnacle(Action), 30=Action, 68=DraftKings, 69=FanDuel, 75=BetMGM, 76=PointsBet
    url = (
        "https://api.actionnetwork.com/web/v1/scoreboard/ncaab"
        f"?period=game&bookIds=15,30,68,69,76,123&date={date_str}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("games", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("Action Network fetch failed: %s", exc)
        return []


def fetch_espn_scoreboard(game_date: date) -> list[dict]:
    """Primary market source: ESPN scoreboard odds (event IDs match pipeline IDs)."""
    date_str = game_date.strftime("%Y%m%d")
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball"
        f"/mens-college-basketball/scoreboard?dates={date_str}&limit=200"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("events", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("ESPN scoreboard fetch failed: %s", exc)
        return []


def fetch_pinnacle_lines() -> list[dict]:
    """Fetch Pinnacle lines. League 493 = NCAAB (487 was NBA)."""
    league_id = 493
    base_url = "https://guest.api.arcadia.pinnacle.com/0.1/leagues"
    try:
        # 1. Fetch matchups to get team names
        m_resp = requests.get(f"{base_url}/{league_id}/matchups", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        m_resp.raise_for_status()
        matchups = m_resp.json()

        # 2. Fetch markets to get the actual lines
        l_resp = requests.get(f"{base_url}/{league_id}/markets/straight", headers=HEADERS, timeout=REQUEST_TIMEOUT)
        l_resp.raise_for_status()
        markets = l_resp.json()

        matchup_map = {m["id"]: m for m in matchups if isinstance(m, dict) and "id" in m}

        combined = []
        for market in markets:
            if not isinstance(market, dict):
                continue
            mid = market.get("matchupId")
            if mid in matchup_map:
                # Merge matchup info (participants) into the market object for the indexer
                merged = {**market, **matchup_map[mid]}
                combined.append(merged)

        log.info("Pinnacle: matched %d markets with matchups", len(combined))
        return combined
    except Exception as exc:  # noqa: BLE001
        log.warning("Pinnacle fetch failed: %s", exc)
        return []


def fetch_draftkings_lines() -> list[dict]:
    # NCAAB Event Group ID
    event_group_id = 92483
    url = f"https://sportsbook.draftkings.com/sites/US-SB/api/v1/eventgroups/{event_group_id}?format=json"
    headers = {
        **HEADERS,
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    }
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
        if isinstance(payload, dict):
            return payload.get("eventGroup", {}).get("events", [])
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning("DraftKings fetch failed: %s", exc)
        return []


def normalize_team_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(name or "").lower())
    tokens = [
        token
        for token in cleaned.split()
        if token not in {"st", "state", "university", "college", "of", "the", "at", "and"}
    ]
    return " ".join(tokens)


def _extract_team_names_from_market(book_game: dict) -> tuple[str, str]:
    participants = book_game.get("participants") if isinstance(book_game, dict) else None
    if isinstance(participants, list) and len(participants) >= 2:
        home = next((p for p in participants if str(p.get("alignment", "")).lower() == "home"), participants[0])
        away = next((p for p in participants if str(p.get("alignment", "")).lower() == "away"), participants[1])
        return str(home.get("name", "")), str(away.get("name", ""))

    home = book_game.get("homeTeam") or book_game.get("homeTeamName") or ""
    away = book_game.get("awayTeam") or book_game.get("awayTeamName") or ""

    if not home and not away:
        name = str(book_game.get("name", ""))
        if "@" in name:
            away, home = [part.strip() for part in name.split("@", 1)]
        elif " vs " in name.lower():
            away, home = [part.strip() for part in re.split(r"\s+vs\.?\s+", name, maxsplit=1, flags=re.IGNORECASE)]

    return str(home), str(away)


def _extract_spread_from_market(book_game: dict) -> Optional[float]:
    if not isinstance(book_game, dict):
        return None

    candidate_keys = ["spread", "homeSpread", "pointSpread", "line", "handicap"]
    for key in candidate_keys:
        value = book_game.get(key)
        if isinstance(value, (float, int)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue

    prices = book_game.get("prices")
    if isinstance(prices, list):
        home_price = next((p for p in prices if str(p.get("designation", "")).lower() == "home"), None)
        value = home_price.get("points") if isinstance(home_price, dict) else None
        if isinstance(value, (float, int)):
            return float(value)

    outcomes = book_game.get("outcomes")
    if isinstance(outcomes, list):
        home_outcome = next((o for o in outcomes if str(o.get("label", "")).lower() in {"home", "1", "h"}), None)
        value = home_outcome.get("line") if isinstance(home_outcome, dict) else None
        if isinstance(value, (float, int)):
            return float(value)

    return None


def build_book_index(book_games: list[dict]) -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    for game in book_games:
        home, away = _extract_team_names_from_market(game)
        if not home or not away:
            continue

        key = (normalize_team_name(home), normalize_team_name(away))
        spread = _extract_spread_from_market(game)
        index[key] = {"spread": spread, "home_team_name": home, "away_team_name": away, "raw": game}
    return index


def parse_action_network_game(game: dict) -> Optional[dict]:
    try:
        teams = game.get("teams", [])
        if len(teams) < 2:
            return None

        home_team = next((t for t in teams if t.get("meta", {}).get("home")), None)
        away_team = next((t for t in teams if not t.get("meta", {}).get("home")), None)

        if not home_team or not away_team:
            # Fallback if meta.home is not present
            away_team = teams[0]
            home_team = teams[1]

        all_odds = game.get("odds", [])
        book_ids = [o.get("book_id") for o in all_odds]
        log.debug("AN Game %s book IDs: %s", game.get("id"), book_ids)

        # Preferred books in order: 30(Action), then anything else
        primary_odds = next((o for o in all_odds if o.get("book_id") == 30), all_odds[0] if all_odds else {})
        dk_odds = next((o for o in all_odds if o.get("book_id") == 68), None)
        # Fallback to book 69 (FanDuel) if DraftKings not found
        if not dk_odds:
            dk_odds = next((o for o in all_odds if o.get("book_id") == 69), None)

        pinn_odds = next((o for o in all_odds if o.get("book_id") == 15), None)
        if not pinn_odds:
            pinn_odds = next((o for o in all_odds if o.get("book_id") == 123), None)

        home_pct_tickets = game.get("home_pct") or primary_odds.get("home_pct")
        home_pct_money = game.get("home_money_pct") or primary_odds.get("home_money_pct")

        return {
            "an_game_id": str(game.get("id", "")),
            "home_team_name": home_team.get("full_name", ""),
            "away_team_name": away_team.get("full_name", ""),
            "game_time_utc": game.get("scheduled_time"),
            "home_spread_open": primary_odds.get("spread_open_home"),
            "home_spread_current": primary_odds.get("spread_current_home"),
            "total_open": primary_odds.get("total_open"),
            "total_current": primary_odds.get("total_current"),
            "home_tickets_pct": home_pct_tickets,
            "away_tickets_pct": 100 - home_pct_tickets if home_pct_tickets is not None else None,
            "home_money_pct": home_pct_money,
            "away_money_pct": 100 - home_pct_money if home_pct_money is not None else None,
            # Extra book data
            "dk_spread": dk_odds.get("spread_home") if dk_odds else None,
            "dk_total": dk_odds.get("total") if dk_odds else None,
            "pinn_spread": pinn_odds.get("spread_home") if pinn_odds else None,
            "pinn_total": pinn_odds.get("total") if pinn_odds else None,
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("Parse error on AN game: %s", exc)
        return None


def parse_espn_event(event: dict) -> Optional[dict]:
    """Parse ESPN scoreboard event into market row fields."""
    try:
        comp = event.get("competitions", [{}])[0]
        odds = (comp.get("odds") or [{}])[0]
        teams = comp.get("competitors", [])
        home_team = next((t for t in teams if t.get("homeAway") == "home"), {})
        away_team = next((t for t in teams if t.get("homeAway") == "away"), {})

        spread_raw = str(odds.get("details") or "").strip().upper()
        home_spread_current = None
        if spread_raw in {"PK", "PICK", "PICKEM", "EVEN"}:
            home_spread_current = 0.0
        elif spread_raw:
            match = re.search(r"([+-]?\d+(?:\.\d+)?)", spread_raw)
            if match:
                try:
                    home_spread_current = float(match.group(1))
                except ValueError:
                    home_spread_current = None

        total_current = odds.get("overUnder")
        try:
            total_current = float(total_current) if total_current is not None else None
        except (TypeError, ValueError):
            total_current = None

        return {
            "event_id": str(event.get("id", "")).strip(),
            "home_team_id": str(home_team.get("id", "")).strip(),
            "away_team_id": str(away_team.get("id", "")).strip(),
            "home_team_name": home_team.get("team", {}).get("displayName", ""),
            "away_team_name": away_team.get("team", {}).get("displayName", ""),
            "game_time_utc": comp.get("date"),
            "home_spread_open": None,
            "home_spread_current": home_spread_current,
            "total_open": None,
            "total_current": total_current,
            "home_tickets_pct": None,
            "away_tickets_pct": None,
            "home_money_pct": None,
            "away_money_pct": None,
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("Parse error on ESPN event: %s", exc)
        return None


def match_to_pipeline_event(market_game: dict, pipeline_predictions: pd.DataFrame) -> Optional[str]:
    home_name = str(market_game.get("home_team_name", "")).lower()
    away_name = str(market_game.get("away_team_name", "")).lower()

    if pipeline_predictions.empty:
        return None

    for _, row in pipeline_predictions.iterrows():
        row_home = str(row.get("home_team", "")).lower()
        row_away = str(row.get("away_team", "")).lower()

        home_match = home_name in row_home or row_home in home_name or any(
            w in row_home for w in home_name.split() if len(w) > 4
        )
        away_match = away_name in row_away or row_away in away_name or any(
            w in row_away for w in away_name.split() if len(w) > 4
        )

        if home_match and away_match:
            event_col = "event_id" if "event_id" in row.index else "game_id"
            return str(row.get(event_col, ""))

    return None


def detect_steam_move(current_line: float, previous_line: float, hours_elapsed: float) -> bool:
    if current_line is None or previous_line is None:
        return False
    movement = abs(current_line - previous_line)
    return movement >= STEAM_MOVE_POINTS and hours_elapsed <= STEAM_MOVE_HOURS


def detect_reverse_line_movement(home_tickets_pct: float, home_spread_open: float, home_spread_current: float) -> dict:
    if any(v is None for v in [home_tickets_pct, home_spread_open, home_spread_current]):
        return {"rlm_flag": False}

    public_on_home = home_tickets_pct >= PUBLIC_BET_THRESHOLD
    public_on_away = home_tickets_pct <= (100 - PUBLIC_BET_THRESHOLD)
    line_moved_away = home_spread_current < home_spread_open - RLM_LINE_MOVE_MIN
    line_moved_home = home_spread_current > home_spread_open + RLM_LINE_MOVE_MIN

    rlm = (public_on_home and line_moved_away) or (public_on_away and line_moved_home)
    if not rlm:
        return {"rlm_flag": False}

    sharp_side = "away" if public_on_home and line_moved_away else "home"
    return {
        "rlm_flag": True,
        "rlm_sharp_side": sharp_side,
        "rlm_public_pct": home_tickets_pct,
        "rlm_line_move": round(home_spread_current - home_spread_open, 1),
        "rlm_note": (
            f"Public {home_tickets_pct:.0f}% on {'home' if public_on_home else 'away'} but "
            f"line moved {abs(home_spread_current - home_spread_open):.1f}pt toward {sharp_side} "
            f"— SHARP MONEY on {sharp_side.upper()}"
        ),
    }


def detect_book_disagreement(pinnacle_spread: Optional[float], draftkings_spread: Optional[float]) -> dict:
    if pinnacle_spread is None or draftkings_spread is None:
        return {"book_disagreement_flag": False}

    diff = abs(pinnacle_spread - draftkings_spread)
    if diff < BOOK_DISAGREE_POINTS:
        return {"book_disagreement_flag": False, "book_spread_diff": round(diff, 2)}

    sharp_side = "home" if pinnacle_spread > draftkings_spread else "away"
    return {
        "book_disagreement_flag": True,
        "book_spread_diff": round(diff, 2),
        "pinnacle_spread": pinnacle_spread,
        "draftkings_spread": draftkings_spread,
        "sharp_side": sharp_side,
        "note": (
            f"Pinnacle {pinnacle_spread:+.1f} vs DraftKings {draftkings_spread:+.1f} "
            f"(diff {diff:.1f}pts) — sharp money on {sharp_side.upper()}"
        ),
    }


def build_market_row(
    event_id: str,
    capture_type: str,
    market_data: dict,
    pinnacle_data: Optional[dict],
    dk_data: Optional[dict],
    existing_rows: pd.DataFrame,
) -> dict:
    now = datetime.now(timezone.utc).isoformat()

    home_spread = market_data.get("home_spread_current")
    spread_open = market_data.get("home_spread_open")
    home_tickets = market_data.get("home_tickets_pct")

    steam_flag = False
    if not existing_rows.empty and "event_id" in existing_rows.columns:
        prior = existing_rows[existing_rows["event_id"] == event_id].sort_values("captured_at_utc")
    else:
        prior = pd.DataFrame()

    if prior.empty:
        steam_flag = False
    elif home_spread is not None:
        last_row = prior.iloc[-1]
        last_spread = last_row.get("home_spread_current")
        last_time = pd.Timestamp(last_row["captured_at_utc"])
        hours_since = (pd.Timestamp.now("UTC") - last_time).total_seconds() / 3600
        try:
            move = abs(float(home_spread) - float(last_spread))
            steam_flag = (
                move >= STEAM_MOVE_POINTS and
                hours_since <= STEAM_MOVE_HOURS and
                float(home_spread) != float(last_spread)
            )
        except (TypeError, ValueError):
            steam_flag = False

    rlm = detect_reverse_line_movement(home_tickets, spread_open, home_spread)

    pinn_spread = pinnacle_data.get("spread") if pinnacle_data else None
    dk_spread = dk_data.get("spread") if dk_data else None
    book_dis = detect_book_disagreement(pinn_spread, dk_spread)

    line_movement = None
    if home_spread is not None and spread_open is not None:
        line_movement = round(float(home_spread) - float(spread_open), 2)

    return {
        "event_id": event_id,
        "capture_type": capture_type,
        "captured_at_utc": now,
        "pulled_at_utc": now,
        "verification_status": "verified",
        "verification_notes": "matched_espn_event",
        "home_spread_open": spread_open,
        "home_spread_current": home_spread,
        "line_movement": line_movement,
        "total_open": market_data.get("total_open"),
        "total_current": market_data.get("total_current"),
        "pinnacle_spread": pinn_spread,
        "draftkings_spread": dk_spread,
        "home_tickets_pct": home_tickets,
        "away_tickets_pct": market_data.get("away_tickets_pct"),
        "home_money_pct": market_data.get("home_money_pct"),
        "away_money_pct": market_data.get("away_money_pct"),
        "steam_flag": steam_flag,
        "rlm_flag": rlm.get("rlm_flag", False),
        "rlm_sharp_side": rlm.get("rlm_sharp_side"),
        "rlm_note": rlm.get("rlm_note"),
        "book_disagreement_flag": book_dis.get("book_disagreement_flag", False),
        "book_spread_diff": book_dis.get("book_spread_diff"),
        "book_sharp_side": book_dis.get("sharp_side"),
        "book_note": book_dis.get("note"),
        "line_freeze_flag": (
            abs(line_movement or 0) < 0.5 and home_tickets is not None and abs((home_tickets or 50) - 50) > 15
        ),
        # Gap-fill targets
        "home_win_prob": market_data.get("home_win_prob"),
        "away_win_prob": market_data.get("away_win_prob"),
        "home_ats_wins": market_data.get("home_ats_wins"),
        "home_ats_losses": market_data.get("home_ats_losses"),
        "away_ats_wins": market_data.get("away_ats_wins"),
        "away_ats_losses": market_data.get("away_ats_losses"),
    }


def append_market_rows(new_rows: list[dict], output_path: Path) -> int:
    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        return 0

    df_new["event_id"] = df_new["event_id"].astype(str)
    if "pulled_at_utc" not in df_new.columns:
        df_new["pulled_at_utc"] = df_new.get("captured_at_utc")
    if "verification_status" not in df_new.columns:
        df_new["verification_status"] = "verified"
    if "verification_notes" not in df_new.columns:
        df_new["verification_notes"] = "matched_espn_event"
    df_new["capture_hour"] = pd.to_datetime(df_new["captured_at_utc"], utc=True).dt.floor("h")

    if output_path.exists():
        df_existing = pd.read_csv(output_path, dtype={"event_id": str})
        df_existing = normalize_numeric_dtypes(df_existing)
        for col in MARKET_LINES_SCHEMA_COLUMNS:
            if col not in df_existing.columns:
                df_existing[col] = pd.NA
        if "capture_hour" not in df_existing.columns:
            df_existing["capture_hour"] = pd.to_datetime(df_existing["captured_at_utc"], utc=True).dt.floor("h").astype(str)
        df_existing["capture_hour"] = df_existing["capture_hour"].astype(str)
        df_new["capture_hour"] = df_new["capture_hour"].astype(str)

        dedup_key = df_existing[["event_id", "capture_type", "capture_hour"]].apply(tuple, axis=1)
        new_key = df_new[["event_id", "capture_type", "capture_hour"]].apply(tuple, axis=1)
        df_new = df_new[~new_key.isin(dedup_key)]

        if df_new.empty:
            log.info("No new market rows to append")
            return 0

        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_new

    inserted = len(df_new)
    df_out.drop(columns=["capture_hour"], errors="ignore", inplace=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_path, index=False)
    log.info("Appended %s market rows (%s total in market_lines.csv)", inserted, len(df_out))
    return inserted


def run_capture(mode: str, data_dir: Path, override_date: Optional[date] = None) -> None:
    today = override_date or date.today()
    log.info("Market capture mode=%s date=%s", mode, today)

<<<<<<< HEAD
    market_path = bootstrap_market_lines_schema(data_dir / "market_lines.csv")
=======
    market_path = bootstrap_market_lines_schema(data_dir)
>>>>>>> 8cfd96c (fix(ingestion): fix duplicate function syntax error and update deprecated pandas api in market_lines.py)
    existing = pd.read_csv(market_path, dtype={"event_id": str}) if market_path.exists() else pd.DataFrame()
    existing = normalize_numeric_dtypes(existing)

    log.info("Fetching ESPN scoreboard...")
    espn_events = fetch_espn_scoreboard(today)
    time.sleep(REQUEST_DELAY)

    log.info("Fetching Action Network...")
    an_games = fetch_action_network(today)
    time.sleep(REQUEST_DELAY)

    log.info("Fetching Pinnacle...")
    pinnacle_games = fetch_pinnacle_lines()
    time.sleep(REQUEST_DELAY)

    log.info("Fetching DraftKings...")
    dk_games = fetch_draftkings_lines()

    pinnacle_by_team = build_book_index(pinnacle_games)
    dk_by_team = build_book_index(dk_games)

    action_by_team: dict[tuple[str, str], dict] = {}
    for game in an_games:
        parsed = parse_action_network_game(game)
        if not parsed:
            continue
        key = (
            normalize_team_name(parsed.get("home_team_name", "")),
            normalize_team_name(parsed.get("away_team_name", "")),
        )
        action_by_team[key] = parsed

    new_rows: list[dict] = []
    matched = 0
    an_matched = 0
    unmatched = 0

    for event in espn_events:
        parsed = parse_espn_event(event)
        if not parsed or not parsed.get("event_id"):
            continue

        event_id = str(parsed.get("event_id", "")).strip()
        if not event_id:
            unmatched += 1
            continue

        espn_key = (
            normalize_team_name(parsed.get("home_team_name", "")),
            normalize_team_name(parsed.get("away_team_name", "")),
        )
        an_enrichment = action_by_team.get(espn_key) or action_by_team.get((espn_key[1], espn_key[0]))
        if an_enrichment:
            an_matched += 1
            parsed["home_tickets_pct"] = an_enrichment.get("home_tickets_pct")
            parsed["away_tickets_pct"] = an_enrichment.get("away_tickets_pct")
            parsed["home_money_pct"] = an_enrichment.get("home_money_pct")
            parsed["away_money_pct"] = an_enrichment.get("away_money_pct")
            if parsed.get("home_spread_open") is None:
                parsed["home_spread_open"] = an_enrichment.get("home_spread_open")
            if parsed.get("total_open") is None:
                parsed["total_open"] = an_enrichment.get("total_open")

        matched += 1
        capture_type = {
            "morning": "opening",
            "pregame": "pregame",
            "postgame": "closing",
            "all": "opening",
        }.get(mode, "pregame")

        team_key = (
            normalize_team_name(str(parsed.get("home_team_name", ""))),
            normalize_team_name(str(parsed.get("away_team_name", ""))),
        )
        pinn_match = pinnacle_by_team.get(team_key) or pinnacle_by_team.get((team_key[1], team_key[0]))
        dk_match = dk_by_team.get(team_key) or dk_by_team.get((team_key[1], team_key[0]))

        # Fallback to Action Network's version of these books if direct fetch failed
        if an_enrichment:
            if not pinn_match and an_enrichment.get("pinn_spread") is not None:
                log.debug("Using Pinnacle fallback for %s: %s", event_id, an_enrichment["pinn_spread"])
                pinn_match = {"spread": an_enrichment["pinn_spread"], "total": an_enrichment.get("pinn_total")}
            if not dk_match and an_enrichment.get("dk_spread") is not None:
                log.debug("Using DraftKings fallback for %s: %s", event_id, an_enrichment["dk_spread"])
                dk_match = {"spread": an_enrichment["dk_spread"], "total": an_enrichment.get("dk_total")}

        row = build_market_row(event_id, capture_type, parsed, pinn_match, dk_match, existing)
        row["home_team_id"] = parsed.get("home_team_id")
        row["away_team_id"] = parsed.get("away_team_id")

        # Attempt unofficial ESPN endpoints only for still-missing gap fields.
        row = fill_market_row_gaps(row)
        new_rows.append(row)

    inserted = append_market_rows(new_rows, market_path) if new_rows else 0
    rejected = max(len(espn_events) - matched, 0)

    if not market_path.exists():
        raise RuntimeError(f"Market lines write failed: file missing at {market_path}")
    written_df = pd.read_csv(market_path, dtype={"event_id": str}, low_memory=False)
    required_cols = ["event_id", "pulled_at_utc", "verification_status", "verification_notes"]
    missing_cols = [c for c in required_cols if c not in written_df.columns]
    if missing_cols:
        raise RuntimeError(f"Market lines write failed: missing columns {missing_cols} in {market_path}")
    log.info(
        "Market post-write assertion passed: pulled=%s inserted=%s rejected=%s",
        len(espn_events),
        inserted,
        rejected,
    )
    rows_with_pinnacle = sum(1 for row in new_rows if row.get("pinnacle_spread") is not None)
    rows_with_dk = sum(1 for row in new_rows if row.get("draftkings_spread") is not None)
    log.info(
        "Market ingest summary: pulled=%s matched=%s an_matched=%s inserted=%s rejected=%s",
        len(espn_events),
        matched,
        an_matched,
        inserted,
        rejected,
    )
    log.info(
        "Market integrity: pinnacle_spread_populated=%s/%s draftkings_spread_populated=%s/%s",
        rows_with_pinnacle,
        len(new_rows),
        rows_with_dk,
        len(new_rows),
    )
    rows_with_win_prob = sum(1 for row in new_rows if row.get("home_win_prob") is not None)
    rows_with_ats = sum(1 for row in new_rows if row.get("home_ats_wins") is not None or row.get("away_ats_wins") is not None)
    log.info(
        "Gap-fill integrity: home_win_prob_populated=%s/%s ats_rows_populated=%s/%s",
        rows_with_win_prob,
        len(new_rows),
        rows_with_ats,
        len(new_rows),
    )

    steam_games = [r for r in new_rows if r.get("steam_flag")]
    if steam_games:
        log.warning("🔥 STEAM DETECTED on %s games: %s", len(steam_games), ", ".join(str(r["event_id"]) for r in steam_games))

    rlm_games = [r for r in new_rows if r.get("rlm_flag")]
    for row in rlm_games:
        log.warning("↔️ REVERSE LINE MOVEMENT: event %s — %s", row["event_id"], row.get("rlm_note"))


def main() -> None:
    try:
        from espn_config import DATA_DIR
    except ImportError:
        DATA_DIR = Path("data")

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["morning", "pregame", "postgame", "all"], default="pregame")
    parser.add_argument("--backfill-days", type=int, default=0)
    args = parser.parse_args()

    bootstrap_market_lines_schema(DATA_DIR)
    if args.backfill_days > 0:
        for d in range(args.backfill_days, -1, -1):
            target = date.today() - timedelta(days=d)
            log.info("Backfill date: %s", target)
            if args.mode == "all":
                for mode in ["morning", "pregame", "postgame"]:
                    run_capture(mode, DATA_DIR, override_date=target)
                    time.sleep(REQUEST_DELAY)
            else:
                run_capture(args.mode, DATA_DIR, override_date=target)
    elif args.mode == "all":
        for mode in ["morning", "pregame", "postgame"]:
            run_capture(mode, DATA_DIR)
            time.sleep(REQUEST_DELAY)
    else:
        run_capture(args.mode, DATA_DIR)


if __name__ == "__main__":
    main()
