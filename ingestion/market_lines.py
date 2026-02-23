"""Capture CBB market lines at opening/pregame/closing checkpoints."""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

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


def fetch_action_network(game_date: date) -> list[dict]:
    date_str = game_date.strftime("%Y%m%d")
    url = (
        "https://api.actionnetwork.com/web/v1/scoreboard/ncaab"
        f"?period=game&bookIds=15,30,76,123&date={date_str}"
    )
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return data.get("games", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("Action Network fetch failed: %s", exc)
        return []


def fetch_pinnacle_lines() -> list[dict]:
    url = "https://guest.api.arcadia.pinnacle.com/0.1/leagues/487/matchups"
    try:
        resp = requests.get(
            url,
            headers={**HEADERS, "X-API-Key": "CmX2KcMrXuFmNg6YFbmTxE0y9CRqvg"},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        payload = resp.json()
        return payload if isinstance(payload, list) else []
    except Exception as exc:  # noqa: BLE001
        log.warning("Pinnacle fetch failed: %s", exc)
        return []


def fetch_draftkings_lines() -> list[dict]:
    url = "https://sportsbook.draftkings.com/api/odds/v1/leagues/ncaab/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json().get("eventGroup", {}).get("events", [])
    except Exception as exc:  # noqa: BLE001
        log.warning("DraftKings fetch failed: %s", exc)
        return []


def parse_action_network_game(game: dict) -> Optional[dict]:
    try:
        teams = game.get("teams", [])
        if len(teams) < 2:
            return None

        home_team = next((t for t in teams if t.get("meta", {}).get("home")), teams[0])
        away_team = next((t for t in teams if not t.get("meta", {}).get("home")), teams[1])
        odds = game.get("odds", [{}])[0] if game.get("odds") else {}

        home_pct_tickets = game.get("home_pct")
        home_pct_money = game.get("home_money_pct")

        return {
            "an_game_id": str(game.get("id", "")),
            "home_team_name": home_team.get("full_name", ""),
            "away_team_name": away_team.get("full_name", ""),
            "game_time_utc": game.get("scheduled_time"),
            "home_spread_open": odds.get("spread_open_home"),
            "home_spread_current": odds.get("spread_current_home"),
            "total_open": odds.get("total_open"),
            "total_current": odds.get("total_current"),
            "home_tickets_pct": home_pct_tickets,
            "away_tickets_pct": 100 - home_pct_tickets if home_pct_tickets is not None else None,
            "home_money_pct": home_pct_money,
            "away_money_pct": 100 - home_pct_money if home_pct_money is not None else None,
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("Parse error on AN game: %s", exc)
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
    prior = existing_rows[existing_rows["event_id"] == event_id].sort_values("captured_at_utc") if not existing_rows.empty else pd.DataFrame()
    if len(prior) > 0 and home_spread is not None:
        last_row = prior.iloc[-1]
        last_spread = last_row.get("home_spread_current")
        last_time = pd.Timestamp(last_row["captured_at_utc"])
        hours_since = (pd.Timestamp.utcnow() - last_time).total_seconds() / 3600
        steam_flag = detect_steam_move(home_spread, last_spread, hours_since)

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
    }


def append_market_rows(new_rows: list[dict], output_path: Path) -> int:
    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        return 0

    df_new["event_id"] = df_new["event_id"].astype(str)
    df_new["capture_hour"] = pd.to_datetime(df_new["captured_at_utc"], utc=True).dt.floor("H")

    if output_path.exists():
        df_existing = pd.read_csv(output_path, dtype={"event_id": str})
        if "capture_hour" not in df_existing.columns:
            df_existing["capture_hour"] = pd.to_datetime(df_existing["captured_at_utc"], utc=True).dt.floor("H").astype(str)
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


def run_capture(mode: str, data_dir: Path) -> None:
    today = date.today()
    log.info("Market capture mode=%s date=%s", mode, today)

    pred_path = data_dir / "predictions_combined_latest.csv"
    predictions = pd.read_csv(pred_path, dtype=str) if pred_path.exists() else pd.DataFrame()

    market_path = data_dir / "market_lines.csv"
    existing = pd.read_csv(market_path, dtype={"event_id": str}) if market_path.exists() else pd.DataFrame()

    log.info("Fetching Action Network...")
    an_games = fetch_action_network(today)
    time.sleep(REQUEST_DELAY)

    log.info("Fetching Pinnacle...")
    pinnacle_games = fetch_pinnacle_lines()
    time.sleep(REQUEST_DELAY)

    log.info("Fetching DraftKings...")
    dk_games = fetch_draftkings_lines()

    new_rows: list[dict] = []
    matched = 0
    unmatched = 0

    for game in an_games:
        parsed = parse_action_network_game(game)
        if not parsed:
            continue

        event_id = match_to_pipeline_event(parsed, predictions)
        if not event_id:
            unmatched += 1
            continue

        matched += 1
        capture_type = {
            "morning": "opening",
            "pregame": "pregame",
            "postgame": "closing",
            "all": "opening",
        }.get(mode, "pregame")

        # TODO: add stronger team-name matching against pinn/dk payload shapes.
        pinn_match = None if not pinnacle_games else None
        dk_match = None if not dk_games else None

        row = build_market_row(event_id, capture_type, parsed, pinn_match, dk_match, existing)
        new_rows.append(row)

    inserted = append_market_rows(new_rows, market_path) if new_rows else 0
    rejected = len(an_games) - matched
    log.info(
        "Market ingest summary: pulled=%s matched=%s inserted=%s rejected=%s",
        len(an_games),
        matched,
        inserted,
        rejected,
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
    args = parser.parse_args()

    if args.mode == "all":
        for mode in ["morning", "pregame", "postgame"]:
            run_capture(mode, DATA_DIR)
            time.sleep(REQUEST_DELAY)
    else:
        run_capture(args.mode, DATA_DIR)


if __name__ == "__main__":
    main()
