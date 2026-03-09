"""Capture CBB market lines at opening/pregame/closing checkpoints."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from pipeline.market_canonical import write_market_canonical_outputs
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
HTTP_RETRIES = 2
HTTP_RETRY_BACKOFF_SECONDS = 0.75

DEFAULT_PIPELINE_TIMEZONE = os.getenv("PIPELINE_TZ", "America/Los_Angeles")
DEFAULT_ESPN_GROUPS = os.getenv("ESPN_SCOREBOARD_GROUPS", "50")
MIN_ENUM_GAMES_FOR_GAP_GUARD = 10
MAX_ENUM_TO_OUTPUT_DROP_RATIO = 0.40

STEAM_MOVE_POINTS = 2.0
STEAM_MOVE_HOURS = 2.0
BOOK_DISAGREE_POINTS = 1.5
PUBLIC_BET_THRESHOLD = 65
RLM_LINE_MOVE_MIN = 0.5

MARKET_LINES_SCHEMA_COLUMNS = [
    "game_id",
    "event_id",
    "home_team_name",
    "away_team_name",
    "home_team_id",
    "away_team_id",
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
    "home_ml",
    "away_ml",
    "spread",
    "dk_spread",
    "total",
    "over_under",
    "book",
    "market_type",
    "source",
]

SNAPSHOT_GROUP_KEYS = ["game_id", "book", "market_type"]


def _safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str) and value.strip() == "":
        return None
    try:
        casted = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(casted):
        return None
    return casted


def _pipeline_now(tz_name: str = DEFAULT_PIPELINE_TIMEZONE) -> datetime:
    try:
        return datetime.now(ZoneInfo(tz_name))
    except Exception:  # noqa: BLE001
        log.warning("Invalid timezone %r; defaulting to UTC", tz_name)
        return datetime.now(timezone.utc)


def _pipeline_today(tz_name: str = DEFAULT_PIPELINE_TIMEZONE) -> date:
    return _pipeline_now(tz_name).date()


def _window_bounds_utc(window_date: date, tz_name: str) -> tuple[datetime, datetime]:
    local_tz = ZoneInfo(tz_name)
    start_local = datetime.combine(window_date, datetime.min.time(), tzinfo=local_tz)
    end_local = start_local + timedelta(days=1)
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)


def _http_get_json(
    url: str,
    *,
    params: Optional[dict[str, object]] = None,
    headers: Optional[dict[str, str]] = None,
    timeout: int = REQUEST_TIMEOUT,
    retries: int = HTTP_RETRIES,
    source: str = "http",
) -> Optional[object]:
    req_headers = headers or HEADERS
    for attempt in range(1, retries + 2):
        try:
            resp = requests.get(url, params=params, headers=req_headers, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status == 403:
                log.warning(
                    "%s fetch forbidden (403): url=%s params=%s. Continuing without this source.",
                    source,
                    url,
                    params,
                )
                return None
            retryable_statuses = {408, 409, 425, 429, 500, 502, 503, 504}
            if status not in retryable_statuses:
                log.warning(
                    "%s fetch failed (non-retryable HTTP %s): url=%s params=%s err=%s",
                    source,
                    status,
                    url,
                    params,
                    exc,
                )
                return None
            if attempt > retries:
                log.warning(
                    "%s fetch failed after %s attempts: url=%s params=%s err=%s",
                    source,
                    attempt,
                    url,
                    params,
                    exc,
                )
                return None
            sleep_s = HTTP_RETRY_BACKOFF_SECONDS * attempt
            log.warning(
                "%s fetch retry %s/%s: url=%s params=%s sleep=%.2fs err=%s",
                source,
                attempt,
                retries + 1,
                url,
                params,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)
        except Exception as exc:  # noqa: BLE001
            if attempt > retries:
                log.warning(
                    "%s fetch failed after %s attempts: url=%s params=%s err=%s",
                    source,
                    attempt,
                    url,
                    params,
                    exc,
                )
                return None
            sleep_s = HTTP_RETRY_BACKOFF_SECONDS * attempt
            log.warning(
                "%s fetch retry %s/%s: url=%s params=%s sleep=%.2fs err=%s",
                source,
                attempt,
                retries + 1,
                url,
                params,
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)
    return None


def bootstrap_market_lines_schema(path: str | Path) -> Path:
    """Ensure market_lines.csv exists and has required schema columns."""
    market_path = Path(path)
    if market_path.is_dir() or (not market_path.suffix and not market_path.exists()):
        market_path = market_path / "market_lines.csv"

    market_path.parent.mkdir(parents=True, exist_ok=True)

    if market_path.exists() and market_path.stat().st_size > 0:
        df = pd.read_csv(market_path, dtype={"event_id": str}, low_memory=False)
    else:
        df = pd.DataFrame(columns=MARKET_LINES_SCHEMA_COLUMNS)

    for col in MARKET_LINES_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    ordered = [c for c in MARKET_LINES_SCHEMA_COLUMNS if c in df.columns]
    remainder = [c for c in df.columns if c not in ordered]
    df = df[ordered + remainder]
    df.to_csv(market_path, index=False)
    log.info("Bootstrapped market lines schema at %s", market_path)
    return market_path


def _find_snapshot_ts_column(df: pd.DataFrame) -> str:
    for candidate in ["captured_at_utc", "pulled_at_utc"]:
        if candidate in df.columns:
            return candidate
    raise RuntimeError("Market snapshots missing timestamp columns: expected captured_at_utc or pulled_at_utc")


def regenerate_market_views(data_dir: Path) -> tuple[int, int]:
    """Build latest/closing line views from append-only snapshots."""
    snapshots_path = data_dir / "market_lines_snapshots.csv"
    latest_path = data_dir / "market_lines_latest.csv"
    closing_path = data_dir / "market_lines_closing.csv"
    games_path = data_dir / "games.csv"

    if not snapshots_path.exists() and (data_dir / "market_lines.csv").exists():
        snapshots_path.write_text((data_dir / "market_lines.csv").read_text())

    bootstrap_market_lines_schema(snapshots_path)
    snapshots = pd.read_csv(snapshots_path, dtype=str, low_memory=False)
    if snapshots.empty and (data_dir / "market_lines.csv").exists():
        snapshots = pd.read_csv(data_dir / "market_lines.csv", dtype=str, low_memory=False)
        if not snapshots.empty:
            snapshots.to_csv(snapshots_path, index=False)

    if snapshots.empty:
        pd.DataFrame(columns=MARKET_LINES_SCHEMA_COLUMNS).to_csv(latest_path, index=False)
        pd.DataFrame(columns=MARKET_LINES_SCHEMA_COLUMNS).to_csv(closing_path, index=False)
        log.info("Market view build skipped: no snapshots present")
        return 0, 0

    for col in MARKET_LINES_SCHEMA_COLUMNS:
        if col not in snapshots.columns:
            snapshots[col] = pd.NA

    ts_col = _find_snapshot_ts_column(snapshots)
    snapshots["captured_at_utc"] = pd.to_datetime(snapshots[ts_col], utc=True, errors="coerce")
    snapshots = snapshots.dropna(subset=["captured_at_utc"]).copy()
    snapshots["book"] = snapshots.get("book", pd.Series(dtype=str)).fillna("consensus")
    snapshots["market_type"] = snapshots.get("market_type", pd.Series(dtype=str)).fillna("spread")
    snapshots["source"] = snapshots.get("source", pd.Series(dtype=str)).fillna("ESPN")
    if "game_id" not in snapshots.columns:
        snapshots["game_id"] = snapshots.get("event_id")
    snapshots["game_id"] = snapshots["game_id"].fillna(snapshots.get("event_id")).astype(str)

    latest = (
        snapshots.sort_values("captured_at_utc")
        .groupby(SNAPSHOT_GROUP_KEYS, as_index=False)
        .tail(1)
        .sort_values(SNAPSHOT_GROUP_KEYS)
    )

    games = pd.read_csv(games_path, dtype=str, low_memory=False) if games_path.exists() else pd.DataFrame()
    if not games.empty:
        games["game_id"] = games.get("game_id", pd.Series(dtype=str)).astype(str)
        games["game_datetime_utc"] = pd.to_datetime(games.get("game_datetime_utc"), utc=True, errors="coerce")
        games["final_score_ts"] = pd.to_datetime(games.get("final_score_timestamp_utc"), utc=True, errors="coerce")
        games["completed_flag"] = games.get("completed", pd.Series(dtype=str)).astype(str).str.lower().isin({"1", "true", "t", "yes", "final"})
        games_min = games[["game_id", "game_datetime_utc", "final_score_ts", "completed_flag"]].drop_duplicates("game_id")
    else:
        games_min = pd.DataFrame(columns=["game_id", "game_datetime_utc", "final_score_ts", "completed_flag"])

    merged = snapshots.merge(games_min, on="game_id", how="left")

    def _pick_closing(group: pd.DataFrame) -> pd.Series:
        g = group.sort_values("captured_at_utc")
        start_ts = g["game_datetime_utc"].iloc[0] if "game_datetime_utc" in g.columns else pd.NaT
        if pd.notna(start_ts):
            before_start = g[g["captured_at_utc"] < start_ts]
            if not before_start.empty:
                return before_start.iloc[-1]

        completed = bool(g["completed_flag"].iloc[0]) if "completed_flag" in g.columns else False
        if completed:
            final_ts = g["final_score_ts"].iloc[0] if "final_score_ts" in g.columns else pd.NaT
            if pd.notna(final_ts):
                before_final = g[g["captured_at_utc"] < final_ts]
                if not before_final.empty:
                    return before_final.iloc[-1]
        return g.iloc[-1]

    closing = merged.groupby(SNAPSHOT_GROUP_KEYS, group_keys=False).apply(_pick_closing).reset_index(drop=True)
    latest["captured_at_utc"] = latest["captured_at_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    closing["captured_at_utc"] = closing["captured_at_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    latest.to_csv(latest_path, index=False)
    closing.to_csv(closing_path, index=False)
    log.info("Regenerated market views from snapshots: latest=%s closing=%s", len(latest), len(closing))
    return len(latest), len(closing)


def fetch_action_network(game_date: date) -> list[dict]:
    date_str = game_date.strftime("%Y%m%d")
    # bookIds: 15=Pinnacle(Action), 30=Action, 68=DraftKings, 69=FanDuel, 75=BetMGM, 76=PointsBet
    url = (
        "https://api.actionnetwork.com/web/v1/scoreboard/ncaab"
        f"?period=game&bookIds=15,30,68,69,76,123&date={date_str}"
    )
    data = _http_get_json(url, source="Action Network")
    if not data:
        return []
    return data.get("games", [])


def fetch_espn_scoreboard(game_date: date, groups: str = DEFAULT_ESPN_GROUPS) -> list[dict]:
    """Primary market source: ESPN scoreboard odds (event IDs match pipeline IDs)."""
    date_str = game_date.strftime("%Y%m%d")
    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    params: dict[str, object] = {"dates": date_str, "limit": 1000}
    groups_clean = str(groups or "").strip()
    if groups_clean:
        params["groups"] = groups_clean
    log.info("Fetching ESPN scoreboard endpoint=%s params=%s", url, params)
    payload = _http_get_json(url, params=params, source="ESPN scoreboard")
    if not payload:
        return []
    events = payload.get("events", [])
    log.info("ESPN scoreboard returned %s events for date=%s groups=%s", len(events), date_str, groups_clean or "<default>")
    return events


def fetch_pinnacle_lines() -> list[dict]:
    """Fetch Pinnacle lines. League 493 = NCAAB (487 was NBA)."""
    league_id = 493
    base_url = "https://guest.api.arcadia.pinnacle.com/0.1/leagues"
    try:
        # 1. Fetch matchups to get team names
        m_payload = _http_get_json(f"{base_url}/{league_id}/matchups", source="Pinnacle matchups")
        matchups = m_payload if isinstance(m_payload, list) else []

        # 2. Fetch markets to get the actual lines
        l_payload = _http_get_json(f"{base_url}/{league_id}/markets/straight", source="Pinnacle markets")
        markets = l_payload if isinstance(l_payload, list) else []

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
        payload = _http_get_json(url, headers=headers, source="DraftKings eventgroup")
        if isinstance(payload, dict):
            return payload.get("eventGroup", {}).get("events", [])
        return []
    except Exception as exc:  # noqa: BLE001
        log.warning("DraftKings fetch failed: %s", exc)
        return []


def normalize_team_name(name: str) -> str:
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(name or "").lower())
    
    # Safely handle 'st' to 'state' so distinct teams don't collapse identically
    cleaned = re.sub(r"\bst\b", "state", cleaned)
    
    tokens = [
        token
        for token in cleaned.split()
        if token not in {"university", "college", "of", "the", "at", "and"}
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
            away_team = teams[0] if len(teams) > 0 else {}
            home_team = teams[1] if len(teams) > 1 else {}
        
        if not isinstance(home_team, dict):
            home_team = {}
        if not isinstance(away_team, dict):
            away_team = {}

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
        all_odds = comp.get("odds") or []
        odds = all_odds[0] if all_odds else {}
        teams = comp.get("competitors", [])
        home_team = next((t for t in teams if t.get("homeAway") == "home"), {})
        away_team = next((t for t in teams if t.get("homeAway") == "away"), {})

        # Scan the full odds array for a DraftKings provider entry.
        # ESPN returns odds from multiple sportsbooks; DK is identified by provider.name.
        # This provides a free fallback when the DraftKings direct API is unavailable.
        dk_odds_entry: dict = {}
        for _o in all_odds:
            _pname = str((_o.get("provider") or {}).get("name", "") or "").lower()
            if "draftkings" in _pname:
                dk_odds_entry = _o
                break

        # Helpers used for both the primary odds entry and the DK provider entry.
        def _norm_label(text: str) -> str:
            return re.sub(r"[^A-Z0-9]+", " ", str(text or "").upper()).strip()

        def _matches(token: str, candidates: set[str]) -> bool:
            token_clean = token.strip()
            if not token_clean:
                return False
            for cand in candidates:
                cand_clean = cand.strip()
                if not cand_clean:
                    continue
                if token_clean == cand_clean or token_clean in cand_clean or cand_clean in token_clean:
                    return True
            return False

        home_tokens = {
            _norm_label(home_team.get("team", {}).get("abbreviation", "")),
            _norm_label(home_team.get("team", {}).get("shortDisplayName", "")),
            _norm_label(home_team.get("team", {}).get("displayName", "")),
        }
        away_tokens = {
            _norm_label(away_team.get("team", {}).get("abbreviation", "")),
            _norm_label(away_team.get("team", {}).get("shortDisplayName", "")),
            _norm_label(away_team.get("team", {}).get("displayName", "")),
        }

        def _parse_details_spread(details_raw: str) -> Optional[float]:
            """Parse an ESPN odds 'details' string (e.g. 'DUKE -7.5') to a home spread."""
            raw = str(details_raw or "").strip().upper()
            if raw in {"PK", "PICK", "PICKEM", "EVEN"}:
                return 0.0
            if not raw:
                return None
            m = re.search(r"([+-]?\d+(?:\.\d+)?)", raw)
            if not m:
                return None
            fav_label = re.sub(r"([+-]?\d.*)$", "", raw).strip()
            try:
                pts = abs(float(m.group(1)))
                if _matches(_norm_label(fav_label), home_tokens):
                    return -pts
                if _matches(_norm_label(fav_label), away_tokens):
                    return pts
                return float(m.group(1))
            except ValueError:
                return None

        spread_raw = str(odds.get("details") or "").strip().upper()
        home_spread_current = _parse_details_spread(spread_raw)

        total_current = odds.get("overUnder")
        try:
            total_current = float(total_current) if total_current is not None else None
        except (TypeError, ValueError):
            total_current = None

        # Extract DraftKings spread/total from the ESPN-embedded provider entry.
        dk_spread: Optional[float] = None
        dk_total: Optional[float] = None
        if dk_odds_entry:
            dk_spread = _parse_details_spread(str(dk_odds_entry.get("details") or ""))
            try:
                dk_total = float(dk_odds_entry.get("overUnder")) if dk_odds_entry.get("overUnder") is not None else None
            except (TypeError, ValueError):
                dk_total = None

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
            # DraftKings line extracted from ESPN's embedded provider odds array.
            # Used as fallback when the DraftKings direct API is unavailable.
            "dk_spread": dk_spread,
            "dk_total": dk_total,
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
    current_f = _safe_float(current_line)
    previous_f = _safe_float(previous_line)
    hours_f = _safe_float(hours_elapsed)
    if current_f is None or previous_f is None or hours_f is None:
        return False
    movement = abs(current_f - previous_f)
    return movement >= STEAM_MOVE_POINTS and hours_f <= STEAM_MOVE_HOURS


def detect_reverse_line_movement(home_tickets_pct: float, home_spread_open: float, home_spread_current: float) -> dict:
    home_tickets_f = _safe_float(home_tickets_pct)
    spread_open_f = _safe_float(home_spread_open)
    spread_current_f = _safe_float(home_spread_current)
    if any(v is None for v in [home_tickets_f, spread_open_f, spread_current_f]):
        return {"rlm_flag": False}

    public_on_home = home_tickets_f >= PUBLIC_BET_THRESHOLD
    public_on_away = home_tickets_f <= (100 - PUBLIC_BET_THRESHOLD)
    line_delta = spread_current_f - spread_open_f

    # Home-spread convention:
    # More negative => movement toward HOME, more positive => movement toward AWAY.
    line_moved_toward_home = line_delta <= -RLM_LINE_MOVE_MIN
    line_moved_toward_away = line_delta >= RLM_LINE_MOVE_MIN

    rlm = (public_on_home and line_moved_toward_away) or (public_on_away and line_moved_toward_home)
    if not rlm:
        return {"rlm_flag": False}

    sharp_side = "away" if public_on_home else "home"
    return {
        "rlm_flag": True,
        "rlm_sharp_side": sharp_side,
        "rlm_public_pct": home_tickets_f,
        "rlm_line_move": round(line_delta, 1),
        "rlm_note": (
            f"Public {home_tickets_f:.0f}% on {'home' if public_on_home else 'away'} but "
            f"line moved {abs(line_delta):.1f}pt toward {sharp_side} - SHARP MONEY on {sharp_side.upper()}"
        ),
    }


def detect_book_disagreement(pinnacle_spread: Optional[float], draftkings_spread: Optional[float]) -> dict:
    pinnacle_f = _safe_float(pinnacle_spread)
    draftkings_f = _safe_float(draftkings_spread)
    if pinnacle_f is None or draftkings_f is None:
        return {"book_disagreement_flag": False}

    diff = abs(pinnacle_f - draftkings_f)
    if diff < BOOK_DISAGREE_POINTS:
        return {"book_disagreement_flag": False, "book_spread_diff": round(diff, 2)}

    sharp_side = "home" if pinnacle_f > draftkings_f else "away"
    note = (
        f"Pinnacle {pinnacle_f:+.1f} vs DraftKings {draftkings_f:+.1f} "
        f"(diff {diff:.1f}pts) - sharp money on {sharp_side.upper()}"
    )
    return {
        "book_disagreement_flag": True,
        "book_spread_diff": round(diff, 2),
        "pinnacle_spread": pinnacle_f,
        "draftkings_spread": draftkings_f,
        "book_sharp_side": sharp_side,
        "book_note": note,
        # Backward-compatible aliases.
        "sharp_side": sharp_side,
        "note": note,
    }


def build_market_row(
    event_id: str,
    capture_type: str,
    market_data: dict,
    pinnacle_data: Optional[dict],
    dk_data: Optional[dict],
    existing_rows: pd.DataFrame,
    pulled_at_utc: Optional[str] = None,
) -> dict:
    captured_now = datetime.now(timezone.utc).isoformat()
    pulled_now = pulled_at_utc or captured_now

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
        last_time = pd.to_datetime(last_row.get("captured_at_utc"), utc=True, errors="coerce")
        if pd.notna(last_time):
            hours_since = (pd.Timestamp.now("UTC") - last_time).total_seconds() / 3600
            steam_flag = detect_steam_move(home_spread, last_spread, hours_since)
        else:
            steam_flag = False

    rlm = detect_reverse_line_movement(home_tickets, spread_open, home_spread)

    pinn_spread = pinnacle_data.get("spread") if pinnacle_data else None
    dk_spread = dk_data.get("spread") if dk_data else None
    book_dis = detect_book_disagreement(pinn_spread, dk_spread)

    line_movement = None
    home_spread_f = _safe_float(home_spread)
    spread_open_f = _safe_float(spread_open)
    home_tickets_f = _safe_float(home_tickets)
    if home_spread_f is not None and spread_open_f is not None:
        line_movement = round(home_spread_f - spread_open_f, 2)

    return {
        "game_id": event_id,
        "event_id": event_id,
        "capture_type": capture_type,
        "captured_at_utc": captured_now,
        "pulled_at_utc": pulled_now,
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
        "book_sharp_side": book_dis.get("book_sharp_side") or book_dis.get("sharp_side"),
        "book_note": book_dis.get("book_note") or book_dis.get("note"),
        "line_freeze_flag": (
            abs(line_movement or 0) < 0.5 and home_tickets_f is not None and abs(home_tickets_f - 50.0) > 15.0
        ),
        # Gap-fill targets
        "home_win_prob": market_data.get("home_win_prob"),
        "away_win_prob": market_data.get("away_win_prob"),
        "home_ats_wins": market_data.get("home_ats_wins"),
        "home_ats_losses": market_data.get("home_ats_losses"),
        "away_ats_wins": market_data.get("away_ats_wins"),
        "away_ats_losses": market_data.get("away_ats_losses"),
        "home_team_name": market_data.get("home_team_name"),
        "away_team_name": market_data.get("away_team_name"),
        "book": "consensus",
        "market_type": "spread",
        "source": "ESPN",
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
    def _pick_dedupe_key(df: pd.DataFrame) -> list[str]:
        # Preferred order from ingestion hardening guidance.
        key_priority = [
            ["game_id", "book", "captured_at_utc"],
            ["game_id", "book", "market_type", "line_type", "snapshot_ts"],
            ["game_id", "book", "home_spread_current", "total_current", "home_ml", "away_ml", "run_id"],
        ]
        for key_cols in key_priority:
            if all(col in df.columns for col in key_cols):
                return key_cols

        fallback_key = ["game_id", "capture_type", "captured_at_utc"]
        available = [col for col in fallback_key if col in df.columns]
        if available:
            return available
        return ["event_id", "capture_type", "captured_at_utc"]

    # Keep a stable game_id alias for dedupe logic.
    if "game_id" not in df_new.columns and "event_id" in df_new.columns:
        df_new["game_id"] = df_new["event_id"]
    if "run_id" not in df_new.columns:
        df_new["run_id"] = df_new.get("pulled_at_utc", df_new.get("captured_at_utc"))

    if output_path.exists():
        df_existing = pd.read_csv(output_path, dtype={"event_id": str})
        df_existing = normalize_numeric_dtypes(df_existing)
        for col in MARKET_LINES_SCHEMA_COLUMNS:
            if col not in df_existing.columns:
                df_existing[col] = pd.NA
        if "game_id" not in df_existing.columns and "event_id" in df_existing.columns:
            df_existing["game_id"] = df_existing["event_id"]
        if "run_id" not in df_existing.columns:
            df_existing["run_id"] = df_existing.get("pulled_at_utc", df_existing.get("captured_at_utc"))

        dedupe_key_cols = _pick_dedupe_key(pd.concat([df_existing, df_new], ignore_index=True, sort=False))
        dedup_key_existing = df_existing[dedupe_key_cols].astype(str).apply(tuple, axis=1)
        dedup_key_new = df_new[dedupe_key_cols].astype(str).apply(tuple, axis=1)
        df_new = df_new[~dedup_key_new.isin(set(dedup_key_existing))]
        if not df_new.empty:
            df_new = df_new.loc[~df_new[dedupe_key_cols].astype(str).apply(tuple, axis=1).duplicated(keep="first")]

        if df_new.empty:
            log.info("No new market rows to append")
            return 0

        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        dedupe_key_cols = _pick_dedupe_key(df_new)
        df_new = df_new.loc[~df_new[dedupe_key_cols].astype(str).apply(tuple, axis=1).duplicated(keep="first")]
        df_out = df_new

    if "captured_at_utc" in df_out.columns:
        df_out = df_out.sort_values("captured_at_utc", kind="mergesort")

    inserted = len(df_new)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    df_out.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)
    log.info("Appended %s market rows (%s total in market_lines.csv)", inserted, len(df_out))
    return inserted


def _atomic_write_csv(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(f".tmp.{uuid.uuid4().hex}")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(output_path)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("append must be true/false")


def _norm_str(value: object) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def _norm_int(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"Invalid integer value: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        raw = value.strip()
        if raw == "":
            return None
        if raw.startswith("+"):
            raw = raw[1:]
        if raw.isdigit():
            return int(raw)
    raise ValueError(f"Invalid integer value: {value!r}")


def _norm_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized == "":
        return default
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def resolve_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
    days_back: Optional[int | str],
    today: Optional[date] = None,
) -> tuple[date, date]:
    start_date = _norm_str(start_date)
    end_date = _norm_str(end_date)
    normalized_days_back = _norm_int(days_back)
    today_ref = today or _pipeline_today(DEFAULT_PIPELINE_TIMEZONE)

    if normalized_days_back is not None:
        if normalized_days_back < 1:
            raise ValueError("--days-back must be >= 1")
        if start_date or end_date:
            log.info("Ignoring --start-date/--end-date because --days-back was provided")
        return today_ref - timedelta(days=normalized_days_back - 1), today_ref

    start = date.fromisoformat(start_date) if start_date else None
    end = date.fromisoformat(end_date) if end_date else None
    if start and end and start > end:
        raise ValueError("--start-date cannot be after --end-date")
    if start and not end:
        return start, today_ref
    if end and not start:
        return end, end
    if start and end:
        return start, end
    return today_ref, today_ref


def _add_date_range_args(parser: argparse.ArgumentParser, data_dir: Path) -> None:
    """Register date-range and master-file related CLI arguments exactly once."""
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--days-back", type=int, default=None)
    parser.add_argument("--append", type=_str_to_bool, default=True)
    parser.add_argument("--master-file", type=Path, default=data_dir / "market_lines_master.csv")


def build_parser(data_dir: Path) -> argparse.ArgumentParser:
    """Build the CLI parser for market line ingestion."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["morning", "pregame", "postgame", "all"], default="pregame")
    parser.add_argument("--backfill-days", type=int, default=0)
    parser.add_argument("--timezone", type=str, default=DEFAULT_PIPELINE_TIMEZONE)
    parser.add_argument("--espn-groups", type=str, default=DEFAULT_ESPN_GROUPS)
    _add_date_range_args(parser, data_dir)
    parser.add_argument("--build-views-only", action="store_true")
    return parser


def write_master_market_file(master_file: Path, new_rows: list[dict], append: bool) -> tuple[int, int, int, int]:
    bootstrap_market_lines_schema(master_file)
    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        df_new = pd.DataFrame(columns=MARKET_LINES_SCHEMA_COLUMNS)

    for col in MARKET_LINES_SCHEMA_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = pd.NA

    prior_master = pd.read_csv(master_file, dtype={"event_id": str}, low_memory=False) if master_file.exists() else pd.DataFrame()
    prior_master_rows = len(prior_master)
    if not prior_master.empty:
        for col in MARKET_LINES_SCHEMA_COLUMNS:
            if col not in prior_master.columns:
                prior_master[col] = pd.NA

    if append:
        df_out = pd.concat([prior_master, df_new], ignore_index=True, sort=False)
        base_key = ["event_id", "capture_type", "captured_at_utc"]
        extended_key = base_key + ["pinnacle_spread", "draftkings_spread", "home_spread_current", "total_current"]
        dedupe_key = extended_key if all(c in df_out.columns for c in extended_key) else base_key
        before_dedupe = len(df_out)
        df_out = df_out.drop_duplicates(subset=dedupe_key, keep="last")
        after_dedupe = len(df_out)
    else:
        before_dedupe = len(df_new)
        after_dedupe = len(df_new)
        df_out = df_new.copy()

    if "captured_at_utc" in df_out.columns:
        df_out = df_out.sort_values("captured_at_utc", kind="mergesort")

    _atomic_write_csv(df_out, master_file)
    return len(df_new), prior_master_rows, after_dedupe, len(df_out)


def write_latest_from_master(master_file: Path, latest_file: Path) -> int:
    if not master_file.exists() or master_file.stat().st_size == 0:
        _atomic_write_csv(pd.DataFrame(columns=MARKET_LINES_SCHEMA_COLUMNS), latest_file)
        return 0

    df = pd.read_csv(master_file, dtype={"event_id": str}, low_memory=False)
    if df.empty:
        _atomic_write_csv(pd.DataFrame(columns=MARKET_LINES_SCHEMA_COLUMNS), latest_file)
        return 0

    for col in MARKET_LINES_SCHEMA_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df["captured_at_utc"] = pd.to_datetime(df["captured_at_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["captured_at_utc"]).copy()
    if df.empty:
        _atomic_write_csv(pd.DataFrame(columns=MARKET_LINES_SCHEMA_COLUMNS), latest_file)
        return 0

    latest = (
        df.sort_values("captured_at_utc")
        .groupby("event_id", as_index=False)
        .tail(1)
        .sort_values("captured_at_utc")
    )
    latest["captured_at_utc"] = latest["captured_at_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
    _atomic_write_csv(latest, latest_file)
    return len(latest)


def _write_merge_report(
    data_dir: Path,
    capture_diagnostics: list[dict],
    rows_for_run: list[dict],
    latest_rows_written: int,
) -> None:
    debug_dir = data_dir.parent / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    report_path = debug_dir / "merge_report.json"

    enumerated_ids: set[str] = set()
    for diag in capture_diagnostics:
        for event_id in diag.get("event_ids_enumerated", []):
            if str(event_id).strip():
                enumerated_ids.add(str(event_id).strip())

    merged_ids = {
        str(row.get("event_id", "")).strip()
        for row in rows_for_run
        if str(row.get("event_id", "")).strip()
    }
    odds_ids = {
        str(row.get("event_id", "")).strip()
        for row in rows_for_run
        if str(row.get("event_id", "")).strip()
        and (
            _safe_float(row.get("home_spread_current")) is not None
            or _safe_float(row.get("total_current")) is not None
        )
    }
    missing_odds_ids = sorted(merged_ids - odds_ids)
    dropped_ids = sorted(enumerated_ids - merged_ids)
    filtered_final_games = sum(int(diag.get("events_filtered_final", 0)) for diag in capture_diagnostics)
    effective_slate_games = sum(
        max(int(diag.get("events_fetched", 0)) - int(diag.get("events_filtered_final", 0)), 0)
        for diag in capture_diagnostics
    )
    coverage_ratio = (len(merged_ids) / effective_slate_games) if effective_slate_games > 0 else None

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "slate_games": len(enumerated_ids),
        "effective_slate_games": effective_slate_games,
        "filtered_final_games": filtered_final_games,
        "coverage_ratio": coverage_ratio,
        "odds_games": len(odds_ids),
        "merged_games": len(merged_ids),
        "missing_odds_games": len(missing_odds_ids),
        "dropped_games": len(dropped_ids),
        "latest_rows_written": latest_rows_written,
        "missing_odds_event_ids_sample": missing_odds_ids[:20],
        "dropped_event_ids_sample": dropped_ids[:20],
        "capture_diagnostics": capture_diagnostics,
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Wrote merge report: %s", report_path)


def _load_existing_market_rows(data_dir: Path, existing: Optional[pd.DataFrame]) -> pd.DataFrame:
    if existing is not None:
        loaded = existing.copy()
    else:
        master_path = data_dir / "market_lines_master.csv"
        if master_path.exists() and master_path.stat().st_size > 0:
            loaded = pd.read_csv(master_path, dtype={"event_id": str}, low_memory=False)
        else:
            loaded = pd.DataFrame(columns=MARKET_LINES_SCHEMA_COLUMNS)

    loaded = normalize_numeric_dtypes(loaded)
    for col in MARKET_LINES_SCHEMA_COLUMNS:
        if col not in loaded.columns:
            loaded[col] = pd.NA
    return loaded


def run_capture(
    mode: str,
    data_dir: Path,
    existing: Optional[pd.DataFrame] = None,
    override_date: Optional[date] = None,
    timezone_name: str = DEFAULT_PIPELINE_TIMEZONE,
    espn_groups: str = DEFAULT_ESPN_GROUPS,
    return_diagnostics: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    today = override_date or _pipeline_today(timezone_name)
    is_backfill = override_date is not None and override_date < _pipeline_today(timezone_name)
    existing_rows = _load_existing_market_rows(data_dir, existing)
    window_start_utc, window_end_utc = _window_bounds_utc(today, timezone_name)
    log.info(
        "Market capture mode=%s date=%s timezone=%s window_utc=[%s, %s)",
        mode,
        today,
        timezone_name,
        window_start_utc.isoformat(),
        window_end_utc.isoformat(),
    )

    pulled_at_utc = datetime.now(timezone.utc).isoformat()
    log.info("Fetching ESPN scoreboard (groups=%s)...", espn_groups or "<default>")
    espn_events = fetch_espn_scoreboard(today, groups=espn_groups)
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
    parsed_ok = 0
    an_matched = 0
    missing_event_id = 0
    filtered_final = 0
    enumerated_event_ids: list[str] = []

    for event in espn_events:
        parsed = parse_espn_event(event)
        if not parsed or not parsed.get("event_id"):
            continue

        event_id = str(parsed.get("event_id", "")).strip()
        if not event_id:
            missing_event_id += 1
            continue

        enumerated_event_ids.append(event_id)
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

        parsed_ok += 1
        capture_type = {
            "morning": "opening",
            "pregame": "pregame",
            "postgame": "closing",
            "all": "opening",
        }.get(mode, "pregame")

        if capture_type == "pregame":
            status_type = (event.get("status") or {}).get("type") or {}
            status_tokens = [
                status_type.get("name"),
                status_type.get("state"),
                status_type.get("description"),
                status_type.get("detail"),
            ]
            status_upper = " ".join(str(token).upper() for token in status_tokens if token)

            if not status_upper:
                log.warning("[MARKET] event_id=%s: game status unavailable, defaulting to pregame", event_id)
            elif "FINAL" in status_upper:
                if is_backfill:
                    capture_type = "closing"
                else:
                    filtered_final += 1
                    continue
            elif "IN_PROGRESS" in status_upper:
                capture_type = "live"
            elif "STATUS_SCHEDULED" in status_upper or "STATUS_DELAYED" in status_upper:
                capture_type = "pregame"
            elif "SCHEDULED" in status_upper or "DELAYED" in status_upper:
                capture_type = "pregame"

        team_key = (
            normalize_team_name(str(parsed.get("home_team_name", ""))),
            normalize_team_name(str(parsed.get("away_team_name", ""))),
        )
        pinn_match = pinnacle_by_team.get(team_key) or pinnacle_by_team.get((team_key[1], team_key[0]))

        # DK line priority: ESPN-embedded (free, already fetched) → DK direct API → Action Network.
        dk_match = None
        if parsed.get("dk_spread") is not None:
            dk_match = {"spread": parsed["dk_spread"], "total": parsed.get("dk_total")}
        if not dk_match:
            dk_match = dk_by_team.get(team_key) or dk_by_team.get((team_key[1], team_key[0]))

        if an_enrichment:
            if not pinn_match and an_enrichment.get("pinn_spread") is not None:
                log.debug("Using Pinnacle fallback for %s: %s", event_id, an_enrichment["pinn_spread"])
                pinn_match = {"spread": an_enrichment["pinn_spread"], "total": an_enrichment.get("pinn_total")}
            if not dk_match and an_enrichment.get("dk_spread") is not None:
                log.debug("Using Action Network DraftKings fallback for %s: %s", event_id, an_enrichment["dk_spread"])
                dk_match = {"spread": an_enrichment["dk_spread"], "total": an_enrichment.get("dk_total")}

        row = build_market_row(
            event_id,
            capture_type,
            parsed,
            pinn_match,
            dk_match,
            existing_rows,
            pulled_at_utc=pulled_at_utc,
        )
        row["home_team_id"] = parsed.get("home_team_id")
        row["away_team_id"] = parsed.get("away_team_id")
        row["home_team_name"] = parsed.get("home_team_name")
        row["away_team_name"] = parsed.get("away_team_name")

        row = fill_market_row_gaps(row)
        new_rows.append(row)

    inserted = len(new_rows)
    rejected = max(len(espn_events) - parsed_ok, 0)

    log.info(
        "Market capture stage counts: fetched=%s parsed_ok=%s missing_event_id=%s filtered_final=%s inserted=%s rejected=%s",
        len(espn_events),
        parsed_ok,
        missing_event_id,
        filtered_final,
        inserted,
        rejected,
    )
    rows_with_pinnacle = sum(1 for row in new_rows if row.get("pinnacle_spread") is not None)
    rows_with_dk = sum(1 for row in new_rows if row.get("draftkings_spread") is not None)
    log.info(
        "Market ingest summary: pulled=%s matched=%s an_matched=%s inserted=%s rejected=%s",
        len(espn_events),
        parsed_ok,
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
        log.warning("STEAM DETECTED on %s games: %s", len(steam_games), ", ".join(str(r["event_id"]) for r in steam_games))

    rlm_games = [r for r in new_rows if r.get("rlm_flag")]
    for row in rlm_games:
        log.warning("REVERSE LINE MOVEMENT: event %s - %s", row["event_id"], row.get("rlm_note"))

    diagnostics = {
        "mode": mode,
        "date": today.isoformat(),
        "timezone": timezone_name,
        "espn_groups": str(espn_groups or ""),
        "window_start_utc": window_start_utc.isoformat(),
        "window_end_utc": window_end_utc.isoformat(),
        "events_fetched": len(espn_events),
        "events_parsed_ok": parsed_ok,
        "events_missing_event_id": missing_event_id,
        "events_filtered_final": filtered_final,
        "rows_inserted": inserted,
        "rows_rejected": rejected,
        "event_ids_enumerated": sorted(set(enumerated_event_ids)),
    }
    if return_diagnostics:
        return new_rows, diagnostics
    return new_rows


def main() -> None:
    try:
        from espn_config import DATA_DIR
    except ImportError:
        DATA_DIR = Path("data")

    parser = build_parser(DATA_DIR)
    args = parser.parse_args()

    args.start_date = _norm_str(args.start_date)
    args.end_date = _norm_str(args.end_date)
    args.timezone = _norm_str(args.timezone) or DEFAULT_PIPELINE_TIMEZONE
    args.espn_groups = _norm_str(args.espn_groups) or DEFAULT_ESPN_GROUPS
    env_days_back = _norm_int(os.getenv("DAYS_BACK"))
    args.days_back = _norm_int(args.days_back) if args.days_back is not None else env_days_back
    args.append = _norm_bool(args.append, default=True)

    log.info(
        "Date arg debug before resolve_date_range: start_date=%r (%s), end_date=%r (%s), days_back=%r (%s)",
        args.start_date,
        type(args.start_date).__name__,
        args.end_date,
        type(args.end_date).__name__,
        args.days_back,
        type(args.days_back).__name__,
    )

    start_date, end_date = resolve_date_range(
        args.start_date,
        args.end_date,
        args.days_back,
        today=_pipeline_today(args.timezone),
    )

    # Pregame default run: extend to tomorrow so the market master includes games up to
    # 30+ hours ahead.  This lets the perplexity pipeline produce picks for any game
    # starting between the 12pm PST run time and 6pm PST the following day.
    # Only applied when no explicit date range or days-back override was provided.
    if (
        args.mode == "pregame"
        and not args.backfill_days
        and args.start_date is None
        and args.end_date is None
        and args.days_back is None
    ):
        tomorrow = _pipeline_today(args.timezone) + timedelta(days=1)
        if end_date < tomorrow:
            end_date = tomorrow
            log.info("Pregame mode: extending capture window to include tomorrow (%s)", end_date)

    log.info(
        "Resolved date range: start=%s end=%s timezone=%s espn_groups=%s",
        start_date,
        end_date,
        args.timezone,
        args.espn_groups,
    )

    snapshots_path = DATA_DIR / "market_lines_snapshots.csv"
    legacy_path = DATA_DIR / "market_lines.csv"
    odds_path = DATA_DIR / "odds_snapshot.csv"
    master_path = args.master_file if args.master_file.is_absolute() else Path(args.master_file)

    bootstrap_market_lines_schema(master_path)
    if args.build_views_only:
        master_view_df = pd.read_csv(master_path, dtype={"event_id": str}, low_memory=False)
        log.info("Build-views-only: writing snapshots/legacy/odds from master rows=%s", len(master_view_df))
        _atomic_write_csv(master_view_df, snapshots_path)
        _atomic_write_csv(master_view_df, legacy_path)
        _atomic_write_csv(master_view_df, odds_path)
        latest_rows, closing_rows = regenerate_market_views(DATA_DIR)
        canonical_paths = write_market_canonical_outputs(DATA_DIR)
        log.info("Rebuilt views from master: latest_rows=%s closing_rows=%s", latest_rows, closing_rows)
        log.info(
            "Canonical market outputs refreshed: latest_by_game=%s canonical=%s",
            canonical_paths.get("latest_by_game_path"),
            canonical_paths.get("canonical_path"),
        )
        return

    if args.backfill_days > 0 and args.days_back is None and not args.start_date and not args.end_date:
        start_date = _pipeline_today(args.timezone) - timedelta(days=args.backfill_days)
        end_date = _pipeline_today(args.timezone)
        log.info("Using --backfill-days fallback range: start=%s end=%s", start_date, end_date)

    existing_master = pd.read_csv(master_path, dtype={"event_id": str}, low_memory=False) if master_path.exists() else pd.DataFrame()
    existing_master = normalize_numeric_dtypes(existing_master)

    all_rows: list[dict] = []
    capture_diagnostics: list[dict] = []
    span = (end_date - start_date).days
    for offset in range(span + 1):
        target = start_date + timedelta(days=offset)
        log.info("Capture date: %s", target)
        if args.mode == "all":
            for mode in ["morning", "pregame", "postgame"]:
                rows, diag = run_capture(
                    mode,
                    DATA_DIR,
                    existing_master,
                    override_date=target,
                    timezone_name=args.timezone,
                    espn_groups=args.espn_groups,
                    return_diagnostics=True,
                )
                all_rows.extend(rows)
                capture_diagnostics.append(diag)
                time.sleep(REQUEST_DELAY)
        else:
            rows, diag = run_capture(
                args.mode,
                DATA_DIR,
                existing_master,
                override_date=target,
                timezone_name=args.timezone,
                espn_groups=args.espn_groups,
                return_diagnostics=True,
            )
            all_rows.extend(rows)
            capture_diagnostics.append(diag)

    # Subtract FINAL-filtered games: in pregame mode, completed games are legitimately
    # excluded from insertion (not lost). Using raw events_fetched as the denominator
    # causes false positives on days with many completed games (e.g. conference tournaments).
    enumerated_total = sum(
        int(d.get("events_fetched", 0)) - int(d.get("events_filtered_final", 0))
        for d in capture_diagnostics
    )
    inserted_total = len(all_rows)
    if enumerated_total >= MIN_ENUM_GAMES_FOR_GAP_GUARD:
        minimum_rows = int(round(enumerated_total * (1.0 - MAX_ENUM_TO_OUTPUT_DROP_RATIO)))
        if inserted_total < minimum_rows:
            raise RuntimeError(
                "Market capture underfilled slate: "
                f"enumerated={enumerated_total} inserted={inserted_total} "
                f"(required >= {minimum_rows}). Check scoreboard filters and parsing."
            )

    new_rows, prior_rows, rows_after_dedupe, rows_written = write_master_market_file(master_path, all_rows, args.append)
    if not master_path.exists():
        raise RuntimeError(f"Market lines write failed: file missing at {master_path}")

    master_df = pd.read_csv(master_path, dtype={"event_id": str}, low_memory=False)
    required_cols = ["event_id", "captured_at_utc", "verification_status", "verification_notes"]
    missing_cols = [c for c in required_cols if c not in master_df.columns]
    if missing_cols:
        raise RuntimeError(f"Market lines write failed: missing columns {missing_cols} in {master_path}")

    log.info("Writing derived snapshots copy: rows=%s -> %s", len(master_df), snapshots_path)
    _atomic_write_csv(master_df, snapshots_path)
    log.info("Writing legacy market_lines copy: rows=%s -> %s", len(master_df), legacy_path)
    _atomic_write_csv(master_df, legacy_path)
    log.info("Writing odds snapshot copy: rows=%s -> %s", len(master_df), odds_path)
    _atomic_write_csv(master_df, odds_path)
    latest_rows, closing_rows = regenerate_market_views(DATA_DIR)
    canonical_paths = write_market_canonical_outputs(DATA_DIR)
    _write_merge_report(DATA_DIR, capture_diagnostics, all_rows, latest_rows)

    log.info("Master merge: new rows fetched=%s", new_rows)
    log.info("Master merge: prior master rows=%s", prior_rows)
    log.info("Master merge: rows after dedupe=%s", rows_after_dedupe)
    log.info("Master merge: rows written=%s", rows_written)
    log.info("Derived latest rows written=%s closing_rows=%s", latest_rows, closing_rows)
    log.info(
        "Canonical market outputs refreshed: latest_by_game=%s canonical=%s",
        canonical_paths.get("latest_by_game_path"),
        canonical_paths.get("canonical_path"),
    )


if __name__ == "__main__":
    main()


