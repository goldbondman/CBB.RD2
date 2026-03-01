"""Unofficial ESPN endpoint gap fillers for odds/win-probability/ATS.

These are best-effort enrichers and only intended to fill missing fields.
They never overwrite already-populated values.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import requests

log = logging.getLogger(__name__)

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
SPORT = "basketball"
LEAGUE = "mens-college-basketball"
CORE = f"https://sports.core.api.espn.com/v2/sports/{SPORT}/leagues/{LEAGUE}"


def _get(url: str, label: str) -> Optional[dict | list]:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code == 200:
            return resp.json()
        log.debug("[%s] HTTP %s for %s", label, resp.status_code, url)
        return None
    except Exception as exc:  # noqa: BLE001
        log.debug("[%s] request failed: %s", label, exc)
        return None


def fetch_game_odds_espn(event_id: str, comp_id: Optional[str] = None) -> dict[str, dict[str, Any]]:
    comp_id = comp_id or event_id
    payload = _get(f"{CORE}/events/{event_id}/competitions/{comp_id}/odds", "core_odds")
    if not payload:
        return {}

    items = payload.get("items", [payload]) if isinstance(payload, dict) else []
    results: dict[str, dict[str, Any]] = {}

    for item in items:
        if not isinstance(item, dict):
            continue
        provider = item.get("provider", {}) if isinstance(item.get("provider"), dict) else {}
        pid = str(provider.get("id", ""))
        pname = str(provider.get("name", "")).lower()

        spread = item.get("spread")
        if spread is None and isinstance(item.get("details"), str):
            details = item.get("details")
            try:
                spread = float(str(details).split()[-1])
            except Exception:
                spread = None

        entry = {
            "spread": spread,
            "over_under": item.get("overUnder"),
            "home_ml": ((item.get("homeTeamOdds") or {}).get("moneyLine") if isinstance(item.get("homeTeamOdds"), dict) else None),
            "away_ml": ((item.get("awayTeamOdds") or {}).get("moneyLine") if isinstance(item.get("awayTeamOdds"), dict) else None),
        }

        if pid:
            results[pid] = entry
        if pid == "41" or "draftkings" in pname:
            results["draftkings"] = entry
        if pid == "25" or "pinnacle" in pname:
            results["pinnacle"] = entry

    return results


def fetch_pickcenter_odds(event_id: str) -> dict[str, dict[str, Any]]:
    payload = _get(
        f"https://site.api.espn.com/apis/site/v2/sports/{SPORT}/{LEAGUE}/summary?event={event_id}",
        "pickcenter",
    )
    if not isinstance(payload, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for pc in payload.get("pickcenter", []) or []:
        if not isinstance(pc, dict):
            continue
        pname = str(((pc.get("provider") or {}).get("name", "")).lower())
        entry = {
            "spread": pc.get("details"),
            "over_under": pc.get("overUnder"),
            "home_ml": ((pc.get("homeTeamOdds") or {}).get("moneyLine") if isinstance(pc.get("homeTeamOdds"), dict) else None),
            "away_ml": ((pc.get("awayTeamOdds") or {}).get("moneyLine") if isinstance(pc.get("awayTeamOdds"), dict) else None),
        }
        if pname:
            out[pname] = entry
        if "draftkings" in pname:
            out["draftkings"] = entry
        if "pinnacle" in pname:
            out["pinnacle"] = entry
    return out


def fetch_win_probability(event_id: str, comp_id: Optional[str] = None) -> dict[str, Any]:
    comp_id = comp_id or event_id
    payload = _get(
        f"{CORE}/events/{event_id}/competitions/{comp_id}/probabilities",
        "probabilities",
    )
    if not payload:
        return {}

    items = payload.get("items", [payload]) if isinstance(payload, dict) else []
    if not items:
        return {}
    first = items[0] if isinstance(items[0], dict) else {}
    return {
        "home_win_prob": first.get("homeWinPercentage"),
        "away_win_prob": first.get("awayWinPercentage"),
    }


def fetch_team_ats_espn(team_id: str, year: int = 2026, season_type: int = 2) -> dict[str, Any]:
    payload = _get(
        f"{CORE}/seasons/{year}/types/{season_type}/teams/{team_id}/ats",
        "team_ats",
    )
    if not isinstance(payload, dict):
        return {}
    return {
        "ats_wins": payload.get("wins") or payload.get("atsWins"),
        "ats_losses": payload.get("losses") or payload.get("atsLosses"),
        "ats_pushes": payload.get("pushes") or payload.get("atsPushes"),
        "ats_pct": payload.get("winPercent") or payload.get("atsWinPercent"),
    }


def fill_market_row_gaps(row: dict[str, Any], year: int = 2026, season_type: int = 2) -> dict[str, Any]:
    """Fill only null gap fields in a market row using unofficial ESPN endpoints."""
    out = dict(row)
    event_id = str(out.get("event_id") or "").strip()
    if not event_id:
        return out

    need_odds = any(
        out.get(k) in (None, "", "nan")
        for k in ["draftkings_spread", "home_ml", "away_ml", "home_spread_current"]
    )
    if need_odds:
        odds = fetch_game_odds_espn(event_id)
        if not odds:
            odds = fetch_pickcenter_odds(event_id)
        dk = odds.get("draftkings", {}) if isinstance(odds, dict) else {}
        if out.get("draftkings_spread") in (None, "", "nan"):
            out["draftkings_spread"] = dk.get("spread")
        if out.get("home_ml") in (None, "", "nan"):
            out["home_ml"] = dk.get("home_ml")
        if out.get("away_ml") in (None, "", "nan"):
            out["away_ml"] = dk.get("away_ml")
        if out.get("over_under") in (None, "", "nan"):
            out["over_under"] = dk.get("over_under")
        # Fill home_spread_current from ESPN Core API DK when ESPN scoreboard
        # omits odds.  ESPN Core API uses home-team perspective so sign matches.
        if out.get("home_spread_current") in (None, "", "nan") and dk.get("spread") is not None:
            out["home_spread_current"] = dk.get("spread")
        if out.get("total_current") in (None, "", "nan") and dk.get("over_under") is not None:
            out["total_current"] = dk.get("over_under")

    if out.get("home_win_prob") in (None, "", "nan") or out.get("away_win_prob") in (None, "", "nan"):
        out.update({k: v for k, v in fetch_win_probability(event_id).items() if v is not None})

    home_team_id = str(out.get("home_team_id") or "").strip()
    away_team_id = str(out.get("away_team_id") or "").strip()
    if home_team_id and out.get("home_ats_wins") in (None, "", "nan"):
        ats_h = fetch_team_ats_espn(home_team_id, year=year, season_type=season_type)
        out.setdefault("home_ats_wins", ats_h.get("ats_wins"))
        out.setdefault("home_ats_losses", ats_h.get("ats_losses"))
    if away_team_id and out.get("away_ats_wins") in (None, "", "nan"):
        ats_a = fetch_team_ats_espn(away_team_id, year=year, season_type=season_type)
        out.setdefault("away_ats_wins", ats_a.get("ats_wins"))
        out.setdefault("away_ats_losses", ats_a.get("ats_losses"))

    return out
