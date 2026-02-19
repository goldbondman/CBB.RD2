"""
ESPN CBB Pipeline — Parsers
Convert raw ESPN JSON into clean flat dictionaries.
One function per data shape. No I/O here.
"""

import logging
from zoneinfo import ZoneInfo
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

TZ_PST = ZoneInfo("America/Los_Angeles")  # handles PST/PDT automatically


def _utc_to_pst(utc_str: str) -> Optional[str]:
    """
    Convert an ISO 8601 UTC string from ESPN to a PST/PDT local datetime string.
    Returns None if the input cannot be parsed.
    Example: '2026-02-18T03:00Z' -> '2026-02-17 07:00 PM PST'
    """
    if not utc_str:
        return None
    try:
        utc_clean = utc_str.replace("Z", "+00:00")
        dt_utc = datetime.fromisoformat(utc_clean)
        dt_pst = dt_utc.astimezone(TZ_PST)
        return dt_pst.strftime("%Y-%m-%d %I:%M %p %Z")
    except Exception:
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_int(val: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_made_attempt(s: Any) -> Tuple[Optional[int], Optional[int]]:
    """Parse '12-20' style strings into (made, attempted)."""
    try:
        parts = str(s).split("-")
        if len(parts) == 2:
            return _safe_int(parts[0]), _safe_int(parts[1])
    except Exception:
        pass
    return None, None


# ── Scoreboard parser ─────────────────────────────────────────────────────────

def parse_scoreboard_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse one event from the ESPN scoreboard response into a flat game row.
    Returns None if the event cannot be meaningfully parsed.
    """
    try:
        game_id = str(event.get("id", "")).strip()
        if not game_id:
            return None

        comps = event.get("competitions", [])
        if not comps:
            return None
        comp = comps[0]

        # Status
        status      = comp.get("status", {})
        status_type = status.get("type", {})
        completed   = bool(status_type.get("completed", False))
        state       = status_type.get("state", "")
        status_desc = status_type.get("description", "")

        # Date/time
        game_dt_utc = comp.get("date", event.get("date", ""))

        # Venue
        venue_obj = comp.get("venue", {})
        venue = venue_obj.get("fullName", "")

        # Neutral site
        neutral_site = bool(comp.get("neutralSite", False))

        # Competitors
        home_team = away_team = ""
        home_id   = away_id   = ""
        home_score = away_score = None

        for c in comp.get("competitors", []):
            ha    = c.get("homeAway", "").lower()
            tname = c.get("team", {}).get("displayName", c.get("team", {}).get("name", ""))
            tid   = str(c.get("team", {}).get("id", "")).strip()
            score = _safe_int(c.get("score"))
            if ha == "home":
                home_team, home_id, home_score = tname, tid, score
            elif ha == "away":
                away_team, away_id, away_score = tname, tid, score

        if not home_team or not away_team:
            return None

        # Odds (best-effort)
        odds      = comp.get("odds", [{}])
        odds_obj  = odds[0] if odds else {}
        spread    = _safe_float(odds_obj.get("spread"))
        over_under = _safe_float(odds_obj.get("overUnder"))
        home_ml   = _safe_float(odds_obj.get("homeTeamOdds", {}).get("moneyLine"))
        away_ml   = _safe_float(odds_obj.get("awayTeamOdds", {}).get("moneyLine"))

        return {
            "game_id":          game_id,
            "game_datetime_utc": game_dt_utc,
            "game_datetime_pst": _utc_to_pst(game_dt_utc),
            "venue":            venue,
            "neutral_site":     neutral_site,
            "home_team":        home_team,
            "home_team_id":     home_id,
            "away_team":        away_team,
            "away_team_id":     away_id,
            "home_score":       home_score,
            "away_score":       away_score,
            "completed":        completed,
            "state":            state,
            "status_desc":      status_desc,
            "spread":           spread,
            "over_under":       over_under,
            "home_ml":          home_ml,
            "away_ml":          away_ml,
        }

    except Exception as exc:
        log.warning(f"parse_scoreboard_event failed for event {event.get('id','?')}: {exc}")
        return None


# ── Summary parser ────────────────────────────────────────────────────────────

def parse_summary(raw: Dict[str, Any], event_id: str) -> Optional[Dict[str, Any]]:
    """
    Parse ESPN summary response into a structured dict with:
      - header:  game-level metadata
      - home:    home team box score stats
      - away:    away team box score stats
      - players: list of player rows
    Returns None on failure.
    """
    try:
        header    = raw.get("header", {})
        boxscore  = raw.get("boxscore", {})
        comps     = header.get("competitions", [])
        comp      = comps[0] if comps else {}

        # ── Game metadata ──
        game_dt_utc  = comp.get("date", "")
        neutral_site = bool(comp.get("neutralSite", False))
        venue        = comp.get("venue", {}).get("fullName", "")
        status_type  = comp.get("status", {}).get("type", {})
        completed    = bool(status_type.get("completed", False))
        state        = status_type.get("state", "")

        # OT detection
        period    = comp.get("status", {}).get("period", 1)
        is_ot     = period > 2
        num_ot    = max(0, period - 2) if is_ot else 0

        # ── Competitors ──
        home_meta = away_meta = {}
        for c in comp.get("competitors", []):
            ha = c.get("homeAway", "").lower()
            if ha == "home":
                home_meta = c
            elif ha == "away":
                away_meta = c

        def _team_meta(meta: Dict) -> Dict:
            team = meta.get("team", {})
            return {
                "team_id":   str(team.get("id", "")).strip(),
                "team":      team.get("displayName", team.get("name", "")),
                "score":     _safe_int(meta.get("score")),
                "home_away": meta.get("homeAway", "").lower(),
            }

        home_info = _team_meta(home_meta)
        away_info = _team_meta(away_meta)

        # ── Box score team stats ──
        def _parse_team_stats(teams_list: List, target_id: str) -> Dict:
            """Extract flat stat dict for a team from boxscore.teams."""
            stats: Dict[str, Any] = {}
            for t in teams_list:
                tid = str(t.get("team", {}).get("id", "")).strip()
                if tid != target_id:
                    continue
                for s in t.get("statistics", []):
                    name  = s.get("name", "")
                    label = s.get("abbreviation", name).lower().replace(" ", "_")
                    dv    = s.get("displayValue", "")

                    # Shooting splits: "12-20" → _made/_attempted
                    if "-" in str(dv) and any(x in name.lower() for x in ["goal", "throw", "three"]):
                        made, att = _parse_made_attempt(dv)
                        if "three" in name.lower() or "3" in name.lower():
                            stats["tpm"], stats["tpa"] = made, att
                        elif "free" in name.lower():
                            stats["ftm"], stats["fta"] = made, att
                        else:
                            stats["fgm"], stats["fga"] = made, att
                    else:
                        # Map abbreviations AND full names → clean column names.
                        # ORB/DRB must be mapped before REB so total rebounds
                        # don't overwrite them if ESPN sends all three.
                        col_map = {
                            # Rebounds — specific before general
                            "oreb": "orb", "offensiverebounds": "orb",
                            "dreb": "drb", "defensiverebounds": "drb",
                            "reb":  "reb", "rebounds": "reb",
                            "totalrebounds": "reb",
                            # Other
                            "ast": "ast", "assists": "ast",
                            "to":  "tov", "turnovers": "tov",
                            "stl": "stl", "steals": "stl",
                            "blk": "blk", "blocks": "blk",
                            "pf":  "pf",  "fouls": "pf",
                        }
                        # Match on abbreviation first, then full name
                        key = col_map.get(label) or col_map.get(name.lower().replace(" ", ""), label)
                        val_parsed = _safe_int(dv) if str(dv).isdigit() else _safe_float(dv)
                        # Only set REB if ORB/DRB not already found
                        if key == "reb" and ("orb" in stats or "drb" in stats):
                            stats.setdefault("reb", val_parsed)
                        else:
                            stats[key] = val_parsed
            return stats

        bs_teams  = boxscore.get("teams", [])
        home_stats = _parse_team_stats(bs_teams, home_info["team_id"])
        away_stats = _parse_team_stats(bs_teams, away_info["team_id"])

        # ── Player rows ──
        players: List[Dict] = []
        for team_block in boxscore.get("players", []):
            tid   = str(team_block.get("team", {}).get("id", "")).strip()
            tname = team_block.get("team", {}).get("displayName", "")
            ha    = "home" if tid == home_info["team_id"] else "away"

            stat_labels: List[str] = []
            for pg in team_block.get("statistics", []):
                stat_labels = [k.lower() for k in pg.get("keys", [])]
                for athlete_entry in pg.get("athletes", []):
                    athlete = athlete_entry.get("athlete", {})
                    vals    = athlete_entry.get("stats", [])
                    did_not_play = athlete_entry.get("didNotPlay", False)

                    # Skip players with no stats and who didn't play
                    if did_not_play and not vals:
                        continue

                    prow: Dict[str, Any] = {
                        "event_id":         event_id,
                        "game_datetime_utc": game_dt_utc,
                        "game_datetime_pst": _utc_to_pst(game_dt_utc),
                        "team_id":          tid,
                        "team":             tname,
                        "home_away":        ha,
                        "athlete_id":       str(athlete.get("id", "")).strip(),
                        "player":           athlete.get("displayName", ""),
                        "starter":          bool(athlete_entry.get("starter", False)),
                        "did_not_play":     did_not_play,
                    }

                    # Map stat labels → clean column names
                    # ORB/DRB before REB to avoid total overwriting splits
                    label_map = {
                        "min":  "min",  "pts":  "pts",
                        "fg":   "_fg_raw",   # parsed as made-attempt below
                        "3pt":  "_3pt_raw",
                        "ft":   "_ft_raw",
                        "oreb": "orb",  "dreb": "drb",  "reb": "reb",
                        "ast":  "ast",  "stl":  "stl",  "blk": "blk",
                        "to":   "tov",  "pf":   "pf",
                    }
                    raw_stats: Dict[str, str] = {}
                    for lbl, val in zip(stat_labels, vals):
                        raw_stats[lbl] = val
                        mapped = label_map.get(lbl)
                        if mapped and not mapped.startswith("_"):
                            prow[mapped] = _safe_int(val) if str(val).isdigit() else _safe_float(val)

                    # Parse shooting splits (e.g. "12-20" → fgm=12, fga=20)
                    for src_key, m_col, a_col in [
                        ("fg",  "fgm", "fga"),
                        ("3pt", "tpm", "tpa"),
                        ("ft",  "ftm", "fta"),
                    ]:
                        if src_key in raw_stats:
                            m, a = _parse_made_attempt(raw_stats[src_key])
                            prow[m_col], prow[a_col] = m, a

                    # Only append if we have at least an athlete_id
                    if prow.get("athlete_id"):
                        players.append(prow)

        return {
            "event_id":         event_id,
            "game_datetime_utc": game_dt_utc,
            "game_datetime_pst": _utc_to_pst(game_dt_utc),
            "venue":            venue,
            "neutral_site":     neutral_site,
            "completed":        completed,
            "state":            state,
            "is_ot":            is_ot,
            "num_ot":           num_ot,
            "home":             {**home_info, **home_stats},
            "away":             {**away_info, **away_stats},
            "players":          players,
        }

    except Exception as exc:
        log.error(f"parse_summary failed for event {event_id}: {exc}")
        return None


def summary_to_team_rows(parsed: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """
    Convert a parsed summary dict into two flat team-game rows (home + away).
    """
    base = {
        "event_id":          parsed["event_id"],
        "game_datetime_utc": parsed["game_datetime_utc"],
        "game_datetime_pst": parsed.get("game_datetime_pst"),
        "venue":             parsed["venue"],
        "neutral_site":      parsed["neutral_site"],
        "completed":         parsed["completed"],
        "state":             parsed["state"],
        "is_ot":             parsed["is_ot"],
        "num_ot":            parsed["num_ot"],
        "home_team":         parsed["home"].get("team"),
        "away_team":         parsed["away"].get("team"),
        "home_team_id":      parsed["home"].get("team_id"),
        "away_team_id":      parsed["away"].get("team_id"),
    }

    def _make_row(side: str) -> Dict:
        me  = parsed[side]
        opp = parsed["away" if side == "home" else "home"]
        return {
            **base,
            "team_id":       me.get("team_id"),
            "team":          me.get("team"),
            "home_away":     side,
            "opponent_id":   opp.get("team_id"),
            "opponent":      opp.get("team"),
            "points_for":    me.get("score"),
            "points_against": opp.get("score"),
            "margin":        (me.get("score") or 0) - (opp.get("score") or 0)
                             if me.get("score") is not None and opp.get("score") is not None
                             else None,
            "fgm":  me.get("fgm"),  "fga":  me.get("fga"),
            "tpm":  me.get("tpm"),  "tpa":  me.get("tpa"),
            "ftm":  me.get("ftm"),  "fta":  me.get("fta"),
            "orb":  me.get("orb"),  "drb":  me.get("drb"),
            "reb":  me.get("reb"),
            "ast":  me.get("ast"),  "stl":  me.get("stl"),
            "blk":  me.get("blk"),  "tov":  me.get("tov"),
            "pf":   me.get("pf"),
        }

    return _make_row("home"), _make_row("away")
