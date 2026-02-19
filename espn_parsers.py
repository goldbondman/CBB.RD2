"""
ESPN CBB Pipeline — Parsers
Convert raw ESPN JSON into clean flat dictionaries.
One function per data shape. No I/O here.
"""

import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

TZ_PST = ZoneInfo("America/Los_Angeles")  # handles PST/PDT automatically


def _utc_to_pst(utc_str: str) -> Optional[str]:
    """
    Convert an ISO 8601 UTC string from ESPN to PST/PDT local datetime string.
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


def _parse_minutes(val: Any) -> Optional[float]:
    """
    Parse ESPN minutes string. ESPN returns minutes as:
      - "32" (integer string)
      - "32:15" (minutes:seconds)
      - "32.0" (float string)
    Returns decimal minutes.
    """
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in ("", "--", "-"):
        return None
    try:
        if ":" in s:
            parts = s.split(":")
            mins = int(parts[0])
            secs = int(parts[1]) if len(parts) > 1 else 0
            return round(mins + secs / 60, 1)
        return float(s)
    except Exception:
        return None


# ── Stat label normalization ──────────────────────────────────────────────────
# ESPN returns player stat keys in various formats depending on endpoint/version.
# This master map covers all known variants (case-insensitive after .lower()).

PLAYER_STAT_MAP = {
    # Minutes
    "min":          "min",
    "minutes":      "min",
    # Points
    "pts":          "pts",
    "points":       "pts",
    # Field goals — shooting split "made-attempted"
    "fg":           "_fg",
    "fgm-a":        "_fg",
    "field goals":  "_fg",
    # Three pointers
    "3pt":          "_3pt",
    "3p":           "_3pt",
    "3-pt":         "_3pt",
    "3pm-a":        "_3pt",
    "three pointers": "_3pt",
    # Free throws
    "ft":           "_ft",
    "ftm-a":        "_ft",
    "free throws":  "_ft",
    # Rebounds
    "oreb":         "orb",
    "or":           "orb",
    "offensive rebounds": "orb",
    "dreb":         "drb",
    "dr":           "drb",
    "defensive rebounds": "drb",
    "reb":          "reb",
    "tr":           "reb",
    "rebounds":     "reb",
    "total rebounds": "reb",
    # Other
    "ast":          "ast",
    "a":            "ast",
    "assists":      "ast",
    "stl":          "stl",
    "s":            "stl",
    "steals":       "stl",
    "blk":          "blk",
    "b":            "blk",
    "blocks":       "blk",
    "to":           "tov",
    "tov":          "tov",
    "turnovers":    "tov",
    "pf":           "pf",
    "fouls":        "pf",
    "personal fouls": "pf",
    "+/-":          "plus_minus",
    "plusminus":    "plus_minus",
}


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
        home_conf = away_conf = ""
        home_conf_id = away_conf_id = ""

        for c in comp.get("competitors", []):
            ha    = c.get("homeAway", "").lower()
            team  = c.get("team", {})
            tname = team.get("displayName", team.get("name", ""))
            tid   = str(team.get("id", "")).strip()
            score = _safe_int(c.get("score"))
            # Conference
            conf     = team.get("conferenceId", "")
            conf_obj = team.get("conference", {})
            conf_name = conf_obj.get("name", conf_obj.get("shortName", ""))

            if ha == "home":
                home_team, home_id, home_score = tname, tid, score
                home_conf, home_conf_id = conf_name, str(conf)
            elif ha == "away":
                away_team, away_id, away_score = tname, tid, score
                away_conf, away_conf_id = conf_name, str(conf)

        if not home_team or not away_team:
            return None

        # Odds (best-effort)
        odds_list  = comp.get("odds", [])
        odds_obj   = odds_list[0] if odds_list else {}
        spread     = _safe_float(odds_obj.get("spread"))
        over_under = _safe_float(odds_obj.get("overUnder"))
        home_ml    = _safe_float(odds_obj.get("homeTeamOdds", {}).get("moneyLine"))
        away_ml    = _safe_float(odds_obj.get("awayTeamOdds", {}).get("moneyLine"))
        odds_provider = odds_obj.get("provider", {}).get("name", "")
        odds_details  = odds_obj.get("details", "")

        return {
            "game_id":           game_id,
            "game_datetime_utc": game_dt_utc,
            "game_datetime_pst": _utc_to_pst(game_dt_utc),
            "venue":             venue,
            "neutral_site":      neutral_site,
            "home_team":         home_team,
            "home_team_id":      home_id,
            "home_conference":   home_conf,
            "home_conf_id":      home_conf_id,
            "away_team":         away_team,
            "away_team_id":      away_id,
            "away_conference":   away_conf,
            "away_conf_id":      away_conf_id,
            "home_score":        home_score,
            "away_score":        away_score,
            "completed":         completed,
            "state":             state,
            "status_desc":       status_desc,
            "spread":            spread,
            "over_under":        over_under,
            "home_ml":           home_ml,
            "away_ml":           away_ml,
            "odds_provider":     odds_provider,
            "odds_details":      odds_details,
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
        header   = raw.get("header", {})
        boxscore = raw.get("boxscore", {})
        comps    = header.get("competitions", [])
        comp     = comps[0] if comps else {}

        # ── Game metadata ──
        game_dt_utc  = comp.get("date", "")
        neutral_site = bool(comp.get("neutralSite", False))
        venue        = comp.get("venue", {}).get("fullName", "")
        status_type  = comp.get("status", {}).get("type", {})
        completed    = bool(status_type.get("completed", False))
        state        = status_type.get("state", "")

        # OT detection
        period = comp.get("status", {}).get("period", 2)
        is_ot  = period > 2
        num_ot = max(0, period - 2) if is_ot else 0

        # Odds from summary (fallback if scoreboard odds missing)
        odds_list     = comp.get("odds", [])
        odds_obj      = odds_list[0] if odds_list else {}
        spread        = _safe_float(odds_obj.get("spread"))
        over_under    = _safe_float(odds_obj.get("overUnder"))
        home_ml       = _safe_float(odds_obj.get("homeTeamOdds", {}).get("moneyLine"))
        away_ml       = _safe_float(odds_obj.get("awayTeamOdds", {}).get("moneyLine"))
        odds_provider = odds_obj.get("provider", {}).get("name", "")
        odds_details  = odds_obj.get("details", "")

        # ── Competitors ──
        home_meta = away_meta = {}
        for c in comp.get("competitors", []):
            ha = c.get("homeAway", "").lower()
            if ha == "home":
                home_meta = c
            elif ha == "away":
                away_meta = c

        def _team_meta(meta: Dict) -> Dict:
            team     = meta.get("team", {})
            conf_obj = team.get("conference", {})
            return {
                "team_id":    str(team.get("id", "")).strip(),
                "team":       team.get("displayName", team.get("name", "")),
                "conference": conf_obj.get("name", conf_obj.get("shortName",
                              str(team.get("conferenceId", "")))),
                "conf_id":    str(team.get("conferenceId", "")),
                "score":      _safe_int(meta.get("score")),
                "home_away":  meta.get("homeAway", "").lower(),
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
                        col_map = {
                            "oreb": "orb", "offensiverebounds": "orb",
                            "dreb": "drb", "defensiverebounds": "drb",
                            "reb":  "reb", "rebounds": "reb",
                            "totalrebounds": "reb",
                            "ast": "ast", "assists": "ast",
                            "to":  "tov", "turnovers": "tov",
                            "stl": "stl", "steals": "stl",
                            "blk": "blk", "blocks": "blk",
                            "pf":  "pf",  "fouls": "pf",
                        }
                        key = col_map.get(label) or col_map.get(
                            name.lower().replace(" ", ""), label)
                        val_parsed = (_safe_int(dv) if str(dv).isdigit()
                                      else _safe_float(dv))
                        if key == "reb" and ("orb" in stats or "drb" in stats):
                            stats.setdefault("reb", val_parsed)
                        else:
                            stats[key] = val_parsed
            return stats

        bs_teams   = boxscore.get("teams", [])
        home_stats = _parse_team_stats(bs_teams, home_info["team_id"])
        away_stats = _parse_team_stats(bs_teams, away_info["team_id"])

        # ── Player rows ──
        players: List[Dict] = []
        for team_block in boxscore.get("players", []):
            tid   = str(team_block.get("team", {}).get("id", "")).strip()
            tname = team_block.get("team", {}).get("displayName", "")
            ha    = "home" if tid == home_info["team_id"] else "away"

            for pg in team_block.get("statistics", []):
                # ESPN returns keys as the column headers for stats array
                raw_keys = pg.get("keys", [])
                stat_labels = [k.lower().strip() for k in raw_keys]

                for athlete_entry in pg.get("athletes", []):
                    athlete      = athlete_entry.get("athlete", {})
                    vals         = athlete_entry.get("stats", [])
                    did_not_play = bool(athlete_entry.get("didNotPlay", False))

                    # Skip DNP with no stats
                    if did_not_play and not any(v not in ("", "--", "-", "0") for v in vals):
                        continue

                    athlete_id = str(athlete.get("id", "")).strip()
                    if not athlete_id:
                        continue

                    prow: Dict[str, Any] = {
                        "event_id":         event_id,
                        "game_datetime_utc": game_dt_utc,
                        "game_datetime_pst": _utc_to_pst(game_dt_utc),
                        "team_id":          tid,
                        "team":             tname,
                        "home_away":        ha,
                        "athlete_id":       athlete_id,
                        "player":           athlete.get("displayName", ""),
                        "jersey":           athlete.get("jersey", ""),
                        "position":         athlete.get("position", {}).get("abbreviation", ""),
                        "starter":          bool(athlete_entry.get("starter", False)),
                        "did_not_play":     did_not_play,
                        # Init all stat cols to None
                        "min":  None, "pts":  None,
                        "fgm":  None, "fga":  None,
                        "tpm":  None, "tpa":  None,
                        "ftm":  None, "fta":  None,
                        "orb":  None, "drb":  None, "reb": None,
                        "ast":  None, "stl":  None, "blk": None,
                        "tov":  None, "pf":   None,
                        "plus_minus": None,
                    }

                    # Map each stat label to its value
                    raw_stats: Dict[str, str] = {}
                    for lbl, val in zip(stat_labels, vals):
                        raw_stats[lbl] = val
                        mapped = PLAYER_STAT_MAP.get(lbl)
                        if mapped is None:
                            continue
                        if mapped == "min":
                            prow["min"] = _parse_minutes(val)
                        elif mapped.startswith("_"):
                            pass  # handled below as shooting split
                        else:
                            cleaned = str(val).strip()
                            if cleaned in ("", "--", "-"):
                                prow[mapped] = None
                            else:
                                prow[mapped] = (_safe_int(cleaned)
                                                if cleaned.isdigit()
                                                else _safe_float(cleaned))

                    # Parse shooting splits: "7-14" → made=7, attempted=14
                    # Each split type is handled independently — no shared break.
                    for fg_lbl in ("fg", "fgm-a"):
                        if fg_lbl in raw_stats:
                            m, a = _parse_made_attempt(raw_stats[fg_lbl])
                            if m is not None:
                                prow["fgm"], prow["fga"] = m, a
                            break
                    for tp_lbl in ("3pt", "3p", "3-pt", "3pm-a"):
                        if tp_lbl in raw_stats:
                            m, a = _parse_made_attempt(raw_stats[tp_lbl])
                            if m is not None:
                                prow["tpm"], prow["tpa"] = m, a
                            break
                    for ft_lbl in ("ft", "ftm-a"):
                        if ft_lbl in raw_stats:
                            m, a = _parse_made_attempt(raw_stats[ft_lbl])
                            if m is not None:
                                prow["ftm"], prow["fta"] = m, a
                            break

                    # Derive pts from shooting if ESPN omits it
                    if prow["pts"] is None:
                        fgm = prow.get("fgm") or 0
                        tpm = prow.get("tpm") or 0
                        ftm = prow.get("ftm") or 0
                        if fgm or tpm or ftm:
                            prow["pts"] = (fgm - tpm) * 2 + tpm * 3 + ftm

                    # Derive total reb if missing
                    if prow["reb"] is None:
                        orb = prow.get("orb") or 0
                        drb = prow.get("drb") or 0
                        if orb or drb:
                            prow["reb"] = orb + drb

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
            "spread":           spread,
            "over_under":       over_under,
            "home_ml":          home_ml,
            "away_ml":          away_ml,
            "odds_provider":    odds_provider,
            "odds_details":     odds_details,
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
        "home_conference":   parsed["home"].get("conference"),
        "away_conference":   parsed["away"].get("conference"),
        # Odds — pulled from summary endpoint
        "spread":            parsed.get("spread"),
        "over_under":        parsed.get("over_under"),
        "home_ml":           parsed.get("home_ml"),
        "away_ml":           parsed.get("away_ml"),
        "odds_provider":     parsed.get("odds_provider"),
        "odds_details":      parsed.get("odds_details"),
    }

    def _make_row(side: str) -> Dict:
        me  = parsed[side]
        opp = parsed["away" if side == "home" else "home"]
        return {
            **base,
            "team_id":        me.get("team_id"),
            "team":           me.get("team"),
            "conference":     me.get("conference"),
            "conf_id":        me.get("conf_id"),
            "home_away":      side,
            "opponent_id":    opp.get("team_id"),
            "opponent":       opp.get("team"),
            "opp_conference": opp.get("conference"),
            "points_for":     me.get("score"),
            "points_against": opp.get("score"),
            "margin":         ((me.get("score") or 0) - (opp.get("score") or 0)
                               if me.get("score") is not None
                               and opp.get("score") is not None else None),
            "fgm": me.get("fgm"), "fga": me.get("fga"),
            "tpm": me.get("tpm"), "tpa": me.get("tpa"),
            "ftm": me.get("ftm"), "fta": me.get("fta"),
            "orb": me.get("orb"), "drb": me.get("drb"), "reb": me.get("reb"),
            "ast": me.get("ast"), "stl": me.get("stl"), "blk": me.get("blk"),
            "tov": me.get("tov"), "pf":  me.get("pf"),
        }

    return _make_row("home"), _make_row("away")
