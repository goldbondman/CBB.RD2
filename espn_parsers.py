"""
ESPN CBB Pipeline — Parsers
Convert raw ESPN JSON into clean flat dictionaries.
One function per data shape. No I/O here.
"""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

TZ_PST = ZoneInfo("America/Los_Angeles")


def _utc_to_pst(utc_str: str) -> Optional[str]:
    if not utc_str:
        return None
    try:
        dt_utc = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        return dt_utc.astimezone(TZ_PST).strftime("%Y-%m-%d %I:%M %p %Z")
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
    """Parse made/attempt strings like '7-13', '7/13', or '7 of 13'."""
    if s is None:
        return None, None
    text = str(s).strip().lower()
    if not text or text in ("--", "-"):
        return None, None

    try:
        normalized = (
            text.replace(" of ", "-")
                .replace("/", "-")
                .replace("\u2212", "-")
        )
        parts = [p.strip() for p in normalized.split("-") if p.strip() != ""]
        if len(parts) == 2:
            return _safe_int(parts[0]), _safe_int(parts[1])
    except Exception:
        pass
    return None, None


def _parse_minutes(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in ("", "--", "-"):
        return None
    try:
        if ":" in s:
            parts = s.split(":")
            return round(int(parts[0]) + int(parts[1]) / 60, 1)
        return float(s)
    except Exception:
        return None


def _parse_record(record_obj: Any) -> Tuple[Optional[int], Optional[int]]:
    """Parse ESPN record object or 'W-L' string → (wins, losses)."""
    if not record_obj:
        return None, None
    # Sometimes it's a dict with summary key like "15-8"
    if isinstance(record_obj, dict):
        summary = record_obj.get("summary", "")
    else:
        summary = str(record_obj)
    try:
        parts = summary.strip().split("-")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return None, None


# ── Stat label normalization ──────────────────────────────────────────────────

PLAYER_STAT_MAP = {
    "min": "min", "minutes": "min",
    "pts": "pts", "points": "pts",
    "fg": "_fg", "fgm-a": "_fg", "field goals": "_fg",
    "3pt": "_3pt", "3p": "_3pt", "3-pt": "_3pt", "3pm-a": "_3pt",
    "three pointers": "_3pt",
    "ft": "_ft", "ftm-a": "_ft", "free throws": "_ft",
    "oreb": "orb", "or": "orb", "offensive rebounds": "orb",
    "dreb": "drb", "dr": "drb", "defensive rebounds": "drb",
    "reb": "reb", "tr": "reb", "rebounds": "reb", "total rebounds": "reb",
    "ast": "ast", "a": "ast", "assists": "ast",
    "stl": "stl", "s": "stl", "steals": "stl",
    "blk": "blk", "b": "blk", "blocks": "blk",
    "to": "tov", "tov": "tov", "turnovers": "tov",
    "pf": "pf", "fouls": "pf", "personal fouls": "pf",
    "+/-": "plus_minus", "plusminus": "plus_minus",
}


# ── Scoreboard parser ─────────────────────────────────────────────────────────

def parse_scoreboard_event(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse one ESPN scoreboard event into a flat game row."""
    try:
        game_id = str(event.get("id", "")).strip()
        if not game_id:
            return None

        comps = event.get("competitions", [])
        if not comps:
            return None
        comp = comps[0]

        status_type = comp.get("status", {}).get("type", {})
        completed   = bool(status_type.get("completed", False))
        state       = status_type.get("state", "")
        status_desc = status_type.get("description", "")
        game_dt_utc = comp.get("date", event.get("date", ""))
        venue       = comp.get("venue", {}).get("fullName", "")
        neutral_site = bool(comp.get("neutralSite", False))

        # ── Competitors ──
        home = away = {}
        for c in comp.get("competitors", []):
            ha = c.get("homeAway", "").lower()
            if ha == "home":
                home = c
            elif ha == "away":
                away = c

        if not home or not away:
            return None

        def _comp_fields(c: Dict, prefix: str) -> Dict:
            team     = c.get("team", {})
            conf_obj = team.get("conference", {})
            # Records — ESPN provides multiple record types
            records  = {r.get("type", ""): r for r in c.get("records", [])}
            overall  = records.get("total", records.get("", {}))
            home_rec = records.get("home", {})
            away_rec = records.get("road", records.get("away", {}))
            conf_rec = records.get("vsconf", records.get("conference", {}))

            ow, ol = _parse_record(overall.get("summary", ""))
            hw, hl = _parse_record(home_rec.get("summary", ""))
            aw, al = _parse_record(away_rec.get("summary", ""))
            cw, cl = _parse_record(conf_rec.get("summary", ""))

            # AP/Coaches ranking
            rank = c.get("curatedRank", {}).get("current")
            rank = _safe_int(rank) if rank and str(rank) != "99" else None

            return {
                f"{prefix}team":        team.get("displayName", team.get("name", "")),
                f"{prefix}team_id":     str(team.get("id", "")).strip(),
                f"{prefix}conference":  conf_obj.get("name", conf_obj.get("shortName",
                                        str(team.get("conferenceId", "")))),
                f"{prefix}conf_id":     str(team.get("conferenceId", "")),
                f"{prefix}score":       _safe_int(c.get("score")),
                f"{prefix}rank":        rank,
                f"{prefix}wins":        ow,
                f"{prefix}losses":      ol,
                f"{prefix}home_wins":   hw,
                f"{prefix}home_losses": hl,
                f"{prefix}away_wins":   aw,
                f"{prefix}away_losses": al,
                f"{prefix}conf_wins":   cw,
                f"{prefix}conf_losses": cl,
            }

        home_fields = _comp_fields(home, "home_")
        away_fields = _comp_fields(away, "away_")

        # ── Period scores ──
        # linescores on each competitor give per-half/OT scores
        home_periods = [_safe_int(ls.get("value")) for ls in home.get("linescores", [])]
        away_periods = [_safe_int(ls.get("value")) for ls in away.get("linescores", [])]

        period_data: Dict = {}
        for i, (hp, ap) in enumerate(zip(home_periods, away_periods), 1):
            label = f"h{i}" if i <= 2 else f"ot{i-2}"
            period_data[f"home_{label}"] = hp
            period_data[f"away_{label}"] = ap

        # ── Odds ──
        odds_obj      = (comp.get("odds") or [{}])[0]
        spread        = _safe_float(odds_obj.get("spread"))
        over_under    = _safe_float(odds_obj.get("overUnder"))
        home_ml       = _safe_float(odds_obj.get("homeTeamOdds", {}).get("moneyLine"))
        away_ml       = _safe_float(odds_obj.get("awayTeamOdds", {}).get("moneyLine"))
        odds_provider = odds_obj.get("provider", {}).get("name", "")
        odds_details  = odds_obj.get("details", "")

        return {
            "game_id":           game_id,
            "game_datetime_utc": game_dt_utc,
            "game_datetime_pst": _utc_to_pst(game_dt_utc),
            "venue":             venue,
            "neutral_site":      neutral_site,
            "completed":         completed,
            "state":             state,
            "status_desc":       status_desc,
            "spread":            spread,
            "over_under":        over_under,
            "home_ml":           home_ml,
            "away_ml":           away_ml,
            "odds_provider":     odds_provider,
            "odds_details":      odds_details,
            **home_fields,
            **away_fields,
            **period_data,
        }

    except Exception as exc:
        log.warning(f"parse_scoreboard_event failed for {event.get('id','?')}: {exc}")
        return None


# ── Summary parser ────────────────────────────────────────────────────────────

def parse_summary(raw: Dict[str, Any], event_id: str) -> Optional[Dict[str, Any]]:
    """Parse ESPN summary JSON into structured game + team + player dicts."""
    try:
        header   = raw.get("header", {})
        boxscore = raw.get("boxscore", {})
        comps    = header.get("competitions", [])
        comp     = comps[0] if comps else {}

        game_dt_utc  = comp.get("date", "")
        neutral_site = bool(comp.get("neutralSite", False))
        venue        = comp.get("venue", {}).get("fullName", "")
        status_type  = comp.get("status", {}).get("type", {})
        completed    = bool(status_type.get("completed", False))
        state        = status_type.get("state", "")
        period       = comp.get("status", {}).get("period", 2)
        is_ot        = period > 2
        num_ot       = max(0, period - 2) if is_ot else 0

        # ── Odds ──
        odds_obj      = (comp.get("odds") or [{}])[0]
        spread        = _safe_float(odds_obj.get("spread"))
        over_under    = _safe_float(odds_obj.get("overUnder"))
        home_ml       = _safe_float(odds_obj.get("homeTeamOdds", {}).get("moneyLine"))
        away_ml       = _safe_float(odds_obj.get("awayTeamOdds", {}).get("moneyLine"))
        odds_provider = odds_obj.get("provider", {}).get("name", "")
        odds_details  = odds_obj.get("details", "")

        # ── Competitors ──
        home_meta = away_meta = {}
        for c in comp.get("competitors", []):
            if c.get("homeAway", "").lower() == "home":
                home_meta = c
            elif c.get("homeAway", "").lower() == "away":
                away_meta = c

        def _team_meta(meta: Dict) -> Dict:
            team     = meta.get("team", {})
            conf_obj = team.get("conference", {})
            records  = {r.get("type", ""): r for r in meta.get("records", [])}
            overall  = records.get("total", records.get("", {}))
            home_rec = records.get("home", {})
            away_rec = records.get("road", records.get("away", {}))
            conf_rec = records.get("vsconf", records.get("conference", {}))

            ow, ol = _parse_record(overall.get("summary", ""))
            hw, hl = _parse_record(home_rec.get("summary", ""))
            aw, al = _parse_record(away_rec.get("summary", ""))
            cw, cl = _parse_record(conf_rec.get("summary", ""))

            rank = meta.get("curatedRank", {}).get("current")
            rank = _safe_int(rank) if rank and str(rank) != "99" else None

            # Per-period scores (linescores)
            periods = [_safe_int(ls.get("value")) for ls in meta.get("linescores", [])]

            return {
                "team_id":    str(team.get("id", "")).strip(),
                "team":       team.get("displayName", team.get("name", "")),
                "conference": conf_obj.get("name", conf_obj.get("shortName",
                              str(team.get("conferenceId", "")))),
                "conf_id":    str(team.get("conferenceId", "")),
                "score":      _safe_int(meta.get("score")),
                "home_away":  meta.get("homeAway", "").lower(),
                "rank":       rank,
                "wins":       ow,   "losses":      ol,
                "home_wins":  hw,   "home_losses": hl,
                "away_wins":  aw,   "away_losses": al,
                "conf_wins":  cw,   "conf_losses": cl,
                "periods":    periods,  # list: [h1, h2, ot1, ot2, ...]
            }

        home_info = _team_meta(home_meta)
        away_info = _team_meta(away_meta)

        # ── Box score team stats ──
        def _parse_team_stats(teams_list: List, target_id: str) -> Dict:
            stats: Dict[str, Any] = {}
            for t in teams_list:
                if str(t.get("team", {}).get("id", "")).strip() != target_id:
                    continue
                for s in t.get("statistics", []):
                    name  = s.get("name", "")
                    label = s.get("abbreviation", name).lower().replace(" ", "_")
                    dv    = s.get("displayValue", "")
                    if "-" in str(dv) and any(x in name.lower() for x in
                                              ["goal", "throw", "three"]):
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
                            "reb": "reb",  "rebounds": "reb",
                            "totalrebounds": "reb",
                            "ast": "ast",  "assists": "ast",
                            "to": "tov",   "turnovers": "tov",
                            "stl": "stl",  "steals": "stl",
                            "blk": "blk",  "blocks": "blk",
                            "pf": "pf",    "fouls": "pf",
                        }
                        key = (col_map.get(label) or
                               col_map.get(name.lower().replace(" ", ""), label))
                        val_p = (_safe_int(dv) if str(dv).isdigit()
                                 else _safe_float(dv))
                        if key == "reb" and ("orb" in stats or "drb" in stats):
                            stats.setdefault("reb", val_p)
                        else:
                            stats[key] = val_p
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
                stat_labels = [k.lower().strip() for k in pg.get("keys", [])]

                for ae in pg.get("athletes", []):
                    athlete      = ae.get("athlete", {})
                    vals         = ae.get("stats", [])
                    did_not_play = bool(ae.get("didNotPlay", False))

                    if did_not_play and not any(
                            v not in ("", "--", "-", "0") for v in vals):
                        continue

                    athlete_id = str(athlete.get("id", "")).strip()
                    if not athlete_id:
                        continue

                    prow: Dict[str, Any] = {
                        "event_id":          event_id,
                        "game_datetime_utc":  game_dt_utc,
                        "game_datetime_pst":  _utc_to_pst(game_dt_utc),
                        "team_id":           tid,
                        "team":              tname,
                        "home_away":         ha,
                        "athlete_id":        athlete_id,
                        "player":            athlete.get("displayName", ""),
                        "jersey":            athlete.get("jersey", ""),
                        "position":          athlete.get("position", {}).get(
                                             "abbreviation", ""),
                        "starter":           bool(ae.get("starter", False)),
                        "did_not_play":      did_not_play,
                        "min": None, "pts": None,
                        "fgm": None, "fga": None,
                        "tpm": None, "tpa": None,
                        "ftm": None, "fta": None,
                        "orb": None, "drb": None, "reb": None,
                        "ast": None, "stl": None, "blk": None,
                        "tov": None, "pf":  None,
                        "plus_minus": None,
                    }

                    raw_stats: Dict[str, str] = {}
                    for lbl, val in zip(stat_labels, vals):
                        raw_stats[lbl] = val
                        mapped = PLAYER_STAT_MAP.get(lbl)
                        if mapped is None:
                            continue
                        if mapped == "min":
                            prow["min"] = _parse_minutes(val)
                        elif not mapped.startswith("_"):
                            cleaned = str(val).strip()
                            prow[mapped] = (None if cleaned in ("", "--", "-")
                                           else (_safe_int(cleaned)
                                                 if cleaned.isdigit()
                                                 else _safe_float(cleaned)))

                    # Shooting splits — independent loops per type
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

                    # Derive pts/reb if ESPN omits them
                    if prow["pts"] is None:
                        fgm = prow.get("fgm") or 0
                        tpm = prow.get("tpm") or 0
                        ftm = prow.get("ftm") or 0
                        if fgm or tpm or ftm:
                            prow["pts"] = (fgm - tpm) * 2 + tpm * 3 + ftm
                    if prow["reb"] is None:
                        orb = prow.get("orb") or 0
                        drb = prow.get("drb") or 0
                        if orb or drb:
                            prow["reb"] = orb + drb

                    # Compatibility aliases expected by some downstream consumers
                    prow["FGA"] = prow.get("fga")
                    prow["FGM"] = prow.get("fgm")
                    prow["FTA"] = prow.get("fta")
                    prow["FTM"] = prow.get("ftm")
                    prow["TPA"] = prow.get("tpa")
                    prow["TPM"] = prow.get("tpm")
                    prow["ORB"] = prow.get("orb")
                    prow["DRB"] = prow.get("drb")
                    prow["RB"] = prow.get("reb")
                    prow["TO"] = prow.get("tov")
                    prow["AST"] = prow.get("ast")

                    players.append(prow)

        return {
            "event_id":          event_id,
            "game_datetime_utc":  game_dt_utc,
            "game_datetime_pst":  _utc_to_pst(game_dt_utc),
            "venue":             venue,
            "neutral_site":      neutral_site,
            "completed":         completed,
            "state":             state,
            "is_ot":             is_ot,
            "num_ot":            num_ot,
            "spread":            spread,
            "over_under":        over_under,
            "home_ml":           home_ml,
            "away_ml":           away_ml,
            "odds_provider":     odds_provider,
            "odds_details":      odds_details,
            "home":              {**home_info, **home_stats},
            "away":              {**away_info, **away_stats},
            "players":           players,
        }

    except Exception as exc:
        log.error(f"parse_summary failed for event {event_id}: {exc}")
        return None


def summary_to_team_rows(parsed: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """Convert parsed summary into two flat team-game rows (home + away)."""

    def _period_cols(side_info: Dict, prefix: str) -> Dict:
        """Flatten per-period scores into named columns."""
        cols = {}
        for i, score in enumerate(side_info.get("periods", []), 1):
            label = f"{prefix}h{i}" if i <= 2 else f"{prefix}ot{i-2}"
            cols[label] = score
        return cols

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
        "home_rank":         parsed["home"].get("rank"),
        "away_rank":         parsed["away"].get("rank"),
        "spread":            parsed.get("spread"),
        "over_under":        parsed.get("over_under"),
        "home_ml":           parsed.get("home_ml"),
        "away_ml":           parsed.get("away_ml"),
        "odds_provider":     parsed.get("odds_provider"),
        "odds_details":      parsed.get("odds_details"),
        # Period scores
        **_period_cols(parsed["home"], "home_"),
        **_period_cols(parsed["away"], "away_"),
    }

    def _make_row(side: str) -> Dict:
        me  = parsed[side]
        opp = parsed["away" if side == "home" else "home"]
        periods = me.get("periods", [])
        h1 = periods[0] if len(periods) > 0 else None
        h2 = periods[1] if len(periods) > 1 else None

        return {
            **base,
            "team_id":          me.get("team_id"),
            "team":             me.get("team"),
            "conference":       me.get("conference"),
            "conf_id":          me.get("conf_id"),
            "home_away":        side,
            "rank":             me.get("rank"),
            "wins":             me.get("wins"),
            "losses":           me.get("losses"),
            "home_wins":        me.get("home_wins"),
            "home_losses":      me.get("home_losses"),
            "away_wins":        me.get("away_wins"),
            "away_losses":      me.get("away_losses"),
            "conf_wins":        me.get("conf_wins"),
            "conf_losses":      me.get("conf_losses"),
            "win_pct":          round(me["wins"] / (me["wins"] + me["losses"]), 3)
                                if me.get("wins") is not None
                                and (me.get("wins", 0) + me.get("losses", 0)) > 0
                                else None,
            "opponent_id":      opp.get("team_id"),
            "opponent":         opp.get("team"),
            "opp_conference":   opp.get("conference"),
            "opp_rank":         opp.get("rank"),
            "opp_wins":         opp.get("wins"),
            "opp_losses":       opp.get("losses"),
            "points_for":       me.get("score"),
            "points_against":   opp.get("score"),
            "margin":           ((me.get("score") or 0) - (opp.get("score") or 0)
                                 if me.get("score") is not None
                                 and opp.get("score") is not None else None),
            "h1_pts":           h1,
            "h2_pts":           h2,
            "h1_pts_against":   opp.get("periods", [None])[0] if opp.get("periods") else None,
            "h2_pts_against":   opp.get("periods", [None, None])[1]
                                if opp.get("periods") and len(opp.get("periods")) > 1 else None,
            "fgm": me.get("fgm"), "fga": me.get("fga"),
            "tpm": me.get("tpm"), "tpa": me.get("tpa"),
            "ftm": me.get("ftm"), "fta": me.get("fta"),
            "orb": me.get("orb"), "drb": me.get("drb"), "reb": me.get("reb"),
            "ast": me.get("ast"), "stl": me.get("stl"), "blk": me.get("blk"),
            "tov": me.get("tov"), "pf":  me.get("pf"),
            # Opponent box score — used by espn_prediction_runner to build
            # accurate GameData objects for the bidirectional prediction model.
            # These are the OPPONENT'S actual stats from the same game.
            "opp_fgm": opp.get("fgm"), "opp_fga": opp.get("fga"),
            "opp_tpm": opp.get("tpm"), "opp_tpa": opp.get("tpa"),
            "opp_ftm": opp.get("ftm"), "opp_fta": opp.get("fta"),
            "opp_orb": opp.get("orb"), "opp_drb": opp.get("drb"),
            "opp_tov": opp.get("tov"), "opp_pf":  opp.get("pf"),
            # Compatibility aliases expected by some downstream consumers
            "FGA": me.get("fga"), "FGM": me.get("fgm"),
            "FTA": me.get("fta"), "FTM": me.get("ftm"),
            "TPA": me.get("tpa"), "TPM": me.get("tpm"),
            "ORB": me.get("orb"), "DRB": me.get("drb"), "RB": me.get("reb"),
            "TO": me.get("tov"), "AST": me.get("ast"),
        }

    return _make_row("home"), _make_row("away")
