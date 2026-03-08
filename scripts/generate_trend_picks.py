#!/usr/bin/env python3
"""
Generate two trend-picks CSVs:

  cbb_trend_picks_today.csv   — HIGH/MED model picks where trend aligns.
                                 Enhanced with: active_trends, agreement_level,
                                 trend_hit_pct (ATS backtest), multi_trend,
                                 tourn_seed_signal (March Madness context).

  cbb_pure_trend_picks_today.csv — ALL games with any strong trend signal,
                                    no model required. Includes: trend_pick
                                    (trend-only), vs_model, per-signal ATS
                                    backtest rates, multi-trend breakdown,
                                    tourn_seed_signal (March Madness context).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TREND_THRESHOLD = 1.5       # net rtg/game considered "trending"
CONF_ALLOWED    = {"HIGH", "MED"}
MIN_BT_GAMES    = 15        # minimum sample to report a backtest rate
L_RECENT        = 6
L_BASE          = 11

ET = ZoneInfo("America/New_York")

# ── March Madness context ─────────────────────────────────────────────────────
# Active when month is March–April and a seed ATS rates file is present.
_TOURN_RATES_PATH = Path("data/march_madness_seed_ats_rates.csv")
_TOURN_MONTHS     = {3, 4}   # March, April


# ── Trend helpers ─────────────────────────────────────────────────────────────

def _classify_active_trends(h: float, a: float) -> list[str]:
    """All trend signals currently active (independent of model direction)."""
    out = []
    if h > TREND_THRESHOLD:   out.append("HOME_UP")
    if a > TREND_THRESHOLD:   out.append("AWAY_UP")
    if h < -TREND_THRESHOLD:  out.append("HOME_DOWN")
    if a < -TREND_THRESHOLD:  out.append("AWAY_DOWN")
    return out


def _trend_aligns(edge: float, h: float, a: float) -> bool:
    """True when the dominant trend supports the model's pick direction."""
    if edge > 0:
        return h > TREND_THRESHOLD or a < -TREND_THRESHOLD
    return a > TREND_THRESHOLD or h < -TREND_THRESHOLD


def _trend_direction(h: float, a: float, edge: float) -> str:
    pick_side = "HOME" if edge > 0 else "AWAY"
    if h > TREND_THRESHOLD and a > TREND_THRESHOLD:
        return f"BOTH_UP ({pick_side} side)"
    if h < -TREND_THRESHOLD and a < -TREND_THRESHOLD:
        return f"BOTH_DOWN ({pick_side} side)"
    if h > TREND_THRESHOLD:   return "HOME_UP"
    if a > TREND_THRESHOLD:   return "AWAY_UP"
    if h < -TREND_THRESHOLD:  return "HOME_DOWN"
    if a < -TREND_THRESHOLD:  return "AWAY_DOWN"
    return "FLAT"


def _agreement_level(edge: float, signals: list[str]) -> str:
    """How strongly do the trend signals agree with the model pick?"""
    if edge > 0:   # model likes home
        supporting = [s for s in signals if s in ("HOME_UP", "AWAY_DOWN")]
    else:           # model likes away
        supporting = [s for s in signals if s in ("AWAY_UP", "HOME_DOWN")]
    if len(supporting) >= 2:
        return "STRONG AGREE"
    if len(supporting) == 1:
        return "AGREE"
    return "PARTIAL"


def _trend_side(h: float, a: float) -> str | None:
    """
    Determine the pure-trend pick side ('home' or 'away') from signal strength.
    Returns None if signals are contradictory with equal strength.
    """
    home_score = 0.0
    away_score = 0.0
    if h > TREND_THRESHOLD:    home_score += h
    if h < -TREND_THRESHOLD:   away_score += abs(h)   # fading falling home = away
    if a > TREND_THRESHOLD:    away_score += a
    if a < -TREND_THRESHOLD:   home_score += abs(a)   # fading falling away = home
    if home_score == away_score == 0:
        return None
    return "home" if home_score >= away_score else "away"


def _trend_team_pick(home: str, away: str, h: float, a: float, edge: float) -> str:
    """
    Human-readable column showing which team the trend clearly favors.

    Format: "<Team> trending (<rate>/game)" or
            "<Team> (<rate>/game) vs <Opp> fading (<rate>/game)" for double signal.
    Always reflects the pick direction (same side as model pick for aligned CSV).
    """
    if edge > 0:   # model & trend both like home
        if h > TREND_THRESHOLD and a < -TREND_THRESHOLD:
            return f"{home} ({h:+.1f}/game) — {away} fading ({a:+.1f}/game)"
        if h > TREND_THRESHOLD:
            return f"{home} trending ({h:+.1f}/game)"
        return f"{home} — fading opponent {away} ({a:+.1f}/game)"
    else:           # model & trend both like away
        if a > TREND_THRESHOLD and h < -TREND_THRESHOLD:
            return f"{away} ({a:+.1f}/game) — {home} fading ({h:+.1f}/game)"
        if a > TREND_THRESHOLD:
            return f"{away} trending ({a:+.1f}/game)"
        return f"{away} — fading opponent {home} ({h:+.1f}/game)"


def _fmt_pick(team: str, vegas_spread, is_home: bool) -> str:
    """Format pick as 'Team +/-line' or just 'Team' if no spread."""
    try:
        spread_val = float(vegas_spread)
        line = spread_val if is_home else -spread_val
        return f"{team} {line:+.1f}"
    except (TypeError, ValueError):
        return team


# ── Time conversion ───────────────────────────────────────────────────────────

def _utc_to_et(dt_str: str) -> str:
    if not dt_str or str(dt_str).strip() in ("", "nan"):
        return ""
    try:
        dt = pd.to_datetime(dt_str, utc=True)
        dt_et = dt.tz_convert(ET)
        h, m = dt_et.hour, dt_et.minute
        ampm = "p" if h >= 12 else "a"
        h12 = h % 12 or 12
        return f"{h12}:{m:02d}{ampm} ET" if m else f"{h12}:00{ampm} ET"
    except Exception:
        return ""


# ── ATS backtest ──────────────────────────────────────────────────────────────

def _compute_trend_backtest(gl_path: Path) -> dict[str, dict]:
    """
    Compute ATS cover rates for each trend signal type using team_game_weighted.csv.

    For each historical matchup:
      1. Compute L6/L11 net_rtg trends (shifted by 1 game — leak-free).
      2. Classify active signals (HOME_UP, AWAY_UP, HOME_DOWN, AWAY_DOWN).
      3. Check cover_margin (>0 means that team covered the spread).

    Signals and their "trend pick" direction:
      HOME_UP       → pick home   (rising home team covers)
      AWAY_UP       → pick away   (rising away team covers)
      HOME_DOWN     → pick away   (fade falling home)
      AWAY_DOWN     → pick home   (fade falling away)
      HOME_UP+AWAY_DOWN → strong home signal (double confirmation)
      AWAY_UP+HOME_DOWN → strong away signal (double confirmation)

    Returns {signal: {"hit_pct": float, "n_games": int}}.
    """
    if not gl_path.exists():
        return {}

    try:
        gl = pd.read_csv(gl_path, low_memory=False)
    except Exception:
        return {}

    # Column discovery
    date_col = next((c for c in ["game_datetime_utc", "game_date", "date"] if c in gl.columns), None)
    ha_col   = next((c for c in ["home_away", "is_home", "location"]        if c in gl.columns), None)
    rtg_col  = next((c for c in ["adj_net_rtg", "net_rtg"]                  if c in gl.columns), None)
    opp_col  = next((c for c in ["opponent_id", "opponent"]                  if c in gl.columns), None)
    cov_col  = "cover_margin" if "cover_margin" in gl.columns else None
    eid_col  = "event_id"     if "event_id"     in gl.columns else None

    if not all([date_col, ha_col, rtg_col, opp_col, "team_id" in gl.columns]):
        return {}

    # Coerce
    gl = gl.copy()
    gl["team_id"] = gl["team_id"].astype(str)
    gl[opp_col]   = gl[opp_col].astype(str)
    gl[rtg_col]   = pd.to_numeric(gl[rtg_col], errors="coerce")
    gl[date_col]  = pd.to_datetime(gl[date_col], errors="coerce", utc=True)
    if cov_col:
        gl[cov_col] = pd.to_numeric(gl[cov_col], errors="coerce")
    gl = gl.dropna(subset=["team_id", rtg_col, date_col]).sort_values(["team_id", date_col])

    # Compute L6/L11 rolling trend per team (shift by 1 to avoid leakage)
    chunks: list[pd.DataFrame] = []
    for _, grp in gl.groupby("team_id", sort=False):
        grp = grp.sort_values(date_col).copy()
        shifted = grp[rtg_col].shift(1)
        l6  = shifted.rolling(L_RECENT, min_periods=L_RECENT).mean()
        l11 = shifted.rolling(L_BASE,   min_periods=L_RECENT).mean()
        grp["_trend"] = (l6 - l11).values
        chunks.append(grp)

    gl = pd.concat(chunks, ignore_index=True).dropna(subset=["_trend"])

    # Separate home and away rows
    def _is_home_val(v) -> bool | None:
        s = str(v).strip().lower()
        if s in {"home", "1", "true"}:   return True
        if s in {"away", "0", "false", "road"}: return False
        return None

    gl["_is_home"] = gl[ha_col].apply(_is_home_val)
    gl = gl.dropna(subset=["_is_home"])

    home_cols = ["team_id", opp_col, "_trend", "_is_home"]
    if cov_col: home_cols.append(cov_col)
    if eid_col: home_cols.append(eid_col)

    home_df = gl[gl["_is_home"] == True][home_cols].copy()
    away_df = gl[gl["_is_home"] == False][["team_id", opp_col, "_trend"] +
                                           ([eid_col] if eid_col else [])].copy()

    home_df = home_df.rename(columns={"team_id": "h_id", opp_col: "a_id",
                                       "_trend": "h_trend",
                                       **({"cover_margin": "h_cov"} if cov_col else {})})
    away_df = away_df.rename(columns={"team_id": "a_id", opp_col: "h_id",
                                       "_trend": "a_trend"})

    # Join on event_id (cleanest) or h_id + a_id
    if eid_col:
        merged = home_df.merge(away_df, on=[eid_col, "h_id", "a_id"], how="inner")
    else:
        merged = home_df.merge(away_df, on=["h_id", "a_id"], how="inner")

    if merged.empty:
        return {}

    # Accumulate outcomes per signal
    buckets: dict[str, list[int]] = {}

    for _, row in merged.iterrows():
        h_trend = float(row["h_trend"])
        a_trend = float(row["a_trend"])
        h_cov   = float(row["h_cov"]) if cov_col and "h_cov" in row and pd.notna(row.get("h_cov")) else None

        sigs = _classify_active_trends(h_trend, a_trend)
        if not sigs:
            continue

        # Per-signal outcome: did the trend pick cover?
        for sig in sigs:
            if h_cov is None:
                continue
            if sig == "HOME_UP":      covered = int(h_cov > 0)
            elif sig == "AWAY_UP":    covered = int(h_cov < 0)
            elif sig == "HOME_DOWN":  covered = int(h_cov < 0)   # fading home
            elif sig == "AWAY_DOWN":  covered = int(h_cov > 0)   # fading away
            else:
                continue
            buckets.setdefault(sig, []).append(covered)

        # Combined signals (double confirmation)
        if h_cov is not None:
            if "HOME_UP" in sigs and "AWAY_DOWN" in sigs:
                buckets.setdefault("HOME_UP+AWAY_DOWN", []).append(int(h_cov > 0))
            if "AWAY_UP" in sigs and "HOME_DOWN" in sigs:
                buckets.setdefault("AWAY_UP+HOME_DOWN", []).append(int(h_cov < 0))

    return {
        sig: {"hit_pct": round(sum(v) / len(v) * 100, 1), "n_games": len(v)}
        for sig, v in buckets.items()
        if len(v) >= MIN_BT_GAMES
    }


def _load_tourn_seed_rates() -> dict[str, dict]:
    """
    Load March Madness seed-tier ATS rates from the pre-computed backtest CSV.
    Returns {seed_tier: {fav_cover_pct, dog_cover_pct, ref_note}} or {} if unavailable.
    Only loaded during March–April (tournament season).
    """
    if datetime.now(ET).month not in _TOURN_MONTHS:
        return {}
    if not _TOURN_RATES_PATH.exists():
        return {}
    try:
        df = pd.read_csv(_TOURN_RATES_PATH)
        result: dict[str, dict] = {}
        for _, row in df.iterrows():
            tier = str(row.get("seed_tier", "")).strip()
            if not tier:
                continue
            result[tier] = {
                "fav_cover_pct": row.get("fav_cover_pct"),
                "dog_cover_pct": row.get("dog_cover_pct"),
                "ref_note":      row.get("ref_note", row.get("interpretation", "")),
            }
        return result
    except Exception:
        return {}


def _tourn_seed_signal(
    home_team: str, away_team: str,
    home_seed, away_seed,
    edge: float,
    rates: dict[str, dict],
) -> str:
    """
    Generate a tournament seed-ATS signal string for a given matchup.
    Returns empty string if seed data or rates are unavailable.

    Format: "5 seed (home) — dog covers 59.6% ATS historically | FADE fav"
    """
    if not rates:
        return ""

    try:
        h_seed = int(home_seed) if home_seed is not None and not pd.isna(home_seed) else None
        a_seed = int(away_seed) if away_seed is not None and not pd.isna(away_seed) else None
    except (TypeError, ValueError):
        return ""

    if h_seed is None or a_seed is None:
        return ""

    fav_seed  = min(h_seed, a_seed)
    dog_seed  = max(h_seed, a_seed)
    tier      = f"{fav_seed}_vs_{dog_seed}"
    rate_info = rates.get(tier)
    if not rate_info:
        return ""

    fav_cover  = rate_info.get("fav_cover_pct")
    dog_cover  = rate_info.get("dog_cover_pct")
    ref_note   = rate_info.get("ref_note", "")

    # Determine which team is the favorite
    fav_team  = home_team if h_seed == fav_seed else away_team
    dog_team  = away_team if h_seed == fav_seed else home_team
    pick_side = "home" if edge > 0 else "away"
    pick_team = home_team if pick_side == "home" else away_team

    parts: list[str] = [f"Seed {fav_seed} vs {dog_seed}"]

    if dog_cover is not None and float(dog_cover) >= 58:
        parts.append(f"{dog_team} ({dog_seed}-seed dog) covers {dog_cover:.1f}% ATS — FADE {fav_team}")
        if pick_team == dog_team:
            parts.append("ALIGNS WITH PICK")
        else:
            parts.append("AGAINST PICK")
    elif fav_cover is not None and float(fav_cover) >= 58:
        parts.append(f"{fav_team} ({fav_seed}-seed fav) covers {fav_cover:.1f}% ATS")
        if pick_team == fav_team:
            parts.append("ALIGNS WITH PICK")
        else:
            parts.append("AGAINST PICK")
    elif ref_note:
        parts.append(ref_note)

    return " | ".join(parts)


def _backtest_for_signals(signals: list[str], backtest: dict) -> tuple[str, str]:
    """
    Return (trend_hit_pct_str, trend_hit_detail_str) for a set of active signals.
    Prefers combined signal key (HOME_UP+AWAY_DOWN etc.) over individual.
    """
    # Check combined key first
    combo_keys = []
    if "HOME_UP" in signals and "AWAY_DOWN" in signals:
        combo_keys.append("HOME_UP+AWAY_DOWN")
    if "AWAY_UP" in signals and "HOME_DOWN" in signals:
        combo_keys.append("AWAY_UP+HOME_DOWN")

    parts: list[str] = []
    best_pct: float | None = None

    for key in combo_keys + signals:
        if key in backtest:
            d = backtest[key]
            parts.append(f"{key}: {d['hit_pct']}% (n={d['n_games']})")
            if best_pct is None or d["hit_pct"] > best_pct:
                best_pct = d["hit_pct"]

    if not parts:
        return "", ""
    return (f"{best_pct:.1f}%" if best_pct else ""), " | ".join(parts)


# ── Game time ─────────────────────────────────────────────────────────────────

def _load_game_times(times_path: Path) -> dict[tuple[str, str], str]:
    game_times: dict[tuple[str, str], str] = {}
    if not times_path.exists():
        return game_times
    try:
        tj = pd.read_csv(times_path, low_memory=False,
                         usecols=["home_team", "away_team", "game_datetime_utc"])
        for _, row in tj.iterrows():
            key = (str(row["away_team"]).strip(), str(row["home_team"]).strip())
            game_times[key] = str(row.get("game_datetime_utc", ""))
    except Exception as e:
        print(f"[WARN] Could not load game times: {e}")
    return game_times


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    picks_path = Path("data/cbb_picks_today.csv")
    if not picks_path.exists():
        print("[STOP] data/cbb_picks_today.csv not found")
        return 1

    picks = pd.read_csv(picks_path, low_memory=False)
    if picks.empty:
        print("[STOP] cbb_picks_today.csv is empty")
        return 1

    # ── Backtest ──────────────────────────────────────────────────────────────
    gl_path = Path("data/team_game_weighted.csv")
    backtest = _compute_trend_backtest(gl_path)
    if backtest:
        print(f"[INFO] Trend backtest computed: {len(backtest)} signal types")
        for sig, d in sorted(backtest.items()):
            print(f"  {sig}: {d['hit_pct']}% ATS (n={d['n_games']})")
    else:
        print("[WARN] Trend backtest unavailable — cover_margin or game log missing")

    # ── March Madness seed context ─────────────────────────────────────────────
    tourn_rates = _load_tourn_seed_rates()
    if tourn_rates:
        print(f"[INFO] Tournament seed ATS rates loaded: {len(tourn_rates)} tiers")
    else:
        print("[INFO] Tournament seed context inactive (non-tournament month or no rates file)")

    # ── Coerce columns ────────────────────────────────────────────────────────
    for col in ["netrtg_trend_home", "netrtg_trend_away"]:
        if col not in picks.columns:
            picks[col] = 0.0
    picks["netrtg_trend_home"] = pd.to_numeric(picks["netrtg_trend_home"], errors="coerce").fillna(0.0)
    picks["netrtg_trend_away"] = pd.to_numeric(picks["netrtg_trend_away"], errors="coerce").fillna(0.0)
    picks["spread_edge"]       = pd.to_numeric(picks["spread_edge"],       errors="coerce").fillna(0.0)

    # Seed columns may or may not be present in picks
    for col in ["home_seed", "away_seed"]:
        if col not in picks.columns:
            picks[col] = None

    game_times = _load_game_times(Path("data/predictions_joint_latest.csv"))

    # ═══════════════════════════════════════════════════════════════════════════
    # CSV 1: cbb_trend_picks_today.csv (model + trend aligned)
    # ═══════════════════════════════════════════════════════════════════════════
    model_rows: list[dict] = []

    for _, row in picks.iterrows():
        conf = str(row.get("spread_conf", ""))
        if conf not in CONF_ALLOWED:
            continue
        if str(row.get("model_pick", "PASS")).strip() == "PASS":
            continue
        if not str(row.get("trend_flag", "")).strip():
            continue

        edge = float(row["spread_edge"])
        h    = float(row["netrtg_trend_home"])
        a    = float(row["netrtg_trend_away"])

        if not _trend_aligns(edge, h, a):
            continue

        away = str(row.get("away_team", ""))
        home = str(row.get("home_team", ""))
        game_time_et = _utc_to_et(game_times.get((away.strip(), home.strip()), ""))

        trend_strength = max(abs(h), abs(a))
        direction      = _trend_direction(h, a, edge)
        signals        = _classify_active_trends(h, a)
        active_str     = " + ".join(signals) if signals else direction
        agree_level    = _agreement_level(edge, signals)
        multi_trend    = len(signals) >= 2

        hit_pct_str, hit_detail = _backtest_for_signals(signals, backtest)

        total_pick = str(row.get("total_pick", "PASS")).strip()
        total_conf = str(row.get("total_conf", ""))
        total_edge = row.get("total_edge", np.nan)

        model_pick    = str(row.get("model_pick", ""))
        pick_tag      = f"{model_pick} [{conf} {edge:+.1f}]"
        trend_flag_tx = str(row.get("trend_flag", ""))
        key_signal_tx = str(row.get("key_signal", ""))

        tourn_sig = _tourn_seed_signal(
            home, away,
            row.get("home_seed"), row.get("away_seed"),
            edge, tourn_rates,
        )

        model_rows.append({
            "game_date":              str(row.get("game_date", ""))[:10],
            "game_time_et":           game_time_et,
            "away_team":              away,
            "home_team":              home,
            "vegas_spread":           row.get("vegas_spread", np.nan),
            "model_pick":             model_pick,
            "spread_edge":            round(edge, 1),
            "spread_conf":            conf,
            "spread_prob":            row.get("spread_prob", np.nan),
            "model_predicted_margin": row.get("model_predicted_margin", np.nan),
            "agreement_level":        agree_level,
            "trend_team_pick":        _trend_team_pick(home, away, h, a, edge),
            "trend_direction":        direction,
            "active_trends":          active_str,
            "multi_trend":            multi_trend,
            "trend_strength":         round(trend_strength, 1),
            "netrtg_trend_home":      round(h, 2),
            "netrtg_trend_away":      round(a, 2),
            "trend_hit_pct":          hit_pct_str,
            "trend_hit_detail":       hit_detail,
            "tourn_seed_signal":      tourn_sig,
            "trend_flag":             trend_flag_tx,
            "trend_flag_pick":        f"{trend_flag_tx} → {pick_tag}" if trend_flag_tx else pick_tag,
            "key_signal":             key_signal_tx,
            "key_signal_pick":        f"{key_signal_tx} → {pick_tag}" if key_signal_tx else pick_tag,
            "total_pick":             total_pick if total_pick != "PASS" else "",
            "total_conf":             total_conf if total_pick != "PASS" else "",
            "total_edge":             round(float(total_edge), 1) if pd.notna(total_edge) and total_pick != "PASS" else np.nan,
        })

    _EMPTY_MODEL_COLS = [
        "game_date", "game_time_et", "away_team", "home_team", "vegas_spread",
        "model_pick", "spread_edge", "spread_conf", "spread_prob",
        "model_predicted_margin", "agreement_level", "trend_team_pick",
        "trend_direction", "active_trends", "multi_trend", "trend_strength",
        "netrtg_trend_home", "netrtg_trend_away",
        "trend_hit_pct", "trend_hit_detail", "tourn_seed_signal",
        "trend_flag", "trend_flag_pick", "key_signal", "key_signal_pick",
        "total_pick", "total_conf", "total_edge",
    ]

    if not model_rows:
        print("[WARN] No trend-aligned model picks — writing empty file")
        pd.DataFrame(columns=_EMPTY_MODEL_COLS).to_csv(
            "data/cbb_trend_picks_today.csv", index=False
        )
    else:
        out_model = pd.DataFrame(model_rows)
        out_model["_cr"] = out_model["spread_conf"].map({"HIGH": 2, "MED": 1}).fillna(0)
        out_model = (out_model.sort_values(["agreement_level", "_cr", "trend_strength"],
                                           ascending=[True, False, False])
                              .drop(columns="_cr"))
        # Sort agreement_level so STRONG AGREE comes first
        level_order = {"STRONG AGREE": 0, "AGREE": 1, "PARTIAL": 2}
        out_model["_lvl"] = out_model["agreement_level"].map(level_order).fillna(3)
        out_model = out_model.sort_values(["_lvl", "trend_strength"], ascending=[True, False]).drop(columns="_lvl")
        out_model.to_csv("data/cbb_trend_picks_today.csv", index=False)

        high_n   = (out_model["spread_conf"] == "HIGH").sum()
        med_n    = (out_model["spread_conf"] == "MED").sum()
        strong_n = (out_model["agreement_level"] == "STRONG AGREE").sum()
        print(f"[OK] cbb_trend_picks_today.csv: {len(out_model)} picks "
              f"({high_n} HIGH, {med_n} MED | {strong_n} STRONG AGREE)")

    # ═══════════════════════════════════════════════════════════════════════════
    # CSV 2: cbb_pure_trend_picks_today.csv (trend-only, all games)
    # ═══════════════════════════════════════════════════════════════════════════
    pure_rows: list[dict] = []

    for _, row in picks.iterrows():
        h = float(pd.to_numeric(row.get("netrtg_trend_home", 0), errors="coerce") or 0)
        a = float(pd.to_numeric(row.get("netrtg_trend_away", 0), errors="coerce") or 0)

        signals = _classify_active_trends(h, a)
        if not signals:
            continue   # no trend signal at all — skip

        away = str(row.get("away_team", ""))
        home = str(row.get("home_team", ""))
        game_time_et = _utc_to_et(game_times.get((away.strip(), home.strip()), ""))

        side = _trend_side(h, a)
        if side is None:
            continue   # contradictory signals of equal strength

        vegas_spread = row.get("vegas_spread", np.nan)
        trend_pick   = _fmt_pick(home if side == "home" else away,
                                  vegas_spread,
                                  is_home=(side == "home"))

        active_str   = " + ".join(signals)
        multi_trend  = len(signals) >= 2
        trend_conf   = "STRONG" if multi_trend else "ALIGNED"

        hit_pct_str, hit_detail = _backtest_for_signals(signals, backtest)

        # vs_model: does trend agree with model pick?
        model_pick = str(row.get("model_pick", "PASS")).strip()
        if model_pick == "PASS":
            vs_model = "NO_MODEL_PICK"
        else:
            edge = float(pd.to_numeric(row.get("spread_edge", 0), errors="coerce") or 0)
            trend_agrees = _trend_aligns(edge, h, a)
            vs_model = "AGREES" if trend_agrees else "DISAGREES"

        # trend_team_pick for pure CSV uses edge=0 proxy: side determines direction
        pure_edge = 1.0 if side == "home" else -1.0
        pure_edge_for_model = float(
            pd.to_numeric(row.get("spread_edge", 0), errors="coerce") or 0
        )
        tourn_sig = _tourn_seed_signal(
            home, away,
            row.get("home_seed"), row.get("away_seed"),
            pure_edge, tourn_rates,
        )
        pure_rows.append({
            "game_date":         str(row.get("game_date", ""))[:10],
            "game_time_et":      game_time_et,
            "away_team":         away,
            "home_team":         home,
            "vegas_spread":      vegas_spread,
            "trend_pick":        trend_pick,
            "trend_team_pick":   _trend_team_pick(home, away, h, a, pure_edge),
            "trend_conf":        trend_conf,
            "active_trends":     active_str,
            "multi_trend":       multi_trend,
            "trend_hit_pct":     hit_pct_str,
            "trend_hit_detail":  hit_detail,
            "tourn_seed_signal": tourn_sig,
            "netrtg_trend_home": round(h, 2),
            "netrtg_trend_away": round(a, 2),
            "vs_model":          vs_model,
            "model_pick":        model_pick if model_pick != "PASS" else "",
            "spread_conf":       str(row.get("spread_conf", "")),
            "spread_edge":       round(pure_edge_for_model, 1),
        })

    _EMPTY_PURE_COLS = [
        "game_date", "game_time_et", "away_team", "home_team", "vegas_spread",
        "trend_pick", "trend_team_pick", "trend_conf", "active_trends", "multi_trend",
        "trend_hit_pct", "trend_hit_detail", "tourn_seed_signal",
        "netrtg_trend_home", "netrtg_trend_away",
        "vs_model", "model_pick", "spread_conf", "spread_edge",
    ]

    if not pure_rows:
        print("[WARN] No pure trend picks found — writing empty file")
        pd.DataFrame(columns=_EMPTY_PURE_COLS).to_csv(
            "data/cbb_pure_trend_picks_today.csv", index=False
        )
    else:
        out_pure = pd.DataFrame(pure_rows)
        # Sort: STRONG first, then by number of active signals, then hit_pct desc
        out_pure["_sig_count"] = out_pure["active_trends"].str.count(r"\+") + 1
        out_pure["_hit_sort"]  = pd.to_numeric(
            out_pure["trend_hit_pct"].str.rstrip("%"), errors="coerce"
        ).fillna(0)
        out_pure = (out_pure.sort_values(
                        ["trend_conf", "_sig_count", "_hit_sort"],
                        ascending=[True, False, False])   # True = STRONG before ALIGNED (alpha)
                            .drop(columns=["_sig_count", "_hit_sort"]))
        # Fix sort: STRONG before ALIGNED
        conf_order = {"STRONG": 0, "ALIGNED": 1}
        out_pure["_co"] = out_pure["trend_conf"].map(conf_order).fillna(2)
        out_pure = out_pure.sort_values(["_co", "trend_hit_pct"],
                                         ascending=[True, False]).drop(columns="_co")
        out_pure.to_csv("data/cbb_pure_trend_picks_today.csv", index=False)

        strong_n = (out_pure["trend_conf"] == "STRONG").sum()
        agrees_n = (out_pure["vs_model"] == "AGREES").sum()
        print(f"[OK] cbb_pure_trend_picks_today.csv: {len(out_pure)} trend picks "
              f"({strong_n} STRONG | {agrees_n} agree with model)")

    # ── Gate ─────────────────────────────────────────────────────────────────
    out_check = pd.read_csv("data/cbb_trend_picks_today.csv")
    gate = {
        "has_active_trends_col": "active_trends" in out_check.columns,
        "has_agreement_level":   "agreement_level" in out_check.columns,
        "has_trend_hit_pct":     "trend_hit_pct" in out_check.columns,
        "pure_trend_csv_exists": Path("data/cbb_pure_trend_picks_today.csv").exists(),
    }
    print("=== GATE_TREND RESULTS ===")
    for check, result in gate.items():
        print(f"  {'PASS' if result else 'FAIL'}  {check}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
