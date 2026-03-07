#!/usr/bin/env python3
"""Generate cbb_trend_picks_today.csv: HIGH/MED picks where trend direction aligns with model."""

from __future__ import annotations

import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

TREND_THRESHOLD = 1.5   # net rtg/game considered "trending"
CONF_ALLOWED = {"HIGH", "MED"}

ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _trend_aligns(edge: float, h: float, a: float) -> bool:
    """True when the dominant trend supports the model's pick direction."""
    if edge > 0:
        # Model likes home: home trending up OR away falling
        return h > TREND_THRESHOLD or a < -TREND_THRESHOLD
    else:
        # Model likes away: away trending up OR home falling
        return a > TREND_THRESHOLD or h < -TREND_THRESHOLD


def _trend_direction(h: float, a: float, edge: float) -> str:
    pick_side = "HOME" if edge > 0 else "AWAY"
    if h > TREND_THRESHOLD and a > TREND_THRESHOLD:
        return f"BOTH_UP ({pick_side} side)"
    if h < -TREND_THRESHOLD and a < -TREND_THRESHOLD:
        return f"BOTH_DOWN ({pick_side} side)"
    if h > TREND_THRESHOLD:
        return "HOME_UP"
    if a > TREND_THRESHOLD:
        return "AWAY_UP"
    if h < -TREND_THRESHOLD:
        return "HOME_DOWN"
    if a < -TREND_THRESHOLD:
        return "AWAY_DOWN"
    return "FLAT"


def _utc_to_et(dt_str: str) -> str:
    """Convert UTC datetime string to ET time string like '12:00p ET'."""
    if not dt_str or str(dt_str).strip() in ("", "nan"):
        return ""
    try:
        dt = pd.to_datetime(dt_str, utc=True)
        dt_et = dt.tz_convert(ET)
        hour = dt_et.hour
        minute = dt_et.minute
        ampm = "p" if hour >= 12 else "a"
        hour12 = hour % 12 or 12
        if minute:
            return f"{hour12}:{minute:02d}{ampm} ET"
        return f"{hour12}:00{ampm} ET"
    except Exception:
        return ""


def main() -> int:
    picks_path = Path("data/cbb_picks_today.csv")
    if not picks_path.exists():
        print("[STOP] data/cbb_picks_today.csv not found")
        return 1

    picks = pd.read_csv(picks_path, low_memory=False)
    if picks.empty:
        print("[STOP] cbb_picks_today.csv is empty")
        return 1

    # Ensure trend columns exist (may be missing from older runs)
    for col in ["netrtg_trend_home", "netrtg_trend_away"]:
        if col not in picks.columns:
            picks[col] = 0.0

    picks["netrtg_trend_home"] = pd.to_numeric(picks["netrtg_trend_home"], errors="coerce").fillna(0.0)
    picks["netrtg_trend_away"] = pd.to_numeric(picks["netrtg_trend_away"], errors="coerce").fillna(0.0)
    picks["spread_edge"] = pd.to_numeric(picks["spread_edge"], errors="coerce").fillna(0.0)

    # Join game times from predictions_joint_latest
    game_times: dict[tuple[str, str], str] = {}
    times_path = Path("data/predictions_joint_latest.csv")
    if times_path.exists():
        try:
            tj = pd.read_csv(times_path, low_memory=False, usecols=["home_team", "away_team", "game_datetime_utc"])
            for _, row in tj.iterrows():
                key = (str(row["away_team"]).strip(), str(row["home_team"]).strip())
                game_times[key] = str(row.get("game_datetime_utc", ""))
        except Exception as e:
            print(f"[WARN] Could not load game times: {e}")

    rows: list[dict] = []
    for _, row in picks.iterrows():
        conf = str(row.get("spread_conf", ""))
        if conf not in CONF_ALLOWED:
            continue
        if str(row.get("model_pick", "PASS")).strip() == "PASS":
            continue
        if not str(row.get("trend_flag", "")).strip():
            continue

        edge = float(row["spread_edge"])
        h = float(row["netrtg_trend_home"])
        a = float(row["netrtg_trend_away"])

        if not _trend_aligns(edge, h, a):
            continue

        away = str(row.get("away_team", ""))
        home = str(row.get("home_team", ""))
        utc_dt = game_times.get((away.strip(), home.strip()), "")
        game_time_et = _utc_to_et(utc_dt)

        trend_strength = max(abs(h), abs(a))
        direction = _trend_direction(h, a, edge)

        # Total context
        total_pick = str(row.get("total_pick", "PASS")).strip()
        total_conf = str(row.get("total_conf", ""))
        total_edge = row.get("total_edge", np.nan)

        rows.append({
            "game_date": str(row.get("game_date", ""))[:10],
            "game_time_et": game_time_et,
            "away_team": away,
            "home_team": home,
            "vegas_spread": row.get("vegas_spread", np.nan),
            "model_pick": row.get("model_pick", ""),
            "spread_edge": round(edge, 1),
            "spread_conf": conf,
            "spread_prob": row.get("spread_prob", np.nan),
            "model_predicted_margin": row.get("model_predicted_margin", np.nan),
            "trend_direction": direction,
            "trend_strength": round(trend_strength, 1),
            "netrtg_trend_home": round(h, 2),
            "netrtg_trend_away": round(a, 2),
            "trend_flag": str(row.get("trend_flag", "")),
            "key_signal": str(row.get("key_signal", "")),
            "total_pick": total_pick if total_pick != "PASS" else "",
            "total_conf": total_conf if total_pick != "PASS" else "",
            "total_edge": round(float(total_edge), 1) if pd.notna(total_edge) and total_pick != "PASS" else np.nan,
        })

    if not rows:
        print("[WARN] No trend-aligned picks found — writing empty file")
        out = pd.DataFrame(columns=[
            "game_date", "game_time_et", "away_team", "home_team", "vegas_spread",
            "model_pick", "spread_edge", "spread_conf", "spread_prob",
            "model_predicted_margin", "trend_direction", "trend_strength",
            "netrtg_trend_home", "netrtg_trend_away", "trend_flag", "key_signal",
            "total_pick", "total_conf", "total_edge",
        ])
        out.to_csv("data/cbb_trend_picks_today.csv", index=False)
        return 0

    out = pd.DataFrame(rows)
    # Sort: HIGH first, then by trend_strength desc
    out["_conf_rank"] = out["spread_conf"].map({"HIGH": 2, "MED": 1}).fillna(0)
    out = out.sort_values(["_conf_rank", "trend_strength"], ascending=[False, False]).drop(columns="_conf_rank")
    out.to_csv("data/cbb_trend_picks_today.csv", index=False)

    high_n = (out["spread_conf"] == "HIGH").sum()
    med_n = (out["spread_conf"] == "MED").sum()
    print(f"[OK] Wrote data/cbb_trend_picks_today.csv ({len(out)} trend-aligned picks: {high_n} HIGH, {med_n} MED)")

    # Gate
    gate = {
        "has_rows": len(out) > 0,
        "has_game_time": out["game_time_et"].notna().any() and (out["game_time_et"] != "").any(),
        "has_trend_flag": out["trend_flag"].notna().all(),
        "no_null_teams": out["home_team"].notna().all() and out["away_team"].notna().all(),
    }
    print("=== GATE_TREND RESULTS ===")
    for check, result in gate.items():
        print(f"  {'PASS' if result else 'FAIL'}  {check}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
