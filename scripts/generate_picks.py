#!/usr/bin/env python3
"""Agent 5: generate upcoming picks using trained models with Gate 5 enforcement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib

SPREAD_EDGE_MIN = 2.0
TOTAL_EDGE_MIN = 2.5


def _fmt_line(value: float) -> str:
    return f"{value:+.1f}"


def _model_pick(row: pd.Series) -> str:
    """Model spread pick - shows team name + line, or PASS."""
    line = row["closing_spread"]
    edge = row["spread_edge"]
    if pd.isna(line) or abs(edge) < SPREAD_EDGE_MIN:
        return "PASS"
    if edge > 0:
        # Model likes home to cover
        return f"{row['home_team']} {_fmt_line(float(line))}"
    else:
        # Model likes away to cover; away line is opposite sign
        away_line = -float(line)
        return f"{row['away_team']} {_fmt_line(away_line)}"


def _total_pick(row: pd.Series) -> str:
    line = row["closing_total"]
    edge = row["total_edge"]
    if pd.isna(line) or abs(edge) < TOTAL_EDGE_MIN:
        return "PASS"
    direction = "OVER" if edge > 0 else "UNDER"
    return f"{direction} {float(line):.1f}"


def _conf_tier(edge_abs: float, prob_dist: float) -> str:
    if edge_abs >= 4.0 and prob_dist >= 0.08:
        return "HIGH"
    if edge_abs >= 2.0 and prob_dist >= 0.04:
        return "MED"
    if edge_abs >= 1.0:
        return "LOW"
    return "PASS"


def _vegas_pred_win(row: pd.Series) -> str:
    line = row.get("closing_spread", np.nan)
    if pd.isna(line):
        return "UNKNOWN"
    line_val = float(line)
    if line_val < 0:
        return str(row.get("home_team", "HOME"))
    if line_val > 0:
        return str(row.get("away_team", "AWAY"))
    return "PICKEM"


def _key_signal(row: pd.Series) -> str:
    """Narrative explaining the primary driver of the spread pick."""
    home = str(row.get("home_team", "Home"))
    away = str(row.get("away_team", "Away"))
    edge = float(row.get("spread_edge", 0.0) or 0.0)
    pick_team = home if edge > 0 else away
    opp_team = away if edge > 0 else home

    # Road dog edge
    if bool(row.get("is_big_dog", 0)) and edge < 0:
        efg = float(row.get("away_efg_L11_diff", 0.0) or 0.0)
        pei = float(row.get("pei_L6_matchup", 0.0) or 0.0)
        parts = []
        if abs(efg) > 0.01:
            parts.append(f"eFG {efg:+.2f}")
        if abs(pei) > 0.01:
            parts.append(f"PEI {pei:+.1f}")
        detail = ", ".join(parts) if parts else "model consensus"
        return f"{away} road dog edge - {detail}"

    # Hot home team
    if bool(row.get("is_hot_home", 0)) and edge > 0:
        h_trend = float(row.get("netrtg_trend_home", 0.0) or 0.0)
        a_trend = float(row.get("netrtg_trend_away", 0.0) or 0.0)
        return (
            f"{home} trending up ({h_trend:+.1f} net rtg/game) "
            f"vs {away} ({a_trend:+.1f})"
        )

    # Signal-based: pick strongest metric
    candidates: list[tuple[float, str]] = []

    netrtg = float(row.get("netrtg_trend_diff", 0.0) or 0.0)
    if abs(netrtg) > 0.1:
        favor = home if netrtg > 0 else away
        candidates.append((abs(netrtg), f"Net rtg trend favors {favor} ({netrtg:+.1f} diff)"))

    odi = float(row.get("odi_star_diff", 0.0) or 0.0)
    if abs(odi) > 0.1:
        favor = home if odi > 0 else away
        candidates.append((abs(odi), f"ODI* edge: {favor} ({odi:+.1f})"))

    pei = float(row.get("pei_L6_matchup", 0.0) or 0.0)
    if abs(pei) > 0.1:
        favor = home if pei > 0 else away
        candidates.append((abs(pei), f"PEI matchup: {favor} ({pei:+.1f})"))

    efg_road = float(row.get("away_efg_L11_diff", 0.0) or 0.0)
    if abs(efg_road) > 0.01:
        favor = away if efg_road > 0 else home
        candidates.append((abs(efg_road) * 10, f"Road eFG edge: {favor} ({efg_road:+.2f})"))

    sch = float(row.get("sch_matchup", 0.0) or 0.0)
    if abs(sch) > 0.1:
        candidates.append((abs(sch), f"Schedule clash: {pick_team} ({sch:+.1f} SOS diff)"))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # Generic fallback: show the edge magnitude
    return f"{pick_team} model edge ({abs(edge):.1f} pts vs line, opp: {opp_team})"


def _trend_flag(row: pd.Series) -> str:
    """Narrative trend flag with team names and net rating values."""
    home = str(row.get("home_team", "Home"))
    away = str(row.get("away_team", "Away"))
    h = float(row.get("netrtg_trend_home", 0.0) or 0.0)
    a = float(row.get("netrtg_trend_away", 0.0) or 0.0)

    if h > 1.5 and a > 1.5:
        return f"Both trending up - {home} {h:+.1f}, {away} {a:+.1f} net rtg/game"
    if h < -1.5 and a < -1.5:
        return f"Both cooling - {home} {h:+.1f}, {away} {a:+.1f} net rtg/game"
    if h > 1.5:
        return f"{home} trending up ({h:+.1f} net rtg/game); {away} flat ({a:+.1f})"
    if a > 1.5:
        return f"{away} trending up ({a:+.1f} net rtg/game); {home} flat ({h:+.1f})"
    return ""


def _total_key_signal(row: pd.Series) -> str:
    """Narrative explaining the primary driver of the total pick."""
    edge = float(row.get("total_edge", 0.0) or 0.0)
    home = str(row.get("home_team", "Home"))
    away = str(row.get("away_team", "Away"))
    model_total = float(row.get("pred_total", 0.0) or 0.0)
    vegas_total = float(row.get("closing_total", np.nan) if pd.notna(row.get("closing_total")) else 0.0)
    direction = "OVER" if edge > 0 else "UNDER"

    if abs(edge) < TOTAL_EDGE_MIN:
        return "No strong total signal"

    # Pace-based signal
    pace_home = float(row.get("pace_home", 0.0) or 0.0)
    pace_away = float(row.get("pace_away", 0.0) or 0.0)
    if abs(pace_home) > 0.1 and abs(pace_away) > 0.1:
        avg_pace = (pace_home + pace_away) / 2
        pace_label = "up-tempo" if avg_pace > 0 else "slow-paced"
        return (
            f"{direction} - {pace_label} matchup ({home} {pace_home:+.1f}, "
            f"{away} {pace_away:+.1f} pace); model {model_total:.1f} vs line {vegas_total:.1f}"
        )

    # Scoring efficiency signal
    ortg_diff = float(row.get("netrtg_trend_diff", 0.0) or 0.0)
    if abs(ortg_diff) > 0.5:
        favor = home if ortg_diff > 0 else away
        return (
            f"{direction} - {favor} scoring efficiency edge ({ortg_diff:+.1f} net rtg diff); "
            f"model total {model_total:.1f} vs line {vegas_total:.1f}"
        )

    # Generic fallback
    return (
        f"{direction} - model projects {model_total:.1f} "
        f"({edge:+.1f} vs {vegas_total:.1f} line)"
    )


def main() -> int:
    df = pd.read_csv("data/internal/matchup_features.csv", low_memory=False)
    if df.empty:
        print("[STOP] matchup_features.csv is empty")
        return 1

    features = joblib.load("models/feature_lists.pkl")
    ridge_s = joblib.load("models/spread_ridge.pkl")
    ridge_t = joblib.load("models/total_ridge.pkl")
    logit_s = joblib.load("models/spread_logit.pkl")
    logit_t = joblib.load("models/total_logit.pkl")
    sc_s = joblib.load("models/spread_scaler.pkl")
    sc_t = joblib.load("models/total_scaler.pkl")

    spread_features = list(features["spread"])
    total_features = list(features["total"])

    df["is_upcoming"] = df["is_upcoming"].astype(str).str.lower().isin({"true", "1"})
    upcoming = df[df["is_upcoming"] == True].copy()
    if upcoming.empty:
        print("[STOP] No upcoming games found in matchup_features.csv")
        return 1

    for col in ["closing_spread", "closing_total"]:
        if col in upcoming.columns:
            upcoming[col] = pd.to_numeric(upcoming[col], errors="coerce")

    x_s = sc_s.transform(upcoming[spread_features].fillna(0.0))
    x_t = sc_t.transform(upcoming[total_features].fillna(0.0))

    upcoming["pred_margin"] = ridge_s.predict(x_s)
    upcoming["pred_total"] = ridge_t.predict(x_t)
    upcoming["spread_prob"] = logit_s.predict_proba(x_s)[:, 1]
    upcoming["total_prob"] = logit_t.predict_proba(x_t)[:, 1]

    # spread_edge: how much model margin exceeds (or falls short of) the market-implied margin.
    # closing_spread is from the home team's perspective: negative = home favored.
    # Market-implied home margin = -closing_spread.
    # Edge = pred_margin - (-closing_spread) = pred_margin + closing_spread.
    upcoming["spread_edge"] = upcoming["pred_margin"] + upcoming["closing_spread"].fillna(0.0)
    upcoming["total_edge"] = upcoming["pred_total"] - upcoming["closing_total"].fillna(0.0)

    picks: list[dict[str, object]] = []
    for _, row in upcoming.iterrows():
        s_edge = round(float(row["spread_edge"]), 1)
        t_edge = round(float(row["total_edge"]), 1)
        s_prob = float(row["spread_prob"])
        t_prob = float(row["total_prob"])
        picks.append(
            {
                "game_date": str(row.get("game_date", ""))[:10],
                "away_team": row.get("away_team", ""),
                "home_team": row.get("home_team", ""),
                "vegas_spread": round(float(row["closing_spread"]), 1) if pd.notna(row["closing_spread"]) else np.nan,
                "vegas_total": round(float(row["closing_total"]), 1) if pd.notna(row["closing_total"]) else np.nan,
                "vegas_pred_win": _vegas_pred_win(row),
                "model_predicted_margin": round(float(row["pred_margin"]), 1),
                "spread_edge": s_edge,
                "model_pick": _model_pick(row),
                "spread_conf": _conf_tier(abs(s_edge), abs(s_prob - 0.5)),
                "spread_prob": round(s_prob, 3),
                "model_predicted_total": round(float(row["pred_total"]), 1),
                "total_edge": t_edge,
                "total_pick": _total_pick(row),
                "total_conf": _conf_tier(abs(t_edge), abs(t_prob - 0.5)),
                "total_prob": round(t_prob, 3),
                "key_signal": _key_signal(row),
                "trend_flag": _trend_flag(row),
                "total_key_signal": _total_key_signal(row),
            }
        )

    out = pd.DataFrame(picks)
    out["_sort"] = out["spread_conf"].map({"HIGH": 3, "MED": 2, "LOW": 1, "PASS": 0}).fillna(0)
    out = out.sort_values(["_sort", "spread_edge"], ascending=[False, False]).drop(columns="_sort")
    ordered_cols = [
        "game_date",
        "away_team",
        "home_team",
        "vegas_spread",
        "vegas_pred_win",
        "model_predicted_margin",
        "spread_edge",
        "model_pick",
        "spread_conf",
        "spread_prob",
        "vegas_total",
        "model_predicted_total",
        "total_edge",
        "total_pick",
        "total_conf",
        "total_prob",
        "key_signal",
        "trend_flag",
        "total_key_signal",
    ]
    out = out[[c for c in ordered_cols if c in out.columns]]
    out.to_csv("data/cbb_picks_today.csv", index=False)

    required = [
        "game_date",
        "away_team",
        "home_team",
        "vegas_spread",
        "vegas_total",
        "vegas_pred_win",
        "model_predicted_margin",
        "model_predicted_total",
        "model_pick",
        "spread_edge",
        "spread_conf",
        "spread_prob",
        "total_pick",
        "total_edge",
        "total_conf",
        "total_prob",
        "key_signal",
        "trend_flag",
        "total_key_signal",
    ]
    missing = [c for c in required if c not in out.columns]
    gate = {
        "schema_valid": len(missing) == 0,
        "has_games": len(out) > 0,
        "not_all_pass": not bool((out["spread_conf"] == "PASS").all()),
        "no_null_teams": out["home_team"].notna().all() and out["away_team"].notna().all(),
    }
    print("=== GATE_5 RESULTS ===")
    for check, result in gate.items():
        print(f"  {'PASS' if result else 'FAIL'}  {check}")
    if missing:
        print(f"  Missing columns: {missing}")

    if not all(gate.values()):
        print("[STOP] Gate 5 failed.")
        if gate.get("not_all_pass") is False and "closing_spread" in upcoming.columns:
            print(
                "  closing_spread range: "
                f"{upcoming['closing_spread'].min(skipna=True):.1f} to {upcoming['closing_spread'].max(skipna=True):.1f}"
            )
            print(
                "  pred_margin range: "
                f"{upcoming['pred_margin'].min(skipna=True):.1f} to {upcoming['pred_margin'].max(skipna=True):.1f}"
            )
        return 1

    print(f"[OK] Gate 5 passed. Wrote data/cbb_picks_today.csv ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
