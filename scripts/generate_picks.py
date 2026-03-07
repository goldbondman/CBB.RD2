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


def _spread_pick(row: pd.Series) -> str:
    line = row["closing_spread"]
    edge = row["spread_edge"]
    if pd.isna(line) or abs(edge) < SPREAD_EDGE_MIN:
        return "PASS"
    if edge > 0:
        return f"HOME {_fmt_line(float(line))}"
    away_line = -float(line)
    return f"AWAY {_fmt_line(away_line)}"


def _total_pick(row: pd.Series) -> str:
    line = row["closing_total"]
    edge = row["total_edge"]
    if pd.isna(line) or abs(edge) < TOTAL_EDGE_MIN:
        return "PASS"
    return f"{'OVER' if edge > 0 else 'UNDER'} {float(line):.1f}"


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
    signals: dict[str, float] = {}
    for label, col in [
        ("Trend edge", "netrtg_trend_diff"),
        ("ODI* edge", "odi_star_diff"),
        ("PEI swing", "pei_L6_matchup"),
        ("eFG road edge", "away_efg_L11_diff"),
        ("SCH clash", "sch_matchup"),
    ]:
        val = row.get(col, np.nan)
        if pd.notna(val):
            signals[label] = abs(float(val))
    if bool(row.get("is_big_dog", 0)) and float(row.get("spread_edge", 0.0)) < 0:
        return "Road dog edge (eFG + PEI)"
    if bool(row.get("is_hot_home", 0)) and float(row.get("spread_edge", 0.0)) > 0:
        return f"Trend edge (home up {float(row.get('netrtg_trend_home', 0.0)):.1f})"
    if not signals:
        return "Model consensus"
    return max(signals, key=signals.get)


def _trend_flag(row: pd.Series) -> str:
    h = float(row.get("netrtg_trend_home", 0.0) or 0.0)
    a = float(row.get("netrtg_trend_away", 0.0) or 0.0)
    if h > 1.5 and a > 1.5:
        return "BOTH_UP"
    if h < -1.5 and a < -1.5:
        return "BOTH_DOWN"
    if h > 1.5:
        return "HOME_UP"
    if a > 1.5:
        return "AWAY_UP"
    return ""


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
    upcoming["spread_edge"] = upcoming["pred_margin"] - upcoming["closing_spread"].fillna(0.0)
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
                "model_predicted_total": round(float(row["pred_total"]), 1),
                "spread_pick": _spread_pick(row),
                "spread_edge": s_edge,
                "spread_conf": _conf_tier(abs(s_edge), abs(s_prob - 0.5)),
                "spread_prob": round(s_prob, 3),
                "total_pick": _total_pick(row),
                "total_edge": t_edge,
                "total_conf": _conf_tier(abs(t_edge), abs(t_prob - 0.5)),
                "total_prob": round(t_prob, 3),
                "key_signal": _key_signal(row),
                "trend_flag": _trend_flag(row),
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
        "spread_pick",
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
        "spread_pick",
        "spread_edge",
        "spread_conf",
        "spread_prob",
        "total_pick",
        "total_edge",
        "total_conf",
        "total_prob",
        "key_signal",
        "trend_flag",
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
