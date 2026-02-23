#!/usr/bin/env python3
"""Weekly + season summary accuracy report (CLV-first)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path("data")
IN_PATH = DATA_DIR / "predictions_graded.csv"
OUT_WEEKLY = DATA_DIR / "model_accuracy_weekly.csv"
OUT_SUMMARY = DATA_DIR / "model_accuracy_summary.csv"


def _compute_roi(df: pd.DataFrame) -> Optional[float]:
    if "home_covered_pred" not in df.columns:
        return None
    graded = df["home_covered_pred"].dropna()
    if len(graded) == 0:
        return None
    wins = float(graded.sum())
    losses = float(len(graded) - wins)
    return round(wins * (100 / 110) - losses, 3)


def compute_weekly_sharpe(df: pd.DataFrame) -> Optional[float]:
    if "week" not in df.columns:
        return None
    returns = []
    for _, grp in df.groupby("week"):
        roi = _compute_roi(grp)
        if roi is not None:
            returns.append(roi)
    if len(returns) < 2:
        return None
    s = pd.Series(returns)
    if s.std(ddof=1) == 0:
        return None
    return round(float(s.mean() / s.std(ddof=1)), 3)


def _compute_clv_trend(graded: pd.DataFrame) -> dict:
    """
    Compute whether CLV vs consensus is improving or
    decaying over time. Uses linear regression on game number.

    This is the earliest warning signal for edge erosion —
    CLV decay appears before ATS% decay because it is
    luck-neutral. If CLV is trending down significantly,
    review model weights and bias corrections immediately.
    """
    if "clv_vs_consensus" not in graded.columns or "game_datetime_utc" not in graded.columns:
        return {"direction": "INSUFFICIENT_DATA", "significant": False}

    clv_data = graded[["game_datetime_utc", "clv_vs_consensus"]].dropna()

    if len(clv_data) < 20:
        return {
            "direction": "INSUFFICIENT_DATA",
            "significant": False,
            "note": f"Only {len(clv_data)} CLV data points (need 20+)",
        }

    clv_data = clv_data.sort_values("game_datetime_utc").copy()
    clv_data["game_num"] = range(len(clv_data))

    from scipy import stats as _stats

    slope, intercept, r, p_val, _ = _stats.linregress(
        clv_data["game_num"], clv_data["clv_vs_consensus"]
    )

    trend = {
        "slope_per_game": round(float(slope), 5),
        "r_squared": round(float(r**2), 4),
        "p_value": round(float(p_val), 4),
        "direction": "IMPROVING" if slope > 0 else "DECAYING",
        "significant": float(p_val) < 0.10,
        "data_points": len(clv_data),
    }

    if trend["significant"]:
        if trend["direction"] == "DECAYING":
            log.warning(
                f"⚠️  CLV EDGE DECAY DETECTED: {slope:.5f} pts/game "
                f"(p={p_val:.3f}, n={len(clv_data)}). "
                "Market may be adjusting to model signals. "
                "Run optimize_weights.py and bias_detector.py."
            )
        else:
            log.info(
                f"✓ CLV IMPROVING: +{slope:.5f} pts/game "
                f"(p={p_val:.3f}, n={len(clv_data)}) — "
                "model finding more edge over time."
            )

    return trend


def compute_weekly(df: pd.DataFrame) -> pd.DataFrame:
    graded = df[df.get("graded", False) == True].copy()
    if graded.empty:
        return pd.DataFrame()

    graded["game_datetime_utc"] = pd.to_datetime(graded.get("game_datetime_utc"), errors="coerce", utc=True)
    graded = graded.dropna(subset=["game_datetime_utc"])
    graded["week"] = graded["game_datetime_utc"].dt.to_period("W")

    rows = []
    for week, grp in graded.groupby("week"):
        ats_pct = round(grp["home_covered_pred"].mean() * 100, 1) if grp["home_covered_pred"].notna().sum() else None
        mae = round(grp["abs_spread_error"].mean(), 2) if "abs_spread_error" in grp.columns else None

        week_clv = grp["clv_vs_consensus"].dropna() if "clv_vs_consensus" in grp.columns else pd.Series(dtype=float)
        clv_mean = round(week_clv.mean(), 3) if len(week_clv) >= 3 else None
        beat_cons_pct = round(grp["beat_consensus"].mean() * 100, 1) if "beat_consensus" in grp.columns else None
        clv_n = len(week_clv)

        pinn_clv = grp["clv_vs_pinnacle"].dropna() if "clv_vs_pinnacle" in grp.columns else pd.Series(dtype=float)
        clv_pinn_mean = round(pinn_clv.mean(), 3) if len(pinn_clv) >= 3 else None

        confirmed = grp[grp["market_confirmed"] == True] if "market_confirmed" in grp.columns else pd.DataFrame()
        contradicted = grp[grp["market_contradicted"] == True] if "market_contradicted" in grp.columns else pd.DataFrame()
        ats_confirmed = round(confirmed["home_covered_pred"].mean() * 100, 1) if len(confirmed) >= 3 else None
        ats_contradicted = round(contradicted["home_covered_pred"].mean() * 100, 1) if len(contradicted) >= 3 else None

        rows.append(
            {
                "week": str(week),
                "clv_vs_consensus_mean": clv_mean,
                "clv_vs_pinnacle_mean": clv_pinn_mean,
                "beat_consensus_pct": beat_cons_pct,
                "clv_games_n": clv_n,
                "ats_market_confirmed": ats_confirmed,
                "ats_market_contradicted": ats_contradicted,
                "market_confirmed_n": len(confirmed),
                "market_contradicted_n": len(contradicted),
                "ats_pct": ats_pct,
                "mae": mae,
                "roi_units": _compute_roi(grp),
            }
        )

    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Season-level accuracy summary.
    Column order: CLV metrics → ATS metrics → market signals.
    """
    graded = df[df["graded"] == True].copy()
    g = len(graded)
    if g == 0:
        log.warning("No graded predictions for summary")
        return pd.DataFrame([
            {
                "as_of_utc": pd.Timestamp.utcnow().isoformat(),
                "clv_vs_pinnacle_mean": None,
                "clv_vs_pinnacle_n": 0,
                "beat_pinnacle_pct": None,
                "clv_vs_consensus_mean": None,
                "beat_consensus_pct": None,
                "clv_vs_dk_mean": None,
                "clv_with_sharp_mean": None,
                "clv_against_sharp_mean": None,
                "clv_trend_direction": "INSUFFICIENT_DATA",
                "clv_trend_significant": False,
                "clv_trend_slope": None,
                "total_graded": 0,
                "ats_pct": None,
                "edge_ats_pct": None,
                "spread_mae": None,
                "roi_units": None,
                "sharpe_ratio": None,
                "weeks_above_breakeven": None,
                "weeks_total": None,
                "ats_market_confirmed": None,
                "ats_market_contradicted": None,
                "market_confirmed_n": 0,
                "market_contradicted_n": 0,
                "market_lift_pp": None,
                "ats_steam_games": None,
                "steam_games_n": 0,
                "ats_rlm_confirmed": None,
                "rlm_confirmed_n": 0,
                "ats_line_freeze": None,
                "freeze_games_n": 0,
            }
        ])

    pinn_clv = graded["clv_vs_pinnacle"].dropna() if "clv_vs_pinnacle" in graded.columns else pd.Series(dtype=float)
    clv_pinn_mean = round(pinn_clv.mean(), 3) if len(pinn_clv) >= 10 else None
    beat_pinn_pct = (
        round(graded["beat_pinnacle"].mean() * 100, 1)
        if "beat_pinnacle" in graded.columns and graded["beat_pinnacle"].notna().sum() >= 10
        else None
    )
    clv_pinn_n = len(pinn_clv)

    cons_clv = graded["clv_vs_consensus"].dropna() if "clv_vs_consensus" in graded.columns else pd.Series(dtype=float)
    clv_cons_mean = round(cons_clv.mean(), 3) if len(cons_clv) >= 10 else None
    beat_cons_pct = (
        round(graded["beat_consensus"].mean() * 100, 1)
        if "beat_consensus" in graded.columns and graded["beat_consensus"].notna().sum() >= 10
        else None
    )

    dk_clv = graded["clv_vs_dk"].dropna() if "clv_vs_dk" in graded.columns else pd.Series(dtype=float)
    clv_dk_mean = round(dk_clv.mean(), 3) if len(dk_clv) >= 10 else None

    with_sharp = graded[graded["clv_direction"] == "WITH_SHARP"]
    against_sharp = graded[graded["clv_direction"] == "AGAINST_SHARP"]
    clv_with_mean = round(with_sharp["clv_vs_consensus"].mean(), 3) if len(with_sharp) >= 5 else None
    clv_against_mean = round(against_sharp["clv_vs_consensus"].mean(), 3) if len(against_sharp) >= 5 else None

    clv_trend = _compute_clv_trend(graded)

    ats_pct = round(graded["home_covered_pred"].mean() * 100, 1)
    mae = round(graded["abs_spread_error"].mean(), 2)
    roi = _compute_roi(graded)
    if "week" not in graded.columns and "game_datetime_utc" in graded.columns:
        graded["week"] = pd.to_datetime(graded["game_datetime_utc"]).dt.to_period("W")
    sharpe = compute_weekly_sharpe(graded)

    edge_games = graded[graded["is_alpha"] == True] if "is_alpha" in graded.columns else pd.DataFrame()
    edge_ats = round(edge_games["home_covered_pred"].mean() * 100, 1) if len(edge_games) >= 5 else None

    if "week" in graded.columns:
        weekly_roi = graded.groupby("week")["home_covered_pred"].apply(
            lambda x: x.sum() * (100 / 110) - (len(x) - x.sum())
        )
        weeks_above = int((weekly_roi > 0).sum())
        weeks_total = len(weekly_roi)
    else:
        weeks_above = None
        weeks_total = None

    confirmed = graded[graded["market_confirmed"] == True] if "market_confirmed" in graded.columns else pd.DataFrame()
    contradicted = graded[graded["market_contradicted"] == True] if "market_contradicted" in graded.columns else pd.DataFrame()
    steam_games = graded[graded["had_steam"] == True] if "had_steam" in graded.columns else pd.DataFrame()
    rlm_games = graded[graded["had_rlm"] == True] if "had_rlm" in graded.columns else pd.DataFrame()
    freeze_games = graded[graded["had_line_freeze"] == True] if "had_line_freeze" in graded.columns else pd.DataFrame()

    rlm_confirmed = rlm_games[rlm_games["market_confirmed"] == True] if len(rlm_games) > 0 else pd.DataFrame()

    def _safe_ats(df_sub, min_n=5):
        if len(df_sub) < min_n:
            return None
        return round(df_sub["home_covered_pred"].mean() * 100, 1)

    ats_confirmed = _safe_ats(confirmed)
    ats_contradicted = _safe_ats(contradicted)
    ats_steam = _safe_ats(steam_games)
    ats_rlm_confirmed = _safe_ats(rlm_confirmed)
    ats_freeze = _safe_ats(freeze_games)

    market_lift = round(float(ats_confirmed) - float(ats_pct), 1) if ats_confirmed is not None else None

    return pd.DataFrame([
        {
            "as_of_utc": pd.Timestamp.utcnow().isoformat(),
            "clv_vs_pinnacle_mean": clv_pinn_mean,
            "clv_vs_pinnacle_n": clv_pinn_n,
            "beat_pinnacle_pct": beat_pinn_pct,
            "clv_vs_consensus_mean": clv_cons_mean,
            "beat_consensus_pct": beat_cons_pct,
            "clv_vs_dk_mean": clv_dk_mean,
            "clv_with_sharp_mean": clv_with_mean,
            "clv_against_sharp_mean": clv_against_mean,
            "clv_trend_direction": clv_trend.get("direction", "INSUFFICIENT_DATA"),
            "clv_trend_significant": clv_trend.get("significant", False),
            "clv_trend_slope": clv_trend.get("slope_per_game"),
            "total_graded": g,
            "ats_pct": ats_pct,
            "edge_ats_pct": edge_ats,
            "spread_mae": mae,
            "roi_units": roi,
            "sharpe_ratio": sharpe,
            "weeks_above_breakeven": weeks_above,
            "weeks_total": weeks_total,
            "ats_market_confirmed": ats_confirmed,
            "ats_market_contradicted": ats_contradicted,
            "market_confirmed_n": len(confirmed),
            "market_contradicted_n": len(contradicted),
            "market_lift_pp": market_lift,
            "ats_steam_games": ats_steam,
            "steam_games_n": len(steam_games),
            "ats_rlm_confirmed": ats_rlm_confirmed,
            "rlm_confirmed_n": len(rlm_confirmed),
            "ats_line_freeze": ats_freeze,
            "freeze_games_n": len(freeze_games),
        }
    ])


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing graded predictions: {IN_PATH}")

    df = pd.read_csv(IN_PATH, low_memory=False)
    weekly = compute_weekly(df)
    summary = compute_summary(df)

    OUT_WEEKLY.parent.mkdir(parents=True, exist_ok=True)
    weekly.to_csv(OUT_WEEKLY, index=False)
    summary.to_csv(OUT_SUMMARY, index=False)

    log.info("Wrote %s", OUT_WEEKLY)
    log.info("Wrote %s", OUT_SUMMARY)


if __name__ == "__main__":
    main()
