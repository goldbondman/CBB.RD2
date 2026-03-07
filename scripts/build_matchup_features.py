#!/usr/bin/env python3
"""Agent 3: build matchup-level features with Gate 3 enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MIN_GAMES = 6


def _pick_case_insensitive(cols: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in cols}
    for item in candidates:
        hit = lookup.get(item.lower())
        if hit:
            return hit
    return None


def _is_away_value(val: object) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, (bool, np.bool_)):
        return not bool(val)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val) == 0.0
    return str(val).strip().lower() in {"away", "a", "0", "false", "road"}


def _m(row: pd.Series | None, col: str, default: float = np.nan) -> float:
    if row is None:
        return default
    val = row.get(col, np.nan)
    if pd.isna(val):
        return default
    try:
        return float(val)
    except Exception:
        return default


def main() -> int:
    audit = json.loads(Path("data/internal/audit_report.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv("data/internal/core_metrics.csv", low_memory=False)
    market = pd.read_csv("data/market_lines_latest_by_game.csv", low_memory=False)

    market_cols = list(market.columns)
    metric_cols = list(metrics.columns)

    date_col = _pick_case_insensitive(
        market_cols,
        ["game_datetime_utc", "date", "game_date", "line_timestamp_utc"],
    )
    if not date_col:
        raise ValueError("Market date column not found")
    home_team_id_col = _pick_case_insensitive(market_cols, ["home_team_id"])
    away_team_id_col = _pick_case_insensitive(market_cols, ["away_team_id"])
    if not home_team_id_col or not away_team_id_col:
        raise ValueError("Market team ID columns not found")

    home_name_col = _pick_case_insensitive(market_cols, ["home_team_name", "home_team"])
    away_name_col = _pick_case_insensitive(market_cols, ["away_team_name", "away_team"])
    event_col_market = _pick_case_insensitive(market_cols, ["event_id", "game_id"])
    spread_col = _pick_case_insensitive(market_cols, ["closing_spread", "spread_line", "opening_spread", "spread"])
    total_col = _pick_case_insensitive(market_cols, ["closing_total", "total_line", "opening_total", "total"])
    result_col = audit.get("upcoming_games", {}).get("result_col")

    metric_date_col = _pick_case_insensitive(metric_cols, ["game_datetime_utc", "date", "game_date"])
    metric_team_col = _pick_case_insensitive(metric_cols, ["team_id", "team"])
    metric_event_col = _pick_case_insensitive(metric_cols, ["event_id", "game_id"])
    metric_ha_col = _pick_case_insensitive(metric_cols, ["home_away", "is_home", "location"])
    metric_pts_col = _pick_case_insensitive(metric_cols, ["points_for", "pts", "points", "score"])
    if not metric_date_col or not metric_team_col:
        raise ValueError("Core metrics missing date or team column")

    market[date_col] = pd.to_datetime(market[date_col], errors="coerce", utc=True)
    metrics[metric_date_col] = pd.to_datetime(metrics[metric_date_col], errors="coerce", utc=True)
    market[home_team_id_col] = pd.to_numeric(market[home_team_id_col], errors="coerce")
    market[away_team_id_col] = pd.to_numeric(market[away_team_id_col], errors="coerce")
    metrics[metric_team_col] = pd.to_numeric(metrics[metric_team_col], errors="coerce")

    if spread_col:
        market["closing_spread"] = pd.to_numeric(market[spread_col], errors="coerce")
    else:
        market["closing_spread"] = np.nan
    for alt in ["closing_spread", "spread_line", "opening_spread"]:
        hit = _pick_case_insensitive(market_cols, [alt])
        if hit:
            market["closing_spread"] = market["closing_spread"].combine_first(pd.to_numeric(market[hit], errors="coerce"))

    if total_col:
        market["closing_total"] = pd.to_numeric(market[total_col], errors="coerce")
    else:
        market["closing_total"] = np.nan
    for alt in ["closing_total", "total_line", "opening_total"]:
        hit = _pick_case_insensitive(market_cols, [alt])
        if hit:
            market["closing_total"] = market["closing_total"].combine_first(pd.to_numeric(market[hit], errors="coerce"))

    # Build completed-score map from core metrics when available.
    score_map: dict[str, tuple[float, float]] = {}
    if metric_event_col and metric_ha_col and metric_pts_col:
        scores = metrics[[metric_event_col, metric_ha_col, metric_pts_col]].copy()
        scores[metric_event_col] = scores[metric_event_col].astype(str)
        scores[metric_pts_col] = pd.to_numeric(scores[metric_pts_col], errors="coerce")
        scores["_is_away"] = scores[metric_ha_col].apply(_is_away_value)
        home_scores = scores[~scores["_is_away"]][[metric_event_col, metric_pts_col]].rename(
            columns={metric_event_col: "event_id", metric_pts_col: "home_score"}
        )
        away_scores = scores[scores["_is_away"]][[metric_event_col, metric_pts_col]].rename(
            columns={metric_event_col: "event_id", metric_pts_col: "away_score"}
        )
        joined_scores = home_scores.merge(away_scores, on="event_id", how="inner")
        for _, row in joined_scores.iterrows():
            score_map[str(row["event_id"])] = (float(row["home_score"]), float(row["away_score"]))

    required_metric_cols = [
        "_NetRtg_L11",
        "_NetRtg_trend",
        "_OffEff_L11",
        "_eFG_L11",
        "_TOV_pct_L11",
        "_ORB_pct_L11",
        "_FTr_L11",
        "_NetRtg_vol_L6",
        "_ORB_pct_L6",
        "_TOV_pct_L6",
        "_Pace_L11",
        "_Pace_L6",
        "_3PA_rate_L11",
        "_FT_PPP_L11",
        "_eFG_L6",
        "_eFG_away_L11",
    ]

    metrics = metrics.sort_values([metric_team_col, metric_date_col]).reset_index(drop=True)
    team_index: dict[float, pd.DataFrame] = {}
    for team_id, tdf in metrics.groupby(metric_team_col, dropna=True):
        team_index[float(team_id)] = tdf.sort_values(metric_date_col).reset_index(drop=True)

    league_cache: dict[tuple[str, str], float] = {}

    def league_avg_as_of(col: str, as_of_date: pd.Timestamp) -> float:
        if col not in metrics.columns:
            return 0.0
        key = (col, as_of_date.isoformat())
        if key in league_cache:
            return league_cache[key]
        prior = metrics[metrics[metric_date_col] < as_of_date]
        val = float(pd.to_numeric(prior[col], errors="coerce").mean()) if len(prior) else 0.0
        if pd.isna(val):
            val = 0.0
        league_cache[key] = val
        return val

    def get_team_metrics_as_of(team_id: float, as_of_date: pd.Timestamp) -> pd.Series | None:
        tdf = team_index.get(float(team_id))
        if tdf is None or tdf.empty:
            return None
        valid = tdf[tdf[metric_date_col] < as_of_date]
        if len(valid) < MIN_GAMES:
            return None
        row = valid.iloc[-1]
        if pd.isna(row.get("_NetRtg_L11", np.nan)):
            return None
        return row

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    now_utc = pd.Timestamp.now(tz="UTC")

    for _, game in market.iterrows():
        gdate = game[date_col]
        home_id = game[home_team_id_col]
        away_id = game[away_team_id_col]
        if pd.isna(gdate) or pd.isna(home_id) or pd.isna(away_id):
            skipped.append(
                {
                    "event_id": str(game.get(event_col_market, "")),
                    "reason": "missing_date_or_team_ids",
                }
            )
            continue

        home_m = get_team_metrics_as_of(float(home_id), gdate)
        away_m = get_team_metrics_as_of(float(away_id), gdate)
        if home_m is None or away_m is None:
            skipped.append(
                {
                    "event_id": str(game.get(event_col_market, "")),
                    "reason": "insufficient_prior_games",
                }
            )
            continue

        league_efg = league_avg_as_of("_eFG_L11", gdate)
        league_tov = league_avg_as_of("_TOV_pct_L11", gdate)
        league_orb = league_avg_as_of("_ORB_pct_L11", gdate)
        league_ftr = league_avg_as_of("_FTr_L11", gdate)
        league_ft_ppp = league_avg_as_of("_FT_PPP_L11", gdate)

        netrtg_l11_diff = _m(home_m, "_NetRtg_L11", 0.0) - _m(away_m, "_NetRtg_L11", 0.0)
        netrtg_trend_home = _m(home_m, "_NetRtg_trend", 0.0)
        netrtg_trend_away = _m(away_m, "_NetRtg_trend", 0.0)
        netrtg_trend_diff = netrtg_trend_home - netrtg_trend_away

        home_efg_off = _m(home_m, "_eFG_L11", league_efg)
        away_efg_def = _m(away_m, "_eFG_away_L11", np.nan)
        if pd.isna(away_efg_def):
            away_efg_def = league_efg - (
                (_m(away_m, "_NetRtg_L11", 0.0) - _m(away_m, "_OffEff_L11", 0.0)) / 100.0
            )
        if pd.isna(away_efg_def):
            away_efg_def = league_efg
        efg_odi = (home_efg_off - league_efg) - (away_efg_def - league_efg)

        home_tov = _m(home_m, "_TOV_pct_L11", league_tov)
        away_tov_forced = _m(away_m, "_TOV_pct_L11", league_tov)
        to_odi = -(home_tov - league_tov) + (away_tov_forced - league_tov)

        home_orb = _m(home_m, "_ORB_pct_L11", league_orb)
        away_orb_allow = league_orb - (_m(away_m, "_ORB_pct_L11", league_orb) - league_orb)
        orb_odi = (home_orb - league_orb) - (away_orb_allow - league_orb)

        home_ftr = _m(home_m, "_FTr_L11", league_ftr)
        away_ftr_allow = _m(away_m, "_FTr_L11", league_ftr)
        ftr_odi = (home_ftr - league_ftr) - (away_ftr_allow - league_ftr)
        odi_star_diff = 0.40 * efg_odi + 0.25 * to_odi + 0.20 * orb_odi + 0.15 * ftr_odi

        efg_matchup_diff = _m(home_m, "_eFG_L11", league_efg) - _m(away_m, "_eFG_L11", league_efg)
        pei_l6 = (_m(home_m, "_ORB_pct_L6", 0.0) - _m(away_m, "_ORB_pct_L6", 0.0)) + (
            _m(away_m, "_TOV_pct_L6", 0.0) - _m(home_m, "_TOV_pct_L6", 0.0)
        )
        away_efg_diff = _m(home_m, "_eFG_away_L11", np.nan) - _m(away_m, "_eFG_away_L11", np.nan)
        vol_l6_diff = _m(home_m, "_NetRtg_vol_L6", 0.0) - _m(away_m, "_NetRtg_vol_L6", 0.0)

        pace_l11_avg = (_m(home_m, "_Pace_L11", 0.0) + _m(away_m, "_Pace_L11", 0.0)) / 2.0
        pace_l6_avg = (_m(home_m, "_Pace_L6", 0.0) + _m(away_m, "_Pace_L6", 0.0)) / 2.0
        posw_l6 = pei_l6 * pace_l6_avg
        sch = (
            abs(_m(home_m, "_Pace_L11", 0.0) - _m(away_m, "_Pace_L11", 0.0))
            + abs(_m(home_m, "_3PA_rate_L11", 0.0) - _m(away_m, "_3PA_rate_L11", 0.0))
            + abs(_m(home_m, "_FTr_L11", 0.0) - _m(away_m, "_FTr_L11", 0.0))
        )
        wl_sum = (_m(home_m, "_FT_PPP_L11", 0.0) - league_ft_ppp) + (
            _m(away_m, "_FT_PPP_L11", 0.0) - league_ft_ppp
        )
        efg_l6_avg = (_m(home_m, "_eFG_L6", league_efg) + _m(away_m, "_eFG_L6", league_efg)) / 2.0
        tc_proxy_diff = -_m(home_m, "_NetRtg_vol_L6", 0.0) - (-_m(away_m, "_NetRtg_vol_L6", 0.0))
        vol_l6_sum = _m(home_m, "_NetRtg_vol_L6", 0.0) + _m(away_m, "_NetRtg_vol_L6", 0.0)

        event_id_value = str(game.get(event_col_market, f"{int(home_id)}_{int(away_id)}_{gdate.date()}"))
        home_score = np.nan
        away_score = np.nan
        if event_id_value in score_map:
            home_score, away_score = score_map[event_id_value]
        has_scores = pd.notna(home_score) and pd.notna(away_score)
        if result_col and result_col in market.columns:
            is_upcoming = pd.isna(game.get(result_col))
        else:
            is_upcoming = (not has_scores) and (pd.isna(gdate) or gdate >= now_utc)

        rows.append(
            {
                "game_id": event_id_value,
                "event_id": event_id_value,
                "game_date": gdate,
                "home_team_id": int(home_id),
                "away_team_id": int(away_id),
                "home_team": game.get(home_name_col, str(home_id)),
                "away_team": game.get(away_name_col, str(away_id)),
                "closing_spread": game.get("closing_spread", np.nan),
                "closing_total": game.get("closing_total", np.nan),
                "is_upcoming": bool(is_upcoming),
                "home_score": home_score,
                "away_score": away_score,
                "actual_margin": (home_score - away_score) if has_scores else np.nan,
                "actual_total": (home_score + away_score) if has_scores else np.nan,
                "netrtg_L11_diff": netrtg_l11_diff,
                "netrtg_trend_diff": netrtg_trend_diff,
                "netrtg_trend_home": netrtg_trend_home,
                "netrtg_trend_away": netrtg_trend_away,
                "odi_star_diff": odi_star_diff,
                "efg_matchup_diff": efg_matchup_diff,
                "pei_L6_matchup": pei_l6,
                "away_efg_L11_diff": away_efg_diff,
                "vol_L6_diff": vol_l6_diff,
                "home_dummy": 1,
                "pace_L11_avg": pace_l11_avg,
                "posw_L6": posw_l6,
                "sch_matchup": sch,
                "wl_L11_sum": wl_sum,
                "efg_L6_avg": efg_l6_avg,
                "tc_proxy_diff": tc_proxy_diff,
                "vol_L6_sum": vol_l6_sum,
                "is_big_dog": int(pd.notna(game.get("closing_spread")) and abs(float(game.get("closing_spread"))) >= 7.0)
                if pd.notna(game.get("closing_spread"))
                else 0,
                "is_hot_home": int(netrtg_trend_home > 1.5),
                "is_cold_away": int(netrtg_trend_away < -1.5),
            }
        )

    df_matchup = pd.DataFrame(rows)
    out_dir = Path("data/internal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "matchup_features.csv"
    df_matchup.to_csv(out_path, index=False)
    if skipped:
        (out_dir / "skipped_games.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    upcoming = df_matchup[df_matchup["is_upcoming"] == True] if "is_upcoming" in df_matchup.columns else pd.DataFrame()
    spread_features = [
        "netrtg_L11_diff",
        "netrtg_trend_diff",
        "odi_star_diff",
        "efg_matchup_diff",
        "pei_L6_matchup",
        "vol_L6_diff",
        "home_dummy",
    ]
    feature_coverage = {
        f: 1.0 - float(df_matchup[f].isna().mean()) for f in spread_features if f in df_matchup.columns
    }
    gate = {
        "has_rows": len(df_matchup) > 50,
        "upcoming_games_exist": len(upcoming) > 0,
        "has_spread_col": "closing_spread" in df_matchup.columns and bool(df_matchup["closing_spread"].notna().any()),
        "features_populated": bool(feature_coverage) and all(v > 0.50 for v in feature_coverage.values()),
        "trend_col_exists": "netrtg_trend_diff" in df_matchup.columns,
    }

    print("=== GATE_3 RESULTS ===")
    for check, result in gate.items():
        print(f"  {'PASS' if result else 'FAIL'}  {check}")
    print("Feature coverage:")
    for feat, cov in feature_coverage.items():
        print(f"  {feat}: {cov:.1%}")

    if not all(gate.values()):
        failed = [k for k, v in gate.items() if not v]
        print(f"[STOP] Gate 3 failed: {failed}")
        return 1

    print(f"[OK] Gate 3 passed. Wrote {out_path} with {len(df_matchup)} rows, upcoming={len(upcoming)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
