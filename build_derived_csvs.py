#!/usr/bin/env python3
"""
build_derived_csvs.py
=====================
Reads existing pipeline CSVs and generates six derived/snapshot CSVs
for the frontend and downstream consumers.

Outputs (written to both data/csv/ and data/ root):
  1. bet_recs.csv              — Betting recommendations (edge >= 3.75)
  2. team_form_snapshot.csv    — Current form / hot-cold dashboard
  3. matchup_preview.csv       — Today's games with full team context
  4. player_leaders.csv        — Season player leaderboard
  5. upset_watch.csv           — Upset watch (uws_total >= 40)
  6. conference_summary.csv    — Conference standings snapshot

Called from update_espn_cbb.yml after "Run tournament metrics" and
before "Validate outputs".
"""

from __future__ import annotations

import pathlib
import sys
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pipeline_csv_utils import safe_write_csv

# ── Paths ──────────────────────────────────────────────────────────────────
DATA        = pathlib.Path("data")
CSV_DIR     = DATA / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

NOW_UTC     = datetime.now(timezone.utc)
NOW_ISO     = NOW_UTC.strftime("%Y-%m-%dT%H:%M:%SZ")
TODAY_STAMP = NOW_UTC.strftime("%Y%m%d")
PST_TZ      = ZoneInfo("America/Los_Angeles")

# Track results for summary table
_results: list[dict] = []

# Config
SPREAD_EDGE_MIN = 3.0
TOTAL_EDGE_MIN = 4.0
ML_EDGE_MIN = 0.05


# ── Helpers ────────────────────────────────────────────────────────────────

def _load(path: str | pathlib.Path, label: str = "") -> Optional[pd.DataFrame]:
    """Load a CSV; return None with a warning if missing or empty."""
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size < 10:
        print(f"[WARN] {label or p.name}: source file missing or empty — {p}")
        return None
    try:
        df = pd.read_csv(p, low_memory=False)
        if df.empty:
            print(f"[WARN] {label or p.name}: loaded but empty")
            return None
        return df
    except Exception as exc:
        print(f"[WARN] {label or p.name}: failed to load — {exc}")
        return None

def _write(df: pd.DataFrame, stem: str, sources: list[str],
           dated_copy: bool = False) -> None:
    """Write df to data/csv/<stem> and data/<stem>; print status."""
    df = df.copy()
    csv_path  = CSV_DIR / stem
    root_path = DATA / stem
    try:
        df["generated_at"] = NOW_ISO
        safe_write_csv(df, csv_path, index=False)
        safe_write_csv(df, root_path, index=False)
        n = len(df)
        print(f"[OK]  {stem}: {n} rows")
        if dated_copy:
            dated = DATA / stem.replace(".csv", f"_{TODAY_STAMP}.csv")
            safe_write_csv(df, dated, index=False)
            print(f"[OK]  {dated.name}: archive copy written")
        _results.append({"file": stem, "rows": n, "sources": ", ".join(sources), "status": "OK"})
    except Exception as exc:
        print(f"[WARN] {stem}: write failed — {exc}")
        traceback.print_exc()
        _results.append({"file": stem, "rows": 0, "sources": ", ".join(sources), "status": f"FAIL: {exc}"})

# ── Matchup enrichment ─────────────────────────────────────────────────────

def enrich_with_matchup_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Join team_matchup_summary.csv onto a predictions-style DataFrame.

    Adds per-team matchup context columns for both home and away sides.
    Designed to be called on matchup_preview.csv and bet_recs.csv after
    they have been built.

    Returns the original df with extra columns appended (or untouched if
    team_matchup_summary.csv is unavailable).
    """
    tms = _load(CSV_DIR / "team_matchup_summary.csv", "team_matchup_summary")
    if tms is None:
        tms = _load(DATA / "team_matchup_summary.csv", "team_matchup_summary")
    if tms is None:
        return df

    # Normalise join keys to strings to prevent silent NaN merges from long game IDs
    for col in ("event_id", "team_id"):
        if col in tms.columns:
            tms[col] = tms[col].astype(str).str.replace(r"\.0$", "", regex=True)

    pick_cols = [
        "star_pcs", "expected_pts_impact", "star_underperform_risk",
        "favorable_conditions", "avg_pcs_starters",
    ]
    pick_cols = [c for c in pick_cols if c in tms.columns]
    if not pick_cols:
        return df

    # For home side
    if "home_team_id" in df.columns and "event_id" in df.columns:
        home_tms = tms.rename(columns={c: f"home_{c}" for c in pick_cols})
        home_tms = home_tms.rename(columns={"team_id": "home_team_id"})
        join_cols_h = [f"home_{c}" for c in pick_cols]
        home_tms = home_tms[["event_id", "home_team_id"] + join_cols_h].copy()
        for c in ("event_id", "home_team_id"):
            df[c] = df[c].astype(str).str.replace(r"\.0$", "", regex=True)
        df = df.merge(home_tms, on=["event_id", "home_team_id"], how="left")

    if "away_team_id" in df.columns and "event_id" in df.columns:
        away_tms = tms.rename(columns={c: f"away_{c}" for c in pick_cols})
        away_tms = away_tms.rename(columns={"team_id": "away_team_id"})
        join_cols_a = [f"away_{c}" for c in pick_cols]
        away_tms = away_tms[["event_id", "away_team_id"] + join_cols_a].copy()
        for c in ("event_id", "away_team_id"):
            df[c] = df[c].astype(str).str.replace(r"\.0$", "", regex=True)
        df = df.merge(away_tms, on=["event_id", "away_team_id"], how="left")

    # Compute matchup_pts_edge (positive = conditions favour home outscoring baseline)
    if "home_expected_pts_impact" in df.columns and "away_expected_pts_impact" in df.columns:
        df["matchup_pts_edge"] = (
            pd.to_numeric(df["home_expected_pts_impact"], errors="coerce") -
            pd.to_numeric(df["away_expected_pts_impact"], errors="coerce")
        ).round(2)

    return df


def build_bet_recs_csv(predictions: pd.DataFrame) -> None:
    """Build bet_recs.csv: Betting recommendations with edge flags."""
    df = predictions.copy()

    # Filter to scheduled/not final
    if "is_final" in df.columns:
        df = df[df["is_final"] == 0]
    elif "game_status" in df.columns:
        df = df[df["game_status"].astype(str).str.lower() == "scheduled"]

    # Keep only today + tomorrow in Pacific time so recs match the intended slate.
    # This avoids pulling in UTC-dated rows that are "yesterday" in PST.
    pst_today = NOW_UTC.astimezone(PST_TZ).date()
    target_dates = {pst_today, pst_today + timedelta(days=1)}

    dt_col = "game_datetime_utc" if "game_datetime_utc" in df.columns else "game_date"
    if dt_col in df.columns:
        before_count = len(df)
        dt_utc = pd.to_datetime(df[dt_col], errors="coerce", utc=True)
        dt_pst_date = dt_utc.dt.tz_convert(PST_TZ).dt.date
        in_window_mask = dt_pst_date.isin(target_dates)
        dropped_invalid = int(dt_utc.isna().sum())
        df = df[in_window_mask].copy()
        print(
            "[INFO] bet_recs: PST date window filter",
            f"kept={len(df)} dropped_outside={before_count - len(df) - dropped_invalid} dropped_invalid_datetime={dropped_invalid}",
            f"target_dates={[d.isoformat() for d in sorted(target_dates)]}",
        )
    else:
        print("[WARN] bet_recs: missing game datetime column; unable to enforce PST date window")

    recs = []

    # Use ens_ spreads if they exist, else primary
    sp_edge_col = next((c for c in ["ens_spread_edge_pts", "spread_edge_pts", "ens_ens_spread_edge_pts"] if c in df.columns), None)
    tot_edge_col = next((c for c in ["ens_total_edge_pts", "total_edge_pts"] if c in df.columns), None)
    ml_edge_col = next((c for c in ["ens_ml_edge_prob", "ml_edge_prob"] if c in df.columns), None)

    if sp_edge_col is None:
        print("[WARN] bet_recs: missing spread edge column — skipping")
        return

    for _, row in df.iterrows():
        # Spread rec
        edge = row.get(sp_edge_col, 0)
        if abs(edge) >= SPREAD_EDGE_MIN:
            model_line = next((row.get(c) for c in ["ens_spread", "pred_spread"] if c in row), 0)
            market_line = row.get("spread_line", row.get("market_spread", 0))
            pick = row.get("spread_pick", "HOME" if model_line < market_line else "AWAY")
            recs.append({
                "game_id": row.get("game_id"),
                "date": row.get("game_date", row.get("game_datetime_utc")),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "bet_type": "SPREAD",
                "pick": pick,
                "market_line": market_line,
                "model_line": model_line,
                "edge": round(edge, 2),
                "model_prob": row.get("mc_cover_probability", row.get("model_confidence")),
                "market_prob": 0.5,
                "expected_roi": row.get("kelly_units", 0),
                "confidence": row.get("mc_confidence_tier", "MEDIUM")
            })

        # Total rec
        if tot_edge_col and abs(row.get(tot_edge_col, 0)) >= TOTAL_EDGE_MIN:
            edge = row.get(tot_edge_col)
            model_total = next((row.get(c) for c in ["ens_total", "pred_total"] if c in row), 140)
            market_total = row.get("total_line", row.get("market_total", 140))
            pick = "OVER" if model_total > market_total else "UNDER"
            recs.append({
                "game_id": row.get("game_id"),
                "date": row.get("game_date", row.get("game_datetime_utc")),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "bet_type": "TOTAL",
                "pick": pick,
                "market_line": market_total,
                "model_line": model_total,
                "edge": round(edge, 1),
                "model_prob": row.get("mc_over_pct" if pick == "OVER" else "mc_under_pct"),
                "market_prob": 0.5,
                "expected_roi": 0,
                "confidence": "MEDIUM"
            })

    out_df = pd.DataFrame(recs)
    if out_df.empty:
        out_df = pd.DataFrame(columns=[
            "game_id", "date", "home_team", "away_team", "bet_type", "pick",
            "market_line", "model_line", "edge", "model_prob", "market_prob",
            "expected_roi", "confidence",
        ])
        out_df["generated_at"] = NOW_ISO
        for out_path in (CSV_DIR / "bet_recs.csv", DATA / "bet_recs.csv"):
            out_df.to_csv(out_path, index=False)
        print("[INFO] bet_recs: wrote empty file after PST date filtering (prevents stale recs)")
        _results.append({"file": "bet_recs.csv", "rows": 0, "sources": "predictions_mc_latest", "status": "OK"})
        return

    _write(out_df, "bet_recs.csv", ["predictions_mc_latest"])


def build_team_form_snapshot_csv() -> None:
    """Build team_form_snapshot.csv: Rolling form dashboard."""
    # Source from team_game_metrics.csv or similar
    tgm = _load(DATA / "team_game_metrics.csv", "team_game_metrics")
    if tgm is None:
        tgm = _load(DATA / "team_game_weighted.csv", "team_game_weighted")
    if tgm is None:
        return

    tgm["game_datetime_utc"] = pd.to_datetime(tgm["game_datetime_utc"], errors="coerce", utc=True)
    snapshot_date = tgm["game_datetime_utc"].max()

    rows = []
    for tid, group in tgm.groupby("team_id"):
        group = group.sort_values("game_datetime_utc")
        l5 = group.tail(5)
        l10 = group.tail(10)

        def _get_net(df):
            if "adj_net_rtg" in df.columns: return df["adj_net_rtg"].mean()
            if "net_rtg" in df.columns: return df["net_rtg"].mean()
            return (df["points_for"] - df["points_against"]).mean()

        rows.append({
            "as_of_date": snapshot_date.strftime("%Y-%m-%d") if pd.notna(snapshot_date) else TODAY_STAMP,
            "team": group["team"].iloc[-1],
            "team_id": tid,
            "last5_w": int(l5["win"].sum()) if "win" in l5.columns else 0,
            "last5_l": len(l5) - int(l5["win"].sum()) if "win" in l5.columns else 0,
            "last5_margin_avg": round((l5["points_for"] - l5["points_against"]).mean(), 1) if "points_for" in l5.columns else 0,
            "last10_w": int(l10["win"].sum()) if "win" in l10.columns else 0,
            "last10_l": len(l10) - int(l10["win"].sum()) if "win" in l10.columns else 0,
            "last10_margin_avg": round((l10["points_for"] - l10["points_against"]).mean(), 1) if "points_for" in l10.columns else 0,
            "last5_net": round(_get_net(l5), 2),
            "last10_net": round(_get_net(l10), 2),
        })

    out_df = pd.DataFrame(rows)
    _write(out_df, "team_form_snapshot.csv", ["team_game_metrics"])


def build_matchup_preview_csv(predictions: pd.DataFrame) -> None:
    """Build matchup_preview.csv: Today's games with team context."""
    df = predictions.copy()

    # Context sources
    form = _load(CSV_DIR / "team_form_snapshot.csv")
    summary = _load(DATA / "team_season_summary.csv")

    if form is None or summary is None:
        print("[WARN] matchup_preview: missing form or summary — skipping")
        return

    # Join context
    for side in ["home", "away"]:
        # Join summary
        s_cols = ["team_id", "adj_ortg", "adj_drtg", "adj_net_rtg", "adj_pace", "adj_net", "tempo"]
        s_sub = summary[[c for c in s_cols if c in summary.columns]].copy()
        s_sub = s_sub.rename(columns={c: f"{side}_{c}" for c in s_sub.columns if c != "team_id"})
        df = df.merge(s_sub, left_on=f"{side}_team_id", right_on="team_id", how="left").drop(columns="team_id")

        # Join form
        f_cols = ["team_id", "last5_net", "last10_net", "last5_margin_avg", "last5_w"]
        f_sub = form[[c for c in f_cols if c in form.columns]].copy()
        f_sub = f_sub.rename(columns={c: f"{side}_{c}" for c in f_sub.columns if c != "team_id"})
        df = df.merge(f_sub, left_on=f"{side}_team_id", right_on="team_id", how="left").drop(columns="team_id")

    # Final columns
    cols = [
        "game_id", "game_date", "home_team", "away_team", "spread_line", "total_line",
        "ens_spread", "ens_total", "mc_home_win_pct",
        "ens_spread_edge_pts", "ens_total_edge_pts", "ens_ml_edge_prob",
        "home_adj_net_rtg", "home_tempo", "home_last5_net", "home_last10_net",
        "away_adj_net_rtg", "away_tempo", "away_last5_net", "away_last10_net"
    ]
    # Handle aliases
    df = df.rename(columns={"pred_spread": "ens_spread", "pred_total": "ens_total"})

    present_cols = [c for c in cols if c in df.columns]
    _write(df[present_cols], "matchup_preview.csv", ["predictions", "form", "summary"])


def build_player_leaders_csv() -> None:
    """Build player_leaders.csv: Season player leaderboard."""
    pss = _load(DATA / "player_season_summary.csv", "player_season_summary")
    if pss is None:
        # Fallback to aggregation from logs if possible
        return

    # Normalize column names if needed
    col_map = {
        "avg_min": "mpg",
        "avg_pts": "ppg",
        "avg_reb": "rpg",
        "avg_ast": "apg",
        "pts_avg": "ppg",
        "reb_avg": "rpg",
        "ast_avg": "apg",
        "min_avg": "mpg",
    }
    for old, new in col_map.items():
        if old in pss.columns and new not in pss.columns:
            pss[new] = pss[old]

    # Filter - require some meaningful participation
    # In test environments with limited games, we lower the threshold.
    min_games = 1
    min_mpg = 5.0

    mask = pd.Series(True, index=pss.index)
    if "games_played" in pss.columns:
        mask &= (pss["games_played"] >= min_games)
    if "mpg" in pss.columns:
        mask &= (pss["mpg"] >= min_mpg)

    df = pss[mask].copy()

    # PPG, RPG, APG
    if "ppg" in df.columns: df["ppg"] = df["ppg"].round(1)
    if "rpg" in df.columns: df["rpg"] = df["rpg"].round(1)
    if "apg" in df.columns: df["apg"] = df["apg"].round(1)
    if "mpg" in df.columns: df["mpg"] = df["mpg"].round(1)

    # Ranks
    df["rank_ppg"] = df["ppg"].rank(ascending=False, method="min")
    df["rank_rpg"] = df["rpg"].rank(ascending=False, method="min")
    df["rank_apg"] = df["apg"].rank(ascending=False, method="min")

    cols = ["team", "player_id", "player_name", "games_played", "mpg", "ppg", "rpg", "apg", "ts_pct", "rank_ppg", "rank_rpg", "rank_apg"]
    present = [c for c in cols if c in df.columns]
    _write(df[present], "player_leaders.csv", ["player_season_summary"])


def build_upset_watch_csv(predictions: pd.DataFrame) -> None:
    """Build upset_watch.csv: Upset watch list."""
    df = predictions.copy()

    # One row per game where market implies meaningful underdog (spread >= 4)
    if "spread_line" not in df.columns: return

    df["abs_spread"] = df["spread_line"].abs()
    dogs = df[df["abs_spread"] >= 4.0].copy()

    if dogs.empty: return

    # Formula components
    # favorite_team / underdog_team
    dogs["favorite_team"] = np.where(dogs["spread_line"] < 0, dogs["home_team"], dogs["away_team"])
    dogs["underdog_team"] = np.where(dogs["spread_line"] < 0, dogs["away_team"], dogs["home_team"])

    # underdog_win_prob
    dogs["underdog_win_prob"] = np.where(dogs["spread_line"] < 0, dogs["mc_away_win_pct"], dogs["mc_home_win_pct"])

    # underdog_spread_edge_pts
    # If home is fav (line < 0), edge for away = model_spread - line (if model > line, e.g. -2 vs -6 -> +4 edge)
    # If away is fav (line > 0), edge for home = line - model_spread (if model < line, e.g. 2 vs 6 -> +4 edge)
    dogs["model_spread"] = dogs["ens_ens_spread"].combine_first(dogs["pred_spread"])
    dogs["underdog_spread_edge_pts"] = np.where(
        dogs["spread_line"] < 0,
        dogs["model_spread"] - dogs["spread_line"],
        dogs["spread_line"] - dogs["model_spread"]
    )

    # UWS Score
    dogs["uws_prob"] = dogs["underdog_win_prob"]
    dogs["uws_edge"] = (dogs["underdog_spread_edge_pts"].clip(lower=0) / 10).clip(upper=1.0)
    dogs["uws_disagree"] = ((dogs["model_spread"] - dogs["spread_line"]).abs() / 10).clip(upper=1.0)

    dogs["uws_total"] = (0.45 * dogs["uws_prob"] + 0.35 * dogs["uws_edge"] + 0.20 * dogs["uws_disagree"]) * 100

    # Filter
    watch = dogs[dogs["uws_total"] >= 40].sort_values("uws_total", ascending=False)

    cols = ["game_id", "game_date", "favorite_team", "underdog_team", "spread_line", "model_spread",
            "underdog_win_prob", "underdog_spread_edge_pts", "uws_prob", "uws_edge", "uws_disagree", "uws_total"]
    present = [c for c in cols if c in watch.columns]
    _write(watch[present], "upset_watch.csv", ["predictions_mc_latest"])


def build_conference_summary_csv() -> None:
    """Build conference_summary.csv: Conference standings snapshot."""
    games = _load(DATA / "games.csv", "games")
    if games is None: return

    # Final only
    finals = games[games["completed"].astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    if finals.empty: return

    # Need team -> conference map
    summary = _load(DATA / "team_season_summary.csv")
    if summary is None: return
    conf_map = summary.set_index("team_id")["conference"].to_dict()

    standings = []
    # This is a bit heavy, maybe just use team_season_summary instead if it has what we need?
    # team_season_summary has wins, losses, conference, etc.

    for _, row in summary.iterrows():
        standings.append({
            "as_of_date": TODAY_STAMP,
            "conference": row.get("conference"),
            "team": row.get("team"),
            "conf_w": 0, # Placeholder, needs conf game flag
            "conf_l": 0,
            "conf_pct": 0,
            "overall_w": row.get("wins"),
            "overall_l": row.get("losses"),
            "overall_pct": round(row.get("wins") / (row.get("wins") + row.get("losses")), 3) if (row.get("wins") + row.get("losses")) > 0 else 0,
            "streak": row.get("cover_streak", 0) # proxy or use actual win streak if available
        })

    _write(pd.DataFrame(standings), "conference_summary.csv", ["team_season_summary"])


def main() -> None:
    """Build all derived CSVs, enriching with matchup data when available."""

    # 0. Load master prediction source
    preds = _load(DATA / "predictions_mc_latest.csv", "predictions_mc_latest")
    if preds is None:
        preds = _load(DATA / "predictions_latest.csv", "predictions_latest")

    if preds is not None:
        # 1. Recommendations
        build_bet_recs_csv(preds)

        # 2. Form Snapshot
        build_team_form_snapshot_csv()

        # 3. Matchup Preview
        build_matchup_preview_csv(preds)

        # 4. Player Leaders
        build_player_leaders_csv()

        # 5. Upset Watch
        build_upset_watch_csv(preds)

        # 6. Conference Summary
        build_conference_summary_csv()

    # Enrich matchup_preview.csv and bet_recs.csv if they now exist
    for stem in ("matchup_preview.csv", "bet_recs.csv"):
        for src_dir in (CSV_DIR, DATA):
            p = src_dir / stem
            df = _load(p, stem)
            if df is not None:
                enriched = enrich_with_matchup_summary(df)
                if len(enriched.columns) > len(df.columns):
                    _write(enriched, stem, ["team_matchup_summary"])
                    print(f"[OK]  {stem}: enriched with matchup summary columns")
                break  # only process once per stem

    # Print results summary
    if _results:
        print("\n" + "="*50)
        print(f"{'Derived File':<25} {'Rows':>6} {'Status'}")
        print("-"*50)
        for r in _results:
            print(f"{r['file']:<25} {r['rows']:>6} {r['status']}")
        print("="*50)


if __name__ == "__main__":
    main()
