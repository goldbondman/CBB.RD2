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
# home/away splits

from __future__ import annotations

import pathlib
import argparse
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

# Runtime overrideable window config (defaults preserve current behavior intent)
WINDOW_START_HOURS = 0.0
WINDOW_BEHIND_HOURS = 0.0
WINDOW_AHEAD_HOURS = 30.0
WINDOW_TIMEZONE = "America/Los_Angeles"
BET_RECS_WINDOW_HOURS = 40.0


EXPECTED_OUTPUT_SCHEMAS = {
    "bet_recs.csv": [
        "game_id", "event_id", "game_datetime_utc", "home_team", "away_team",
        "spread_pick", "spread_edge", "total_pick", "total_edge",
        "ml_pick", "ml_edge", "recommendation_tier",
    ],
    "team_form_snapshot.csv": [
        "team_id", "team", "conference", "games_last_10", "record_last_10",
        "avg_margin_last_10", "form_score",
    ],
    "matchup_preview.csv": [
        "game_id", "event_id", "game_datetime_utc", "home_team", "away_team",
        "home_team_id", "away_team_id", "home_adj_net", "away_adj_net",
        "pred_spread", "pred_total",
    ],
    "player_leaders.csv": [
        "player_id", "player_name", "team_id", "team", "conference",
        "games", "minutes", "points", "rebounds", "assists",
    ],
    "upset_watch.csv": [
        "game_id", "event_id", "game_datetime_utc", "favorite_team", "underdog_team",
        "market_spread", "model_spread", "upset_probability", "uws_total",
    ],
    "conference_summary.csv": [
        "conference", "team_id", "team", "wins", "losses", "conference_win_pct",
        "ats_wins", "ats_losses", "ats_pct",
    ],
}



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


def _select_prediction_source() -> tuple[Optional[pd.DataFrame], Optional[str]]:
    """Load the freshest predictions source available for derived outputs."""
    candidates: list[tuple[str, pathlib.Path]] = [
        ("predictions_latest", DATA / "predictions_latest.csv"),
        ("predictions_mc_latest", DATA / "predictions_mc_latest.csv"),
    ]
    best_df: Optional[pd.DataFrame] = None
    best_label: Optional[str] = None
    best_ts: pd.Timestamp | None = None

    for label, path in candidates:
        df = _load(path, label)
        if df is None:
            continue

        max_ts: pd.Timestamp | None = None
        for ts_col in ["game_time_utc", "game_datetime_utc", "generated_at_utc", "generated_at"]:
            if ts_col in df.columns:
                ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
                if ts.notna().any():
                    max_ts = ts.max()
                    break

        if best_df is None:
            best_df, best_label, best_ts = df, label, max_ts
            continue

        if best_ts is None and max_ts is not None:
            best_df, best_label, best_ts = df, label, max_ts
            continue

        if max_ts is not None and best_ts is not None and max_ts > best_ts:
            best_df, best_label, best_ts = df, label, max_ts
            continue

        if max_ts == best_ts and len(df) > len(best_df):
            best_df, best_label, best_ts = df, label, max_ts

    if best_label is not None:
        freshness = best_ts.isoformat() if best_ts is not None else "unknown"
        print(f"[INFO] using {best_label} for derived CSVs (freshness={freshness})")
    return best_df, best_label


def _filter_upcoming_window(
    df: pd.DataFrame,
    label: str,
    start_hours: float,
    behind_hours: float,
    ahead_hours: float,
    timezone_name: str,
) -> pd.DataFrame:
    """Filter rows into a rolling time window in the requested timezone."""
    if ahead_hours <= 0:
        print(f"[WARN] {label}: window-ahead-hours <= 0 ({ahead_hours}); returning empty dataset")
        return df.iloc[0:0].copy()

    dt_candidates = [
        "game_time_utc",
        "game_datetime_utc",
        "game_datetime",
        "date",
        "game_date",
    ]
    dt_col = next((col for col in dt_candidates if col in df.columns), None)
    if dt_col is None:
        print(f"[WARN] {label}: no datetime column found in {dt_candidates}; unable to enforce time window")
        return df

    tz = ZoneInfo(timezone_name)
    run_local = NOW_UTC.astimezone(tz)
    window_start = run_local - timedelta(hours=float(behind_hours)) + timedelta(hours=float(start_hours))
    window_end = run_local + timedelta(hours=float(ahead_hours))

    dt_local = pd.to_datetime(df[dt_col], errors="coerce", utc=True).dt.tz_convert(tz)
    before_count = len(df)
    invalid_count = int(dt_local.isna().sum())
    mask = dt_local.notna() & (dt_local >= window_start) & (dt_local <= window_end)
    filtered = df[mask].copy()

    print(
        f"[INFO] {label}: rolling window filter",
        f"timezone={timezone_name}",
        f"start={window_start.isoformat()}",
        f"end={window_end.isoformat()}",
        f"behind_hours={float(behind_hours)}",
        f"kept={len(filtered)}",
        f"dropped_outside={before_count - len(filtered) - invalid_count}",
        f"dropped_invalid_datetime={invalid_count}",
        f"datetime_col={dt_col}",
    )
    return filtered


def _parse_game_time_to_utc(raw_value: object) -> tuple[pd.Timestamp, str | None]:
    """Parse a single game timestamp into an aware UTC pandas Timestamp.

    Policy: timezone-naive timestamps are rejected (not assumed UTC).
    """
    if pd.isna(raw_value):
        return pd.NaT, "empty"

    if isinstance(raw_value, pd.Timestamp):
        ts = raw_value
        if ts.tzinfo is None:
            return pd.NaT, "naive_timestamp_rejected"
        return ts.tz_convert("UTC"), None

    if isinstance(raw_value, datetime):
        if raw_value.tzinfo is None:
            return pd.NaT, "naive_timestamp_rejected"
        return pd.Timestamp(raw_value.astimezone(timezone.utc)), None

    text = str(raw_value).strip()
    if not text:
        return pd.NaT, "empty"

    try:
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        dt_value = datetime.fromisoformat(normalized)
    except ValueError:
        return pd.NaT, "unparseable"

    if dt_value.tzinfo is None:
        return pd.NaT, "naive_timestamp_rejected"

    return pd.Timestamp(dt_value.astimezone(timezone.utc)), None


def _filter_bet_recs_window(df: pd.DataFrame) -> pd.DataFrame:
    """Strict bet recs window filter: [now_local, now_local+40h) in UTC."""
    dt_candidates = ["game_datetime_utc", "game_datetime", "start_time", "commence_time", "game_time", "date", "game_date"]
    dt_col = next((col for col in dt_candidates if col in df.columns), None)
    if dt_col is None:
        raise RuntimeError(f"bet_recs: missing datetime column; expected one of {dt_candidates}")

    tz_local = ZoneInfo("America/Los_Angeles")
    run_time_local = NOW_UTC.astimezone(tz_local)
    window_start_utc = run_time_local.astimezone(timezone.utc)
    window_end_utc = (run_time_local + timedelta(hours=BET_RECS_WINDOW_HOURS)).astimezone(timezone.utc)

    parsed_values: list[pd.Timestamp] = []
    parse_failures = 0
    before_count = len(df)

    for _, row in df.iterrows():
        parsed, reason = _parse_game_time_to_utc(row.get(dt_col))
        parsed_values.append(parsed)
        if reason is not None:
            parse_failures += 1
            print(
                f"[WARN] bet_recs: dropping row due to invalid game time "
                f"game_id={row.get('game_id', row.get('event_id', '<unknown>'))} "
                f"raw={row.get(dt_col)!r} reason={reason}"
            )

    out = df.copy()
    out["_game_time_utc"] = pd.to_datetime(parsed_values, utc=True, errors="coerce")

    if before_count > 0 and (parse_failures / before_count) > 0.30:
        raise RuntimeError(
            "bet_recs: parse failure ratio exceeds 30% "
            f"({parse_failures}/{before_count}); likely upstream schema/timestamp format change"
        )

    if out["_game_time_utc"].notna().any() and out["_game_time_utc"].max() < pd.Timestamp(window_start_utc):
        raise RuntimeError(
            "bet_recs: source data appears stale; latest game time is before current run window start "
            f"(latest={out['_game_time_utc'].max().isoformat()}, window_start_utc={window_start_utc.isoformat()})"
        )

    mask = (
        out["_game_time_utc"].notna()
        & (out["_game_time_utc"] >= pd.Timestamp(window_start_utc))
        & (out["_game_time_utc"] < pd.Timestamp(window_end_utc))
    )
    filtered = out.loc[mask].copy()

    min_time = filtered["_game_time_utc"].min()
    max_time = filtered["_game_time_utc"].max()
    print("[DEBUG] bet_recs time window")
    print(f"  run_time_local={run_time_local.isoformat()}")
    print(f"  window_start_utc={window_start_utc.isoformat()}")
    print(f"  window_end_utc={window_end_utc.isoformat()}")
    print(f"  min_game_time_utc={min_time.isoformat() if pd.notna(min_time) else 'None'}")
    print(f"  max_game_time_utc={max_time.isoformat() if pd.notna(max_time) else 'None'}")
    print(f"  count_before_filter={before_count}")
    print(f"  count_after_filter={len(filtered)}")

    if not filtered.empty:
        preview = filtered[[c for c in ["home_team", "away_team", "_game_time_utc"] if c in filtered.columns]].head(10)
        print("[DEBUG] bet_recs first 10 filtered rows")
        print(preview.to_string(index=False))

    return filtered

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
            tms[col] = tms[col].astype(str).str.strip().str.lstrip("0").str.replace(r"^$", "0", regex=True)

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
            df[c] = df[c].astype(str).str.strip().str.lstrip("0").str.replace(r"^$", "0", regex=True)
        df = df.merge(home_tms, on=["event_id", "home_team_id"], how="left")

    if "away_team_id" in df.columns and "event_id" in df.columns:
        away_tms = tms.rename(columns={c: f"away_{c}" for c in pick_cols})
        away_tms = away_tms.rename(columns={"team_id": "away_team_id"})
        join_cols_a = [f"away_{c}" for c in pick_cols]
        away_tms = away_tms[["event_id", "away_team_id"] + join_cols_a].copy()
        for c in ("event_id", "away_team_id"):
            df[c] = df[c].astype(str).str.strip().str.lstrip("0").str.replace(r"^$", "0", regex=True)
        df = df.merge(away_tms, on=["event_id", "away_team_id"], how="left")

    # Compute matchup_pts_edge (positive = conditions favour home outscoring baseline)
    if "home_expected_pts_impact" in df.columns and "away_expected_pts_impact" in df.columns:
        df["matchup_pts_edge"] = (
            pd.to_numeric(df["home_expected_pts_impact"], errors="coerce") -
            pd.to_numeric(df["away_expected_pts_impact"], errors="coerce")
        ).round(2)

    return df


def _norm_id(series: pd.Series) -> pd.Series:
    """Normalize join keys to strings without trailing .0 artifacts."""
    return series.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def enrich_ensemble_team_names() -> None:
    """Backfill team names + matchup text on ensemble prediction exports.

    Integrity behavior:
      - Never drops rows.
      - Only fills missing/blank home_team and away_team.
      - Writes updated files only when at least one missing name is recovered.
    """
    ensemble_paths = [
        p for p in sorted(DATA.glob("ensemble_predictions*.csv"))
        if p.exists() and p.stat().st_size > 10
    ]
    if not ensemble_paths:
        return

    lookup_sources: list[pd.DataFrame] = []
    for p in (DATA / "predictions_latest.csv", DATA / "games.csv"):
        df = _load(p, p.stem)
        if df is None:
            continue
        required = {"game_id", "home_team", "away_team"}
        if required.issubset(df.columns):
            keep = [c for c in ["game_id", "home_team_id", "away_team_id", "home_team", "away_team"] if c in df.columns]
            lookup_sources.append(df[keep].copy())

    if not lookup_sources:
        print("[WARN] ensemble enrichment: no lookup source with team names")
        return

    lookup = pd.concat(lookup_sources, ignore_index=True)
    lookup["game_id"] = _norm_id(lookup["game_id"])
    if "home_team_id" in lookup.columns:
        lookup["home_team_id"] = _norm_id(lookup["home_team_id"])
    if "away_team_id" in lookup.columns:
        lookup["away_team_id"] = _norm_id(lookup["away_team_id"])
    lookup["home_team"] = lookup["home_team"].astype(str).str.strip()
    lookup["away_team"] = lookup["away_team"].astype(str).str.strip()
    lookup = lookup[(lookup["home_team"] != "") & (lookup["away_team"] != "")]
    lookup = lookup.drop_duplicates(subset=["game_id"], keep="first")

    for ensemble_path in ensemble_paths:
        ens = _load(ensemble_path, ensemble_path.name)
        if ens is None or "game_id" not in ens.columns:
            continue

        ens = ens.copy()
        ens["game_id"] = _norm_id(ens["game_id"])
        if "home_team_id" in ens.columns:
            ens["home_team_id"] = _norm_id(ens["home_team_id"])
        if "away_team_id" in ens.columns:
            ens["away_team_id"] = _norm_id(ens["away_team_id"])

        for c in ("home_team", "away_team"):
            if c not in ens.columns:
                ens[c] = ""
            ens[c] = ens[c].astype(str).replace({"nan": "", "None": ""}).str.strip()

        before_missing = ((ens["home_team"] == "") | (ens["away_team"] == "")).sum()

        merged = ens.merge(
            lookup[["game_id", "home_team", "away_team"]].rename(
                columns={"home_team": "_home_team_lu", "away_team": "_away_team_lu"}
            ),
            on="game_id",
            how="left",
        )

        home_missing = merged["home_team"] == ""
        away_missing = merged["away_team"] == ""
        merged.loc[home_missing, "home_team"] = merged.loc[home_missing, "_home_team_lu"].fillna("")
        merged.loc[away_missing, "away_team"] = merged.loc[away_missing, "_away_team_lu"].fillna("")
        merged = merged.drop(columns=["_home_team_lu", "_away_team_lu"])

        if "matchup" not in merged.columns:
            merged["matchup"] = ""
        matchup_missing = merged["matchup"].astype(str).str.strip().isin(["", "nan", "None"])
        merged.loc[matchup_missing, "matchup"] = (
            merged.loc[matchup_missing, "away_team"].astype(str).str.strip() +
            " @ " +
            merged.loc[matchup_missing, "home_team"].astype(str).str.strip()
        ).str.strip()
        merged.loc[merged["matchup"].str.startswith("@") | merged["matchup"].str.endswith("@"), "matchup"] = ""

        after_missing = ((merged["home_team"] == "") | (merged["away_team"] == "")).sum()
        recovered = int(before_missing - after_missing)
        if recovered > 0:
            safe_write_csv(merged, ensemble_path, index=False)
            print(f"[OK]  {ensemble_path.name}: backfilled team names for {recovered} rows")


def build_bet_recs_csv(predictions: pd.DataFrame) -> None:
    """Build bet_recs.csv: Betting recommendations with edge flags."""
    df = predictions.copy()

    # Filter to scheduled/not final
    if "is_final" in df.columns:
        df = df[df["is_final"] == 0]
    elif "game_status" in df.columns:
        df = df[df["game_status"].astype(str).str.lower() == "scheduled"]

    df = _filter_bet_recs_window(df)

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
            # Identify model triggers (which models agree with this pick)
            triggers = []
            sub_models = [
                ("M1_FF", "fourfactors_spread"),
                ("M2_Eff", "adjefficiency_spread"),
                ("M3_Pyth", "pythagorean_spread"),
                ("M4_Mom", "momentum_spread"),
                ("M5_ATS", "atsintelligence_spread"),
                ("M6_Sit", "situational_spread"),
                ("M7_CAGE", "cagerankings_spread"),
                ("M8_HA", "regressedeff_spread"),
            ]
            for label, col in sub_models:
                if col in row and pd.notna(row[col]):
                    sub_val = float(row[col])
                    # If model and consensus pick the same side
                    if (pick == "HOME" and sub_val < 0) or (pick == "AWAY" and sub_val > 0):
                        triggers.append(label)

            recs.append({
                "game_id": row.get("game_id"),
                "date": row.get("_game_time_utc", row.get("game_datetime_utc")),
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
                "confidence": row.get("mc_confidence_tier", "MEDIUM"),
                "triggers": "|".join(triggers) if triggers else "CONSENSUS"
            })

        # Total rec
        if tot_edge_col and abs(row.get(tot_edge_col, 0)) >= TOTAL_EDGE_MIN:
            edge = row.get(tot_edge_col)
            model_total = next((row.get(c) for c in ["ens_total", "pred_total"] if c in row), 140)
            market_total = row.get("total_line", row.get("market_total", 140))
            pick = "OVER" if model_total > market_total else "UNDER"
            recs.append({
                "game_id": row.get("game_id"),
                "date": row.get("_game_time_utc", row.get("game_datetime_utc")),
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
    df = _filter_upcoming_window(
        df,
        label="matchup_preview",
        start_hours=WINDOW_START_HOURS,
        behind_hours=WINDOW_BEHIND_HOURS,
        ahead_hours=WINDOW_AHEAD_HOURS,
        timezone_name=WINDOW_TIMEZONE,
    )

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
    df = _filter_upcoming_window(
        df,
        label="upset_watch",
        start_hours=WINDOW_START_HOURS,
        behind_hours=WINDOW_BEHIND_HOURS,
        ahead_hours=WINDOW_AHEAD_HOURS,
        timezone_name=WINDOW_TIMEZONE,
    )

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
    spread_candidates = [
        "ens_ens_spread",
        "pred_spread",
        "predicted_spread",
        "mc_spread_median",
    ]
    spread_candidates = [c for c in spread_candidates if c in dogs.columns]
    dogs["model_spread"] = np.nan
    selected_spread_col = None
    for col in spread_candidates:
        candidate = pd.to_numeric(dogs[col], errors="coerce")
        unique_values = candidate.dropna().nunique()
        if selected_spread_col is None and candidate.notna().any():
            dogs["model_spread"] = candidate
            selected_spread_col = col
            continue
        # Data quality guard: avoid using a degenerate near-constant series
        # when a later fallback has per-game variation.
        if candidate.notna().any() and unique_values > 1 and pd.to_numeric(dogs["model_spread"], errors="coerce").dropna().nunique() <= 1:
            dogs["model_spread"] = candidate
            selected_spread_col = col

    if selected_spread_col:
        print(f"[INFO] upset_watch: model_spread source = {selected_spread_col}")

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




def write_stub_output(stem: str, reason: str) -> None:
    """Write an empty, schema-correct stub output for a documented derived CSV."""
    cols = EXPECTED_OUTPUT_SCHEMAS.get(stem, [])
    print(f"[WARN] {stem}: stub writer active — {reason}")
    _write(pd.DataFrame(columns=cols), stem, ["stub"], dated_copy=False)


def ensure_documented_outputs() -> None:
    """Ensure all six documented outputs exist with at least schema headers."""
    for stem in EXPECTED_OUTPUT_SCHEMAS:
        csv_path = CSV_DIR / stem
        root_path = DATA / stem
        if not csv_path.exists() or csv_path.stat().st_size < 10 or not root_path.exists() or root_path.stat().st_size < 10:
            write_stub_output(stem, "builder unavailable or produced no output")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build derived CBB CSVs")
    parser.add_argument(
        "--window-ahead-hours",
        type=float,
        default=30.0,
        help="How many hours ahead from run time to keep games (default: 30).",
    )
    parser.add_argument(
        "--window-start-hours",
        type=float,
        default=0.0,
        help="Offset in hours from run time before starting the filter window (default: 0).",
    )
    parser.add_argument(
        "--window-behind-hours",
        type=float,
        default=0.0,
        help="Hours behind run time to include before the window start (default: 0).",
    )
    parser.add_argument(
        "--window-timezone",
        type=str,
        default="America/Los_Angeles",
        help="IANA timezone used for window calculations (default: America/Los_Angeles).",
    )
    return parser.parse_args()


def main() -> None:
    """Build all derived CSVs, enriching with matchup data when available."""
    global WINDOW_AHEAD_HOURS, WINDOW_START_HOURS, WINDOW_BEHIND_HOURS, WINDOW_TIMEZONE
    args = parse_args()
    WINDOW_AHEAD_HOURS = args.window_ahead_hours
    WINDOW_START_HOURS = args.window_start_hours
    WINDOW_BEHIND_HOURS = args.window_behind_hours
    WINDOW_TIMEZONE = args.window_timezone

    # 0. Load freshest prediction source
    preds, _pred_source = _select_prediction_source()

    if preds is not None:
        enrich_ensemble_team_names()

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
    else:
        print("[WARN] predictions input unavailable — writing documented derived CSV stubs")
        for stem in EXPECTED_OUTPUT_SCHEMAS:
            write_stub_output(stem, "predictions source missing")

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

    ensure_documented_outputs()

    for r in _results:
        print(f"[BUILD] {r}")

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
