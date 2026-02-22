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

import pandas as pd

from pipeline_csv_utils import safe_write_csv

# ── Paths ──────────────────────────────────────────────────────────────────
DATA        = pathlib.Path("data")
CSV_DIR     = DATA / "csv"
CSV_DIR.mkdir(parents=True, exist_ok=True)

NOW_UTC     = datetime.now(timezone.utc)
NOW_ISO     = NOW_UTC.strftime("%Y-%m-%dT%H:%M:%SZ")
TODAY_STAMP = NOW_UTC.strftime("%Y%m%d")

# Track results for summary table
_results: list[dict] = []


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

    # Normalise join keys
    for col in ("event_id", "team_id"):
        if col in tms.columns:
            tms[col] = pd.to_numeric(tms[col], errors="coerce")

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
        home_tms["event_id"] = home_tms["event_id"]
        home_tms = home_tms.rename(columns={"team_id": "home_team_id"})
        join_cols_h = [f"home_{c}" for c in pick_cols]
        home_tms = home_tms[["event_id", "home_team_id"] + join_cols_h].copy()
        for c in ("event_id", "home_team_id"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.merge(home_tms, on=["event_id", "home_team_id"], how="left")

    if "away_team_id" in df.columns and "event_id" in df.columns:
        away_tms = tms.rename(columns={c: f"away_{c}" for c in pick_cols})
        away_tms = away_tms.rename(columns={"team_id": "away_team_id"})
        join_cols_a = [f"away_{c}" for c in pick_cols]
        away_tms = away_tms[["event_id", "away_team_id"] + join_cols_a].copy()
        for c in ("event_id", "away_team_id"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.merge(away_tms, on=["event_id", "away_team_id"], how="left")

    # Compute matchup_pts_edge (positive = conditions favour home outscoring baseline)
    if "home_expected_pts_impact" in df.columns and "away_expected_pts_impact" in df.columns:
        df["matchup_pts_edge"] = (
            pd.to_numeric(df["home_expected_pts_impact"], errors="coerce") -
            pd.to_numeric(df["away_expected_pts_impact"], errors="coerce")
        ).round(2)

    return df


def main() -> None:
    """Build all derived CSVs, enriching with matchup data when available."""
    # Enrich matchup_preview.csv and bet_recs.csv if they already exist
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


if __name__ == "__main__":
    main()