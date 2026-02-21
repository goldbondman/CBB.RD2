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
        df.to_csv(csv_path,  index=False)
        df.to_csv(root_path, index=False)
        n = len(df)
        print(f"[OK]  {stem}: {n} rows")
        if dated_copy:
            dated = DATA / stem.replace(".csv", f"_{TODAY_STAMP}.csv")
            df.to_csv(dated, index=False)
            print(f"[OK]  {dated.name}: archive copy written")
        _results.append({"file": stem, "rows": n, "sources": ", ".join(sources), "status": "OK"})
    except Exception as exc:
        print(f"[WARN] {stem}: write failed — {exc}")
        traceback.print_exc()
        _results.append({"file": stem, "rows": 0, "sources": ", ".join(sources), "status": f"FAIL: {exc}"})

# ── MC Integration Helpers ─────────────────────────────────────────────────

def _mc_columns_available(df: pd.DataFrame) -> bool:
    """Check if MC enrichment columns are present in the DataFrame."""
    return "mc_cover_probability" in df.columns


def _choose_confidence_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    For bet_recs: prefer mc_cover_probability over raw model_confidence
    when MC output is available.  Adds confidence_source column.
    """
    df = df.copy()
    if _mc_columns_available(df):
        df["confidence_used"] = df["mc_cover_probability"].combine_first(
            df.get("model_confidence", pd.Series(index=df.index, dtype=float))
        )
        df["confidence_source"] = df["mc_cover_probability"].apply(
            lambda v: "MC" if pd.notna(v) else "ENSEMBLE"
        )
    else:
        df["confidence_used"] = df.get(
            "model_confidence", pd.Series(index=df.index, dtype=float)
        )
        df["confidence_source"] = "ENSEMBLE"
    return df


def _choose_upset_probability(df: pd.DataFrame) -> pd.DataFrame:
    """
    For upset_watch: prefer mc_upset_probability over uws_total
    when MC output is available.
    """
    df = df.copy()
    if "mc_upset_probability" in df.columns:
        df["upset_probability_display"] = df["mc_upset_probability"]
        df["upset_source"] = "MC"
    else:
        uws_col = next(
            (c for c in ["uws_total", "ens_uws_total"] if c in df.columns),
            None,
        )
        if uws_col:
            df["upset_probability_display"] = pd.to_numeric(
                df[uws_col], errors="coerce"
            ) / 100.0
            df["upset_source"] = "UWS"
        else:
            df["upset_probability_display"] = None
            df["upset_source"] = "NONE"
    return df


# ... [rest of the script omitted for brevity] ...

if __name__ == "__main__":
    main()