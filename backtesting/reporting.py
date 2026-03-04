#!/usr/bin/env python3
"""
backtesting/reporting.py
=========================
CSV output writers and console report for backtest results.

Outputs:
  signals_library.py    — qualified signals as Python constants
  user_signals.csv      — 59%+ signals with stale tracking
  backtest_results.csv  — ALL tested combos (appended each run)
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timezone
from typing import Any

import pandas as pd

DATA     = pathlib.Path("data")
ROOT     = pathlib.Path(".")
NOW_UTC  = datetime.now(timezone.utc)
RUN_DATE = NOW_UTC.strftime("%Y-%m-%d")
RUN_TS   = NOW_UTC.strftime("%Y%m%dT%H%M%SZ")


# ─────────────────────────────────────────────────────────────────────────────
# signals_library.py
# ─────────────────────────────────────────────────────────────────────────────

def write_signals_library(
    qualified: list[dict[str, Any]],
    path: pathlib.Path = ROOT / "signals_library.py",
) -> None:
    """Write qualified signals to signals_library.py as Python constants."""
    existing_lines: list[str] = []
    if path.exists() and path.stat().st_size > 0:
        existing_lines = path.read_text(encoding="utf-8").splitlines()
        # Keep any manually added content above the AUTO-GENERATED block
        auto_idx = next(
            (i for i, l in enumerate(existing_lines) if "AUTO-GENERATED" in l), None
        )
        if auto_idx is not None:
            existing_lines = existing_lines[:auto_idx]

    new_blocks = [
        f"# AUTO-GENERATED — {RUN_DATE}",
        f"# Run: {RUN_TS}",
        f"# Signals added: {len(qualified)}",
        "",
    ]

    for i, r in enumerate(qualified):
        combo_id = f"COMBO_{RUN_TS}_{i:03d}"
        signals_repr = repr(list(r["combo"]))
        thresh_repr  = json.dumps(r.get("thresholds", {}), indent=4)
        block = (
            f"# SU={r.get('hit_rate_su',0):.3f}  "
            f"holdout={r.get('holdout_su') or 'N/A'}  "
            f"ATS={r.get('hit_rate_ats') or 'N/A'}  "
            f"n={r.get('n',0)}  "
            f"p={r.get('p_value',1):.5f}  "
            f"fire_rate={r.get('fire_rate',0):.3f}  "
            f"seasons={r.get('seasons_passing',0)}/1\n"
            f"SIGNAL_{combo_id} = {{\n"
            f"    'signals':         {signals_repr},\n"
            f"    'thresholds':      {thresh_repr},\n"
            f"    'hit_rate_su':     {r.get('hit_rate_su',0):.4f},\n"
            f"    'hit_rate_ats':    {r.get('hit_rate_ats') or 'None'},\n"
            f"    'holdout_su':      {r.get('holdout_su') or 'None'},\n"
            f"    'holdout_ats':     {r.get('holdout_ats') or 'None'},\n"
            f"    'p_value':         {r.get('p_value',1):.6f},\n"
            f"    'n':               {r.get('n',0)},\n"
            f"    'fire_rate':       {r.get('fire_rate',0):.4f},\n"
            f"    'seasons_passing': {r.get('seasons_passing',0)},\n"
            f"    'lx_days':         {r.get('lx_days', False)},\n"
            f"    'consistent':      {r.get('consistent', False)},\n"
            f"}}\n"
        )
        new_blocks.append(block)

    content = "\n".join(existing_lines) + "\n" + "\n".join(new_blocks) + "\n"
    path.write_text(content, encoding="utf-8")
    print(f"[write] {path} — {len(qualified)} signals")


# ─────────────────────────────────────────────────────────────────────────────
# user_signals.csv
# ─────────────────────────────────────────────────────────────────────────────

USER_SIGNALS_COLS = [
    "combo_id", "combo_type", "signals", "thresholds",
    "hit_rate_su", "hit_rate_ats", "holdout_su", "holdout_ats",
    "p_value", "n", "fire_rate", "seasons_passing", "lx_days_flag",
    "season_hit_rates", "date_added", "auto_merged", "stale",
]


def write_user_csv(
    qualified: list[dict[str, Any]],
    path: pathlib.Path = ROOT / "user_signals.csv",
) -> None:
    """Append qualified signals to user_signals.csv."""
    rows = []
    for i, r in enumerate(qualified):
        combo_id = f"{RUN_TS}_{i:03d}"
        rows.append({
            "combo_id":        combo_id,
            "combo_type":      r.get("combo_type", "unknown"),
            "signals":         "|".join(r["combo"]),
            "thresholds":      json.dumps(r.get("thresholds", {})),
            "hit_rate_su":     r.get("hit_rate_su"),
            "hit_rate_ats":    r.get("hit_rate_ats"),
            "holdout_su":      r.get("holdout_su"),
            "holdout_ats":     r.get("holdout_ats"),
            "p_value":         r.get("p_value"),
            "n":               r.get("n"),
            "fire_rate":       r.get("fire_rate"),
            "seasons_passing": r.get("seasons_passing", 0),
            "lx_days_flag":    r.get("lx_days", False),
            "season_hit_rates": json.dumps(r.get("season_hit_rates", {})),
            "date_added":      RUN_DATE,
            "auto_merged":     True,
            "stale":           False,
        })

    if not rows:
        print("[write] user_signals.csv — 0 qualified signals, skipping append")
        return

    new_df = pd.DataFrame(rows, columns=USER_SIGNALS_COLS)

    if path.exists() and path.stat().st_size > 10:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    print(f"[write] {path} — {len(new_df)} new rows ({len(combined)} total)")


# ─────────────────────────────────────────────────────────────────────────────
# backtest_results.csv
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_COLS = [
    "combo_id", "combo_type", "signals", "thresholds",
    "n", "fire_rate",
    "hit_rate_su", "hit_rate_ats", "holdout_su", "holdout_ats",
    "p_value", "consistent", "seasons_passing", "lx_days", "merged",
    "timestamp",
]


def write_backtest_results(
    all_results: list[dict[str, Any]],
    path: pathlib.Path = ROOT / "backtest_results.csv",
) -> None:
    """Append ALL tested combos to backtest_results.csv (sorted by SU hit rate)."""
    rows = []
    for i, r in enumerate(all_results):
        combo_id = f"{RUN_TS}_{i:05d}"
        rows.append({
            "combo_id":        combo_id,
            "combo_type":      r.get("combo_type", "unknown"),
            "signals":         "|".join(str(s) for s in r.get("combo", [])),
            "thresholds":      json.dumps(r.get("thresholds", {})),
            "n":               r.get("n", 0),
            "fire_rate":       r.get("fire_rate", 0),
            "hit_rate_su":     r.get("hit_rate_su", 0),
            "hit_rate_ats":    r.get("hit_rate_ats"),
            "holdout_su":      r.get("holdout_su"),
            "holdout_ats":     r.get("holdout_ats"),
            "p_value":         r.get("p_value"),
            "consistent":      r.get("consistent", False),
            "seasons_passing": r.get("seasons_passing", 0),
            "lx_days":         r.get("lx_days", False),
            "merged":          r.get("merged", False),
            "timestamp":       RUN_TS,
        })

    new_df = pd.DataFrame(rows, columns=RESULTS_COLS)
    new_df = new_df.sort_values("hit_rate_su", ascending=False)

    if path.exists() and path.stat().st_size > 10:
        existing = pd.read_csv(path)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(path, index=False)
    print(f"[write] {path} — {len(new_df)} new rows ({len(combined)} total)")


# ─────────────────────────────────────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────────────────────────────────────

def print_final_report(summary: dict[str, Any]) -> None:
    qualified  = summary.get("qualified", [])
    validated  = summary.get("validated", [])
    smoke_ok   = sum(1 for r in summary.get("smoke_results", []) if r.get("n", 0) >= 30)
    lx_merged  = sum(1 for r in qualified if r.get("lx_days"))
    n_pair_p2  = sum(1 for r in validated
                     if r.get("combo_type") == "pair" and r.get("hit_rate_su", 0) >= 0.55)

    print("\n" + "=" * 60)
    print(f"BACKTEST COMPLETE — {RUN_DATE}")
    print("=" * 60)
    print(f"Stale signals detected:   (see stale.py)")
    print(f"Smoke test:               {smoke_ok}/20 valid")
    print()
    print(f"Pair Pass 1:              {summary.get('n_pairs', 0)} tested")
    print(f"Pair Pass 2 (refined):    {n_pair_p2} re-run (55%+ from Pass 1)")
    print(f"Trios:                    {summary.get('n_trios', 0)} tested "
          f"({summary.get('n_qualifying_pairs', 0)} qualifying pairs)")
    print(f"Quads:                    {summary.get('n_quads', 0)} (priority list only)")
    print()
    print(f"Skipped (collinear):      {summary.get('n_collinear', 0)}")
    print(f"Skipped (fire rate):      {summary.get('n_low_fire', 0)}")
    print(f"Skipped (p >= 0.05):      {summary.get('n_insig', 0)}")
    print(f"Skipped (1-season):       {summary.get('n_inconsistent', 0)}")
    print()
    print(f"Passed all merge gates:   {len(qualified)}")
    print(f"Lx-days combos merged:    {lx_merged}")

    if qualified:
        print(f"\nTop 10 by SU hit rate:")
        header = f"{'Combo':<55}  {'Type':5}  {'n':5}  {'FR':5}  {'SU':5}  {'Ho-SU':5}  {'ATS':5}  {'p':7}  {'Lx':3}  {'S':3}"
        print(header)
        print("-" * len(header))
        for r in qualified[:10]:
            ats  = f"{r.get('hit_rate_ats', 0) or 0:.3f}"
            ho   = f"{r.get('holdout_su') or 0:.3f}"
            pval = f"{r.get('p_value', 1):.4f}"
            print(
                f"  {' + '.join(r['combo'])[:53]:<55}  "
                f"{r.get('combo_type',''):5}  "
                f"{r.get('n',0):5d}  "
                f"{r.get('fire_rate',0):5.3f}  "
                f"{r.get('hit_rate_su',0):5.3f}  "
                f"{ho:5}  "
                f"{ats:5}  "
                f"{pval:7}  "
                f"{'Y' if r.get('lx_days') else 'N':3}  "
                f"{r.get('seasons_passing',0):3}"
            )

    print()
    print(f"Timeout checkpoint hit:   {'YES' if summary.get('timeout_hit') else 'NO'}")
    print(f"Auto-merged to repo:      {'YES' if qualified else 'NO'}")
    print(f"user_signals.csv updated: YES")
    elapsed = summary.get("elapsed_sec", 0)
    print(f"Elapsed:                  {elapsed/60:.1f} min")
    print("=" * 60)
