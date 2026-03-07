#!/usr/bin/env python3
"""
grade_signal_picks.py
======================
Grade tracked signal disagreement picks against actual results.

Reads:
  data/signal_picks_tracker.csv   — picks logged by apply_signals.py
  data/backtest_results_latest.csv — graded game results (actual_margin etc.)

Writes:
  data/signal_picks_tracker.csv   — updated with actual_margin / signal_correct

Run manually or add to cbb_pipeline.yml after grade_predictions step.

Usage:
  py grade_signal_picks.py
  py grade_signal_picks.py --results data/backtest_results_20260306.csv
"""

from __future__ import annotations

import argparse
import pathlib

import pandas as pd

DATA = pathlib.Path("data")
TRACKER_PATH = DATA / "signal_picks_tracker.csv"
RESULTS_PATH = DATA / "backtest_results_latest.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tracker", type=pathlib.Path, default=TRACKER_PATH)
    p.add_argument("--results", type=pathlib.Path, default=RESULTS_PATH)
    return p.parse_args()


def grade(tracker_path: pathlib.Path, results_path: pathlib.Path) -> None:
    if not tracker_path.exists():
        print("[grade] No signal_picks_tracker.csv found — run apply_signals.py first")
        return
    if not results_path.exists():
        print(f"[grade] Results file not found: {results_path}")
        return

    tracker = pd.read_csv(tracker_path)
    results = pd.read_csv(results_path)

    # Merge on game_id
    results_slim = results[["game_id", "actual_margin", "home_score_actual", "away_score_actual"]].drop_duplicates("game_id")
    merged = tracker.merge(results_slim, on="game_id", how="left", suffixes=("", "_new"))

    n_graded = 0
    for idx, row in merged.iterrows():
        margin = row.get("actual_margin_new")
        if pd.isna(margin):
            continue
        if pd.notna(row.get("actual_margin")) and pd.notna(row.get("signal_correct")):
            continue  # already graded

        margin = float(margin)
        merged.at[idx, "actual_margin"]  = margin
        merged.at[idx, "home_score_actual"] = row.get("home_score_actual_new")
        merged.at[idx, "away_score_actual"] = row.get("away_score_actual_new")

        home_won = margin > 0
        model_picked_home = str(row.get("model_pick", "")) == str(row.get("home_team", ""))

        merged.at[idx, "signal_correct"] = bool(home_won)   # signal always picks home
        merged.at[idx, "model_correct"]  = bool(home_won == model_picked_home)
        n_graded += 1

    # Drop the _new merge columns
    merged = merged[[c for c in merged.columns if not c.endswith("_new")]]
    merged.to_csv(tracker_path, index=False)
    print(f"[grade] {n_graded} picks newly graded")

    # Print summary
    graded = merged[merged["signal_correct"].notna()].copy()
    if graded.empty:
        print("[grade] No graded picks yet — results not available")
        return

    sig_acc  = graded["signal_correct"].mean()
    mod_acc  = graded["model_correct"].mean()
    n        = len(graded)
    pending  = len(merged) - n

    print(f"\n{'='*55}")
    print(f"SIGNAL DISAGREEMENT PICK TRACKER")
    print(f"{'='*55}")
    print(f"Graded:  {n}   Pending: {pending}")
    print(f"Signal (home) accuracy:  {sig_acc:.1%}  ({int(graded['signal_correct'].sum())}/{n})")
    print(f"Model accuracy on same:  {mod_acc:.1%}  ({int(graded['model_correct'].sum())}/{n})")
    print()

    # Per-pick breakdown — sort by signal_n_fired desc
    graded = graded.sort_values("signal_n_fired", ascending=False)
    print(f"{'Home team':<32} {'Away team':<28} {'Signals':>7} {'HoldSU':>7} {'Result':>8} {'Sig':>4} {'Mod':>4}")
    print("-" * 95)
    for _, r in graded.iterrows():
        margin = r["actual_margin"]
        winner = r["home_team"] if margin > 0 else r["away_team"]
        result_str = f"{r['home_team']} +{margin:.0f}" if margin > 0 else f"{r['away_team']} +{abs(margin):.0f}"
        sig_ok  = "WIN" if r["signal_correct"] else "LOSS"
        mod_ok  = "WIN" if r["model_correct"]  else "LOSS"
        print(
            f"  {str(r['home_team']):<30} {str(r['away_team']):<28} "
            f"{int(r['signal_n_fired']):>7,} {float(r['signal_best_holdout_su']):>7.3f} "
            f"{result_str[:20]:>20}  {sig_ok:>4}  {mod_ok:>4}"
        )

    print(f"{'='*55}")

    # Highlight upsets where signal was right and model was wrong
    signal_wins_model_loses = graded[(graded["signal_correct"]) & (~graded["model_correct"].astype(bool))]
    if not signal_wins_model_loses.empty:
        print(f"\nSignal RIGHT, Model WRONG ({len(signal_wins_model_loses)} games):")
        for _, r in signal_wins_model_loses.sort_values("signal_n_fired", ascending=False).iterrows():
            print(f"  {r['home_team']} def {r['away_team']}  "
                  f"[{int(r['signal_n_fired']):,} signals, holdout SU {r['signal_best_holdout_su']:.3f}]")


if __name__ == "__main__":
    args = parse_args()
    grade(args.tracker, args.results)
