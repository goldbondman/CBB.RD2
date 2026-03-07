#!/usr/bin/env python3
"""
apply_signals.py
================
Overlay backtested signals onto the current predictions.

Reads:
  data/predictions_combined_latest.csv   — today's predictions
  data/team_game_metrics_advanced.csv    — latest per-team rolling metrics
  user_signals.csv                       — qualified non-stale signals

Writes:
  data/predictions_with_signals.csv      — predictions + signal columns

Signal columns added:
  signal_n_fired          — how many signals fired for this game
  signal_best_su          — in-sample SU of the top fired signal
  signal_best_holdout_su  — holdout SU of the top fired signal
  signal_best_combo       — metric names of the top fired signal
  signal_top_combos       — top 5 fired combos (semicolon-separated)
  signal_agrees_model     — True if signals and model pick the same side

Usage:
  py apply_signals.py
  py apply_signals.py --predictions data/predictions_20260306.csv
"""

from __future__ import annotations

import argparse
import pathlib
import sys

DATA = pathlib.Path("data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Apply backtested signals to predictions")
    p.add_argument(
        "--predictions",
        type=pathlib.Path,
        default=DATA / "predictions_combined_latest.csv",
        help="Path to predictions CSV (default: data/predictions_combined_latest.csv)",
    )
    p.add_argument(
        "--output",
        type=pathlib.Path,
        default=DATA / "predictions_with_signals.csv",
        help="Output path (default: data/predictions_with_signals.csv)",
    )
    p.add_argument(
        "--signals",
        type=pathlib.Path,
        default=pathlib.Path("user_signals.csv"),
        help="Path to user_signals.csv",
    )
    p.add_argument(
        "--adv-metrics",
        type=pathlib.Path,
        default=DATA / "team_game_metrics_advanced.csv",
        help="Path to team_game_metrics_advanced.csv",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.predictions.exists():
        print(f"[HALT] Predictions file not found: {args.predictions}")
        sys.exit(1)
    if not args.adv_metrics.exists():
        print(f"[HALT] Advanced metrics not found: {args.adv_metrics}")
        sys.exit(1)
    if not args.signals.exists() or args.signals.stat().st_size < 10:
        print("[HALT] user_signals.csv not found or empty — run run_backtest.py first")
        sys.exit(1)

    from backtesting.signal_evaluator import run_signal_overlay

    result = run_signal_overlay(
        predictions_path=args.predictions,
        adv_path=args.adv_metrics,
        signals_path=args.signals,
        output_path=args.output,
    )

    # Print a quick preview of games where signals fired
    fired = result[result["signal_n_fired"] > 0].copy()
    if fired.empty:
        print("\n[signals] No signals fired for any current game.")
        return

    fired = fired.sort_values("signal_n_fired", ascending=False)
    print(f"\n{'='*70}")
    print("GAMES WITH ACTIVE SIGNALS")
    print(f"{'='*70}")
    cols = ["home_team", "away_team", "pred_spread", "home_win_prob",
            "signal_n_fired", "signal_best_holdout_su",
            "signal_agrees_model", "signal_best_combo"]
    cols = [c for c in cols if c in fired.columns]
    print(fired[cols].to_string(index=False))
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
