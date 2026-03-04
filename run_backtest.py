#!/usr/bin/env python3
"""
run_backtest.py
================
Main entry point for the CBB metric backtesting pipeline.

Execution order (Steps 1–17 from the plan):
  1.  load_metrics()               confirm schema + row counts
  2.  build_rest_flags()           (done in compute_metrics)
  3.  build_signal_direction_map() direction_map.py
  4.  check_stale_signals()        stale.py
  5.  run_smoke_test()             engine.py
  6.  build_signal_pool()          combos.py
  7.  run_pair_pass1()             engine.py
  8.  run_pair_pass2()             engine.py
  9.  run_trio_backtest()          engine.py
  10. run_quad_priority_list()     engine.py
  11. run_statistical_validation() engine.py
  12. filter_qualified()           engine.py (merge gates)
  13. write_signals_library()      reporting.py
  14. write_user_csv()             reporting.py
  15. write_backtest_results()     reporting.py
  16. git_push()                   git_ops.py
  17. print_final_report()         reporting.py

Usage:
  python run_backtest.py
  python run_backtest.py --skip-compute   (if matchup_metrics.csv already exists)
  python run_backtest.py --dry-run        (skip git push)
  python run_backtest.py --smoke-only     (only run smoke test, then exit)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd

DATA = pathlib.Path("data")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CBB Metric Backtester")
    p.add_argument("--skip-compute", action="store_true",
                   help="Skip metric computation (use existing matchup_metrics.csv)")
    p.add_argument("--dry-run", action="store_true",
                   help="Skip git push")
    p.add_argument("--smoke-only", action="store_true",
                   help="Run smoke test only, then exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Step 1: Load / compute metrics ──────────────────────────────────────
    matchup_path = DATA / "matchup_metrics.csv"
    advanced_path = DATA / "team_game_metrics_advanced.csv"

    if args.skip_compute and matchup_path.exists():
        print(f"[step1] --skip-compute: loading {matchup_path}")
        matchup = pd.read_csv(matchup_path, low_memory=False)
        print(f"[step1] matchup_metrics: {len(matchup):,} rows, {len(matchup.columns)} cols")
    else:
        print("[step1] Running metric computation...")
        from backtesting.compute_metrics import run as compute_run
        _, matchup = compute_run(write_files=True)

    # Confirm required columns
    required = ["home_win", "game_dt"]
    missing = [c for c in required if c not in matchup.columns]

    # home_win might be named differently
    if "home_win" not in matchup.columns and "win" in matchup.columns:
        matchup = matchup.rename(columns={"win": "home_win"})
    if "game_dt" not in matchup.columns and "game_datetime_utc" in matchup.columns:
        matchup["game_dt"] = pd.to_datetime(matchup["game_datetime_utc"], utc=True, errors="coerce")

    missing = [c for c in required if c not in matchup.columns]
    if missing:
        print(f"[HALT] matchup_metrics missing required columns: {missing}")
        sys.exit(1)

    # Drop games with no outcome
    matchup = matchup[matchup["home_win"].notna()].copy()
    print(f"[step1] Games with outcomes: {len(matchup):,}")

    # home_cover column
    if "home_cover" not in matchup.columns:
        if "cover" in matchup.columns:
            matchup["home_cover"] = matchup["cover"]
        else:
            matchup["home_cover"] = float("nan")

    # ── Step 3: Signal direction map (validation) ───────────────────────────
    from backtesting.direction_map import validate_registry
    reg = validate_registry(list(matchup.columns))
    print(f"\n[step3] Signal registry: {len(reg['present'])} present, {len(reg['missing'])} missing")
    if reg["missing"]:
        print(f"  Missing: {reg['missing'][:10]}{'...' if len(reg['missing'])>10 else ''}")

    # ── Step 4: Stale signal check ───────────────────────────────────────────
    from backtesting.stale import check_stale_signals
    n_stale = check_stale_signals(matchup)

    if args.smoke_only:
        # ── Smoke test only ──────────────────────────────────────────────────
        from backtesting.thresholds import build_median_thresholds
        from backtesting.engine import run_smoke_test
        median_thresholds = build_median_thresholds(matchup)
        run_smoke_test(matchup, median_thresholds)
        print("\n[smoke-only] Done.")
        return

    # ── Steps 5–12: Full backtest engine ────────────────────────────────────
    from backtesting.engine import run_backtest
    summary = run_backtest(matchup)
    summary["n_stale"] = n_stale

    qualified  = summary["qualified"]
    validated  = summary["validated"]
    timeout_hit = summary.get("timeout_hit", False)

    # ── Step 13: Write signals_library.py ───────────────────────────────────
    from backtesting.reporting import write_signals_library, write_user_csv, write_backtest_results, print_final_report
    write_signals_library(qualified)

    # ── Step 14: Write user_signals.csv ─────────────────────────────────────
    write_user_csv(qualified)

    # ── Step 15: Write backtest_results.csv ─────────────────────────────────
    write_backtest_results(validated)

    # ── Step 16: Git push ────────────────────────────────────────────────────
    if not args.dry_run and not timeout_hit:
        from backtesting.git_ops import git_push
        git_push(n_merged=len(qualified))
    elif args.dry_run:
        print("[git] --dry-run: skipping push")
    elif timeout_hit:
        # Still write checkpoint results but don't push
        write_backtest_results(validated)
        print("[git] Timeout checkpoint written — push deferred to next run")

    # ── Step 17: Final report ────────────────────────────────────────────────
    print_final_report(summary)


if __name__ == "__main__":
    main()
