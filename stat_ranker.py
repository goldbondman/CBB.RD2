#!/usr/bin/env python3
"""
stat_ranker.py
==============
Optimizes ATS hit rate across 10 team-differential statistics for all
completed games this season that have a Vegas spread in the CSVs.

Usage:
    python stat_ranker.py

Outputs:
    data/stat_rankings.csv    — per-stat and ensemble performance table
    data/optimal_weights.json — optimized weight vector

Methodology:
    1. Load backtest_training_data.csv, filter to graded ATS games
    2. Compute each stat as (home_value - away_value)
    3. Sort chronologically, split 60% train / 40% test (walk-forward, no lookahead)
    4. Individual stat analysis: hit rate + ROI at -110 on test set
    5. Ensemble optimization: 1000 random weight starts via scipy.optimize
       Objective: maximize ATS hit rate on training set
       Constraint: weights sum to 1, all >= 0
    6. Validate ensemble on test set; flag if overfit (train-test gap > 3pts)
    7. Print full results; save CSVs and JSON
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

DATA_DIR = pathlib.Path("data")
OUT_RANKINGS = DATA_DIR / "stat_rankings.csv"
OUT_WEIGHTS  = DATA_DIR / "optimal_weights.json"
TRAINING_FRAC = 0.60
N_RANDOM_STARTS = 1000
OVERFIT_THRESHOLD = 3.0   # percentage points
JUICE = 1.10              # -110 standard juice

# ── Stat → (home_col, away_col, notes) ────────────────────────────────────────
# For each requested stat, map to home/away columns in backtest_training_data.csv.
# If columns are missing or have < 50 non-null training rows → flag and skip.
STAT_MAP: dict[str, tuple[str, str, str]] = {
    "adj_efg_diff":            ("home_efg_pct_l5",   "away_efg_pct_l5",   "L5 eFG% differential"),
    "adj_tov_diff":            ("home_tov_pct_l5",   "away_tov_pct_l5",   "L5 TOV% differential"),
    "weighted_drtg_diff":      ("home_adj_drtg",     "away_adj_drtg",     "Opponent-quality-adj defensive rating diff"),
    "ftr_diff":                ("home_ftr_l5",       "away_ftr_l5",       "L5 Free-throw rate differential"),
    "momentum_score_diff":     ("home_momentum_score","away_momentum_score","Momentum score differential"),
    "true_orb_pct_diff":       ("home_orb_pct_l10",  "away_orb_pct_l10",  "L10 offensive rebound % differential"),
    "adj_pace_diff":           ("home_pace_l5",      "away_pace_l5",      "L5 pace differential"),
    "cover_rate_l10_diff":     ("home_cover_rate_l10","away_cover_rate_l10","L10 ATS cover rate differential"),
    "rest_days_diff":          ("home_rest_days",    "away_rest_days",    "Rest days differential"),
    "close_game_win_pct_diff": ("home_luck_score",   "away_luck_score",   "Luck-adjusted close-game proxy (home_luck_score - away_luck_score)"),
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def roi_at_110(n_wins: int, n_losses: int) -> float:
    """ROI per unit wagered at -110 juice."""
    total = n_wins + n_losses
    if total == 0:
        return float("nan")
    return round((n_wins * (1 / JUICE) - n_losses) / total * 100, 2)


def ats_hit_rate(predictions: np.ndarray, outcomes: np.ndarray) -> float:
    """
    predictions: +1 (bet home) or -1 (bet away) for each game
    outcomes:    1 (home covered) or 0 (away covered)

    Returns hit rate in [0, 1].
    """
    n = len(predictions)
    if n == 0:
        return float("nan")
    # A win = prediction matches outcome
    wins = np.sum(
        ((predictions > 0) & (outcomes == 1)) |
        ((predictions < 0) & (outcomes == 0))
    )
    return wins / n


def composite_predictions(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Composite score = X @ weights.  Positive → bet home."""
    return X @ weights


def negative_hit_rate(weights: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Objective for scipy: minimize = maximize hit rate."""
    scores = composite_predictions(X, weights)
    # exclude near-zero scores (no pick)
    preds = np.sign(scores)
    preds[preds == 0] = 1   # tie-break: bet home
    return -ats_hit_rate(preds, y)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("STAT RANKER — ATS Hit Rate Optimization")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    btd_path = DATA_DIR / "backtest_training_data.csv"
    if not btd_path.exists():
        sys.exit(f"[ERROR] {btd_path} not found. Run build_training_data.py first.")

    df = pd.read_csv(btd_path, low_memory=False)
    print(f"\nLoaded {len(df):,} total rows from backtest_training_data.csv")

    # Filter to completed games with a graded ATS outcome
    df = df[df["home_covered_ats"].notna()].copy()
    df["home_covered_ats"] = pd.to_numeric(df["home_covered_ats"], errors="coerce")
    df = df[df["home_covered_ats"].isin([0, 1])].copy()
    print(f"Graded ATS games (have Vegas spread): {len(df):,}")

    if len(df) < 20:
        sys.exit("[ERROR] Fewer than 20 graded ATS games — cannot optimize.")

    # Sort chronologically
    if "game_datetime_utc" in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
        df = df.sort_values("_sort_dt").reset_index(drop=True)
    elif "game_date" in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["game_date"], errors="coerce")
        df = df.sort_values("_sort_dt").reset_index(drop=True)
    else:
        print("[WARN] No datetime column found; assuming chronological order.")

    # Train / test split
    n_train = int(len(df) * TRAINING_FRAC)
    train = df.iloc[:n_train].copy()
    test  = df.iloc[n_train:].copy()
    print(f"Split: {len(train)} train / {len(test)} test  (60/40 walk-forward)")
    print(f"Train date range: {df.iloc[0].get('_sort_dt', '?')} to {df.iloc[n_train-1].get('_sort_dt', '?')}")
    print(f"Test  date range: {df.iloc[n_train].get('_sort_dt', '?')} to {df.iloc[-1].get('_sort_dt', '?')}")

    if len(train) < 10 or len(test) < 5:
        sys.exit("[ERROR] Too few rows for meaningful train/test evaluation.")

    y_train = train["home_covered_ats"].values
    y_test  = test["home_covered_ats"].values

    # ── Compute stat diffs ────────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("STAT AVAILABILITY CHECK")
    print("-" * 70)

    active_stats: list[str] = []
    skipped_stats: list[str] = []
    diff_cols: dict[str, str] = {}   # stat_name → computed diff column name

    for stat, (hcol, acol, desc) in STAT_MAP.items():
        h_ok = hcol in df.columns
        a_ok = acol in df.columns
        if not h_ok or not a_ok:
            missing = []
            if not h_ok: missing.append(hcol)
            if not a_ok: missing.append(acol)
            print(f"  SKIP  {stat:<30} missing cols: {missing}")
            skipped_stats.append(stat)
            continue

        diff_col = f"_diff_{stat}"
        df[diff_col]    = pd.to_numeric(df[hcol], errors="coerce") - pd.to_numeric(df[acol], errors="coerce")
        train[diff_col] = pd.to_numeric(train[hcol], errors="coerce") - pd.to_numeric(train[acol], errors="coerce")
        test[diff_col]  = pd.to_numeric(test[hcol], errors="coerce") - pd.to_numeric(test[acol], errors="coerce")

        n_train_ok = train[diff_col].notna().sum()
        n_test_ok  = test[diff_col].notna().sum()

        if n_train_ok < 30:
            print(f"  SKIP  {stat:<30} only {n_train_ok} non-null training rows (need >=30)")
            skipped_stats.append(stat)
            continue

        print(f"  OK    {stat:<30} train:{n_train_ok}/{len(train)} test:{n_test_ok}/{len(test)}  [{desc}]")
        active_stats.append(stat)
        diff_cols[stat] = diff_col

    if not active_stats:
        sys.exit("[ERROR] No usable stats found — cannot proceed.")

    print(f"\nActive stats: {len(active_stats)}  |  Skipped: {len(skipped_stats)}")

    # ── Individual stat analysis ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("INDIVIDUAL STAT PERFORMANCE (test set)")
    print("=" * 70)
    print(f"  {'STAT':<30} {'TRAIN_HR':>8} {'TEST_HR':>8} {'TEST_ROI':>9} {'N_TEST':>7} {'GAP':>6}")
    print("  " + "-" * 68)

    stat_results: list[dict] = []

    for stat in active_stats:
        dc = diff_cols[stat]

        # Training hit rate
        tr_mask = train[dc].notna()
        if tr_mask.sum() < 5:
            continue
        tr_scores = np.sign(train.loc[tr_mask, dc].values)
        tr_scores[tr_scores == 0] = 1
        tr_hr = ats_hit_rate(tr_scores, y_train[tr_mask.values]) * 100

        # Test hit rate
        te_mask = test[dc].notna()
        te_n = te_mask.sum()
        if te_n < 5:
            te_hr = float("nan")
            te_roi = float("nan")
            te_wins, te_losses = 0, 0
        else:
            te_scores = np.sign(test.loc[te_mask, dc].values)
            te_scores[te_scores == 0] = 1
            te_outcomes = y_test[te_mask.values]
            te_hr = ats_hit_rate(te_scores, te_outcomes) * 100
            te_wins   = int(np.sum(((te_scores > 0) & (te_outcomes == 1)) | ((te_scores < 0) & (te_outcomes == 0))))
            te_losses = te_n - te_wins
            te_roi = roi_at_110(te_wins, te_losses)

        gap = tr_hr - te_hr if not np.isnan(te_hr) else float("nan")
        gap_str = f"{gap:+.1f}" if not np.isnan(gap) else "  n/a"
        print(f"  {stat:<30} {tr_hr:>7.1f}% {te_hr:>7.1f}% {te_roi:>8.1f}% {te_n:>6}  {gap_str}")

        stat_results.append({
            "stat": stat,
            "description": STAT_MAP[stat][2],
            "individual_train_hit_rate": round(tr_hr, 2),
            "individual_test_hit_rate":  round(te_hr, 2) if not np.isnan(te_hr) else None,
            "individual_roi":            te_roi if not np.isnan(te_roi) else None,
            "train_test_gap":            round(gap, 2) if not np.isnan(gap) else None,
            "sample_size_train":         int(tr_mask.sum()),
            "sample_size_test":          int(te_n),
        })

    # Sort by test hit rate descending
    stat_results.sort(key=lambda r: r["individual_test_hit_rate"] or 0, reverse=True)

    print("\n  Ranked by test hit rate:")
    for i, r in enumerate(stat_results, 1):
        te_hr = f"{r['individual_test_hit_rate']:.1f}%" if r['individual_test_hit_rate'] is not None else "  n/a"
        print(f"  #{i}  {r['stat']:<30}  test HR: {te_hr}")

    # ── Ensemble optimization ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ENSEMBLE WEIGHT OPTIMIZATION (training set)")
    print("=" * 70)

    # Build X_train: fill NaN with 0 (no signal)
    X_train = np.column_stack([
        train[diff_cols[s]].fillna(0).values for s in active_stats
    ])
    X_test = np.column_stack([
        test[diff_cols[s]].fillna(0).values for s in active_stats
    ])

    # Normalize columns so scale differences don't dominate
    col_scales = np.nanstd(X_train, axis=0)
    col_scales[col_scales == 0] = 1.0
    X_train_norm = X_train / col_scales
    X_test_norm  = X_test  / col_scales

    n_stats = len(active_stats)
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    bounds = [(0.0, 1.0)] * n_stats

    print(f"Running {N_RANDOM_STARTS:,} random starts to find global optimum...")

    best_val   = float("inf")
    best_weights = np.ones(n_stats) / n_stats   # equal weight baseline

    rng = np.random.default_rng(42)
    for _ in range(N_RANDOM_STARTS):
        w0 = rng.dirichlet(np.ones(n_stats))
        result = minimize(
            negative_hit_rate,
            w0,
            args=(X_train_norm, y_train),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 500},
        )
        if result.fun < best_val:
            best_val = result.fun
            best_weights = result.x.copy()

    # Evaluate on train
    train_scores = composite_predictions(X_train_norm, best_weights)
    train_preds  = np.sign(train_scores)
    train_preds[train_preds == 0] = 1
    train_hr = ats_hit_rate(train_preds, y_train) * 100
    train_wins   = int(np.sum(((train_preds > 0) & (y_train == 1)) | ((train_preds < 0) & (y_train == 0))))
    train_losses = len(y_train) - train_wins
    train_roi = roi_at_110(train_wins, train_losses)

    # Evaluate on test
    test_scores = composite_predictions(X_test_norm, best_weights)
    test_preds  = np.sign(test_scores)
    test_preds[test_preds == 0] = 1
    test_hr = ats_hit_rate(test_preds, y_test) * 100
    test_wins   = int(np.sum(((test_preds > 0) & (y_test == 1)) | ((test_preds < 0) & (y_test == 0))))
    test_losses = len(y_test) - test_wins
    test_roi = roi_at_110(test_wins, test_losses)

    # Equal-weight baseline on test
    eq_weights = np.ones(n_stats) / n_stats
    eq_scores = composite_predictions(X_test_norm, eq_weights)
    eq_preds  = np.sign(eq_scores)
    eq_preds[eq_preds == 0] = 1
    eq_hr  = ats_hit_rate(eq_preds, y_test) * 100
    eq_wins   = int(np.sum(((eq_preds > 0) & (y_test == 1)) | ((eq_preds < 0) & (y_test == 0))))
    eq_losses = len(y_test) - eq_wins
    eq_roi = roi_at_110(eq_wins, eq_losses)

    overfit_flag = (train_hr - test_hr) > OVERFIT_THRESHOLD

    print(f"\n  Optimized ensemble results:")
    print(f"  Training  hit rate: {train_hr:.1f}%  ({train_wins}-{train_losses})  ROI: {train_roi:+.1f}%")
    print(f"  Test      hit rate: {test_hr:.1f}%  ({test_wins}-{test_losses})  ROI: {test_roi:+.1f}%")
    print(f"  Equal-weight test:  {eq_hr:.1f}%  ({eq_wins}-{eq_losses})  ROI: {eq_roi:+.1f}%")
    print(f"  Train-test gap:     {train_hr - test_hr:+.1f} pts", end="")
    if overfit_flag:
        print(f"  *** LIKELY OVERFIT (gap > {OVERFIT_THRESHOLD}pts) ***")
    else:
        print()

    print(f"\n  Optimized weights:")
    for stat, w in zip(active_stats, best_weights):
        print(f"    {stat:<30}  {w:.4f}")

    # ── Sample size warning ────────────────────────────────────────────────────
    print(f"\n  NOTE: Only {len(df)} games have ATS data (ESPN spread coverage ~3%).")
    print(f"  Test set = {len(test)} games. Results have high variance; interpret cautiously.")

    # ── Output ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OUTPUTS")
    print("=" * 70)

    # Build stat_rankings.csv
    # Add skipped stats as rows with nulls
    for sk in skipped_stats:
        stat_results.append({
            "stat": sk,
            "description": STAT_MAP[sk][2],
            "individual_train_hit_rate": None,
            "individual_test_hit_rate": None,
            "individual_roi": None,
            "train_test_gap": None,
            "sample_size_train": 0,
            "sample_size_test": 0,
        })

    # Add ensemble weights
    weights_map = dict(zip(active_stats, best_weights))
    for r in stat_results:
        r["ensemble_weight"] = round(weights_map.get(r["stat"], 0.0), 6)
        r["ensemble_train_hit_rate"] = round(train_hr, 2)
        r["ensemble_test_hit_rate"]  = round(test_hr, 2)
        r["ensemble_train_roi"]      = round(train_roi, 2)
        r["ensemble_test_roi"]       = round(test_roi, 2)
        r["equal_weight_test_hit_rate"] = round(eq_hr, 2)
        r["equal_weight_test_roi"]   = round(eq_roi, 2)
        r["overfit_flag"]            = overfit_flag
        r["n_ats_games_total"]       = len(df)

    rankings_df = pd.DataFrame(stat_results)[[
        "stat", "description",
        "individual_train_hit_rate", "individual_test_hit_rate", "individual_roi",
        "train_test_gap", "sample_size_train", "sample_size_test",
        "ensemble_weight",
        "ensemble_train_hit_rate", "ensemble_test_hit_rate",
        "ensemble_train_roi", "ensemble_test_roi",
        "equal_weight_test_hit_rate", "equal_weight_test_roi",
        "overfit_flag", "n_ats_games_total",
    ]]
    rankings_df.to_csv(OUT_RANKINGS, index=False)
    print(f"[OK] {OUT_RANKINGS}: {len(rankings_df)} rows")

    # Build optimal_weights.json
    weights_json = {
        "active_stats": active_stats,
        "skipped_stats": skipped_stats,
        "weights": {s: round(float(w), 6) for s, w in zip(active_stats, best_weights)},
        "col_scales": {s: round(float(cs), 6) for s, cs in zip(active_stats, col_scales)},
        "performance": {
            "train_hit_rate": round(train_hr, 4),
            "test_hit_rate":  round(test_hr, 4),
            "train_roi":      round(train_roi, 4),
            "test_roi":       round(test_roi, 4),
            "equal_weight_test_hit_rate": round(eq_hr, 4),
            "equal_weight_test_roi":      round(eq_roi, 4),
            "overfit_flag":   bool(overfit_flag),
            "n_train":        int(len(train)),
            "n_test":         int(len(test)),
            "n_ats_total":    int(len(df)),
        },
        "methodology": (
            f"60/40 walk-forward split, {N_RANDOM_STARTS} random scipy SLSQP starts, "
            "weights constrained to sum=1 all>=0, columns z-score normalized before optimization"
        ),
    }
    with open(OUT_WEIGHTS, "w") as fh:
        json.dump(weights_json, fh, indent=2)
    print(f"[OK] {OUT_WEIGHTS}")

    print("\nDone.")


if __name__ == "__main__":
    main()
