#!/usr/bin/env python3
"""
backtesting/engine.py
======================
Core backtest loop. Implements the two-pass pair approach, trio pass,
quad priority list, statistical validation, and merge gates.

Parallelisation: joblib.Parallel(n_jobs=2) — explicit.
Runtime guard: 90-minute timeout with checkpoint.
"""

from __future__ import annotations

import itertools
import time
import warnings
from datetime import datetime
from typing import Any, Iterator

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import binomtest

from .combos import (
    SMOKE_TEST_COMBOS,
    generate_pairs,
    generate_trios_from_pairs,
    has_collinear_signals,
    has_sum_without_diff,
    available_signals,
)
from .direction_map import combo_fires, SIGNAL_DIRECTION
from .thresholds import FULL_GRID, build_median_thresholds

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MIN_SAMPLE       = 30      # minimum fires to be valid
MIN_FIRE_RATE    = 0.05    # must fire on >= 5% of total games
PVALUE_THRESH    = 0.05    # max p-value for binomtest
MERGE_SU_INSAMPLE = 0.59   # minimum in-sample SU hit rate
MERGE_SU_HOLDOUT  = 0.54   # minimum holdout SU hit rate
SEASON_PASS_THRESH = 0.59  # season-level rate for consistency gate
SEASON_PASS_MIN    = 1     # must pass in ≥ N seasons (1 of 1 — single season flag)
PAIR_PASS1_THRESH  = 0.55  # pair must reach this to proceed to Pass 2
TRIO_SOURCE_THRESH = 0.55  # pairs must hit this to donate signals to trio pool
TIMEOUT_SECS       = 90 * 60  # 90 minutes

# ─────────────────────────────────────────────────────────────────────────────
# Single-combo evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_combo(
    combo: tuple[str, ...],
    df: pd.DataFrame,
    threshold_dict: dict[str, float],
    n_total: int,
) -> dict[str, Any] | None:
    """
    Evaluate a single combo against the full in-sample dataset.
    Returns a result dict or None if below fire-rate/sample filters.

    Outcome column: 'home_win' (1 = home/Team A wins).
    ATS outcome: 'home_cover' (1 = home/Team A covers).
    """
    # Find rows where ALL signals fire
    try:
        fires_mask = pd.Series(True, index=df.index)
        for sig in combo:
            direction = SIGNAL_DIRECTION.get(sig, "A")
            thresh = threshold_dict.get(sig, 0.0)
            val_series = df.get(sig)
            if val_series is None:
                return None
            null_mask = val_series.isna()
            if direction == "A":
                fires_mask &= (val_series > thresh) & ~null_mask
            elif direction == "B":
                fires_mask &= (val_series < thresh) & ~null_mask
            elif direction == "X":
                fires_mask &= (val_series.abs() > thresh) & ~null_mask

        n_fires = fires_mask.sum()
        fire_rate = n_fires / n_total if n_total > 0 else 0.0

        if fire_rate < MIN_FIRE_RATE or n_fires < MIN_SAMPLE:
            return None

        fired_df = df[fires_mask]

        n_hits_su  = fired_df["home_win"].sum()
        hit_rate_su = n_hits_su / n_fires

        # ATS: only count games with valid cover data (exclude NaN/no-spread games)
        if "home_cover" in fired_df.columns:
            ats_valid = fired_df["home_cover"].dropna()
            hit_rate_ats = float(ats_valid.mean()) if len(ats_valid) >= 5 else np.nan
        else:
            hit_rate_ats = np.nan

        return {
            "combo":        combo,
            "combo_str":    " + ".join(combo),
            "n":            int(n_fires),
            "fire_rate":    round(float(fire_rate), 4),
            "hit_rate_su":  round(float(hit_rate_su), 4),
            "hit_rate_ats": round(float(hit_rate_ats), 4) if not np.isnan(hit_rate_ats) else None,
            "thresholds":   {k: threshold_dict.get(k, 0.0) for k in combo},
        }
    except Exception:
        return None


def evaluate_combo_grid(
    combo: tuple[str, ...],
    df: pd.DataFrame,
    n_total: int,
) -> dict[str, Any] | None:
    """
    Pass 2: full threshold grid search. Returns the best threshold combo.
    """
    # Build per-signal grid (only for signals in FULL_GRID)
    signal_grids: list[list[float]] = []
    for sig in combo:
        if sig in FULL_GRID:
            signal_grids.append(FULL_GRID[sig])
        else:
            signal_grids.append([0.0])  # boolean / fixed threshold

    best: dict[str, Any] | None = None

    for threshold_combo in itertools.product(*signal_grids):
        thresh_dict = dict(zip(combo, threshold_combo))
        result = evaluate_combo(combo, df, thresh_dict, n_total)
        if result is None:
            continue
        if best is None or result["hit_rate_su"] > best["hit_rate_su"]:
            best = result

    return best


# ─────────────────────────────────────────────────────────────────────────────
# Statistical validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_result(
    result: dict[str, Any],
    df_insample: pd.DataFrame,
    df_holdout: pd.DataFrame,
) -> dict[str, Any]:
    """
    Add p_value, holdout metrics, per-season rates, and consistency flag.
    """
    # Binomial test vs 50% baseline
    pval = binomtest(
        int(result["n"] * result["hit_rate_su"]),
        result["n"],
        p=0.50,
        alternative="greater",
    ).pvalue
    result["p_value"] = round(float(pval), 6)

    # Holdout validation
    ho_result = evaluate_combo(
        result["combo"],
        df_holdout,
        result["thresholds"],
        n_total=len(df_holdout),
    )
    if ho_result:
        result["holdout_su"]  = ho_result["hit_rate_su"]
        result["holdout_ats"] = ho_result.get("hit_rate_ats")
        result["holdout_n"]   = ho_result["n"]
    else:
        result["holdout_su"]  = None
        result["holdout_ats"] = None
        result["holdout_n"]   = 0

    # Per-season hit rates (single season → one entry)
    if "game_dt" in df_insample.columns:
        df_insample = df_insample.copy()
        df_insample["season_yr"] = pd.to_datetime(
            df_insample["game_dt"], utc=True, errors="coerce"
        ).dt.year.where(
            pd.to_datetime(df_insample["game_dt"], utc=True, errors="coerce").dt.month >= 11,
            pd.to_datetime(df_insample["game_dt"], utc=True, errors="coerce").dt.year - 1,
        )
        season_rates: dict[str, float] = {}
        for yr, season_df in df_insample.groupby("season_yr"):
            s_result = evaluate_combo(
                result["combo"], season_df, result["thresholds"], len(season_df)
            )
            if s_result:
                season_rates[str(yr)] = s_result["hit_rate_su"]
        result["season_hit_rates"] = season_rates
        passing = sum(1 for r in season_rates.values() if r >= SEASON_PASS_THRESH)
        result["seasons_passing"] = passing
        result["consistent"] = passing >= SEASON_PASS_MIN
    else:
        result["season_hit_rates"] = {}
        result["seasons_passing"]  = 0
        result["consistent"]       = False

    # lx_days flag
    result["lx_days"] = any(
        "rest_" in s or "_rest" in s for s in result["combo"]
    )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Merge gate check
# ─────────────────────────────────────────────────────────────────────────────

def passes_merge_gate(result: dict[str, Any]) -> bool:
    return (
        result.get("hit_rate_su", 0) >= MERGE_SU_INSAMPLE
        and result.get("holdout_su") is not None
        and result.get("holdout_su", 0) >= MERGE_SU_HOLDOUT
        and result.get("n", 0) >= MIN_SAMPLE
        and result.get("fire_rate", 0) >= MIN_FIRE_RATE
        and result.get("p_value", 1) < PVALUE_THRESH
        and result.get("consistent", False)
    )


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def run_smoke_test(
    df: pd.DataFrame,
    median_thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Run the 20 priority combos (Steps 3 in the plan).
    Returns results. Halts (via assertion) if <3 valid.
    """
    n_total = len(df)
    results = []
    print("\n[smoke] Running 20 priority combos...")

    for combo in SMOKE_TEST_COMBOS:
        # Filter to signals that exist in df
        valid_combo = tuple(s for s in combo if s in df.columns)
        if len(valid_combo) < 2:
            print(f"  SKIP (cols missing): {combo}")
            continue

        result = evaluate_combo(valid_combo, df, median_thresholds, n_total)
        if result is None:
            status = f"FAIL (n<{MIN_SAMPLE} or fire_rate<{MIN_FIRE_RATE:.0%})"
        else:
            status = f"n={result['n']:3d}  SU={result['hit_rate_su']:.3f}  ATS={result.get('hit_rate_ats') or 'N/A'}"
            results.append(result)
        print(f"  {'|'.join(combo):60s}  {status}")

    n_valid = len(results)
    print(f"\n[smoke] {n_valid}/20 combos returned valid results (need >= 3)")

    if n_valid < 3:
        print("[HALT] Fewer than 3 valid smoke-test combos. Debug pipeline before continuing.")
        raise RuntimeError(f"Smoke test failed: only {n_valid}/20 valid — check metric computation.")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pair passes
# ─────────────────────────────────────────────────────────────────────────────

def run_pair_pass1(
    df: pd.DataFrame,
    median_thresholds: dict[str, float],
    start_time: float,
) -> tuple[list[dict], int, int]:
    """
    Pass 1: median-split discovery over all valid pairs.
    Returns (results, n_skipped_collinear, n_skipped_fire_rate).
    """
    signal_pool = available_signals(list(df.columns))
    pairs = list(generate_pairs(signal_pool, list(df.columns)))
    n_total = len(df)
    n_collinear = len(list(itertools.combinations(signal_pool, 2))) - len(pairs)

    print(f"\n[pair-p1] Testing {len(pairs):,} pairs with median thresholds...")

    def _eval(combo):
        if time.time() - start_time > TIMEOUT_SECS:
            return None
        return evaluate_combo(combo, df, median_thresholds, n_total)

    raw = Parallel(n_jobs=2, prefer="threads")(
        delayed(_eval)(combo) for combo in pairs
    )

    results = [r for r in raw if r is not None]
    n_low_fire = len(pairs) - len(results)

    results.sort(key=lambda x: x["hit_rate_su"], reverse=True)
    print(f"[pair-p1] Valid: {len(results)} | Collinear skipped: {n_collinear} | Below filter: {n_low_fire}")
    if results:
        top5 = results[:5]
        for r in top5:
            print(f"  {r['combo_str'][:60]:60s}  SU={r['hit_rate_su']:.3f}  n={r['n']}")

    return results, n_collinear, n_low_fire


def run_pair_pass2(
    df: pd.DataFrame,
    pass1_results: list[dict],
    start_time: float,
) -> list[dict]:
    """
    Pass 2: full grid refinement on pairs that hit >= 55% SU in Pass 1.
    """
    candidates = [r for r in pass1_results if r["hit_rate_su"] >= PAIR_PASS1_THRESH]
    n_total = len(df)

    print(f"\n[pair-p2] Refining {len(candidates)} pairs with full grid...")

    def _eval_grid(result):
        if time.time() - start_time > TIMEOUT_SECS:
            return result
        refined = evaluate_combo_grid(result["combo"], df, n_total)
        return refined if refined else result

    refined = Parallel(n_jobs=2, prefer="threads")(
        delayed(_eval_grid)(r) for r in candidates
    )
    refined = [r for r in refined if r is not None]

    # Merge refined back into pass1 results (replace matching combos)
    refined_map = {r["combo"]: r for r in refined}
    final = []
    for r in pass1_results:
        if r["combo"] in refined_map:
            final.append(refined_map[r["combo"]])
        else:
            final.append(r)

    final.sort(key=lambda x: x["hit_rate_su"], reverse=True)
    print(f"[pair-p2] Refined: {len(refined)} pairs")
    return final


# ─────────────────────────────────────────────────────────────────────────────
# Trio pass
# ─────────────────────────────────────────────────────────────────────────────

def run_trio_backtest(
    df: pd.DataFrame,
    pair_results: list[dict],
    median_thresholds: dict[str, float],
    start_time: float,
) -> tuple[list[dict], int]:
    """
    Build and test trios from signals in 55%+ pairs.
    Returns (results, n_qualifying_pairs).
    """
    qualifying = [r for r in pair_results if r["hit_rate_su"] >= TRIO_SOURCE_THRESH]
    n_total = len(df)

    trios = list(generate_trios_from_pairs(qualifying, list(df.columns)))
    print(f"\n[trio] {len(qualifying)} qualifying pairs → {len(trios):,} trios to test...")

    def _eval(combo):
        if time.time() - start_time > TIMEOUT_SECS:
            return None
        return evaluate_combo(combo, df, median_thresholds, n_total)

    raw = Parallel(n_jobs=2, prefer="threads")(
        delayed(_eval)(combo) for combo in trios
    )
    results = [r for r in raw if r is not None]
    results.sort(key=lambda x: x["hit_rate_su"], reverse=True)

    print(f"[trio] Valid results: {len(results)}")
    return results, len(qualifying)


# ─────────────────────────────────────────────────────────────────────────────
# Quad priority list
# ─────────────────────────────────────────────────────────────────────────────

def run_quad_priority_list(
    df: pd.DataFrame,
    median_thresholds: dict[str, float],
    pair_results: list[dict],
) -> list[dict]:
    """
    Run only the 20 hardcoded priority quads.
    Use refined thresholds from Pass 2 where available.
    """
    from .combos import SMOKE_TEST_COMBOS
    quads = [c for c in SMOKE_TEST_COMBOS if len(c) == 4]
    n_total = len(df)

    # Build best-known thresholds from pair refinement
    refined_thresh: dict[str, float] = dict(median_thresholds)
    for r in pair_results:
        for sig, val in r["thresholds"].items():
            refined_thresh[sig] = val

    results = []
    print(f"\n[quad] Testing {len(quads)} priority quads...")
    for combo in quads:
        valid_combo = tuple(s for s in combo if s in df.columns)
        if len(valid_combo) < 4:
            print(f"  SKIP (missing cols): {combo}")
            continue
        result = evaluate_combo(valid_combo, df, refined_thresh, n_total)
        if result:
            results.append(result)
            print(f"  {result['combo_str'][:60]:60s}  SU={result['hit_rate_su']:.3f}  n={result['n']}")
        else:
            print(f"  FAIL (filter): {'|'.join(valid_combo)}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(df: pd.DataFrame) -> dict[str, Any]:
    """
    Orchestrate all backtest phases.
    df must be matchup_metrics.csv (game-level, one row per game).
    Returns a summary dict with all results.
    """
    start_time = time.time()

    # Ensure outcome columns exist
    if "home_win" not in df.columns:
        raise ValueError("matchup_metrics must have 'home_win' column")
    if "home_cover" not in df.columns:
        df = df.copy()
        if "home_cover" in df.columns:
            pass
        elif "cover" in df.columns:
            df["home_cover"] = df["cover"]
        else:
            df["home_cover"] = np.nan

    # Train/holdout split — last 20% of games by date
    df = df.sort_values("game_dt").copy()
    n_holdout = max(1, int(len(df) * 0.20))
    df_insample = df.iloc[:-n_holdout].copy()
    df_holdout  = df.iloc[-n_holdout:].copy()

    n_total = len(df_insample)
    print(f"\n[split] In-sample: {len(df_insample):,} | Holdout: {len(df_holdout):,}")

    # Median thresholds on in-sample only (no leakage)
    median_thresholds = build_median_thresholds(df_insample)

    # ── Phase 1: Smoke test
    smoke_results = run_smoke_test(df_insample, median_thresholds)

    # ── Phase 2: Pair Pass 1
    timeout_hit = False
    pair_results, n_collinear, n_low_fire = run_pair_pass1(
        df_insample, median_thresholds, start_time
    )

    if time.time() - start_time > TIMEOUT_SECS:
        print("\n[TIMEOUT] 90-minute checkpoint reached after pair Pass 1.")
        timeout_hit = True
        return _checkpoint(pair_results, [], [], smoke_results, n_collinear, n_low_fire, timeout_hit)

    # ── Phase 3: Pair Pass 2 (refinement)
    pair_results = run_pair_pass2(df_insample, pair_results, start_time)

    if time.time() - start_time > TIMEOUT_SECS:
        print("\n[TIMEOUT] 90-minute checkpoint reached after pair Pass 2.")
        timeout_hit = True
        return _checkpoint(pair_results, [], [], smoke_results, n_collinear, n_low_fire, timeout_hit)

    # ── Phase 4: Trio backtest
    trio_results, n_qualifying_pairs = run_trio_backtest(
        df_insample, pair_results, median_thresholds, start_time
    )

    if time.time() - start_time > TIMEOUT_SECS:
        print("\n[TIMEOUT] 90-minute checkpoint reached after trios.")
        timeout_hit = True
        return _checkpoint(
            pair_results, trio_results, [], smoke_results,
            n_collinear, n_low_fire, timeout_hit
        )

    # ── Phase 5: Quad priority list
    quad_results = run_quad_priority_list(df_insample, median_thresholds, pair_results)

    # ── Phase 6: Statistical validation on all candidates
    all_candidates = pair_results + trio_results + quad_results
    print(f"\n[validate] Running stat validation on {len(all_candidates)} candidates...")

    validated = []
    for r in all_candidates:
        r = validate_result(r, df_insample, df_holdout)
        validated.append(r)

    # ── Phase 7: Apply merge gates
    qualified = [r for r in validated if passes_merge_gate(r)]
    n_insig = sum(1 for r in validated if r.get("p_value", 1) >= PVALUE_THRESH)
    n_inconsistent = sum(1 for r in validated if not r.get("consistent", False))

    # Tag combo type
    for r in validated:
        r["combo_type"] = {2: "pair", 3: "trio", 4: "quad"}.get(len(r["combo"]), "other")
        r["merged"] = r in qualified

    elapsed = time.time() - start_time
    print(f"\n[done] Elapsed: {elapsed/60:.1f} min | Qualified: {len(qualified)}")

    return {
        "validated":        validated,
        "qualified":        qualified,
        "smoke_results":    smoke_results,
        "n_pairs":          len(pair_results),
        "n_trios":          len(trio_results),
        "n_quads":          len(quad_results),
        "n_qualifying_pairs": n_qualifying_pairs,
        "n_collinear":      n_collinear,
        "n_low_fire":       n_low_fire,
        "n_insig":          n_insig,
        "n_inconsistent":   n_inconsistent,
        "timeout_hit":      timeout_hit,
        "elapsed_sec":      elapsed,
        "df_insample":      df_insample,
        "df_holdout":       df_holdout,
    }


def _checkpoint(
    pair_results, trio_results, quad_results,
    smoke_results, n_collinear, n_low_fire, timeout_hit,
) -> dict[str, Any]:
    """Return a partial results dict for timeout checkpoint."""
    all_r = pair_results + trio_results + quad_results
    for r in all_r:
        r["combo_type"] = {2: "pair", 3: "trio", 4: "quad"}.get(len(r["combo"]), "other")
        r["merged"] = False
        r.setdefault("p_value", None)
        r.setdefault("holdout_su", None)
        r.setdefault("consistent", False)
    return {
        "validated":        all_r,
        "qualified":        [],
        "smoke_results":    smoke_results,
        "n_pairs":          len(pair_results),
        "n_trios":          len(trio_results),
        "n_quads":          len(quad_results),
        "n_qualifying_pairs": 0,
        "n_collinear":      n_collinear,
        "n_low_fire":       n_low_fire,
        "n_insig":          0,
        "n_inconsistent":   0,
        "timeout_hit":      timeout_hit,
        "elapsed_sec":      TIMEOUT_SECS,
        "df_insample":      pd.DataFrame(),
        "df_holdout":       pd.DataFrame(),
    }
