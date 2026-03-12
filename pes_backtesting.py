"""PES A/B/C backtesting framework for CAGE.

Mode A (Validation):
    Tests whether PES agreement/disagreement with model edge correlates with
    outcome quality.
Mode B (Enhancement):
    Tests PES as an additive tournament-only layer on top of model edge.
Mode C (Flip):
    Tests whether strong PES disagreement can justify flipping away from
    the base model side in tournament games.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from pes_metric import PESConfig, compute_pes_matchup, get_pes_tier, plot_pes_quadrant


@dataclass(frozen=True)
class BacktestConfig:
    """Column mapping for PES backtesting workflows."""

    model_edge_col: str = "model_edge"
    covered_col: str = "covered"
    ml_won_col: str = "ml_won"
    total_covered_col: str = "total_covered"
    final_margin_col: str = "final_margin"
    season_col: str = "season"
    is_tournament_col: str = "is_tournament"
    tourney_round_col: str = "tourney_round"
    seed_team_col: str = "seed_team"
    seed_opponent_col: str = "seed_opponent"
    is_underdog_col: str = "is_underdog"
    pes_team_col: str = "pes_team"
    pes_opponent_col: str = "pes_opponent"
    pes_diff_col: str = "pes_diff"
    pgs_team_col: str = "pgs_team"
    ces_team_col: str = "ces_team"
    pgs_opponent_col: str = "pgs_opponent"
    ces_opponent_col: str = "ces_opponent"
    pes_quadrant_col: str = "pes_quadrant"


def _safe_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return numeric series (NaN fallback if missing)."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _safe_bool(df: pd.DataFrame, col: str) -> pd.Series:
    """Return robust boolean series for any dtype."""
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype="bool")
    raw = df[col]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(False)
    if pd.api.types.is_numeric_dtype(raw):
        return pd.to_numeric(raw, errors="coerce").fillna(0.0) != 0.0
    txt = raw.astype(str).str.strip().str.lower()
    return txt.isin({"1", "true", "t", "yes", "y"})


def _valid_binary_mask(series: pd.Series) -> pd.Series:
    """Mask rows with valid binary outcomes."""
    v = pd.to_numeric(series, errors="coerce")
    return v.isin([0, 1])


def compute_hit_rate(series: pd.Series) -> tuple[float, int]:
    """Compute hit-rate from a 0/1-like outcome series."""
    valid = _valid_binary_mask(series)
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0
    hits = int(pd.to_numeric(series[valid], errors="coerce").fillna(0).sum())
    return float(hits / n), n


def proportion_z_test(p1: float, n1: int, p2: float, n2: int) -> tuple[float, float]:
    """One-tailed two-proportion z-test, returning (z_stat, p_value)."""
    if n1 <= 0 or n2 <= 0 or np.isnan(p1) or np.isnan(p2):
        return np.nan, np.nan
    pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    denom = np.sqrt(max(1e-12, pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2)))
    if denom == 0:
        return np.nan, np.nan
    z_stat = (p1 - p2) / denom
    p_value = 1.0 - stats.norm.cdf(z_stat)
    return float(z_stat), float(p_value)


def wilson_confidence_interval(hits: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson interval for Bernoulli hit-rate."""
    if n <= 0:
        return np.nan, np.nan
    z = float(stats.norm.ppf((1.0 + confidence) / 2.0))
    phat = hits / n
    denom = 1.0 + (z**2 / n)
    center = (phat + (z**2 / (2 * n))) / denom
    half = (z * np.sqrt((phat * (1 - phat) + (z**2 / (4 * n))) / n)) / denom
    return float(center - half), float(center + half)


def compute_roi(series: pd.Series, odds: float = -110) -> float:
    """Compute flat-stake ROI from 0/1 outcomes at supplied American odds."""
    valid = _valid_binary_mask(series)
    n = int(valid.sum())
    if n == 0:
        return np.nan
    results = pd.to_numeric(series[valid], errors="coerce").fillna(0).astype(int)
    wins = int(results.sum())
    losses = n - wins
    if odds < 0:
        profit_per_win = 100.0 / abs(odds)
    else:
        profit_per_win = odds / 100.0
    pnl = wins * profit_per_win - losses * 1.0
    return float(pnl / n)


def _verdict(lift: float, p_value: float, n: int, min_sample: int) -> str:
    """Return verdict string using the situational module emoji convention."""
    if n < min_sample:
        return "⛔ INSUFFICIENT SAMPLE"
    if np.isnan(lift):
        return "⛔ INSUFFICIENT SAMPLE"
    if lift <= 0:
        return "❌ NEGATIVE"
    if lift >= 0.03 and pd.notna(p_value) and p_value <= 0.05:
        return "✅ VALID"
    if lift < 0.03:
        return "⚠️ REDUNDANT"
    return "🔍 PROMISING"


def _evaluate_population(
    *,
    frame: pd.DataFrame,
    population: pd.Series,
    result_col: str,
    base_rate: float,
    base_n: int,
    min_sample: int,
) -> Dict[str, Any]:
    """Evaluate one population against a base population."""
    subset = frame.loc[population].copy()
    rate, n = compute_hit_rate(subset[result_col]) if result_col in subset.columns else (np.nan, 0)
    hits = int(pd.to_numeric(subset[result_col], errors="coerce").fillna(0).sum()) if result_col in subset.columns else 0
    ci_low, ci_high = wilson_confidence_interval(hits, n)
    z_stat, p_value = proportion_z_test(rate, n, base_rate, base_n)
    lift = rate - base_rate if pd.notna(rate) and pd.notna(base_rate) else np.nan
    verdict = _verdict(lift, p_value, n, min_sample)
    return {
        "n": n,
        "hit_rate": rate,
        "lift_over_base": lift,
        "z_stat": z_stat,
        "p_value": p_value,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "roi": compute_roi(subset[result_col]) if result_col in subset.columns else np.nan,
        "verdict": verdict,
    }


def _result_cols_or_default(df: pd.DataFrame, cfg: BacktestConfig, result_cols: Optional[list[str]]) -> List[str]:
    """Return the active result columns present in dataframe."""
    default_cols = [cfg.covered_col, cfg.ml_won_col, cfg.total_covered_col]
    chosen = default_cols if result_cols is None else list(result_cols)
    return [c for c in chosen if c in df.columns]


def _write_csv_with_metadata(path: str | Path, frame: pd.DataFrame, metadata: Dict[str, Any]) -> None:
    """Write CSV preceded by metadata comment rows."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        for k, v in metadata.items():
            handle.write(f"# {k}: {v}\n")
        frame.to_csv(handle, index=False)


def _format_pct(x: Any) -> str:
    """Format decimal as percentage text."""
    x_num = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    if pd.isna(x_num):
        return "nan"
    return f"{100.0 * float(x_num):.2f}%"


def _ascii_clean(val: Any) -> str:
    """Strip non-ascii characters for Windows terminal safety."""
    return str(val).encode("ascii", "ignore").decode("ascii")


def _quadrant_from_scores(pgs: pd.Series, ces: pd.Series, threshold: float = 0.60) -> pd.Series:
    """Assign quadrant labels from pgs/ces series."""
    return pd.Series(
        np.select(
            [
                (pgs >= threshold) & (ces >= threshold),
                (ces >= threshold) & (pgs < threshold),
                (pgs >= threshold) & (ces < threshold),
            ],
            ["elite", "efficient_only", "possession_only"],
            default="fodder",
        ),
        index=pgs.index,
        dtype="object",
    )


def _ascii_threshold_plot(sweep_df: pd.DataFrame, label: str) -> None:
    """Render an in-terminal ASCII chart for threshold sweep results."""
    if sweep_df.empty:
        print(f"{label}: no sweep rows available.")
        return
    rates = pd.to_numeric(sweep_df["hit_rate"], errors="coerce").dropna()
    if rates.empty:
        print(f"{label}: no valid hit-rate rows available.")
        return
    print(f"\n{label}")
    max_rate = float(rates.max())
    min_rate = float(rates.min())
    span = max(1e-9, max_rate - min_rate)
    for _, row in sweep_df.iterrows():
        cutoff = float(row["cutoff"])
        rate = pd.to_numeric(pd.Series([row.get("hit_rate")]), errors="coerce").iloc[0]
        n = int(pd.to_numeric(pd.Series([row.get("n")]), errors="coerce").fillna(0).iloc[0])
        if pd.isna(rate):
            print(f"cut={cutoff:>4.1f} n={n:>4d} hit=   nan")
            continue
        rate = float(rate)
        bar_len = int(round(((rate - min_rate) / span) * 40))
        bar = "#" * max(1, bar_len)
        sig = "*" if bool(row.get("significant", False)) else ""
        print(f"cut={cutoff:>4.1f} n={n:>4d} hit={rate:>6.3f} {bar}{sig}")


def run_validation_backtest(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    edge_threshold: float = 4.0,
    pes_diff_threshold: float = 5.0,
    min_sample: int = 150,
    result_cols: list[str] | None = None,
) -> dict:
    """Mode A: PES validation against base model edge populations."""
    work = df.copy()
    markets = _result_cols_or_default(work, cfg, result_cols)
    edge = _safe_series(work, cfg.model_edge_col)
    pes_diff = _safe_series(work, cfg.pes_diff_col)
    tournament = _safe_bool(work, cfg.is_tournament_col)

    populations = {
        "base": edge >= float(edge_threshold),
        "pes_agrees": (edge >= float(edge_threshold)) & (pes_diff >= float(pes_diff_threshold)),
        "pes_disagrees": (edge >= float(edge_threshold)) & (pes_diff <= -float(pes_diff_threshold)),
        "pes_neutral": (edge >= float(edge_threshold)) & (pes_diff.abs() < float(pes_diff_threshold)),
    }

    def evaluate_scope(scope_name: str, scope_mask: pd.Series) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for result_col in markets:
            base_mask = scope_mask & populations["base"] & _valid_binary_mask(work[result_col])
            base_rate, base_n = compute_hit_rate(work.loc[base_mask, result_col])
            for pop_name, pop_mask in populations.items():
                eval_mask = scope_mask & pop_mask & _valid_binary_mask(work[result_col])
                stats_row = _evaluate_population(
                    frame=work,
                    population=eval_mask,
                    result_col=result_col,
                    base_rate=base_rate,
                    base_n=base_n,
                    min_sample=min_sample,
                )
                rows.append(
                    {
                        "scope": scope_name,
                        "market": result_col,
                        "population": pop_name,
                        "base_n": base_n,
                        "base_hit_rate": base_rate,
                        **stats_row,
                    }
                )
        return pd.DataFrame(rows)

    full_df = evaluate_scope("full", pd.Series(True, index=work.index))
    tourney_df = evaluate_scope("tournament", tournament)

    print("\n=== PES Validation Backtest (Mode A) ===")
    if not full_df.empty:
        for market in markets:
            sub = full_df[(full_df["market"] == market) & (full_df["population"] != "base")]
            if sub.empty:
                continue
            best = sub.sort_values(["lift_over_base", "p_value"], ascending=[False, True]).iloc[0]
            print(
                f"{market} best(full): {best['population']} "
                f"hit={_format_pct(best['hit_rate'])} lift={_format_pct(best['lift_over_base'])} "
                f"n={int(best['n'])} p={best['p_value']:.4f}"
            )
    if not tourney_df.empty:
        for market in markets:
            sub = tourney_df[(tourney_df["market"] == market) & (tourney_df["population"] != "base")]
            if sub.empty:
                continue
            best = sub.sort_values(["lift_over_base", "p_value"], ascending=[False, True]).iloc[0]
            print(
                f"{market} best(tournament): {best['population']} "
                f"hit={_format_pct(best['hit_rate'])} lift={_format_pct(best['lift_over_base'])} "
                f"n={int(best['n'])} p={best['p_value']:.4f}"
            )

    return {"full": full_df, "tournament": tourney_df}


def run_enhancement_backtest(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    edge_threshold: float = 4.0,
    min_sample: int = 150,
    result_cols: list[str] | None = None,
) -> dict:
    """Mode B: PES enhancement testing on tournament-only populations."""
    work = df.copy()
    markets = _result_cols_or_default(work, cfg, result_cols)
    edge = _safe_series(work, cfg.model_edge_col)
    is_tour = _safe_bool(work, cfg.is_tournament_col)
    pes_diff = _safe_series(work, cfg.pes_diff_col)
    pgs_team = _safe_series(work, cfg.pgs_team_col)
    ces_team = _safe_series(work, cfg.ces_team_col)
    pgs_opp = _safe_series(work, cfg.pgs_opponent_col)
    ces_opp = _safe_series(work, cfg.ces_opponent_col)
    team_q = work.get(cfg.pes_quadrant_col, _quadrant_from_scores(pgs_team, ces_team)).astype(str).str.lower()
    opp_q = _quadrant_from_scores(pgs_opp, ces_opp)

    base_tourney = (edge >= float(edge_threshold)) & is_tour
    populations = {
        "base_tournament": base_tourney,
        "pes_elite_team": base_tourney & (team_q == "elite"),
        "pes_diff_5_plus": base_tourney & (pes_diff >= 5.0),
        "pes_diff_8_plus": base_tourney & (pes_diff >= 8.0),
        "pes_diff_12_plus": base_tourney & (pes_diff >= 12.0),
        "pes_diff_15_plus": base_tourney & (pes_diff >= 15.0),
        "quadrant_mismatch": base_tourney & (team_q == "elite") & opp_q.isin(["fodder", "possession_only"]),
        "possession_dominance": base_tourney & ((pgs_team - pgs_opp) >= 0.15),
        "efficiency_dominance": base_tourney & ((ces_team - ces_opp) >= 0.15),
        "both_dominate": base_tourney & ((pgs_team - pgs_opp) >= 0.10) & ((ces_team - ces_opp) >= 0.10),
    }

    rows: List[Dict[str, Any]] = []
    for market in markets:
        base_mask = populations["base_tournament"] & _valid_binary_mask(work[market])
        base_rate, base_n = compute_hit_rate(work.loc[base_mask, market])
        for pop_name, pop_mask in populations.items():
            eval_mask = pop_mask & _valid_binary_mask(work[market])
            stats_row = _evaluate_population(
                frame=work,
                population=eval_mask,
                result_col=market,
                base_rate=base_rate,
                base_n=base_n,
                min_sample=min_sample,
            )
            stats_row["lift_over_base_tournament"] = stats_row["lift_over_base"]
            rows.append(
                {
                    "market": market,
                    "population": pop_name,
                    "base_tournament_n": base_n,
                    "base_tournament_hit_rate": base_rate,
                    **stats_row,
                }
            )

    results = pd.DataFrame(rows)

    # Threshold sweeper equivalent on PES differential.
    sweep_rows: List[Dict[str, Any]] = []
    sweep_market = markets[0] if markets else cfg.covered_col
    if sweep_market in work.columns:
        base_mask = base_tourney & _valid_binary_mask(work[sweep_market])
        base_rate, base_n = compute_hit_rate(work.loc[base_mask, sweep_market])
        for cutoff in np.arange(2.0, 20.0 + 0.001, 0.5):
            cut_mask = base_tourney & (pes_diff >= cutoff) & _valid_binary_mask(work[sweep_market])
            rate, n = compute_hit_rate(work.loc[cut_mask, sweep_market])
            lift = rate - base_rate if pd.notna(rate) and pd.notna(base_rate) else np.nan
            _, p_value = proportion_z_test(rate, n, base_rate, base_n)
            sweep_rows.append(
                {
                    "cutoff": float(round(cutoff, 1)),
                    "n": n,
                    "hit_rate": rate,
                    "lift": lift,
                    "p_value": p_value,
                    "significant": bool(pd.notna(p_value) and p_value <= 0.05 and n >= min_sample),
                }
            )
    sweep_df = pd.DataFrame(sweep_rows)
    if not sweep_df.empty:
        _ascii_threshold_plot(sweep_df, label=f"PES diff threshold sweep ({sweep_market})")

    print("\n=== PES Enhancement Backtest (Mode B) ===")
    if not results.empty:
        top = results.sort_values(["lift_over_base_tournament", "p_value"], ascending=[False, True]).head(8)
        cols = ["market", "population", "n", "hit_rate", "lift_over_base_tournament", "p_value", "verdict"]
        view = top[cols].copy()
        if "verdict" in view.columns:
            view["verdict"] = view["verdict"].map(_ascii_clean)
        print(view.to_string(index=False))

    return {"results": results, "threshold_sweep": sweep_df}


def run_flip_backtest(
    df: pd.DataFrame,
    cfg: BacktestConfig,
    edge_threshold: float = 4.0,
    pes_flip_threshold: float = 5.0,
    min_sample: int = 100,
    result_cols: list[str] | None = None,
) -> dict:
    """Mode C: Test flipping model side when PES strongly disagrees."""
    work = df.copy()
    markets = _result_cols_or_default(work, cfg, result_cols or [cfg.covered_col, cfg.ml_won_col])
    edge = _safe_series(work, cfg.model_edge_col)
    pes_diff = _safe_series(work, cfg.pes_diff_col)
    is_tour = _safe_bool(work, cfg.is_tournament_col)
    seed_team = _safe_series(work, cfg.seed_team_col)

    flip_mask = (edge >= float(edge_threshold)) & (pes_diff <= -float(pes_flip_threshold)) & is_tour
    rows: List[Dict[str, Any]] = []
    for market in markets:
        valid = flip_mask & _valid_binary_mask(work[market])
        model_series = pd.to_numeric(work.loc[valid, market], errors="coerce")
        flip_series = 1 - model_series

        model_rate, model_n = compute_hit_rate(model_series)
        flip_rate, flip_n = compute_hit_rate(flip_series)
        z_stat, p_value = proportion_z_test(flip_rate, flip_n, model_rate, model_n)
        lift = flip_rate - model_rate if pd.notna(flip_rate) and pd.notna(model_rate) else np.nan
        verdict = _verdict(lift, p_value, flip_n, min_sample)
        if flip_n < 75:
            verdict = f"{verdict} | VERY LOW SAMPLE"

        rows.append(
            {
                "market": market,
                "flip_n": flip_n,
                "flip_hit_rate": flip_rate,
                "model_hit_rate_same_games": model_rate,
                "lift_of_flip_over_model": lift,
                "z_stat": z_stat,
                "p_value": p_value,
                "roi": compute_roi(flip_series),
                "verdict": verdict,
            }
        )
    flip_results = pd.DataFrame(rows)

    seed_band_rows: List[Dict[str, Any]] = []
    seed_bands = {
        "seed_1_4": (seed_team >= 1) & (seed_team <= 4),
        "seed_5_8": (seed_team >= 5) & (seed_team <= 8),
        "seed_9_13": (seed_team >= 9) & (seed_team <= 13),
    }
    for band_name, band_mask in seed_bands.items():
        band_valid = flip_mask & band_mask
        ats_flip = 1 - pd.to_numeric(work.loc[band_valid & _valid_binary_mask(work[cfg.covered_col]), cfg.covered_col], errors="coerce")
        ml_flip = 1 - pd.to_numeric(work.loc[band_valid & _valid_binary_mask(work[cfg.ml_won_col]), cfg.ml_won_col], errors="coerce")
        ats_rate, ats_n = compute_hit_rate(ats_flip)
        ml_rate, ml_n = compute_hit_rate(ml_flip)
        seed_band_rows.append({"seed_band": band_name, "ats_n": ats_n, "ats_flip_hit_rate": ats_rate, "ml_n": ml_n, "ml_flip_hit_rate": ml_rate})
    seed_band_df = pd.DataFrame(seed_band_rows)

    mag_rows: List[Dict[str, Any]] = []
    abs_disagree = pes_diff.abs()
    mag_bands = {
        "5_8": (abs_disagree >= 5.0) & (abs_disagree < 8.0),
        "8_12": (abs_disagree >= 8.0) & (abs_disagree < 12.0),
        "12_plus": abs_disagree >= 12.0,
    }
    for band_name, band_mask in mag_bands.items():
        band_valid = flip_mask & band_mask
        ats_flip = 1 - pd.to_numeric(work.loc[band_valid & _valid_binary_mask(work[cfg.covered_col]), cfg.covered_col], errors="coerce")
        ml_flip = 1 - pd.to_numeric(work.loc[band_valid & _valid_binary_mask(work[cfg.ml_won_col]), cfg.ml_won_col], errors="coerce")
        ats_rate, ats_n = compute_hit_rate(ats_flip)
        ml_rate, ml_n = compute_hit_rate(ml_flip)
        mag_rows.append({"magnitude_band": band_name, "ats_n": ats_n, "ats_flip_hit_rate": ats_rate, "ml_n": ml_n, "ml_flip_hit_rate": ml_rate})
    magnitude_df = pd.DataFrame(mag_rows)

    print("\n=== PES Flip Backtest (Mode C) ===")
    if not flip_results.empty:
        view = flip_results.copy()
        if "verdict" in view.columns:
            view["verdict"] = view["verdict"].map(_ascii_clean)
        print(view.to_string(index=False))
    print("\nInterpretation guidance:")
    print("- If flip_hit_rate > 52% with p < 0.10: PES likely carries independent signal.")
    print("- If flip_hit_rate < 48%: base model edge dominates in these conflicts.")
    print("- If flip_hit_rate is 48-52% with large n: monitor, evidence inconclusive.")

    return {"results": flip_results, "seed_bands": seed_band_df, "magnitude_bands": magnitude_df}


def run_pes_full_backtest(
    df: pd.DataFrame,
    pes_cfg: PESConfig,
    backtest_cfg: BacktestConfig,
    edge_threshold: float = 4.0,
    pes_diff_threshold: float = 5.0,
    pes_flip_threshold: float = 5.0,
    min_sample: int = 150,
    result_cols: list[str] | None = None,
    export_csv: bool = True,
) -> dict:
    """Run full PES A/B/C backtest suite and return aggregate verdict."""
    base = df.copy()
    regular = compute_pes_matchup(base, cfg=pes_cfg, tournament=False)
    tournament = compute_pes_matchup(base, cfg=pes_cfg, tournament=True)

    is_tour = _safe_bool(base, backtest_cfg.is_tournament_col)
    combined = base.copy()
    pes_cols = [
        backtest_cfg.pgs_team_col,
        backtest_cfg.ces_team_col,
        backtest_cfg.pes_team_col,
        backtest_cfg.pgs_opponent_col,
        backtest_cfg.ces_opponent_col,
        backtest_cfg.pes_opponent_col,
        backtest_cfg.pes_diff_col,
        backtest_cfg.pes_quadrant_col,
    ]
    for col in pes_cols:
        if col in regular.columns and col in tournament.columns:
            combined[col] = np.where(is_tour, tournament[col], regular[col])
        elif col in tournament.columns:
            combined[col] = tournament[col]

    validation = run_validation_backtest(
        df=combined,
        cfg=backtest_cfg,
        edge_threshold=edge_threshold,
        pes_diff_threshold=pes_diff_threshold,
        min_sample=min_sample,
        result_cols=result_cols,
    )
    enhancement = run_enhancement_backtest(
        df=combined,
        cfg=backtest_cfg,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        result_cols=result_cols,
    )
    flip = run_flip_backtest(
        df=combined,
        cfg=backtest_cfg,
        edge_threshold=edge_threshold,
        pes_flip_threshold=pes_flip_threshold,
        min_sample=max(100, min_sample // 2),
        result_cols=result_cols,
    )

    val_tour = validation.get("tournament", pd.DataFrame())
    enh_df = enhancement.get("results", pd.DataFrame())
    flip_df = flip.get("results", pd.DataFrame())
    sweep_df = enhancement.get("threshold_sweep", pd.DataFrame())

    agree_lift = np.nan
    disagree_lift = np.nan
    if not val_tour.empty:
        agree_rows = val_tour[val_tour["population"] == "pes_agrees"]
        disagree_rows = val_tour[val_tour["population"] == "pes_disagrees"]
        if not agree_rows.empty:
            agree_lift = float(pd.to_numeric(agree_rows["lift_over_base"], errors="coerce").max())
        if not disagree_rows.empty:
            disagree_lift = float(pd.to_numeric(disagree_rows["lift_over_base"], errors="coerce").min())

    best_enhancement_threshold = np.nan
    best_enhancement_lift = np.nan
    if not sweep_df.empty:
        sig = sweep_df[sweep_df["significant"]]
        target = sig if not sig.empty else sweep_df
        best = target.sort_values(["lift", "n"], ascending=[False, False]).head(1)
        if not best.empty:
            best_enhancement_threshold = float(best["cutoff"].iloc[0])
            best_enhancement_lift = float(best["lift"].iloc[0])

    flip_recommendation = "monitor"
    flip_best_rate = np.nan
    flip_best_p = np.nan
    if not flip_df.empty:
        flip_best = flip_df.sort_values(["flip_hit_rate", "p_value"], ascending=[False, True]).iloc[0]
        flip_best_rate = float(flip_best["flip_hit_rate"]) if pd.notna(flip_best["flip_hit_rate"]) else np.nan
        flip_best_p = float(flip_best["p_value"]) if pd.notna(flip_best["p_value"]) else np.nan
        if pd.notna(flip_best_rate) and flip_best_rate > 0.52 and pd.notna(flip_best_p) and flip_best_p < 0.10:
            flip_recommendation = "investigate_flip_signal"
        elif pd.notna(flip_best_rate) and flip_best_rate < 0.48:
            flip_recommendation = "avoid_flip_signal"

    tourney_n = int(_safe_bool(combined, backtest_cfg.is_tournament_col).sum())
    overall_verdict = "PES NOT ADDITIVE"
    if tourney_n < min_sample:
        overall_verdict = "INSUFFICIENT TOURNAMENT SAMPLE"
    else:
        has_validation_spread = pd.notna(agree_lift) and pd.notna(disagree_lift) and agree_lift > 0 and disagree_lift < 0
        has_enhancement = pd.notna(best_enhancement_lift) and best_enhancement_lift >= 0.03
        has_flip = pd.notna(flip_best_rate) and flip_best_rate > 0.52 and pd.notna(flip_best_p) and flip_best_p < 0.10
        if has_validation_spread and has_enhancement:
            overall_verdict = "PES VALIDATES AND ENHANCES MODEL"
        elif has_validation_spread:
            overall_verdict = "PES VALIDATES ONLY"
        elif has_flip:
            overall_verdict = "PES INDEPENDENT SIGNAL"

    print("\n=== PES Unified Summary ===")
    print(f"PES agreement lift (Mode A best tournament): {_format_pct(agree_lift)}")
    print(f"PES disagreement drop (Mode A tournament): {_format_pct(disagree_lift)}")
    print(
        f"Best enhancement threshold (Mode B sweep): "
        f"{best_enhancement_threshold if pd.notna(best_enhancement_threshold) else 'nan'} "
        f"lift={_format_pct(best_enhancement_lift)}"
    )
    print(f"Flip recommendation (Mode C): {flip_recommendation}")
    print(f"Overall verdict: {overall_verdict}")

    if export_csv:
        stamp = pd.Timestamp.now(tz="UTC").isoformat()
        metadata = {
            "timestamp_utc": stamp,
            "edge_threshold": edge_threshold,
            "pes_diff_threshold": pes_diff_threshold,
            "total_tournament_games": tourney_n,
            "pes_weight_context": "mixed_regular_and_tournament",
        }
        validation_export = pd.concat(
            [validation.get("full", pd.DataFrame()), validation.get("tournament", pd.DataFrame())],
            ignore_index=True,
            sort=False,
        )
        _write_csv_with_metadata("pes_validation_results.csv", validation_export, metadata)
        _write_csv_with_metadata("pes_enhancement_results.csv", enh_df, metadata)
        flip_export = pd.concat(
            [flip.get("results", pd.DataFrame()), flip.get("seed_bands", pd.DataFrame()), flip.get("magnitude_bands", pd.DataFrame())],
            ignore_index=True,
            sort=False,
        )
        _write_csv_with_metadata("pes_flip_results.csv", flip_export, metadata)

    return {
        "validation": validation,
        "enhancement": enhancement,
        "flip": flip,
        "overall_verdict": overall_verdict,
        "best_enhancement_threshold": best_enhancement_threshold,
        "flip_recommendation": flip_recommendation,
        "data_with_pes": combined,
    }


if __name__ == "__main__":
    pes_cfg = PESConfig()
    bt_cfg = BacktestConfig()

    rng = np.random.default_rng(42)
    n = 3000

    is_tournament = rng.random(n) < 0.15
    model_edge = rng.normal(loc=2.0, scale=5.5, size=n)
    model_cover_prob = 1.0 / (1.0 + np.exp(-(model_edge - 0.5) / 5.0))
    model_ml_prob = 1.0 / (1.0 + np.exp(-(model_edge + 1.5) / 4.5))
    total_over_prob = np.clip(rng.normal(0.50, 0.08, size=n), 0.1, 0.9)

    synthetic = pd.DataFrame(
        {
            bt_cfg.model_edge_col: model_edge,
            bt_cfg.covered_col: rng.binomial(1, model_cover_prob),
            bt_cfg.ml_won_col: rng.binomial(1, model_ml_prob),
            bt_cfg.total_covered_col: rng.binomial(1, total_over_prob),
            bt_cfg.final_margin_col: rng.normal(loc=model_edge, scale=11.0, size=n),
            bt_cfg.season_col: rng.choice([2022, 2023, 2024, 2025], size=n),
            bt_cfg.is_tournament_col: is_tournament.astype(int),
            bt_cfg.tourney_round_col: np.where(is_tournament, rng.choice([1, 2], size=n, p=[0.7, 0.3]), 0),
            bt_cfg.seed_team_col: rng.integers(1, 17, size=n),
            bt_cfg.seed_opponent_col: rng.integers(1, 17, size=n),
            bt_cfg.is_underdog_col: (rng.random(n) < 0.47).astype(int),
            "team_name": [f"team_{i}" for i in range(n)],
            # PES inputs (team/opponent paired)
            f"{pes_cfg.oreb_pct}_team": np.clip(rng.normal(0.31, 0.05, size=n), 0.18, 0.45),
            f"{pes_cfg.oreb_pct}_opponent": np.clip(rng.normal(0.30, 0.05, size=n), 0.18, 0.45),
            f"{pes_cfg.tov_pct}_team": np.clip(rng.normal(0.16, 0.03, size=n), 0.08, 0.28),
            f"{pes_cfg.tov_pct}_opponent": np.clip(rng.normal(0.17, 0.03, size=n), 0.08, 0.30),
            f"{pes_cfg.pace_rank}_team": rng.integers(1, 364, size=n),
            f"{pes_cfg.pace_rank}_opponent": rng.integers(1, 364, size=n),
            f"{pes_cfg.forced_tov_rate}_team": np.clip(rng.normal(0.18, 0.04, size=n), 0.08, 0.35),
            f"{pes_cfg.forced_tov_rate}_opponent": np.clip(rng.normal(0.17, 0.04, size=n), 0.08, 0.35),
            f"{pes_cfg.efg_pct}_team": np.clip(rng.normal(0.52, 0.04, size=n), 0.40, 0.68),
            f"{pes_cfg.efg_pct}_opponent": np.clip(rng.normal(0.51, 0.04, size=n), 0.40, 0.68),
            f"{pes_cfg.ppp}_team": np.clip(rng.normal(1.05, 0.11, size=n), 0.70, 1.50),
            f"{pes_cfg.ppp}_opponent": np.clip(rng.normal(1.04, 0.11, size=n), 0.70, 1.50),
            f"{pes_cfg.fta_rate}_team": np.clip(rng.normal(0.30, 0.06, size=n), 0.10, 0.50),
            f"{pes_cfg.fta_rate}_opponent": np.clip(rng.normal(0.29, 0.06, size=n), 0.10, 0.50),
            f"{pes_cfg.opp_efg_pct}_team": np.clip(rng.normal(0.49, 0.04, size=n), 0.36, 0.62),
            f"{pes_cfg.opp_efg_pct}_opponent": np.clip(rng.normal(0.50, 0.04, size=n), 0.36, 0.62),
        }
    )

    with_pes = compute_pes_matchup(synthetic, cfg=pes_cfg, tournament=True)
    print("\nPES tier preview:")
    preview = with_pes[["team_name", "pes_team", "pes_diff"]].head(5).copy()
    preview["pes_tier"] = preview["pes_team"].map(get_pes_tier)
    print(preview.to_string(index=False))

    plot_pes_quadrant(with_pes, label_col="team_name", highlight_seeds=[1, 5, 12], tournament_only=True)

    full_results = run_pes_full_backtest(
        df=synthetic,
        pes_cfg=pes_cfg,
        backtest_cfg=bt_cfg,
        edge_threshold=4.0,
        pes_diff_threshold=5.0,
        pes_flip_threshold=5.0,
        min_sample=150,
        result_cols=[bt_cfg.covered_col, bt_cfg.ml_won_col, bt_cfg.total_covered_col],
        export_csv=True,
    )
    print("\nFinal overall verdict:", full_results["overall_verdict"])
