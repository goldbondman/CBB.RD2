"""Situational layer backtesting module for CAGE.

This module provides reusable utilities for evaluating situational filters on top
of base model signals for totals-over and underdog ATS/ML markets.

Design goals:
- Single-file, self-contained implementation.
- Column names are fully configurable via ``ColumnConfig``.
- Strict anti-overfitting controls (p-value, minimum sample, season consistency,
  and collinearity checks) are applied throughout.
- Lightweight dependencies: pandas, numpy, scipy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class ColumnConfig:
    """Canonical mapping for all source column names used by the module.

    Notes
    -----
    - All game-data references in this module route through this dataclass.
    - Additional optional fields are included to support richer layer logic while
      keeping backward compatibility when columns are unavailable.
    """

    model_edge: str = "model_edge"
    covered: str = "covered"
    total_covered: str = "total_covered"
    ml_won: str = "ml_won"
    season: str = "season"
    is_underdog: str = "is_underdog"
    model_projects_over: str = "model_projects_over"
    opponent_name: str = "opponent_name"
    rest_days_team: str = "rest_days_team"
    rest_days_opponent: str = "rest_days_opponent"
    pace_team: str = "pace_team"
    pace_opponent: str = "pace_opponent"
    efg_pct_team: str = "efg_pct_team"
    efg_pct_opponent: str = "efg_pct_opponent"
    efg_allowed_team: str = "efg_allowed_team"
    efg_allowed_opponent: str = "efg_allowed_opponent"
    oreb_pct_team: str = "oreb_pct_team"
    dreb_pct_opp: str = "dreb_pct_opp"
    three_pa_rate_team: str = "three_pa_rate_team"
    three_pa_rate_opponent: str = "three_pa_rate_opponent"
    fta_rate_team: str = "fta_rate_team"
    fta_rate_opponent: str = "fta_rate_opponent"
    def_rating_team: str = "def_rating_team"
    def_rating_opponent: str = "def_rating_opponent"
    mti_team: str = "mti_team"
    mti_opponent: str = "mti_opponent"
    spr_team: str = "spr_team"
    spr_opponent: str = "spr_opponent"
    odi_team: str = "odi_team"
    odi_opponent: str = "odi_opponent"
    pmi: str = "pmi"
    dpc_team: str = "dpc_team"
    dpc_opponent: str = "dpc_opponent"
    pei_gap: str = "pei_gap"
    sci_team: str = "sci_team"
    sci_opponent: str = "sci_opponent"
    conference_team: str = "conference_team"
    conference_opponent: str = "conference_opponent"
    is_neutral_site: str = "is_neutral_site"
    is_tournament: str = "is_tournament"
    tourney_round: str = "tourney_round"
    prior_margin_team: str = "prior_margin_team"
    prior_opponent_name: str = "prior_opponent_name"
    meeting_number: str = "meeting_number"
    line_open: str = "line_open"
    line_close: str = "line_close"
    public_pct_favorite: str = "public_pct_favorite"

    # Optional enrichments for layer definitions
    to_pct_team: str = "to_pct_team"
    to_pct_opponent: str = "to_pct_opponent"
    pace_allowed_rank_team: str = "pace_allowed_rank_team"
    pace_allowed_rank_opponent: str = "pace_allowed_rank_opponent"
    opp_two_pt_allowed_rank_team: str = "opp_two_pt_allowed_rank_team"
    opp_two_pt_allowed_rank_opponent: str = "opp_two_pt_allowed_rank_opponent"
    previous_ot_team: str = "previous_ot_team"
    previous_ot_opponent: str = "previous_ot_opponent"
    is_conference_tournament: str = "is_conference_tournament"
    season_low_total_team: str = "season_low_total_team"
    season_low_total_opponent: str = "season_low_total_opponent"
    projected_total: str = "projected_total"
    market_total_close: str = "market_total_close"
    total_edge: str = "total_edge"

    opponent_rank: str = "opponent_rank"
    opponent_win_streak: str = "opponent_win_streak"
    top25_loss_share_team: str = "top25_loss_share_team"
    favorite_soft_schedule_index: str = "favorite_soft_schedule_index"
    sos_adj_eff_gap: str = "sos_adj_eff_gap"
    favorite_fastbreak_points_allowed: str = "favorite_fastbreak_points_allowed"
    opponent_three_pt_allowed_pct: str = "opponent_three_pt_allowed_pct"
    elimination_game_flag: str = "elimination_game_flag"

    # V2 extension fields
    final_margin: str = "final_margin"
    ml_implied_prob: str = "ml_implied_prob"
    model_win_prob: str = "model_win_prob"
    model_projects_under: str = "model_projects_under"
    pace_rank_team: str = "pace_rank_team"
    pace_rank_opponent: str = "pace_rank_opponent"
    three_pt_def_rank_team: str = "three_pt_def_rank_team"
    three_pt_def_rank_opponent: str = "three_pt_def_rank_opponent"
    def_rating_rank_team: str = "def_rating_rank_team"
    def_rating_rank_opponent: str = "def_rating_rank_opponent"
    adj_efficiency_margin_team: str = "adj_efficiency_margin_team"
    adj_efficiency_margin_opponent: str = "adj_efficiency_margin_opponent"
    fast_break_pts_team: str = "fast_break_pts_team"
    fast_break_pts_allowed_opponent: str = "fast_break_pts_allowed_opponent"
    tov_pct_team: str = "tov_pct_team"
    tov_pct_opponent: str = "tov_pct_opponent"
    steal_rate_team: str = "steal_rate_team"
    steal_rate_opponent: str = "steal_rate_opponent"
    block_rate_team: str = "block_rate_team"
    block_rate_opponent: str = "block_rate_opponent"
    bench_minutes_pct_team: str = "bench_minutes_pct_team"
    bench_minutes_pct_opponent: str = "bench_minutes_pct_opponent"
    second_half_margin_team: str = "second_half_margin_team"
    second_half_def_eff_opponent: str = "second_half_def_eff_opponent"
    ane_team: str = "ane_team"
    ane_opponent: str = "ane_opponent"
    is_power_conf_team: str = "is_power_conf_team"
    is_power_conf_opponent: str = "is_power_conf_opponent"
    prev_game_ot_team: str = "prev_game_ot_team"
    prev_game_ot_opponent: str = "prev_game_ot_opponent"
    consecutive_road_games_opponent: str = "consecutive_road_games_opponent"
    days_since_last_game_team: str = "days_since_last_game_team"
    days_since_last_game_opponent: str = "days_since_last_game_opponent"
    win_streak_opponent: str = "win_streak_opponent"
    sos_rank_team: str = "sos_rank_team"
    sos_rank_opponent: str = "sos_rank_opponent"
    losses_vs_top25_pct_team: str = "losses_vs_top25_pct_team"
    wins_vs_bottom_half_pct_opponent: str = "wins_vs_bottom_half_pct_opponent"
    ft_pct_team: str = "ft_pct_team"
    ft_pct_opponent: str = "ft_pct_opponent"
    ppp_team: str = "ppp_team"
    ppp_opponent: str = "ppp_opponent"
    half_court_rate_team: str = "half_court_rate_team"
    half_court_rate_opponent: str = "half_court_rate_opponent"
    transition_rate_team: str = "transition_rate_team"
    transition_rate_opponent: str = "transition_rate_opponent"
    interior_def_rank_team: str = "interior_def_rank_team"
    interior_def_rank_opponent: str = "interior_def_rank_opponent"
    tourney_experience_score_team: str = "tourney_experience_score_team"
    tourney_experience_score_opponent: str = "tourney_experience_score_opponent"
    conf_strength_rank_team: str = "conf_strength_rank_team"
    conf_strength_rank_opponent: str = "conf_strength_rank_opponent"
    coach_upset_rate_team: str = "coach_upset_rate_team"
    coach_upset_rate_opponent: str = "coach_upset_rate_opponent"
    seed_team: str = "seed_team"
    seed_opponent: str = "seed_opponent"
    seed_diff: str = "seed_diff"
    region_team: str = "region_team"
    region_opponent: str = "region_opponent"
    pgs_team: str = "pgs_team"
    ces_team: str = "ces_team"
    pes_team: str = "pes_team"
    pgs_opponent: str = "pgs_opponent"
    ces_opponent: str = "ces_opponent"
    pes_opponent: str = "pes_opponent"
    pes_diff: str = "pes_diff"
    pes_quadrant: str = "pes_quadrant"


# -----------------------------
# Internal helper utilities
# -----------------------------

def _safe_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    """Return a numeric series for ``col`` or a default-valued fallback series."""
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def _safe_bool(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    """Return a boolean series for ``col`` with robust coercion and fallback."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="bool")
    raw = df[col]
    if pd.api.types.is_bool_dtype(raw):
        return raw.fillna(default)
    if pd.api.types.is_numeric_dtype(raw):
        return raw.fillna(0).astype(float) != 0
    normalized = raw.astype(str).str.strip().str.lower()
    return normalized.isin({"1", "true", "t", "yes", "y"})


def _safe_text(df: pd.DataFrame, col: str, default: str = "") -> pd.Series:
    """Return text series for ``col`` or fallback values."""
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].fillna(default).astype(str)


def _positive_result_mask(series: pd.Series) -> pd.Series:
    """Mask rows where binary outcome is available and valid (0/1)."""
    v = pd.to_numeric(series, errors="coerce")
    return v.isin([0, 1])


def _result_hits(series: pd.Series) -> int:
    """Count positive hits from a binary 0/1-like series."""
    v = pd.to_numeric(series, errors="coerce")
    return int(v.fillna(0).sum())


def _quantile_or_default(s: pd.Series, q: float, default: float) -> float:
    """Return quantile if available; otherwise return fallback."""
    clean = pd.to_numeric(s, errors="coerce").dropna()
    if clean.empty:
        return default
    return float(clean.quantile(q))


# -----------------------------
# Core statistical utilities
# -----------------------------

def compute_hit_rate(series: pd.Series) -> Tuple[float, int]:
    """Compute hit rate and sample size from a binary 0/1-like result series.

    Parameters
    ----------
    series:
        Binary-like outcome series where 1 indicates success and 0 indicates fail.

    Returns
    -------
    (hit_rate, n):
        ``hit_rate`` is float in [0, 1] or NaN when no valid rows exist;
        ``n`` is valid sample count.
    """
    valid = _positive_result_mask(series)
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0
    hits = _result_hits(series[valid])
    return float(hits / n), n


def compute_blowout_rate(df: pd.DataFrame, margin_col: str, threshold: float) -> Tuple[float, int]:
    """Compute blowout-rate from a realized margin column.

    Parameters
    ----------
    df:
        Source DataFrame.
    margin_col:
        Column containing realized margin values.
    threshold:
        Blowout threshold applied to absolute margin.

    Returns
    -------
    (rate, n):
        ``rate`` is the share of rows with ``abs(margin) >= threshold`` and
        ``n`` is sample size with non-null margin values.
    """
    if margin_col not in df.columns:
        return np.nan, 0
    margin = pd.to_numeric(df[margin_col], errors="coerce")
    valid = margin.notna()
    n = int(valid.sum())
    if n == 0:
        return np.nan, 0
    rate = float((margin.loc[valid].abs() >= float(threshold)).mean())
    return rate, n


def proportion_z_test(
    p1: float,
    n1: int,
    p2: float,
    n2: int,
) -> Tuple[float, float]:
    """One-tailed z-test for difference in proportions.

    Tests H1: p1 > p2.

    Returns
    -------
    (z_stat, p_value)
    """
    if any(
        [
            n1 <= 0,
            n2 <= 0,
            np.isnan(p1),
            np.isnan(p2),
            p1 < 0,
            p1 > 1,
            p2 < 0,
            p2 > 1,
        ]
    ):
        return np.nan, np.nan

    x1 = p1 * n1
    x2 = p2 * n2
    pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(pooled * (1.0 - pooled) * (1.0 / n1 + 1.0 / n2))
    if se == 0:
        return np.nan, np.nan

    z = (p1 - p2) / se
    p_val = float(stats.norm.sf(z))  # one-tailed, right side
    return float(z), p_val


def wilson_confidence_interval(
    hits: int,
    n: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Wilson score confidence interval for a Bernoulli proportion."""
    if n <= 0:
        return np.nan, np.nan
    alpha = 1.0 - confidence
    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    phat = hits / n
    denom = 1.0 + z**2 / n
    center = (phat + z**2 / (2.0 * n)) / denom
    margin = z * np.sqrt((phat * (1.0 - phat) + z**2 / (4.0 * n)) / n) / denom
    return float(center - margin), float(center + margin)


def chi2_independence_test(
    df: pd.DataFrame,
    layer_col: str,
    result_col: str,
) -> Tuple[float, float]:
    """Chi-square test of independence between layer activation and result."""
    if layer_col not in df.columns or result_col not in df.columns:
        return np.nan, np.nan

    table = pd.crosstab(df[layer_col].fillna(False), pd.to_numeric(df[result_col], errors="coerce"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, np.nan

    stat, p_value, _, _ = stats.chi2_contingency(table)
    return float(stat), float(p_value)


def redundancy_check(
    df: pd.DataFrame,
    layer_cols: Sequence[str],
    edge_col: str,
    result_col: str,
    edge_threshold: float,
) -> pd.DataFrame:
    """Compute Spearman pairwise correlations among layers and flag high overlap.

    Parameters
    ----------
    df:
        Full game-level frame.
    layer_cols:
        Boolean/indicator columns representing candidate layers.
    edge_col:
        Numeric base edge column.
    result_col:
        Binary outcome column, used only to restrict valid rows.
    edge_threshold:
        Minimum edge for inclusion in redundancy sample.

    Returns
    -------
    DataFrame with columns:
    ``layer_a``, ``layer_b``, ``spearman_corr``, ``abs_corr``,
    ``high_collinearity`` (abs corr > 0.5).
    """
    use_cols = [c for c in layer_cols if c in df.columns]
    if len(use_cols) < 2:
        return pd.DataFrame(columns=["layer_a", "layer_b", "spearman_corr", "abs_corr", "high_collinearity"])

    edge = _safe_series(df, edge_col)
    result_ok = _positive_result_mask(df[result_col]) if result_col in df.columns else pd.Series(False, index=df.index)
    scope = df.loc[(edge >= edge_threshold) & result_ok, use_cols].copy()

    if scope.empty:
        return pd.DataFrame(columns=["layer_a", "layer_b", "spearman_corr", "abs_corr", "high_collinearity"])

    for c in use_cols:
        scope[c] = _safe_bool(scope, c).astype(int)

    corr = scope.corr(method="spearman")
    rows: List[Dict[str, Any]] = []
    for i, a in enumerate(use_cols):
        for b in use_cols[i + 1 :]:
            value = float(corr.loc[a, b]) if not pd.isna(corr.loc[a, b]) else np.nan
            rows.append(
                {
                    "layer_a": a,
                    "layer_b": b,
                    "spearman_corr": value,
                    "abs_corr": abs(value) if not np.isnan(value) else np.nan,
                    "high_collinearity": bool(abs(value) > 0.5) if not np.isnan(value) else False,
                }
            )

    return pd.DataFrame(rows).sort_values(["high_collinearity", "abs_corr"], ascending=[False, False])


def season_consistency_check(
    df: pd.DataFrame,
    layer_col: str,
    season_col: str,
    result_col: str,
    edge_col: str,
    edge_threshold: float,
) -> pd.DataFrame:
    """Evaluate per-season consistency of a layer over base edge-qualified games.

    A warning is printed when the layer has positive lift in fewer than 60% of
    seasons with adequate samples.
    """
    required = [layer_col, season_col, result_col, edge_col]
    if any(col not in df.columns for col in required):
        return pd.DataFrame(
            columns=[
                "season",
                "base_n",
                "base_hit_rate",
                "layer_n",
                "layer_hit_rate",
                "lift",
                "positive_lift",
            ]
        )

    edge = _safe_series(df, edge_col)
    result_mask = _positive_result_mask(df[result_col])
    base_scope = df.loc[(edge >= edge_threshold) & result_mask].copy()
    if base_scope.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "base_n",
                "base_hit_rate",
                "layer_n",
                "layer_hit_rate",
                "lift",
                "positive_lift",
            ]
        )

    base_scope[layer_col] = _safe_bool(base_scope, layer_col)
    rows: List[Dict[str, Any]] = []
    for season_value, chunk in base_scope.groupby(season_col, dropna=False):
        base_rate, base_n = compute_hit_rate(chunk[result_col])
        layer_chunk = chunk.loc[chunk[layer_col]]
        layer_rate, layer_n = compute_hit_rate(layer_chunk[result_col])
        lift = layer_rate - base_rate if base_n > 0 and layer_n > 0 else np.nan
        rows.append(
            {
                "season": season_value,
                "base_n": base_n,
                "base_hit_rate": base_rate,
                "layer_n": layer_n,
                "layer_hit_rate": layer_rate,
                "lift": lift,
                "positive_lift": bool(lift > 0) if not np.isnan(lift) else False,
            }
        )

    result = pd.DataFrame(rows)
    denom = int((result["layer_n"] > 0).sum()) if not result.empty else 0
    if denom > 0:
        positive_ratio = float(result["positive_lift"].sum() / denom)
        if positive_ratio < 0.60:
            print(
                f"[WARN] Layer '{layer_col}' positive in only {positive_ratio:.1%} "
                "of seasons (<60%)."
            )

    return result.sort_values("season")


# -----------------------------
# Threshold sweeper
# -----------------------------


class ThresholdSweeper:
    """Sweep thresholds for a continuous feature and measure filtered lift.

    Parameters
    ----------
    df:
        Game-level dataframe.
    metric_col:
        Continuous metric to threshold.
    result_col:
        Binary result column (0/1).
    edge_col:
        Base edge column used for candidate scope.
    edge_threshold:
        Minimum edge value for inclusion.
    sweep_range:
        Tuple of (min_cutoff, max_cutoff).
    step:
        Step size for threshold sweep.
    direction:
        ``"above"`` keeps rows metric >= cutoff; ``"below"`` keeps metric <= cutoff.
    min_sample:
        Minimum sample size needed for a threshold row to be retained.
    p_threshold:
        Significance threshold.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        metric_col: str,
        result_col: str,
        edge_col: str,
        edge_threshold: float,
        sweep_range: Tuple[float, float],
        step: float,
        direction: str = "above",
        min_sample: int = 150,
        p_threshold: float = 0.05,
    ) -> None:
        self.df = df.copy()
        self.metric_col = metric_col
        self.result_col = result_col
        self.edge_col = edge_col
        self.edge_threshold = float(edge_threshold)
        self.sweep_range = sweep_range
        self.step = float(step)
        self.direction = direction
        self.min_sample = int(min_sample)
        self.p_threshold = float(p_threshold)
        self._results: Optional[pd.DataFrame] = None

    def run(self) -> pd.DataFrame:
        """Run the threshold sweep and return ranked results."""
        if self.metric_col not in self.df.columns:
            self._results = pd.DataFrame(
                columns=["cutoff", "n", "hit_rate", "lift", "p_value", "significant"]
            )
            return self._results

        metric = _safe_series(self.df, self.metric_col)
        edge = _safe_series(self.df, self.edge_col)
        valid_result = _positive_result_mask(self.df[self.result_col])
        base_mask = (edge >= self.edge_threshold) & valid_result & metric.notna()

        base_hit, base_n = compute_hit_rate(self.df.loc[base_mask, self.result_col])
        if base_n == 0:
            self._results = pd.DataFrame(
                columns=["cutoff", "n", "hit_rate", "lift", "p_value", "significant"]
            )
            return self._results

        low, high = self.sweep_range
        cutoffs = np.arange(low, high + (self.step / 2.0), self.step)
        rows: List[Dict[str, Any]] = []
        for cutoff in cutoffs:
            if self.direction == "above":
                mask = base_mask & (metric >= cutoff)
            else:
                mask = base_mask & (metric <= cutoff)

            hit_rate, n = compute_hit_rate(self.df.loc[mask, self.result_col])
            if n < self.min_sample:
                continue

            lift = hit_rate - base_hit
            z_stat, p_value = proportion_z_test(hit_rate, n, base_hit, base_n)
            rows.append(
                {
                    "cutoff": float(np.round(cutoff, 6)),
                    "n": int(n),
                    "hit_rate": float(hit_rate),
                    "lift": float(lift),
                    "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
                    "significant": bool(
                        (not np.isnan(p_value)) and p_value <= self.p_threshold and lift > 0
                    ),
                }
            )

        self._results = pd.DataFrame(rows)
        if self._results.empty:
            return self._results

        return self._results.sort_values(
            ["significant", "lift", "p_value", "n"],
            ascending=[False, False, True, False],
        ).reset_index(drop=True)

    def plot_ascii(self, width: int = 36) -> None:
        """Print an ASCII bar chart of hit rates across swept thresholds."""
        if self._results is None:
            _ = self.run()
        if self._results is None or self._results.empty:
            print("No sweep results to plot.")
            return

        ordered = self._results.sort_values("cutoff")
        min_hr = float(ordered["hit_rate"].min())
        max_hr = float(ordered["hit_rate"].max())
        span = max(max_hr - min_hr, 1e-9)

        print(f"\nASCII hit-rate sweep: {self.metric_col} ({self.direction})")
        for _, row in ordered.iterrows():
            scaled = int(((float(row["hit_rate"]) - min_hr) / span) * width)
            bar = "#" * max(1, scaled)
            sig = "*" if bool(row["significant"]) else " "
            print(
                f"{row['cutoff']:>7.3f} | {bar:<{width}} "
                f"hr={row['hit_rate']:.3f} n={int(row['n'])} {sig}"
            )


# -----------------------------
# Layer analysis core
# -----------------------------


def analyze_layer(
    df: pd.DataFrame,
    layer_col: str,
    result_col: str,
    edge_col: str,
    edge_threshold: float,
    label: Optional[str] = None,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    lift_threshold: float = 0.03,
) -> Dict[str, Any]:
    """Analyze one boolean layer against a binary outcome.

    Returns
    -------
    Dict containing base, layered, anti-layer metrics and verdict metadata.
    """
    name = label or layer_col
    if layer_col not in df.columns or result_col not in df.columns or edge_col not in df.columns:
        return {
            "label": name,
            "base_n": 0,
            "base_hit_rate": np.nan,
            "layered_n": 0,
            "layered_hit_rate": np.nan,
            "anti_layer_n": 0,
            "anti_layer_hit_rate": np.nan,
            "lift": np.nan,
            "z_stat": np.nan,
            "p_value": np.nan,
            "chi2_p": np.nan,
            "ci_95": (np.nan, np.nan),
            "sufficient_sample": False,
            "significant": False,
            "meaningful_lift": False,
            "verdict": "⛔ INSUFFICIENT SAMPLE",
        }

    result_ok = _positive_result_mask(df[result_col])
    edge = _safe_series(df, edge_col)
    layer = _safe_bool(df, layer_col)

    base_mask = (edge >= edge_threshold) & result_ok
    layered_mask = base_mask & layer
    anti_mask = base_mask & (~layer)

    base_rate, base_n = compute_hit_rate(df.loc[base_mask, result_col])
    layered_rate, layered_n = compute_hit_rate(df.loc[layered_mask, result_col])
    anti_rate, anti_n = compute_hit_rate(df.loc[anti_mask, result_col])

    lift = layered_rate - base_rate if layered_n > 0 and base_n > 0 else np.nan

    if layered_n > 0:
        layered_hits = _result_hits(df.loc[layered_mask, result_col])
        ci_95 = wilson_confidence_interval(layered_hits, layered_n, confidence=0.95)
    else:
        ci_95 = (np.nan, np.nan)

    if layered_n > 0 and anti_n > 0:
        z_stat, p_value = proportion_z_test(layered_rate, layered_n, anti_rate, anti_n)
    else:
        z_stat, p_value = np.nan, np.nan

    chi2_stat, chi2_p = chi2_independence_test(df.loc[base_mask, [layer_col, result_col]], layer_col, result_col)
    _ = chi2_stat

    sufficient_sample = layered_n >= min_sample
    meaningful_lift = (not np.isnan(lift)) and (lift >= lift_threshold)
    significant = (not np.isnan(p_value)) and (p_value <= p_threshold)

    if not sufficient_sample:
        verdict = "⛔ INSUFFICIENT SAMPLE"
    elif np.isnan(lift) or lift < 0:
        verdict = "❌ NEGATIVE"
    elif meaningful_lift and significant:
        verdict = "✅ VALID"
    elif not meaningful_lift:
        verdict = "⚠️ REDUNDANT"
    else:
        verdict = "🔍 PROMISING"

    return {
        "label": name,
        "base_n": int(base_n),
        "base_hit_rate": float(base_rate) if not np.isnan(base_rate) else np.nan,
        "layered_n": int(layered_n),
        "layered_hit_rate": float(layered_rate) if not np.isnan(layered_rate) else np.nan,
        "anti_layer_n": int(anti_n),
        "anti_layer_hit_rate": float(anti_rate) if not np.isnan(anti_rate) else np.nan,
        "lift": float(lift) if not np.isnan(lift) else np.nan,
        "z_stat": float(z_stat) if not np.isnan(z_stat) else np.nan,
        "p_value": float(p_value) if not np.isnan(p_value) else np.nan,
        "chi2_p": float(chi2_p) if not np.isnan(chi2_p) else np.nan,
        "ci_95": ci_95,
        "sufficient_sample": bool(sufficient_sample),
        "significant": bool(significant),
        "meaningful_lift": bool(meaningful_lift),
        "verdict": verdict,
    }


# -----------------------------
# Analyzer helpers
# -----------------------------


def _compute_total_edge(df: pd.DataFrame, cfg: ColumnConfig) -> pd.Series:
    """Compute total edge series with safe fallback precedence."""
    if cfg.total_edge in df.columns:
        return _safe_series(df, cfg.total_edge)

    projected_total = _safe_series(df, cfg.projected_total)
    market_total = _safe_series(df, cfg.market_total_close)
    if projected_total.notna().any() and market_total.notna().any():
        return projected_total - market_total

    # fallback for mixed schemas
    return _safe_series(df, cfg.model_edge)


def _is_early_round(series: pd.Series) -> pd.Series:
    """Flag early-round tournament labels from textual round names."""
    txt = series.fillna("").astype(str).str.lower()
    return txt.str.contains("first|round of 64|opening|play-in", regex=True)


def _bool_name(name: str) -> str:
    """Generate stable temporary boolean column names for layer masks."""
    return f"_layer_{name}"


def analyze_over_layers(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    edge_threshold: float,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    output_path: str | Path = "over_layer_results.csv",
) -> pd.DataFrame:
    """Analyze over-market situational layers on top of model-over base signals.

    Base scope:
    - model projects over
    - projected total edge >= ``edge_threshold``
    """
    work = df.copy()

    model_over = _safe_bool(work, cfg.model_projects_over)
    total_edge = _compute_total_edge(work, cfg)
    valid_result = _positive_result_mask(work[cfg.total_covered]) if cfg.total_covered in work.columns else pd.Series(False, index=work.index)
    base_scope = model_over & (total_edge >= edge_threshold) & valid_result

    # Dynamic thresholds
    pace_team = _safe_series(work, cfg.pace_team)
    pace_opp = _safe_series(work, cfg.pace_opponent)
    def_team = _safe_series(work, cfg.def_rating_team)
    def_opp = _safe_series(work, cfg.def_rating_opponent)

    fast_cut = _quantile_or_default(pd.concat([pace_team, pace_opp]), 0.60, 70.0)
    poor_def_cut = _quantile_or_default(pd.concat([def_team, def_opp]), 0.70, 105.0)
    pmi_cut = _quantile_or_default(_safe_series(work, cfg.pmi), 0.65, 0.55)

    # Layer definitions
    layers: Dict[str, pd.Series] = {}

    layers["both_teams_fast_tempo"] = (pace_team >= fast_cut) & (pace_opp >= fast_cut)
    layers["pace_mismatch_over"] = (pace_team - pace_opp).abs() >= 5.0

    pace_allow_rank_team = _safe_series(work, cfg.pace_allowed_rank_team)
    pace_allow_rank_opp = _safe_series(work, cfg.pace_allowed_rank_opponent)
    if pace_allow_rank_team.notna().any() and pace_allow_rank_opp.notna().any():
        layers["neither_team_slows"] = (pace_allow_rank_team > 100) & (pace_allow_rank_opp > 100)
    else:
        layers["neither_team_slows"] = (pace_team >= fast_cut * 0.95) & (pace_opp >= fast_cut * 0.95)

    fta_team = _safe_series(work, cfg.fta_rate_team)
    fta_opp = _safe_series(work, cfg.fta_rate_opponent)
    layers["low_foul_rate_both"] = (fta_team < 0.28) & (fta_opp < 0.28)

    to_team = _safe_series(work, cfg.to_pct_team)
    to_opp = _safe_series(work, cfg.to_pct_opponent)
    layers["both_high_live_tov"] = (to_team >= 0.17) & (to_opp >= 0.17)

    efg_team = _safe_series(work, cfg.efg_pct_team)
    efg_opp = _safe_series(work, cfg.efg_pct_opponent)
    layers["both_efg_above_52"] = (efg_team >= 0.52) & (efg_opp >= 0.52)

    tpa_team = _safe_series(work, cfg.three_pa_rate_team)
    tpa_opp = _safe_series(work, cfg.three_pa_rate_opponent)
    layers["high_combined_3pa_rate"] = ((tpa_team + tpa_opp) / 2.0) >= 0.35

    layers["both_poor_defenses"] = (def_team >= poor_def_cut) & (def_opp >= poor_def_cut)
    layers["low_ft_rate_both"] = (fta_team < 0.28) & (fta_opp < 0.28)

    twopt_rank_team = _safe_series(work, cfg.opp_two_pt_allowed_rank_team)
    twopt_rank_opp = _safe_series(work, cfg.opp_two_pt_allowed_rank_opponent)
    if twopt_rank_team.notna().any() and twopt_rank_opp.notna().any():
        layers["neither_elite_interior_d"] = (twopt_rank_team > 50) & (twopt_rank_opp > 50)
    else:
        efg_allow_team = _safe_series(work, cfg.efg_allowed_team)
        efg_allow_opp = _safe_series(work, cfg.efg_allowed_opponent)
        layers["neither_elite_interior_d"] = (efg_allow_team > 0.48) & (efg_allow_opp > 0.48)

    layers["both_mti_trending_up"] = (_safe_series(work, cfg.mti_team) > 0) & (_safe_series(work, cfg.mti_opponent) > 0)
    layers["high_combined_spr"] = (_safe_series(work, cfg.spr_team) > 0.65) & (_safe_series(work, cfg.spr_opponent) > 0.65)
    layers["both_odi_offensive"] = (_safe_series(work, cfg.odi_team) > 0) & (_safe_series(work, cfg.odi_opponent) > 0)
    layers["high_pmi"] = _safe_series(work, cfg.pmi) > pmi_cut
    layers["both_low_dpc"] = (_safe_series(work, cfg.dpc_team) < 0.40) & (_safe_series(work, cfg.dpc_opponent) < 0.40)

    same_conf = _safe_text(work, cfg.conference_team).str.lower() == _safe_text(work, cfg.conference_opponent).str.lower()
    layers["first_meeting_different_conf"] = (_safe_series(work, cfg.meeting_number) <= 1) & (~same_conf)
    layers["both_coming_off_ot"] = _safe_bool(work, cfg.previous_ot_team) & _safe_bool(work, cfg.previous_ot_opponent)
    layers["early_conf_tourney"] = _safe_bool(work, cfg.is_conference_tournament) & _is_early_round(_safe_text(work, cfg.tourney_round))

    season_low_team = _safe_series(work, cfg.season_low_total_team)
    season_low_opp = _safe_series(work, cfg.season_low_total_opponent)
    market_total = _safe_series(work, cfg.market_total_close)
    poor_opp_def = (_safe_series(work, cfg.def_rating_opponent) >= poor_def_cut) | (_safe_series(work, cfg.def_rating_team) >= poor_def_cut)
    layers["market_anchored_low"] = (
        ((market_total - season_low_team).abs() <= 3.0) | ((market_total - season_low_opp).abs() <= 3.0)
    ) & poor_opp_def

    rows: List[Dict[str, Any]] = []
    temp_cols: List[str] = []

    for layer_name, condition in layers.items():
        temp_col = _bool_name(layer_name)
        work[temp_col] = condition.fillna(False)
        temp_cols.append(temp_col)

        layer_result = analyze_layer(
            df=work.loc[base_scope].copy(),
            layer_col=temp_col,
            result_col=cfg.total_covered,
            edge_col=cfg.model_edge,
            edge_threshold=-np.inf,
            label=layer_name,
            min_sample=min_sample,
            p_threshold=p_threshold,
            lift_threshold=0.03,
        )
        rows.append(layer_result)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        result_df.to_csv(output_path, index=False)
        return result_df

    # Additional anti-overfitting diagnostics on positive candidates
    pass_mask = (result_df["lift"] > 0) & (result_df["layered_n"] >= min_sample)
    pass_layers = result_df.loc[pass_mask, "label"].tolist()

    for layer_name in pass_layers:
        temp_col = _bool_name(layer_name)
        season_df = season_consistency_check(
            df=work.loc[base_scope].copy(),
            layer_col=temp_col,
            season_col=cfg.season,
            result_col=cfg.total_covered,
            edge_col=cfg.model_edge,
            edge_threshold=-np.inf,
        )
        positive_ratio = np.nan
        if not season_df.empty:
            valid_seasons = season_df["layer_n"] > 0
            denom = int(valid_seasons.sum())
            if denom > 0:
                positive_ratio = float((season_df.loc[valid_seasons, "lift"] > 0).mean())
        result_df.loc[result_df["label"] == layer_name, "season_positive_ratio"] = positive_ratio

    red = redundancy_check(
        df=work.loc[base_scope].copy(),
        layer_cols=[_bool_name(name) for name in pass_layers],
        edge_col=cfg.model_edge,
        result_col=cfg.total_covered,
        edge_threshold=-np.inf,
    )
    high_col_pairs = int(red["high_collinearity"].sum()) if not red.empty else 0
    result_df["high_collinearity_pair_count"] = high_col_pairs

    ordered = result_df.sort_values(["verdict", "lift", "p_value", "layered_n"], ascending=[True, False, True, False])
    ordered.to_csv(output_path, index=False)
    return ordered


def _analyze_layer_dual(
    work: pd.DataFrame,
    base_scope: pd.Series,
    layer_col: str,
    cfg: ColumnConfig,
    min_sample: int,
    p_threshold: float,
) -> Dict[str, Any]:
    """Evaluate a layer against both ATS and ML outcomes."""
    ats = analyze_layer(
        df=work.loc[base_scope].copy(),
        layer_col=layer_col,
        result_col=cfg.covered,
        edge_col=cfg.model_edge,
        edge_threshold=-np.inf,
        label=layer_col,
        min_sample=min_sample,
        p_threshold=p_threshold,
        lift_threshold=0.03,
    )
    ml = analyze_layer(
        df=work.loc[base_scope].copy(),
        layer_col=layer_col,
        result_col=cfg.ml_won,
        edge_col=cfg.model_edge,
        edge_threshold=-np.inf,
        label=layer_col,
        min_sample=min_sample,
        p_threshold=p_threshold,
        lift_threshold=0.03,
    )

    return {
        "layer": layer_col.replace("_layer_", ""),
        "ats_n": ats["layered_n"],
        "ats_hit_rate": ats["layered_hit_rate"],
        "ats_lift": ats["lift"],
        "ats_p_value": ats["p_value"],
        "ats_verdict": ats["verdict"],
        "ml_n": ml["layered_n"],
        "ml_hit_rate": ml["layered_hit_rate"],
        "ml_lift": ml["lift"],
        "ml_p_value": ml["p_value"],
        "ml_verdict": ml["verdict"],
    }


def analyze_underdog_layers(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    edge_threshold: float,
    blue_blood_list: Optional[Sequence[str]] = None,
    power_conferences: Optional[Sequence[str]] = None,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    output_path: str | Path = "underdog_layer_results.csv",
) -> pd.DataFrame:
    """Analyze underdog situational layers for ATS and ML outcomes.

    Base scope:
    - game-side is underdog
    - model_edge >= edge_threshold
    """
    work = df.copy()
    blue_bloods = {t.lower().strip() for t in (blue_blood_list or ["duke", "kentucky", "kansas", "north carolina", "michigan state"])}
    power_confs = {c.lower().strip() for c in (power_conferences or ["acc", "big ten", "big 12", "sec", "big east", "pac-12"])}

    is_dog = _safe_bool(work, cfg.is_underdog)
    edge = _safe_series(work, cfg.model_edge)
    base_scope = is_dog & (edge >= edge_threshold)

    opp_name = _safe_text(work, cfg.opponent_name).str.lower().str.strip()
    conf_team = _safe_text(work, cfg.conference_team).str.lower().str.strip()
    conf_opp = _safe_text(work, cfg.conference_opponent).str.lower().str.strip()

    layers: Dict[str, pd.Series] = {}

    # Market bias layers
    layers["blue_blood_opponent"] = opp_name.isin(blue_bloods)
    layers["opponent_top_25_ranked"] = _safe_series(work, cfg.opponent_rank) <= 25
    layers["opponent_on_win_streak"] = _safe_series(work, cfg.opponent_win_streak) >= 5
    layers["underdog_just_lost_big"] = _safe_series(work, cfg.prior_margin_team) <= -20
    layers["mid_major_underdog"] = ~conf_team.isin(power_confs)

    # Efficiency mismatch
    sos_gap = _safe_series(work, cfg.sos_adj_eff_gap)
    spread_abs = _safe_series(work, cfg.line_close).abs()
    if sos_gap.notna().any():
        layers["efficiency_closer_than_spread"] = sos_gap.abs() <= (spread_abs / 2.0)
    else:
        layers["efficiency_closer_than_spread"] = _safe_series(work, cfg.pei_gap).abs() <= (spread_abs / 2.0)

    layers["underdog_losses_vs_top25"] = _safe_series(work, cfg.top25_loss_share_team) >= 0.50
    soft_sched = _safe_series(work, cfg.favorite_soft_schedule_index)
    if soft_sched.notna().any():
        layers["favorite_soft_schedule"] = soft_sched >= 0.60
    else:
        layers["favorite_soft_schedule"] = (
            _safe_series(work, cfg.sci_team) - _safe_series(work, cfg.sci_opponent)
        ) >= 0.10

    layers["pei_gap_small"] = _safe_series(work, cfg.pei_gap).abs() <= np.maximum(1.0, spread_abs * 0.5)

    # Matchup layers
    oreb_cut = _quantile_or_default(_safe_series(work, cfg.oreb_pct_team), 0.75, 0.32)
    dreb_cut = _quantile_or_default(_safe_series(work, cfg.dreb_pct_opp), 0.25, 0.68)
    layers["underdog_oreb_vs_poor_dreb"] = (
        _safe_series(work, cfg.oreb_pct_team) >= oreb_cut
    ) & (_safe_series(work, cfg.dreb_pct_opp) <= dreb_cut)

    pace_cut = _quantile_or_default(_safe_series(work, cfg.pace_team), 0.65, 70.0)
    fb_weak_cut = _quantile_or_default(_safe_series(work, cfg.favorite_fastbreak_points_allowed), 0.65, 12.0)
    layers["underdog_fast_vs_transition_weak"] = (
        _safe_series(work, cfg.pace_team) >= pace_cut
    ) & (_safe_series(work, cfg.favorite_fastbreak_points_allowed) >= fb_weak_cut)

    three_rate_cut = _quantile_or_default(_safe_series(work, cfg.three_pa_rate_team), 0.65, 0.36)
    poor_perimeter_cut = _quantile_or_default(_safe_series(work, cfg.opponent_three_pt_allowed_pct), 0.65, 0.35)
    layers["underdog_3pt_vs_poor_perimeter"] = (
        _safe_series(work, cfg.three_pa_rate_team) >= three_rate_cut
    ) & (_safe_series(work, cfg.opponent_three_pt_allowed_pct) >= poor_perimeter_cut)

    layers["underdog_low_spr_favorite_high"] = (
        _safe_series(work, cfg.spr_team) <= 0.45
    ) & (_safe_series(work, cfg.spr_opponent) >= 0.65)

    # Situational layers
    layers["rested_underdog_fatigued_favorite"] = (
        _safe_series(work, cfg.rest_days_team) >= 2
    ) & (_safe_series(work, cfg.rest_days_opponent) <= 1)

    layers["neutral_site_underdog"] = _safe_bool(work, cfg.is_neutral_site)
    layers["revenge_spot"] = (
        (_safe_text(work, cfg.prior_opponent_name).str.lower().str.strip() == opp_name)
        & (_safe_series(work, cfg.prior_margin_team) <= -10)
    )

    elimination_flag = _safe_bool(work, cfg.elimination_game_flag)
    if elimination_flag.any():
        layers["elimination_game"] = elimination_flag
    else:
        layers["elimination_game"] = _safe_bool(work, cfg.is_tournament)

    same_conf = conf_team == conf_opp
    layers["third_meeting_underdog"] = (_safe_series(work, cfg.meeting_number) >= 3) & same_conf

    line_open = _safe_series(work, cfg.line_open)
    line_close = _safe_series(work, cfg.line_close)
    public_fav = _safe_series(work, cfg.public_pct_favorite)
    layers["reverse_line_movement"] = (line_close < line_open) & (public_fav >= 0.65)

    # CAGE confirmations
    layers["underdog_mti_up_favorite_flat"] = (
        _safe_series(work, cfg.mti_team) > 0
    ) & (_safe_series(work, cfg.mti_opponent) <= 0)
    layers["underdog_sci_battle_tested"] = _safe_series(work, cfg.sci_team) >= _quantile_or_default(_safe_series(work, cfg.sci_team), 0.60, 0.55)
    layers["underdog_odi_defensive"] = _safe_series(work, cfg.odi_team) < 0

    rows: List[Dict[str, Any]] = []
    pass_layer_cols: List[str] = []

    for layer_name, condition in layers.items():
        temp_col = _bool_name(layer_name)
        work[temp_col] = condition.fillna(False)

        dual = _analyze_layer_dual(
            work=work,
            base_scope=base_scope,
            layer_col=temp_col,
            cfg=cfg,
            min_sample=min_sample,
            p_threshold=p_threshold,
        )
        rows.append(dual)

        if (
            (pd.notna(dual["ats_lift"]) and dual["ats_lift"] > 0 and dual["ats_n"] >= min_sample)
            or (pd.notna(dual["ml_lift"]) and dual["ml_lift"] > 0 and dual["ml_n"] >= min_sample)
        ):
            pass_layer_cols.append(temp_col)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        result_df.to_csv(output_path, index=False)
        return result_df

    # Diagnostics
    if pass_layer_cols:
        red_ats = redundancy_check(
            df=work.loc[base_scope].copy(),
            layer_cols=pass_layer_cols,
            edge_col=cfg.model_edge,
            result_col=cfg.covered,
            edge_threshold=-np.inf,
        )
        red_ml = redundancy_check(
            df=work.loc[base_scope].copy(),
            layer_cols=pass_layer_cols,
            edge_col=cfg.model_edge,
            result_col=cfg.ml_won,
            edge_threshold=-np.inf,
        )
        result_df["high_collinearity_pair_count_ats"] = int(red_ats["high_collinearity"].sum()) if not red_ats.empty else 0
        result_df["high_collinearity_pair_count_ml"] = int(red_ml["high_collinearity"].sum()) if not red_ml.empty else 0

        season_ratios: Dict[str, float] = {}
        for temp_col in pass_layer_cols:
            sc = season_consistency_check(
                df=work.loc[base_scope].copy(),
                layer_col=temp_col,
                season_col=cfg.season,
                result_col=cfg.covered,
                edge_col=cfg.model_edge,
                edge_threshold=-np.inf,
            )
            if sc.empty:
                season_ratios[temp_col] = np.nan
                continue
            valid = sc["layer_n"] > 0
            denom = int(valid.sum())
            season_ratios[temp_col] = float((sc.loc[valid, "lift"] > 0).mean()) if denom > 0 else np.nan

        result_df["season_positive_ratio_ats"] = result_df["layer"].map(lambda n: season_ratios.get(_bool_name(n), np.nan))

    result_df = result_df.sort_values(["ats_lift", "ml_lift", "ats_p_value", "ml_p_value"], ascending=[False, False, True, True])
    result_df.to_csv(output_path, index=False)
    return result_df


def analyze_over_underdog_intersection(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    edge_threshold: float,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    power_conferences: Optional[Sequence[str]] = None,
    output_path: str | Path = "intersection_results.csv",
) -> pd.DataFrame:
    """Analyze intersection cases where over signal and underdog edge co-occur."""
    work = df.copy()
    power_confs = {c.lower().strip() for c in (power_conferences or ["acc", "big ten", "big 12", "sec", "big east", "pac-12"])}

    model_over = _safe_bool(work, cfg.model_projects_over)
    is_dog = _safe_bool(work, cfg.is_underdog)
    edge = _safe_series(work, cfg.model_edge)

    base = model_over & is_dog & (edge >= edge_threshold)

    pace_team = _safe_series(work, cfg.pace_team)
    pace_opp = _safe_series(work, cfg.pace_opponent)
    fast_cut = _quantile_or_default(pd.concat([pace_team, pace_opp]), 0.60, 70.0)
    combined_pace = pace_team + pace_opp

    def_rank_team = _safe_series(work, cfg.def_rating_rank_team)
    def_rank_opp = _safe_series(work, cfg.def_rating_rank_opponent)
    max_rank = int(
        np.nanmax(
            [
                np.nanmax(def_rank_team.values) if def_rank_team.notna().any() else np.nan,
                np.nanmax(def_rank_opp.values) if def_rank_opp.notna().any() else np.nan,
                364,
            ]
        )
    )
    bottom_100_cut = max(1, max_rank - 99)

    eff_gap = (_safe_series(work, cfg.adj_efficiency_margin_team) - _safe_series(work, cfg.adj_efficiency_margin_opponent)).abs()
    seed_gap = _safe_series(work, cfg.seed_diff).abs()
    round_num = _safe_series(work, cfg.tourney_round)
    if round_num.isna().all():
        round_txt = _safe_text(work, cfg.tourney_round).str.lower()
        round_num = pd.Series(np.nan, index=work.index, dtype=float)
        round_num = np.where(
            round_txt.str.contains("round of 64|first", regex=True),
            1.0,
            np.where(round_txt.str.contains("round of 32|second", regex=True), 2.0, np.nan),
        )
        round_num = pd.Series(round_num, index=work.index, dtype=float)

    combos: Dict[str, pd.Series] = {
        "dog_edge4_and_both_fast": (edge >= 4.0) & (pace_team >= fast_cut) & (pace_opp >= fast_cut),
        "dog_plus6_and_both_efg51": (_safe_series(work, cfg.line_close) >= 6.0)
        & (_safe_series(work, cfg.efg_pct_team) >= 0.51)
        & (_safe_series(work, cfg.efg_pct_opponent) >= 0.51),
        "rested_dog_and_high_spr": (_safe_series(work, cfg.rest_days_team) >= 2)
        & (_safe_series(work, cfg.rest_days_opponent) <= 1)
        & (_safe_series(work, cfg.spr_team) > 0.65)
        & (_safe_series(work, cfg.spr_opponent) > 0.65),
        "mid_major_neutral_and_mti_up": (~_safe_text(work, cfg.conference_team).str.lower().str.strip().isin(power_confs))
        & _safe_bool(work, cfg.is_neutral_site)
        & (_safe_series(work, cfg.mti_team) > 0)
        & (_safe_series(work, cfg.mti_opponent) > 0),
        "tournament_dog_both_poor_def_eff_gap_lt8": _safe_bool(work, cfg.is_tournament)
        & (def_rank_team >= bottom_100_cut)
        & (def_rank_opp >= bottom_100_cut)
        & (eff_gap < 8.0),
        "rd1_seed_mismatch_eff_gap_lt6_high_pace": (round_num == 1.0)
        & (seed_gap >= 4.0)
        & (eff_gap < 6.0)
        & (combined_pace >= _quantile_or_default(combined_pace, 0.65, 140.0)),
    }

    ats_base_rate, ats_base_n = compute_hit_rate(work.loc[base, cfg.covered])
    rows: List[Dict[str, Any]] = []

    for name, combo in combos.items():
        mask = base & combo
        ats_rate, n_ats = compute_hit_rate(work.loc[mask, cfg.covered])
        ml_rate, n_ml = compute_hit_rate(work.loc[mask, cfg.ml_won])
        over_rate, n_over = compute_hit_rate(work.loc[mask, cfg.total_covered])

        z_stat, p_value = proportion_z_test(ats_rate, n_ats, ats_base_rate, ats_base_n)
        if np.isnan(p_value):
            p_value = np.nan

        auto_merge = bool(
            (pd.notna(ats_rate) and ats_rate >= 0.59)
            or (pd.notna(ml_rate) and ml_rate >= 0.59)
            or (pd.notna(over_rate) and over_rate >= 0.59)
        )

        rows.append(
            {
                "combo": name,
                "n": int(n_ats),
                "ats_hit_rate": ats_rate,
                "ml_hit_rate": ml_rate,
                "over_hit_rate": over_rate,
                "p_value": p_value,
                "auto_merge_candidate": auto_merge,
            }
        )

    result_df = pd.DataFrame(rows).sort_values(["auto_merge_candidate", "ats_hit_rate", "ml_hit_rate", "n"], ascending=[False, False, False, False])
    result_df.to_csv(output_path, index=False)
    return result_df


def run_full_backtest(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    edge_threshold: float = 4.0,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    blowout_margin: float = 15.0,
    blowout_over_threshold: float = 20.0,
    blowout_under_threshold: float = 20.0,
    blue_blood_list: Optional[Sequence[str]] = None,
    power_conferences: Optional[Sequence[str]] = None,
    tourney_rounds: Optional[Sequence[int]] = None,
    output_dir: str | Path = ".",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Run all situational layer analyzers and write output CSV artifacts.

    Returns
    -------
    dict:
        Keys ``over``, ``underdog``, and ``intersection`` mapped to their result
        DataFrames.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    over_path = out / "over_layer_results.csv"
    dog_path = out / "underdog_layer_results.csv"
    inter_path = out / "intersection_results.csv"
    blowout_win_path = out / "blowout_win_layer_results.csv"
    underdog_ml_win_path = out / "underdog_ml_win_layer_results.csv"
    blowout_over_path = out / "blowout_over_layer_results.csv"
    blowout_under_path = out / "blowout_under_layer_results.csv"
    march_path = out / "march_madness_rd1_rd2_layer_results.csv"

    over = analyze_over_layers(
        df=df,
        cfg=cfg,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        p_threshold=p_threshold,
        output_path=over_path,
    )
    underdog = analyze_underdog_layers(
        df=df,
        cfg=cfg,
        edge_threshold=edge_threshold,
        blue_blood_list=blue_blood_list,
        power_conferences=power_conferences,
        min_sample=min_sample,
        p_threshold=p_threshold,
        output_path=dog_path,
    )
    intersection = analyze_over_underdog_intersection(
        df=df,
        cfg=cfg,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        p_threshold=p_threshold,
        power_conferences=power_conferences,
        output_path=inter_path,
    )
    blowout_win = analyze_blowout_win_layers(
        df=df,
        cfg=cfg,
        edge_threshold=edge_threshold,
        blowout_margin=blowout_margin,
        min_sample=min_sample,
        p_threshold=p_threshold,
        output_path=blowout_win_path,
        **kwargs,
    )
    underdog_ml_win = analyze_underdog_ml_win_layers(
        df=df,
        cfg=cfg,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        p_threshold=p_threshold,
        output_path=underdog_ml_win_path,
        **kwargs,
    )
    blowout_over = analyze_blowout_over_layers(
        df=df,
        cfg=cfg,
        blowout_over_threshold=blowout_over_threshold,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        p_threshold=p_threshold,
        output_path=blowout_over_path,
        **kwargs,
    )
    blowout_under = analyze_blowout_under_layers(
        df=df,
        cfg=cfg,
        blowout_under_threshold=blowout_under_threshold,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        p_threshold=p_threshold,
        output_path=blowout_under_path,
        **kwargs,
    )
    march = analyze_march_madness_rd1_rd2_layers(
        df=df,
        cfg=cfg,
        edge_threshold=edge_threshold,
        blue_blood_list=list(blue_blood_list) if blue_blood_list is not None else None,
        power_conferences=list(power_conferences) if power_conferences is not None else None,
        tourney_rounds=list(tourney_rounds) if tourney_rounds is not None else None,
        min_sample=min_sample,
        p_threshold=p_threshold,
        output_path=march_path,
        **kwargs,
    )

    # Re-write existing v1 exports with metadata header rows for consistency.
    over_base_n = int(
        (
            _safe_bool(df, cfg.model_projects_over)
            & (_compute_total_edge(df, cfg) >= edge_threshold)
            & _positive_result_mask(df[cfg.total_covered])
        ).sum()
    ) if cfg.total_covered in df.columns else 0
    dog_base_n = int(
        (
            _safe_bool(df, cfg.is_underdog)
            & (_safe_series(df, cfg.model_edge) >= edge_threshold)
            & _positive_result_mask(df[cfg.covered])
        ).sum()
    ) if cfg.covered in df.columns else 0
    inter_base_n = int(
        (
            _safe_bool(df, cfg.model_projects_over)
            & _safe_bool(df, cfg.is_underdog)
            & (_safe_series(df, cfg.model_edge) >= edge_threshold)
        ).sum()
    )
    _write_csv_with_metadata(over_path, over, edge_threshold=edge_threshold, min_sample=min_sample, base_population_n=over_base_n)
    _write_csv_with_metadata(dog_path, underdog, edge_threshold=edge_threshold, min_sample=min_sample, base_population_n=dog_base_n)
    _write_csv_with_metadata(inter_path, intersection, edge_threshold=edge_threshold, min_sample=min_sample, base_population_n=inter_base_n)

    print("\n=== Situational Layer Backtest Summary ===")
    print(f"Over layers: {len(over)} rows -> {over_path}")
    print(f"Underdog layers: {len(underdog)} rows -> {dog_path}")
    print(f"Intersection layers: {len(intersection)} rows -> {inter_path}")
    print(f"Blowout win layers: {len(blowout_win.get('results', pd.DataFrame()))} rows -> {blowout_win_path}")
    print(f"Underdog ML win layers: {len(underdog_ml_win.get('results', pd.DataFrame()))} rows -> {underdog_ml_win_path}")
    print(f"Blowout over layers: {len(blowout_over.get('results', pd.DataFrame()))} rows -> {blowout_over_path}")
    print(f"Blowout under layers: {len(blowout_under.get('results', pd.DataFrame()))} rows -> {blowout_under_path}")
    print(f"March Madness R1/R2 layers: {len(march.get('results', pd.DataFrame()))} rows -> {march_path}")

    if not over.empty:
        # Avoid terminal encoding failures on Windows cp1252 from emoji verdicts.
        top_over = over.head(5)[["label", "layered_hit_rate", "lift", "p_value"]]
        print("\nTop over layers:")
        print(top_over.to_string(index=False))

    if not underdog.empty:
        top_dog = underdog.head(5)[["layer", "ats_hit_rate", "ats_lift", "ml_hit_rate", "ml_lift"]]
        print("\nTop underdog layers:")
        print(top_dog.to_string(index=False))

    if not intersection.empty:
        print("\nIntersection summary:")
        print(intersection.to_string(index=False))

    leaderboard_rows: List[Dict[str, Any]] = []
    for _, row in over.iterrows():
        leaderboard_rows.append(
            {
                "analyzer": "over_layers",
                "layer": row.get("label"),
                "market": "total",
                "n": row.get("layered_n"),
                "hit_rate": row.get("layered_hit_rate"),
                "lift": row.get("lift"),
                "p_value": row.get("p_value"),
                "verdict": row.get("verdict"),
            }
        )

    for _, row in underdog.iterrows():
        leaderboard_rows.append(
            {
                "analyzer": "underdog_layers",
                "layer": row.get("layer"),
                "market": "ats",
                "n": row.get("ats_n"),
                "hit_rate": row.get("ats_hit_rate"),
                "lift": row.get("ats_lift"),
                "p_value": row.get("ats_p_value"),
                "verdict": row.get("ats_verdict"),
            }
        )
        leaderboard_rows.append(
            {
                "analyzer": "underdog_layers",
                "layer": row.get("layer"),
                "market": "ml",
                "n": row.get("ml_n"),
                "hit_rate": row.get("ml_hit_rate"),
                "lift": row.get("ml_lift"),
                "p_value": row.get("ml_p_value"),
                "verdict": row.get("ml_verdict"),
            }
        )

    for _, row in intersection.iterrows():
        for market_name, hit_col in [("ats", "ats_hit_rate"), ("ml", "ml_hit_rate"), ("total", "over_hit_rate")]:
            leaderboard_rows.append(
                {
                    "analyzer": "intersection",
                    "layer": row.get("combo"),
                    "market": market_name,
                    "n": row.get("n"),
                    "hit_rate": row.get(hit_col),
                    "lift": np.nan,
                    "p_value": row.get("p_value"),
                    "verdict": "AUTO_MERGE" if bool(row.get("auto_merge_candidate", False)) else "CHECK",
                }
            )

    def _append_result_rows(analyzer_name: str, result_df: pd.DataFrame, market_key: str = "market") -> None:
        for _, r in result_df.iterrows():
            leaderboard_rows.append(
                {
                    "analyzer": analyzer_name,
                    "layer": r.get("label", r.get("layer")),
                    "market": market_key,
                    "n": r.get("layered_n", r.get("n")),
                    "hit_rate": r.get("layered_hit_rate", r.get("hit_rate")),
                    "lift": r.get("lift"),
                    "p_value": r.get("p_value"),
                    "verdict": r.get("verdict"),
                }
            )

    _append_result_rows("blowout_win_layers", blowout_win.get("results", pd.DataFrame()), "blowout_win")
    _append_result_rows("underdog_ml_win_layers", underdog_ml_win.get("results", pd.DataFrame()), "ml")
    _append_result_rows("blowout_over_layers", blowout_over.get("results", pd.DataFrame()), "blowout_over")
    _append_result_rows("blowout_under_layers", blowout_under.get("results", pd.DataFrame()), "blowout_under")

    march_df = march.get("results", pd.DataFrame())
    if not march_df.empty:
        for _, r in march_df.iterrows():
            for mk, n_col, hit_col, lift_col, p_col, verdict_col in [
                ("ats", "ats_n", "ats_hit_rate", "ats_lift", "ats_p_value", "ats_verdict"),
                ("ml", "ml_n", "ml_hit_rate", "ml_lift", "ml_p_value", "ml_verdict"),
                ("total", "total_n", "total_hit_rate", "total_lift", "total_p_value", "total_verdict"),
            ]:
                leaderboard_rows.append(
                    {
                        "analyzer": "march_madness_rd1_rd2_layers",
                        "layer": r.get("layer"),
                        "market": mk,
                        "n": r.get(n_col),
                        "hit_rate": r.get(hit_col),
                        "lift": r.get(lift_col),
                        "p_value": r.get(p_col),
                        "verdict": r.get(verdict_col),
                    }
                )

    leaderboard = pd.DataFrame(leaderboard_rows)
    if not leaderboard.empty:
        sig = pd.to_numeric(leaderboard["p_value"], errors="coerce") <= p_threshold
        leaderboard["_sig"] = sig.fillna(False)
        leaderboard["_lift"] = pd.to_numeric(leaderboard["lift"], errors="coerce")
        leaderboard = leaderboard.sort_values(["_sig", "_lift"], ascending=[False, False]).drop(columns=["_sig", "_lift"])
        display_leaderboard = leaderboard.copy()
        if "verdict" in display_leaderboard.columns:
            display_leaderboard["verdict"] = (
                display_leaderboard["verdict"]
                .astype(str)
                .map(lambda x: x.encode("ascii", "ignore").decode("ascii"))
            )
        print("\nCross-analyzer leaderboard:")
        print(display_leaderboard.head(40).to_string(index=False))

    return {
        "over": over,
        "underdog": underdog,
        "intersection": intersection,
        "blowout_win": blowout_win,
        "underdog_ml_win": underdog_ml_win,
        "blowout_over": blowout_over,
        "blowout_under": blowout_under,
        "march_madness_rd1_rd2": march,
        "leaderboard": leaderboard if "leaderboard" in locals() else pd.DataFrame(),
    }


def _synthetic_example_df(cfg: ColumnConfig, n: int = 5000, seed: int = 7) -> pd.DataFrame:
    """Build a synthetic demo dataframe for local smoke tests."""
    rng = np.random.default_rng(seed)

    seasons = rng.choice([2022, 2023, 2024, 2025], size=n, p=[0.2, 0.25, 0.3, 0.25])
    team_confs = rng.choice(["acc", "sec", "big ten", "big 12", "mvc", "a10", "wcc"], size=n)
    opp_confs = rng.choice(["acc", "sec", "big ten", "big 12", "mvc", "a10", "wcc"], size=n)

    model_edge = rng.normal(loc=2.0, scale=5.5, size=n)
    is_dog = rng.random(n) < 0.47

    pace_team = rng.normal(69.8, 3.8, size=n)
    pace_opp = rng.normal(69.2, 4.0, size=n)
    efg_team = np.clip(rng.normal(0.515, 0.04, size=n), 0.40, 0.65)
    efg_opp = np.clip(rng.normal(0.510, 0.04, size=n), 0.40, 0.65)
    spr_team = np.clip(rng.normal(0.55, 0.13, size=n), 0.1, 0.95)
    spr_opp = np.clip(rng.normal(0.54, 0.13, size=n), 0.1, 0.95)

    total_edge = rng.normal(1.5, 10.0, size=n)
    model_over = total_edge > 0

    # Inject mild structured signal so demo has meaningful outputs
    ats_p = 0.5 + 0.02 * (model_edge >= 4) + 0.02 * ((spr_team < 0.45) & (spr_opp > 0.65))
    ml_p = 0.45 + 0.04 * (model_edge >= 4) + 0.03 * ((pace_team > 71) & (pace_opp > 71))
    over_p = 0.5 + 0.03 * model_over + 0.02 * ((pace_team > 70) & (pace_opp > 70))

    ats = (rng.random(n) < np.clip(ats_p, 0.02, 0.98)).astype(int)
    ml = (rng.random(n) < np.clip(ml_p, 0.02, 0.98)).astype(int)
    over = (rng.random(n) < np.clip(over_p, 0.02, 0.98)).astype(int)

    line_open = rng.normal(4.5, 5.0, size=n)
    line_close = line_open + rng.normal(0.0, 1.1, size=n)

    df = pd.DataFrame(
        {
            cfg.model_edge: model_edge,
            cfg.covered: ats,
            cfg.total_covered: over,
            cfg.ml_won: ml,
            cfg.season: seasons,
            cfg.is_underdog: is_dog.astype(int),
            cfg.model_projects_over: model_over.astype(int),
            cfg.opponent_name: rng.choice(["Duke", "Kansas", "UCLA", "VCU", "Gonzaga", "UNC"], size=n),
            cfg.rest_days_team: rng.integers(0, 4, size=n),
            cfg.rest_days_opponent: rng.integers(0, 4, size=n),
            cfg.pace_team: pace_team,
            cfg.pace_opponent: pace_opp,
            cfg.efg_pct_team: efg_team,
            cfg.efg_pct_opponent: efg_opp,
            cfg.efg_allowed_team: np.clip(rng.normal(0.50, 0.03, size=n), 0.42, 0.60),
            cfg.efg_allowed_opponent: np.clip(rng.normal(0.505, 0.03, size=n), 0.42, 0.60),
            cfg.oreb_pct_team: np.clip(rng.normal(0.31, 0.04, size=n), 0.2, 0.45),
            cfg.dreb_pct_opp: np.clip(rng.normal(0.71, 0.04, size=n), 0.55, 0.85),
            cfg.three_pa_rate_team: np.clip(rng.normal(0.37, 0.07, size=n), 0.15, 0.60),
            cfg.three_pa_rate_opponent: np.clip(rng.normal(0.35, 0.07, size=n), 0.15, 0.60),
            cfg.fta_rate_team: np.clip(rng.normal(0.29, 0.08, size=n), 0.10, 0.60),
            cfg.fta_rate_opponent: np.clip(rng.normal(0.285, 0.08, size=n), 0.10, 0.60),
            cfg.def_rating_team: rng.normal(101, 7, size=n),
            cfg.def_rating_opponent: rng.normal(102, 7, size=n),
            cfg.mti_team: rng.normal(0.05, 0.20, size=n),
            cfg.mti_opponent: rng.normal(0.02, 0.20, size=n),
            cfg.spr_team: spr_team,
            cfg.spr_opponent: spr_opp,
            cfg.odi_team: rng.normal(0.02, 0.30, size=n),
            cfg.odi_opponent: rng.normal(0.01, 0.30, size=n),
            cfg.pmi: np.clip(rng.normal(0.5, 0.2, size=n), 0.0, 1.0),
            cfg.dpc_team: np.clip(rng.normal(0.45, 0.15, size=n), 0.0, 1.0),
            cfg.dpc_opponent: np.clip(rng.normal(0.46, 0.15, size=n), 0.0, 1.0),
            cfg.pei_gap: rng.normal(0, 5, size=n),
            cfg.sci_team: np.clip(rng.normal(0.55, 0.2, size=n), 0.0, 1.0),
            cfg.sci_opponent: np.clip(rng.normal(0.53, 0.2, size=n), 0.0, 1.0),
            cfg.conference_team: team_confs,
            cfg.conference_opponent: opp_confs,
            cfg.is_neutral_site: (rng.random(n) < 0.25).astype(int),
            cfg.is_tournament: (rng.random(n) < 0.22).astype(int),
            cfg.tourney_round: rng.choice(["first round", "round of 32", "semifinal", "final"], size=n),
            cfg.prior_margin_team: rng.normal(0, 13, size=n),
            cfg.prior_opponent_name: rng.choice(["duke", "kansas", "ucla", "vcu", "unc"], size=n),
            cfg.meeting_number: rng.integers(1, 4, size=n),
            cfg.line_open: line_open,
            cfg.line_close: line_close,
            cfg.public_pct_favorite: np.clip(rng.normal(0.63, 0.12, size=n), 0.2, 0.95),
            cfg.to_pct_team: np.clip(rng.normal(0.17, 0.03, size=n), 0.08, 0.30),
            cfg.to_pct_opponent: np.clip(rng.normal(0.17, 0.03, size=n), 0.08, 0.30),
            cfg.pace_allowed_rank_team: rng.integers(1, 364, size=n),
            cfg.pace_allowed_rank_opponent: rng.integers(1, 364, size=n),
            cfg.opp_two_pt_allowed_rank_team: rng.integers(1, 364, size=n),
            cfg.opp_two_pt_allowed_rank_opponent: rng.integers(1, 364, size=n),
            cfg.previous_ot_team: (rng.random(n) < 0.12).astype(int),
            cfg.previous_ot_opponent: (rng.random(n) < 0.11).astype(int),
            cfg.is_conference_tournament: (rng.random(n) < 0.16).astype(int),
            cfg.season_low_total_team: rng.normal(126, 10, size=n),
            cfg.season_low_total_opponent: rng.normal(127, 10, size=n),
            cfg.projected_total: 137 + total_edge,
            cfg.market_total_close: 137 + rng.normal(0, 8, size=n),
            cfg.total_edge: total_edge,
            cfg.opponent_rank: rng.integers(1, 200, size=n),
            cfg.opponent_win_streak: rng.integers(0, 9, size=n),
            cfg.top25_loss_share_team: np.clip(rng.normal(0.4, 0.25, size=n), 0, 1),
            cfg.favorite_soft_schedule_index: np.clip(rng.normal(0.5, 0.2, size=n), 0, 1),
            cfg.sos_adj_eff_gap: rng.normal(0, 6, size=n),
            cfg.favorite_fastbreak_points_allowed: np.clip(rng.normal(11, 4, size=n), 2, 25),
            cfg.opponent_three_pt_allowed_pct: np.clip(rng.normal(0.345, 0.03, size=n), 0.28, 0.45),
            cfg.elimination_game_flag: (rng.random(n) < 0.15).astype(int),
            cfg.final_margin: rng.normal(0.0, 14.0, size=n),
            cfg.ml_implied_prob: np.clip(rng.normal(0.50, 0.15, size=n), 0.02, 0.98),
            cfg.model_win_prob: np.clip(rng.normal(0.52, 0.16, size=n), 0.02, 0.98),
            cfg.model_projects_under: (~model_over).astype(int),
            cfg.pace_rank_team: rng.integers(1, 364, size=n),
            cfg.pace_rank_opponent: rng.integers(1, 364, size=n),
            cfg.three_pt_def_rank_team: rng.integers(1, 364, size=n),
            cfg.three_pt_def_rank_opponent: rng.integers(1, 364, size=n),
            cfg.def_rating_rank_team: rng.integers(1, 364, size=n),
            cfg.def_rating_rank_opponent: rng.integers(1, 364, size=n),
            cfg.adj_efficiency_margin_team: rng.normal(12, 8, size=n),
            cfg.adj_efficiency_margin_opponent: rng.normal(11, 8, size=n),
            cfg.fast_break_pts_team: np.clip(rng.normal(12.5, 3.5, size=n), 2, 25),
            cfg.fast_break_pts_allowed_opponent: np.clip(rng.normal(12.0, 3.5, size=n), 2, 25),
            cfg.tov_pct_team: np.clip(rng.normal(0.17, 0.03, size=n), 0.08, 0.30),
            cfg.tov_pct_opponent: np.clip(rng.normal(0.17, 0.03, size=n), 0.08, 0.30),
            cfg.steal_rate_team: np.clip(rng.normal(0.09, 0.02, size=n), 0.02, 0.20),
            cfg.steal_rate_opponent: np.clip(rng.normal(0.09, 0.02, size=n), 0.02, 0.20),
            cfg.block_rate_team: np.clip(rng.normal(0.08, 0.02, size=n), 0.01, 0.18),
            cfg.block_rate_opponent: np.clip(rng.normal(0.08, 0.02, size=n), 0.01, 0.18),
            cfg.bench_minutes_pct_team: np.clip(rng.normal(0.31, 0.08, size=n), 0.05, 0.55),
            cfg.bench_minutes_pct_opponent: np.clip(rng.normal(0.30, 0.08, size=n), 0.05, 0.55),
            cfg.second_half_margin_team: rng.normal(0, 7, size=n),
            cfg.second_half_def_eff_opponent: np.clip(rng.normal(1.02, 0.12, size=n), 0.70, 1.40),
            cfg.ane_team: rng.normal(13, 9, size=n),
            cfg.ane_opponent: rng.normal(12, 9, size=n),
            cfg.is_power_conf_team: pd.Series(team_confs).isin(["acc", "sec", "big ten", "big 12", "big east", "pac-12"]).astype(int),
            cfg.is_power_conf_opponent: pd.Series(opp_confs).isin(["acc", "sec", "big ten", "big 12", "big east", "pac-12"]).astype(int),
            cfg.prev_game_ot_team: (rng.random(n) < 0.12).astype(int),
            cfg.prev_game_ot_opponent: (rng.random(n) < 0.11).astype(int),
            cfg.consecutive_road_games_opponent: rng.integers(0, 6, size=n),
            cfg.days_since_last_game_team: rng.integers(0, 5, size=n),
            cfg.days_since_last_game_opponent: rng.integers(0, 5, size=n),
            cfg.win_streak_opponent: rng.integers(0, 9, size=n),
            cfg.sos_rank_team: rng.integers(1, 364, size=n),
            cfg.sos_rank_opponent: rng.integers(1, 364, size=n),
            cfg.losses_vs_top25_pct_team: np.clip(rng.normal(0.35, 0.20, size=n), 0.0, 1.0),
            cfg.wins_vs_bottom_half_pct_opponent: np.clip(rng.normal(0.55, 0.20, size=n), 0.0, 1.0),
            cfg.ft_pct_team: np.clip(rng.normal(0.72, 0.08, size=n), 0.45, 0.95),
            cfg.ft_pct_opponent: np.clip(rng.normal(0.71, 0.08, size=n), 0.45, 0.95),
            cfg.ppp_team: np.clip(rng.normal(1.05, 0.12, size=n), 0.70, 1.45),
            cfg.ppp_opponent: np.clip(rng.normal(1.04, 0.12, size=n), 0.70, 1.45),
            cfg.half_court_rate_team: np.clip(rng.normal(0.50, 0.10, size=n), 0.20, 0.80),
            cfg.half_court_rate_opponent: np.clip(rng.normal(0.50, 0.10, size=n), 0.20, 0.80),
            cfg.transition_rate_team: np.clip(rng.normal(0.19, 0.06, size=n), 0.05, 0.40),
            cfg.transition_rate_opponent: np.clip(rng.normal(0.19, 0.06, size=n), 0.05, 0.40),
            cfg.interior_def_rank_team: rng.integers(1, 364, size=n),
            cfg.interior_def_rank_opponent: rng.integers(1, 364, size=n),
            cfg.tourney_experience_score_team: np.clip(rng.normal(0.50, 0.20, size=n), 0.0, 1.0),
            cfg.tourney_experience_score_opponent: np.clip(rng.normal(0.50, 0.20, size=n), 0.0, 1.0),
            cfg.conf_strength_rank_team: rng.integers(1, 33, size=n),
            cfg.conf_strength_rank_opponent: rng.integers(1, 33, size=n),
            cfg.coach_upset_rate_team: np.clip(rng.normal(0.15, 0.08, size=n), 0.0, 0.60),
            cfg.coach_upset_rate_opponent: np.clip(rng.normal(0.15, 0.08, size=n), 0.0, 0.60),
            cfg.seed_team: rng.integers(1, 17, size=n),
            cfg.seed_opponent: rng.integers(1, 17, size=n),
            cfg.seed_diff: rng.integers(-15, 16, size=n),
            cfg.region_team: rng.choice(["east", "west", "south", "midwest"], size=n),
            cfg.region_opponent: rng.choice(["east", "west", "south", "midwest"], size=n),
        }
    )

    return df


def _write_csv_with_metadata(
    output_path: str | Path,
    frame: pd.DataFrame,
    *,
    edge_threshold: float,
    min_sample: int,
    base_population_n: int,
) -> None:
    """Write analyzer output CSV with metadata header rows."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.now(tz="UTC").isoformat()
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# timestamp_utc: {stamp}\n")
        handle.write(f"# edge_threshold: {edge_threshold}\n")
        handle.write(f"# min_sample: {min_sample}\n")
        handle.write(f"# base_population_n: {int(base_population_n)}\n")
        frame.to_csv(handle, index=False)


def _rank_top_mask(rank_series: pd.Series, top_k: int = 50) -> pd.Series:
    """Mask top-K rank rows (lower rank number indicates better)."""
    return pd.to_numeric(rank_series, errors="coerce") <= float(top_k)


def _rank_bottom_mask(rank_series: pd.Series, bottom_k: int = 50) -> pd.Series:
    """Mask bottom-K rank rows (higher rank number indicates worse)."""
    rank = pd.to_numeric(rank_series, errors="coerce")
    max_rank = int(
        np.nanmax(
            [
                np.nanmax(rank.values) if rank.notna().any() else np.nan,
                364,
            ]
        )
    )
    cutoff = max(1, max_rank - int(bottom_k) + 1)
    return rank >= float(cutoff)


def _round_number_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Parse tournament round values as numeric with text fallback."""
    numeric = _safe_series(df, col)
    if numeric.notna().any():
        return numeric
    txt = _safe_text(df, col).str.lower()
    parsed = np.where(
        txt.str.contains("round of 64|first", regex=True),
        1.0,
        np.where(
            txt.str.contains("round of 32|second", regex=True),
            2.0,
            np.where(txt.str.contains("sweet 16|third", regex=True), 3.0, np.nan),
        ),
    )
    return pd.Series(parsed, index=df.index, dtype=float)


def _evaluate_named_layers(
    *,
    frame: pd.DataFrame,
    base_scope: pd.Series,
    layers: Dict[str, pd.Series],
    result_col: str,
    cfg: ColumnConfig,
    min_sample: int,
    p_threshold: float,
    lift_threshold: float = 0.03,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a dictionary of named boolean layer masks with shared controls."""
    work = frame.copy()
    rows: List[Dict[str, Any]] = []
    for layer_name, mask in layers.items():
        temp_col = _bool_name(layer_name)
        work[temp_col] = mask.fillna(False).astype(bool)
        layer_result = analyze_layer(
            df=work.loc[base_scope].copy(),
            layer_col=temp_col,
            result_col=result_col,
            edge_col=cfg.model_edge,
            edge_threshold=-np.inf,
            label=layer_name,
            min_sample=min_sample,
            p_threshold=p_threshold,
            lift_threshold=lift_threshold,
        )
        layer_result["layer"] = layer_name
        rows.append(layer_result)

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df, pd.DataFrame(), pd.DataFrame()

    pass_layers = result_df.loc[(result_df["lift"] > 0) & (result_df["layered_n"] >= min_sample), "layer"].tolist()
    pass_temp = [_bool_name(x) for x in pass_layers]

    red = redundancy_check(
        df=work.loc[base_scope].copy(),
        layer_cols=pass_temp,
        edge_col=cfg.model_edge,
        result_col=result_col,
        edge_threshold=-np.inf,
    ) if pass_temp else pd.DataFrame()

    season_frames: List[pd.DataFrame] = []
    for layer_name in pass_layers:
        temp_col = _bool_name(layer_name)
        season_df = season_consistency_check(
            df=work.loc[base_scope].copy(),
            layer_col=temp_col,
            season_col=cfg.season,
            result_col=result_col,
            edge_col=cfg.model_edge,
            edge_threshold=-np.inf,
        )
        if not season_df.empty:
            season_df["layer"] = layer_name
            season_frames.append(season_df)
            valid = season_df["layer_n"] > 0
            denom = int(valid.sum())
            ratio = float((season_df.loc[valid, "lift"] > 0).mean()) if denom > 0 else np.nan
            result_df.loc[result_df["layer"] == layer_name, "season_positive_ratio"] = ratio

    result_df["high_collinearity_pair_count"] = int(red["high_collinearity"].sum()) if not red.empty else 0
    result_df = result_df.sort_values(["significant", "lift", "p_value", "layered_n"], ascending=[False, False, True, False]).reset_index(drop=True)
    season_out = pd.concat(season_frames, ignore_index=True, sort=False) if season_frames else pd.DataFrame()
    return result_df, red, season_out

def analyze_blowout_win_layers(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    edge_threshold: float = 4.0,
    blowout_margin: float = 15.0,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    output_path: str | Path = "blowout_win_layer_results.csv",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Analyze situational layers for model-side blowout wins."""
    _ = kwargs
    work = df.copy()
    edge = _safe_series(work, cfg.model_edge)
    final_margin = _safe_series(work, cfg.final_margin)
    work["_result_blowout_win"] = ((final_margin >= float(blowout_margin)) & final_margin.notna()).astype(int)
    base_scope = (edge >= float(edge_threshold)) & final_margin.notna()

    ane_gap = (_safe_series(work, cfg.ane_team) - _safe_series(work, cfg.ane_opponent)).abs()
    pei_abs = _safe_series(work, cfg.pei_gap).abs()
    bench_team = _safe_series(work, cfg.bench_minutes_pct_team)
    bench_opp = _safe_series(work, cfg.bench_minutes_pct_opponent)
    oreb_team = _safe_series(work, cfg.oreb_pct_team)
    dreb_opp = _safe_series(work, cfg.dreb_pct_opp)
    pace_team = _safe_series(work, cfg.pace_team)
    fta_team = _safe_series(work, cfg.fta_rate_team)
    fta_opp = _safe_series(work, cfg.fta_rate_opponent)
    fb_team = _safe_series(work, cfg.fast_break_pts_team)
    fb_allow_opp = _safe_series(work, cfg.fast_break_pts_allowed_opponent)
    tov_opp = _safe_series(work, cfg.tov_pct_opponent)
    steal_team = _safe_series(work, cfg.steal_rate_team)
    sec_half_def_opp = _safe_series(work, cfg.second_half_def_eff_opponent)
    rest_team = _safe_series(work, cfg.days_since_last_game_team).where(
        _safe_series(work, cfg.days_since_last_game_team).notna(),
        _safe_series(work, cfg.rest_days_team),
    )
    rest_opp = _safe_series(work, cfg.days_since_last_game_opponent).where(
        _safe_series(work, cfg.days_since_last_game_opponent).notna(),
        _safe_series(work, cfg.rest_days_opponent),
    )
    sci_team = _safe_series(work, cfg.sci_team)
    sci_opp = _safe_series(work, cfg.sci_opponent)
    odi_team = _safe_series(work, cfg.odi_team)
    odi_opp = _safe_series(work, cfg.odi_opponent)
    mti_team = _safe_series(work, cfg.mti_team)
    mti_opp = _safe_series(work, cfg.mti_opponent)
    road_games_opp = _safe_series(work, cfg.consecutive_road_games_opponent)

    q_bench_hi = _quantile_or_default(bench_team, 0.75, 0.34)
    q_bench_lo = _quantile_or_default(bench_opp, 0.25, 0.24)
    q_oreb_hi = _quantile_or_default(oreb_team, 0.75, 0.32)
    q_dreb_lo = _quantile_or_default(dreb_opp, 0.25, 0.68)
    q_fb_hi = _quantile_or_default(fb_team, 0.75, 14.0)
    q_fb_allow_lo = _quantile_or_default(fb_allow_opp, 0.75, 13.0)
    q_steal_hi = _quantile_or_default(steal_team, 0.75, 0.09)
    q_sec_def_bad = _quantile_or_default(sec_half_def_opp, 0.75, 1.08)
    q_pei_ext = _quantile_or_default(pei_abs, 0.90, 8.0)

    layers: Dict[str, pd.Series] = {
        "large_ane_gap": ane_gap >= 12.0,
        "both_top50_off_opponent_bottom50_def": _rank_top_mask(_safe_series(work, cfg.pace_rank_team), 50) & _rank_bottom_mask(_safe_series(work, cfg.def_rating_rank_opponent), 50),
        "pei_gap_extreme": pei_abs >= q_pei_ext,
        "opponent_short_rotation": bench_opp <= q_bench_lo,
        "depth_advantage": (bench_team >= q_bench_hi) & (bench_opp <= q_bench_lo),
        "team_fast_opponent_transition_weak": (pace_team >= _quantile_or_default(pace_team, 0.75, 71.0)) & (fb_allow_opp >= q_fb_allow_lo),
        "team_elite_ft_drawing_opponent_foul_prone": (fta_team >= _quantile_or_default(fta_team, 0.75, 0.33)) & (fta_opp >= _quantile_or_default(fta_opp, 0.75, 0.33)),
        "team_high_oreb_opponent_poor_dreb": (oreb_team >= q_oreb_hi) & (dreb_opp <= q_dreb_lo),
        "team_top25_fastbreak_opponent_bottom25_allowed": (fb_team >= q_fb_hi) & (fb_allow_opp >= q_fb_allow_lo),
        "opponent_high_tov_team_forces_turnovers": (tov_opp >= 0.20) & (steal_team >= q_steal_hi),
        "opponent_poor_ft_team_draws_fouls": (_safe_series(work, cfg.ft_pct_opponent) <= 0.65) & (fta_team >= _quantile_or_default(fta_team, 0.75, 0.33)),
        "opponent_short_rotation_fast_tempo": (bench_opp <= q_bench_lo) & (pace_team >= _quantile_or_default(pace_team, 0.75, 71.0)),
        "opponent_bottom25_second_half_def": sec_half_def_opp >= q_sec_def_bad,
        "rested_team_opponent_game3_in3days": (rest_team >= 2.0) & (rest_opp <= 1.0) & (road_games_opp >= 3.0),
        "opponent_prev_game_ot": _safe_bool(work, cfg.prev_game_ot_opponent) | _safe_bool(work, cfg.previous_ot_opponent),
        "no_tournament_implications_opponent": (~_safe_bool(work, cfg.is_tournament)) & (_safe_series(work, cfg.sos_rank_opponent) >= _quantile_or_default(_safe_series(work, cfg.sos_rank_opponent), 0.70, 240.0)),
        "opponent_pseudo_road_neutral": _safe_bool(work, cfg.is_neutral_site) & (road_games_opp >= 2.0),
        "mti_crossover": (mti_team > 0) & (mti_opp < 0),
        "sci_battle_tested_vs_untested": (sci_team >= _quantile_or_default(sci_team, 0.75, 0.65)) & (sci_opp <= _quantile_or_default(sci_opp, 0.25, 0.40)),
        "odi_extreme_mismatch": (odi_team >= _quantile_or_default(odi_team, 0.75, 0.20)) & (odi_opp <= _quantile_or_default(odi_opp, 0.25, -0.20)),
        "ane_gap_plus_rest": (ane_gap >= 12.0) & (rest_team - rest_opp >= 1.0),
    }

    results, redundancy, season_consistency = _evaluate_named_layers(
        frame=work,
        base_scope=base_scope,
        layers=layers,
        result_col="_result_blowout_win",
        cfg=cfg,
        min_sample=min_sample,
        p_threshold=p_threshold,
        lift_threshold=0.03,
    )

    _write_csv_with_metadata(
        output_path,
        results,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        base_population_n=int(base_scope.sum()),
    )
    return {
        "results": results,
        "redundancy": redundancy,
        "season_consistency": season_consistency,
        "base_population_n": int(base_scope.sum()),
        "output_path": str(output_path),
    }


def analyze_underdog_ml_win_layers(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    edge_threshold: float = 4.0,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    output_path: str | Path = "underdog_ml_win_layer_results.csv",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Analyze underdog moneyline layers using model-vs-market win-probability edge."""
    _ = kwargs
    work = df.copy()
    is_dog = _safe_bool(work, cfg.is_underdog)
    model_prob = _safe_series(work, cfg.model_win_prob).where(
        _safe_series(work, cfg.model_win_prob).notna(),
        (_safe_series(work, cfg.model_edge) / 12.0).pipe(lambda s: 1.0 / (1.0 + np.exp(-s))),
    )
    market_prob = _safe_series(work, cfg.ml_implied_prob)
    base_scope = is_dog & (model_prob > (market_prob + 0.05)) & _positive_result_mask(work[cfg.ml_won])

    adj_gap = (_safe_series(work, cfg.adj_efficiency_margin_team) - _safe_series(work, cfg.adj_efficiency_margin_opponent)).abs()
    pei_abs = _safe_series(work, cfg.pei_gap).abs()
    three_rate = _safe_series(work, cfg.three_pa_rate_team)
    opp_perim_rank = _safe_series(work, cfg.three_pt_def_rank_opponent)
    oreb_team = _safe_series(work, cfg.oreb_pct_team)
    dreb_opp = _safe_series(work, cfg.dreb_pct_opp)
    rest_team = _safe_series(work, cfg.days_since_last_game_team).where(
        _safe_series(work, cfg.days_since_last_game_team).notna(),
        _safe_series(work, cfg.rest_days_team),
    )
    rest_opp = _safe_series(work, cfg.days_since_last_game_opponent).where(
        _safe_series(work, cfg.days_since_last_game_opponent).notna(),
        _safe_series(work, cfg.rest_days_opponent),
    )

    layers: Dict[str, pd.Series] = {
        "efficiency_gap_under_8": adj_gap < 8.0,
        "both_within_8_adj_efficiency": adj_gap <= 8.0,
        "underdog_top50_adj_eff_despite_odds": _rank_top_mask(_safe_series(work, cfg.pace_rank_team), 50) & (market_prob < 0.35),
        "pei_gap_under_5": pei_abs <= 5.0,
        "underdog_elite_def_vs_favorite_halfcourt_struggles": _rank_top_mask(_safe_series(work, cfg.def_rating_rank_team), 25) & (_safe_series(work, cfg.half_court_rate_opponent) >= _quantile_or_default(_safe_series(work, cfg.half_court_rate_opponent), 0.70, 0.55)),
        "underdog_slow_tempo_elite_def": _rank_bottom_mask(_safe_series(work, cfg.pace_rank_team), 80) & _rank_top_mask(_safe_series(work, cfg.def_rating_rank_team), 25),
        "underdog_3pt_heavy_vs_poor_perimeter": (three_rate >= _quantile_or_default(three_rate, 0.75, 0.38)) & _rank_bottom_mask(opp_perim_rank, 100),
        "underdog_superior_ft_drawing_vs_foul_prone": (_safe_series(work, cfg.fta_rate_team) >= _quantile_or_default(_safe_series(work, cfg.fta_rate_team), 0.75, 0.33)) & (_safe_series(work, cfg.fta_rate_opponent) >= _quantile_or_default(_safe_series(work, cfg.fta_rate_opponent), 0.75, 0.33)),
        "underdog_top25_oreb_vs_poor_dreb": (oreb_team >= _quantile_or_default(oreb_team, 0.75, 0.32)) & (dreb_opp <= _quantile_or_default(dreb_opp, 0.25, 0.68)),
        "efficiency_gap_under_10_elimination": (adj_gap < 10.0) & _safe_bool(work, cfg.is_tournament),
        "underdog_tourney_experience_advantage": _safe_series(work, cfg.tourney_experience_score_team) > _safe_series(work, cfg.tourney_experience_score_opponent),
        "underdog_physical_conf_vs_finesse_conf": (_safe_series(work, cfg.conf_strength_rank_team) + 8.0) < _safe_series(work, cfg.conf_strength_rank_opponent),
        "underdog_elite_perimeter_scorer_vs_poor_perimeter_def": (three_rate >= _quantile_or_default(three_rate, 0.75, 0.38)) & _rank_bottom_mask(opp_perim_rank, 100),
        "revenge_game_large_prior_loss": (_safe_series(work, cfg.prior_margin_team) <= -15.0) & (_safe_text(work, cfg.prior_opponent_name).str.lower().str.strip() == _safe_text(work, cfg.opponent_name).str.lower().str.strip()),
        "letdown_spot_favorite": (_safe_series(work, cfg.win_streak_opponent) >= 5.0) & (_safe_series(work, cfg.opponent_rank) <= 10.0),
        "rested_underdog_fatigued_favorite": (rest_team >= 2.0) & (rest_opp <= 1.0),
        "neutral_site_underdog": _safe_bool(work, cfg.is_neutral_site),
        "public_80pct_plus_on_favorite": _safe_series(work, cfg.public_pct_favorite) >= 0.80,
        "reverse_line_movement_underdog": (_safe_series(work, cfg.line_close) < _safe_series(work, cfg.line_open)) & (_safe_series(work, cfg.public_pct_favorite) >= 0.65),
        "mti_momentum_crossover": (_safe_series(work, cfg.mti_team) > 0) & (_safe_series(work, cfg.mti_opponent) <= 0),
        "underdog_odi_defensive_favorite_odi_offensive": (_safe_series(work, cfg.odi_team) < 0) & (_safe_series(work, cfg.odi_opponent) > 0),
        "underdog_sci_highest_quartile": _safe_series(work, cfg.sci_team) >= _quantile_or_default(_safe_series(work, cfg.sci_team), 0.75, 0.65),
        "low_spr_underdog_high_spr_favorite": (_safe_series(work, cfg.spr_team) < 0.45) & (_safe_series(work, cfg.spr_opponent) > 0.65),
    }

    results, redundancy, season_consistency = _evaluate_named_layers(
        frame=work,
        base_scope=base_scope,
        layers=layers,
        result_col=cfg.ml_won,
        cfg=cfg,
        min_sample=min_sample,
        p_threshold=p_threshold,
        lift_threshold=0.03,
    )
    _write_csv_with_metadata(
        output_path,
        results,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        base_population_n=int(base_scope.sum()),
    )
    return {
        "results": results,
        "redundancy": redundancy,
        "season_consistency": season_consistency,
        "base_population_n": int(base_scope.sum()),
        "output_path": str(output_path),
    }


def analyze_blowout_over_layers(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    blowout_over_threshold: float = 20.0,
    edge_threshold: float = 4.0,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    output_path: str | Path = "blowout_over_layer_results.csv",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Analyze layers where model-over calls win by large margin versus total line."""
    _ = kwargs
    work = df.copy()
    model_over = _safe_bool(work, cfg.model_projects_over)
    total_margin = _safe_series(work, cfg.final_margin)
    work["_result_blowout_over"] = ((total_margin >= float(blowout_over_threshold)) & total_margin.notna()).astype(int)
    base_scope = model_over & total_margin.notna()

    pace_rank_team = _safe_series(work, cfg.pace_rank_team)
    pace_rank_opp = _safe_series(work, cfg.pace_rank_opponent)
    def_rank_team = _safe_series(work, cfg.def_rating_rank_team)
    def_rank_opp = _safe_series(work, cfg.def_rating_rank_opponent)
    efg_team = _safe_series(work, cfg.efg_pct_team)
    efg_opp = _safe_series(work, cfg.efg_pct_opponent)
    efg_allow_team = _safe_series(work, cfg.efg_allowed_team)
    efg_allow_opp = _safe_series(work, cfg.efg_allowed_opponent)
    ppp_team = _safe_series(work, cfg.ppp_team)
    ppp_opp = _safe_series(work, cfg.ppp_opponent)
    pace_team = _safe_series(work, cfg.pace_team)
    pace_opp = _safe_series(work, cfg.pace_opponent)
    market_total = _safe_series(work, cfg.market_total_close)
    implied_poss = market_total / (ppp_team + ppp_opp).replace(0.0, np.nan)
    proj_poss = (pace_team + pace_opp) / 2.0

    layers: Dict[str, pd.Series] = {
        "both_teams_top50_tempo": _rank_top_mask(pace_rank_team, 50) & _rank_top_mask(pace_rank_opp, 50),
        "neither_team_defensive_identity": _rank_bottom_mask(def_rank_team, 100) & _rank_bottom_mask(def_rank_opp, 100),
        "both_teams_top25_fastbreak": (_safe_series(work, cfg.fast_break_pts_team) >= _quantile_or_default(_safe_series(work, cfg.fast_break_pts_team), 0.75, 14.0)) & (_safe_series(work, cfg.fast_break_pts_allowed_opponent) >= _quantile_or_default(_safe_series(work, cfg.fast_break_pts_allowed_opponent), 0.75, 13.0)),
        "both_low_halfcourt_rate": (_safe_series(work, cfg.half_court_rate_team) <= _quantile_or_default(_safe_series(work, cfg.half_court_rate_team), 0.25, 0.45)) & (_safe_series(work, cfg.half_court_rate_opponent) <= _quantile_or_default(_safe_series(work, cfg.half_court_rate_opponent), 0.25, 0.45)),
        "combined_possessions_exceed_implied": (proj_poss - implied_poss) >= 8.0,
        "both_def_rating_bottom50": _rank_bottom_mask(def_rank_team, 50) & _rank_bottom_mask(def_rank_opp, 50),
        "both_efg_allowed_above_54": (efg_allow_team >= 0.54) & (efg_allow_opp >= 0.54),
        "both_poor_perimeter_defense": _rank_bottom_mask(_safe_series(work, cfg.three_pt_def_rank_team), 100) & _rank_bottom_mask(_safe_series(work, cfg.three_pt_def_rank_opponent), 100),
        "both_poor_interior_defense": _rank_bottom_mask(_safe_series(work, cfg.interior_def_rank_team), 100) & _rank_bottom_mask(_safe_series(work, cfg.interior_def_rank_opponent), 100),
        "neither_team_forces_turnovers": _rank_bottom_mask(_safe_series(work, cfg.steal_rate_team).rank(ascending=False, method="average"), 50) & _rank_bottom_mask(_safe_series(work, cfg.steal_rate_opponent).rank(ascending=False, method="average"), 50),
        "both_efg_above_54": (efg_team >= 0.54) & (efg_opp >= 0.54),
        "four_way_3pt_confirmation": (_safe_series(work, cfg.three_pa_rate_team) >= _quantile_or_default(_safe_series(work, cfg.three_pa_rate_team), 0.70, 0.37)) & (_safe_series(work, cfg.three_pa_rate_opponent) >= _quantile_or_default(_safe_series(work, cfg.three_pa_rate_opponent), 0.70, 0.37)) & _rank_bottom_mask(_safe_series(work, cfg.three_pt_def_rank_team), 100) & _rank_bottom_mask(_safe_series(work, cfg.three_pt_def_rank_opponent), 100),
        "both_low_ft_rate": (_safe_series(work, cfg.fta_rate_team) <= _quantile_or_default(_safe_series(work, cfg.fta_rate_team), 0.25, 0.24)) & (_safe_series(work, cfg.fta_rate_opponent) <= _quantile_or_default(_safe_series(work, cfg.fta_rate_opponent), 0.25, 0.24)),
        "both_ppp_above_threshold": (ppp_team >= 1.10) & (ppp_opp >= 1.10),
        "first_meeting_different_conf": (_safe_series(work, cfg.meeting_number) <= 1) & (_safe_text(work, cfg.conference_team).str.lower().str.strip() != _safe_text(work, cfg.conference_opponent).str.lower().str.strip()),
        "both_coming_off_low_scoring_games": (_safe_series(work, cfg.prior_margin_team).abs() <= _quantile_or_default(_safe_series(work, cfg.prior_margin_team).abs(), 0.40, 8.0)),
        "early_tournament_round": _safe_bool(work, cfg.is_tournament) & _round_number_series(work, cfg.tourney_round).isin([1.0, 2.0]),
        "neither_team_true_home_court": _safe_bool(work, cfg.is_neutral_site),
        "both_high_spr": (_safe_series(work, cfg.spr_team) > 0.65) & (_safe_series(work, cfg.spr_opponent) > 0.65),
        "both_odi_offensive": (_safe_series(work, cfg.odi_team) > 0) & (_safe_series(work, cfg.odi_opponent) > 0),
        "both_mti_trending_up": (_safe_series(work, cfg.mti_team) > 0) & (_safe_series(work, cfg.mti_opponent) > 0),
        "high_pmi_both_fast": (_safe_series(work, cfg.pmi) >= _quantile_or_default(_safe_series(work, cfg.pmi), 0.70, 0.60)) & _rank_top_mask(pace_rank_team, 120) & _rank_top_mask(pace_rank_opp, 120),
    }

    results, redundancy, season_consistency = _evaluate_named_layers(
        frame=work,
        base_scope=base_scope,
        layers=layers,
        result_col="_result_blowout_over",
        cfg=cfg,
        min_sample=min_sample,
        p_threshold=p_threshold,
        lift_threshold=0.03,
    )
    _write_csv_with_metadata(
        output_path,
        results,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        base_population_n=int(base_scope.sum()),
    )
    return {
        "results": results,
        "redundancy": redundancy,
        "season_consistency": season_consistency,
        "base_population_n": int(base_scope.sum()),
        "output_path": str(output_path),
    }


def analyze_blowout_under_layers(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    blowout_under_threshold: float = 20.0,
    edge_threshold: float = 4.0,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    output_path: str | Path = "blowout_under_layer_results.csv",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Analyze layers where model-under calls win by large margin versus total line."""
    _ = kwargs
    work = df.copy()
    model_under = _safe_bool(work, cfg.model_projects_under)
    if model_under.sum() == 0:
        model_under = ~_safe_bool(work, cfg.model_projects_over)
    total_margin = _safe_series(work, cfg.final_margin)
    work["_result_blowout_under"] = ((total_margin <= -float(blowout_under_threshold)) & total_margin.notna()).astype(int)
    base_scope = model_under & total_margin.notna()

    pace_rank_team = _safe_series(work, cfg.pace_rank_team)
    pace_rank_opp = _safe_series(work, cfg.pace_rank_opponent)
    def_rank_team = _safe_series(work, cfg.def_rating_rank_team)
    def_rank_opp = _safe_series(work, cfg.def_rating_rank_opponent)
    pace_team = _safe_series(work, cfg.pace_team)
    pace_opp = _safe_series(work, cfg.pace_opponent)
    ppp_team = _safe_series(work, cfg.ppp_team)
    ppp_opp = _safe_series(work, cfg.ppp_opponent)
    market_total = _safe_series(work, cfg.market_total_close)
    implied_poss = market_total / (ppp_team + ppp_opp).replace(0.0, np.nan)
    proj_poss = (pace_team + pace_opp) / 2.0

    layers: Dict[str, pd.Series] = {
        "both_bottom25_tempo": _rank_bottom_mask(pace_rank_team, 90) & _rank_bottom_mask(pace_rank_opp, 90),
        "both_top25_force_slow_pace": _rank_top_mask(_safe_series(work, cfg.pace_allowed_rank_team), 25) & _rank_top_mask(_safe_series(work, cfg.pace_allowed_rank_opponent), 25),
        "combined_possessions_below_implied": (implied_poss - proj_poss) >= 8.0,
        "both_high_halfcourt_rate": (_safe_series(work, cfg.half_court_rate_team) >= _quantile_or_default(_safe_series(work, cfg.half_court_rate_team), 0.75, 0.55)) & (_safe_series(work, cfg.half_court_rate_opponent) >= _quantile_or_default(_safe_series(work, cfg.half_court_rate_opponent), 0.75, 0.55)),
        "both_high_foul_rate": (_safe_series(work, cfg.fta_rate_team) >= _quantile_or_default(_safe_series(work, cfg.fta_rate_team), 0.75, 0.33)) & (_safe_series(work, cfg.fta_rate_opponent) >= _quantile_or_default(_safe_series(work, cfg.fta_rate_opponent), 0.75, 0.33)),
        "both_top25_adj_def_efficiency": _rank_top_mask(def_rank_team, 25) & _rank_top_mask(def_rank_opp, 25),
        "both_top25_efg_allowed": (_safe_series(work, cfg.efg_allowed_team) <= 0.46) & (_safe_series(work, cfg.efg_allowed_opponent) <= 0.46),
        "both_force_high_tov": (_safe_series(work, cfg.tov_pct_opponent) >= 0.18) & (_safe_series(work, cfg.tov_pct_team) >= 0.18),
        "both_top25_block_rate": _rank_top_mask(_safe_series(work, cfg.block_rate_team).rank(ascending=False, method="average"), 25) & _rank_top_mask(_safe_series(work, cfg.block_rate_opponent).rank(ascending=False, method="average"), 25),
        "both_top25_opponent_3pt_allowed": _rank_top_mask(_safe_series(work, cfg.three_pt_def_rank_team), 25) & _rank_top_mask(_safe_series(work, cfg.three_pt_def_rank_opponent), 25),
        "both_efg_below_48": (_safe_series(work, cfg.efg_pct_team) <= 0.48) & (_safe_series(work, cfg.efg_pct_opponent) <= 0.48),
        "both_low_3pa_rate": (_safe_series(work, cfg.three_pa_rate_team) <= 0.30) & (_safe_series(work, cfg.three_pa_rate_opponent) <= 0.30),
        "both_poor_oreb": (_safe_series(work, cfg.oreb_pct_team) <= 0.28) & (_safe_series(work, cfg.dreb_pct_opp) >= _quantile_or_default(_safe_series(work, cfg.dreb_pct_opp), 0.60, 0.72)),
        "both_high_ft_dependency": (_safe_series(work, cfg.fta_rate_team) >= _quantile_or_default(_safe_series(work, cfg.fta_rate_team), 0.75, 0.33)) & (_safe_series(work, cfg.fta_rate_opponent) >= _quantile_or_default(_safe_series(work, cfg.fta_rate_opponent), 0.75, 0.33)),
        "third_meeting_same_conf": (_safe_series(work, cfg.meeting_number) >= 3.0) & (_safe_text(work, cfg.conference_team).str.lower().str.strip() == _safe_text(work, cfg.conference_opponent).str.lower().str.strip()),
        "both_coming_off_high_scoring_games": (_safe_series(work, cfg.prior_margin_team).abs() >= _quantile_or_default(_safe_series(work, cfg.prior_margin_team).abs(), 0.60, 10.0)),
        "high_stakes_elimination_both": _safe_bool(work, cfg.is_tournament) & _safe_bool(work, cfg.elimination_game_flag),
        "back_to_back_one_team": (_safe_series(work, cfg.days_since_last_game_team) <= 1.0) | (_safe_series(work, cfg.days_since_last_game_opponent) <= 1.0),
        "both_low_spr": (_safe_series(work, cfg.spr_team) < 0.45) & (_safe_series(work, cfg.spr_opponent) < 0.45),
        "both_odi_defensive": (_safe_series(work, cfg.odi_team) < 0) & (_safe_series(work, cfg.odi_opponent) < 0),
        "both_mti_flat_or_declining": (_safe_series(work, cfg.mti_team) <= 0) & (_safe_series(work, cfg.mti_opponent) <= 0),
        "low_combined_pmi": _safe_series(work, cfg.pmi) <= _quantile_or_default(_safe_series(work, cfg.pmi), 0.30, 0.40),
    }

    results, redundancy, season_consistency = _evaluate_named_layers(
        frame=work,
        base_scope=base_scope,
        layers=layers,
        result_col="_result_blowout_under",
        cfg=cfg,
        min_sample=min_sample,
        p_threshold=p_threshold,
        lift_threshold=0.03,
    )
    _write_csv_with_metadata(
        output_path,
        results,
        edge_threshold=edge_threshold,
        min_sample=min_sample,
        base_population_n=int(base_scope.sum()),
    )
    return {
        "results": results,
        "redundancy": redundancy,
        "season_consistency": season_consistency,
        "base_population_n": int(base_scope.sum()),
        "output_path": str(output_path),
    }


def analyze_march_madness_rd1_rd2_layers(
    df: pd.DataFrame,
    cfg: ColumnConfig,
    edge_threshold: float = 4.0,
    blue_blood_list: Optional[List[str]] = None,
    power_conferences: Optional[List[str]] = None,
    tourney_rounds: Optional[List[int]] = None,
    min_sample: int = 150,
    p_threshold: float = 0.05,
    output_path: str | Path = "march_madness_rd1_rd2_layer_results.csv",
    **kwargs: Any,
) -> Dict[str, Any]:
    """Analyze NCAA round 1/2 layers across ATS, ML, and totals outcomes."""
    _ = kwargs
    work = df.copy()
    blue_bloods = {x.lower().strip() for x in (blue_blood_list or ["duke", "kentucky", "kansas", "north carolina", "michigan state"])}
    power_confs = {x.lower().strip() for x in (power_conferences or ["acc", "sec", "big ten", "big 12", "big east", "pac-12"])}
    rounds = set(int(x) for x in (tourney_rounds or [1, 2]))

    round_num = _round_number_series(work, cfg.tourney_round)
    base_scope = _safe_bool(work, cfg.is_tournament) & round_num.isin(list(rounds))

    opp_name = _safe_text(work, cfg.opponent_name).str.lower().str.strip()
    conf_team = _safe_text(work, cfg.conference_team).str.lower().str.strip()
    conf_opp = _safe_text(work, cfg.conference_opponent).str.lower().str.strip()
    seed_team = _safe_series(work, cfg.seed_team)
    seed_opp = _safe_series(work, cfg.seed_opponent)
    seed_gap = _safe_series(work, cfg.seed_diff).abs().where(_safe_series(work, cfg.seed_diff).notna(), (seed_team - seed_opp).abs())
    eff_gap = (_safe_series(work, cfg.adj_efficiency_margin_team) - _safe_series(work, cfg.adj_efficiency_margin_opponent)).abs()
    pace_sum = _safe_series(work, cfg.pace_team) + _safe_series(work, cfg.pace_opponent)
    public_fav = _safe_series(work, cfg.public_pct_favorite)

    layers: Dict[str, pd.Series] = {
        "five_twelve_matchup": ((seed_team == 5) & (seed_opp == 12)) | ((seed_team == 12) & (seed_opp == 5)),
        "six_eleven_matchup": ((seed_team == 6) & (seed_opp == 11)) | ((seed_team == 11) & (seed_opp == 6)),
        "eight_nine_matchup": ((seed_team == 8) & (seed_opp == 9)) | ((seed_team == 9) & (seed_opp == 8)),
        "double_digit_seed_high_efficiency": (seed_team >= 10) & _rank_top_mask(_safe_series(work, cfg.pace_rank_team), 75),
        "blue_blood_favorite_inflated": opp_name.isin(blue_bloods) & (_safe_series(work, cfg.line_close) > 0),
        "seed_diff_misleads_efficiency": (seed_gap >= 4.0) & (eff_gap < 6.0),
        "mid_major_top100_efficiency_underdog": (~conf_team.isin(power_confs)) & _safe_bool(work, cfg.is_underdog) & _rank_top_mask(_safe_series(work, cfg.pace_rank_team), 100) & (_safe_series(work, cfg.line_close) >= 6.0),
        "rd1_bye_advantage": (round_num == 1.0) & ((_safe_series(work, cfg.days_since_last_game_team) - _safe_series(work, cfg.days_since_last_game_opponent)) >= 1.0),
        "opponent_conf_tourney_champ_fatigue": (round_num == 1.0) & (_safe_series(work, cfg.days_since_last_game_opponent) <= 1.0) & (_safe_series(work, cfg.consecutive_road_games_opponent) >= 3.0),
        "rest_differential_2plus_days": (_safe_series(work, cfg.days_since_last_game_team) - _safe_series(work, cfg.days_since_last_game_opponent)) >= 2.0,
        "opponent_conf_tourney_runner_up_fatigue": (round_num == 1.0) & (_safe_series(work, cfg.days_since_last_game_opponent) <= 1.0) & _safe_bool(work, cfg.prev_game_ot_opponent),
        "pseudo_home_pod_game": _safe_bool(work, cfg.is_neutral_site) & (_safe_text(work, cfg.region_team).str.lower().str.strip() == _safe_text(work, cfg.region_opponent).str.lower().str.strip()),
        "pseudo_road_opponent": _safe_bool(work, cfg.is_neutral_site) & (_safe_text(work, cfg.region_team).str.lower().str.strip() != _safe_text(work, cfg.region_opponent).str.lower().str.strip()),
        "both_true_neutral": _safe_bool(work, cfg.is_neutral_site),
        "travel_distance_gap": _safe_bool(work, cfg.is_neutral_site) & (_safe_series(work, cfg.consecutive_road_games_opponent) >= 2.0),
        "first_meeting_cross_conf": (_safe_series(work, cfg.meeting_number) <= 1.0) & (conf_team != conf_opp),
        "rd2_rematch_same_conf": (round_num == 2.0) & (_safe_series(work, cfg.meeting_number) >= 3.0) & (conf_team == conf_opp),
        "underdog_coach_upset_history": _safe_bool(work, cfg.is_underdog) & (_safe_series(work, cfg.coach_upset_rate_team) >= _quantile_or_default(_safe_series(work, cfg.coach_upset_rate_team), 0.50, 0.15)),
        "favorite_coach_low_tourney_experience": (~_safe_bool(work, cfg.is_underdog)) & (_safe_series(work, cfg.tourney_experience_score_team) <= _quantile_or_default(_safe_series(work, cfg.tourney_experience_score_team), 0.30, 0.30)),
        "efficiency_gap_under_8_rd1": (round_num == 1.0) & (eff_gap < 8.0),
        "efficiency_gap_under_8_rd2": (round_num == 2.0) & (eff_gap < 8.0),
        "underdog_physical_conf_vs_finesse": _safe_bool(work, cfg.is_underdog) & ((_safe_series(work, cfg.conf_strength_rank_team) + 8.0) < _safe_series(work, cfg.conf_strength_rank_opponent)),
        "underdog_elite_perimeter_scorer": _safe_bool(work, cfg.is_underdog) & (_safe_series(work, cfg.three_pa_rate_team) >= _quantile_or_default(_safe_series(work, cfg.three_pa_rate_team), 0.75, 0.38)) & _rank_bottom_mask(_safe_series(work, cfg.three_pt_def_rank_opponent), 100),
        "underdog_tourney_experience_advantage": _safe_bool(work, cfg.is_underdog) & (_safe_series(work, cfg.tourney_experience_score_team) > _safe_series(work, cfg.tourney_experience_score_opponent)),
        "public_75pct_plus_rd1_favorite": (round_num == 1.0) & (public_fav >= 0.75),
        "reverse_line_movement_rd1": (round_num == 1.0) & (_safe_series(work, cfg.line_close) < _safe_series(work, cfg.line_open)) & (public_fav >= 0.65),
        "name_brand_inflation": opp_name.isin(blue_bloods) & (eff_gap < 8.0),
        "cinderella_narrative_overreaction": (round_num == 2.0) & (~_safe_bool(work, cfg.is_underdog)) & (_safe_series(work, cfg.prior_margin_team) <= -10.0),
        "rd1_both_fast_different_conf": (round_num == 1.0) & (pace_sum >= _quantile_or_default(pace_sum, 0.70, 140.0)) & (conf_team != conf_opp),
        "rd1_both_poor_defenses_neutral": (round_num == 1.0) & _safe_bool(work, cfg.is_neutral_site) & _rank_bottom_mask(_safe_series(work, cfg.def_rating_rank_team), 75) & _rank_bottom_mask(_safe_series(work, cfg.def_rating_rank_opponent), 75),
        "rd2_both_survived_close_rd1": (round_num == 2.0) & (_safe_series(work, cfg.prior_margin_team).abs() <= 10.0),
        "rd1_two_defensive_identity_teams": (round_num == 1.0) & _rank_top_mask(_safe_series(work, cfg.def_rating_rank_team), 50) & _rank_top_mask(_safe_series(work, cfg.def_rating_rank_opponent), 50),
        "rd1_conf_tourney_champ_fatigue_under": (round_num == 1.0) & (_safe_series(work, cfg.days_since_last_game_opponent) <= 1.0) & (_safe_series(work, cfg.consecutive_road_games_opponent) >= 3.0),
        "underdog_mti_up_favorite_flat": _safe_bool(work, cfg.is_underdog) & (_safe_series(work, cfg.mti_team) > 0) & (_safe_series(work, cfg.mti_opponent) <= 0),
        "underdog_odi_defensive_elimination": _safe_bool(work, cfg.is_underdog) & _safe_bool(work, cfg.is_tournament) & (_safe_series(work, cfg.odi_team) < 0),
        "mid_major_sci_battle_tested": _safe_bool(work, cfg.is_underdog) & (~conf_team.isin(power_confs)) & (_safe_series(work, cfg.sci_team) >= _quantile_or_default(_safe_series(work, cfg.sci_team), 0.75, 0.65)),
        "low_spr_underdog_high_spr_favorite": _safe_bool(work, cfg.is_underdog) & (_safe_series(work, cfg.spr_team) < 0.45) & (_safe_series(work, cfg.spr_opponent) > 0.65),
        "possession_dominance_tournament": (_safe_series(work, cfg.oreb_pct_team) >= 0.33) & (_safe_series(work, cfg.tov_pct_team) <= 0.14) & (_safe_series(work, cfg.pace_rank_team) <= 100.0) & (_safe_series(work, cfg.steal_rate_team) >= _quantile_or_default(_safe_series(work, cfg.steal_rate_team), 0.60, 0.09)) & _safe_bool(work, cfg.is_tournament),
        "fast_efficient_vs_slow_poor": (_safe_series(work, cfg.pace_rank_team) <= 75.0) & (_safe_series(work, cfg.efg_pct_team) >= 0.52) & (_safe_series(work, cfg.pace_rank_opponent) >= 200.0) & (_safe_series(work, cfg.efg_pct_opponent) <= 0.49) & _safe_bool(work, cfg.is_tournament),
        "low_tov_high_oreb_neutral_site": (_safe_series(work, cfg.tov_pct_team) <= 0.14) & (_safe_series(work, cfg.oreb_pct_team) >= 0.32) & _safe_bool(work, cfg.is_neutral_site) & _safe_bool(work, cfg.is_tournament),
        "seed_efficiency_mismatch": (seed_gap >= 4.0) & ((_safe_series(work, cfg.ane_team) - _safe_series(work, cfg.ane_opponent)).abs() < 6.0),
    }

    rows: List[Dict[str, Any]] = []
    pass_layers: List[str] = []
    work_eval = work.copy()
    for layer_name, mask in layers.items():
        temp_col = _bool_name(layer_name)
        work_eval[temp_col] = mask.fillna(False).astype(bool)
        ats = analyze_layer(df=work_eval.loc[base_scope].copy(), layer_col=temp_col, result_col=cfg.covered, edge_col=cfg.model_edge, edge_threshold=-np.inf, label=layer_name, min_sample=min_sample, p_threshold=p_threshold, lift_threshold=0.03)
        ml = analyze_layer(df=work_eval.loc[base_scope].copy(), layer_col=temp_col, result_col=cfg.ml_won, edge_col=cfg.model_edge, edge_threshold=-np.inf, label=layer_name, min_sample=min_sample, p_threshold=p_threshold, lift_threshold=0.03)
        tot = analyze_layer(df=work_eval.loc[base_scope].copy(), layer_col=temp_col, result_col=cfg.total_covered, edge_col=cfg.model_edge, edge_threshold=-np.inf, label=layer_name, min_sample=min_sample, p_threshold=p_threshold, lift_threshold=0.03)
        rows.append({
            "layer": layer_name,
            "ats_n": ats["layered_n"], "ats_hit_rate": ats["layered_hit_rate"], "ats_lift": ats["lift"], "ats_p_value": ats["p_value"], "ats_verdict": ats["verdict"],
            "ml_n": ml["layered_n"], "ml_hit_rate": ml["layered_hit_rate"], "ml_lift": ml["lift"], "ml_p_value": ml["p_value"], "ml_verdict": ml["verdict"],
            "total_n": tot["layered_n"], "total_hit_rate": tot["layered_hit_rate"], "total_lift": tot["lift"], "total_p_value": tot["p_value"], "total_verdict": tot["verdict"],
        })
        if any([(pd.notna(ats["lift"]) and ats["lift"] > 0 and ats["layered_n"] >= min_sample), (pd.notna(ml["lift"]) and ml["lift"] > 0 and ml["layered_n"] >= min_sample), (pd.notna(tot["lift"]) and tot["lift"] > 0 and tot["layered_n"] >= min_sample)]):
            pass_layers.append(layer_name)

    result_df = pd.DataFrame(rows)
    red = redundancy_check(df=work_eval.loc[base_scope].copy(), layer_cols=[_bool_name(x) for x in pass_layers], edge_col=cfg.model_edge, result_col=cfg.covered, edge_threshold=-np.inf) if pass_layers else pd.DataFrame()
    result_df["high_collinearity_pair_count"] = int(red["high_collinearity"].sum()) if not red.empty else 0

    season_rows: List[pd.DataFrame] = []
    for layer_name in pass_layers:
        temp_col = _bool_name(layer_name)
        for market_key, result_col, metric_name in [("ats", cfg.covered, "season_positive_ratio_ats"), ("ml", cfg.ml_won, "season_positive_ratio_ml"), ("total", cfg.total_covered, "season_positive_ratio_total")]:
            sc = season_consistency_check(df=work_eval.loc[base_scope].copy(), layer_col=temp_col, season_col=cfg.season, result_col=result_col, edge_col=cfg.model_edge, edge_threshold=-np.inf)
            if sc.empty:
                continue
            sc["layer"] = layer_name
            sc["market"] = market_key
            season_rows.append(sc)
            valid = sc["layer_n"] > 0
            denom = int(valid.sum())
            ratio = float((sc.loc[valid, "lift"] > 0).mean()) if denom > 0 else np.nan
            result_df.loc[result_df["layer"] == layer_name, metric_name] = ratio

    result_df = result_df.sort_values(["ats_lift", "ml_lift", "total_lift"], ascending=[False, False, False]).reset_index(drop=True)
    season_out = pd.concat(season_rows, ignore_index=True, sort=False) if season_rows else pd.DataFrame()
    _write_csv_with_metadata(output_path, result_df, edge_threshold=edge_threshold, min_sample=min_sample, base_population_n=int(base_scope.sum()))
    return {"results": result_df, "redundancy": red, "season_consistency": season_out, "base_population_n": int(base_scope.sum()), "output_path": str(output_path)}


if __name__ == "__main__":
    config = ColumnConfig()
    demo_df = _synthetic_example_df(config, n=5000, seed=11)

    print("Running demo full backtest on synthetic data...")
    _ = run_full_backtest(
        df=demo_df,
        cfg=config,
        edge_threshold=4.0,
        min_sample=150,
        p_threshold=0.05,
        output_dir=Path("."),
    )

    # Demonstrate threshold sweeper separately
    sweeper = ThresholdSweeper(
        df=demo_df,
        metric_col=config.efg_pct_team,
        result_col=config.covered,
        edge_col=config.model_edge,
        edge_threshold=4.0,
        sweep_range=(0.44, 0.58),
        step=0.01,
        direction="above",
        min_sample=150,
    )
    sweep_results = sweeper.run()
    print("\nThreshold sweep top rows:")
    print(sweep_results.head(10).to_string(index=False))
    sweeper.plot_ascii()

    print("\nRunning V2 analyzer demos...")
    demo_blowout_win = analyze_blowout_win_layers(
        df=demo_df,
        cfg=config,
        edge_threshold=4.0,
        blowout_margin=15.0,
        min_sample=150,
        p_threshold=0.05,
        output_path=Path("blowout_win_layer_results.csv"),
    )
    print("\nBlowout win layers (top 5):")
    print(demo_blowout_win["results"].head(5).drop(columns=["verdict"], errors="ignore").to_string(index=False))

    demo_ud_ml = analyze_underdog_ml_win_layers(
        df=demo_df,
        cfg=config,
        edge_threshold=4.0,
        min_sample=150,
        p_threshold=0.05,
        output_path=Path("underdog_ml_win_layer_results.csv"),
    )
    print("\nUnderdog ML layers (top 5):")
    print(demo_ud_ml["results"].head(5).drop(columns=["verdict"], errors="ignore").to_string(index=False))

    demo_blowout_over = analyze_blowout_over_layers(
        df=demo_df,
        cfg=config,
        blowout_over_threshold=20.0,
        min_sample=150,
        p_threshold=0.05,
        output_path=Path("blowout_over_layer_results.csv"),
    )
    print("\nBlowout over layers (top 5):")
    print(demo_blowout_over["results"].head(5).drop(columns=["verdict"], errors="ignore").to_string(index=False))

    demo_blowout_under = analyze_blowout_under_layers(
        df=demo_df,
        cfg=config,
        blowout_under_threshold=20.0,
        min_sample=150,
        p_threshold=0.05,
        output_path=Path("blowout_under_layer_results.csv"),
    )
    print("\nBlowout under layers (top 5):")
    print(demo_blowout_under["results"].head(5).drop(columns=["verdict"], errors="ignore").to_string(index=False))

    demo_march = analyze_march_madness_rd1_rd2_layers(
        df=demo_df,
        cfg=config,
        edge_threshold=4.0,
        min_sample=150,
        p_threshold=0.05,
        output_path=Path("march_madness_rd1_rd2_layer_results.csv"),
    )
    print("\nMarch Madness R1/R2 layers (top 5):")
    print(demo_march["results"].head(5).to_string(index=False))
