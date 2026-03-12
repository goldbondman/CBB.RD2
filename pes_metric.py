"""PES metric computation utilities for CAGE.

This module introduces the Possession Efficiency Score (PES) and its
subcomponents:
- PGS: Possession Generation Score
- CES: Conversion Efficiency Score

The implementation is fully vectorized and configurable via ``PESConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


REGULAR_SEASON_WEIGHTS: Dict[str, float] = {
    # PGS weights
    "oreb": 0.25,
    "tov_inv": 0.25,
    "pace": 0.20,
    "forced_tov": 0.15,
    # CES weights
    "efg": 0.35,
    "ppp": 0.25,
    "fta_rate": 0.15,
    "opp_efg_inv": 0.25,
}

TOURNAMENT_WEIGHTS: Dict[str, float] = {
    # PGS weights -- possession levers elevated in single elimination
    "oreb": 0.30,
    "tov_inv": 0.30,
    "pace": 0.20,
    "forced_tov": 0.20,
    # CES weights
    "efg": 0.35,
    "ppp": 0.25,
    "fta_rate": 0.10,
    "opp_efg_inv": 0.30,
}

_PGS_KEYS: Tuple[str, ...] = ("oreb", "tov_inv", "pace", "forced_tov")
_CES_KEYS: Tuple[str, ...] = ("efg", "ppp", "fta_rate", "opp_efg_inv")
_QUADRANT_THRESHOLD: float = 0.60


@dataclass(frozen=True)
class PESConfig:
    """Column mapping for PES computation."""

    oreb_pct: str = "oreb_pct"
    tov_pct: str = "tov_pct"
    pace_rank: str = "pace_rank"
    forced_tov_rate: str = "forced_tov_rate"
    efg_pct: str = "efg_pct"
    ppp: str = "ppp"
    fta_rate: str = "fta_rate"
    opp_efg_pct: str = "opp_efg_pct"
    is_tournament: str = "is_tournament"
    seed: str = "seed"
    season: str = "season"


def normalize_series(s: pd.Series, method: str = "minmax") -> pd.Series:
    """Normalize a series for PES computation.

    Parameters
    ----------
    s:
        Source numeric series.
    method:
        ``"minmax"`` for 0-1 scaling or ``"zscore"`` for standard-normal
        scaling.

    Returns
    -------
    pd.Series
        Normalized series with NaN-safe handling.
    """
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.empty:
        return pd.Series(dtype="float64", index=s.index)

    fill_value = float(numeric.dropna().median()) if numeric.notna().any() else 0.0
    clean = numeric.fillna(fill_value).astype(float)

    method_norm = method.strip().lower()
    if method_norm == "zscore":
        std = float(clean.std(ddof=0))
        if std == 0.0 or np.isnan(std):
            return pd.Series(0.0, index=clean.index, dtype="float64")
        mean = float(clean.mean())
        return (clean - mean) / std

    if method_norm != "minmax":
        raise ValueError(f"Unsupported normalization method: {method}")

    min_v = float(clean.min())
    max_v = float(clean.max())
    if np.isclose(max_v, min_v):
        return pd.Series(0.5, index=clean.index, dtype="float64")
    return (clean - min_v) / (max_v - min_v)


def _to_unit_interval(s: pd.Series, method: str) -> pd.Series:
    """Map normalized values onto [0, 1]."""
    if method.strip().lower() == "zscore":
        # Convert z-scores to probabilities (normal CDF approximation).
        arr = pd.to_numeric(s, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        probs = 0.5 * (1.0 + np.erf(arr / np.sqrt(2.0)))
        return pd.Series(np.clip(probs, 0.0, 1.0), index=s.index, dtype="float64")
    return pd.to_numeric(s, errors="coerce").fillna(0.0).clip(0.0, 1.0)


def _select_weights(weights: Optional[Dict[str, float]], tournament: bool) -> Dict[str, float]:
    """Select and validate the active PES weight dictionary."""
    selected = dict(weights) if weights is not None else dict(TOURNAMENT_WEIGHTS if tournament else REGULAR_SEASON_WEIGHTS)
    missing = [k for k in (*_PGS_KEYS, *_CES_KEYS) if k not in selected]
    if missing:
        raise ValueError(f"Missing PES weight keys: {missing}")
    return selected


def _normalize_weight_group(weights: Dict[str, float], keys: Iterable[str]) -> Dict[str, float]:
    """Normalize a subset of weights so the group sums to 1."""
    values = {k: float(weights[k]) for k in keys}
    total = float(sum(values.values()))
    if total <= 0.0:
        raise ValueError(f"Weight group has non-positive total for keys={list(keys)}")
    return {k: v / total for k, v in values.items()}


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    """Return numeric series for a column with NaN fallback."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def compute_pes(
    df: pd.DataFrame,
    cfg: PESConfig,
    weights: Dict[str, float] | None = None,
    tournament: bool = False,
    normalize_method: str = "minmax",
    suffix: str = "",
) -> pd.DataFrame:
    """Compute PGS, CES, and PES for every row.

    Notes
    -----
    - PGS and CES are produced on a [0, 1] scale.
    - PES is ``PGS * CES * 100`` (0-100 readability scale).
    - Input ``df`` is never mutated.
    """
    out = df.copy(deep=True)
    active_weights = _select_weights(weights, tournament=tournament)
    pgs_w = _normalize_weight_group(active_weights, _PGS_KEYS)
    ces_w = _normalize_weight_group(active_weights, _CES_KEYS)

    comp_oreb = _to_unit_interval(normalize_series(_safe_numeric(out, cfg.oreb_pct), method=normalize_method), normalize_method)
    comp_tov = _to_unit_interval(normalize_series(_safe_numeric(out, cfg.tov_pct), method=normalize_method), normalize_method)
    comp_pace_rank = _to_unit_interval(normalize_series(_safe_numeric(out, cfg.pace_rank), method=normalize_method), normalize_method)
    comp_forced_tov = _to_unit_interval(normalize_series(_safe_numeric(out, cfg.forced_tov_rate), method=normalize_method), normalize_method)
    comp_efg = _to_unit_interval(normalize_series(_safe_numeric(out, cfg.efg_pct), method=normalize_method), normalize_method)
    comp_ppp = _to_unit_interval(normalize_series(_safe_numeric(out, cfg.ppp), method=normalize_method), normalize_method)
    comp_fta = _to_unit_interval(normalize_series(_safe_numeric(out, cfg.fta_rate), method=normalize_method), normalize_method)
    comp_opp_efg = _to_unit_interval(normalize_series(_safe_numeric(out, cfg.opp_efg_pct), method=normalize_method), normalize_method)

    pgs = (
        pgs_w["oreb"] * comp_oreb
        + pgs_w["tov_inv"] * (1.0 - comp_tov)
        + pgs_w["pace"] * (1.0 - comp_pace_rank)
        + pgs_w["forced_tov"] * comp_forced_tov
    ).clip(0.0, 1.0)
    ces = (
        ces_w["efg"] * comp_efg
        + ces_w["ppp"] * comp_ppp
        + ces_w["fta_rate"] * comp_fta
        + ces_w["opp_efg_inv"] * (1.0 - comp_opp_efg)
    ).clip(0.0, 1.0)
    pes = (pgs * ces * 100.0).clip(0.0, 100.0)

    out[f"pgs{suffix}"] = pgs
    out[f"ces{suffix}"] = ces
    out[f"pes{suffix}"] = pes
    return out


def _resolve_side_column(df: pd.DataFrame, base_col: str, side_suffix: str, fallback_cols: Tuple[str, ...]) -> pd.Series:
    """Resolve side-specific column with fallback lookup order."""
    candidates = (f"{base_col}{side_suffix}", *fallback_cols, base_col)
    for candidate in candidates:
        if candidate in df.columns:
            return pd.to_numeric(df[candidate], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _assign_quadrant(pgs: pd.Series, ces: pd.Series) -> pd.Series:
    """Assign PES quadrant labels from team PGS/CES values."""
    cond_elite = (pgs >= _QUADRANT_THRESHOLD) & (ces >= _QUADRANT_THRESHOLD)
    cond_efficient = (ces >= _QUADRANT_THRESHOLD) & (pgs < _QUADRANT_THRESHOLD)
    cond_possession = (pgs >= _QUADRANT_THRESHOLD) & (ces < _QUADRANT_THRESHOLD)
    return pd.Series(
        np.select(
            [cond_elite, cond_efficient, cond_possession],
            ["elite", "efficient_only", "possession_only"],
            default="fodder",
        ),
        index=pgs.index,
        dtype="object",
    )


def compute_pes_matchup(
    df: pd.DataFrame,
    cfg: PESConfig,
    team_suffix: str = "_team",
    opponent_suffix: str = "_opponent",
    tournament: bool = False,
) -> pd.DataFrame:
    """Compute side-by-side PES metrics and matchup differentials."""
    out = df.copy(deep=True)

    team_frame = pd.DataFrame(index=out.index)
    opp_frame = pd.DataFrame(index=out.index)

    for feature in (
        cfg.oreb_pct,
        cfg.tov_pct,
        cfg.pace_rank,
        cfg.forced_tov_rate,
        cfg.efg_pct,
        cfg.ppp,
        cfg.fta_rate,
    ):
        team_frame[feature] = _resolve_side_column(out, feature, team_suffix, tuple())
        opp_frame[feature] = _resolve_side_column(out, feature, opponent_suffix, tuple())

    team_frame[cfg.opp_efg_pct] = _resolve_side_column(
        out,
        cfg.opp_efg_pct,
        team_suffix,
        (f"{cfg.efg_pct}{opponent_suffix}",),
    )
    opp_frame[cfg.opp_efg_pct] = _resolve_side_column(
        out,
        cfg.opp_efg_pct,
        opponent_suffix,
        (f"{cfg.efg_pct}{team_suffix}",),
    )

    for shared in (cfg.is_tournament, cfg.seed, cfg.season):
        if shared in out.columns:
            team_frame[shared] = out[shared]
            opp_frame[shared] = out[shared]
        else:
            team_frame[shared] = np.nan
            opp_frame[shared] = np.nan

    team_scores = compute_pes(team_frame, cfg=cfg, tournament=tournament)
    opp_scores = compute_pes(opp_frame, cfg=cfg, tournament=tournament)

    out["pgs_team"] = team_scores["pgs"]
    out["ces_team"] = team_scores["ces"]
    out["pes_team"] = team_scores["pes"]
    out["pgs_opponent"] = opp_scores["pgs"]
    out["ces_opponent"] = opp_scores["ces"]
    out["pes_opponent"] = opp_scores["pes"]
    out["pgs_diff"] = out["pgs_team"] - out["pgs_opponent"]
    out["ces_diff"] = out["ces_team"] - out["ces_opponent"]
    out["pes_diff"] = out["pes_team"] - out["pes_opponent"]
    out["pes_quadrant"] = _assign_quadrant(out["pgs_team"], out["ces_team"])
    return out


def plot_pes_quadrant(
    df: pd.DataFrame,
    label_col: str = "team_name",
    highlight_seeds: Optional[list[int]] = None,
    width: int = 80,
    height: int = 30,
    tournament_only: bool = True,
) -> None:
    """Print an ASCII PGS-vs-CES quadrant plot."""
    if "pgs_team" not in df.columns or "ces_team" not in df.columns:
        print("PES quadrant plot unavailable: missing pgs_team/ces_team columns.")
        return

    plot_df = df.copy()
    if tournament_only and "is_tournament" in plot_df.columns:
        tourney = pd.to_numeric(plot_df["is_tournament"], errors="coerce").fillna(0) != 0
        plot_df = plot_df.loc[tourney].copy()

    if plot_df.empty:
        print("PES quadrant plot unavailable: no rows to plot.")
        return

    pgs = pd.to_numeric(plot_df["pgs_team"], errors="coerce").clip(0.0, 1.0)
    ces = pd.to_numeric(plot_df["ces_team"], errors="coerce").clip(0.0, 1.0)
    pes = pd.to_numeric(plot_df.get("pes_team"), errors="coerce")
    quadrant = plot_df.get("pes_quadrant", _assign_quadrant(pgs, ces)).astype(str)

    med_x = float(pgs.median())
    med_y = float(ces.median())

    grid = [[" " for _ in range(max(10, width))] for _ in range(max(10, height))]
    w = len(grid[0])
    h = len(grid)

    x_mid = int(np.clip(round(med_x * (w - 1)), 0, w - 1))
    y_mid = int(np.clip(round((1.0 - med_y) * (h - 1)), 0, h - 1))
    for x in range(w):
        grid[y_mid][x] = "-"
    for y in range(h):
        grid[y][x_mid] = "|"
    grid[y_mid][x_mid] = "+"

    highlight = set(highlight_seeds or [])
    if "seed" in plot_df.columns:
        seeds = pd.to_numeric(plot_df["seed"], errors="coerce")
    else:
        seeds = pd.Series(np.nan, index=plot_df.index, dtype="float64")

    symbol_map = {
        "elite": "*",
        "efficient_only": "o",
        "possession_only": "+",
        "fodder": ".",
    }

    for idx in plot_df.index:
        xv = pgs.loc[idx]
        yv = ces.loc[idx]
        if pd.isna(xv) or pd.isna(yv):
            continue
        gx = int(np.clip(round(float(xv) * (w - 1)), 0, w - 1))
        gy = int(np.clip(round((1.0 - float(yv)) * (h - 1)), 0, h - 1))

        seed_val = seeds.loc[idx] if idx in seeds.index else np.nan
        if pd.notna(seed_val) and int(seed_val) in highlight:
            ch = str(int(seed_val))[-1]
        else:
            ch = symbol_map.get(quadrant.loc[idx], ".")

        if grid[gy][gx] not in {" ", "-", "|", "+"}:
            grid[gy][gx] = "#"
        else:
            grid[gy][gx] = ch

    print("PES Quadrant Plot (X=PGS, Y=CES)")
    print("Top-left: possession_only | Top-right: elite")
    print("Bottom-left: fodder       | Bottom-right: efficient_only")
    for row in grid:
        print("".join(row))

    counts = quadrant.value_counts(dropna=False)
    print("\nQuadrant counts:")
    for key in ("elite", "efficient_only", "possession_only", "fodder"):
        print(f"  {key}: {int(counts.get(key, 0))}")

    label_series = plot_df[label_col].astype(str) if label_col in plot_df.columns else pd.Series(plot_df.index.astype(str), index=plot_df.index)
    elite_labels = label_series.loc[quadrant == "elite"].head(20).tolist()
    if elite_labels:
        print("\nElite labels:")
        print("  " + ", ".join(elite_labels))

    ranking = pd.DataFrame({"label": label_series, "pes": pes}).dropna(subset=["pes"])
    top5 = ranking.sort_values("pes", ascending=False).head(5)
    bot5 = ranking.sort_values("pes", ascending=True).head(5)

    print("\nTop 5 PES:")
    if top5.empty:
        print("  (none)")
    else:
        for _, r in top5.iterrows():
            print(f"  {r['label']}: {float(r['pes']):.2f}")

    print("\nBottom 5 PES:")
    if bot5.empty:
        print("  (none)")
    else:
        for _, r in bot5.iterrows():
            print(f"  {r['label']}: {float(r['pes']):.2f}")


def get_pes_tier(pes_score: float) -> str:
    """Classify PES score into tournament context tiers."""
    if pes_score >= 75:
        return "Elite"
    if pes_score >= 60:
        return "Dangerous"
    if pes_score >= 45:
        return "Competitive"
    if pes_score >= 30:
        return "Vulnerable"
    return "Fodder"
