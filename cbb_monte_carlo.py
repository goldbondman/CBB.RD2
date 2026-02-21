#!/usr/bin/env python3
"""
cbb_monte_carlo.py — Monte Carlo Simulation Layer for CBB Predictions

Post-processing enrichment that sits between ensemble model outputs and
the final prediction CSV.  Uses the 7-model spread/total estimates as
distribution parameters and runs N simulations per game to produce
confidence intervals, distribution-based confidence scores, and a
simulation-grounded upset probability that feeds the UWS system.

Pipeline integration:
    predictions_combined_latest.csv  →  cbb_monte_carlo.py  →  predictions_mc_latest.csv
                                                              →  mc_game_cards.csv

CLI:
    python cbb_monte_carlo.py
    python cbb_monte_carlo.py --input data/predictions_combined_latest.csv
    python cbb_monte_carlo.py --n-sims 10000
    python cbb_monte_carlo.py --dry-run
    python cbb_monte_carlo.py --game-id 401XXXXXX
    python cbb_monte_carlo.py --report
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

DATA_DIR = Path("data")
CSV_DIR = DATA_DIR / "csv"
WEIGHTS_PATH = DATA_DIR / "backtest_optimized_weights.json"

# League-average fallback values for missing rankings data
LEAGUE_AVG_PROFILES = {
    "consistency_score": 50.0,
    "cage_em": 0.0,
    "floor_em": -10.0,
    "ceiling_em": 10.0,
    "net_rtg_l5": 0.0,
    "cage_t": 68.0,
    "suffocation": 50.0,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GameSimInput:
    """All inputs needed to simulate a single game."""

    game_id: str
    home_team: str
    away_team: str
    home_team_id: str
    away_team_id: str

    # From ensemble
    ensemble_spread: float      # negative = home favored
    ensemble_total: float       # projected combined score
    spread_std: Optional[float] = None   # ensemble spread std if available
    total_std: Optional[float] = None    # ensemble total std if available

    # From rankings (joined by team_id)
    home_cage_em: float = 0.0
    away_cage_em: float = 0.0
    home_consistency: float = 50.0   # 0-100, used to scale variance
    away_consistency: float = 50.0
    home_floor_em: float = -10.0
    away_floor_em: float = -10.0
    home_ceiling_em: float = 10.0
    away_ceiling_em: float = 10.0
    home_net_rtg_l5: float = 0.0
    away_net_rtg_l5: float = 0.0
    home_cage_t: float = 68.0       # adjusted tempo
    away_cage_t: float = 68.0
    home_suffocation: float = 50.0   # defensive rating 0-100
    away_suffocation: float = 50.0

    # Betting context
    spread_line: Optional[float] = None
    total_line: Optional[float] = None
    neutral_site: bool = False

    # Edge flag from ensemble (for mc_edge_confirmed / mc_edge_contradicted)
    edge_flag: int = 0


@dataclass
class SimResult:
    """Full Monte Carlo simulation result for one game."""

    game_id: str
    n_sims: int

    # Spread distribution
    spread_mean: float = 0.0
    spread_median: float = 0.0
    spread_std_realized: float = 0.0
    spread_p10: float = 0.0
    spread_p25: float = 0.0
    spread_p75: float = 0.0
    spread_p90: float = 0.0

    # Cover probabilities (only populated if spread_line exists)
    home_covers_pct: Optional[float] = None
    away_covers_pct: Optional[float] = None
    push_pct: Optional[float] = None
    cover_probability: Optional[float] = None  # max of home/away covers

    # Total distribution
    total_mean: float = 0.0
    total_median: float = 0.0
    total_p10: float = 0.0
    total_p90: float = 0.0
    over_pct: Optional[float] = None
    under_pct: Optional[float] = None

    # Win probability
    home_win_pct: float = 0.5
    away_win_pct: float = 0.5
    upset_probability: float = 0.0

    # Flags
    high_variance_flag: bool = False
    low_variance_flag: bool = False
    model_alignment: str = "LEAN"   # "STRONG" / "SPLIT" / "LEAN"

    # Calibrated confidence
    mc_confidence: float = 0.5
    mc_confidence_tier: str = "LOW"

    # Edge enrichment
    mc_edge_confirmed: bool = False
    mc_edge_contradicted: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANCE MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def compute_spread_std(inp: GameSimInput, base_std: float = 10.5) -> float:
    """
    CBB spread outcomes have empirically ~10-11 point std dev.
    Adjust based on team characteristics:
      - consistency: low consistency teams → higher variance
      - pace: faster games → more possessions → regression to mean → lower variance
      - suffocation defense: elite defenses compress score distributions
    """
    avg_consistency = (inp.home_consistency + inp.away_consistency) / 2.0
    consistency_adj = 1.0 + (100.0 - avg_consistency) / 200.0  # range ~1.0 to 1.5

    avg_pace = (inp.home_cage_t + inp.away_cage_t) / 2.0
    pace_adj = 1.0 - (avg_pace - 68.0) / 100.0  # range ~0.95 to 1.05

    max_suffocation = max(inp.home_suffocation, inp.away_suffocation)
    defense_adj = 1.0 - (max_suffocation - 50.0) / 200.0  # range ~0.85 to 1.0

    return base_std * consistency_adj * pace_adj * defense_adj


def compute_total_std(inp: GameSimInput, base_std: float = 10.0) -> float:
    """
    Total variance model.  Uses pace more heavily (more possessions =
    total regresses toward mean faster).
    """
    avg_consistency = (inp.home_consistency + inp.away_consistency) / 2.0
    consistency_adj = 1.0 + (100.0 - avg_consistency) / 250.0  # slightly less than spread

    avg_pace = (inp.home_cage_t + inp.away_cage_t) / 2.0
    pace_adj = 1.0 - (avg_pace - 68.0) / 80.0  # heavier pace weighting than spread

    max_suffocation = max(inp.home_suffocation, inp.away_suffocation)
    defense_adj = 1.0 - (max_suffocation - 50.0) / 250.0

    return base_std * consistency_adj * pace_adj * defense_adj


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_game(
    inp: GameSimInput,
    n_sims: int = 5000,
    seed: Optional[int] = None,
) -> SimResult:
    """
    Run vectorized Monte Carlo simulation for a single game.

    Draws correlated spread and total samples from a bivariate normal
    distribution.  Spread and total are mildly negatively correlated in
    CBB (~-0.15) because defensive games compress both spread range and total.
    """
    rng = np.random.default_rng(seed)

    spread_std = inp.spread_std if inp.spread_std is not None else compute_spread_std(inp)
    total_std = inp.total_std if inp.total_std is not None else compute_total_std(inp)

    # Ensure positive standard deviations
    spread_std = max(spread_std, 1.0)
    total_std = max(total_std, 1.0)

    # Build covariance matrix with mild negative correlation
    correlation = -0.15
    cov = np.array([
        [spread_std ** 2, correlation * spread_std * total_std],
        [correlation * spread_std * total_std, total_std ** 2],
    ])

    # Add small jitter to diagonal for numerical stability (Cholesky safety)
    cov[0, 0] += 1e-8
    cov[1, 1] += 1e-8

    samples = rng.multivariate_normal(
        mean=[inp.ensemble_spread, inp.ensemble_total],
        cov=cov,
        size=n_sims,
    )
    spread_sims = samples[:, 0]
    total_sims = samples[:, 1]

    # ── Spread distribution ───────────────────────────────────────────────
    spread_mean = float(np.mean(spread_sims))
    spread_median = float(np.median(spread_sims))
    spread_std_realized = float(np.std(spread_sims))
    spread_p10 = float(np.percentile(spread_sims, 10))
    spread_p25 = float(np.percentile(spread_sims, 25))
    spread_p75 = float(np.percentile(spread_sims, 75))
    spread_p90 = float(np.percentile(spread_sims, 90))

    # ── Total distribution ────────────────────────────────────────────────
    total_mean = float(np.mean(total_sims))
    total_median = float(np.median(total_sims))
    total_p10 = float(np.percentile(total_sims, 10))
    total_p90 = float(np.percentile(total_sims, 90))

    # ── Win probability (who wins outright, ignoring spread line) ─────────
    # spread < 0 means home wins
    home_win_pct = float(np.mean(spread_sims < 0))
    away_win_pct = 1.0 - home_win_pct

    # ── Upset probability ─────────────────────────────────────────────────
    # Underdog = team not favored by the ensemble spread
    if inp.ensemble_spread < 0:
        # Home is favored → upset = away wins
        upset_probability = away_win_pct
    elif inp.ensemble_spread > 0:
        # Away is favored → upset = home wins
        upset_probability = home_win_pct
    else:
        upset_probability = 0.5

    # ── Cover probabilities (only if spread_line exists) ──────────────────
    home_covers_pct = None
    away_covers_pct = None
    push_pct = None
    cover_probability = None

    if inp.spread_line is not None:
        # Home covers if actual spread < spread_line (more negative = bigger win)
        diffs = spread_sims - inp.spread_line
        home_covers_pct = float(np.mean(diffs < -0.5))
        away_covers_pct = float(np.mean(diffs > 0.5))
        push_pct = float(np.mean(np.abs(diffs) <= 0.5))
        cover_probability = max(home_covers_pct, away_covers_pct)

    # ── Over/under probabilities (only if total_line exists) ──────────────
    over_pct = None
    under_pct = None

    if inp.total_line is not None:
        over_pct = float(np.mean(total_sims > inp.total_line))
        under_pct = float(np.mean(total_sims < inp.total_line))

    # ── Variance flags ────────────────────────────────────────────────────
    high_variance_flag = spread_std_realized > 13.0
    low_variance_flag = spread_std_realized < 7.5

    # ── Model alignment ───────────────────────────────────────────────────
    if spread_p10 > 0 or spread_p90 < 0:
        # p10 and p90 same sign → very strong
        model_alignment = "STRONG"
    elif (spread_p25 > 0 or spread_p75 < 0):
        # p25/p75 same sign but p10/p90 straddle → lean
        model_alignment = "LEAN"
    else:
        # p25 and p75 straddle zero
        model_alignment = "SPLIT"

    # ── MC confidence ─────────────────────────────────────────────────────
    if cover_probability is not None:
        raw_confidence = max(cover_probability, 1.0 - cover_probability)
    else:
        # No line → use win probability as confidence proxy
        raw_confidence = max(home_win_pct, away_win_pct)

    # Ensure always >= 0.5
    raw_confidence = max(raw_confidence, 0.5)

    # Apply calibration if available
    mc_confidence = _apply_calibration_from_file(raw_confidence)

    # Confidence tier
    if mc_confidence >= 0.72:
        mc_confidence_tier = "ELITE"
    elif mc_confidence >= 0.65:
        mc_confidence_tier = "HIGH"
    elif mc_confidence >= 0.58:
        mc_confidence_tier = "MEDIUM"
    else:
        mc_confidence_tier = "LOW"

    # ── Edge enrichment ───────────────────────────────────────────────────
    mc_edge_confirmed = False
    mc_edge_contradicted = False
    if cover_probability is not None:
        mc_edge_confirmed = (
            cover_probability > 0.60 and inp.edge_flag == 1
        )
        mc_edge_contradicted = (
            inp.edge_flag == 1 and cover_probability < 0.52
        )

    return SimResult(
        game_id=inp.game_id,
        n_sims=n_sims,
        spread_mean=round(spread_mean, 2),
        spread_median=round(spread_median, 2),
        spread_std_realized=round(spread_std_realized, 2),
        spread_p10=round(spread_p10, 1),
        spread_p25=round(spread_p25, 1),
        spread_p75=round(spread_p75, 1),
        spread_p90=round(spread_p90, 1),
        home_covers_pct=_round_or_none(home_covers_pct, 4),
        away_covers_pct=_round_or_none(away_covers_pct, 4),
        push_pct=_round_or_none(push_pct, 4),
        cover_probability=_round_or_none(cover_probability, 4),
        total_mean=round(total_mean, 1),
        total_median=round(total_median, 1),
        total_p10=round(total_p10, 1),
        total_p90=round(total_p90, 1),
        over_pct=_round_or_none(over_pct, 4),
        under_pct=_round_or_none(under_pct, 4),
        home_win_pct=round(home_win_pct, 4),
        away_win_pct=round(away_win_pct, 4),
        upset_probability=round(upset_probability, 4),
        high_variance_flag=high_variance_flag,
        low_variance_flag=low_variance_flag,
        model_alignment=model_alignment,
        mc_confidence=round(mc_confidence, 4),
        mc_confidence_tier=mc_confidence_tier,
        mc_edge_confirmed=mc_edge_confirmed,
        mc_edge_contradicted=mc_edge_contradicted,
    )


def _round_or_none(value: Optional[float], decimals: int) -> Optional[float]:
    """Round a value, returning None if input is None."""
    return round(value, decimals) if value is not None else None


# ═══════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_calibration(raw_confidence: float, calibration_table: dict) -> float:
    """
    Apply calibration adjustment from backtest_optimized_weights.json.

    Find the correct bucket (50-59, 60-69, etc.) and apply the adj multiplier.
    Clamp output to [0.50, 0.99].
    """
    bucket = int(raw_confidence * 100) // 10 * 10
    adj = calibration_table.get(str(bucket), 1.0)
    calibrated = raw_confidence * float(adj)
    return max(0.50, min(0.99, calibrated))


def _load_calibration_table() -> Optional[dict]:
    """Load confidence_calibration from backtest_optimized_weights.json."""
    if not WEIGHTS_PATH.exists() or WEIGHTS_PATH.stat().st_size < 10:
        return None
    try:
        payload = json.loads(WEIGHTS_PATH.read_text())
        table = payload.get("confidence_calibration")
        if isinstance(table, dict):
            return table
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return None


# Module-level cache for calibration table (loaded once)
_CALIBRATION_TABLE: Optional[dict] = None
_CALIBRATION_LOADED: bool = False


def _apply_calibration_from_file(raw_confidence: float) -> float:
    """Apply calibration from file if available, else return raw."""
    global _CALIBRATION_TABLE, _CALIBRATION_LOADED
    if not _CALIBRATION_LOADED:
        _CALIBRATION_TABLE = _load_calibration_table()
        _CALIBRATION_LOADED = True
    if _CALIBRATION_TABLE is not None:
        return apply_calibration(raw_confidence, _CALIBRATION_TABLE)
    return raw_confidence


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_slate(
    games: list,
    n_sims: int = 5000,
    n_jobs: int = 1,
    verbose: bool = True,
) -> list:
    """
    Run simulations for a full slate of games.

    Parameters
    ----------
    games : list[GameSimInput]
    n_sims : int
    n_jobs : int
        Reserved for future parallelism, not needed now.
    verbose : bool
        Print progress.

    Returns
    -------
    list[SimResult]
    """
    results = []
    t0 = time.time()
    total = len(games)

    for i, game in enumerate(games, 1):
        if verbose:
            print(f"[SIM] Game {i}/{total}: {game.home_team} vs {game.away_team}")
        result = simulate_game(game, n_sims=n_sims)
        results.append(result)

    elapsed = time.time() - t0
    total_draws = total * n_sims
    if verbose:
        print(
            f"[SIM] {total} games × {n_sims:,} sims = "
            f"{total_draws:,} draws ({elapsed:.1f}s)"
        )

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RANKINGS JOIN HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def load_team_profiles_for_sim(
    rankings_path: str = "data/cbb_rankings.csv",
) -> dict:
    """
    Returns dict keyed by team_id with all fields needed for GameSimInput.

    Falls back to league-average values for any missing team.
    Logs a warning for each team not found.
    """
    path = Path(rankings_path)
    profiles: dict = {}

    if not path.exists() or path.stat().st_size < 10:
        print("[WARN] Rankings not found — using league-average variance parameters")
        return profiles

    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception as exc:
        print(f"[WARN] Failed to read rankings: {exc}")
        return profiles

    # Identify the team_id column
    id_col = "team_id"
    if id_col not in df.columns:
        for candidate in ["id", "Team_ID", "team"]:
            if candidate in df.columns:
                id_col = candidate
                break

    if id_col not in df.columns:
        print("[WARN] No team_id column found in rankings CSV")
        return profiles

    for _, row in df.iterrows():
        raw_id = row.get(id_col, "")
        # Handle numeric IDs that pandas reads as float (1.0 → "1")
        try:
            if isinstance(raw_id, float) and raw_id == int(raw_id):
                tid = str(int(raw_id))
            else:
                tid = str(raw_id)
        except (ValueError, OverflowError):
            tid = str(raw_id)
        if not tid or tid == "nan":
            continue

        def _g(col, default=0.0):
            v = row.get(col, default)
            try:
                return float(v) if pd.notna(v) else default
            except (TypeError, ValueError):
                return default

        profiles[tid] = {
            "consistency_score": _g("consistency_score", LEAGUE_AVG_PROFILES["consistency_score"]),
            "cage_em": _g("cage_em", LEAGUE_AVG_PROFILES["cage_em"]),
            "floor_em": _g("floor_em", LEAGUE_AVG_PROFILES["floor_em"]),
            "ceiling_em": _g("ceiling_em", LEAGUE_AVG_PROFILES["ceiling_em"]),
            "net_rtg_l5": _g("net_rtg_l5", LEAGUE_AVG_PROFILES["net_rtg_l5"]),
            "cage_t": _g("cage_t", LEAGUE_AVG_PROFILES["cage_t"]),
            "suffocation": _g("suffocation", LEAGUE_AVG_PROFILES["suffocation"]),
        }

    print(f"[INFO] Loaded {len(profiles)} team profiles for MC simulation")
    return profiles


def _get_team_field(
    profiles: dict,
    team_id: str,
    field_name: str,
    default: float,
) -> float:
    """Safely get a field from team profiles, falling back to default."""
    team_data = profiles.get(team_id, {})
    return team_data.get(field_name, default)


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD GameSimInput FROM CSV ROW
# ═══════════════════════════════════════════════════════════════════════════════

def build_game_sim_inputs(
    df: pd.DataFrame,
    team_profiles: dict,
) -> list:
    """
    Convert a predictions DataFrame into a list of GameSimInput objects.

    Expects columns from predictions_combined_latest.csv (primary + ensemble merged).
    """
    inputs = []

    for _, row in df.iterrows():
        game_id = str(row.get("game_id", ""))
        home_team = str(row.get("home_team", ""))
        away_team = str(row.get("away_team", ""))
        home_team_id = str(row.get("home_team_id", ""))
        away_team_id = str(row.get("away_team_id", ""))

        # Get ensemble spread: prefer ens_ensemble_spread (renamed), else pred_spread
        ensemble_spread = _safe_float(row.get("ens_ens_spread"))
        if ensemble_spread is None:
            ensemble_spread = _safe_float(row.get("ens_spread"))
        if ensemble_spread is None:
            ensemble_spread = _safe_float(row.get("pred_spread"))
        if ensemble_spread is None:
            continue  # Can't simulate without a spread

        # Get ensemble total
        ensemble_total = _safe_float(row.get("ens_ens_total"))
        if ensemble_total is None:
            ensemble_total = _safe_float(row.get("ens_total"))
        if ensemble_total is None:
            ensemble_total = _safe_float(row.get("pred_total"))
        if ensemble_total is None:
            ensemble_total = 140.0  # fallback

        # Spread/total std from ensemble if available
        spread_std = _safe_float(row.get("ens_ens_spread_std"))
        if spread_std is None:
            spread_std = _safe_float(row.get("ens_spread_std"))
        total_std = None  # ensemble doesn't expose total_std currently

        # Betting context
        spread_line = _safe_float(row.get("spread_line"))
        total_line = _safe_float(row.get("total_line"))
        neutral_site = bool(row.get("neutral_site", False))

        # Edge flag
        edge_flag = int(_safe_float(row.get("edge_flag", 0)) or 0)
        if edge_flag == 0:
            edge_flag = int(_safe_float(row.get("ens_edge_flag_spread", 0)) or 0)

        # Team profile fields
        D = LEAGUE_AVG_PROFILES
        inp = GameSimInput(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            ensemble_spread=ensemble_spread,
            ensemble_total=ensemble_total,
            spread_std=spread_std,
            total_std=total_std,
            home_cage_em=_get_team_field(team_profiles, home_team_id, "cage_em", D["cage_em"]),
            away_cage_em=_get_team_field(team_profiles, away_team_id, "cage_em", D["cage_em"]),
            home_consistency=_get_team_field(team_profiles, home_team_id, "consistency_score", D["consistency_score"]),
            away_consistency=_get_team_field(team_profiles, away_team_id, "consistency_score", D["consistency_score"]),
            home_floor_em=_get_team_field(team_profiles, home_team_id, "floor_em", D["floor_em"]),
            away_floor_em=_get_team_field(team_profiles, away_team_id, "floor_em", D["floor_em"]),
            home_ceiling_em=_get_team_field(team_profiles, home_team_id, "ceiling_em", D["ceiling_em"]),
            away_ceiling_em=_get_team_field(team_profiles, away_team_id, "ceiling_em", D["ceiling_em"]),
            home_net_rtg_l5=_get_team_field(team_profiles, home_team_id, "net_rtg_l5", D["net_rtg_l5"]),
            away_net_rtg_l5=_get_team_field(team_profiles, away_team_id, "net_rtg_l5", D["net_rtg_l5"]),
            home_cage_t=_get_team_field(team_profiles, home_team_id, "cage_t", D["cage_t"]),
            away_cage_t=_get_team_field(team_profiles, away_team_id, "cage_t", D["cage_t"]),
            home_suffocation=_get_team_field(team_profiles, home_team_id, "suffocation", D["suffocation"]),
            away_suffocation=_get_team_field(team_profiles, away_team_id, "suffocation", D["suffocation"]),
            spread_line=spread_line,
            total_line=total_line,
            neutral_site=neutral_site,
            edge_flag=edge_flag,
        )
        inputs.append(inp)

    return inputs


def _safe_float(val) -> Optional[float]:
    """Convert a value to float, returning None if not possible."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT ENRICHMENT
# ═══════════════════════════════════════════════════════════════════════════════

def results_to_mc_columns(results: list) -> pd.DataFrame:
    """Convert SimResult list to DataFrame with mc_ prefixed columns."""
    rows = []
    for r in results:
        rows.append({
            "game_id": r.game_id,
            "mc_spread_median": r.spread_median,
            "mc_spread_p10": r.spread_p10,
            "mc_spread_p25": r.spread_p25,
            "mc_spread_p75": r.spread_p75,
            "mc_spread_p90": r.spread_p90,
            "mc_spread_std": r.spread_std_realized,
            "mc_total_median": r.total_median,
            "mc_total_p10": r.total_p10,
            "mc_total_p90": r.total_p90,
            "mc_home_win_pct": r.home_win_pct,
            "mc_away_win_pct": r.away_win_pct,
            "mc_cover_probability": r.cover_probability,
            "mc_over_pct": r.over_pct,
            "mc_under_pct": r.under_pct,
            "mc_upset_probability": r.upset_probability,
            "mc_confidence": r.mc_confidence,
            "mc_confidence_tier": r.mc_confidence_tier,
            "mc_high_variance": r.high_variance_flag,
            "mc_low_variance": r.low_variance_flag,
            "mc_model_alignment": r.model_alignment,
            "mc_edge_confirmed": r.mc_edge_confirmed,
            "mc_edge_contradicted": r.mc_edge_contradicted,
            "mc_n_sims": r.n_sims,
        })
    return pd.DataFrame(rows)


def build_game_cards(
    combined_df: pd.DataFrame,
    mc_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build frontend-ready mc_game_cards.csv by joining combined predictions
    with MC output.  One row per game.
    """
    from datetime import datetime, timezone

    if "game_id" not in combined_df.columns or "game_id" not in mc_df.columns:
        return pd.DataFrame()

    merged = combined_df.merge(mc_df, on="game_id", how="inner")
    if merged.empty:
        return pd.DataFrame()

    cards = pd.DataFrame()
    cards["game_id"] = merged["game_id"]

    # Game context
    for col in ["game_date", "game_time_pst", "game_datetime_utc"]:
        if col in merged.columns:
            cards[col] = merged[col]

    cards["home_team"] = merged.get("home_team", "")
    cards["away_team"] = merged.get("away_team", "")

    if "home_conference" in merged.columns:
        cards["conference"] = merged["home_conference"]
    if "venue" in merged.columns:
        cards["venue"] = merged["venue"]

    # Prediction context
    cards["pred_spread"] = merged.get("pred_spread")
    cards["spread_line"] = merged.get("spread_line")

    # Edge info
    if "edge_flag" in merged.columns:
        cards["edge"] = merged["edge_flag"]
    if "spread_diff_vs_line" in merged.columns:
        cards["edge_tier"] = merged["spread_diff_vs_line"]

    cards["pred_total"] = merged.get("pred_total")
    cards["total_line"] = merged.get("total_line")

    # MC columns
    cards["mc_cover_probability"] = merged.get("mc_cover_probability")
    cards["mc_confidence_tier"] = merged.get("mc_confidence_tier")

    # Formatted ranges
    cards["mc_spread_range"] = merged.apply(
        lambda r: f"{r.get('mc_spread_p25', ''):.0f} to {r.get('mc_spread_p75', ''):.0f}"
        if pd.notna(r.get("mc_spread_p25")) and pd.notna(r.get("mc_spread_p75"))
        else "",
        axis=1,
    )
    cards["mc_total_range"] = merged.apply(
        lambda r: f"{r.get('mc_total_p10', ''):.0f} to {r.get('mc_total_p90', ''):.0f}"
        if pd.notna(r.get("mc_total_p10")) and pd.notna(r.get("mc_total_p90"))
        else "",
        axis=1,
    )

    cards["mc_home_win_pct"] = merged.get("mc_home_win_pct")
    cards["mc_away_win_pct"] = merged.get("mc_away_win_pct")
    cards["mc_upset_probability"] = merged.get("mc_upset_probability")
    cards["mc_model_alignment"] = merged.get("mc_model_alignment")
    cards["mc_high_variance"] = merged.get("mc_high_variance")
    cards["mc_edge_confirmed"] = merged.get("mc_edge_confirmed")
    cards["mc_edge_contradicted"] = merged.get("mc_edge_contradicted")

    # UWS alert (from ensemble)
    uws_col = next(
        (c for c in ["uws_total", "ens_uws_total"] if c in merged.columns),
        None,
    )
    if uws_col:
        cards["uws_alert"] = pd.to_numeric(merged[uws_col], errors="coerce") >= 55
    else:
        cards["uws_alert"] = False

    # Ensemble spread for comparison
    ens_spread_col = next(
        (c for c in ["ens_ens_spread", "ens_ensemble_spread", "ens_spread"]
         if c in merged.columns),
        None,
    )
    if ens_spread_col:
        cards["ensemble_spread"] = merged[ens_spread_col]

    # Model agreement
    if "ens_model_agreement" in merged.columns:
        cards["model_agreement"] = merged["ens_model_agreement"]
    elif "ens_agreement" in merged.columns:
        cards["model_agreement"] = merged["ens_agreement"]
    elif "ens_ens_agreement" in merged.columns:
        cards["model_agreement"] = merged["ens_ens_agreement"]

    cards["generated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    return cards


def print_report(results: list) -> None:
    """Print summary statistics for a slate of simulation results."""
    if not results:
        print("[REPORT] No simulation results to report")
        return

    n = len(results)
    print(f"\n{'='*70}")
    print(f"  MONTE CARLO SIMULATION REPORT — {n} game(s)")
    print(f"{'='*70}")

    # Confidence tier distribution
    tiers = {"ELITE": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for r in results:
        tiers[r.mc_confidence_tier] = tiers.get(r.mc_confidence_tier, 0) + 1

    print(f"\n  Confidence tiers:")
    for tier, count in tiers.items():
        pct = count / n * 100
        print(f"    {tier:8s}: {count:3d} ({pct:5.1f}%)")

    # Alignment distribution
    alignments = {}
    for r in results:
        alignments[r.model_alignment] = alignments.get(r.model_alignment, 0) + 1
    print(f"\n  Model alignment:")
    for align, count in alignments.items():
        print(f"    {align:8s}: {count:3d}")

    # Variance flags
    hv = sum(1 for r in results if r.high_variance_flag)
    lv = sum(1 for r in results if r.low_variance_flag)
    print(f"\n  High variance games: {hv}")
    print(f"  Low variance games:  {lv}")

    # Edge enrichment
    confirmed = sum(1 for r in results if r.mc_edge_confirmed)
    contradicted = sum(1 for r in results if r.mc_edge_contradicted)
    print(f"\n  Edges confirmed:     {confirmed}")
    print(f"  Edges contradicted:  {contradicted}")

    # Average confidence
    avg_conf = sum(r.mc_confidence for r in results) / n
    print(f"\n  Average MC confidence: {avg_conf:.3f}")

    # Cover probabilities (for games with lines)
    cover_games = [r for r in results if r.cover_probability is not None]
    if cover_games:
        avg_cover = sum(r.cover_probability for r in cover_games) / len(cover_games)
        print(f"  Average cover probability: {avg_cover:.3f} ({len(cover_games)} games with lines)")

    print(f"{'='*70}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation layer for CBB predictions",
    )
    parser.add_argument(
        "--input",
        default="data/predictions_combined_latest.csv",
        help="Path to predictions CSV (default: data/predictions_combined_latest.csv)",
    )
    parser.add_argument(
        "--n-sims", type=int, default=5000,
        help="Number of simulations per game (default: 5000)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print results, don't write CSV",
    )
    parser.add_argument(
        "--game-id",
        help="Run simulation for a single game only",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Summary stats only, no CSV write",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists() or input_path.stat().st_size < 10:
        print(f"[WARN] Input file not found or empty: {input_path}")
        return

    print(f"[MC] Loading predictions from {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    if df.empty:
        print("[WARN] Predictions CSV is empty — nothing to simulate")
        return

    # Filter to single game if requested
    if args.game_id:
        if "game_id" in df.columns:
            df = df[df["game_id"].astype(str) == str(args.game_id)]
        if df.empty:
            print(f"[WARN] Game {args.game_id} not found in predictions")
            return

    print(f"[MC] {len(df)} games to simulate")

    # Load team profiles from rankings
    team_profiles = load_team_profiles_for_sim()

    # Build simulation inputs
    game_inputs = build_game_sim_inputs(df, team_profiles)
    if not game_inputs:
        print("[WARN] No valid games to simulate (missing spread data?)")
        return

    print(f"[MC] {len(game_inputs)} games with valid inputs")

    # Run simulations
    results = simulate_slate(game_inputs, n_sims=args.n_sims)

    # Report mode — print summary and exit
    if args.report:
        print_report(results)
        return

    # Print report in all modes
    print_report(results)

    # Dry run — don't write files
    if args.dry_run:
        print("[MC] Dry run — skipping file writes")
        return

    # Build MC columns DataFrame
    mc_df = results_to_mc_columns(results)

    # Ensure output directories exist
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    # Merge MC columns into combined predictions
    if "game_id" in df.columns and "game_id" in mc_df.columns:
        df["game_id"] = df["game_id"].astype(str)
        mc_df["game_id"] = mc_df["game_id"].astype(str)
        enriched = df.merge(mc_df, on="game_id", how="left")
    else:
        enriched = pd.concat([df.reset_index(drop=True), mc_df.reset_index(drop=True)], axis=1)

    # Write enriched predictions
    mc_path = DATA_DIR / "predictions_mc_latest.csv"
    enriched.to_csv(mc_path, index=False)
    print(f"[OK] MC-enriched predictions: {len(enriched)} rows → {mc_path}")

    # Overwrite combined in place so downstream consumers get MC columns
    enriched.to_csv(input_path, index=False)
    print(f"[OK] Updated {input_path} with MC columns")

    # Copy to csv/ for frontend
    csv_mc_path = CSV_DIR / "predictions_mc_latest.csv"
    enriched.to_csv(csv_mc_path, index=False)
    print(f"[OK] Frontend copy → {csv_mc_path}")

    # Build game cards
    cards = build_game_cards(df, mc_df)
    if not cards.empty:
        cards_path = CSV_DIR / "mc_game_cards.csv"
        cards.to_csv(cards_path, index=False)
        print(f"[OK] Game cards: {len(cards)} rows → {cards_path}")
    else:
        print("[WARN] No game cards generated")


if __name__ == "__main__":
    main()
