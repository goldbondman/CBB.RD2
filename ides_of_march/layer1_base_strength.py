from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODEL_A_WEIGHTS, MODEL_B_WEIGHTS

# ---------------------------------------------------------------------------
# Season anchor weights
# ---------------------------------------------------------------------------
# Regular season: mostly recent form (L12), small season anchor
# Tournament: mostly season average — stops Big-12/ACC end-of-season slumps
# from making elite teams appear worse than mid-major conference champions.
SEASON_ANCHOR_WEIGHT_DEFAULT    = 0.25   # 75% L12 / 25% season
SEASON_ANCHOR_WEIGHT_TOURNAMENT = 0.90   # 10% L12 / 90% season

# ---------------------------------------------------------------------------
# KenPom blend weight (NCAA tournament only)
# ---------------------------------------------------------------------------
# KenPom AdjEM is a stable, full-season absolute efficiency margin calibrated
# against the entire D-I field.  It is far more discriminating than the rolling
# ESPN adj_net_rtg for extreme mismatches (elite Power-6 vs single-bid teams).
# For tournament games we blend the ESPN-based blended_adj_em toward the KenPom
# AdjEM margin.  Weight ramps from 0 (no KenPom available) to KENPOM_MAX_WEIGHT.
KENPOM_TOURNAMENT_WEIGHT = 0.70   # 70% KenPom / 30% ESPN-based for tournament games

# ---------------------------------------------------------------------------
# Mismatch calibration constants
# ---------------------------------------------------------------------------
MISMATCH_THRESHOLD = 8.0    # adj_em gap where raw coefficients start underestimating
MISMATCH_SCALE     = 0.35   # boost per point above threshold
MISMATCH_CAP       = 8.0    # max additional points from mismatch boost


def apply_base_strength(game_frame: pd.DataFrame) -> pd.DataFrame:
    out = game_frame.copy()

    is_ncaa_tourney = out.get(
        "is_ncaa_tournament", pd.Series(False, index=out.index)
    ).astype(bool)

    # ── Primary efficiency margin signals ─────────────────────────────────────
    out["adj_em_margin"] = pd.to_numeric(out.get("adj_em_margin_l12"), errors="coerce")
    out["efg_margin"] = pd.to_numeric(out.get("efg_margin_l5"), errors="coerce")
    out["to_margin"] = pd.to_numeric(out.get("to_margin_l5"), errors="coerce")
    out["oreb_margin"] = pd.to_numeric(out.get("oreb_margin_l5"), errors="coerce")
    out["ftr_margin"] = pd.to_numeric(out.get("ftr_margin_l5"), errors="coerce")
    out["ft_scoring_pressure_margin"] = pd.to_numeric(
        out.get("ft_scoring_pressure_margin_l5"), errors="coerce"
    )

    # ── Season-long talent anchor (ESPN-based) ─────────────────────────────────
    # Blends the rolling-12 adj_em with the season-expanding average.
    # For tournament games use a very high season weight so that a Big12/ACC team
    # that stumbled in its last 12 regular-season games still shows its true
    # strength relative to a mid-major conference champion.
    adj_em_season = pd.to_numeric(out.get("adj_em_margin_season"), errors="coerce")
    adj_em_l12 = out["adj_em_margin"].fillna(0.0)
    season_weight = np.where(is_ncaa_tourney, SEASON_ANCHOR_WEIGHT_TOURNAMENT, SEASON_ANCHOR_WEIGHT_DEFAULT)
    has_season = adj_em_season.notna()
    espn_blended_adj_em = np.where(
        has_season,
        (1.0 - season_weight) * adj_em_l12 + season_weight * adj_em_season.fillna(adj_em_l12),
        adj_em_l12,
    )

    # ── KenPom absolute talent blend (NCAA tournament only) ───────────────────
    # When KenPom AdjEM margin is available for tournament games, blend it with
    # the ESPN-based estimate.  KenPom is calibrated against the full D-I field
    # and correctly distinguishes Duke (+37) from Iowa State (+31) from Tennessee
    # State (-4) — differences the rolling ESPN adj_net_rtg cannot capture.
    kenpom_margin = pd.to_numeric(out.get("kenpom_adj_em_margin"), errors="coerce")
    has_kenpom = kenpom_margin.notna() & is_ncaa_tourney

    blended_adj_em = np.where(
        has_kenpom,
        (1.0 - KENPOM_TOURNAMENT_WEIGHT) * espn_blended_adj_em
        + KENPOM_TOURNAMENT_WEIGHT * kenpom_margin.fillna(espn_blended_adj_em),
        espn_blended_adj_em,
    )
    out["adj_em_blended"] = blended_adj_em
    out["kenpom_used"] = has_kenpom

    blended_s = pd.Series(blended_adj_em, index=out.index)

    # ── Model A (efficiency-heavy) ─────────────────────────────────────────────
    scale_factor = 100.0
    out["model_a_margin"] = (
        MODEL_A_WEIGHTS["adj_em_margin"] * blended_s
        + scale_factor * MODEL_A_WEIGHTS["efg_margin"] * out["efg_margin"].fillna(0.0)
        + scale_factor * MODEL_A_WEIGHTS["to_margin"] * out["to_margin"].fillna(0.0)
        + scale_factor * MODEL_A_WEIGHTS["oreb_margin"] * out["oreb_margin"].fillna(0.0)
        + scale_factor * MODEL_A_WEIGHTS["ftr_margin"] * out["ftr_margin"].fillna(0.0)
    )

    # ── Model B (balanced) ────────────────────────────────────────────────────
    out["model_b_margin"] = (
        MODEL_B_WEIGHTS["adj_em_margin"] * blended_s
        + scale_factor * MODEL_B_WEIGHTS["efg_margin"] * out["efg_margin"].fillna(0.0)
        + scale_factor * MODEL_B_WEIGHTS["to_margin"] * out["to_margin"].fillna(0.0)
        + scale_factor * MODEL_B_WEIGHTS["oreb_margin"] * out["oreb_margin"].fillna(0.0)
        + scale_factor * MODEL_B_WEIGHTS["ft_scoring_pressure_margin"] * out["ft_scoring_pressure_margin"].fillna(0.0)
    )

    out["model_a_margin"] = out["model_a_margin"].clip(-40, 40)
    out["model_b_margin"] = out["model_b_margin"].clip(-40, 40)
    out["base_margin_blend"] = 0.5 * out["model_a_margin"] + 0.5 * out["model_b_margin"]

    # ── Mismatch calibration boost ─────────────────────────────────────────────
    # The blended adj_em coefficients (0.58/0.45) under-predict when the talent
    # gap is large.  Boost proportionally beyond the threshold.
    excess = (blended_s.abs() - MISMATCH_THRESHOLD).clip(lower=0.0)
    mismatch_boost = np.sign(blended_s) * (MISMATCH_SCALE * excess).clip(upper=MISMATCH_CAP)
    out["mismatch_boost"] = mismatch_boost
    out["base_margin_blend"] = (out["base_margin_blend"] + mismatch_boost).clip(-45, 45)

    # Stability proxy: higher when model variants agree.
    out["base_model_stability"] = (
        1.0 - (out["model_a_margin"] - out["model_b_margin"]).abs().clip(0, 20) / 20.0
    )
    out["base_model_stability"] = out["base_model_stability"].fillna(0.5)

    out["base_model_side"] = np.where(
        out["base_margin_blend"] > 0, "HOME",
        np.where(out["base_margin_blend"] < 0, "AWAY", "NEUTRAL")
    )
    return out
