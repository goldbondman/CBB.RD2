from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Seed-model blend constants (NCAA tournament only)
# ---------------------------------------------------------------------------
# For tournament games where seeds are available, we blend the adj_em model
# output with a seed-calibrated estimate.  Seed differentials predict
# tournament spreads more accurately than per-game rolling efficiency for
# extreme mismatches (elite programs that lost a few Big12/ACC games vs
# conference champions from weak leagues).
#
# seed_model_margin = SEED_MARGIN_COEFF × seed_differential
#   seed_diff > 0 → home team is the lower (better) seed
# seed_weight ramps from 0 (equal seeds) to SEED_MAX_WEIGHT (diff = 15)
#   so for a 1 vs 16 matchup the final margin is ~85% seed-model.
#
# Calibrated so that:
#   1 vs 16 (diff=15): seed_model=33  → blended ≈ 28-30 (market 25-35)
#   2 vs 15 (diff=13): seed_model=28  → blended ≈ 22-24 (market 20-25)
#   1 vs  8 (diff= 7): seed_model=15  → blended ≈ 11-13 (market  7-12)
#   5 vs 12 (diff= 7): seed_model=15  → blended ≈  8-10 (market  3- 8)
#   4 vs  5 (diff= 1): seed_model= 2  → barely changes adj_em model
SEED_MARGIN_COEFF = 2.2
SEED_MAX_WEIGHT   = 0.85


def apply_context_adjustments(game_frame: pd.DataFrame) -> pd.DataFrame:
    out = game_frame.copy()

    is_neutral = out.get("is_neutral", pd.Series(False, index=out.index)).astype(bool)
    is_conf_tourney = out.get("is_conference_tournament", pd.Series(False, index=out.index)).astype(bool)
    is_ncaa_tourney = out.get("is_ncaa_tournament", pd.Series(False, index=out.index)).astype(bool)
    is_postseason = is_conf_tourney | is_ncaa_tourney

    home_bonus_eligible = out.get(
        "home_bonus_eligible",
        ((~is_neutral) & (~is_postseason)),
    ).astype(bool)
    hca = np.where(home_bonus_eligible, 2.7, 0.0)

    form_diff = pd.to_numeric(out.get("form_delta_diff"), errors="coerce").fillna(0.0)
    sos_diff  = pd.to_numeric(out.get("sos_diff"),        errors="coerce").fillna(0.0)
    rest_diff = pd.to_numeric(out.get("rest_diff"),       errors="coerce").fillna(0.0)

    out["context_hca"]  = hca
    out["context_form"] = 0.22 * form_diff
    out["context_rest"] = 0.15 * rest_diff

    # SOS weight raised for postseason — schedule quality divergence is widest
    # at tournament time (Power-6 teams vs single-bid conference champions).
    sos_weight = np.where(is_postseason, 0.18, 0.12)
    out["context_sos"] = sos_weight * sos_diff

    # ── Soft context adjustment (non-seed) ────────────────────────────────────
    context_cap = np.where(is_postseason, 8.0, 6.0)
    raw_context = (
        out["context_hca"]
        + out["context_form"]
        + out["context_sos"]
        + out["context_rest"]
    )
    out["context_adjustment"] = raw_context.clip(-context_cap, context_cap)

    out["margin_ctx_a"]     = pd.to_numeric(out.get("model_a_margin"),    errors="coerce") + out["context_adjustment"]
    out["margin_ctx_b"]     = pd.to_numeric(out.get("model_b_margin"),    errors="coerce") + out["context_adjustment"]
    out["margin_ctx_blend"] = pd.to_numeric(out.get("base_margin_blend"), errors="coerce") + out["context_adjustment"]

    # ── Seed-model blend (NCAA tournament only) ────────────────────────────────
    # When seeds are available, blend the adj_em-derived margin toward a
    # seed-calibrated estimate.  The blend weight increases with seed
    # differential so that extreme mismatches (1 vs 16) are ~85% seed-driven
    # while close matchups (4 vs 5) remain ~97% adj_em-driven.
    home_seed = pd.to_numeric(out.get("home_seed"), errors="coerce")
    away_seed = pd.to_numeric(out.get("away_seed"), errors="coerce")
    seed_data_available = home_seed.notna() & away_seed.notna() & is_ncaa_tourney

    # seed_diff > 0  →  home team has lower (better) seed  →  home favoured
    seed_diff = (away_seed - home_seed).fillna(0.0)
    seed_margin_model = SEED_MARGIN_COEFF * seed_diff          # seed-based expected margin
    seed_diff_abs     = seed_diff.abs()
    seed_weight       = pd.Series(
        np.where(
            seed_data_available,
            (seed_diff_abs / 15.0 * SEED_MAX_WEIGHT).clip(0, SEED_MAX_WEIGHT),
            0.0,
        ),
        index=out.index,
    )

    model_margin  = out["margin_ctx_blend"].fillna(0.0)
    blended_margin = (1.0 - seed_weight) * model_margin + seed_weight * seed_margin_model

    out["context_seed_weight"] = seed_weight
    out["margin_ctx_blend"] = np.where(seed_data_available, blended_margin, model_margin)

    # ── Totals ─────────────────────────────────────────────────────────────────
    pace             = pd.to_numeric(out.get("expected_pace"),  errors="coerce").fillna(68.0)
    total_from_ratings = pd.to_numeric(out.get("expected_total"), errors="coerce")
    total_from_market  = pd.to_numeric(out.get("market_total"),   errors="coerce")

    out["projected_total_ctx"] = total_from_ratings
    out["projected_total_ctx"] = out["projected_total_ctx"].where(
        out["projected_total_ctx"].notna(), 136.0 + 0.35 * (pace - 68.0)
    )
    out["projected_total_ctx"] = out["projected_total_ctx"].where(
        out["projected_total_ctx"].notna(), total_from_market
    )
    out["projected_total_ctx"] = out["projected_total_ctx"].clip(110.0, 180.0)

    out["context_summary"] = (
        "hca="   + out["context_hca"].round(2).astype(str)
        + "|home_ok=" + home_bonus_eligible.astype(int).astype(str)
        + "|post="    + is_postseason.astype(int).astype(str)
        + "|rest="    + out["context_rest"].round(2).astype(str)
        + "|form="    + out["context_form"].round(2).astype(str)
        + "|sos="     + out["context_sos"].round(2).astype(str)
        + "|seed_w="  + seed_weight.round(2).astype(str)
    )
    return out
