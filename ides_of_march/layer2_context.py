from __future__ import annotations

import numpy as np
import pandas as pd


def apply_context_adjustments(game_frame: pd.DataFrame) -> pd.DataFrame:
    out = game_frame.copy()

    is_neutral = out.get("is_neutral", pd.Series(False, index=out.index)).astype(bool)
    is_conf_tourney = out.get("is_conference_tournament", pd.Series(False, index=out.index)).astype(bool)
    is_ncaa_tourney = out.get("is_ncaa_tournament", pd.Series(False, index=out.index)).astype(bool)
    home_bonus_eligible = out.get(
        "home_bonus_eligible",
        ((~is_neutral) & (~is_conf_tourney) & (~is_ncaa_tourney)),
    ).astype(bool)
    hca = np.where(home_bonus_eligible, 2.7, 0.0)

    form_diff = pd.to_numeric(out.get("form_delta_diff"), errors="coerce").fillna(0.0)
    sos_diff = pd.to_numeric(out.get("sos_diff"), errors="coerce").fillna(0.0)
    rest_diff = pd.to_numeric(out.get("rest_diff"), errors="coerce").fillna(0.0)

    out["context_hca"] = hca
    out["context_form"] = 0.22 * form_diff
    out["context_sos"] = 0.12 * sos_diff
    out["context_rest"] = 0.15 * rest_diff

    out["context_adjustment"] = (
        out["context_hca"]
        + out["context_form"]
        + out["context_sos"]
        + out["context_rest"]
    ).clip(-6.0, 6.0)

    out["margin_ctx_a"] = pd.to_numeric(out.get("model_a_margin"), errors="coerce") + out["context_adjustment"]
    out["margin_ctx_b"] = pd.to_numeric(out.get("model_b_margin"), errors="coerce") + out["context_adjustment"]
    out["margin_ctx_blend"] = pd.to_numeric(out.get("base_margin_blend"), errors="coerce") + out["context_adjustment"]

    pace = pd.to_numeric(out.get("expected_pace"), errors="coerce").fillna(68.0)
    total_from_ratings = pd.to_numeric(out.get("expected_total"), errors="coerce")
    total_from_market = pd.to_numeric(out.get("market_total"), errors="coerce")

    out["projected_total_ctx"] = total_from_ratings
    out["projected_total_ctx"] = out["projected_total_ctx"].where(out["projected_total_ctx"].notna(), 136.0 + 0.35 * (pace - 68.0))
    out["projected_total_ctx"] = out["projected_total_ctx"].where(
        out["projected_total_ctx"].notna(),
        total_from_market,
    )
    out["projected_total_ctx"] = out["projected_total_ctx"].clip(110.0, 180.0)

    out["context_summary"] = (
        "hca=" + out["context_hca"].round(2).astype(str)
        + "|home_ok=" + home_bonus_eligible.astype(int).astype(str)
        + "|post=" + (is_conf_tourney | is_ncaa_tourney).astype(int).astype(str)
        + "|rest=" + out["context_rest"].round(2).astype(str)
        + "|form=" + out["context_form"].round(2).astype(str)
        + "|sos=" + out["context_sos"].round(2).astype(str)
    )
    return out
