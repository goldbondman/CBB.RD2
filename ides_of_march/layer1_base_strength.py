from __future__ import annotations

import numpy as np
import pandas as pd

from .config import MODEL_A_WEIGHTS, MODEL_B_WEIGHTS


def apply_base_strength(game_frame: pd.DataFrame) -> pd.DataFrame:
    out = game_frame.copy()

    out["adj_em_margin"] = pd.to_numeric(out.get("adj_em_margin_l12"), errors="coerce")
    out["efg_margin"] = pd.to_numeric(out.get("efg_margin_l5"), errors="coerce")
    out["to_margin"] = pd.to_numeric(out.get("to_margin_l5"), errors="coerce")
    out["oreb_margin"] = pd.to_numeric(out.get("oreb_margin_l5"), errors="coerce")
    out["ftr_margin"] = pd.to_numeric(out.get("ftr_margin_l5"), errors="coerce")
    out["ft_scoring_pressure_margin"] = pd.to_numeric(out.get("ft_scoring_pressure_margin_l5"), errors="coerce")

    scale_factor = 100.0
    out["model_a_margin"] = (
        MODEL_A_WEIGHTS["adj_em_margin"] * out["adj_em_margin"].fillna(0.0)
        + scale_factor * MODEL_A_WEIGHTS["efg_margin"] * out["efg_margin"].fillna(0.0)
        + scale_factor * MODEL_A_WEIGHTS["to_margin"] * out["to_margin"].fillna(0.0)
        + scale_factor * MODEL_A_WEIGHTS["oreb_margin"] * out["oreb_margin"].fillna(0.0)
        + scale_factor * MODEL_A_WEIGHTS["ftr_margin"] * out["ftr_margin"].fillna(0.0)
    )

    out["model_b_margin"] = (
        MODEL_B_WEIGHTS["adj_em_margin"] * out["adj_em_margin"].fillna(0.0)
        + scale_factor * MODEL_B_WEIGHTS["efg_margin"] * out["efg_margin"].fillna(0.0)
        + scale_factor * MODEL_B_WEIGHTS["to_margin"] * out["to_margin"].fillna(0.0)
        + scale_factor * MODEL_B_WEIGHTS["oreb_margin"] * out["oreb_margin"].fillna(0.0)
        + scale_factor * MODEL_B_WEIGHTS["ft_scoring_pressure_margin"] * out["ft_scoring_pressure_margin"].fillna(0.0)
    )

    out["model_a_margin"] = out["model_a_margin"].clip(-35, 35)
    out["model_b_margin"] = out["model_b_margin"].clip(-35, 35)
    out["base_margin_blend"] = 0.5 * out["model_a_margin"] + 0.5 * out["model_b_margin"]

    # Stability proxy: higher when model variants agree.
    out["base_model_stability"] = 1.0 - (out["model_a_margin"] - out["model_b_margin"]).abs().clip(0, 20) / 20.0
    out["base_model_stability"] = out["base_model_stability"].fillna(0.5)

    out["base_model_side"] = np.where(out["base_margin_blend"] > 0, "HOME", np.where(out["base_margin_blend"] < 0, "AWAY", "NEUTRAL"))
    return out
