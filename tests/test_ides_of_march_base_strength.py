import pandas as pd

from ides_of_march.layer1_base_strength import apply_base_strength


def test_model_b_formula_uses_configured_weights():
    frame = pd.DataFrame(
        [
            {
                "adj_em_margin_l12": 5.0,
                "efg_margin_l5": 0.05,
                "to_margin_l5": 0.03,
                "oreb_margin_l5": 0.02,
                "ftr_margin_l5": 0.04,
                "ft_scoring_pressure_margin_l5": 0.01,
            }
        ]
    )

    out = apply_base_strength(frame)
    expected = 0.45 * 5.0 + 100.0 * (0.22 * 0.05 + 0.14 * 0.03 + 0.11 * 0.02 + 0.08 * 0.01)
    assert round(float(out.loc[0, "model_b_margin"]), 6) == round(expected, 6)


def test_base_model_stability_drops_when_models_disagree():
    frame = pd.DataFrame(
        [
            {
                "adj_em_margin_l12": 12.0,
                "efg_margin_l5": 0.10,
                "to_margin_l5": -0.08,
                "oreb_margin_l5": 0.12,
                "ftr_margin_l5": -0.04,
                "ft_scoring_pressure_margin_l5": 0.06,
            }
        ]
    )
    out = apply_base_strength(frame)
    stability = float(out.loc[0, "base_model_stability"])
    assert 0.0 <= stability <= 1.0
