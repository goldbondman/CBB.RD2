import pandas as pd

from ides_of_march.layer6_decision import apply_decision_layer


def test_decision_layer_probability_ranges_and_filter_behavior():
    frame = pd.DataFrame(
        [
            {
                "projected_margin_pre_mc": 4.0,
                "market_spread": -2.0,
                "market_total": 142.0,
                "base_model_stability": 0.8,
                "situational_confidence_boost": 4.0,
                "mc_volatility": 10.0,
                "mc_home_cover_prob": 0.58,
                "mc_filter_pass": True,
            },
            {
                "projected_margin_pre_mc": -3.0,
                "market_spread": 1.0,
                "market_total": 137.0,
                "base_model_stability": 0.9,
                "situational_confidence_boost": 3.0,
                "mc_volatility": 15.0,
                "mc_home_cover_prob": 0.42,
                "mc_filter_pass": False,
            },
        ]
    )

    out = apply_decision_layer(frame, direct_win_model=None, mc_mode="confidence_filter")

    assert ((out["win_prob_home"] >= 0) & (out["win_prob_home"] <= 1)).all()
    assert ((out["ats_cover_prob_home"] >= 0) & (out["ats_cover_prob_home"] <= 1)).all()
    assert ((out["confidence_score"] >= 0) & (out["confidence_score"] <= 100)).all()
    assert out.loc[1, "bet_recommendation"] == "PASS"
