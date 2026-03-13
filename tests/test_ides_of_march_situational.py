import pandas as pd

from ides_of_march.layer3_situational import apply_situational_layer, discover_situational_rules


def test_discover_rules_enforces_min_sample_and_shrinkage():
    rows = []
    for i in range(60):
        rows.append(
            {
                "rest_diff": 2.5,
                "form_delta_diff": 1.2,
                "market_spread": -3.5,
                "actual_home_covered": bool(i < 38),
            }
        )
    hist = pd.DataFrame(rows)

    rulebook = discover_situational_rules(hist, min_sample=50, shrink_k=75.0)
    row = rulebook[rulebook["rule_id"] == "home_rest_form_stack"].iloc[0]

    assert int(row["sample_size"]) == 60
    assert bool(row["accepted"]) is True
    assert 0.5 < float(row["shrunk_ats_rate"]) < 0.7


def test_apply_situational_layer_sets_active_rules_and_adjustment():
    rulebook = pd.DataFrame(
        [
            {
                "rule_id": "home_rest_form_stack",
                "description": "Home rest + form stack",
                "direction": 1,
                "sample_size": 80,
                "raw_ats_rate": 0.62,
                "shrunk_ats_rate": 0.58,
                "effect": 0.08,
                "accepted": True,
            }
        ]
    )
    frame = pd.DataFrame(
        [
            {
                "rest_diff": 3.0,
                "form_delta_diff": 1.1,
                "market_spread": -2.0,
                "away_days_rest": 1,
                "home_orb_pct_l5": 0.30,
                "away_drb_pct_l5": 0.69,
                "away_three_par_l5": 0.35,
                "home_pace_l5": 68.5,
                "home_Form_Delta": 1.5,
                "home_sos_pre": -0.2,
            }
        ]
    )

    out = apply_situational_layer(frame, rulebook)
    assert out.loc[0, "situational_active_rules"] == "home_rest_form_stack"
    assert float(out.loc[0, "situational_spread_adjustment"]) > 0
