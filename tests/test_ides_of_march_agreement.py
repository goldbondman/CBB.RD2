import pandas as pd

from ides_of_march.layer5_agreement import apply_agreement_layer, assign_agreement_bucket


def test_assign_agreement_bucket_expected_cases():
    assert assign_agreement_bucket(1, 0, 0) == "Base model only"
    assert assign_agreement_bucket(0, 1, 0) == "Situational signal only"
    assert assign_agreement_bucket(0, 0, -1) == "Monte Carlo signal only"
    assert assign_agreement_bucket(1, 1, 0) == "Base + Situational agree"
    assert assign_agreement_bucket(1, -1, 0) == "Base + Situational conflict"
    assert assign_agreement_bucket(-1, 0, -1) == "Base + Monte Carlo agree"
    assert assign_agreement_bucket(1, 1, 1) == "Base + Situational + Monte Carlo agree"


def test_apply_agreement_layer_adds_bucket_column():
    frame = pd.DataFrame(
        [
            {"edge_home": 2.0, "situational_signal": 1, "mc_signal": 1},
            {"edge_home": -1.8, "situational_signal": 0, "mc_signal": 1},
        ]
    )
    out = apply_agreement_layer(frame)

    assert "agreement_bucket" in out.columns
    assert out.loc[0, "agreement_bucket"] == "Base + Situational + Monte Carlo agree"
    assert out.loc[1, "agreement_bucket"] == "Base + Monte Carlo conflict"
