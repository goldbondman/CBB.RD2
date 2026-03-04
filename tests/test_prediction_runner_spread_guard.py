from espn_prediction_runner import _resolve_predicted_spread


def test_resolve_predicted_spread_uses_model_value_when_finite():
    prediction = {"predicted_spread": -4.27, "home_net_eff": 12.0, "away_net_eff": 8.0}
    home_ctx = {"adj_net_rtg": 10.0}
    away_ctx = {"adj_net_rtg": 7.0}

    out = _resolve_predicted_spread(prediction, home_ctx, away_ctx, neutral_site=False)

    assert out == -4.27


def test_resolve_predicted_spread_falls_back_to_net_eff_when_nan():
    prediction = {"predicted_spread": float("nan"), "home_net_eff": 18.0, "away_net_eff": 10.0}
    home_ctx = {}
    away_ctx = {}

    out = _resolve_predicted_spread(prediction, home_ctx, away_ctx, neutral_site=False)

    # -(home-away+hca) => -(8+1.0) = -9.0
    assert out == -9.0


def test_resolve_predicted_spread_returns_none_when_no_inputs_available():
    prediction = {"predicted_spread": float("nan")}
    home_ctx = {"adj_net_rtg": None, "net_eff": None, "net_rtg": None}
    away_ctx = {"adj_net_rtg": None, "net_eff": None, "net_rtg": None}

    out = _resolve_predicted_spread(prediction, home_ctx, away_ctx, neutral_site=True)

    assert out is None
