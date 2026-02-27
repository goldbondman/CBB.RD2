import pandas as pd
import pytest

from cbb_backtester import (
    _normalize_selected_models,
    _normalize_weight_overrides,
    optimize_weights,
)


def test_normalize_selected_models_accepts_case_insensitive_subset():
    models = _normalize_selected_models(["fourfactors", "Variance"])
    assert models == ["FourFactors", "Variance"]


def test_normalize_selected_models_rejects_unknown_names():
    with pytest.raises(ValueError, match="Unknown model"):
        _normalize_selected_models(["FourFactors", "notamodel"])


def test_normalize_weight_overrides_normalizes_and_expands_missing_selected():
    weights = _normalize_weight_overrides(
        {"fourfactors": 3, "variance": 1},
        active_models=["FourFactors", "Variance", "Situational"],
    )
    assert weights == {"FourFactors": 0.75, "Variance": 0.25, "Situational": 0.0}


def test_normalize_weight_overrides_rejects_unknown_or_negative_weights():
    with pytest.raises(ValueError, match="unknown/non-selected"):
        _normalize_weight_overrides(
            {"fourfactors": 1, "unknown": 1},
            active_models=["FourFactors"],
        )

    with pytest.raises(ValueError, match=">= 0"):
        _normalize_weight_overrides(
            {"fourfactors": -1},
            active_models=["FourFactors"],
        )


def test_optimize_weights_supports_single_model_subset():
    records = pd.DataFrame(
        {
            "market_spread": [-5.0, -2.0, 1.5, 3.0],
            "actual_margin": [7.0, -1.0, 0.0, -4.0],
            "fourfactors_spread": [-6.0, -1.0, 1.0, 2.0],
        }
    )
    weights, metric = optimize_weights(records, model_names=["FourFactors"], metric="ats")
    assert set(weights.keys()) == {"FourFactors"}
    assert pytest.approx(weights["FourFactors"], rel=1e-6) == 1.0
    assert isinstance(metric, float)
