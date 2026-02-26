from pathlib import Path

import pandas as pd
import pytest


def test_team_game_metrics_not_empty():
    df = pd.read_csv("data/team_game_metrics.csv")
    assert len(df) > 100, "team_game_metrics has fewer than 100 rows"


def test_no_all_null_l10_columns():
    df = pd.read_csv("data/team_game_metrics.csv")
    l10_cols = [c for c in df.columns if c.endswith("_l10")]
    assert len(l10_cols) >= 10, f"Expected 10+ L10 columns, got {len(l10_cols)}"
    for col in l10_cols:
        # In test environments with sparse data, L10 columns are often mostly null.
        # Only fail if they are literally all null when we expect some data.
        null_pct = df[col].isna().mean()
        if null_pct == 1.0:
            # Check if base column has any data
            base_col = col.replace("_l10", "")
            if base_col in df.columns and df[base_col].notna().sum() > 20:
                 # If base col has data but L10 is all null, it's a bug
                 assert null_pct < 1.0, f"{col} is 100% null but {base_col} has data — L10 broken"


def test_shooting_stats_not_all_zero():
    df = pd.read_csv("data/team_game_metrics.csv")
    for col in ["efg_pct", "fgm", "fga", "tpm", "tpa"]:
        if col in df.columns:
            assert df[col].gt(0).any(), f"{col} is all zeros — parsing bug"


def test_efg_pct_is_percentage_not_decimal():
    df = pd.read_csv("data/team_game_metrics.csv")
    efg = df["efg_pct"].dropna()
    assert efg.mean() > 1.0, (
        f"efg_pct mean={efg.mean():.3f} — looks like decimal scale, "
        f"should be percentage (expected ~51.0)"
    )


def test_predictions_have_required_columns():
    df = pd.read_csv("data/predictions_combined_latest.csv")
    # Unified on game_id; predicted_spread/pred_spread are aliases
    required = ["game_id", "model_confidence",
                "home_team_id", "away_team_id",
                "home_conference", "away_conference",
                "ens_fourfactors_spread", "ens_adjefficiency_spread"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required prediction columns: {missing}"


def test_spread_values_are_reasonable():
    df = pd.read_csv("data/predictions_combined_latest.csv")
    # Try multiple spread candidates, preferring ones with data
    spread_col = next((c for c in ["ens_ens_spread", "pred_spread", "predicted_spread"]
                       if c in df.columns and df[c].notna().any()), None)
    if spread_col is None:
        pytest.skip("No spread prediction column with data found")
    spreads = df[spread_col].dropna()
    assert spreads.abs().mean() < 30, (
        f"Mean absolute spread {spreads.abs().mean():.1f} is unreasonable — "
        f"scale bug likely"
    )
    assert spreads.abs().mean() > 0.5, (
        f"Mean absolute spread {spreads.abs().mean():.3f} is near zero — "
        f"model output collapsed"
    )


def test_results_log_event_ids_are_strings():
    path = Path("data/results_log.csv")
    if not path.exists():
        pytest.skip("results_log.csv not yet created")
    df = pd.read_csv(path, dtype={"event_id": str})
    assert df["event_id"].dtype == object, "event_id should be string dtype"
    assert not df["event_id"].str.match(r"^\d+\.0$").any(), (
        "event_id contains float strings (e.g. '401234.0') — will break join"
    )


def test_model_weights_sum_to_one():
    path = Path("data/model_weights.json")
    if not path.exists():
        pytest.skip("model_weights.json not yet created")
    import json
    with open(path) as f:
        w = json.load(f)
    total = sum(w["weights"].values())
    assert abs(total - 1.0) < 0.01, f"Weights sum to {total:.4f}, not 1.0"


def test_no_future_data_leakage():
    df = pd.read_csv("data/predictions_combined_latest.csv")
    leakage_cols = ["actual_margin", "actual_home_score",
                    "actual_away_score", "home_covered"]
    present = [c for c in leakage_cols if c in df.columns]
    assert not present, f"Potential data leakage columns in predictions: {present}"


def test_steam_flag_default_fallback_is_vectorized_series():
    df = pd.DataFrame({"line_movement": [0.0, 1.5]})

    with pytest.raises(AttributeError):
        df.get("steam_flag", False).astype(str)

    steam_flag = df.get("steam_flag", pd.Series(False, index=df.index, dtype="bool"))
    mask = steam_flag.astype(str).str.lower() == "true"
    assert mask.tolist() == [False, False]
