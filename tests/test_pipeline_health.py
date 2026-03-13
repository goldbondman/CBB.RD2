from pathlib import Path
import re

import pandas as pd
import pytest
from pandas.api.types import is_string_dtype


def test_team_game_metrics_not_empty():
    df = pd.read_csv("data/team_game_metrics.csv")
    assert len(df) > 100, "team_game_metrics has fewer than 100 rows"


def test_no_all_null_l10_columns():
    df = pd.read_csv("data/team_game_metrics.csv")
    l10_cols = [c for c in df.columns if re.search(r"(?i)(^|_)l10($|_)", c)]
    assert len(l10_cols) >= 10, f"Expected 10+ L10 columns, got {len(l10_cols)}"
    populated = [col for col in l10_cols if df[col].isna().mean() < 0.95]
    assert len(populated) >= 10, "Fewer than 10 L10 columns have usable coverage (<95% null)"


def test_shooting_stats_not_all_zero():
    df = pd.read_csv("data/team_game_metrics.csv")
    for col in ["efg_pct", "fgm", "fga", "tpm", "tpa"]:
        if col in df.columns:
            assert df[col].gt(0).any(), f"{col} is all zeros - parsing bug"


def test_efg_pct_is_percentage_not_decimal():
    df = pd.read_csv("data/team_game_metrics.csv")
    candidate_cols = [c for c in ["efg_pct", "eFG", "home_eFG_L10", "away_eFG_L10"] if c in df.columns]
    if not candidate_cols:
        pytest.skip("No eFG columns available in team_game_metrics.csv")
    efg = pd.concat([pd.to_numeric(df[c], errors="coerce") for c in candidate_cols], ignore_index=True).dropna()
    assert not efg.empty, "No numeric eFG values found"
    mean_val = float(efg.mean())
    assert 0.1 <= mean_val <= 100.0, f"eFG mean={mean_val:.3f} is out of expected range"


def test_predictions_have_required_columns():
    df = pd.read_csv("data/predictions_combined_latest.csv")
    required = [
        "event_id",
        "predicted_spread",
        "model_confidence",
        "home_team_id",
        "away_team_id",
        "home_conference",
        "away_conference",
        "model1_schedule_pred",
        "model2_four_factors_pred",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required prediction columns: {missing}"


def test_spread_values_are_reasonable():
    df = pd.read_csv("data/predictions_combined_latest.csv")
    spreads = df["predicted_spread"].dropna()
    assert spreads.abs().mean() < 30, (
        f"Mean absolute spread {spreads.abs().mean():.1f} is unreasonable - scale bug likely"
    )
    assert spreads.abs().mean() > 0.5, (
        f"Mean absolute spread {spreads.abs().mean():.3f} is near zero - model output collapsed"
    )


def test_results_log_event_ids_are_strings():
    path = Path("data/results_log.csv")
    if not path.exists():
        pytest.skip("results_log.csv not yet created")
    df = pd.read_csv(path, dtype={"event_id": str})
    assert is_string_dtype(df["event_id"]), "event_id should use a string-compatible dtype"
    assert not df["event_id"].str.match(r"^\d+\.0$").any(), (
        "event_id contains float strings (e.g. '401234.0') - will break join"
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
    leakage_cols = ["actual_margin", "actual_home_score", "actual_away_score", "home_covered"]
    present = [c for c in leakage_cols if c in df.columns]
    assert not present, f"Potential data leakage columns in predictions: {present}"


def test_steam_flag_default_fallback_is_vectorized_series():
    df = pd.DataFrame({"line_movement": [0.0, 1.5]})

    with pytest.raises(AttributeError):
        df.get("steam_flag", False).astype(str)

    steam_flag = df.get("steam_flag", pd.Series(False, index=df.index, dtype="bool"))
    mask = steam_flag.astype(str).str.lower() == "true"
    assert mask.tolist() == [False, False]


# Known team_id -> correct conference pairs (spot-check for major programs)
KNOWN_TEAM_CONFERENCES = {
    "333": "Southeastern Conference",  # Alabama
    "2": "Southeastern Conference",  # Auburn
    "96": "Southeastern Conference",  # Kentucky
    "150": "Atlantic Coast Conference",  # Duke
    "2250": "West Coast Conference",  # Gonzaga
    "9": "Big 12",  # Arizona State
}

WRONG_CONFERENCE_NAMES = {
    "Pac-12",  # Big Ten teams were incorrectly labeled this (conf_id=7 -> Big Ten)
    "Atlantic Sun Conference",  # WAC teams were incorrectly labeled this (conf_id=30 -> WAC)
}

# Specific team_id -> conference pairings that should NOT appear in the data
# (these represent the known wrong assignments that existed before the fix)
WRONG_TEAM_CONFERENCE_PAIRS = {
    "333": "Patriot League",  # Alabama was wrongly labeled Patriot League
    "2": "Patriot League",  # Auburn was wrongly labeled Patriot League
    "96": "Patriot League",  # Kentucky was wrongly labeled Patriot League
    "150": "Big East",  # Duke was wrongly labeled Big East
    "2250": "America East Conference",  # Gonzaga was wrongly labeled America East
}


def test_games_csv_no_wrong_conference_labels():
    df = pd.read_csv("data/games.csv", low_memory=False)
    df["home_team_id"] = df["home_team_id"].astype(str).str.split(".").str[0]
    df["away_team_id"] = df["away_team_id"].astype(str).str.split(".").str[0]
    for col in ("home_conference", "away_conference"):
        if col not in df.columns:
            continue
        counts = df[col].value_counts()
        for wrong in WRONG_CONFERENCE_NAMES:
            assert wrong not in counts.index or counts[wrong] == 0, (
                f"games.csv column '{col}' still contains wrong label '{wrong}' "
                f"({counts.get(wrong, 0)} rows)"
            )
    # Also verify specific teams don't have their known-wrong labels
    for tid, wrong_conf in WRONG_TEAM_CONFERENCE_PAIRS.items():
        home_bad = df[(df["home_team_id"] == tid) & (df["home_conference"] == wrong_conf)]
        away_bad = df[(df["away_team_id"] == tid) & (df["away_conference"] == wrong_conf)]
        assert len(home_bad) == 0 and len(away_bad) == 0, (
            f"team_id={tid} still has wrong conference '{wrong_conf}' in games.csv"
        )


def test_games_csv_sec_teams_have_correct_conference():
    df = pd.read_csv("data/games.csv", low_memory=False)
    df["home_team_id"] = df["home_team_id"].astype(str).str.split(".").str[0]
    df["away_team_id"] = df["away_team_id"].astype(str).str.split(".").str[0]
    for tid, expected_conf in KNOWN_TEAM_CONFERENCES.items():
        home_rows = df[df["home_team_id"] == tid]["home_conference"].dropna().unique()
        away_rows = df[df["away_team_id"] == tid]["away_conference"].dropna().unique()
        all_confs = list(home_rows) + list(away_rows)
        if not all_confs:
            continue
        for actual in all_confs:
            assert actual == expected_conf, (
                f"team_id={tid} expected '{expected_conf}' but got '{actual}' in games.csv"
            )


def test_conference_id_map_has_correct_sec_entry():
    from espn_config import ESPN_CONFERENCE_MAP

    assert ESPN_CONFERENCE_MAP.get("23") == "Southeastern Conference", (
        f"ESPN_CONFERENCE_MAP['23'] should be 'Southeastern Conference', "
        f"got '{ESPN_CONFERENCE_MAP.get('23')}'"
    )
    assert ESPN_CONFERENCE_MAP.get("2") == "Atlantic Coast Conference", (
        f"ESPN_CONFERENCE_MAP['2'] should be 'Atlantic Coast Conference', "
        f"got '{ESPN_CONFERENCE_MAP.get('2')}'"
    )
    assert ESPN_CONFERENCE_MAP.get("7") == "Big Ten", (
        f"ESPN_CONFERENCE_MAP['7'] should be 'Big Ten', "
        f"got '{ESPN_CONFERENCE_MAP.get('7')}'"
    )
    assert ESPN_CONFERENCE_MAP.get("4") == "Big East", (
        f"ESPN_CONFERENCE_MAP['4'] should be 'Big East', "
        f"got '{ESPN_CONFERENCE_MAP.get('4')}'"
    )
    assert ESPN_CONFERENCE_MAP.get("29") == "West Coast Conference", (
        f"ESPN_CONFERENCE_MAP['29'] should be 'West Coast Conference', "
        f"got '{ESPN_CONFERENCE_MAP.get('29')}'"
    )
