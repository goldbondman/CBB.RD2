import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from cbb_output_schemas import (
    OUTPUT_FILE_SCHEMAS,
    completeness_report,
    validate_output,
)


# ── validate_output ──────────────────────────────────────────────────────────

def test_validate_output_passes_when_all_columns_present():
    cols = OUTPUT_FILE_SCHEMAS["games"]
    df = pd.DataFrame([{c: "x" for c in cols}])
    missing = validate_output(df, "games")
    assert missing == []


def test_validate_output_returns_missing_columns():
    df = pd.DataFrame([{"game_id": "1"}])
    missing = validate_output(df, "games")
    assert "home_team" in missing
    assert "away_team" in missing


def test_validate_output_strict_raises_on_missing():
    df = pd.DataFrame([{"game_id": "1"}])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_output(df, "games", strict=True)


def test_validate_output_strict_passes_when_complete():
    cols = OUTPUT_FILE_SCHEMAS["games"]
    df = pd.DataFrame([{c: "x" for c in cols}])
    validate_output(df, "games", strict=True)


def test_validate_output_unknown_schema_raises():
    df = pd.DataFrame()
    with pytest.raises(KeyError, match="Unknown output schema"):
        validate_output(df, "nonexistent_schema")


# ── team_game_logs schema ────────────────────────────────────────────────────

def test_team_game_logs_schema_includes_alias_columns():
    schema = OUTPUT_FILE_SCHEMAS["team_game_logs"]
    for alias in ["FGA", "FGM", "FTA", "FTM", "TPA", "TPM", "ORB", "DRB", "RB", "TO", "AST"]:
        assert alias in schema, f"Alias column {alias} missing from team_game_logs schema"


def test_player_game_logs_schema_includes_alias_columns():
    schema = OUTPUT_FILE_SCHEMAS["player_game_logs"]
    for alias in ["FGA", "FGM", "FTA", "FTM", "TPA", "TPM", "ORB", "DRB", "RB", "TO", "AST"]:
        assert alias in schema, f"Alias column {alias} missing from player_game_logs schema"


# ── predictions schema ───────────────────────────────────────────────────────

def test_predictions_schema_includes_team_metadata():
    schema = OUTPUT_FILE_SCHEMAS["predictions"]
    for col in ["home_conference", "home_wins", "home_losses",
                "away_conference", "away_wins", "away_losses"]:
        assert col in schema


def test_predictions_schema_includes_box_score_fields():
    schema = OUTPUT_FILE_SCHEMAS["predictions"]
    for prefix in ["home_", "away_"]:
        for stat in ["FGA", "FGM", "FTA", "FTM", "TPA", "TPM",
                     "ORB", "DRB", "RB", "TO", "AST"]:
            assert f"{prefix}{stat}" in schema


# ── completeness_report ──────────────────────────────────────────────────────

def test_completeness_report_identifies_complete_output():
    cols = OUTPUT_FILE_SCHEMAS["games"]
    df = pd.DataFrame([{c: "val" for c in cols}])
    report = completeness_report({"games": df})

    assert len(report) == 1
    row = report.iloc[0]
    assert row["output"] == "games"
    assert row["missing_cols"] == 0
    assert row["missing_list"] == ""
    assert row["null_pct"] == 0.0


def test_completeness_report_identifies_missing_columns():
    df = pd.DataFrame([{"game_id": "1"}])
    report = completeness_report({"games": df})

    row = report.iloc[0]
    assert row["missing_cols"] > 0
    assert "home_team" in row["missing_list"]


def test_completeness_report_reports_null_percentage():
    cols = OUTPUT_FILE_SCHEMAS["games"]
    data = {c: [None] for c in cols}
    df = pd.DataFrame(data)
    report = completeness_report({"games": df})

    row = report.iloc[0]
    assert row["null_pct"] == 100.0


def test_completeness_report_ignores_unknown_schemas():
    df = pd.DataFrame([{"a": 1}])
    report = completeness_report({"unknown_file": df})
    assert len(report) == 0


def test_completeness_report_handles_empty_dataframe():
    cols = OUTPUT_FILE_SCHEMAS["games"]
    df = pd.DataFrame(columns=cols)
    report = completeness_report({"games": df})

    row = report.iloc[0]
    assert row["rows"] == 0
    assert row["null_pct"] is None
