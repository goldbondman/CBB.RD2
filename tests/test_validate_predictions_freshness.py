from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from infra.ci.validate_predictions_freshness import select_prediction_source, validate, validate_file


def _base_now() -> datetime:
    return datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)


def test_validator_passes_with_real_schema_game_datetime_utc() -> None:
    df = pd.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "game_datetime_utc": ["2026-03-01T20:45Z", "2026-03-01T22:15Z"],
            "home_team": ["A", "B"],
            "away_team": ["C", "D"],
            "predicted_at_utc": ["2026-03-01T17:00:00Z", "2026-03-01T17:00:00Z"],
        }
    )

    summary = validate(
        df,
        selected_file="data/predictions_latest.csv",
        now_utc=_base_now(),
        timezone_local="America/Los_Angeles",
        max_hours=6,
    )

    assert summary.result == "PASS"
    assert summary.time_column_used == "game_datetime_utc"
    assert summary.parseable_rate == 1.0


def test_validator_passes_with_canonical_game_time_utc() -> None:
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "generated_at_utc": ["2026-03-01T17:10:00Z", "2026-03-01T17:10:00Z"],
            "game_time_utc": ["2026-03-01T20:45:00Z", "2026-03-01T21:45:00+00:00"],
        }
    )

    summary = validate(
        df,
        selected_file="data/predictions_latest.csv",
        now_utc=_base_now(),
        timezone_local="America/Los_Angeles",
        max_hours=6,
    )

    assert summary.result == "PASS"
    assert summary.time_column_used == "game_time_utc"


def test_validator_missing_time_column_diagnostics(capsys: pytest.CaptureFixture[str]) -> None:
    df = pd.DataFrame(
        {
            "event_id": ["e1"],
            "generated_at_utc": ["2026-03-01T17:10:00Z"],
            "home_team": ["A"],
        }
    )

    with pytest.raises(RuntimeError, match="No game time column found"):
        validate(
            df,
            selected_file="data/predictions_latest.csv",
            now_utc=_base_now(),
            timezone_local="America/Los_Angeles",
            max_hours=6,
        )

    out = capsys.readouterr().out
    assert "Columns found" in out
    assert "Detected time-like columns" in out


def test_best_column_selection_prefers_more_parseable_candidate() -> None:
    df = pd.DataFrame(
        {
            "generated_at_utc": ["2026-03-01T17:10:00Z"] * 10,
            "start_time": [
                "2026-03-01 20:45:00",
                "2026-03-01 20:45:00",
                "2026-03-01T20:45:00Z",
                "bad",
                "",
                "2026-03-01 21:00:00",
                "2026-03-01T22:00:00Z",
                "bad",
                "2026-03-01 23:00:00",
                "bad",
            ],
            "game_datetime_utc": [f"2026-03-01T{20 + (i % 3)}:45:00Z" for i in range(10)],
        }
    )

    summary = validate(
        df,
        selected_file="data/predictions_latest.csv",
        now_utc=_base_now(),
        timezone_local="America/Los_Angeles",
        max_hours=6,
    )

    assert summary.time_column_used == "game_datetime_utc"
    assert summary.parseable_rate == 1.0



def test_select_prediction_source_parity_with_build_derived(tmp_path, monkeypatch) -> None:
    import build_derived_csvs as bdc

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    latest = pd.DataFrame({"game_id": ["g1"], "game_time_utc": ["2026-03-01T18:00:00Z"]})
    mc_latest = pd.DataFrame(
        {"game_id": ["g2", "g3"], "game_time_utc": ["2026-03-01T20:00:00Z", "2026-03-01T19:00:00Z"]}
    )
    latest.to_csv(data_dir / "predictions_latest.csv", index=False)
    mc_latest.to_csv(data_dir / "predictions_mc_latest.csv", index=False)

    monkeypatch.chdir(tmp_path)

    df_a, label_a, path_a = select_prediction_source()
    df_b, label_b = bdc._select_prediction_source()

    assert label_a == label_b == "predictions_mc_latest"
    assert path_a.resolve() == (tmp_path / "data" / "predictions_mc_latest.csv").resolve()
    pd.testing.assert_frame_equal(df_a.reset_index(drop=True), df_b.reset_index(drop=True))


def test_validate_file_wrapper_pass(tmp_path) -> None:
    file_path = tmp_path / "predictions_latest.csv"
    pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "generated_at_utc": ["2026-03-01T16:30:00Z", "2026-03-01T16:30:00Z"],
            "game_time_utc": ["2026-03-01T19:00:00Z", "2026-03-01T20:00:00Z"],
        }
    ).to_csv(file_path, index=False)

    summary = validate_file(
        file_path,
        now_utc=_base_now(),
        timezone_local="America/Los_Angeles",
        max_hours=6,
    )

    assert summary.result == "PASS"
    assert summary.selected_file == str(file_path)
