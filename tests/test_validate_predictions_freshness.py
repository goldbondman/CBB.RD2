from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from infra.ci.validate_predictions_freshness import (
    select_prediction_source,
    validate,
    validate_file,
    validate_predictions_freshness,
)


def _base_now() -> datetime:
    return datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "generated_at_utc": ["2026-03-01T17:10:00Z", "2026-03-01T17:10:00Z"],
            "game_datetime_utc": ["2026-03-01T20:45:00Z", "2026-03-01T21:45:00Z"],
        }
    )


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


def test_validate_prefers_game_datetime_utc_regression() -> None:
    now_utc = _base_now()
    df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "generated_at_utc": ["2026-03-01T17:30:00Z", "2026-03-01T17:30:00Z"],
            "game_datetime_utc": ["2026-03-01T20:45Z", "2026-03-01T21:45Z"],
            "start_time": ["2026-03-01 20:45:00", "2026-03-01 21:45:00"],
        }
    )

    summary = validate(
        df,
        selected_file="data/predictions_latest.csv",
        now_utc=now_utc,
        timezone_local="America/Los_Angeles",
        max_hours=6,
    )

    assert summary.result == "PASS"
    assert summary.time_column_used == "game_datetime_utc"
    assert summary.time_parseable_rate == 1.0


def test_validate_predictions_freshness_fail_stale() -> None:
    now_utc = _base_now()
    df = _base_frame()
    df["generated_at_utc"] = "2026-03-01T00:00:00Z"

    with pytest.raises(RuntimeError, match="stale predictions"):
        validate(
            df,
            selected_file="data/predictions_latest.csv",
            now_utc=now_utc,
            timezone_local="America/Los_Angeles",
            max_hours=6,
        )


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


def _write_predictions(path, columns: dict[str, list[str]]) -> None:
    df = pd.DataFrame(columns)
    df.to_csv(path, index=False)


def _freeze_now(monkeypatch, now: datetime) -> None:
    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz else now.replace(tzinfo=None)

    monkeypatch.setattr("infra.ci.validate_predictions_freshness.datetime", FrozenDateTime)


def test_validate_predictions_freshness_passes_for_recent_and_future_games(tmp_path, monkeypatch):
    now = datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc)
    _freeze_now(monkeypatch, now)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_predictions(
        data_dir / "predictions_latest.csv",
        {
            "game_id": ["g1", "g2"],
            "generated_at_utc": ["2026-03-03T22:30:00Z", "2026-03-03T22:30:00Z"],
            "game_datetime_utc": ["2026-03-04T01:00:00Z", "2026-03-05T00:00:00Z"],
        },
    )

    validate_predictions_freshness(data_dir, max_age_hours=6)


def test_validate_predictions_freshness_fails_for_stale_runtime_timestamp(tmp_path, monkeypatch):
    now = datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc)
    _freeze_now(monkeypatch, now)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_predictions(
        data_dir / "predictions_latest.csv",
        {
            "game_id": ["g1"],
            "generated_at_utc": ["2026-03-03T10:00:00Z"],
            "game_datetime_utc": ["2026-03-04T01:00:00Z"],
        },
    )

    with pytest.raises(RuntimeError, match="freshness gate failed"):
        validate_predictions_freshness(data_dir, max_age_hours=6)


def test_validate_predictions_freshness_fails_for_past_only_schedule(tmp_path, monkeypatch):
    now = datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc)
    _freeze_now(monkeypatch, now)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_predictions(
        data_dir / "predictions_latest.csv",
        {
            "game_id": ["g1"],
            "generated_at_utc": ["2026-03-03T22:40:00Z"],
            "game_datetime_utc": ["2026-03-03T12:00:00Z"],
        },
    )

    with pytest.raises(RuntimeError, match="forward-looking schedule gate failed"):
        validate_predictions_freshness(data_dir, max_age_hours=6)


def test_freshness_prefers_generated_at_over_predicted_at(tmp_path, monkeypatch):
    now = datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc)
    _freeze_now(monkeypatch, now)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_predictions(
        data_dir / "predictions_latest.csv",
        {
            "game_id": ["g1"],
            "generated_at_utc": ["2026-03-03T22:50:00Z"],
            "predicted_at_utc": ["2026-02-28T00:00:00Z"],
            "game_datetime_utc": ["2026-03-04T01:00:00Z"],
        },
    )

    validate_predictions_freshness(data_dir, max_age_hours=6)


def test_falls_back_to_file_mtime_when_runtime_columns_missing(tmp_path, monkeypatch, capsys):
    now = datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc)
    _freeze_now(monkeypatch, now)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    path = data_dir / "predictions_latest.csv"
    _write_predictions(
        path,
        {
            "game_id": ["g1"],
            "predicted_at_utc": ["2026-02-20T00:00:00Z"],
            "game_datetime_utc": ["2026-03-04T01:00:00Z"],
        },
    )
    recent_epoch = datetime(2026, 3, 3, 22, 53, tzinfo=timezone.utc).timestamp()
    path.touch()
    path.chmod(0o644)
    import os

    os.utime(path, (recent_epoch, recent_epoch))

    validate_predictions_freshness(data_dir, max_age_hours=6)
    out = capsys.readouterr().out
    assert "falling back to file mtime" in out
    assert "freshness_timestamp_source=file_mtime" in out


def test_artifact_marker_is_reported(tmp_path, monkeypatch, capsys):
    now = datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc)
    _freeze_now(monkeypatch, now)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / ".artifact_marker.txt").write_text("artifact_downloaded_at_utc=2026-03-03T22:50:00Z\n", encoding="utf-8")
    _write_predictions(
        data_dir / "predictions_latest.csv",
        {
            "game_id": ["g1"],
            "generated_at_utc": ["2026-03-03T22:50:00Z"],
            "game_datetime_utc": ["2026-03-04T01:00:00Z"],
        },
    )

    validate_predictions_freshness(data_dir, max_age_hours=6)
    out = capsys.readouterr().out
    assert "artifact_marker=artifact_downloaded_at_utc=2026-03-03T22:50:00Z" in out
