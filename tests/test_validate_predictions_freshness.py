from datetime import datetime, timezone

import pandas as pd
import pytest

from infra.ci.validate_predictions_freshness import validate_predictions_freshness


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
