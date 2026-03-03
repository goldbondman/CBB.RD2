from datetime import datetime, timezone

import pandas as pd
import pytest

from infra.ci.validate_predictions_freshness import validate_predictions_freshness


def _write_predictions(path, generated_at_utc: str, game_times: list[str]) -> None:
    df = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(len(game_times))],
            "generated_at_utc": [generated_at_utc for _ in game_times],
            "game_datetime_utc": game_times,
        }
    )
    df.to_csv(path, index=False)


def test_validate_predictions_freshness_passes_for_recent_and_future_games(tmp_path, monkeypatch):
    now = datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz else now.replace(tzinfo=None)

    monkeypatch.setattr("infra.ci.validate_predictions_freshness.datetime", FrozenDateTime)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_predictions(
        data_dir / "predictions_latest.csv",
        generated_at_utc="2026-03-03T22:30:00Z",
        game_times=["2026-03-04T01:00:00Z", "2026-03-05T00:00:00Z"],
    )

    validate_predictions_freshness(data_dir, max_age_hours=6)


def test_validate_predictions_freshness_fails_for_stale_timestamp(tmp_path, monkeypatch):
    now = datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz else now.replace(tzinfo=None)

    monkeypatch.setattr("infra.ci.validate_predictions_freshness.datetime", FrozenDateTime)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_predictions(
        data_dir / "predictions_latest.csv",
        generated_at_utc="2026-03-03T10:00:00Z",
        game_times=["2026-03-04T01:00:00Z"],
    )

    with pytest.raises(RuntimeError, match="artifact is stale"):
        validate_predictions_freshness(data_dir, max_age_hours=6)


def test_validate_predictions_freshness_fails_for_past_only_schedule(tmp_path, monkeypatch):
    now = datetime(2026, 3, 3, 22, 54, tzinfo=timezone.utc)

    class FrozenDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now if tz else now.replace(tzinfo=None)

    monkeypatch.setattr("infra.ci.validate_predictions_freshness.datetime", FrozenDateTime)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_predictions(
        data_dir / "predictions_latest.csv",
        generated_at_utc="2026-03-03T22:40:00Z",
        game_times=["2026-03-03T12:00:00Z"],
    )

    with pytest.raises(RuntimeError, match="schedule is stale"):
        validate_predictions_freshness(data_dir, max_age_hours=6)
