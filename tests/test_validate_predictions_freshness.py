from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from infra.ci.validate_predictions_freshness import validate, validate_file


def _base_now() -> datetime:
    return datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)


def _base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "generated_at_utc": ["2026-03-01T16:30:00Z", "2026-03-01T16:30:00Z"],
            "game_time_utc": ["2026-03-01T19:00:00Z", "2026-03-01T20:00:00Z"],
        }
    )


def test_validate_predictions_freshness_pass_case() -> None:
    now_utc = _base_now()
    df = _base_frame()

    summary = validate(
        df,
        selected_file="data/predictions_latest.csv",
        now_utc=now_utc,
        timezone_local="America/Los_Angeles",
        max_hours=6,
    )

    assert summary.result == "PASS"
    assert summary.parse_fail_rate == 0
    assert summary.freshness_age_hours < 6


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


def test_validate_predictions_freshness_fail_forward_looking() -> None:
    now_utc = _base_now()
    df = _base_frame()
    df["game_time_utc"] = ["2026-03-01T12:00:00Z", "2026-03-01T12:30:00Z"]

    with pytest.raises(RuntimeError, match="not forward-looking"):
        validate(
            df,
            selected_file="data/predictions_latest.csv",
            now_utc=now_utc,
            timezone_local="America/Los_Angeles",
            max_hours=6,
        )


def test_validate_predictions_freshness_fail_parse_rate() -> None:
    now_utc = _base_now()
    df = pd.DataFrame(
        {
            "game_id": [f"g{i}" for i in range(11)],
            "generated_at_utc": ["2026-03-01T17:00:00Z"] * 11,
            "game_time_utc": [
                "2026-03-01T19:00:00Z",
                "2026-03-01 20:00:00",  # naive -> invalid
                "2026-03-01 21:00:00",  # naive -> invalid
                "2026-03-01 22:00:00",  # naive -> invalid
                "2026-03-01 23:00:00",  # naive -> invalid
                "2026-03-01T23:00:00Z",
                "2026-03-02T00:00:00Z",
                "2026-03-02T01:00:00Z",
                "2026-03-02T02:00:00Z",
                "2026-03-02T03:00:00Z",
                "2026-03-02T04:00:00Z",
            ],
        }
    )

    with pytest.raises(RuntimeError, match="parse_fail_rate"):
        validate(
            df,
            selected_file="data/predictions_latest.csv",
            now_utc=now_utc,
            timezone_local="America/Los_Angeles",
            max_hours=6,
        )


def test_select_prediction_source_parity_with_build_derived(tmp_path, monkeypatch) -> None:
    import build_derived_csvs as bdc
    from infra.ci.validate_predictions_freshness import select_prediction_source

    data_dir = tmp_path / "data"
    data_dir.mkdir()

    latest = pd.DataFrame(
        {
            "game_id": ["g1"],
            "game_datetime_utc": ["2026-03-01T18:00:00Z"],
        }
    )
    mc_latest = pd.DataFrame(
        {
            "game_id": ["g2", "g3"],
            "game_datetime_utc": ["2026-03-01T20:00:00Z", "2026-03-01T19:00:00Z"],
        }
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
    now_utc = _base_now()
    file_path = tmp_path / "predictions_latest.csv"
    _base_frame().to_csv(file_path, index=False)

    summary = validate_file(
        file_path,
        now_utc=now_utc,
        timezone_local="America/Los_Angeles",
        max_hours=6,
    )

    assert summary.result == "PASS"
    assert summary.selected_file == str(file_path)
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
