from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from pipeline.integrity import run_integrity_gate
from pipeline.update_policy import evaluate_update_eligibility


def _write_csv(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_identity_join_integrity(tmp_path: Path) -> None:
    _write_csv(tmp_path / "games.csv", [{"game_id": "g1", "game_datetime_utc": "2026-01-01T00:00:00Z"}])
    _write_csv(tmp_path / "market_lines.csv", [{"game_id": "g1", "pulled_at": "2025-12-31T00:00:00Z"}])
    _write_csv(tmp_path / "predictions_combined_latest.csv", [{"game_id": "missing"}])
    _write_csv(tmp_path / "results_log.csv", [{"game_id": "g1"}])

    result = run_integrity_gate(data_dir=tmp_path, as_of=pd.Timestamp("2026-01-01", tz="UTC"), mode="backtest")
    assert not result.ok
    assert any("not in games.csv" in e for e in result.errors)


def test_leakage_cutoff_enforcement(tmp_path: Path) -> None:
    _write_csv(tmp_path / "games.csv", [{"game_id": "g1", "game_datetime_utc": "2026-01-02T00:00:00Z"}])
    _write_csv(tmp_path / "market_lines.csv", [{"game_id": "g1", "pulled_at": "2026-01-03T00:00:00Z"}])
    _write_csv(tmp_path / "team_game_weighted.csv", [{"game_id": "g1"}])

    result = run_integrity_gate(data_dir=tmp_path, as_of=pd.Timestamp("2026-01-01", tz="UTC"), mode="predict")
    assert not result.ok
    assert any("newer than as_of" in e for e in result.errors)


def test_update_policy_cooldown_behavior(tmp_path: Path) -> None:
    pd.DataFrame([
        {"run_tag": "BAD", "graded_games": 80},
        {"run_tag": "BAD", "graded_games": 80},
        {"run_tag": "BAD", "graded_games": 80},
    ]).to_csv(tmp_path / "run_evaluations.csv", index=False)

    (tmp_path / "last_update.json").write_text(
        json.dumps(
            {
                "model_version": "v1",
                "promotion_timestamp": datetime.now(timezone.utc).isoformat(),
                "games_since_update": 240,
                "update_reason": "prior",
            }
        ),
        encoding="utf-8",
    )

    status = evaluate_update_eligibility(
        run_history_path=tmp_path / "run_evaluations.csv",
        last_update_path=tmp_path / "last_update.json",
        force_update=False,
        override_reason=None,
    )
    assert not status["eligible"]
    assert not status["cooldown_ok"]


def test_update_policy_three_of_five_trigger(tmp_path: Path) -> None:
    pd.DataFrame([
        {"run_tag": "GOOD", "graded_games": 40},
        {"run_tag": "BAD", "graded_games": 40},
        {"run_tag": "BAD", "graded_games": 40},
        {"run_tag": "GOOD", "graded_games": 40},
        {"run_tag": "BAD", "graded_games": 40},
    ]).to_csv(tmp_path / "run_evaluations.csv", index=False)

    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    (tmp_path / "last_update.json").write_text(
        json.dumps(
            {
                "model_version": "v1",
                "promotion_timestamp": old_ts,
                "games_since_update": 200,
                "update_reason": "prior",
            }
        ),
        encoding="utf-8",
    )

    status = evaluate_update_eligibility(
        run_history_path=tmp_path / "run_evaluations.csv",
        last_update_path=tmp_path / "last_update.json",
        force_update=False,
        override_reason=None,
    )
    assert status["eligible"]
    assert status["bad_runs_recent"] == 3
