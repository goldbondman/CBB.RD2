from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import espn_sos


def _base_rows() -> list[dict[str, object]]:
    return [
        {
            "team_id": "A",
            "opponent_id": "B",
            "event_id": "1",
            "game_datetime_utc": "2025-01-01T00:00:00Z",
            "ortg": 100.0,
            "drtg": 95.0,
            "net_rtg": 5.0,
            "efg_pct": 0.50,
            "pace": 69.0,
            "orb_pct": 0.30,
            "drb_pct": 0.70,
            "tov_pct": 0.16,
            "ftr": 0.28,
        },
        {
            "team_id": "B",
            "opponent_id": "A",
            "event_id": "1",
            "game_datetime_utc": "2025-01-01T00:00:00Z",
            "ortg": 95.0,
            "drtg": 100.0,
            "net_rtg": -5.0,
            "efg_pct": 0.48,
            "pace": 69.0,
            "orb_pct": 0.28,
            "drb_pct": 0.72,
            "tov_pct": 0.17,
            "ftr": 0.25,
        },
        {
            "team_id": "A",
            "opponent_id": "C",
            "event_id": "2",
            "game_datetime_utc": "2025-01-02T00:00:00Z",
            "ortg": 102.0,
            "drtg": 97.0,
            "net_rtg": 5.0,
            "efg_pct": 0.51,
            "pace": 70.0,
            "orb_pct": 0.31,
            "drb_pct": 0.69,
            "tov_pct": 0.15,
            "ftr": 0.27,
        },
        {
            "team_id": "C",
            "opponent_id": "A",
            "event_id": "2",
            "game_datetime_utc": "2025-01-02T00:00:00Z",
            "ortg": 97.0,
            "drtg": 102.0,
            "net_rtg": -5.0,
            "efg_pct": 0.47,
            "pace": 70.0,
            "orb_pct": 0.27,
            "drb_pct": 0.73,
            "tov_pct": 0.18,
            "ftr": 0.24,
        },
    ]


def test_compute_sos_metrics_dedupes_team_event_input(tmp_path, monkeypatch):
    audit_path = tmp_path / "sos_size_audit.json"
    dup_sample = tmp_path / "sos_duplicate_input_samples.csv"
    merge_sample = tmp_path / "sos_merge_violation_samples.csv"
    monkeypatch.setattr(espn_sos, "SOS_SIZE_AUDIT_PATH", audit_path)
    monkeypatch.setattr(espn_sos, "SOS_DUP_SAMPLE_PATH", dup_sample)
    monkeypatch.setattr(espn_sos, "SOS_MERGE_VIOLATION_PATH", merge_sample)

    rows = _base_rows()
    rows.append(rows[0].copy())  # duplicate team-event row
    df = pd.DataFrame(rows)
    out = espn_sos.compute_sos_metrics(df)

    assert len(out) == 4
    assert out[["team_id", "event_id"]].drop_duplicates().shape[0] == 4
    assert audit_path.exists()
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["input_dedup"]["duplicate_keys"] == 1


def test_compute_sos_metrics_fallback_dedupe_without_event_id(tmp_path, monkeypatch):
    audit_path = tmp_path / "sos_size_audit.json"
    dup_sample = tmp_path / "sos_duplicate_input_samples.csv"
    merge_sample = tmp_path / "sos_merge_violation_samples.csv"
    monkeypatch.setattr(espn_sos, "SOS_SIZE_AUDIT_PATH", audit_path)
    monkeypatch.setattr(espn_sos, "SOS_DUP_SAMPLE_PATH", dup_sample)
    monkeypatch.setattr(espn_sos, "SOS_MERGE_VIOLATION_PATH", merge_sample)

    df = pd.DataFrame(_base_rows()).drop(columns=["event_id"])
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicate fallback key
    out = espn_sos.compute_sos_metrics(df)

    assert len(out) == 4
    assert out[["team_id", "opponent_id", "game_datetime_utc"]].drop_duplicates().shape[0] == 4


def test_row_guard_raises_on_explosion():
    audit: dict[str, object] = {}
    with pytest.raises(RuntimeError, match="row explosion detected"):
        espn_sos._enforce_row_guard("test_guard", 100, 200, audit)  # noqa: SLF001
