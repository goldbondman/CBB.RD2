from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def test_context_overlay_run_all(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    (data_dir / "tournaments").mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {"game_id": "g1", "event_id": "e1", "game_datetime_utc": "2026-03-10T00:00:00Z", "game_type": "conf_tournament"},
            {"game_id": "g2", "event_id": "e2", "game_datetime_utc": "2026-03-11T00:00:00Z", "game_type": "regular"},
        ]
    ).to_csv(data_dir / "games.csv", index=False)
    pd.DataFrame([{"game_id": "g1", "event_id": "e1", "fatigue_tier": "high"}]).to_csv(
        data_dir / "tournaments" / "fatigue_flags.csv",
        index=False,
    )

    out_csv = data_dir / "context" / "context_overlay_latest.csv"
    out_md = data_dir / "context" / "context_overlay_exec_summary.md"
    proc = _run(
        [
            sys.executable,
            "context_layers/run_all.py",
            "--data-dir",
            str(data_dir),
            "--output-csv",
            str(out_csv),
            "--output-md",
            str(out_md),
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    out = pd.read_csv(out_csv)
    assert len(out) == 2
    assert "fatigue_tier" in out.columns
    assert "tournament_layers_active" in out.columns
    assert out_md.exists()


def test_gate_builder_runs_with_minimal_inputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "event_id": "e1",
                "game_id": "g1",
                "game_datetime_utc": "2026-03-01T00:00:00Z",
                "home_team_id": "H1",
                "away_team_id": "A1",
                "pred_spread": -6.0,
                "market_spread": -3.0,
                "actual_margin": 8.0,
                "market_total": 139.0,
            },
            {
                "event_id": "e2",
                "game_id": "g2",
                "game_datetime_utc": "2026-03-02T00:00:00Z",
                "home_team_id": "H2",
                "away_team_id": "A2",
                "pred_spread": -1.0,
                "market_spread": -4.0,
                "actual_margin": -2.0,
                "market_total": 127.0,
            },
        ]
    ).to_csv(data_dir / "results_log.csv", index=False)

    out_results = data_dir / "gates" / "gate_results.csv"
    out_best = data_dir / "gates" / "gate_rules_best.csv"
    out_md = data_dir / "gates" / "gate_exec_summary.md"
    proc = _run(
        [
            sys.executable,
            "scripts/gate_builder.py",
            "--data-dir",
            str(data_dir),
            "--output-results-csv",
            str(out_results),
            "--output-best-csv",
            str(out_best),
            "--output-md",
            str(out_md),
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out_results.exists()
    assert out_best.exists()
    assert out_md.exists()
    df = pd.read_csv(out_results)
    assert not df.empty
    assert "gate_family" in df.columns
