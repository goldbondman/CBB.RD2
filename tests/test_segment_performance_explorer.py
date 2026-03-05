from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def test_segment_performance_explorer_outputs_and_min_sample(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    (data_dir / "teams").mkdir(parents=True, exist_ok=True)
    (data_dir / "matchups").mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [
            {
                "event_id": "e1",
                "game_id": "g1",
                "game_datetime_utc": "2026-02-01T00:00:00Z",
                "home_team_id": "H1",
                "away_team_id": "A1",
                "home_team": "Home1",
                "away_team": "Away1",
                "pred_spread": -8.0,
                "spread_line": -5.0,
                "actual_margin": 10.0,
                "pred_total": 150.0,
                "market_total": 145.0,
                "actual_total": 148.0,
            },
            {
                "event_id": "e2",
                "game_id": "g2",
                "game_datetime_utc": "2026-02-02T00:00:00Z",
                "home_team_id": "H2",
                "away_team_id": "A2",
                "home_team": "Home2",
                "away_team": "Away2",
                "pred_spread": -1.0,
                "spread_line": -3.0,
                "actual_margin": -5.0,
                "pred_total": 138.0,
                "market_total": 140.0,
                "actual_total": 130.0,
            },
            {
                "event_id": "e3",
                "game_id": "g3",
                "game_datetime_utc": "2026-02-03T00:00:00Z",
                "home_team_id": "H3",
                "away_team_id": "A3",
                "home_team": "Home3",
                "away_team": "Away3",
                "pred_spread": -3.0,
                "spread_line": -2.0,
                "actual_margin": 1.0,
                "pred_total": 142.0,
                "market_total": 141.0,
                "actual_total": 141.0,
            },
        ]
    )
    results.to_csv(data_dir / "results_log_graded.csv", index=False)

    # Conference lookup should be discoverable from another table.
    pd.DataFrame(
        [
            {"event_id": "e1", "game_id": "g1", "home_conference": "ACC", "away_conference": "ACC"},
            {"event_id": "e2", "game_id": "g2", "home_conference": "SEC", "away_conference": "B12"},
            {"event_id": "e3", "game_id": "g3", "home_conference": "BE", "away_conference": "BE"},
        ]
    ).to_csv(data_dir / "predictions_history.csv", index=False)

    pd.DataFrame(
        [
            {"event_id": "e1", "team_id": "H1", "volatility_tier": "high"},
            {"event_id": "e2", "team_id": "A2", "volatility_tier": "low"},
            {"event_id": "e3", "team_id": "H3", "volatility_tier": "med"},
        ]
    ).to_csv(data_dir / "teams" / "team_volatility.csv", index=False)

    pd.DataFrame(
        [
            {"event_id": "e1", "team_id": "H1", "similarity_score": 0.8},
            {"event_id": "e2", "team_id": "A2", "similarity_score": 0.1},
            {"event_id": "e3", "team_id": "H3", "similarity_score": 0.5},
        ]
    ).to_csv(data_dir / "matchups" / "opponent_style_similarity.csv", index=False)

    out_csv = data_dir / "analytics" / "segment_performance.csv"
    out_md = data_dir / "analytics" / "segment_exec_summary.md"
    proc = _run(
        [
            sys.executable,
            "scripts/segment_performance_explorer.py",
            "--data-dir",
            str(data_dir),
            "--output-csv",
            str(out_csv),
            "--output-md",
            str(out_md),
            "--min-sample",
            "2",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out_csv.exists()
    assert out_md.exists()

    out = pd.read_csv(out_csv)
    assert not out.empty
    assert set(["segment_name", "segment_value", "market_type", "sample_size", "win_rate", "roi", "avg_edge", "avg_error"]).issubset(
        out.columns
    )
    assert "edge_bucket" in set(out["segment_name"])
    assert "total_bucket" in set(out["segment_name"])
    assert out["excluded_by_min_sample"].astype(bool).any()
    assert "excluded segment details" in out_md.read_text(encoding="utf-8")
