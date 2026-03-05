from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def test_window_grid_evaluator_outputs_all_combos(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    results = []
    for i in range(1, 16):
        event = f"e{i}"
        dt = f"2026-01-{i:02d}T00:00:00Z"
        # Home row for A and away row for B per event.
        rows.append(
            {
                "event_id": event,
                "game_datetime_utc": dt,
                "team_id": "A",
                "home_away": "home",
                "net_rtg": float(10 + i),
            }
        )
        rows.append(
            {
                "event_id": event,
                "game_datetime_utc": dt,
                "team_id": "B",
                "home_away": "away",
                "net_rtg": float(5 + (i % 3)),
            }
        )
        results.append(
            {
                "event_id": event,
                "game_id": event,
                "game_datetime_utc": dt,
                "market_spread": -4.0,
                "actual_margin": 6.0 + (i % 2),
            }
        )

    pd.DataFrame(rows).to_csv(data_dir / "team_game_weighted.csv", index=False)
    pd.DataFrame(results).to_csv(data_dir / "results_log.csv", index=False)

    out_csv = data_dir / "analytics" / "window_grid_results.csv"
    out_md = data_dir / "analytics" / "window_grid_exec_summary.md"
    proc = _run(
        [
            sys.executable,
            "scripts/window_grid_evaluator.py",
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
    assert out_csv.exists()
    assert out_md.exists()

    out = pd.read_csv(out_csv)
    expected = {"4/8", "4/12", "4/8/12", "5/10", "6/11", "7/12"}
    assert set(out["window_combo"]) == expected
    assert "stability_score" in out.columns
    # Longer windows should have fewer eligible rows in this synthetic setup.
    sample_48 = int(out.loc[out["window_combo"] == "4/8", "sample_size"].iloc[0])
    sample_712 = int(out.loc[out["window_combo"] == "7/12", "sample_size"].iloc[0])
    assert sample_48 >= sample_712
