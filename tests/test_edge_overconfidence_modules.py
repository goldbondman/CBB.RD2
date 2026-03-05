from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True, check=False)


def test_edge_bucketing_engine_outputs_expected_buckets(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, 101):
        rows.append(
            {
                "event_id": str(i),
                "game_id": str(i),
                "game_datetime_utc": "2026-03-05T00:00:00Z",
                "pred_spread": float(i % 10),
                "spread_line": float((i % 10) - 1),
                "actual_margin": float((i % 10) - 2),
                "pred_total": float(140 + (i % 11)),
                "market_total": float(138 + (i % 11)),
                "actual_total": float(137 + (i % 11)),
                "primary_confidence": float(0.5 + ((i % 10) / 20)),
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "results_log_graded.csv", index=False)

    out_csv = data_dir / "analytics" / "edge_buckets.csv"
    out_md = data_dir / "analytics" / "edge_buckets_exec_summary.md"
    proc = _run(
        [
            sys.executable,
            "scripts/edge_bucketing_engine.py",
            "--data-dir",
            str(data_dir),
            "--output-csv",
            str(out_csv),
            "--output-summary-md",
            str(out_md),
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_csv.exists()
    assert out_md.exists()
    df = pd.read_csv(out_csv)
    assert set(df["bucket"]).issubset({"0-2", "2-4", "4-6", "6-8", "8+"})
    assert set(df["market_type"]) == {"spread", "total"}


def test_overconfidence_detector_quantile_flags(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(1, 101):
        rows.append(
            {
                "event_id": str(i),
                "game_id": str(i),
                "game_datetime_utc": "2026-03-05T00:00:00Z",
                "pred_spread": float(i % 9),
                "spread_line": float((i % 9) - 0.5),
                "actual_margin": float((i % 9) - (3 if i % 5 == 0 else 1)),
                "pred_total": float(142 + (i % 7)),
                "market_total": float(140 + (i % 7)),
                "actual_total": float(139 + (i % 7) + (4 if i % 6 == 0 else 0)),
                "primary_confidence": float(0.45 + (i % 20) / 40),
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "results_log_graded.csv", index=False)

    out_csv = data_dir / "analytics" / "overconfidence_report.csv"
    out_md = data_dir / "analytics" / "overconfidence_exec_summary.md"
    proc = _run(
        [
            sys.executable,
            "scripts/overconfidence_detector.py",
            "--data-dir",
            str(data_dir),
            "--output-csv",
            str(out_csv),
            "--output-md",
            str(out_md),
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out_csv.exists()
    assert out_md.exists()
    df = pd.read_csv(out_csv)
    assert not df.empty
    assert "overconfidence_flag" in df.columns
    assert df["overconfidence_flag"].astype(bool).sum() > 0
