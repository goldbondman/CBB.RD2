from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def test_feature_null_scanner_sample_limit(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "event_id": [f"g{i}" for i in range(120)],
            "feat_a": [None if i % 3 == 0 else float(i) for i in range(120)],
            "feat_b": [float(i) if i % 5 else None for i in range(120)],
        }
    )
    df.to_csv(data_dir / "rotation_features.csv", index=False)

    out_csv = data_dir / "quality" / "feature_null_report.csv"
    out_md = data_dir / "quality" / "feature_null_exec_summary.md"

    proc = _run(
        [
            sys.executable,
            "scripts/feature_null_scanner.py",
            "--data-dir",
            str(data_dir),
            "--output-csv",
            str(out_csv),
            "--output-md",
            str(out_md),
            "--sample-limit",
            "50",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out_csv.exists()
    assert out_md.exists()

    report = pd.read_csv(out_csv)
    assert not report.empty
    assert (report["scanned_rows"] <= 50).all()
    assert "status: `OK`" in out_md.read_text(encoding="utf-8")


def test_missing_market_detector_sample_limit(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    base = pd.DataFrame(
        {
            "game_id": [str(i) for i in range(1, 101)],
            "game_datetime_utc": ["2026-03-05T00:00:00Z"] * 100,
        }
    )
    market = pd.DataFrame({"event_id": [str(i) for i in range(1, 31)]})
    base.to_csv(data_dir / "predictions_combined_latest.csv", index=False)
    market.to_csv(data_dir / "market_lines.csv", index=False)

    out_csv = data_dir / "quality" / "missing_market_report.csv"
    out_md = data_dir / "quality" / "missing_market_exec_summary.md"

    proc = _run(
        [
            sys.executable,
            "scripts/missing_market_detector.py",
            "--data-dir",
            str(data_dir),
            "--output-csv",
            str(out_csv),
            "--output-md",
            str(out_md),
            "--sample-limit",
            "50",
        ],
        repo_root,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert out_csv.exists()
    assert out_md.exists()

    report = pd.read_csv(out_csv)
    assert len(report) == 50
    assert report["missing_market"].sum() == 20
    assert "status: `OK`" in out_md.read_text(encoding="utf-8")
