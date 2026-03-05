from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run(cmd: list[str], repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True, check=False)


def test_team_volatility_leak_free_windows_and_tiers(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(12):
        rows.append(
            {
                "event_id": f"g{i+1}",
                "game_datetime_utc": f"2026-01-{i+1:02d}T00:00:00Z",
                "season": 2026,
                "team_id": "100",
                "team": "Test Team",
                "home_away": "home" if i % 2 == 0 else "away",
                "margin": float(i - 3),
                "poss": float(68 + (i % 5)),
                "NetRtg": float(10 + i),
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "team_game_metrics.csv", index=False)

    out_csv = data_dir / "teams" / "team_volatility.csv"
    out_md = data_dir / "teams" / "team_volatility_exec_summary.md"
    proc = _run(
        [
            sys.executable,
            "scripts/team_volatility_score.py",
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
    assert len(out) == 12
    assert out.loc[0:2, "net_rating_std_l10"].isna().all()
    assert not math.isnan(float(out.loc[3, "net_rating_std_l10"]))

    expected_std = pd.Series([10.0, 11.0, 12.0]).std(ddof=0)
    actual_std = float(out.loc[3, "net_rating_std_l10"])
    assert abs(actual_std - float(expected_std)) < 1e-9

    non_null_tiers = out["volatility_tier"].dropna()
    assert not non_null_tiers.empty
    assert set(non_null_tiers.unique()).issubset({"low", "med", "high"})
    assert "status: `OK`" in out_md.read_text(encoding="utf-8")


def test_team_volatility_uses_margin_per_poss_when_net_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(15):
        rows.append(
            {
                "event_id": f"h{i+1}",
                "game_datetime_utc": f"2026-02-{(i % 9) + 1:02d}T00:00:00Z",
                "season": 2026,
                "team_id": "200",
                "team": "Fallback Team",
                "margin": float((i % 7) - 3),
                "poss": float(65 + (i % 6)),
                "pace": float(63 + (i % 4)),
            }
        )
    pd.DataFrame(rows).to_csv(data_dir / "team_game_metrics.csv", index=False)

    out_csv = data_dir / "teams" / "team_volatility.csv"
    out_md = data_dir / "teams" / "team_volatility_exec_summary.md"
    proc = _run(
        [
            sys.executable,
            "scripts/team_volatility_score.py",
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
    assert "derived_margin_per_poss" in str(out.loc[0, "net_metric_source"])
    assert out["net_rating_std_l10"].notna().sum() > 0
