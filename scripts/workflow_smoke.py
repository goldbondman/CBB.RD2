#!/usr/bin/env python3
"""Local workflow smoke runner for predictions, analytics, and player overlay."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"


def _run(cmd: list[str], *, allow_fail: bool = False) -> int:
    print(f"[RUN] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env)
    if proc.returncode != 0 and not allow_fail:
        raise SystemExit(f"[ERROR] command failed ({proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def _supports_limit(script_path: str) -> bool:
    proc = subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    help_text = (proc.stdout or "") + (proc.stderr or "")
    return "--limit" in help_text


def _require(path: Path, label: str) -> None:
    if not path.exists():
        raise SystemExit(f"[BLOCKED] missing required file for {label}: {path}")
    if path.suffix == ".csv" and path.stat().st_size == 0:
        raise SystemExit(f"[BLOCKED] empty required CSV for {label}: {path}")


def _copy_if_exists(src: Path, dst_dir: Path) -> None:
    if src.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / src.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="", help="Optional smoke run id")
    parser.add_argument("--date", default="", help="Optional prediction date YYYYMMDD")
    parser.add_argument("--live", action="store_true", help="Run live prediction pull (networked)")
    args = parser.parse_args()

    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    smoke_dir = DATA_DIR / "smoke" / run_id
    smoke_dir.mkdir(parents=True, exist_ok=True)

    _require(DATA_DIR / "games.csv", "predictions smoke")
    _require(DATA_DIR / "team_game_weighted.csv", "predictions smoke")

    # Predictions stage
    if args.live:
        pred_cmd = [sys.executable, "espn_prediction_runner.py"]
        if args.date.strip():
            pred_cmd.extend(["--date", args.date.strip()])
        if _supports_limit("espn_prediction_runner.py"):
            pred_cmd.extend(["--limit", "200"])
        _run(pred_cmd)
    else:
        print("[OFFLINE] Skipping espn_prediction_runner.py (use --live to enable networked pull)")
        if not (DATA_DIR / "predictions_latest.csv").exists():
            print("[BLOCKED] predictions_latest.csv missing in offline mode")

    _run([sys.executable, "cbb_ensemble.py"], allow_fail=True)
    _run(
        [
            sys.executable,
            "cbb_monte_carlo.py",
            "--input",
            "data/predictions_latest.csv",
            "--n-sims",
            "500",
        ],
        allow_fail=True,
    )

    # Analytics stage
    _run([sys.executable, "evaluation/predictions_graded.py"], allow_fail=True)
    tracker_cmd = [sys.executable, "cbb_results_tracker.py", "--reprocess-days", "3"]
    if _supports_limit("cbb_results_tracker.py"):
        tracker_cmd.extend(["--limit", "200"])
    _run(tracker_cmd, allow_fail=True)
    _run([sys.executable, "cbb_accuracy_report.py"], allow_fail=True)

    # Player overlay stage (only if standalone overlay input exists)
    overlay_input = DATA_DIR / "player_overlay_input.csv"
    overlay_output = DATA_DIR / "player_overlay_predictions.csv"
    if overlay_input.exists() and overlay_input.stat().st_size > 0:
        _run(
            [
                sys.executable,
                "scripts/run_player_overlay.py",
                "--input",
                str(overlay_input),
                "--output",
                str(overlay_output),
            ],
            allow_fail=True,
        )
    else:
        print("[BLOCKED] player overlay smoke skipped: data/player_overlay_input.csv missing/empty")

    # Integrity reports
    for workflow in ("cbb_predictions_rolling", "cbb_analytics", "player_overlay"):
        manifest = REPO_ROOT / ".github" / "pipeline_contracts" / f"{workflow}.contract.json"
        if not manifest.exists():
            print(f"[BLOCKED] missing manifest: {manifest}")
            continue
        _run(
            [
                sys.executable,
                ".github/scripts/data_integrity_report.py",
                "--manifest",
                str(manifest),
                "--workflow-name",
                workflow,
                "--run-id",
                f"{run_id}-smoke",
            ],
            allow_fail=True,
        )

    # Copy key outputs into smoke folder
    for output in [
        DATA_DIR / "predictions_latest.csv",
        DATA_DIR / "predictions_combined_latest.csv",
        DATA_DIR / "predictions_history.csv",
        DATA_DIR / "predictions_graded.csv",
        DATA_DIR / "results_log.csv",
        DATA_DIR / "results_summary.csv",
        DATA_DIR / "player_overlay_predictions.csv",
    ]:
        _copy_if_exists(output, smoke_dir)

    # Copy integrity summaries
    integrity_root = DATA_DIR / "integrity_reports"
    if integrity_root.exists():
        shutil.copytree(integrity_root, smoke_dir / "integrity_reports", dirs_exist_ok=True)

    print(f"[OK] smoke outputs: {smoke_dir}")
    print(f"[OK] integrity summary root: {smoke_dir / 'integrity_reports'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
