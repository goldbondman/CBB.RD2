#!/usr/bin/env python3
"""
backtesting/git_ops.py
=======================
Git tag and push automation for qualified signal batches.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

NOW_UTC  = datetime.now(timezone.utc)
RUN_DATE = NOW_UTC.strftime("%Y-%m-%d")

FILES_TO_COMMIT = [
    "signals_library.py",
    "backtest_results.csv",
    "user_signals.csv",
]


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def git_push(n_merged: int, dry_run: bool = False) -> bool:
    """
    Stage, commit, tag, and push qualifying signals.
    Returns True on success.
    """
    if n_merged == 0:
        print("[git] No qualified signals — skipping push")
        return True

    tag = f"signal-batch-{RUN_DATE}"
    msg = f"AUTO: Add {n_merged} signals at 59%+ hit rate — {RUN_DATE}"

    try:
        if dry_run:
            print(f"[git DRY-RUN] Would commit: {FILES_TO_COMMIT}")
            print(f"[git DRY-RUN] Message: {msg}")
            print(f"[git DRY-RUN] Tag: {tag}")
            return True

        # Stage
        _run(["git", "add"] + FILES_TO_COMMIT)

        # Check if there's anything to commit
        status = _run(["git", "status", "--porcelain"])
        if not status.stdout.strip():
            print("[git] Nothing to commit")
            return True

        # Commit
        _run(["git", "commit", "-m", msg])

        # Tag (overwrite if exists)
        _run(["git", "tag", "-f", tag], check=False)

        # Push
        _run(["git", "push", "origin", "main"])
        _run(["git", "push", "origin", "--tags", "--force"], check=False)

        print(f"[git] Pushed {n_merged} signals — tag: {tag}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[git] ERROR: {e.stderr or e}")
        return False
