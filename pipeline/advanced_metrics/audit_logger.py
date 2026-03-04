"""Audit logging utilities for feature engine runs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_git_sha(repo_root: Path | None = None) -> str:
    cwd = repo_root or Path.cwd()
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True, stderr=subprocess.DEVNULL)
            .strip()
        )
    except Exception:
        return "unknown"


def build_nan_report(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    report: dict[str, int] = {}
    for col in columns:
        if col in df.columns:
            report[col] = int(df[col].isna().sum())
    return report


def write_run_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
