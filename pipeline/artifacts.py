from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACTS_ROOT = ROOT / "artifacts"


def resolve_run_dir(run_id: str) -> Path:
    run_dir = ARTIFACTS_ROOT / run_id
    for rel in [
        "manifest",
        "predictions",
        "backtest/detail",
        "backtest/summary",
        "evaluation",
        "logs",
    ]:
        (run_dir / rel).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")
