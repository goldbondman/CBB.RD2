#!/usr/bin/env python3
"""Deploy, reject, inspect, and rollback model weight configurations."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

DATA_DIR = Path("data")
CANDIDATE_PATH = DATA_DIR / "candidate_weights.json"
ACTIVE_PATH = DATA_DIR / "active_weights.json"
WALK_FORWARD_PATH = DATA_DIR / "walk_forward_results.csv"
HISTORY_DIR = DATA_DIR / "weight_history"


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _load_candidate_weights() -> Dict[str, Any]:
    if not CANDIDATE_PATH.exists():
        raise FileNotFoundError(f"Missing candidate weights file: {CANDIDATE_PATH}")
    return json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))


def _extract_validation_metadata() -> Dict[str, Any]:
    if not WALK_FORWARD_PATH.exists():
        raise FileNotFoundError(f"Missing walk-forward results file: {WALK_FORWARD_PATH}")

    df = pd.read_csv(WALK_FORWARD_PATH)
    if df.empty:
        raise ValueError(f"Walk-forward results file is empty: {WALK_FORWARD_PATH}")

    row = df.iloc[-1].to_dict()
    normalized = {str(k).strip().lower(): v for k, v in row.items()}

    recommendation = str(normalized.get("recommendation", "")).strip().upper()
    if recommendation not in {"DEPLOY", "REJECT", "INSUFFICIENT_DATA"}:
        raise ValueError(
            "walk_forward_results.csv must include recommendation with one of "
            "DEPLOY/REJECT/INSUFFICIENT_DATA"
        )

    improvement_value = normalized.get("improvement_pp", 0.0)
    try:
        improvement_pp = float(improvement_value)
    except (TypeError, ValueError):
        improvement_pp = 0.0

    return {
        "recommendation": recommendation,
        "improvement_pp": improvement_pp,
        "source_file": str(WALK_FORWARD_PATH),
        "source_row": row,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _deploy(force: bool = False) -> int:
    candidate_weights = _load_candidate_weights()
    validation = _extract_validation_metadata()
    recommendation = "DEPLOY" if force else validation["recommendation"]
    timestamp = _utc_timestamp()

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if recommendation == "DEPLOY":
        if ACTIVE_PATH.exists():
            shutil.copy2(ACTIVE_PATH, HISTORY_DIR / f"weights_{timestamp}.json")

        payload = dict(candidate_weights)
        payload["deployed_at"] = datetime.now(timezone.utc).isoformat()
        payload["validation"] = validation
        _save_json(ACTIVE_PATH, payload)

        print(f"Deployed new weights. CLV improvement: +{validation['improvement_pp']:.3f}")
        return 0

    if recommendation == "REJECT":
        _save_json(HISTORY_DIR / f"rejected_{timestamp}.json", candidate_weights)
        print("Candidate weights rejected. Current weights unchanged.")
        return 0

    print("Insufficient data for weight update. Need 4+ walk-forward folds.")
    return 0


def _status() -> int:
    if not ACTIVE_PATH.exists():
        print("No active weights deployed yet.")
        return 0

    payload = json.loads(ACTIVE_PATH.read_text(encoding="utf-8"))
    print(json.dumps(payload, indent=2))
    return 0


def _rollback() -> int:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    candidates = sorted(HISTORY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        print("No weight history entries found to rollback.")
        return 1

    source = candidates[0]
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["rolled_back_from"] = source.name
    payload["rolled_back_at"] = datetime.now(timezone.utc).isoformat()
    _save_json(ACTIVE_PATH, payload)

    print(f"Rolled back active weights from {source.name}.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Deploy candidate weights regardless of recommendation")
    parser.add_argument("--status", action="store_true", help="Print current active weights")
    parser.add_argument("--rollback", action="store_true", help="Restore the most recent weight_history entry")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.status:
        return _status()
    if args.rollback:
        return _rollback()
    return _deploy(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
