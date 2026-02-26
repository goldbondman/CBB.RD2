#!/usr/bin/env python3
"""Validated deployment gatekeeper for model weights with rollback support."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

DATA_DIR = Path("data")
CANDIDATE_PATH = DATA_DIR / "candidate_weights.json"
ACTIVE_PATH = DATA_DIR / "active_weights.json"
VALIDATION_PATH = DATA_DIR / "walk_forward_validation.json"
HISTORY_DIR = DATA_DIR / "weight_history"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _validation_summary(validation: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "recommendation": validation.get("recommendation"),
        "improvement_pp": validation.get("improvement_pp"),
        "folds_candidate_won": validation.get("folds_candidate_won"),
        "folds_total": validation.get("folds_total"),
        "evaluated_at": validation.get("evaluated_at"),
    }


def reject(candidate_weights: Dict[str, Any], stamp: str) -> int:
    _save_json(HISTORY_DIR / f"rejected_{stamp}.json", candidate_weights)
    print("Rejected — current weights unchanged.")
    return 0


def deploy(force: bool = False) -> int:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    validation = _load_json(VALIDATION_PATH)
    candidate_weights = _load_json(CANDIDATE_PATH)

    recommendation = str(validation.get("recommendation", "")).strip().upper()
    if recommendation not in {"DEPLOY", "REJECT", "INSUFFICIENT_DATA"}:
        raise ValueError("walk_forward_validation.json recommendation must be DEPLOY/REJECT/INSUFFICIENT_DATA")

    if force:
        recommendation = "DEPLOY"

    stamp = _timestamp()
    if recommendation == "INSUFFICIENT_DATA":
        print("Insufficient data — need 4+ walk-forward folds.")
        return 0

    if recommendation == "REJECT":
        return reject(candidate_weights, stamp)

    if ACTIVE_PATH.exists():
        shutil.copy2(ACTIVE_PATH, HISTORY_DIR / f"weights_{stamp}.json")

    improvement_pp = float(validation.get("improvement_pp", 0.0) or 0.0)
    folds_candidate_won = int(validation.get("folds_candidate_won", 0) or 0)
    folds_total = int(validation.get("folds_total", 0) or 0)

    payload = dict(candidate_weights)
    payload["deployed_at"] = datetime.now(timezone.utc).isoformat()
    payload["validation"] = _validation_summary(validation)
    payload["improvement_pp"] = improvement_pp
    payload["folds_candidate_won"] = folds_candidate_won
    payload["folds_total"] = folds_total

    _save_json(ACTIVE_PATH, payload)
    print(
        f"Deployed. CLV improvement: +{improvement_pp:.4f} "
        f"({folds_candidate_won}/{folds_total} folds)"
    )
    return 0


def rollback() -> int:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    history_files = sorted(
        [p for p in HISTORY_DIR.glob("*.json") if not p.name.startswith("rejected_")],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not history_files:
        print("No rollback file available.")
        return 1

    restore_file = history_files[0]
    shutil.copy2(restore_file, ACTIVE_PATH)
    print(f"Rolled back active weights from {restore_file.name}")
    return 0


def status() -> int:
    if not ACTIVE_PATH.exists():
        print("No active weights — using ModelConfig defaults")
        return 0

    weights = _load_json(ACTIVE_PATH)
    for key, value in weights.items():
        if key == "validation":
            continue
        print(f"{key}: {value}")

    validation = weights.get("validation", {})
    if isinstance(validation, dict) and "improvement_pp" in validation:
        print(f"validation_improvement_pp: {validation['improvement_pp']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Deploy regardless of walk-forward recommendation")
    parser.add_argument("--rollback", action="store_true", help="Restore the most recent non-rejected history file")
    parser.add_argument("--status", action="store_true", help="Show active weights")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.status:
        return status()
    if args.rollback:
        return rollback()
    return deploy(force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
