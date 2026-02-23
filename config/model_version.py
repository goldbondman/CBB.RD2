"""
model_version.py
─────────────────
Generates a deterministic version hash from the exact state of
all model configuration files at prediction time.

Two prediction runs with identical config files produce the
same model_version_hash. Any config change produces a new hash.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd


def _file_hash(path: Path) -> str:
    """SHA256 of file contents. Empty string if file missing."""
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def compute_model_version(
    data_dir: Path,
    config_dir: Optional[Path] = None,
) -> dict:
    """
    Compute a version snapshot from all config files that affect
    prediction output. Called once per prediction run.

    Returns dict with:
    - model_version_hash: short deterministic hash
    - component_hashes:   per-file hashes for debugging
    - config_snapshot:    actual weight values at time of run
    - created_at_utc:     timestamp
    """
    if config_dir is None:
        config_dir = data_dir

    config_files = {
        "model_weights": config_dir / "model_weights.json",
        "confidence_calibration": config_dir / "confidence_calibration.json",
        "model_bias_table": data_dir / "model_bias_table.csv",
    }

    component_hashes = {
        name: _file_hash(path)
        for name, path in config_files.items()
    }

    combined = "|".join(f"{k}:{v}" for k, v in sorted(component_hashes.items()))
    full_hash = hashlib.sha256(combined.encode()).hexdigest()
    short_hash = full_hash[:8]

    config_snapshot = {}
    weights_path = config_files["model_weights"]
    if weights_path.exists():
        try:
            weights_data = json.loads(weights_path.read_text())
            config_snapshot["weights"] = weights_data.get("weights", {})
            config_snapshot["weights_by_tier"] = list(
                weights_data.get("weights_by_tier", {}).keys()
            )
        except Exception:
            config_snapshot["weights"] = "parse_error"

    bias_path = config_files["model_bias_table"]
    if bias_path.exists():
        try:
            bt = pd.read_csv(bias_path)
            actionable = bt[bt.get("actionable", False) == True]
            config_snapshot["active_bias_corrections"] = len(actionable)
            config_snapshot["bias_dimensions"] = (
                actionable["dimension"].unique().tolist()
                if len(actionable) > 0 else []
            )
        except Exception:
            config_snapshot["active_bias_corrections"] = "parse_error"

    return {
        "model_version_hash": short_hash,
        "model_version_full": full_hash,
        "component_hashes": component_hashes,
        "config_snapshot": config_snapshot,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_run_id": _get_pipeline_run_id(),
    }


def _get_pipeline_run_id() -> str:
    """
    Use GitHub Actions run ID if available, otherwise timestamp.
    Unique per Actions run, not per prediction.
    """
    import os

    return os.environ.get(
        "GITHUB_RUN_ID",
        datetime.now(timezone.utc).strftime("local_%Y%m%d_%H%M%S"),
    )


def save_version_to_history(
    version: dict,
    history_path: Path,
) -> None:
    """
    Append version snapshot to model_version_history.json.
    Never overwrites — append only.
    Only appends if hash is new (deduplicates reruns).
    """
    history = []
    if history_path.exists():
        try:
            history = json.loads(history_path.read_text())
        except Exception:
            history = []

    existing_hashes = {v.get("model_version_hash") for v in history}
    if version["model_version_hash"] not in existing_hashes:
        history.append(version)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(json.dumps(history, indent=2))
