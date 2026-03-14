"""Backtest calibration policy: build, load, and resolve spread thresholds."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_backtest_calibration_policy(path: Path) -> dict[str, Any] | None:
    """Load a calibration policy JSON from *path*.

    Returns the parsed dict on success, or ``None`` if the file does not
    exist, is empty, or cannot be decoded.
    """
    try:
        if not path.exists() or path.stat().st_size == 0:
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_backtest_calibration_policy(
    scorecard: pd.DataFrame,
    bet_ledger: pd.DataFrame,
    *,
    generated_at_utc: str,
    default_thresholds: dict[str, float],
) -> dict[str, Any]:
    """Build a calibration policy dict from backtest *scorecard* results.

    For each ``mc_mode`` present in the scorecard the best-performing
    variant (by ``ats_accuracy`` then ``five_unit_annualized``) is chosen
    and its aggregate statistics are encoded as per-mode spread thresholds.
    The returned dict is intended to be serialised to JSON and later
    consumed by :func:`resolve_spread_thresholds`.
    """
    edge_min_default = float(default_thresholds.get("edge_min", 0.5))
    confidence_min_default = float(default_thresholds.get("confidence_min", 51.0))
    ats_prob_edge_min_default = float(default_thresholds.get("ats_prob_edge_min", 0.02))

    thresholds_by_mode: dict[str, dict[str, Any]] = {}

    if scorecard is not None and not scorecard.empty and "mc_mode" in scorecard.columns:
        for mc_mode, grp in scorecard.groupby("mc_mode", observed=True):
            mc_mode_str = str(mc_mode)

            sort_cols = [c for c in ["ats_accuracy", "five_unit_annualized"] if c in grp.columns]
            if sort_cols:
                best_row = (
                    grp.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
                    .iloc[0]
                )
            else:
                best_row = grp.iloc[0]

            variant_id = str(best_row.get("variant_id", "")) if hasattr(best_row, "get") else ""

            # Derive an edge threshold from historical performance.  If
            # avg_spread_edge is available and positive, target 80 % of the
            # observed mean (a safety margin below the historical average) but
            # cap at 3× the static default to prevent overly aggressive
            # thresholds on small samples.
            _EDGE_SAFETY_FACTOR = 0.8
            _EDGE_MAX_MULTIPLIER = 3
            avg_edge = pd.to_numeric(best_row.get("avg_spread_edge", np.nan) if hasattr(best_row, "get") else np.nan, errors="coerce")
            if pd.notna(avg_edge) and float(avg_edge) > edge_min_default:
                calibrated_edge = round(min(float(avg_edge) * _EDGE_SAFETY_FACTOR, edge_min_default * _EDGE_MAX_MULTIPLIER), 4)
            else:
                calibrated_edge = edge_min_default

            thresholds_by_mode[mc_mode_str] = {
                "edge_min": calibrated_edge,
                "confidence_min": confidence_min_default,
                "ats_prob_edge_min": ats_prob_edge_min_default,
                "policy_scope": "calibrated",
                "policy_variant": variant_id,
            }

    return {
        "generated_at_utc": generated_at_utc,
        "default_thresholds": {
            "edge_min": edge_min_default,
            "confidence_min": confidence_min_default,
            "ats_prob_edge_min": ats_prob_edge_min_default,
        },
        "spread_thresholds_by_mc_mode": thresholds_by_mode,
    }


def resolve_spread_thresholds(
    preds: pd.DataFrame,
    *,
    policy: dict[str, Any] | None,
    mc_mode: str,
    default_thresholds: dict[str, float],
) -> dict[str, Any]:
    """Return the spread thresholds to use for *mc_mode*.

    If *policy* contains a ``spread_thresholds_by_mc_mode`` entry for
    *mc_mode* those values are used; otherwise *default_thresholds* are
    returned.  Scalar threshold values are returned as-is (they broadcast
    when assigned to a DataFrame column).  ``policy_scope`` and
    ``policy_variant`` are returned as :class:`pandas.Series` aligned to
    *preds*.
    """
    edge_min = float(default_thresholds.get("edge_min", 0.5))
    confidence_min = float(default_thresholds.get("confidence_min", 51.0))
    ats_prob_edge_min = float(default_thresholds.get("ats_prob_edge_min", 0.02))
    scope = "defaults"
    variant = ""

    if policy and isinstance(policy, dict):
        mode_branch = policy.get("spread_thresholds_by_mc_mode", {}).get(mc_mode, {})
        if mode_branch and isinstance(mode_branch, dict):
            edge_min = float(pd.to_numeric(mode_branch.get("edge_min", edge_min), errors="coerce") or edge_min)
            confidence_min = float(pd.to_numeric(mode_branch.get("confidence_min", confidence_min), errors="coerce") or confidence_min)
            ats_prob_edge_min = float(pd.to_numeric(mode_branch.get("ats_prob_edge_min", ats_prob_edge_min), errors="coerce") or ats_prob_edge_min)
            scope = str(mode_branch.get("policy_scope", "calibrated"))
            variant = str(mode_branch.get("policy_variant", ""))

    idx = preds.index
    return {
        "edge_min": edge_min,
        "confidence_min": confidence_min,
        "ats_prob_edge_min": ats_prob_edge_min,
        "policy_scope": pd.Series(scope, index=idx, dtype=str),
        "policy_variant": pd.Series(variant, index=idx, dtype=str),
    }
