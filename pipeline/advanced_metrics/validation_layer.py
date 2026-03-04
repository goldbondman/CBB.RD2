"""Backward-compatible wrappers for feature integrity checks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .feature_registry import get_registry
from .integrity_gate import evaluate_integrity


@dataclass(frozen=True)
class MetricValidationResult:
    metric_name: str
    grain: str
    status: str
    missing_inputs: tuple[str, ...]


def validate_metric_inputs(df: pd.DataFrame, *, grain: str) -> dict[str, MetricValidationResult]:
    mapped_grain = "team_game" if grain == "team" else "matchup"
    registry = get_registry(mapped_grain)
    results = evaluate_integrity(df, registry)
    return {
        name: MetricValidationResult(
            metric_name=name,
            grain=grain,
            status=result.status,
            missing_inputs=result.missing_columns,
        )
        for name, result in results.items()
    }
