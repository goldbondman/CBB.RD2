"""Integrity gate for feature input contracts."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .feature_registry import FeatureSpec


@dataclass(frozen=True)
class FeatureIntegrityResult:
    feature_name: str
    status: str  # ACTIVE or BLOCKED
    missing_columns: tuple[str, ...]


def validate_feature_inputs(df: pd.DataFrame, spec: FeatureSpec) -> FeatureIntegrityResult:
    available = set(df.columns)
    missing = tuple(sorted(col for col in spec.required_inputs if col not in available))
    status = "ACTIVE" if not missing else "BLOCKED"
    return FeatureIntegrityResult(
        feature_name=spec.name,
        status=status,
        missing_columns=missing,
    )


def evaluate_integrity(
    df: pd.DataFrame,
    registry: dict[str, FeatureSpec],
    feature_names: list[str] | None = None,
) -> dict[str, FeatureIntegrityResult]:
    selected = feature_names if feature_names is not None else list(registry.keys())
    return {name: validate_feature_inputs(df, registry[name]) for name in selected}
