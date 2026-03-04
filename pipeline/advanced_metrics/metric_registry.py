"""Backward-compatible shim to the feature registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .feature_registry import MATCHUP_FEATURE_REGISTRY, TEAM_GAME_FEATURE_REGISTRY


@dataclass(frozen=True)
class MetricDefinition:
    metric_name: str
    required_inputs: tuple[str, ...]
    derived_inputs: tuple[str, ...]
    compute_fn: Callable[[pd.DataFrame], pd.DataFrame]
    output_columns: tuple[str, ...]
    grain: str


TEAM_METRIC_REGISTRY: dict[str, MetricDefinition] = {
    name: MetricDefinition(
        metric_name=spec.name,
        required_inputs=spec.required_inputs,
        derived_inputs=spec.derived_inputs,
        compute_fn=spec.compute_fn,
        output_columns=spec.output_cols,
        grain="team",
    )
    for name, spec in TEAM_GAME_FEATURE_REGISTRY.items()
}

MATCHUP_METRIC_REGISTRY: dict[str, MetricDefinition] = {
    name: MetricDefinition(
        metric_name=spec.name,
        required_inputs=spec.required_inputs,
        derived_inputs=spec.derived_inputs,
        compute_fn=spec.compute_fn,
        output_columns=spec.output_cols,
        grain="matchup",
    )
    for name, spec in MATCHUP_FEATURE_REGISTRY.items()
}


def team_metric_names() -> list[str]:
    return list(TEAM_METRIC_REGISTRY.keys())


def matchup_metric_names() -> list[str]:
    return list(MATCHUP_METRIC_REGISTRY.keys())
