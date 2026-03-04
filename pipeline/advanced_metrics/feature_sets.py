"""Curated feature bundles for downstream models."""

from __future__ import annotations

from .feature_registry import MATCHUP_FEATURE_REGISTRY, TEAM_GAME_FEATURE_REGISTRY


FEATURE_SET_SPREAD_V1 = {
    "team_game": (
        "ANE",
        "SVI",
        "PEQ",
        "POSW",
        "WL",
        "ODI",
        "TC",
        "TIN",
        "VOL",
        "DPC",
        "FFC",
        "PXP",
        "SCI",
        "factor_ODIs",
    ),
    "matchup": (
        "ODI_A",
        "ODI_B",
        "odi_diff",
        "odi_sum",
        "PEI_matchup",
        "POSW",
        "MTI",
        "SCI",
        "factor_ODIs",
        "factor_ODIs_diffs_sums",
    ),
}


FEATURE_SET_TOTAL_V1 = {
    "team_game": (
        "PEQ",
        "POSW",
        "TC",
        "TIN",
        "VOL",
        "FFC",
        "PXP",
        "SCI",
        "ODI",
        "factor_ODIs",
    ),
    "matchup": (
        "PEI_matchup",
        "POSW",
        "odi_sum",
        "MTI",
        "SCI",
        "factor_ODIs",
        "factor_ODIs_diffs_sums",
    ),
}


FEATURE_SET_ML_V1 = {
    "team_game": (
        "ANE",
        "SVI",
        "PEQ",
        "POSW",
        "WL",
        "ODI",
        "TC",
        "TIN",
        "VOL",
        "DPC",
        "FFC",
        "PXP",
        "SCI",
        "factor_ODIs",
    ),
    "matchup": (
        "ODI_A",
        "ODI_B",
        "odi_diff",
        "odi_sum",
        "PEI_matchup",
        "POSW",
        "MTI",
        "SCI",
        "factor_ODIs",
        "factor_ODIs_diffs_sums",
    ),
}


def validate_feature_set(feature_set: dict[str, tuple[str, ...]]) -> None:
    for feature_name in feature_set.get("team_game", ()):
        if feature_name not in TEAM_GAME_FEATURE_REGISTRY:
            raise ValueError(f"Unknown team_game feature in set: {feature_name}")
    for feature_name in feature_set.get("matchup", ()):
        if feature_name not in MATCHUP_FEATURE_REGISTRY:
            raise ValueError(f"Unknown matchup feature in set: {feature_name}")
