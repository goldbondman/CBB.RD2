"""Registry of Feature Engine specifications."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from typing import Callable, Literal

import pandas as pd

from .metric_library import (
    f_ANE,
    f_DPC,
    f_FFC,
    f_MTI,
    f_ODI,
    f_ODI_A,
    f_ODI_B,
    f_PEI_matchup,
    f_PEQ,
    f_POSW,
    f_POSW_matchup,
    f_PXP,
    f_SCI,
    f_SCI_matchup,
    f_SVI,
    f_TC,
    f_TIN,
    f_VOL,
    f_WL,
    f_factor_ODIs,
    f_factor_ODIs_AB,
    f_factor_ODIs_diffs_sums,
    f_odi_diff,
    f_odi_sum,
)

FeatureGrain = Literal["team_game", "matchup"]


@dataclass(frozen=True)
class FeatureCacheSpec:
    enabled: bool
    key_fields: tuple[str, ...]
    version_hash: str


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    grain: FeatureGrain
    required_inputs: tuple[str, ...]
    derived_inputs: tuple[str, ...]
    dependencies: tuple[str, ...]
    compute_fn: Callable[[pd.DataFrame], pd.DataFrame]
    output_cols: tuple[str, ...]
    cache: FeatureCacheSpec


def _compute_version_hash(
    *,
    name: str,
    grain: FeatureGrain,
    required_inputs: tuple[str, ...],
    derived_inputs: tuple[str, ...],
    dependencies: tuple[str, ...],
    output_cols: tuple[str, ...],
    key_fields: tuple[str, ...],
    compute_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> str:
    try:
        fn_source = inspect.getsource(compute_fn)
    except OSError:
        fn_source = repr(compute_fn)

    mapping_payload = {
        "name": name,
        "grain": grain,
        "required_inputs": list(required_inputs),
        "derived_inputs": list(derived_inputs),
        "dependencies": list(dependencies),
        "output_cols": list(output_cols),
        "key_fields": list(key_fields),
        "compute_fn_source": fn_source,
    }
    encoded = json.dumps(mapping_payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _make_feature(
    *,
    name: str,
    grain: FeatureGrain,
    required_inputs: tuple[str, ...],
    derived_inputs: tuple[str, ...],
    dependencies: tuple[str, ...],
    compute_fn: Callable[[pd.DataFrame], pd.DataFrame],
    output_cols: tuple[str, ...],
    cache_enabled: bool = True,
    key_fields: tuple[str, ...] = ("season_id", "window_id", "feature_name"),
) -> FeatureSpec:
    version_hash = _compute_version_hash(
        name=name,
        grain=grain,
        required_inputs=required_inputs,
        derived_inputs=derived_inputs,
        dependencies=dependencies,
        output_cols=output_cols,
        key_fields=key_fields,
        compute_fn=compute_fn,
    )
    return FeatureSpec(
        name=name,
        grain=grain,
        required_inputs=required_inputs,
        derived_inputs=derived_inputs,
        dependencies=dependencies,
        compute_fn=compute_fn,
        output_cols=output_cols,
        cache=FeatureCacheSpec(
            enabled=cache_enabled,
            key_fields=key_fields,
            version_hash=version_hash,
        ),
    )


TEAM_GAME_FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    "WL": _make_feature(
        name="WL",
        grain="team_game",
        required_inputs=("points_for", "points_against"),
        derived_inputs=(),
        dependencies=(),
        compute_fn=f_WL,
        output_cols=("WL",),
    ),
    "ANE": _make_feature(
        name="ANE",
        grain="team_game",
        required_inputs=("NetRtg", "opp_pre_NetRtg_season"),
        derived_inputs=(
            "NetRtg = OffEff - DefEff",
            "opp_pre_NetRtg_season = opponent pregame expanding mean(NetRtg)",
        ),
        dependencies=(),
        compute_fn=f_ANE,
        output_cols=("ANE",),
    ),
    "SVI": _make_feature(
        name="SVI",
        grain="team_game",
        required_inputs=("WL", "opp_pre_NetRtg_season"),
        derived_inputs=("opp_pre_NetRtg_season = opponent pregame expanding mean(NetRtg)",),
        dependencies=("WL",),
        compute_fn=f_SVI,
        output_cols=("SVI",),
    ),
    "PEQ": _make_feature(
        name="PEQ",
        grain="team_game",
        required_inputs=("OffEff", "TOV%"),
        derived_inputs=("OffEff = 100*points_for/poss", "TOV% = tov/poss"),
        dependencies=(),
        compute_fn=f_PEQ,
        output_cols=("PEQ",),
    ),
    "POSW": _make_feature(
        name="POSW",
        grain="team_game",
        required_inputs=("ORB%", "DRB%", "TOV%"),
        derived_inputs=("ORB%, DRB% from team/opponent rebound shares",),
        dependencies=(),
        compute_fn=f_POSW,
        output_cols=("POSW",),
    ),
    "ODI": _make_feature(
        name="ODI",
        grain="team_game",
        required_inputs=("eFG", "TOV%", "ORB%", "FTr", "opp_eFG", "opp_TOV%", "opp_ORB%", "opp_FTr"),
        derived_inputs=("opp_* from opponent row linked by (event_id, opponent_id)",),
        dependencies=(),
        compute_fn=f_ODI,
        output_cols=("ODI",),
    ),
    "factor_ODIs": _make_feature(
        name="factor_ODIs",
        grain="team_game",
        required_inputs=("eFG", "TOV%", "ORB%", "FTr", "opp_eFG", "opp_TOV%", "opp_ORB%", "opp_FTr"),
        derived_inputs=("factor edges derived from four-factor deltas vs opponent",),
        dependencies=(),
        compute_fn=f_factor_ODIs,
        output_cols=("eFG_ODI", "TO_ODI", "ORB_ODI", "FTR_ODI"),
    ),
    "TC": _make_feature(
        name="TC",
        grain="team_game",
        required_inputs=("poss", "opp_pre_poss_season"),
        derived_inputs=("poss = fga-orb+tov+0.44*fta", "opp_pre_poss_season = opponent pregame expanding mean(poss)"),
        dependencies=(),
        compute_fn=f_TC,
        output_cols=("TC",),
    ),
    "TIN": _make_feature(
        name="TIN",
        grain="team_game",
        required_inputs=("poss", "pre_poss_season"),
        derived_inputs=("pre_poss_season = team pregame expanding mean(poss)",),
        dependencies=(),
        compute_fn=f_TIN,
        output_cols=("TIN",),
    ),
    "VOL": _make_feature(
        name="VOL",
        grain="team_game",
        required_inputs=("NetRtg", "pre_NetRtg_season"),
        derived_inputs=("pre_NetRtg_season = team pregame expanding mean(NetRtg)",),
        dependencies=(),
        compute_fn=f_VOL,
        output_cols=("VOL",),
    ),
    "DPC": _make_feature(
        name="DPC",
        grain="team_game",
        required_inputs=("bench_minutes_share", "TS_bench", "REB_rate_bench"),
        derived_inputs=(
            "bench_minutes_share from starter_bench_helper",
            "TS_bench from starter_bench_helper",
            "REB_rate_bench from starter_bench_helper",
        ),
        dependencies=(),
        compute_fn=f_DPC,
        output_cols=("DPC",),
    ),
    "FFC": _make_feature(
        name="FFC",
        grain="team_game",
        required_inputs=("eFG", "TOV%", "ORB%", "FTr"),
        derived_inputs=(),
        dependencies=(),
        compute_fn=f_FFC,
        output_cols=("FFC",),
    ),
    "PXP": _make_feature(
        name="PXP",
        grain="team_game",
        required_inputs=("NetRtg", "pre_NetRtg_season"),
        derived_inputs=(),
        dependencies=(),
        compute_fn=f_PXP,
        output_cols=("PXP",),
    ),
    "SCI": _make_feature(
        name="SCI",
        grain="team_game",
        required_inputs=("ODI", "season", "team_id", "event_id", "game_datetime_utc"),
        derived_inputs=("SCI = abs(within-season leak-free zscore(ODI))",),
        dependencies=("ODI",),
        compute_fn=f_SCI,
        output_cols=("SCI",),
    ),
}


MATCHUP_FEATURE_REGISTRY: dict[str, FeatureSpec] = {
    "ODI_A": _make_feature(
        name="ODI_A",
        grain="matchup",
        required_inputs=("ODI_A",),
        derived_inputs=("ODI_A derived from side-A team ODI",),
        dependencies=(),
        compute_fn=f_ODI_A,
        output_cols=("ODI_A",),
    ),
    "ODI_B": _make_feature(
        name="ODI_B",
        grain="matchup",
        required_inputs=("ODI_B",),
        derived_inputs=("ODI_B derived from side-B team ODI",),
        dependencies=(),
        compute_fn=f_ODI_B,
        output_cols=("ODI_B",),
    ),
    "PEI_matchup": _make_feature(
        name="PEI_matchup",
        grain="matchup",
        required_inputs=("PEQ_A", "PEQ_B"),
        derived_inputs=("PEQ_A/PEQ_B derived from team PEQ joined by event sides",),
        dependencies=(),
        compute_fn=f_PEI_matchup,
        output_cols=("PEI_matchup",),
    ),
    "POSW": _make_feature(
        name="POSW",
        grain="matchup",
        required_inputs=("POSW_A", "POSW_B"),
        derived_inputs=("POSW_A/POSW_B derived from team POSW joined by event sides",),
        dependencies=(),
        compute_fn=f_POSW_matchup,
        output_cols=("POSW",),
    ),
    "odi_diff": _make_feature(
        name="odi_diff",
        grain="matchup",
        required_inputs=("ODI_A", "ODI_B"),
        derived_inputs=(),
        dependencies=("ODI_A", "ODI_B"),
        compute_fn=f_odi_diff,
        output_cols=("ODI_diff",),
    ),
    "odi_sum": _make_feature(
        name="odi_sum",
        grain="matchup",
        required_inputs=("ODI_A", "ODI_B"),
        derived_inputs=(),
        dependencies=("ODI_A", "ODI_B"),
        compute_fn=f_odi_sum,
        output_cols=("ODI_sum",),
    ),
    "MTI": _make_feature(
        name="MTI",
        grain="matchup",
        required_inputs=("ANE_A", "ANE_B"),
        derived_inputs=("ANE_A/ANE_B derived from team ANE joined by event sides",),
        dependencies=(),
        compute_fn=f_MTI,
        output_cols=("MTI",),
    ),
    "SCI": _make_feature(
        name="SCI",
        grain="matchup",
        required_inputs=("SCI_A", "SCI_B"),
        derived_inputs=("SCI_A/SCI_B derived from team SCI joined by event sides",),
        dependencies=(),
        compute_fn=f_SCI_matchup,
        output_cols=("SCI",),
    ),
    "factor_ODIs": _make_feature(
        name="factor_ODIs",
        grain="matchup",
        required_inputs=(
            "eFG_ODI_A",
            "eFG_ODI_B",
            "TO_ODI_A",
            "TO_ODI_B",
            "ORB_ODI_A",
            "ORB_ODI_B",
            "FTR_ODI_A",
            "FTR_ODI_B",
        ),
        derived_inputs=("factor ODI A/B columns derived from team factor_ODIs joined by event sides",),
        dependencies=(),
        compute_fn=f_factor_ODIs_AB,
        output_cols=(
            "eFG_ODI_A",
            "eFG_ODI_B",
            "TO_ODI_A",
            "TO_ODI_B",
            "ORB_ODI_A",
            "ORB_ODI_B",
            "FTR_ODI_A",
            "FTR_ODI_B",
        ),
    ),
    "factor_ODIs_diffs_sums": _make_feature(
        name="factor_ODIs_diffs_sums",
        grain="matchup",
        required_inputs=(
            "eFG_ODI_A",
            "eFG_ODI_B",
            "TO_ODI_A",
            "TO_ODI_B",
            "ORB_ODI_A",
            "ORB_ODI_B",
            "FTR_ODI_A",
            "FTR_ODI_B",
        ),
        derived_inputs=(),
        dependencies=("factor_ODIs",),
        compute_fn=f_factor_ODIs_diffs_sums,
        output_cols=(
            "eFG_ODI_diff",
            "eFG_ODI_sum",
            "TO_ODI_diff",
            "TO_ODI_sum",
            "ORB_ODI_diff",
            "ORB_ODI_sum",
            "FTR_ODI_diff",
            "FTR_ODI_sum",
        ),
    ),
}


def get_registry(grain: FeatureGrain) -> dict[str, FeatureSpec]:
    if grain == "team_game":
        return TEAM_GAME_FEATURE_REGISTRY
    if grain == "matchup":
        return MATCHUP_FEATURE_REGISTRY
    raise ValueError(f"Unsupported grain: {grain}")


def registry_feature_names(grain: FeatureGrain) -> list[str]:
    return list(get_registry(grain).keys())


def registry_output_columns(grain: FeatureGrain) -> list[str]:
    cols: list[str] = []
    for spec in get_registry(grain).values():
        for col in spec.output_cols:
            if col not in cols:
                cols.append(col)
    return cols
