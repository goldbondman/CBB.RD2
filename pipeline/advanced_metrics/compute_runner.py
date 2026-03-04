"""Compute runner for the registry-driven Feature Engine."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit_logger import build_nan_report, get_git_sha, utc_now_iso, write_run_manifest
from .cache_layer import load_cached_feature, save_cached_feature
from .feature_dag import resolve_execution_order
from .feature_registry import FeatureSpec, get_registry
from .integrity_gate import validate_feature_inputs
from .rolling_window_layer import add_leak_free_windows
from .shared_derivations import add_shared_derivations
from .starter_bench_helper import compute_starter_bench_features

REPO_ROOT = Path(__file__).resolve().parents[2]
TEAM_GAME_INPUT_PATH = REPO_ROOT / "data" / "team_game_logs.csv"
PLAYER_GAME_INPUT_PATH = REPO_ROOT / "data" / "player_game_logs.csv"
TEAM_GAME_OUTPUT_PATH = REPO_ROOT / "data" / "team_game_metrics.csv"
MATCHUP_OUTPUT_PATH = REPO_ROOT / "data" / "matchup_metrics.csv"
RUN_MANIFEST_PATH = REPO_ROOT / "data" / "feature_runs" / "feature_run_manifest.json"


def _normalize_id_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.lstrip("0")
    out = out.replace("", "0")
    return out


def _derive_season(dt_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(dt_series, utc=True, errors="coerce")
    season = dt.dt.year.where(dt.dt.month < 10, dt.dt.year + 1)
    return season.astype("Int64")


def _attach_team_pregame_baselines(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["team_id", "season", "game_datetime_utc", "event_id"]).reset_index(drop=True)
    grp = out.groupby(["team_id", "season"], sort=False)
    out["pre_NetRtg_season"] = grp["NetRtg"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding().mean())
    out["pre_poss_season"] = grp["poss"].transform(lambda s: pd.to_numeric(s, errors="coerce").shift(1).expanding().mean())
    return out


def _attach_opponent_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    opp_lookup = out[
        [
            "event_id",
            "team_id",
            "eFG",
            "TOV%",
            "ORB%",
            "FTr",
            "pre_NetRtg_season",
            "pre_poss_season",
        ]
    ].rename(
        columns={
            "team_id": "opponent_id",
            "eFG": "opp_eFG",
            "TOV%": "opp_TOV%",
            "ORB%": "opp_ORB%",
            "FTr": "opp_FTr",
            "pre_NetRtg_season": "opp_pre_NetRtg_season",
            "pre_poss_season": "opp_pre_poss_season",
        }
    )
    return out.merge(opp_lookup, on=["event_id", "opponent_id"], how="left")


def _merge_starter_bench(team_df: pd.DataFrame, player_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = team_df.copy()
    helper_cols = [
        "bench_minutes_share",
        "TS_bench",
        "TS_starters",
        "REB_rate_bench",
        "REB_rate_starters",
    ]
    blocked_columns: list[str] = []
    required = {"event_id", "team_id", "min", "pts", "fga", "fta", "reb"}
    missing = sorted(required - set(player_df.columns))
    if missing:
        blocked_columns = missing
        for col in helper_cols:
            out[col] = np.nan
        return out, blocked_columns

    sb = compute_starter_bench_features(player_df)
    out = out.merge(sb, on=["event_id", "team_id"], how="left")
    return out, blocked_columns


def _grain_key_columns(grain: str, df: pd.DataFrame) -> list[str]:
    if grain == "team_game":
        return [col for col in ("event_id", "team_id") if col in df.columns]
    if grain == "matchup":
        preferred = [col for col in ("event_id", "team_id_A", "team_id_B") if col in df.columns]
        return preferred if preferred else [col for col in ("event_id",) if col in df.columns]
    raise ValueError(f"Unsupported grain: {grain}")


def _season_values(df: pd.DataFrame) -> list[Any]:
    if "season" not in df.columns:
        return ["all"]
    vals = [v for v in df["season"].dropna().unique().tolist()]
    return sorted(vals) if vals else ["all"]


def _mask_for_season(df: pd.DataFrame, season_value: Any) -> pd.Series:
    if "season" not in df.columns or season_value == "all":
        return pd.Series(True, index=df.index)
    return df["season"] == season_value


def _execute_feature(
    df: pd.DataFrame,
    *,
    grain: str,
    spec: FeatureSpec,
    rebuild: bool,
    blocked: list[dict[str, Any]],
    cache_stats: dict[str, int],
) -> pd.DataFrame:
    blocked_names = {
        entry["feature_name"]
        for entry in blocked
        if entry.get("grain") == grain and isinstance(entry.get("feature_name"), str)
    }
    blocked_deps = sorted(dep for dep in spec.dependencies if dep in blocked_names)
    if blocked_deps:
        for col in spec.output_cols:
            if col not in df.columns:
                df[col] = np.nan
        blocked.append(
            {
                "grain": grain,
                "feature_name": spec.name,
                "missing_columns": [f"dependency_blocked:{dep}" for dep in blocked_deps],
            }
        )
        return df

    gate = validate_feature_inputs(df, spec)
    if gate.status == "BLOCKED":
        for col in spec.output_cols:
            if col not in df.columns:
                df[col] = np.nan
        blocked.append(
            {
                "grain": grain,
                "feature_name": spec.name,
                "missing_columns": list(gate.missing_columns),
            }
        )
        return df

    out = df.copy()
    keys = _grain_key_columns(grain, out)
    if not keys:
        raise ValueError(f"Missing key columns for grain={grain} feature={spec.name}")

    for output_col in spec.output_cols:
        if output_col not in out.columns:
            out[output_col] = np.nan

    for season_value in _season_values(out):
        mask = _mask_for_season(out, season_value)
        if not mask.any():
            continue

        season_slice = out.loc[mask].copy().reset_index()
        season_slice.rename(columns={"index": "_orig_index"}, inplace=True)
        for key_col in keys:
            season_slice[key_col] = season_slice[key_col].astype(str)

        cached = None
        if spec.cache.enabled and not rebuild:
            cached = load_cached_feature(
                grain=grain,
                feature_name=spec.name,
                season_id=season_value,
                version_hash=spec.cache.version_hash,
            )

        if cached is not None:
            required_cached = set(keys).union(spec.output_cols)
            if required_cached.issubset(set(cached.columns)):
                for key_col in keys:
                    cached[key_col] = cached[key_col].astype(str)
                merged = season_slice[keys].merge(cached[list(required_cached)], on=keys, how="left")
                for col in spec.output_cols:
                    out.loc[season_slice["_orig_index"], col] = pd.to_numeric(merged[col], errors="coerce").values
                cache_stats["hits"] += 1
                continue

        computed = spec.compute_fn(season_slice)
        if not isinstance(computed, pd.DataFrame):
            raise TypeError(f"Feature '{spec.name}' compute_fn must return a DataFrame")
        missing_outputs = [col for col in spec.output_cols if col not in computed.columns]
        if missing_outputs:
            raise ValueError(f"Feature '{spec.name}' missing output columns: {missing_outputs}")
        if len(computed) != len(season_slice):
            raise ValueError(f"Feature '{spec.name}' returned {len(computed)} rows for {len(season_slice)} input rows")

        for col in spec.output_cols:
            out.loc[season_slice["_orig_index"], col] = pd.to_numeric(computed[col], errors="coerce").values

        if spec.cache.enabled:
            cache_frame = season_slice[keys].copy()
            for col in spec.output_cols:
                cache_frame[col] = pd.to_numeric(computed[col], errors="coerce").values
            save_cached_feature(
                df=cache_frame,
                grain=grain,
                feature_name=spec.name,
                season_id=season_value,
                version_hash=spec.cache.version_hash,
                key_fields=spec.cache.key_fields,
            )
            cache_stats["writes"] += 1

        cache_stats["misses"] += 1

    return out


def _execute_registry_features(
    df: pd.DataFrame,
    *,
    grain: str,
    requested_features: list[str] | None,
    rebuild: bool,
    blocked: list[dict[str, Any]],
    cache_stats: dict[str, int],
) -> tuple[pd.DataFrame, list[str]]:
    registry = get_registry(grain)
    order = resolve_execution_order(registry, requested_features)
    out = df.copy()
    for feature_name in order:
        spec = registry[feature_name]
        out = _execute_feature(
            out,
            grain=grain,
            spec=spec,
            rebuild=rebuild,
            blocked=blocked,
            cache_stats=cache_stats,
        )
    return out, order


def _build_matchup_base(team_metrics: pd.DataFrame, team_feature_order: list[str]) -> pd.DataFrame:
    side_base = team_metrics.copy()
    side_base["home_away"] = side_base["home_away"].astype(str).str.lower()

    feature_cols: list[str] = []
    registry = get_registry("team_game")
    for name in team_feature_order:
        for col in registry[name].output_cols:
            if col in side_base.columns and col not in feature_cols:
                feature_cols.append(col)

    base_cols = [col for col in ("event_id", "game_datetime_utc", "season", "neutral_site") if col in side_base.columns]

    home = side_base.loc[side_base["home_away"] == "home", base_cols + ["team_id"] + feature_cols].copy()
    away = side_base.loc[side_base["home_away"] == "away", base_cols + ["team_id"] + feature_cols].copy()
    if home.empty or away.empty:
        cols = base_cols + ["team_id_A", "team_id_B", "home_team_id", "away_team_id"]
        return pd.DataFrame(columns=cols)

    home = home.rename(columns={"team_id": "team_id_A", **{c: f"{c}_A" for c in feature_cols}})
    away = away.rename(columns={"team_id": "team_id_B", **{c: f"{c}_B" for c in feature_cols}})

    matchup = home.merge(away, on=base_cols, how="inner")
    matchup["home_team_id"] = matchup["team_id_A"]
    matchup["away_team_id"] = matchup["team_id_B"]
    return matchup


def _prepare_inputs(
    *,
    season_id: int | None,
    limit_games: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    team_df = pd.read_csv(TEAM_GAME_INPUT_PATH, low_memory=False)
    player_df = pd.read_csv(PLAYER_GAME_INPUT_PATH, low_memory=False)

    required_team_cols = {
        "event_id",
        "game_datetime_utc",
        "team_id",
        "opponent_id",
        "home_away",
        "points_for",
        "points_against",
        "fgm",
        "fga",
        "tpm",
        "tpa",
        "ftm",
        "fta",
        "orb",
        "drb",
        "tov",
        "opp_fgm",
        "opp_fga",
        "opp_tpm",
        "opp_tpa",
        "opp_ftm",
        "opp_fta",
        "opp_orb",
        "opp_drb",
        "opp_tov",
    }
    missing_team = sorted(required_team_cols - set(team_df.columns))
    if missing_team:
        raise ValueError(f"Canonical team_game input is missing required columns: {missing_team}")

    for col in ("event_id", "team_id", "opponent_id"):
        if col in team_df.columns:
            team_df[col] = _normalize_id_series(team_df[col])
    for col in ("event_id", "team_id", "athlete_id"):
        if col in player_df.columns:
            player_df[col] = _normalize_id_series(player_df[col])

    team_df["game_datetime_utc"] = pd.to_datetime(team_df["game_datetime_utc"], utc=True, errors="coerce")
    player_df["game_datetime_utc"] = pd.to_datetime(player_df["game_datetime_utc"], utc=True, errors="coerce")
    team_df["season"] = _derive_season(team_df["game_datetime_utc"])

    if season_id is not None:
        team_df = team_df.loc[team_df["season"] == season_id].copy()
        allowed_events = set(team_df["event_id"].astype(str))
        player_df = player_df.loc[player_df["event_id"].astype(str).isin(allowed_events)].copy()

    if limit_games is not None:
        ordered = team_df.sort_values(["game_datetime_utc", "event_id"], na_position="last")
        event_ids = ordered["event_id"].drop_duplicates().head(limit_games)
        keep = set(event_ids.astype(str))
        team_df = team_df.loc[team_df["event_id"].astype(str).isin(keep)].copy()
        player_df = player_df.loc[player_df["event_id"].astype(str).isin(keep)].copy()

    return team_df, player_df


def _prepare_team_frame(team_df: pd.DataFrame, player_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = team_df.copy()
    out = add_shared_derivations(out)
    out = _attach_team_pregame_baselines(out)
    out = _attach_opponent_context(out)
    out, missing_player_cols = _merge_starter_bench(out, player_df)
    return out, missing_player_cols


def _feature_output_columns(grain: str, feature_names: list[str]) -> list[str]:
    registry = get_registry(grain)
    cols: list[str] = []
    for name in feature_names:
        for col in registry[name].output_cols:
            if col not in cols:
                cols.append(col)
    return cols


def compute_features(
    season_id: int | None = None,
    limit_games: int | None = None,
    rebuild: bool = False,
    *,
    team_features: list[str] | None = None,
    matchup_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute team-game and matchup metrics from canonical box score tables."""
    run_started_at = utc_now_iso()
    blocked: list[dict[str, Any]] = []
    cache_stats: dict[str, int] = {"hits": 0, "misses": 0, "writes": 0}

    team_source, player_source = _prepare_inputs(season_id=season_id, limit_games=limit_games)
    team_frame, missing_player_cols = _prepare_team_frame(team_source, player_source)

    if missing_player_cols:
        blocked.append(
            {
                "grain": "team_game",
                "feature_name": "starter_bench_helper",
                "missing_columns": missing_player_cols,
            }
        )

    team_with_features, team_feature_order = _execute_registry_features(
        team_frame,
        grain="team_game",
        requested_features=team_features,
        rebuild=rebuild,
        blocked=blocked,
        cache_stats=cache_stats,
    )
    team_metric_cols = _feature_output_columns("team_game", team_feature_order)
    team_with_windows = add_leak_free_windows(
        team_with_features,
        metric_columns=team_metric_cols,
        group_columns=("team_id",),
        season_column="season",
        date_column="game_datetime_utc",
        event_column="event_id",
    )

    matchup_base = _build_matchup_base(team_with_windows, team_feature_order)
    matchup_with_features, matchup_feature_order = _execute_registry_features(
        matchup_base,
        grain="matchup",
        requested_features=matchup_features,
        rebuild=rebuild,
        blocked=blocked,
        cache_stats=cache_stats,
    )
    matchup_metric_cols = _feature_output_columns("matchup", matchup_feature_order)
    matchup_with_windows = add_leak_free_windows(
        matchup_with_features,
        metric_columns=matchup_metric_cols,
        group_columns=("team_id_A",),
        season_column="season",
        date_column="game_datetime_utc",
        event_column="event_id",
    )

    TEAM_GAME_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    MATCHUP_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    team_with_windows.to_csv(TEAM_GAME_OUTPUT_PATH, index=False)
    matchup_with_windows.to_csv(MATCHUP_OUTPUT_PATH, index=False)

    run_finished_at = utc_now_iso()
    run_manifest = {
        "run_started_at_utc": run_started_at,
        "run_finished_at_utc": run_finished_at,
        "git_sha": get_git_sha(REPO_ROOT),
        "inputs": {
            "team_game_path": str(TEAM_GAME_INPUT_PATH),
            "player_game_path": str(PLAYER_GAME_INPUT_PATH),
            "team_game_rows": int(len(team_source)),
            "player_game_rows": int(len(player_source)),
            "team_game_columns": list(team_source.columns),
            "player_game_columns": list(player_source.columns),
        },
        "parameters": {
            "season_id": season_id,
            "limit_games": limit_games,
            "rebuild": rebuild,
            "team_features": team_features,
            "matchup_features": matchup_features,
        },
        "execution": {
            "team_feature_order": team_feature_order,
            "matchup_feature_order": matchup_feature_order,
            "cache_stats": cache_stats,
        },
        "blocked": blocked,
        "nan_report": {
            "team_game_metrics": build_nan_report(team_with_windows, team_metric_cols),
            "matchup_metrics": build_nan_report(matchup_with_windows, matchup_metric_cols),
        },
        "outputs": {
            "team_game_metrics_path": str(TEAM_GAME_OUTPUT_PATH),
            "matchup_metrics_path": str(MATCHUP_OUTPUT_PATH),
            "team_game_metrics_rows": int(len(team_with_windows)),
            "matchup_metrics_rows": int(len(matchup_with_windows)),
        },
    }
    write_run_manifest(RUN_MANIFEST_PATH, run_manifest)

    return team_with_windows, matchup_with_windows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute registry-driven advanced features")
    parser.add_argument("--season-id", type=int, default=None, help="Optional season filter (e.g., 2025)")
    parser.add_argument("--limit-games", type=int, default=None, help="Optional limit on number of games")
    parser.add_argument("--rebuild", action="store_true", help="Ignore cache and rebuild all feature outputs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compute_features(
        season_id=args.season_id,
        limit_games=args.limit_games,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
