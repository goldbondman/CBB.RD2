from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    ModelLabConfig,
    canonicalize_game_id,
    dataframe_nan_report,
    derive_season_id,
)


@dataclass
class FrameBuildResult:
    spread_frame: pd.DataFrame
    total_frame: pd.DataFrame
    ml_frame: pd.DataFrame
    blocked_reasons: list[str]
    source_paths: dict[str, str]
    frame_row_counts: dict[str, int]
    nan_report: dict[str, dict[str, dict[str, float]]]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _normalize_binary(series: pd.Series) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    out = out.fillna(0)
    return (out > 0).astype(int)


def _dedupe_latest(df: pd.DataFrame, key_col: str, time_cols: list[str]) -> pd.DataFrame:
    if df.empty or key_col not in df.columns:
        return df

    work = df.copy()
    sort_cols = []
    for col in time_cols:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce", utc=True)
            sort_cols.append(col)

    if sort_cols:
        work = work.sort_values(sort_cols)
    work = work.drop_duplicates(subset=[key_col], keep="last")
    return work


def _coalesce_columns(df: pd.DataFrame, targets: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for target, candidates in targets.items():
        if target in out.columns:
            base = out[target]
        else:
            base = pd.Series(pd.NA, index=out.index)

        for col in candidates:
            if col not in out.columns:
                continue
            base = base.where(base.notna(), out[col])

        out[target] = base
    return out


def _build_labels(config: ModelLabConfig, blocked: list[str]) -> pd.DataFrame:
    sources: list[tuple[str, Path]] = [
        ("results_log_graded", config.repo_root / config.results_graded_path),
        ("results_log", config.repo_root / config.results_log_path),
        ("backtest_training_data", config.repo_root / config.backtest_training_path),
        ("backtest_results_latest", config.repo_root / config.backtest_results_path),
    ]

    rows: list[pd.DataFrame] = []
    for source_name, path in sources:
        df = _read_csv(path)
        if df.empty:
            blocked.append(f"labels_missing:{source_name}")
            continue

        df = _coalesce_columns(
            df,
            {
                "event_id": ["game_id"],
                "game_id": ["event_id"],
                "game_datetime_utc": ["game_datetime"],
                "game_date": ["date"],
                "actual_margin": ["actual_spread"],
                "actual_total": [],
                "home_won": [],
                "neutral_site": [],
            },
        )

        if "home_won" not in df.columns or df["home_won"].isna().all():
            if "actual_margin" in df.columns:
                margin = pd.to_numeric(df["actual_margin"], errors="coerce")
                df["home_won"] = (margin > 0).astype("Int64")

        df["game_id"] = df["game_id"].map(canonicalize_game_id)
        df["event_id"] = df["event_id"].map(canonicalize_game_id)
        df["game_id"] = df["game_id"].where(df["game_id"].astype(str) != "", df["event_id"])
        df["event_id"] = df["event_id"].where(df["event_id"].astype(str) != "", df["game_id"])

        use_cols = [
            "game_id",
            "event_id",
            "game_datetime_utc",
            "game_date",
            "neutral_site",
            "actual_margin",
            "actual_total",
            "home_won",
        ]
        for col in use_cols:
            if col not in df.columns:
                df[col] = pd.NA

        picked = df[use_cols].copy()
        picked["label_source"] = source_name
        picked["source_rank"] = len(rows)
        rows.append(picked)

    if not rows:
        return pd.DataFrame(
            columns=[
                "game_id",
                "event_id",
                "game_datetime_utc",
                "game_date",
                "neutral_site",
                "actual_margin",
                "actual_total",
                "home_won",
            ]
        )

    merged = pd.concat(rows, ignore_index=True)
    merged["game_datetime_utc"] = pd.to_datetime(merged["game_datetime_utc"], utc=True, errors="coerce")
    merged = merged.sort_values(["source_rank", "game_datetime_utc"], na_position="last")

    def _first_non_null(group: pd.DataFrame, col: str) -> Any:
        s = group[col]
        s = s[s.notna()]
        return s.iloc[0] if not s.empty else pd.NA

    grouped_rows: list[dict[str, Any]] = []
    for game_id, group in merged.groupby("game_id", dropna=False):
        if not game_id:
            continue
        row = {
            "game_id": game_id,
            "event_id": _first_non_null(group, "event_id"),
            "game_datetime_utc": _first_non_null(group, "game_datetime_utc"),
            "game_date": _first_non_null(group, "game_date"),
            "neutral_site": _first_non_null(group, "neutral_site"),
            "actual_margin": pd.to_numeric(_first_non_null(group, "actual_margin"), errors="coerce"),
            "actual_total": pd.to_numeric(_first_non_null(group, "actual_total"), errors="coerce"),
            "home_won": pd.to_numeric(_first_non_null(group, "home_won"), errors="coerce"),
        }
        grouped_rows.append(row)

    out = pd.DataFrame(grouped_rows)
    if out.empty:
        return out

    out["home_won"] = pd.to_numeric(out["home_won"], errors="coerce").astype("Float64")
    out["neutral_site"] = _normalize_binary(out["neutral_site"])
    out["season_id"] = derive_season_id(out["game_datetime_utc"])
    return out


def _build_market_lines(config: ModelLabConfig, blocked: list[str]) -> pd.DataFrame:
    sources: list[tuple[str, Path]] = [
        ("market_lines_closing", config.repo_root / config.market_closing_path),
        ("market_lines", config.repo_root / config.market_lines_path),
        ("games", config.repo_root / config.games_path),
    ]

    rows: list[pd.DataFrame] = []
    for source_name, path in sources:
        df = _read_csv(path)
        if df.empty:
            blocked.append(f"market_missing:{source_name}")
            continue

        df = _coalesce_columns(
            df,
            {
                "event_id": ["game_id"],
                "game_id": ["event_id"],
                "game_datetime_utc": ["game_datetime", "final_score_ts"],
                "game_date": ["date"],
                "spread_open": ["home_spread_open"],
                "spread_close": ["home_spread_current", "home_spread", "spread"],
                "spread_line": ["home_spread_current", "home_spread", "spread", "spread_line"],
                "total_open": ["total_open"],
                "total_close": ["total_current", "over_under", "total", "total_line"],
                "total_line": ["total_current", "over_under", "total", "total_line"],
                "home_ml": ["home_ml"],
                "away_ml": ["away_ml"],
            },
        )

        for col in [
            "spread_open",
            "spread_close",
            "spread_line",
            "total_open",
            "total_close",
            "total_line",
            "home_ml",
            "away_ml",
        ]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["game_id"] = df["game_id"].map(canonicalize_game_id)
        df["event_id"] = df["event_id"].map(canonicalize_game_id)
        df["game_id"] = df["game_id"].where(df["game_id"].astype(str) != "", df["event_id"])

        keep_cols = [
            "game_id",
            "event_id",
            "game_datetime_utc",
            "game_date",
            "spread_open",
            "spread_close",
            "spread_line",
            "total_open",
            "total_close",
            "total_line",
            "home_ml",
            "away_ml",
        ]
        for col in keep_cols:
            if col not in df.columns:
                df[col] = pd.NA

        picked = df[keep_cols].copy()
        picked["market_source"] = source_name
        picked["source_rank"] = len(rows)
        rows.append(picked)

    if not rows:
        return pd.DataFrame(
            columns=[
                "game_id",
                "event_id",
                "game_datetime_utc",
                "game_date",
                "spread_open",
                "spread_close",
                "spread_line",
                "total_open",
                "total_close",
                "total_line",
                "home_ml",
                "away_ml",
            ]
        )

    merged = pd.concat(rows, ignore_index=True)
    merged["game_datetime_utc"] = pd.to_datetime(merged["game_datetime_utc"], utc=True, errors="coerce")
    merged = merged.sort_values(["source_rank", "game_datetime_utc"], na_position="last")

    out = _dedupe_latest(merged, "game_id", ["game_datetime_utc"])
    out = out.drop(columns=[c for c in ["market_source", "source_rank"] if c in out.columns])
    return out


def _base_from_matchup(config: ModelLabConfig, blocked: list[str]) -> pd.DataFrame:
    matchup_path = config.repo_root / config.matchup_metrics_path
    matchup = _read_csv(matchup_path)
    if matchup.empty:
        blocked.append("matchup_metrics_missing")
        return pd.DataFrame()

    if "event_id" not in matchup.columns:
        blocked.append("matchup_metrics_missing:event_id")
        return pd.DataFrame()

    base = matchup.copy()
    base["event_id"] = base["event_id"].map(canonicalize_game_id)
    base["game_id"] = base["event_id"]
    if "game_datetime_utc" in base.columns:
        base["game_datetime_utc"] = pd.to_datetime(base["game_datetime_utc"], utc=True, errors="coerce")
    else:
        base["game_datetime_utc"] = pd.NaT

    base["season_id"] = derive_season_id(base["game_datetime_utc"], base.get("season"))
    if "neutral_site" not in base.columns:
        base["neutral_site"] = 0
    base["neutral_site"] = _normalize_binary(base["neutral_site"])

    for col in ["home_team_id", "away_team_id"]:
        if col in base.columns:
            base[col] = base[col].astype(str).str.strip()

    base = _dedupe_latest(base, "game_id", ["game_datetime_utc"])
    return base


def build_frames(config: ModelLabConfig) -> FrameBuildResult:
    blocked: list[str] = []
    source_paths = {
        "matchup_metrics": str((config.repo_root / config.matchup_metrics_path).resolve()),
        "team_game_metrics": str((config.repo_root / config.team_game_metrics_path).resolve()),
        "results_log_graded": str((config.repo_root / config.results_graded_path).resolve()),
        "results_log": str((config.repo_root / config.results_log_path).resolve()),
        "backtest_training_data": str((config.repo_root / config.backtest_training_path).resolve()),
        "backtest_results_latest": str((config.repo_root / config.backtest_results_path).resolve()),
        "market_lines_closing": str((config.repo_root / config.market_closing_path).resolve()),
        "market_lines": str((config.repo_root / config.market_lines_path).resolve()),
        "games": str((config.repo_root / config.games_path).resolve()),
    }

    base = _base_from_matchup(config, blocked)
    if base.empty:
        empty = pd.DataFrame()
        return FrameBuildResult(
            spread_frame=empty,
            total_frame=empty,
            ml_frame=empty,
            blocked_reasons=blocked,
            source_paths=source_paths,
            frame_row_counts={"spread_frame": 0, "total_frame": 0, "ml_frame": 0},
            nan_report={"spread_frame": {}, "total_frame": {}, "ml_frame": {}},
        )

    labels = _build_labels(config, blocked)
    market = _build_market_lines(config, blocked)

    work = base.merge(
        labels,
        on="game_id",
        how="left",
        suffixes=("", "_lbl"),
    )
    for col in ["event_id", "game_datetime_utc", "game_date", "neutral_site", "season_id"]:
        lbl_col = f"{col}_lbl"
        if lbl_col in work.columns:
            work[col] = work[col].where(work[col].notna(), work[lbl_col])
            work = work.drop(columns=[lbl_col])

    work = work.merge(
        market,
        on="game_id",
        how="left",
        suffixes=("", "_mkt"),
    )
    for col in ["event_id", "game_datetime_utc", "game_date"]:
        mkt_col = f"{col}_mkt"
        if mkt_col in work.columns:
            work[col] = work[col].where(work[col].notna(), work[mkt_col])
            work = work.drop(columns=[mkt_col])

    if "event_id" not in work.columns:
        work["event_id"] = work["game_id"]
    work["event_id"] = work["event_id"].map(canonicalize_game_id)
    work["game_id"] = work["game_id"].map(canonicalize_game_id)
    work["game_datetime_utc"] = pd.to_datetime(work.get("game_datetime_utc"), utc=True, errors="coerce")
    work["season_id"] = derive_season_id(work["game_datetime_utc"], work.get("season_id"))
    work["neutral_site"] = _normalize_binary(work.get("neutral_site", pd.Series(0, index=work.index)))

    label_cols = ["actual_margin", "actual_total", "home_won"]
    market_cols = [
        "spread_open",
        "spread_close",
        "spread_line",
        "total_open",
        "total_close",
        "total_line",
        "home_ml",
        "away_ml",
    ]

    for col in label_cols + market_cols:
        if col not in work.columns:
            work[col] = pd.NA

    for col in ["actual_margin", "actual_total", "home_won"] + market_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    if "game_date" not in work.columns:
        work["game_date"] = pd.to_datetime(work["game_datetime_utc"], utc=True, errors="coerce").dt.date.astype(str)

    key_cols = [
        "season_id",
        "game_id",
        "event_id",
        "game_datetime_utc",
        "game_date",
        "home_team_id",
        "away_team_id",
        "neutral_site",
    ]
    for col in key_cols:
        if col not in work.columns:
            work[col] = pd.NA

    spread_frame = work[work["actual_margin"].notna()].copy()
    total_frame = work[work["actual_total"].notna()].copy()
    ml_frame = work[work["home_won"].notna()].copy()

    sort_cols = [c for c in ["season_id", "game_datetime_utc", "game_id"] if c in spread_frame.columns]
    if sort_cols:
        spread_frame = spread_frame.sort_values(sort_cols).reset_index(drop=True)
        total_frame = total_frame.sort_values(sort_cols).reset_index(drop=True)
        ml_frame = ml_frame.sort_values(sort_cols).reset_index(drop=True)

    frame_row_counts = {
        "spread_frame": int(len(spread_frame)),
        "total_frame": int(len(total_frame)),
        "ml_frame": int(len(ml_frame)),
    }

    nan_report = {
        "spread_frame": dataframe_nan_report(spread_frame),
        "total_frame": dataframe_nan_report(total_frame),
        "ml_frame": dataframe_nan_report(ml_frame),
    }

    return FrameBuildResult(
        spread_frame=spread_frame,
        total_frame=total_frame,
        ml_frame=ml_frame,
        blocked_reasons=sorted(set(blocked)),
        source_paths=source_paths,
        frame_row_counts=frame_row_counts,
        nan_report=nan_report,
    )


def write_frames(result: FrameBuildResult, run_dir: Path) -> dict[str, str]:
    outputs = {
        "spread_frame": str((run_dir / "spread_frame.csv").resolve()),
        "total_frame": str((run_dir / "total_frame.csv").resolve()),
        "ml_frame": str((run_dir / "ml_frame.csv").resolve()),
    }

    result.spread_frame.to_csv(run_dir / "spread_frame.csv", index=False)
    result.total_frame.to_csv(run_dir / "total_frame.csv", index=False)
    result.ml_frame.to_csv(run_dir / "ml_frame.csv", index=False)
    return outputs
