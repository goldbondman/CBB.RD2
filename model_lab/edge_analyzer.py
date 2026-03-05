from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_MODEL_NAMES,
    ModelLabConfig,
    canonicalize_game_id,
    dataframe_nan_report,
)
from .data_builder import build_frames, write_frames
from .metrics import american_odds_profit, clv_spread, clv_total
from .model_wrappers import load_predictions
from .splits import Fold, build_rolling_folds


EDGE_BINS = [0.0, 1.0, 2.0, 3.0, 4.0, float("inf")]
EDGE_LABELS = ["0-1", "1-2", "2-3", "3-4", "4+"]
CORE_SNAPSHOT_FEATURES = ["ODI_diff", "PEQ", "SVI", "WL", "VOL", "TIN", "MTI", "SCI", "DPC", "PXP"]


@dataclass
class EdgeAnalyzerResult:
    run_id: str
    market: str
    model_name: str
    rows_analyzed: int
    edge_bucket_report_path: Path
    segment_report_path: Path
    worst_misses_path: Path
    exec_summary_path: Path
    run_manifest_path: Path
    blocked_reasons: list[str]
    blocked_segments: list[dict[str, Any]]


def _read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_model_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name)).strip("_") or "model"


def _coalesce(out: pd.DataFrame, target: str, candidates: list[str]) -> pd.Series:
    base = out[target] if target in out.columns else pd.Series(pd.NA, index=out.index)
    for col in candidates:
        if col in out.columns:
            base = base.where(base.notna(), out[col])
    return base


def _load_or_build_market_frame(
    run_dir: Path,
    config: ModelLabConfig,
    market: str,
    blocked: list[str],
) -> pd.DataFrame:
    frame_path = run_dir / f"{market}_frame.csv"
    frame = _read_csv_optional(frame_path)
    if not frame.empty:
        frame["game_id"] = frame["game_id"].map(canonicalize_game_id)
        return frame

    frames = build_frames(config)
    write_frames(frames, run_dir)
    if market == "spread":
        frame = frames.spread_frame.copy()
    elif market == "total":
        frame = frames.total_frame.copy()
    elif market == "ml":
        frame = frames.ml_frame.copy()
    else:
        raise ValueError(f"Unsupported market: {market}")
    if frame.empty:
        blocked.append(f"edge_analyzer_empty_frame:{market}")
    frame["game_id"] = frame["game_id"].map(canonicalize_game_id)
    return frame


def _load_ensemble_predictions(config: ModelLabConfig) -> pd.DataFrame:
    sources: list[Path] = [
        (config.repo_root / config.ensemble_latest_path).resolve(),
        (config.repo_root / config.results_log_path).resolve(),
        (config.repo_root / config.results_graded_path).resolve(),
        (config.repo_root / config.backtest_results_path).resolve(),
    ]
    parts: list[pd.DataFrame] = []
    for path in sources:
        df = _read_csv_optional(path)
        if df.empty:
            continue
        has_spread = "ens_spread" in df.columns and pd.to_numeric(df["ens_spread"], errors="coerce").notna().any()
        has_total = "ens_total" in df.columns and pd.to_numeric(df["ens_total"], errors="coerce").notna().any()
        if not has_spread and not has_total:
            continue

        out = pd.DataFrame(index=df.index)
        out["event_id"] = _coalesce(df, "event_id", ["game_id"])
        out["game_id"] = _coalesce(df, "game_id", ["event_id"])
        out["game_datetime_utc"] = pd.to_datetime(_coalesce(df, "game_datetime_utc", ["game_datetime"]), utc=True, errors="coerce")
        out["game_date"] = _coalesce(df, "game_date", ["date"])
        out["home_team_id"] = _coalesce(df, "home_team_id", [])
        out["away_team_id"] = _coalesce(df, "away_team_id", [])
        out["neutral_site"] = pd.to_numeric(_coalesce(df, "neutral_site", []), errors="coerce")
        out["pred_spread"] = pd.to_numeric(_coalesce(df, "ens_spread", []), errors="coerce")
        out["pred_total"] = pd.to_numeric(_coalesce(df, "ens_total", []), errors="coerce")
        out["pred_conf"] = pd.to_numeric(_coalesce(df, "ens_confidence", []), errors="coerce")
        out["source_file"] = str(path)
        out["model_name"] = "ensemble"
        out["game_id"] = out["game_id"].map(canonicalize_game_id)
        out["event_id"] = out["event_id"].map(canonicalize_game_id)
        out = out[out["game_id"].astype(str) != ""].copy()
        parts.append(out)

    if not parts:
        return pd.DataFrame()
    merged = pd.concat(parts, ignore_index=True)
    merged = merged.sort_values(["game_datetime_utc"], na_position="last")
    merged = merged.drop_duplicates(subset=["game_id"], keep="last")
    return merged.reset_index(drop=True)


def _load_model_predictions(config: ModelLabConfig, model_name: str) -> pd.DataFrame:
    if model_name == "ensemble":
        return _load_ensemble_predictions(config)
    return load_predictions(config, model_name).copy()


def _attach_predictions(frame: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or preds.empty:
        return pd.DataFrame()

    keep_cols = [
        "game_id",
        "event_id",
        "game_datetime_utc",
        "game_date",
        "home_team_id",
        "away_team_id",
        "neutral_site",
        "pred_spread",
        "pred_total",
        "pred_conf",
        "source_file",
    ]
    local_preds = preds.copy()
    for col in keep_cols:
        if col not in local_preds.columns:
            local_preds[col] = pd.NA
    local_preds["game_id"] = local_preds["game_id"].map(canonicalize_game_id)
    local_preds = local_preds[local_preds["game_id"].astype(str) != ""].copy()
    local_preds["game_datetime_utc"] = pd.to_datetime(local_preds["game_datetime_utc"], utc=True, errors="coerce")
    local_preds = local_preds.sort_values(["game_datetime_utc"], na_position="last")
    local_preds = local_preds.drop_duplicates(subset=["game_id"], keep="last")

    merged = frame.merge(local_preds[keep_cols], on="game_id", how="inner", suffixes=("", "_predsrc"))
    for col in ["event_id", "game_datetime_utc", "game_date", "home_team_id", "away_team_id", "neutral_site"]:
        src = f"{col}_predsrc"
        if src in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[src])
            merged.drop(columns=[src], inplace=True)
    merged["game_id"] = merged["game_id"].map(canonicalize_game_id)
    merged = merged[merged["game_id"].astype(str) != ""].copy()
    return merged.reset_index(drop=True)


def _attach_fold_test_rows(df: pd.DataFrame, frame: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    if df.empty or frame.empty or not folds:
        return pd.DataFrame()

    pieces: list[pd.DataFrame] = []
    frame_game = frame.copy()
    frame_game["game_id"] = frame_game["game_id"].map(canonicalize_game_id)
    for fold in folds:
        test_index = [i for i in fold.test_index if i in frame_game.index]
        if not test_index:
            continue
        game_ids = set(frame_game.loc[test_index, "game_id"].astype(str).tolist())
        fold_df = df[df["game_id"].astype(str).isin(game_ids)].copy()
        if fold_df.empty:
            continue
        fold_df["fold_id"] = fold.fold_id
        fold_df["fold_mode"] = fold.mode
        pieces.append(fold_df)

    if not pieces:
        return pd.DataFrame()
    out = pd.concat(pieces, ignore_index=True)
    out["game_datetime_utc"] = pd.to_datetime(out.get("game_datetime_utc"), utc=True, errors="coerce")
    return out


def _american_to_prob(odds: pd.Series) -> pd.Series:
    value = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(np.nan, index=value.index, dtype=float)
    pos = value > 0
    neg = value < 0
    out.loc[pos] = 100.0 / (value.loc[pos] + 100.0)
    out.loc[neg] = value.loc[neg].abs() / (value.loc[neg].abs() + 100.0)
    return out


def _resolve_ml_implied_home(df: pd.DataFrame) -> pd.Series:
    if "home_ml" not in df.columns or "away_ml" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    home_prob = _american_to_prob(df["home_ml"])
    away_prob = _american_to_prob(df["away_ml"])
    denom = home_prob + away_prob
    implied = home_prob / denom
    implied = implied.where(denom > 0)
    return implied


def _prepare_predictions(df: pd.DataFrame, market: str) -> pd.Series:
    pred_spread = pd.to_numeric(df.get("pred_spread"), errors="coerce")
    pred_total = pd.to_numeric(df.get("pred_total"), errors="coerce")
    pred_conf = pd.to_numeric(df.get("pred_conf"), errors="coerce")

    if market == "spread":
        return pred_spread
    if market == "total":
        return pred_total
    if market == "ml":
        from_spread = 1.0 / (1.0 + np.exp(pred_spread / 6.0))
        from_conf = pred_conf.copy()
        outside = (from_conf < 0.0) | (from_conf > 1.0)
        from_conf.loc[outside] = 1.0 / (1.0 + np.exp(-from_conf.loc[outside] / 10.0))
        out = from_spread.where(from_spread.notna(), from_conf)
        return out.clip(0.0, 1.0)
    raise ValueError(f"Unsupported market: {market}")


def _row_grade(test_df: pd.DataFrame, market: str, default_odds: int) -> pd.DataFrame:
    out = test_df.copy()
    out["model_pred"] = _prepare_predictions(out, market)

    out["outcome"] = np.nan
    out["profit"] = np.nan
    out["stake"] = np.nan
    out["edge_abs"] = np.nan
    out["abs_error"] = np.nan
    out["clv"] = np.nan
    out["pick_home"] = pd.NA
    out["pred_home_prob"] = np.nan
    out["implied_home_prob"] = np.nan
    out["selected_implied_prob"] = np.nan

    if market == "spread":
        yt = pd.to_numeric(out.get("actual_margin"), errors="coerce")
        line = pd.to_numeric(out.get("spread_line"), errors="coerce")
        pred = pd.to_numeric(out.get("model_pred"), errors="coerce")
        valid = yt.notna() & line.notna() & pred.notna()
        if valid.any():
            pick_home = pred < line
            ats_actual = yt + line
            push = ats_actual == 0
            wins = ((pick_home & (ats_actual > 0)) | ((~pick_home) & (ats_actual < 0)))
            out.loc[valid, "pick_home"] = pick_home.loc[valid].astype(int)
            out.loc[valid, "outcome"] = np.where(push.loc[valid], 0.5, np.where(wins.loc[valid], 1.0, 0.0))
            out.loc[valid, "stake"] = 1.0
            odds = pd.Series(default_odds, index=out.index, dtype=float)
            profits = []
            for idx in out.index[valid]:
                outcome = out.at[idx, "outcome"]
                if outcome == 0.5:
                    profits.append(0.0)
                else:
                    profits.append(american_odds_profit(1.0, float(odds.at[idx]), bool(outcome == 1.0)))
            out.loc[valid, "profit"] = profits
            out.loc[valid, "edge_abs"] = (pred.loc[valid] - line.loc[valid]).abs()
            out.loc[valid, "abs_error"] = (yt.loc[valid] - (-pred.loc[valid])).abs()

            spread_open = pd.to_numeric(out.get("spread_open"), errors="coerce")
            spread_close = pd.to_numeric(out.get("spread_close"), errors="coerce")
            clv = clv_spread(pd.to_numeric(out["pick_home"], errors="coerce"), spread_open, spread_close)
            out["clv"] = pd.to_numeric(clv, errors="coerce")
        return out

    if market == "total":
        yt = pd.to_numeric(out.get("actual_total"), errors="coerce")
        line = pd.to_numeric(out.get("total_line"), errors="coerce")
        pred = pd.to_numeric(out.get("model_pred"), errors="coerce")
        valid = yt.notna() & line.notna() & pred.notna()
        if valid.any():
            pick_over = pred > line
            push = yt == line
            wins = ((pick_over & (yt > line)) | ((~pick_over) & (yt < line)))
            out.loc[valid, "outcome"] = np.where(push.loc[valid], 0.5, np.where(wins.loc[valid], 1.0, 0.0))
            out.loc[valid, "stake"] = 1.0
            odds = pd.Series(default_odds, index=out.index, dtype=float)
            profits = []
            for idx in out.index[valid]:
                outcome = out.at[idx, "outcome"]
                if outcome == 0.5:
                    profits.append(0.0)
                else:
                    profits.append(american_odds_profit(1.0, float(odds.at[idx]), bool(outcome == 1.0)))
            out.loc[valid, "profit"] = profits
            out.loc[valid, "edge_abs"] = (pred.loc[valid] - line.loc[valid]).abs()
            out.loc[valid, "abs_error"] = (yt.loc[valid] - pred.loc[valid]).abs()

            total_open = pd.to_numeric(out.get("total_open"), errors="coerce")
            total_close = pd.to_numeric(out.get("total_close"), errors="coerce")
            clv = clv_total((pred > line).astype(int), total_open, total_close)
            out["clv"] = pd.to_numeric(clv, errors="coerce")
        return out

    if market == "ml":
        yt = pd.to_numeric(out.get("home_won"), errors="coerce")
        pred_prob = pd.to_numeric(out.get("model_pred"), errors="coerce").clip(0.0, 1.0)
        implied_home = _resolve_ml_implied_home(out)
        valid = yt.notna() & pred_prob.notna()
        if valid.any():
            pred_home = pred_prob >= 0.5
            outcome = (pred_home.astype(int) == yt.astype(int)).astype(float)
            out.loc[valid, "pred_home_prob"] = pred_prob.loc[valid]
            out.loc[valid, "outcome"] = outcome.loc[valid]
            out.loc[valid, "stake"] = 1.0
            out.loc[valid, "abs_error"] = (yt.loc[valid] - pred_prob.loc[valid]).abs()
            out.loc[valid, "implied_home_prob"] = implied_home.loc[valid]
            selected_implied = np.where(pred_home.loc[valid], implied_home.loc[valid], 1.0 - implied_home.loc[valid])
            out.loc[valid, "selected_implied_prob"] = selected_implied
            out.loc[valid & implied_home.notna(), "edge_abs"] = (
                pred_prob.loc[valid & implied_home.notna()] - implied_home.loc[valid & implied_home.notna()]
            ).abs() * 100.0

            home_ml = pd.to_numeric(out.get("home_ml"), errors="coerce")
            away_ml = pd.to_numeric(out.get("away_ml"), errors="coerce")
            chosen_odds = pd.Series(default_odds, index=out.index, dtype=float)
            has_home = home_ml.notna()
            has_away = away_ml.notna()
            choose_home = pred_home & has_home
            choose_away = (~pred_home) & has_away
            chosen_odds.loc[choose_home] = home_ml.loc[choose_home]
            chosen_odds.loc[choose_away] = away_ml.loc[choose_away]

            profits = []
            for idx in out.index[valid]:
                profits.append(
                    american_odds_profit(1.0, float(chosen_odds.at[idx]), bool(out.at[idx, "outcome"] == 1.0))
                )
            out.loc[valid, "profit"] = profits
        return out

    raise ValueError(f"Unsupported market: {market}")


def _aggregate_group_metrics(group: pd.DataFrame, min_n: int) -> dict[str, Any]:
    outcome = pd.to_numeric(group.get("outcome"), errors="coerce")
    n = int(len(group))
    graded_n = int(outcome.notna().sum())
    if graded_n == 0:
        hit_rate = float("nan")
    else:
        non_push = outcome[outcome != 0.5]
        hit_rate = float(non_push.mean()) if len(non_push) > 0 else float("nan")

    stake = pd.to_numeric(group.get("stake"), errors="coerce").fillna(0.0)
    profit = pd.to_numeric(group.get("profit"), errors="coerce").fillna(0.0)
    total_stake = float(stake.sum())
    roi = float(profit.sum() / total_stake) if total_stake > 0 else float("nan")

    edge = pd.to_numeric(group.get("edge_abs"), errors="coerce")
    abs_error = pd.to_numeric(group.get("abs_error"), errors="coerce")
    clv = pd.to_numeric(group.get("clv"), errors="coerce")

    return {
        "n": n,
        "graded_n": graded_n,
        "hit_rate": hit_rate,
        "roi": roi,
        "avg_edge": float(edge.mean(skipna=True)) if edge.notna().any() else float("nan"),
        "avg_abs_error": float(abs_error.mean(skipna=True)) if abs_error.notna().any() else float("nan"),
        "clv_mean": float(clv.mean(skipna=True)) if clv.notna().any() else float("nan"),
        "small_sample_flag": bool(n < min_n),
    }


def _missing_edge_columns(df: pd.DataFrame, market: str) -> list[str]:
    required = {
        "spread": ["spread_line"],
        "total": ["total_line"],
        "ml": ["home_ml", "away_ml"],
    }[market]
    missing: list[str] = []
    for col in required:
        if col not in df.columns:
            missing.append(col)
            continue
        if pd.to_numeric(df[col], errors="coerce").notna().sum() == 0:
            missing.append(col)
    return missing


def _edge_bucket_report(df: pd.DataFrame, run_id: str, market: str, model_name: str, min_n: int) -> tuple[pd.DataFrame, list[str]]:
    blocked: list[str] = []
    rows: list[dict[str, Any]] = []
    edge = pd.to_numeric(df.get("edge_abs"), errors="coerce")
    usable = df.loc[edge.notna()].copy()
    if usable.empty:
        missing = _missing_edge_columns(df, market)
        rows.append(
            {
                "run_id": run_id,
                "market": market,
                "model_name": model_name,
                "bucket_label": "ALL",
                "edge_lo": np.nan,
                "edge_hi": np.nan,
                "status": "BLOCKED",
                "blocked_reason": "edge_unavailable",
                "missing_columns": ",".join(missing),
                "n": 0,
                "graded_n": 0,
                "hit_rate": np.nan,
                "roi": np.nan,
                "avg_edge": np.nan,
                "avg_abs_error": np.nan,
                "clv_mean": np.nan,
                "small_sample_flag": True,
            }
        )
        blocked.append(f"edge_bucket_blocked:{market}:{model_name}:missing={','.join(missing)}")
        return pd.DataFrame(rows), blocked

    usable["edge_bucket"] = pd.cut(
        pd.to_numeric(usable["edge_abs"], errors="coerce"),
        bins=EDGE_BINS,
        labels=EDGE_LABELS,
        include_lowest=True,
        right=False,
    )
    for label, lo, hi in zip(EDGE_LABELS, EDGE_BINS[:-1], EDGE_BINS[1:]):
        group = usable.loc[usable["edge_bucket"].astype(str) == label].copy()
        agg = _aggregate_group_metrics(group, min_n)
        rows.append(
            {
                "run_id": run_id,
                "market": market,
                "model_name": model_name,
                "bucket_label": label,
                "edge_lo": lo,
                "edge_hi": hi if np.isfinite(hi) else np.nan,
                "status": "OK",
                "blocked_reason": "",
                "missing_columns": "",
                **agg,
            }
        )
    return pd.DataFrame(rows), blocked


def _quantile_segment(series: pd.Series, q: int, prefix: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = values.notna()
    if valid.sum() < q:
        return out
    labels = [f"{prefix}Q{i}" for i in range(1, q + 1)]
    try:
        binned = pd.qcut(values[valid], q=q, labels=labels, duplicates="drop")
    except Exception:
        return out
    out.loc[valid] = binned.astype(str)
    return out


def _segment_rows(
    df: pd.DataFrame,
    run_id: str,
    market: str,
    model_name: str,
    family: str,
    values: pd.Series | None,
    *,
    min_n: int,
    blocked_reason: str = "",
    missing_columns: list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    blocked_segments: list[dict[str, Any]] = []
    missing = missing_columns or []
    if values is None:
        rows.append(
            {
                "run_id": run_id,
                "market": market,
                "model_name": model_name,
                "segment_family": family,
                "segment_value": "ALL",
                "status": "BLOCKED",
                "blocked_reason": blocked_reason,
                "missing_columns": ",".join(missing),
                "n": 0,
                "graded_n": 0,
                "hit_rate": np.nan,
                "roi": np.nan,
                "avg_edge": np.nan,
                "avg_abs_error": np.nan,
                "clv_mean": np.nan,
                "small_sample_flag": True,
            }
        )
        blocked_segments.append(
            {
                "segment_family": family,
                "reason": blocked_reason,
                "missing_columns": missing,
            }
        )
        return rows, blocked_segments

    local = df.copy()
    local["_segment_value"] = values
    local = local[local["_segment_value"].notna()].copy()
    if local.empty:
        rows.append(
            {
                "run_id": run_id,
                "market": market,
                "model_name": model_name,
                "segment_family": family,
                "segment_value": "ALL",
                "status": "BLOCKED",
                "blocked_reason": blocked_reason or "no_rows_in_segment",
                "missing_columns": ",".join(missing),
                "n": 0,
                "graded_n": 0,
                "hit_rate": np.nan,
                "roi": np.nan,
                "avg_edge": np.nan,
                "avg_abs_error": np.nan,
                "clv_mean": np.nan,
                "small_sample_flag": True,
            }
        )
        blocked_segments.append(
            {
                "segment_family": family,
                "reason": blocked_reason or "no_rows_in_segment",
                "missing_columns": missing,
            }
        )
        return rows, blocked_segments

    for segment_value in sorted(local["_segment_value"].astype(str).unique().tolist()):
        group = local[local["_segment_value"].astype(str) == segment_value].copy()
        agg = _aggregate_group_metrics(group, min_n)
        rows.append(
            {
                "run_id": run_id,
                "market": market,
                "model_name": model_name,
                "segment_family": family,
                "segment_value": segment_value,
                "status": "OK",
                "blocked_reason": "",
                "missing_columns": "",
                **agg,
            }
        )
    return rows, blocked_segments


def _segment_report(df: pd.DataFrame, run_id: str, market: str, model_name: str, min_n: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    blocked_segments: list[dict[str, Any]] = []

    for family, col in [("VOL", "VOL"), ("MTI", "MTI"), ("SCI", "SCI")]:
        if col not in df.columns:
            r, b = _segment_rows(
                df,
                run_id,
                market,
                model_name,
                family,
                None,
                min_n=min_n,
                blocked_reason="missing_columns",
                missing_columns=[col],
            )
            rows.extend(r)
            blocked_segments.extend(b)
            continue
        values = _quantile_segment(df[col], 5, f"{family}_")
        if values.notna().sum() == 0:
            r, b = _segment_rows(
                df,
                run_id,
                market,
                model_name,
                family,
                None,
                min_n=min_n,
                blocked_reason="insufficient_non_null_for_quantiles",
                missing_columns=[],
            )
            rows.extend(r)
            blocked_segments.extend(b)
            continue
        r, b = _segment_rows(df, run_id, market, model_name, family, values, min_n=min_n)
        rows.extend(r)
        blocked_segments.extend(b)

    missing_loc = [c for c in ["home_away", "neutral_site"] if c not in df.columns]
    if missing_loc:
        r, b = _segment_rows(
            df,
            run_id,
            market,
            model_name,
            "location",
            None,
            min_n=min_n,
            blocked_reason="missing_columns",
            missing_columns=missing_loc,
        )
        rows.extend(r)
        blocked_segments.extend(b)
    else:
        neutral = pd.to_numeric(df["neutral_site"], errors="coerce").fillna(0)
        home_away = df["home_away"].astype(str).str.lower()
        values = pd.Series(pd.NA, index=df.index, dtype="object")
        values.loc[neutral == 1] = "neutral"
        values.loc[(neutral != 1) & (home_away.isin(["home", "h"]))] = "home"
        values.loc[(neutral != 1) & (home_away.isin(["away", "a"]))] = "away"
        r, b = _segment_rows(df, run_id, market, model_name, "location", values, min_n=min_n)
        rows.extend(r)
        blocked_segments.extend(b)

    if market == "spread":
        if "spread_line" not in df.columns:
            r, b = _segment_rows(
                df,
                run_id,
                market,
                model_name,
                "favorite_dog",
                None,
                min_n=min_n,
                blocked_reason="missing_columns",
                missing_columns=["spread_line"],
            )
            rows.extend(r)
            blocked_segments.extend(b)
        else:
            line = pd.to_numeric(df["spread_line"], errors="coerce")
            pick_home = pd.to_numeric(df["pick_home"], errors="coerce")
            values = pd.Series(pd.NA, index=df.index, dtype="object")
            pickem = line == 0
            favorite = ((pick_home == 1) & (line < 0)) | ((pick_home == 0) & (line > 0))
            underdog = ((pick_home == 1) & (line > 0)) | ((pick_home == 0) & (line < 0))
            values.loc[pickem] = "pickem"
            values.loc[favorite] = "favorite"
            values.loc[underdog] = "underdog"
            r, b = _segment_rows(df, run_id, market, model_name, "favorite_dog", values, min_n=min_n)
            rows.extend(r)
            blocked_segments.extend(b)
    elif market == "ml":
        missing_ml = [c for c in ["home_ml", "away_ml"] if c not in df.columns]
        implied = pd.to_numeric(df.get("selected_implied_prob"), errors="coerce")
        if missing_ml or implied.notna().sum() == 0:
            r, b = _segment_rows(
                df,
                run_id,
                market,
                model_name,
                "favorite_dog",
                None,
                min_n=min_n,
                blocked_reason="missing_columns" if missing_ml else "missing_implied_prob_rows",
                missing_columns=missing_ml if missing_ml else ["home_ml", "away_ml"],
            )
            rows.extend(r)
            blocked_segments.extend(b)
        else:
            values = pd.cut(
                implied,
                bins=[0.0, 0.4, 0.5, 0.6, 1.000001],
                labels=["P<0.40", "P0.40-0.50", "P0.50-0.60", "P0.60+"],
                include_lowest=True,
                right=False,
            ).astype("object")
            r, b = _segment_rows(df, run_id, market, model_name, "favorite_dog", values, min_n=min_n)
            rows.extend(r)
            blocked_segments.extend(b)
    else:
        r, b = _segment_rows(
            df,
            run_id,
            market,
            model_name,
            "favorite_dog",
            None,
            min_n=min_n,
            blocked_reason="not_applicable_for_market",
            missing_columns=[],
        )
        rows.extend(r)
        blocked_segments.extend(b)

    if market == "total":
        if "total_line" not in df.columns:
            r, b = _segment_rows(
                df,
                run_id,
                market,
                model_name,
                "total_regime",
                None,
                min_n=min_n,
                blocked_reason="missing_columns",
                missing_columns=["total_line"],
            )
            rows.extend(r)
            blocked_segments.extend(b)
        else:
            values = _quantile_segment(df["total_line"], 3, "total_")
            mapper = {"total_Q1": "low", "total_Q2": "med", "total_Q3": "high"}
            values = values.map(mapper).astype("object")
            if values.notna().sum() == 0:
                r, b = _segment_rows(
                    df,
                    run_id,
                    market,
                    model_name,
                    "total_regime",
                    None,
                    min_n=min_n,
                    blocked_reason="insufficient_non_null_for_quantiles",
                    missing_columns=[],
                )
                rows.extend(r)
                blocked_segments.extend(b)
            else:
                r, b = _segment_rows(df, run_id, market, model_name, "total_regime", values, min_n=min_n)
                rows.extend(r)
                blocked_segments.extend(b)
    else:
        r, b = _segment_rows(
            df,
            run_id,
            market,
            model_name,
            "total_regime",
            None,
            min_n=min_n,
            blocked_reason="not_applicable_for_market",
            missing_columns=[],
        )
        rows.extend(r)
        blocked_segments.extend(b)

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values(["segment_family", "status", "roi"], ascending=[True, True, False], na_position="last")
    return report.reset_index(drop=True), blocked_segments


def _worst_misses(
    df: pd.DataFrame,
    run_id: str,
    market: str,
    model_name: str,
) -> pd.DataFrame:
    actual_col = {"spread": "actual_margin", "total": "actual_total", "ml": "home_won"}[market]
    line_col = {"spread": "spread_line", "total": "total_line", "ml": "implied_home_prob"}[market]
    local = df.copy()
    local["pred_value"] = pd.to_numeric(local.get("model_pred"), errors="coerce")
    local["actual_value"] = pd.to_numeric(local.get(actual_col), errors="coerce")
    local["market_line"] = pd.to_numeric(local.get(line_col), errors="coerce")
    local["edge"] = pd.to_numeric(local.get("edge_abs"), errors="coerce")
    local["abs_error"] = pd.to_numeric(local.get("abs_error"), errors="coerce")
    local = local[local["abs_error"].notna()].copy()
    local = local.sort_values("abs_error", ascending=False).head(20).reset_index(drop=True)
    local["rank"] = np.arange(1, len(local) + 1)

    missing_snapshot = [col for col in CORE_SNAPSHOT_FEATURES if col not in local.columns]
    for col in CORE_SNAPSHOT_FEATURES:
        if col not in local.columns:
            local[col] = np.nan

    out_cols = [
        "rank",
        "fold_id",
        "season_id",
        "game_id",
        "event_id",
        "game_date",
        "game_datetime_utc",
        "home_team_id",
        "away_team_id",
        "home_team",
        "away_team",
        "neutral_site",
        "pred_value",
        "actual_value",
        "market_line",
        "edge",
        "abs_error",
    ]
    for col in out_cols:
        if col not in local.columns:
            local[col] = pd.NA

    local["run_id"] = run_id
    local["market"] = market
    local["model_name"] = model_name
    for col in CORE_SNAPSHOT_FEATURES:
        local[f"snapshot_{col}"] = local[col]
    local["snapshot_missing_columns"] = ",".join(missing_snapshot)

    final_cols = [
        "run_id",
        "market",
        "model_name",
        *out_cols,
        *[f"snapshot_{col}" for col in CORE_SNAPSHOT_FEATURES],
        "snapshot_missing_columns",
    ]
    return local[final_cols].copy()


def _build_exec_summary(
    run_id: str,
    market: str,
    model_name: str,
    bucket_report: pd.DataFrame,
    segment_report: pd.DataFrame,
    min_n: int,
) -> str:
    lines: list[str] = []
    lines.append("# Edge Analyzer Executive Summary")
    lines.append("")
    lines.append(f"- Run ID: `{run_id}`")
    lines.append(f"- Market: `{market}`")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Generated (UTC): `{_now_utc()}`")
    lines.append("")

    ok_segments = segment_report[
        (segment_report.get("status", "") == "OK")
        & pd.to_numeric(segment_report.get("roi"), errors="coerce").notna()
        & (pd.to_numeric(segment_report.get("n"), errors="coerce") >= int(min_n))
    ].copy()
    if ok_segments.empty:
        lines.append("## ROI Drivers")
        lines.append("- No segments met minimum sample size for stable ROI interpretation.")
    else:
        top = ok_segments.sort_values("roi", ascending=False).head(3)
        bottom = ok_segments.sort_values("roi", ascending=True).head(3)
        lines.append("## Where ROI Came From (Top 3)")
        for _, row in top.iterrows():
            lines.append(
                f"- `{row['segment_family']}:{row['segment_value']}` -> ROI `{float(row['roi']):.4f}`, "
                f"hit `{float(row['hit_rate']):.4f}`, n `{int(row['n'])}`"
            )
        lines.append("")
        lines.append("## Where Model Bleeds (Bottom 3)")
        for _, row in bottom.iterrows():
            lines.append(
                f"- `{row['segment_family']}:{row['segment_value']}` -> ROI `{float(row['roi']):.4f}`, "
                f"hit `{float(row['hit_rate']):.4f}`, n `{int(row['n'])}`"
            )

    lines.append("")
    lines.append("## Sample Size Warnings")
    bucket_small = bucket_report[(bucket_report.get("status", "") == "OK") & bucket_report.get("small_sample_flag", False)]
    segment_small = segment_report[(segment_report.get("status", "") == "OK") & segment_report.get("small_sample_flag", False)]
    if bucket_small.empty and segment_small.empty:
        lines.append("- None.")
    else:
        if not bucket_small.empty:
            labels = ", ".join(sorted(bucket_small["bucket_label"].astype(str).unique().tolist()))
            lines.append(f"- Edge buckets below `min_n={min_n}`: {labels}")
        if not segment_small.empty:
            labels = ", ".join(
                sorted(
                    (
                        segment_small["segment_family"].astype(str)
                        + ":"
                        + segment_small["segment_value"].astype(str)
                    ).unique().tolist()
                )
            )
            lines.append(f"- Segments below `min_n={min_n}`: {labels}")

    blocked = segment_report[segment_report.get("status", "") == "BLOCKED"].copy()
    lines.append("")
    lines.append("## Blocked Segments")
    if blocked.empty:
        lines.append("- None.")
    else:
        for _, row in blocked.iterrows():
            lines.append(
                f"- `{row['segment_family']}` blocked ({row['blocked_reason']}); "
                f"missing columns: `{row['missing_columns']}`"
            )

    lines.append("")
    lines.append("## Recommended Gates (Rules)")
    stable_buckets = bucket_report[
        (bucket_report.get("status", "") == "OK")
        & pd.to_numeric(bucket_report.get("roi"), errors="coerce").notna()
        & (pd.to_numeric(bucket_report.get("n"), errors="coerce") >= int(min_n))
    ].copy()
    if not stable_buckets.empty:
        best_bucket = stable_buckets.sort_values("roi", ascending=False).iloc[0]
        lines.append(
            f"- Gate 1: only fire when absolute edge bucket is `{best_bucket['bucket_label']}` or stronger "
            f"(historical ROI `{float(best_bucket['roi']):.4f}`, n `{int(best_bucket['n'])}`)."
        )
    else:
        lines.append("- Gate 1: require minimum edge bucket sample before enabling edge-size gating.")

    if not ok_segments.empty:
        worst = ok_segments.sort_values("roi", ascending=True).iloc[0]
        lines.append(
            f"- Gate 2: downweight/skip segment `{worst['segment_family']}:{worst['segment_value']}` "
            f"(historical ROI `{float(worst['roi']):.4f}`)."
        )
    else:
        lines.append("- Gate 2: defer segment gating until at least one segment reaches stable sample size.")

    lines.append(f"- Gate 3: enforce `n >= {int(min_n)}` before trusting any bucket/segment edge rule.")
    lines.append("")
    return "\n".join(lines) + "\n"


def _edge_nan_report(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    cols = [
        "model_pred",
        "actual_margin",
        "actual_total",
        "home_won",
        "spread_line",
        "spread_open",
        "spread_close",
        "total_line",
        "total_open",
        "total_close",
        "home_ml",
        "away_ml",
        "edge_abs",
        "abs_error",
        "outcome",
        "clv",
    ]
    present = [c for c in cols if c in df.columns]
    if not present:
        return {}
    return dataframe_nan_report(df[present].copy())


def run_edge_analysis(
    run_dir: Path,
    config: ModelLabConfig,
    *,
    market: str,
    model_name: str,
    min_n: int = 50,
    limit: int | None = None,
) -> EdgeAnalyzerResult:
    run_id = run_dir.name
    blocked: list[str] = []
    blocked_segments: list[dict[str, Any]] = []

    frame = _load_or_build_market_frame(run_dir, config, market, blocked)
    if frame.empty:
        output_dir = (run_dir / "edge_analyzer" / market / _safe_model_name(model_name)).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        edge_path = output_dir / "edge_bucket_report.csv"
        segment_path = output_dir / "segment_report.csv"
        worst_path = output_dir / "worst_misses.csv"
        summary_path = output_dir / "EDGE_EXEC_SUMMARY.md"
        pd.DataFrame().to_csv(edge_path, index=False)
        pd.DataFrame().to_csv(segment_path, index=False)
        pd.DataFrame().to_csv(worst_path, index=False)
        summary_path.write_text("# Edge Analyzer Executive Summary\n\n- BLOCKED: market frame is missing/empty.\n", encoding="utf-8")
        manifest_path = (run_dir / "edge_analyzer" / "run_manifest.json").resolve()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "generated_at_utc": _now_utc(),
                    "market": market,
                    "model_name": model_name,
                    "min_n": int(max(1, min_n)),
                    "limit": int(limit) if limit is not None else None,
                    "rows_frame": 0,
                    "rows_predictions": 0,
                    "rows_joined": 0,
                    "rows_analyzed": 0,
                    "fold_count": 0,
                    "blocked_reasons": sorted(set(blocked)),
                    "blocked_segments": [],
                    "nan_report": {},
                    "artifacts": {
                        "edge_bucket_report": str(edge_path),
                        "segment_report": str(segment_path),
                        "worst_misses": str(worst_path),
                        "edge_exec_summary": str(summary_path),
                    },
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return EdgeAnalyzerResult(
            run_id=run_id,
            market=market,
            model_name=model_name,
            rows_analyzed=0,
            edge_bucket_report_path=edge_path,
            segment_report_path=segment_path,
            worst_misses_path=worst_path,
            exec_summary_path=summary_path,
            run_manifest_path=manifest_path,
            blocked_reasons=sorted(set(blocked)),
            blocked_segments=[],
        )

    frame = frame.sort_values(["season_id", "game_datetime_utc", "game_id"], na_position="last").reset_index(drop=True)
    if limit is not None and int(limit) > 0:
        frame = frame.tail(int(limit)).reset_index(drop=True)

    folds, split_blocked = build_rolling_folds(frame, config)
    blocked.extend(split_blocked)
    if not folds:
        blocked.append(f"edge_analyzer_no_folds:{market}")

    preds = _load_model_predictions(config, model_name)
    if preds.empty:
        blocked.append(f"edge_analyzer_missing_predictions:{model_name}")

    merged = _attach_predictions(frame, preds)
    if merged.empty:
        blocked.append(f"edge_analyzer_no_prediction_join_rows:{market}:{model_name}")

    test_rows = _attach_fold_test_rows(merged, frame, folds) if folds else pd.DataFrame()
    if test_rows.empty:
        blocked.append(f"edge_analyzer_no_test_rows:{market}:{model_name}")

    graded = _row_grade(test_rows, market, config.default_odds) if not test_rows.empty else pd.DataFrame()
    bucket_report, bucket_blocked = _edge_bucket_report(
        graded if not graded.empty else pd.DataFrame(),
        run_id,
        market,
        model_name,
        int(max(1, min_n)),
    )
    blocked.extend(bucket_blocked)

    segment_report, segment_blocked = _segment_report(
        graded if not graded.empty else pd.DataFrame(),
        run_id,
        market,
        model_name,
        int(max(1, min_n)),
    )
    blocked_segments.extend(segment_blocked)
    for item in segment_blocked:
        missing = ",".join(item.get("missing_columns", []))
        blocked.append(f"edge_segment_blocked:{item.get('segment_family')}:{item.get('reason')}:{missing}")

    worst_misses = _worst_misses(graded if not graded.empty else pd.DataFrame(), run_id, market, model_name)
    summary_text = _build_exec_summary(
        run_id,
        market,
        model_name,
        bucket_report,
        segment_report,
        int(max(1, min_n)),
    )

    output_dir = (run_dir / "edge_analyzer" / market / _safe_model_name(model_name)).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    edge_path = output_dir / "edge_bucket_report.csv"
    segment_path = output_dir / "segment_report.csv"
    worst_path = output_dir / "worst_misses.csv"
    summary_path = output_dir / "EDGE_EXEC_SUMMARY.md"
    bucket_report.to_csv(edge_path, index=False)
    segment_report.to_csv(segment_path, index=False)
    worst_misses.to_csv(worst_path, index=False)
    summary_path.write_text(summary_text, encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "generated_at_utc": _now_utc(),
        "market": market,
        "model_name": model_name,
        "available_models": DEFAULT_MODEL_NAMES + ["ensemble"],
        "min_n": int(max(1, min_n)),
        "limit": int(limit) if limit is not None else None,
        "rows_frame": int(len(frame)),
        "rows_predictions": int(len(preds)),
        "rows_joined": int(len(merged)),
        "rows_analyzed": int(len(graded)),
        "fold_count": int(len(folds)),
        "blocked_reasons": sorted(set(blocked)),
        "blocked_segments": blocked_segments,
        "nan_report": _edge_nan_report(graded if not graded.empty else pd.DataFrame()),
        "artifacts": {
            "edge_bucket_report": str(edge_path),
            "segment_report": str(segment_path),
            "worst_misses": str(worst_path),
            "edge_exec_summary": str(summary_path),
        },
    }
    manifest_path = (run_dir / "edge_analyzer" / "run_manifest.json").resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return EdgeAnalyzerResult(
        run_id=run_id,
        market=market,
        model_name=model_name,
        rows_analyzed=int(len(graded)),
        edge_bucket_report_path=edge_path,
        segment_report_path=segment_path,
        worst_misses_path=worst_path,
        exec_summary_path=summary_path,
        run_manifest_path=manifest_path,
        blocked_reasons=sorted(set(blocked)),
        blocked_segments=blocked_segments,
    )
