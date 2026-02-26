#!/usr/bin/env python3
"""
clv_analyzer.py — CLV (Closing Line Value) Analysis & Reporting

Computes CLV for the ensemble and individual models by comparing
predicted spreads against market opening/closing lines. Emits
performance reports used for model calibration and accuracy tracking.

Convention: line - pred (positive = model found value)
  - If line is -5 and model is -7: -5 - (-7) = +2.0 (good)
  - If line is -5 and model is -3: -5 - (-3) = -2.0 (bad)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from pipeline_csv_utils import compute_clv_pts

LOG = logging.getLogger(__name__)

# Defaults
DATA_DIR = Path("data")
ACCURACY_PATH = DATA_DIR / "model_accuracy_report.csv"
PRED_CONTEXT_PATH = DATA_DIR / "predictions_with_context.csv"
PRED_FALLBACK_PATHS = [
    DATA_DIR / "predictions_mc_latest.csv",
    DATA_DIR / "predictions_combined_latest.csv",
]

OUT_CLV_REPORT = DATA_DIR / "clv_report.csv"
OUT_CLV_BY_SUBMODEL = DATA_DIR / "clv_by_submodel.csv"

SUBMODEL_SPREAD_COLS = [
    "ens_ens_spread",
    "ens_fourfactors_spread",
    "ens_adjefficiency_spread",
    "ens_pythagorean_spread",
    "ens_momentum_spread",
    "ens_situational_spread",
    "ens_cagerankings_spread",
    "ens_luckregression_spread",
    "ens_variance_spread",
    "ens_homeawayform_spread",
]


def _write_empty_outputs() -> None:
    report_cols = [
        "game_id",
        "game_datetime_utc",
        "home_team",
        "away_team",
        "pred_spread",
        "home_spread_open",
        "home_spread_current",
        "clv_vs_open",
        "clv_vs_close",
        "actual_margin",
        "covered",
        "spread_error",
        "model_confidence",
        "game_tier",
    ]
    submodel_cols = [
        "model_name",
        "n_games_with_clv",
        "mean_clv_vs_open",
        "mean_clv_vs_close",
        "pct_positive_clv",
        "mean_abs_error",
        "correlation_clv_to_outcome",
    ]
    OUT_CLV_REPORT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=report_cols).to_csv(OUT_CLV_REPORT, index=False)
    pd.DataFrame(columns=submodel_cols).to_csv(OUT_CLV_BY_SUBMODEL, index=False)


def _prepare_join_key(df: pd.DataFrame, name: str) -> pd.DataFrame:
    out = df.copy()
    from pipeline_csv_utils import normalize_game_id
    for col in ("game_id", "event_id"):
        if col in out.columns:
            out[col] = out[col].apply(normalize_game_id)

    if "game_id" in out.columns and out["game_id"].str.len().gt(0).any():
        out["join_game_id"] = out["game_id"]
    elif "event_id" in out.columns and out["event_id"].str.len().gt(0).any():
        out["join_game_id"] = out["event_id"]
    else:
        raise ValueError(f"{name} missing both game_id and event_id")
    return out


def _safe_read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        LOG.warning("%s not found: %s", label, path)
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
        if df.empty:
            LOG.warning("%s is empty (no rows): %s", label, path)
        return df
    except EmptyDataError:
        LOG.warning("%s is empty (0 bytes): %s", label, path)
        return pd.DataFrame()


def _first_non_null(df: pd.DataFrame, candidates: list[str], fallback_dtype: str = "float") -> pd.Series:
    series = pd.Series([pd.NA] * len(df), index=df.index, dtype="object")
    for col in candidates:
        if col in df.columns:
            series = series.where(series.notna(), df[col])
    if fallback_dtype == "float":
        return pd.to_numeric(series, errors="coerce")
    return series


def build_clv_reports(
    accuracy_path: Path = ACCURACY_PATH,
    pred_context_path: Path = PRED_CONTEXT_PATH,
    out_report: Path = OUT_CLV_REPORT,
    out_submodel: Path = OUT_CLV_BY_SUBMODEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = _safe_read_csv(accuracy_path, "model_accuracy_report")
    pred = _safe_read_csv(pred_context_path, "predictions_with_context")

    if pred.empty:
        for fallback in PRED_FALLBACK_PATHS:
            pred = _safe_read_csv(fallback, f"fallback predictions ({fallback.name})")
            if not pred.empty:
                LOG.warning("Using fallback predictions source for CLV report: %s", fallback)
                break

    if pred.empty:
        LOG.warning("No prediction rows available for CLV report. Writing empty outputs.")
        _write_empty_outputs()
        return pd.DataFrame(), pd.DataFrame()

    pred = _prepare_join_key(pred, "predictions_source")

    if acc.empty:
        LOG.warning("model_accuracy_report has no rows; continuing with predictions-only metrics.")
        merged = pred.copy()
    else:
        acc = _prepare_join_key(acc, "model_accuracy_report")
        merged = pred.merge(
            acc,
            on="join_game_id",
            how="left",
            suffixes=("", "_acc"),
        )

    # Coalesce key columns
    merged["game_id"] = _first_non_null(merged, ["game_id", "event_id", "game_id_acc", "event_id_acc"], fallback_dtype="str")
    merged["pred_spread"] = _first_non_null(merged, ["pred_spread", "predicted_spread", "ens_ens_spread"])
    merged["home_spread_open"] = _first_non_null(merged, ["home_spread_open", "spread_open", "opening_spread"])
    merged["home_spread_current"] = _first_non_null(
        merged,
        ["home_spread_current", "market_spread", "spread", "spread_line", "closing_spread"],
    )

    # Apply CLV convention via helper
    merged["clv_vs_open"] = merged.apply(
        lambda r: compute_clv_pts(r.get("home_spread_open"), r.get("pred_spread")), axis=1
    )
    merged["clv_vs_close"] = merged.apply(
        lambda r: compute_clv_pts(r.get("home_spread_current"), r.get("pred_spread")), axis=1
    )

    merged["actual_margin"] = _first_non_null(merged, ["actual_margin", "actual_margin_acc", "home_margin", "margin"])

    if "spread_error" not in merged.columns:
        merged["spread_error"] = (merged["actual_margin"] - merged["pred_spread"]).abs().round(3)
    else:
        merged["spread_error"] = pd.to_numeric(merged["spread_error"], errors="coerce")

    if "covered" not in merged.columns:
        merged["covered"] = pd.NA

    game_cols = [
        "game_id",
        "game_datetime_utc",
        "home_team",
        "away_team",
        "pred_spread",
        "home_spread_open",
        "home_spread_current",
        "clv_vs_open",
        "clv_vs_close",
        "actual_margin",
        "covered",
        "spread_error",
        "model_confidence",
        "game_tier",
    ]
    for col in game_cols:
        if col not in merged.columns:
            merged[col] = pd.NA

    game_report = merged[game_cols].copy()
    # Filter to games with at least some market data
    game_report = game_report[
        game_report["clv_vs_open"].notna() | game_report["clv_vs_close"].notna()
    ].sort_values(["game_datetime_utc", "game_id"], na_position="last")

    LOG.info("CLV game report: %d rows", len(game_report))

    # Submodel metrics
    submodel_rows = []
    for col in [c for c in SUBMODEL_SPREAD_COLS if c in merged.columns]:
        pred_col = pd.to_numeric(merged[col], errors="coerce")
        # Reuse convention: line - pred
        clv_open = merged["home_spread_open"] - pred_col
        clv_close = merged["home_spread_current"] - pred_col
        abs_error = (merged["actual_margin"] - pred_col).abs()
        outcome_edge = merged["actual_margin"] - merged["home_spread_current"]

        valid_close = clv_close.notna()
        valid_open = clv_open.notna()
        if valid_close.any():
            corr_mask = valid_close & outcome_edge.notna()
            corr_val = np.nan
            if int(corr_mask.sum()) >= 2:
                corr_val = clv_close[corr_mask].corr(outcome_edge[corr_mask])

            abs_error_mask = valid_close & abs_error.notna()
            mean_abs_error = round(float(abs_error[abs_error_mask].mean()), 4) if abs_error_mask.any() else pd.NA
            mean_clv_vs_open = round(float(clv_open[valid_open].mean()), 4) if valid_open.any() else pd.NA

            submodel_rows.append(
                {
                    "model_name": col,
                    "n_games_with_clv": int(valid_close.sum()),
                    "mean_clv_vs_open": mean_clv_vs_open,
                    "mean_clv_vs_close": round(float(clv_close[valid_close].mean()), 4),
                    "pct_positive_clv": round(float((clv_close[valid_close] > 0).mean()), 4),
                    "mean_abs_error": mean_abs_error,
                    "correlation_clv_to_outcome": (
                        round(float(corr_val), 4) if pd.notna(corr_val) else pd.NA
                    ),
                }
            )

    submodel_df = pd.DataFrame(submodel_rows)
    if not submodel_df.empty:
        submodel_df = submodel_df.sort_values("mean_clv_vs_close", ascending=False)

    out_report.parent.mkdir(parents=True, exist_ok=True)
    game_report.to_csv(out_report, index=False)
    submodel_df.to_csv(out_submodel, index=False)

    LOG.info("Wrote submodel CLV report: %d models -> %s", len(submodel_df), out_submodel)
    return game_report, submodel_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_clv_reports()
