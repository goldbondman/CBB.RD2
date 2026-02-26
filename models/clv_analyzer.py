"""Generate game-level and submodel CLV reports."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

DATA_DIR = Path("data")
ACCURACY_PATH = DATA_DIR / "model_accuracy_report.csv"
PRED_CONTEXT_PATH = DATA_DIR / "predictions_with_context.csv"
OUT_CLV_REPORT = DATA_DIR / "clv_report.csv"
OUT_CLV_BY_SUBMODEL = DATA_DIR / "clv_by_submodel.csv"
PRED_FALLBACK_PATHS = [
    DATA_DIR / "predictions_combined_latest.csv",
    DATA_DIR / "predictions_latest.csv",
    DATA_DIR / "predictions_primary.csv",
]

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

SUBMODEL_SPREAD_COLS = [
    "ens_ens_spread",
    "ens_fourfactors_spread",
    "ens_adjefficiency_spread",
    "ens_pythagorean_spread",
    "ens_momentum_spread",
    "ens_situational_spread",
    "ens_cagerankings_spread",
    "ens_regressedeff_spread",
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
    for col in ("game_id", "event_id"):
        if col in out.columns:
            out[col] = out[col].astype(str).str.strip()
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
        return pd.read_csv(path, low_memory=False)
    except EmptyDataError:
        LOG.warning("%s is empty (0 bytes or no rows): %s", label, path)
        return pd.DataFrame()


def _first_non_null(df: pd.DataFrame, candidates: list[str], fallback_dtype: str = "float") -> pd.Series:
    series = pd.Series([pd.NA] * len(df), index=df.index)
    for col in candidates:
        if col in df.columns:
            series = series.fillna(df[col])
    if fallback_dtype == "float":
        return pd.to_numeric(series, errors="coerce")
    return series


def build_clv_reports(
    accuracy_path: Path = ACCURACY_PATH,
    pred_context_path: Path = PRED_CONTEXT_PATH,
    out_report: Path = OUT_CLV_REPORT,
    out_submodel: Path = OUT_CLV_BY_SUBMODEL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not accuracy_path.exists() or not pred_context_path.exists():
        LOG.warning(
            "Missing CLV input(s): accuracy=%s exists=%s | predictions=%s exists=%s. Writing empty outputs.",
            accuracy_path,
            accuracy_path.exists(),
            pred_context_path,
            pred_context_path.exists(),
        )
    acc = _safe_read_csv(accuracy_path, "model_accuracy_report.csv")
    pred = _safe_read_csv(pred_context_path, "predictions_with_context.csv")
    if pred.empty:
        for fallback in PRED_FALLBACK_PATHS:
            pred = _safe_read_csv(fallback, f"fallback predictions ({fallback.name})")
            if not pred.empty:
                LOG.warning(
                    "Using fallback predictions source for CLV report: %s (%d rows)",
                    fallback,
                    len(pred),
                )
                break

    if pred.empty:
        LOG.warning("No prediction rows available for CLV report. Writing empty outputs.")
        _write_empty_outputs()
        return pd.DataFrame(), pd.DataFrame()

    pred = _prepare_join_key(pred, "predictions_with_context.csv")

    if acc.empty:
        LOG.warning("model_accuracy_report.csv has no rows; CLV report will omit outcome metrics.")
        merged = pred.copy()
    else:
        acc = _prepare_join_key(acc, "model_accuracy_report.csv")
        merged = pred.merge(
            acc,
            on="join_game_id",
            how="left",
            suffixes=("", "_acc"),
        )

    merged["game_id"] = _first_non_null(merged, ["game_id", "event_id", "game_id_acc", "event_id_acc"], fallback_dtype="str")
    merged["pred_spread"] = _first_non_null(merged, ["pred_spread", "predicted_spread", "ens_ens_spread"])
    merged["home_spread_open"] = _first_non_null(merged, ["home_spread_open", "spread_open", "opening_spread", "spread_line"])
    merged["home_spread_current"] = _first_non_null(
        merged,
        ["home_spread_current", "market_spread", "spread", "spread_line", "closing_spread"],
    )
    merged["clv_vs_open"] = (merged["home_spread_open"] - merged["pred_spread"]).round(3)
    merged["clv_vs_close"] = (merged["home_spread_current"] - merged["pred_spread"]).round(3)

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
    game_report = game_report[
        game_report["clv_vs_open"].notna() | game_report["clv_vs_close"].notna()
    ].sort_values(["game_datetime_utc", "game_id"], na_position="last")

    submodel_rows = []
    for col in [c for c in SUBMODEL_SPREAD_COLS if c in merged.columns]:
        pred_col = pd.to_numeric(merged[col], errors="coerce")
        clv_open = merged["home_spread_open"] - pred_col
        clv_close = merged["home_spread_current"] - pred_col
        abs_error = (merged["actual_margin"] - pred_col).abs()
        outcome_edge = merged["actual_margin"] - merged["home_spread_current"]

        valid = clv_open.notna() & clv_close.notna()
        if valid.any():
            corr_mask = valid & outcome_edge.notna()
            corr_val = np.nan
            if int(corr_mask.sum()) >= 2:
                corr_val = clv_close[corr_mask].corr(outcome_edge[corr_mask])

            submodel_rows.append(
                {
                    "model_name": col,
                    "n_games_with_clv": int(valid.sum()),
                    "mean_clv_vs_open": round(float(clv_open[valid].mean()), 4),
                    "mean_clv_vs_close": round(float(clv_close[valid].mean()), 4),
                    "pct_positive_clv": round(float((clv_close[valid] > 0).mean()), 4),
                    "mean_abs_error": round(float(abs_error[valid].mean()), 4),
                    "correlation_clv_to_outcome": (
                        round(float(corr_val), 4) if pd.notna(corr_val) else pd.NA
                    ),
                }
            )

    submodel_df = pd.DataFrame(
        submodel_rows,
        columns=[
            "model_name",
            "n_games_with_clv",
            "mean_clv_vs_open",
            "mean_clv_vs_close",
            "pct_positive_clv",
            "mean_abs_error",
            "correlation_clv_to_outcome",
        ],
    )

    out_report.parent.mkdir(parents=True, exist_ok=True)
    game_report.to_csv(out_report, index=False)
    submodel_df.to_csv(out_submodel, index=False)

    LOG.info("Wrote game CLV report: %d rows -> %s", len(game_report), out_report)
    LOG.info("Wrote submodel CLV report: %d rows -> %s", len(submodel_df), out_submodel)
    return game_report, submodel_df


if __name__ == "__main__":
    build_clv_reports()
