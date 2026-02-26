from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _roi_from_win_rate(win_rate: float, odds: int = -110) -> float:
    if np.isnan(win_rate):
        return np.nan
    return (win_rate * (100 / abs(odds))) - (1 - win_rate)


def build_standard_evaluation(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    graded = pd.read_csv(data_dir / "results_log_graded.csv", low_memory=False)

    spread_mae = pd.to_numeric(graded.get("primary_margin_error"), errors="coerce").abs().mean()
    total_mae = pd.to_numeric(graded.get("total_error"), errors="coerce").abs().mean()

    if {"primary_confidence", "primary_wins_game"}.issubset(graded.columns):
        p = pd.to_numeric(graded["primary_confidence"], errors="coerce").clip(0, 1)
        y = pd.to_numeric(graded["primary_wins_game"], errors="coerce")
        mask = p.notna() & y.notna()
        brier = float(np.mean((p[mask] - y[mask]) ** 2)) if mask.any() else np.nan
    else:
        brier = np.nan

    ats = pd.to_numeric(graded.get("primary_ats_correct"), errors="coerce")
    ou = pd.to_numeric(graded.get("primary_ou_correct"), errors="coerce")
    ats_hit = float(ats.mean()) if ats.notna().any() else np.nan
    ou_hit = float(ou.mean()) if ou.notna().any() else np.nan

    roi_ats = _roi_from_win_rate(ats_hit)
    roi_ou = _roi_from_win_rate(ou_hit)

    clv_spread = pd.to_numeric(graded.get("clv_vs_close"), errors="coerce").median()
    clv_ml = pd.to_numeric(graded.get("clv_ml_implied_prob"), errors="coerce").median()

    edge = pd.to_numeric(graded.get("edge_abs"), errors="coerce")
    edge_bucket = pd.cut(edge, bins=[-np.inf, 2, 4, 6, np.inf], labels=["0-2", "2-4", "4-6", "6+"])
    edge_perf = (
        graded.assign(edge_bucket=edge_bucket)
        .groupby("edge_bucket", observed=False)["primary_ats_correct"]
        .mean()
        .reset_index()
        .rename(columns={"primary_ats_correct": "ats_hit_rate"})
    )

    out_rows = [
        ("spread_mae", spread_mae),
        ("total_mae", total_mae),
        ("brier_score", brier),
        ("ats_hit_rate", ats_hit),
        ("ou_hit_rate", ou_hit),
        ("roi_ats", roi_ats),
        ("roi_ou", roi_ou),
        ("median_clv_spread", clv_spread),
        ("median_clv_ml", clv_ml),
    ]
    metrics_df = pd.DataFrame(out_rows, columns=["metric", "value"])

    summary = {k: (None if pd.isna(v) else float(v)) for k, v in out_rows}
    summary["edge_bucket_performance"] = edge_perf.fillna(np.nan).to_dict(orient="records")
    return metrics_df, summary


def write_evaluation_outputs(data_dir: Path) -> dict:
    metrics_df, summary = build_standard_evaluation(data_dir)
    (data_dir / "evaluation.csv").write_text(metrics_df.to_csv(index=False), encoding="utf-8")
    (data_dir / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
