#!/usr/bin/env python3
"""Feature predictiveness and redundancy audit for graded CBB outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

DATA_DIR = Path("data")
INPUT_PATH = DATA_DIR / "results_log_graded.csv"
OUTPUT_PATH = DATA_DIR / "feature_audit.csv"


def audit_feature_predictiveness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Point-biserial correlation between numeric features and ATS outcome.
    """
    from scipy import stats

    if "home_covered_pred" not in df.columns and "ats_correct" in df.columns:
        df = df.copy()
        df["home_covered_pred"] = pd.to_numeric(df["ats_correct"], errors="coerce")

    outcome = pd.to_numeric(df.get("home_covered_pred"), errors="coerce").dropna()
    if outcome.empty:
        return pd.DataFrame(columns=["feature", "correlation", "p_value", "predictive", "sample_n"])

    feature_cols = [
        c
        for c in df.columns
        if c.endswith(("_l10", "_l5", "_diff", "_score")) and c in df.columns
    ]

    results = []
    for col in feature_cols:
        aligned = pd.to_numeric(df[col], errors="coerce").reindex(outcome.index).dropna()
        outcome_aligned = outcome.reindex(aligned.index)
        if len(aligned) < 20:
            continue
        r, p = stats.pointbiserialr(outcome_aligned, aligned)
        results.append(
            {
                "feature": col,
                "correlation": round(float(r), 4),
                "p_value": round(float(p), 4),
                "predictive": bool(abs(r) > 0.10 and p < 0.10),
                "sample_n": int(len(aligned)),
            }
        )

    result_df = pd.DataFrame(results)
    if result_df.empty:
        return result_df

    result_df = result_df.sort_values("correlation", key=lambda s: s.abs(), ascending=False)

    strong = result_df[result_df["correlation"].abs() > 0.20]["feature"].tolist()
    noise = result_df[result_df["correlation"].abs() < 0.03]["feature"].tolist()
    if strong:
        log.info("Strong signals (|r| > 0.20): %s", ", ".join(strong))
    if noise:
        log.info("Noise candidates (|r| < 0.03): %s", ", ".join(noise))

    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    corr_matrix = numeric.corr().abs()
    redundant = set()
    for i, col_a in enumerate(corr_matrix.columns):
        for col_b in corr_matrix.columns[i + 1 :]:
            val = corr_matrix.loc[col_a, col_b]
            if pd.notna(val) and val > 0.90:
                redundant.add((col_a, col_b, round(float(val), 4)))

    if redundant:
        pairs = ", ".join([f"{a}~{b} ({v})" for a, b, v in sorted(redundant)])
        log.warning("Redundant feature pairs (>0.90 correlation): %s", pairs)

    return result_df


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Missing graded file for audit: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, low_memory=False)
    if len(df) < 100:
        log.info("Skipping feature audit: need at least 100 graded games, found %s", len(df))
        return

    result_df = audit_feature_predictiveness(df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(OUTPUT_PATH, index=False)
    log.info("Feature audit rows written: %s", len(result_df))


if __name__ == "__main__":
    main()
