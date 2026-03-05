#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def _null_rate(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return 1.0
    return float(df[col].isna().mean())


def _check_columns(df: pd.DataFrame, cols: List[str], label: str, errors: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        errors.append(f"{label}: missing columns: {missing}")


def _enforce_min_non_null(df: pd.DataFrame, cols: List[str], min_non_null: float, label: str, errors: List[str]) -> Dict[str, float]:
    rates: Dict[str, float] = {}
    for col in cols:
        rate = 1.0 - _null_rate(df, col)
        rates[col] = rate
        if rate < min_non_null:
            errors.append(f"{label}: non-null rate {rate:.3f} < {min_non_null:.3f} for column '{col}'")
    return rates


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate backtest context outputs (columns + null rates).")
    parser.add_argument("--training", default="data/backtest_training_data.csv")
    parser.add_argument("--results", default="data/backtest_results_latest.csv")
    parser.add_argument("--min-market-non-null", type=float, default=0.05)
    parser.add_argument("--min-engineered-non-null", type=float, default=0.70)
    parser.add_argument("--min-pred-non-null", type=float, default=0.10)
    parser.add_argument("--report", default="debug/backtest_context_validation.json")
    args = parser.parse_args()

    errors: List[str] = []
    report: Dict[str, object] = {
        "files": {"training": args.training, "results": args.results},
        "thresholds": {
            "min_market_non_null": args.min_market_non_null,
            "min_engineered_non_null": args.min_engineered_non_null,
            "min_pred_non_null": args.min_pred_non_null,
        },
    }

    training_path = Path(args.training)
    results_path = Path(args.results)
    if not training_path.exists() or training_path.stat().st_size == 0:
        raise SystemExit(f"[ERROR] Missing/empty training file: {training_path}")
    if not results_path.exists() or results_path.stat().st_size == 0:
        raise SystemExit(f"[ERROR] Missing/empty results file: {results_path}")

    tdf = pd.read_csv(training_path, low_memory=False)
    rdf = pd.read_csv(results_path, low_memory=False)
    report["rows"] = {"training": int(len(tdf)), "results": int(len(rdf))}

    training_required = [
        "game_id", "event_id",
        "home_net_rtg_l5", "away_net_rtg_l5",
        "espn_spread", "espn_total", "opening_spread", "closing_spread", "spread_line", "total_line",
        "pred_spread", "pred_total", "clv_delta",
    ]
    results_required = [
        "game_id", "market_spread", "market_total",
        "opening_spread", "closing_spread", "spread_line", "total_line",
        "pred_spread", "pred_total", "clv_delta",
    ]

    _check_columns(tdf, training_required, "training", errors)
    _check_columns(rdf, results_required, "results", errors)

    market_cols = ["espn_spread", "espn_total", "closing_spread", "spread_line", "total_line"]
    optional_market_cols = ["opening_spread"]
    engineered_cols = [
        "home_net_rtg_l5", "home_net_rtg_l10", "home_adj_ortg", "home_adj_drtg", "home_adj_net_rtg",
        "home_efg_pct_l5", "home_efg_pct_l10", "home_tov_pct_l5", "home_orb_pct_l10", "home_ftr_l5",
        "home_pace_l5", "home_ortg_l5", "home_drtg_l5", "home_margin_l5", "home_margin_l10",
        "away_net_rtg_l5", "away_net_rtg_l10", "away_adj_ortg", "away_adj_drtg", "away_adj_net_rtg",
        "away_efg_pct_l5", "away_efg_pct_l10", "away_tov_pct_l5", "away_orb_pct_l10", "away_ftr_l5",
        "away_pace_l5", "away_ortg_l5", "away_drtg_l5", "away_margin_l5", "away_margin_l10",
    ]
    pred_cols = ["pred_spread", "pred_total"]

    report["non_null_rates"] = {
        "training_market": _enforce_min_non_null(tdf, [c for c in market_cols if c in tdf.columns], args.min_market_non_null, "training_market", errors),
        "training_market_optional": {c: (1.0 - _null_rate(tdf, c)) for c in optional_market_cols if c in tdf.columns},
        "training_engineered": _enforce_min_non_null(tdf, [c for c in engineered_cols if c in tdf.columns], args.min_engineered_non_null, "training_engineered", errors),
        "training_pred": _enforce_min_non_null(tdf, [c for c in pred_cols if c in tdf.columns], args.min_pred_non_null, "training_pred", errors),
        "results_market": _enforce_min_non_null(rdf, [c for c in ["market_spread", "market_total", "closing_spread", "spread_line", "total_line"] if c in rdf.columns], args.min_market_non_null, "results_market", errors),
        "results_market_optional": {c: (1.0 - _null_rate(rdf, c)) for c in optional_market_cols if c in rdf.columns},
        "results_pred": _enforce_min_non_null(rdf, [c for c in pred_cols if c in rdf.columns], args.min_pred_non_null, "results_pred", errors),
    }

    top_missing = {}
    if not tdf.empty:
        top_missing["training_top10_null"] = (
            tdf.isna().mean().sort_values(ascending=False).head(10).round(4).to_dict()
        )
    if not rdf.empty:
        top_missing["results_top10_null"] = (
            rdf.isna().mean().sort_values(ascending=False).head(10).round(4).to_dict()
        )
    report["top_missing"] = top_missing

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[INFO] wrote validation report: {report_path}")
    if errors:
        print("[ERROR] backtest context validation failed:")
        for err in errors[:25]:
            print(f"  - {err}")
        return 1

    print("[OK] backtest context validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
