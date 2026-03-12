from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from phase_common import derive_phase, ensure_parent, resolve_paths, safe_read_csv, to_num


log = logging.getLogger("calibration_attribution")
logging.basicConfig(level=logging.INFO, format="[PHASE3] %(levelname)-8s %(message)s")


def _compute_ats_hit(actual_margin: pd.Series, market_spread: pd.Series) -> pd.Series:
    margin = to_num(actual_margin)
    spread = to_num(market_spread)
    return ((margin + spread) > 0).astype(float)


def _bucket_confidence(series: pd.Series) -> pd.Series:
    values = to_num(series)
    bins = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.01]
    labels = [
        "50-55%",
        "55-60%",
        "60-65%",
        "65-70%",
        "70-75%",
        "75-80%",
        "80-90%",
        "90%+",
    ]
    return pd.cut(values, bins=bins, labels=labels, include_lowest=True, right=False)


def build_calibration(backtest: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    frame = backtest.copy()
    frame["market_spread"] = to_num(frame.get("market_spread"))
    frame["ens_spread"] = to_num(frame.get("ens_spread"))
    frame["actual_margin"] = to_num(frame.get("actual_margin"))
    frame["ens_confidence"] = to_num(frame.get("ens_confidence"))
    frame = frame[
        frame["market_spread"].notna()
        & frame["actual_margin"].notna()
        & frame["ens_confidence"].between(0, 1, inclusive="both")
    ].copy()

    if frame.empty:
        buckets = pd.DataFrame(
            [{"bucket": "N/A", "n": 0, "expected_win_rate": np.nan, "actual_win_rate": np.nan, "calibration_gap": np.nan}]
        )
        summary = pd.DataFrame([{"metric": "status", "value": "BLOCKED: no ATS-calibratable rows"}])
    else:
        frame["ats_hit"] = _compute_ats_hit(frame["actual_margin"], frame["market_spread"])
        frame["bucket"] = _bucket_confidence(frame["ens_confidence"])
        buckets = (
            frame.groupby("bucket", dropna=True)
            .agg(
                n=("ats_hit", "size"),
                expected_win_rate=("ens_confidence", "mean"),
                actual_win_rate=("ats_hit", "mean"),
            )
            .reset_index()
        )
        buckets["calibration_gap"] = buckets["actual_win_rate"] - buckets["expected_win_rate"]

        brier = np.mean((frame["ens_confidence"] - frame["ats_hit"]) ** 2) if len(frame) else np.nan
        weighted_gap = (
            np.average(np.abs(buckets["calibration_gap"]), weights=buckets["n"]) if len(buckets) else np.nan
        )
        summary = pd.DataFrame(
            [
                {"metric": "rows_used", "value": int(len(frame))},
                {"metric": "bucket_count", "value": int(len(buckets))},
                {"metric": "brier_score", "value": float(brier) if pd.notna(brier) else np.nan},
                {"metric": "weighted_abs_calibration_gap", "value": float(weighted_gap) if pd.notna(weighted_gap) else np.nan},
            ]
        )

    bucket_path = out_dir / "calibration_buckets.csv"
    report_path = out_dir / "calibration_report.csv"
    buckets.to_csv(bucket_path, index=False)
    summary.to_csv(report_path, index=False)
    return {"buckets": bucket_path, "report": report_path}


def build_attribution(backtest: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    frame = backtest.copy()
    frame["actual_margin"] = to_num(frame.get("actual_margin"))
    frame["market_spread"] = to_num(frame.get("market_spread"))
    frame["actual_ats_home"] = _compute_ats_hit(frame["actual_margin"], frame["market_spread"])

    model_spread_cols = [
        c
        for c in frame.columns
        if c.endswith("_spread")
        and c
        not in {
            "market_spread",
            "opening_spread",
            "closing_spread",
            "spread_line",
            "pred_spread",
            "predicted_spread",
            "home_market_spread",
            "away_market_spread",
        }
    ]
    rows: list[dict[str, Any]] = []
    for col in sorted(model_spread_cols):
        pred = to_num(frame[col])
        valid_margin = pred.notna() & frame["actual_margin"].notna()
        mae = np.mean(np.abs(pred[valid_margin] - frame.loc[valid_margin, "actual_margin"])) if valid_margin.any() else np.nan
        corr = pred[valid_margin].corr(frame.loc[valid_margin, "actual_margin"]) if valid_margin.any() else np.nan

        valid_ats = valid_margin & frame["market_spread"].notna()
        if valid_ats.any():
            pred_home_cover = (pred[valid_ats] < frame.loc[valid_ats, "market_spread"]).astype(float)
            ats_hit = np.where(
                pred_home_cover == 1.0,
                frame.loc[valid_ats, "actual_ats_home"],
                1.0 - frame.loc[valid_ats, "actual_ats_home"],
            )
            ats_rate = float(np.nanmean(ats_hit))
            roi = float(np.nanmean(np.where(np.array(ats_hit) >= 0.5, 100 / 110, -1.0)))
            n_ats = int(valid_ats.sum())
        else:
            ats_rate = np.nan
            roi = np.nan
            n_ats = 0

        rows.append(
            {
                "model_component": col,
                "n_margin": int(valid_margin.sum()),
                "mae_vs_actual_margin": float(mae) if pd.notna(mae) else np.nan,
                "corr_vs_actual_margin": float(corr) if pd.notna(corr) else np.nan,
                "n_ats": n_ats,
                "ats_hit_rate": ats_rate,
                "ats_roi_flat_-110": roi,
            }
        )

    attribution = pd.DataFrame(rows).sort_values(["ats_hit_rate", "corr_vs_actual_margin"], ascending=False)
    attr_path = out_dir / "performance_attribution.csv"
    attribution.to_csv(attr_path, index=False)

    if model_spread_cols:
        corr = frame[model_spread_cols].apply(to_num).corr()
        corr_long = (
            corr.stack()
            .reset_index()
            .rename(columns={"level_0": "model_a", "level_1": "model_b", 0: "corr"})
        )
        corr_long = corr_long[corr_long["model_a"] < corr_long["model_b"]].sort_values("corr", ascending=False)
    else:
        corr_long = pd.DataFrame(columns=["model_a", "model_b", "corr"])
    red_path = out_dir / "metric_redundancy_report.csv"
    corr_long.to_csv(red_path, index=False)

    return {"attribution": attr_path, "redundancy": red_path}


def build_phase_performance(backtest: pd.DataFrame, out_dir: Path) -> Path:
    frame = backtest.copy()
    dt_col = "game_datetime" if "game_datetime" in frame.columns else "game_datetime_utc"
    frame["phase"] = derive_phase(frame.get(dt_col, pd.Series(np.nan, index=frame.index)))
    frame["market_spread"] = to_num(frame.get("market_spread"))
    frame["actual_margin"] = to_num(frame.get("actual_margin"))
    frame["actual_total"] = to_num(frame.get("actual_total"))
    frame["market_total"] = to_num(frame.get("market_total"))
    frame["ens_spread"] = to_num(frame.get("ens_spread"))
    frame["ens_total"] = to_num(frame.get("ens_total"))

    frame["ats_hit"] = np.where(
        frame["market_spread"].notna(),
        np.where(
            frame["ens_spread"] < frame["market_spread"],
            (frame["actual_margin"] + frame["market_spread"] > 0).astype(float),
            (frame["actual_margin"] + frame["market_spread"] < 0).astype(float),
        ),
        np.nan,
    )
    frame["total_hit"] = np.where(
        frame["market_total"].notna(),
        np.where(
            frame["ens_total"] > frame["market_total"],
            (frame["actual_total"] > frame["market_total"]).astype(float),
            (frame["actual_total"] < frame["market_total"]).astype(float),
        ),
        np.nan,
    )
    frame["ml_hit"] = np.where(
        frame["ens_spread"].notna(),
        np.where(frame["ens_spread"] < 0, frame["actual_margin"] > 0, frame["actual_margin"] < 0).astype(float),
        np.nan,
    )

    out = (
        frame.groupby("phase", dropna=False)
        .agg(
            n_games=("game_id", "size"),
            ats_n=("ats_hit", lambda x: int(pd.Series(x).notna().sum())),
            ats_hit_rate=("ats_hit", "mean"),
            total_n=("total_hit", lambda x: int(pd.Series(x).notna().sum())),
            total_hit_rate=("total_hit", "mean"),
            ml_n=("ml_hit", lambda x: int(pd.Series(x).notna().sum())),
            ml_hit_rate=("ml_hit", "mean"),
        )
        .reset_index()
    )
    out_path = out_dir / "phase_performance.csv"
    out.to_csv(out_path, index=False)
    return out_path


def build_seed_calibration(paths, out_dir: Path) -> Path:
    # Seed-linked model probabilities are not present in current backtest artifacts.
    out = pd.DataFrame(
        [
            {
                "status": "BLOCKED",
                "reason": "No seed-linked calibrated prediction probability artifact found.",
                "required_inputs": "seed_team, seed_opponent, model_probability, outcome",
            }
        ]
    )
    out_path = out_dir / "seed_calibration.csv"
    out.to_csv(out_path, index=False)
    return out_path


def build_pes_lift(paths, out_dir: Path) -> Path:
    layer_input = safe_read_csv(paths.rd2_root / "data" / "layer_backtests" / "layer_backtest_input.csv")
    if layer_input.empty:
        out = pd.DataFrame(
            [
                {
                    "status": "BLOCKED",
                    "reason": "layer_backtest_input.csv missing or empty",
                }
            ]
        )
    else:
        work = layer_input.copy()
        work["model_edge"] = to_num(work.get("model_edge"))
        work["ml_won"] = to_num(work.get("ml_won"))
        work["pes_diff"] = to_num(work.get("pes_diff"))
        work["is_tournament"] = to_num(work.get("is_tournament")).fillna(0) != 0
        scope = work["is_tournament"] & work["ml_won"].isin([0, 1]) & work["model_edge"].notna()
        base = work.loc[scope & (work["model_edge"] >= 4.0)]
        enh = work.loc[scope & (work["model_edge"] >= 4.0) & (work["pes_diff"] >= 5.0)]
        base_rate = float(base["ml_won"].mean()) if len(base) else np.nan
        enh_rate = float(enh["ml_won"].mean()) if len(enh) else np.nan
        out = pd.DataFrame(
            [
                {
                    "base_n": int(len(base)),
                    "base_hit_rate": base_rate,
                    "pes_filtered_n": int(len(enh)),
                    "pes_filtered_hit_rate": enh_rate,
                    "incremental_lift": (enh_rate - base_rate) if pd.notna(base_rate) and pd.notna(enh_rate) else np.nan,
                    "status": "PASS" if len(enh) >= 30 else "PROVISIONAL",
                }
            ]
        )
    out_path = out_dir / "pes_incremental_lift.csv"
    out.to_csv(out_path, index=False)
    return out_path


def build_edge_decay(backtest: pd.DataFrame, out_dir: Path) -> Path:
    frame = backtest.copy()
    frame["opening_spread"] = to_num(frame.get("opening_spread"))
    frame["closing_spread"] = to_num(frame.get("closing_spread"))
    frame["ens_spread"] = to_num(frame.get("ens_spread"))
    frame["market_spread"] = to_num(frame.get("market_spread"))
    frame["actual_margin"] = to_num(frame.get("actual_margin"))
    frame["clv_delta"] = to_num(frame.get("clv_delta"))

    valid = frame[
        frame["opening_spread"].notna()
        & frame["closing_spread"].notna()
        & frame["ens_spread"].notna()
        & frame["market_spread"].notna()
        & frame["actual_margin"].notna()
    ].copy()
    if valid.empty:
        out = pd.DataFrame(
            [{"status": "BLOCKED", "reason": "No rows with opening/closing/model spread and outcomes."}]
        )
    else:
        valid["line_move"] = valid["closing_spread"] - valid["opening_spread"]
        valid["line_move_abs"] = valid["line_move"].abs()
        valid["model_pref_home"] = valid["ens_spread"] < valid["closing_spread"]
        valid["market_move_to_home"] = valid["line_move"] < 0
        valid["move_against_model"] = valid["model_pref_home"] ^ valid["market_move_to_home"]
        valid["ats_hit"] = np.where(
            valid["model_pref_home"],
            (valid["actual_margin"] + valid["closing_spread"] > 0).astype(float),
            (valid["actual_margin"] + valid["closing_spread"] < 0).astype(float),
        )
        valid["move_bucket"] = pd.cut(
            valid["line_move_abs"],
            bins=[-0.001, 0.25, 0.5, 1.0, 2.0, 100.0],
            labels=["0-0.25", "0.25-0.5", "0.5-1.0", "1.0-2.0", "2.0+"],
        )
        out = (
            valid.groupby(["move_bucket", "move_against_model"], dropna=False)
            .agg(
                n=("ats_hit", "size"),
                ats_hit_rate=("ats_hit", "mean"),
                mean_clv=("clv_delta", "mean"),
                mean_line_move=("line_move", "mean"),
            )
            .reset_index()
        )
    out_path = out_dir / "edge_decay_report.csv"
    out.to_csv(out_path, index=False)
    return out_path


def write_recommendations(path: Path, artifacts: dict[str, Path]) -> None:
    ensure_parent(path)
    lines = [
        "# Calibration and Attribution Recommendations",
        "",
        f"Generated at UTC: {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## Outputs",
    ]
    for name, apath in artifacts.items():
        lines.append(f"- {name}: {apath}")

    lines.extend(
        [
            "",
            "## Recommended Actions",
            "1. Reweight submodels using `performance_attribution.csv` after filtering to models with stable ATS sample >= 200.",
            "2. Hold seed-specific deployment claims until a seed-linked probability artifact exists (`seed_calibration.csv` currently BLOCKED).",
            "3. Use `edge_decay_report.csv` to reduce stake when `move_against_model=true` and move bucket >= 1.0.",
            "4. Re-run this report after market-line backfill expands ATS sample beyond the current constrained set.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 calibration and performance attribution")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    paths = resolve_paths()
    out_dir = Path(args.output_dir)
    out_dir = out_dir if out_dir.is_absolute() else paths.rd2_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    backtest = safe_read_csv(paths.backtest_csv)
    if backtest.empty:
        log.error("Backtest source missing: %s", paths.backtest_csv)
        return 2

    artifacts: dict[str, Path] = {}
    artifacts.update(build_calibration(backtest, out_dir))
    artifacts.update(build_attribution(backtest, out_dir))
    artifacts["phase_performance"] = build_phase_performance(backtest, out_dir)
    artifacts["seed_calibration"] = build_seed_calibration(paths, out_dir)
    artifacts["pes_incremental_lift"] = build_pes_lift(paths, out_dir)
    artifacts["edge_decay"] = build_edge_decay(backtest, out_dir)

    rec_path = paths.rd2_root / "docs" / "reports" / "calibration_recommendations.md"
    write_recommendations(rec_path, artifacts)
    log.info("Wrote calibration/attribution outputs to %s", out_dir)
    log.info("Recommendations -> %s", rec_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
