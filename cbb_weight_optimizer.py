#!/usr/bin/env python3
"""Walk-forward optimizer for ATS metric weights."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("cbb_weight_optimizer")

DATA_PATH = Path("data/backtest_training_data.csv")
LEGACY_WEIGHTS_PATH = Path("data/backtest_optimized_weights.json")
OPT_DIR = Path("data/optimization")
HISTORY_PATH = OPT_DIR / "weight_history.csv"
ACTIVE_MODEL_PATH = OPT_DIR / "active_model.json"

EDGE_THRESHOLD = 1.5
MIN_VALIDATION_ROWS = 20
MIN_METRICS = 4
MIN_FOLDS = 3
RANDOM_CANDIDATES = 500
NEIGHBORHOOD_SIZE = 50
TOP_K_STAGE1 = 20
TOP_K_STAGE2 = 5

DEPLOY_THRESHOLDS = {
    "min_picks": 75,
    "min_hit_rate": 0.515,
    "max_log_loss": 0.695,
}


@dataclass(frozen=True)
class VariantConfig:
    name: str
    metrics: Tuple[str, ...]
    min_rows_required: int
    require_clv: bool = False


VARIANTS: Dict[str, VariantConfig] = {
    "A": VariantConfig(
        name="A",
        metrics=(
            "net_rtg_delta_l10",
            "cage_em_diff",
            "efg_delta_l10",
            "to_rate_delta_l5",
            "home_field",
            "rest_delta",
        ),
        min_rows_required=50,
    ),
    "B": VariantConfig(
        name="B",
        metrics=(
            "net_rtg_delta_l10",
            "net_rtg_delta_l5",
            "adj_ortg_delta",
            "adj_drtg_delta",
            "cage_em_diff",
            "cage_t_diff",
            "efg_delta_l10",
            "to_rate_delta_l5",
            "orb_delta_l10",
            "ftrate_delta_l5",
            "pace_delta_l5",
            "home_field",
            "rest_delta",
            "travel_fatigue_delta",
            "rot_efg_delta",
            "exec_tax_diff",
            "three_pt_fragility_diff",
            "rot_minshare_sd_diff",
            "star_availability_delta",
            "lineup_continuity_delta",
        ),
        min_rows_required=150,
    ),
    "C": VariantConfig(
        name="C",
        metrics=(
            "net_rtg_delta_l10",
            "cage_em_diff",
            "efg_delta_l10",
            "to_rate_delta_l5",
            "home_field",
            "rest_delta",
            "clv_delta",
            "exec_tax_diff",
        ),
        min_rows_required=100,
        require_clv=True,
    ),
}


@dataclass
class Fold:
    fold_no: int
    fold_date: pd.Timestamp
    train_idx: pd.Index
    val_idx: pd.Index


def project_to_simplex(weights: np.ndarray) -> np.ndarray:
    clipped = np.clip(weights, 0, None)
    total = clipped.sum()
    if total <= 0:
        return np.full_like(clipped, 1.0 / len(clipped))
    return clipped / total


def score_candidate(
    weights: np.ndarray,
    metrics: pd.DataFrame,
    metric_cols: Sequence[str],
    outcomes: pd.Series,
    edge_threshold: float = EDGE_THRESHOLD,
) -> Dict[str, float]:
    raw_score = (metrics[list(metric_cols)] * weights).sum(axis=1)

    picks_mask = raw_score.abs() >= edge_threshold
    picks = raw_score[picks_mask]
    pick_outcomes = outcomes[picks_mask].dropna()
    aligned_picks = picks.loc[pick_outcomes.index]

    if len(pick_outcomes) < 10:
        return {
            "log_loss": 0.693,
            "hit_rate": 0.5,
            "roi": 0.0,
            "n_picks": int(len(pick_outcomes)),
            "n_games": int(len(outcomes)),
        }

    home_picks = aligned_picks > 0
    correct = (home_picks & (pick_outcomes == 1)) | (~home_picks & (pick_outcomes == 0))
    hit_rate = float(correct.mean())
    roi = float((correct.sum() * 0.909 - (~correct).sum()) / len(correct))

    prob_home = 1 / (1 + np.exp(-raw_score))
    valid = outcomes.dropna()
    prob_aligned = prob_home.loc[valid.index].clip(0.001, 0.999)
    ll = float(-(valid * np.log(prob_aligned) + (1 - valid) * np.log(1 - prob_aligned)).mean())

    return {
        "log_loss": ll,
        "hit_rate": hit_rate,
        "roi": roi,
        "n_picks": int(len(correct)),
        "n_games": int(len(outcomes)),
    }


def build_folds(df: pd.DataFrame, min_train: int = 100, fold_days: int = 7, val_days: int = 14) -> List[Fold]:
    sorted_df = df.sort_values("game_date")
    if sorted_df.empty:
        return []

    start_date = sorted_df["game_date"].min() + pd.Timedelta(days=fold_days)
    end_date = sorted_df["game_date"].max()
    fold_date = start_date
    folds: List[Fold] = []
    fold_no = 1

    while fold_date <= end_date:
        train_idx = sorted_df.index[sorted_df["game_date"] < fold_date]
        val_end = fold_date + pd.Timedelta(days=val_days)
        val_idx = sorted_df.index[
            (sorted_df["game_date"] >= fold_date) & (sorted_df["game_date"] < val_end)
        ]
        if len(train_idx) >= min_train and len(val_idx) >= MIN_VALIDATION_ROWS:
            folds.append(Fold(fold_no=fold_no, fold_date=fold_date, train_idx=train_idx, val_idx=val_idx))
            fold_no += 1
        fold_date += pd.Timedelta(days=fold_days)

    return folds


def evaluate_weights(df: pd.DataFrame, folds: Sequence[Fold], metrics: Sequence[str], weights: np.ndarray) -> Tuple[float, List[Dict[str, float]]]:
    weights = project_to_simplex(weights)
    fold_metrics: List[Dict[str, float]] = []
    for fold in folds:
        train_df = df.loc[fold.train_idx]
        val_df = df.loc[fold.val_idx]

        scaler = StandardScaler()
        train_x = pd.DataFrame(
            scaler.fit_transform(train_df[list(metrics)]),
            columns=metrics,
            index=train_df.index,
        )
        val_x = pd.DataFrame(
            scaler.transform(val_df[list(metrics)]),
            columns=metrics,
            index=val_df.index,
        )

        # Score validation fold only; scaler is fit on training-only rows.
        scored = score_candidate(weights, val_x, metrics, val_df["home_covered_ats"])
        scored.update(
            {
                "fold_no": fold.fold_no,
                "fold_date": fold.fold_date.strftime("%Y-%m-%d"),
                "n_train": int(len(train_df)),
                "n_val": int(len(val_df)),
            }
        )
        fold_metrics.append(scored)

    median_log_loss = float(np.median([f["log_loss"] for f in fold_metrics])) if fold_metrics else float("inf")
    return median_log_loss, fold_metrics


def optimize_variant(df: pd.DataFrame, variant: VariantConfig, min_rows_override: Optional[int] = None) -> Dict[str, object]:
    variant_df = df.copy()
    if variant.require_clv:
        variant_df = variant_df[variant_df["clv_delta"].notna()]

    min_rows = min_rows_override if min_rows_override is not None else variant.min_rows_required
    if len(variant_df) < min_rows:
        return {
            "status": "skipped",
            "reason": f"insufficient_rows:{len(variant_df)}<{min_rows}",
        }

    available = [m for m in variant.metrics if m in variant_df.columns]
    coverage = {m: float(variant_df[m].notna().mean()) for m in available}
    selected_metrics = [m for m in available if coverage[m] >= 0.4]
    dropped_metrics = [m for m in available if m not in selected_metrics]

    if len(selected_metrics) < MIN_METRICS:
        return {
            "status": "skipped",
            "reason": "fewer_than_4_metrics_after_coverage_filter",
            "dropped_metrics": dropped_metrics,
        }

    model_df = variant_df.dropna(subset=["game_date", "home_covered_ats", *selected_metrics]).copy()
    folds = build_folds(model_df, min_train=min_rows)
    if len(folds) < MIN_FOLDS:
        return {
            "status": "skipped",
            "reason": f"insufficient_folds:{len(folds)}<{MIN_FOLDS}",
            "dropped_metrics": dropped_metrics,
            "metrics": selected_metrics,
        }

    n_metrics = len(selected_metrics)
    rng = np.random.default_rng(42)

    stage1 = rng.dirichlet(np.ones(n_metrics), size=RANDOM_CANDIDATES)
    scored_stage1: List[Tuple[float, np.ndarray]] = []
    for candidate in stage1:
        median_ll, _ = evaluate_weights(model_df, folds, selected_metrics, candidate)
        scored_stage1.append((median_ll, candidate))
    scored_stage1.sort(key=lambda item: item[0])
    top_candidates = [w for _, w in scored_stage1[:TOP_K_STAGE1]]

    neighbors: List[np.ndarray] = []
    for candidate in top_candidates:
        for _ in range(NEIGHBORHOOD_SIZE):
            perturb = rng.uniform(-0.05, 0.05, size=n_metrics)
            neighbors.append(project_to_simplex(candidate + perturb))

    scored_stage2: List[Tuple[float, np.ndarray]] = []
    for candidate in neighbors:
        median_ll, _ = evaluate_weights(model_df, folds, selected_metrics, candidate)
        scored_stage2.append((median_ll, candidate))
    scored_stage2.sort(key=lambda item: item[0])
    top5 = [w for _, w in scored_stage2[:TOP_K_STAGE2]]

    best_weights: Optional[np.ndarray] = None
    best_median_ll = float("inf")
    best_fold_metrics: List[Dict[str, float]] = []
    for candidate in top5:
        result = minimize(
            lambda w: evaluate_weights(model_df, folds, selected_metrics, project_to_simplex(w))[0],
            x0=candidate,
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-4},
        )
        polished = project_to_simplex(result.x)
        median_ll, fold_metrics = evaluate_weights(model_df, folds, selected_metrics, polished)
        if median_ll < best_median_ll:
            best_median_ll = median_ll
            best_weights = polished
            best_fold_metrics = fold_metrics

    assert best_weights is not None
    aggregate = {
        "log_loss": float(np.median([m["log_loss"] for m in best_fold_metrics])),
        "hit_rate": float(np.mean([m["hit_rate"] for m in best_fold_metrics])),
        "roi": float(np.mean([m["roi"] for m in best_fold_metrics])),
        "n_picks": int(sum(m["n_picks"] for m in best_fold_metrics)),
        "n_folds": int(len(best_fold_metrics)),
    }

    full_scaler = StandardScaler()
    full_scaler.fit(model_df[selected_metrics])

    return {
        "status": "optimized",
        "metrics": selected_metrics,
        "weights": best_weights.tolist(),
        "aggregate_metrics": aggregate,
        "fold_metrics": best_fold_metrics,
        "dropped_metrics": dropped_metrics,
        "scaler_mean": full_scaler.mean_.tolist(),
        "scaler_scale": full_scaler.scale_.tolist(),
    }


def should_deploy(metrics: Dict[str, float], prior_metrics: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
    if metrics["n_picks"] < DEPLOY_THRESHOLDS["min_picks"]:
        return False, "insufficient_picks"
    if metrics["hit_rate"] < DEPLOY_THRESHOLDS["min_hit_rate"]:
        return False, "hit_rate_below_floor"
    if metrics["log_loss"] > DEPLOY_THRESHOLDS["max_log_loss"]:
        return False, "log_loss_above_floor"
    return True, "approved"


def ensure_history_file() -> None:
    OPT_DIR.mkdir(parents=True, exist_ok=True)
    if HISTORY_PATH.exists():
        return
    with HISTORY_PATH.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "run_id",
                "model_variant",
                "deployed_at",
                "log_loss",
                "hit_rate",
                "roi",
                "n_picks",
                "n_folds",
                "deployment_status",
                "rejection_reason",
                "metrics_used",
                "n_metrics",
            ]
        )


def append_history_row(row: Dict[str, object], dry_run: bool) -> None:
    if dry_run:
        return
    ensure_history_file()
    with HISTORY_PATH.open("a", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                row["run_id"],
                row["model_variant"],
                row["deployed_at"],
                row["log_loss"],
                row["hit_rate"],
                row["roi"],
                row["n_picks"],
                row["n_folds"],
                row["deployment_status"],
                row["rejection_reason"],
                row["metrics_used"],
                row["n_metrics"],
            ]
        )


def run_report_only() -> int:
    if not HISTORY_PATH.exists():
        print("No weight history found (data/optimization/weight_history.csv missing).")
        return 0
    hist = pd.read_csv(HISTORY_PATH)
    if hist.empty:
        print("Weight history exists but has no rows.")
        return 0
    summary = hist.groupby("model_variant").agg(
        runs=("run_id", "count"),
        avg_log_loss=("log_loss", "mean"),
        avg_hit_rate=("hit_rate", "mean"),
        avg_roi=("roi", "mean"),
        approvals=("deployment_status", lambda s: int((s == "approved").sum())),
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    return 0


def load_training_df() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    if "game_date" not in df.columns:
        if "game_datetime_utc" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_datetime_utc"], errors="coerce").dt.normalize()
        else:
            raise ValueError("Training data requires game_date or game_datetime_utc column")
    else:
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.normalize()

    if "home_covered_ats" not in df.columns:
        raise ValueError("Training data requires home_covered_ats column")

    df["home_covered_ats"] = pd.to_numeric(df["home_covered_ats"], errors="coerce")
    return df


def weight_file_for_variant(name: str) -> Path:
    return OPT_DIR / f"weights_model_{name.lower()}.json"


def fold_results_file(run_id: str) -> Path:
    safe = run_id.replace(":", "").replace("-", "")
    return OPT_DIR / f"fold_results_{safe}.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward ATS metric weight optimizer")
    parser.add_argument("--variant", choices=sorted(VARIANTS.keys()))
    parser.add_argument("--min-rows", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    if args.report_only:
        return run_report_only()

    try:
        df = load_training_df()
    except (FileNotFoundError, ValueError) as exc:
        log.warning("%s", exc)
        return 0

    OPT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).isoformat()
    season = int(df.get("season", pd.Series([datetime.now().year])).dropna().max())
    chosen_variants = [args.variant] if args.variant else ["A", "B", "C"]

    approved: List[Tuple[str, Dict[str, object], Path]] = []

    for variant_name in chosen_variants:
        cfg = VARIANTS[variant_name]
        result = optimize_variant(df, cfg, min_rows_override=args.min_rows)
        if result["status"] != "optimized":
            log.warning("Variant %s skipped: %s", variant_name, result.get("reason"))
            continue

        deploy_ok, reason = should_deploy(result["aggregate_metrics"])
        deployment_status = "approved" if deploy_ok else "rejected"

        weights_payload = {
            "model_variant": variant_name,
            "run_id": run_id,
            "season": season,
            "metrics": result["metrics"],
            "weights": result["weights"],
            "scaler_mean": result["scaler_mean"],
            "scaler_scale": result["scaler_scale"],
            "edge_threshold": EDGE_THRESHOLD,
            "aggregate_metrics": result["aggregate_metrics"],
            "deployment_status": deployment_status,
            "rejection_reason": None if deploy_ok else reason,
            "dropped_metrics": result.get("dropped_metrics", []),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        log.info(
            "Variant %s aggregate: log_loss=%.4f hit_rate=%.4f roi=%.4f picks=%s folds=%s status=%s dropped=%s",
            variant_name,
            result["aggregate_metrics"]["log_loss"],
            result["aggregate_metrics"]["hit_rate"],
            result["aggregate_metrics"]["roi"],
            result["aggregate_metrics"]["n_picks"],
            result["aggregate_metrics"]["n_folds"],
            deployment_status,
            result.get("dropped_metrics", []),
        )

        history_row = {
            "run_id": run_id,
            "model_variant": variant_name,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
            "log_loss": result["aggregate_metrics"]["log_loss"],
            "hit_rate": result["aggregate_metrics"]["hit_rate"],
            "roi": result["aggregate_metrics"]["roi"],
            "n_picks": result["aggregate_metrics"]["n_picks"],
            "n_folds": result["aggregate_metrics"]["n_folds"],
            "deployment_status": deployment_status,
            "rejection_reason": "" if deploy_ok else reason,
            "metrics_used": "|".join(result["metrics"]),
            "n_metrics": len(result["metrics"]),
        }
        append_history_row(history_row, dry_run=args.dry_run)

        if not args.dry_run:
            fold_df = pd.DataFrame(result["fold_metrics"])
            fold_df.to_csv(fold_results_file(run_id), index=False)

            if deploy_ok:
                output_path = weight_file_for_variant(variant_name)
                output_path.write_text(json.dumps(weights_payload, indent=2))
                approved.append((variant_name, weights_payload, output_path))
        elif deploy_ok:
            approved.append((variant_name, weights_payload, weight_file_for_variant(variant_name)))

    if approved:
        chosen = sorted(approved, key=lambda item: item[1]["aggregate_metrics"]["log_loss"])[0]
        active_payload = {
            "active_variant": chosen[0],
            "weights_file": str(chosen[2]),
            "selected_at": datetime.now(timezone.utc).isoformat(),
            "selection_reason": "lowest log_loss among approved variants",
        }
        if args.dry_run:
            log.info("Dry run active model would be %s (%s)", chosen[0], chosen[2])
        else:
            ACTIVE_MODEL_PATH.write_text(json.dumps(active_payload, indent=2))
    else:
        log.warning("No variants approved; existing weights remain active.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
