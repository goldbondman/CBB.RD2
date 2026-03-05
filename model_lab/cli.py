from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    DEFAULT_MODEL_NAMES,
    ModelLabConfig,
    default_run_id,
    ensure_run_dir,
    get_git_sha,
    load_manifest,
    update_manifest,
    utc_now_iso,
    write_manifest,
)
from .data_builder import FrameBuildResult, build_frames, write_frames
from .edge_analyzer import run_edge_analysis
from .ensemble import build_market_dataset, evaluate_ensemble
from .executive_summary import generate_exec_summary
from .evaluators import evaluate_predictions
from .feature_selector import select_features_from_run
from .feature_tests import build_generated_feature_set_payloads, run_feature_tests
from .model_wrappers import load_all_available_predictions
from .splits import Fold, build_rolling_folds
from .window_grid import evaluate_window_grid


def _merge_blocked(existing: list[str] | None, incoming: list[str]) -> list[str]:
    values = set(existing or [])
    values.update(incoming)
    return sorted(values)


def _base_manifest(config: ModelLabConfig, run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "git_sha": get_git_sha(config.repo_root),
        "blocked_reasons": [],
        "frames": {},
        "folds": {},
        "nan_report": {},
        "artifacts": {},
    }


def _ensure_manifest(config: ModelLabConfig, run_dir: Path, run_id: str) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    if manifest:
        return manifest
    manifest = _base_manifest(config, run_id)
    write_manifest(run_dir, manifest)
    return manifest


def _record_frame_build(run_dir: Path, result: FrameBuildResult, frame_artifacts: dict[str, str]) -> None:
    manifest = load_manifest(run_dir)
    manifest["updated_at_utc"] = utc_now_iso()
    manifest["frames"] = {
        "paths": frame_artifacts,
        "row_counts": result.frame_row_counts,
        "sources": result.source_paths,
    }
    manifest["nan_report"] = result.nan_report
    manifest["blocked_reasons"] = _merge_blocked(manifest.get("blocked_reasons"), result.blocked_reasons)
    manifest.setdefault("artifacts", {}).update(frame_artifacts)
    write_manifest(run_dir, manifest)


def _frame_for_market(frames: FrameBuildResult, market: str) -> pd.DataFrame:
    if market == "spread":
        return frames.spread_frame
    if market == "total":
        return frames.total_frame
    if market == "ml":
        return frames.ml_frame
    raise ValueError(f"Unsupported market: {market}")


def _evaluate_model_on_fold(market: str, test_df: pd.DataFrame, pred: pd.Series, config: ModelLabConfig) -> dict[str, float]:
    if market == "spread":
        return evaluate_predictions(
            y_true=test_df["actual_margin"],
            y_pred=pred,
            market_line=test_df["spread_line"],
            odds=None,
            market="spread",
            default_odds=config.default_odds,
            line_open=test_df.get("spread_open"),
            line_close=test_df.get("spread_close"),
        )
    if market == "total":
        return evaluate_predictions(
            y_true=test_df["actual_total"],
            y_pred=pred,
            market_line=test_df["total_line"],
            odds=None,
            market="total",
            default_odds=config.default_odds,
            line_open=test_df.get("total_open"),
            line_close=test_df.get("total_close"),
        )
    if market == "ml":
        return evaluate_predictions(
            y_true=test_df["home_won"],
            y_pred=pred,
            market_line=None,
            odds={"home_ml": test_df.get("home_ml"), "away_ml": test_df.get("away_ml")},
            market="ml",
            default_odds=config.default_odds,
        )
    raise ValueError(f"Unsupported market: {market}")


def _aggregate_fold_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "n_folds": 0.0,
            "n_games": 0.0,
            "hit_rate": float("nan"),
            "roi": float("nan"),
            "clv_mean": float("nan"),
            "mae": float("nan"),
            "brier": float("nan"),
            "calibration_ece": float("nan"),
        }

    df = pd.DataFrame(rows)
    games = pd.to_numeric(df.get("graded_n", 0), errors="coerce").fillna(0.0)
    total_games = float(games.sum())

    def weighted(col: str) -> float:
        vals = pd.to_numeric(df.get(col), errors="coerce")
        mask = vals.notna() & (games > 0)
        if mask.sum() == 0:
            return float("nan")
        return float((vals[mask] * games[mask]).sum() / games[mask].sum())

    return {
        "n_folds": float(df["fold_id"].nunique()),
        "n_games": total_games,
        "hit_rate": weighted("hit_rate"),
        "roi": weighted("roi"),
        "clv_mean": weighted("clv_mean"),
        "mae": weighted("mae"),
        "brier": weighted("brier"),
        "calibration_ece": weighted("calibration_ece"),
    }


def cmd_build_frames(args: argparse.Namespace, config: ModelLabConfig) -> int:
    run_id = args.run_id or default_run_id()
    run_dir = ensure_run_dir(config, run_id)
    _ensure_manifest(config, run_dir, run_id)

    frames = build_frames(config)
    artifacts = write_frames(frames, run_dir)
    _record_frame_build(run_dir, frames, artifacts)

    print(f"run_id={run_id}")
    print(json.dumps(frames.frame_row_counts, indent=2))
    return 0


def cmd_score_models(args: argparse.Namespace, config: ModelLabConfig) -> int:
    run_id = args.run_id or default_run_id()
    run_dir = ensure_run_dir(config, run_id)
    _ensure_manifest(config, run_dir, run_id)

    frames = build_frames(config)
    frame_artifacts = write_frames(frames, run_dir)
    _record_frame_build(run_dir, frames, frame_artifacts)

    predictions_by_model = load_all_available_predictions(config, DEFAULT_MODEL_NAMES)
    blocked = []
    rows: list[dict[str, Any]] = []
    fold_manifest: dict[str, list[dict[str, Any]]] = {}

    for market in ["spread", "total", "ml"]:
        dataset = build_market_dataset(predictions_by_model, market)
        folds, split_blocked = build_rolling_folds(dataset, config)
        blocked.extend(split_blocked)
        fold_manifest[market] = [f.to_manifest() for f in folds]

        if dataset.empty:
            blocked.append(f"score_models_empty_dataset:{market}")
            continue
        if not folds:
            blocked.append(f"score_models_no_folds:{market}")
            continue

        model_cols = [c for c in dataset.columns if c.startswith("model_")]
        for model_col in model_cols:
            model_name = model_col.replace("model_", "")
            fold_rows = []
            for fold in folds:
                test_idx = [i for i in fold.test_index if i in dataset.index]
                test_df = dataset.loc[test_idx].copy()
                if test_df.empty:
                    continue

                pred = pd.to_numeric(test_df[model_col], errors="coerce")
                metrics = _evaluate_model_on_fold(market, test_df, pred, config)
                metrics["fold_id"] = fold.fold_id
                fold_rows.append(metrics)

            agg = _aggregate_fold_metrics(fold_rows)
            rows.append(
                {
                    "market": market,
                    "model_name": model_name,
                    "n_folds": agg["n_folds"],
                    "n_games": agg["n_games"],
                    "hit_rate": agg["hit_rate"],
                    "roi": agg["roi"],
                    "clv_mean": agg["clv_mean"],
                    "mae": agg["mae"],
                    "brier": agg["brier"],
                    "calibration_ece": agg["calibration_ece"],
                }
            )

    scorecard = pd.DataFrame(rows)
    scorecard_path = run_dir / "model_scorecard.csv"
    scorecard.to_csv(scorecard_path, index=False)

    manifest = load_manifest(run_dir)
    manifest["updated_at_utc"] = utc_now_iso()
    manifest["folds"] = fold_manifest
    manifest["blocked_reasons"] = _merge_blocked(manifest.get("blocked_reasons"), blocked)
    manifest.setdefault("artifacts", {})["model_scorecard"] = str(scorecard_path.resolve())
    write_manifest(run_dir, manifest)

    print(f"run_id={run_id}")
    print(f"model_scorecard={scorecard_path}")
    return 0


def cmd_feature_signal(args: argparse.Namespace, config: ModelLabConfig) -> int:
    run_id = args.run_id or default_run_id()
    run_dir = ensure_run_dir(config, run_id)
    _ensure_manifest(config, run_dir, run_id)

    frames = build_frames(config)
    frame_artifacts = write_frames(frames, run_dir)
    _record_frame_build(run_dir, frames, frame_artifacts)

    market = args.market
    frame = _frame_for_market(frames, market)

    folds, split_blocked = build_rolling_folds(frame, config)
    result = run_feature_tests(
        frame,
        folds,
        market=market,
        random_seed=args.random_seed,
        max_features=args.max_features,
        stability_min=config.selector_stability_min,
        sign_consistency_min=config.selector_sign_consistency_min,
        permutation_delta_min=config.selector_permutation_delta_min,
        ablation_delta_min=config.selector_ablation_delta_min,
        correlation_max=config.selector_correlation_max,
        cap_conservative=config.feature_cap_conservative,
        cap_balanced=config.feature_cap_balanced,
        cap_aggressive=config.feature_cap_aggressive,
    )

    scorecard_path = run_dir / "feature_scorecard.csv"
    stability_path = run_dir / "feature_stability.csv"
    window_grid_path = run_dir / "window_grid_scorecard.csv"
    feature_set_report_path = run_dir / "feature_set_report.md"
    result.feature_scorecard.to_csv(scorecard_path, index=False)
    result.feature_stability.to_csv(stability_path, index=False)
    result.window_grid_scorecard.to_csv(window_grid_path, index=False)
    feature_set_report_path.write_text(result.feature_set_report, encoding="utf-8")

    feature_set_dir = (config.repo_root / "feature_sets" / "generated").resolve()
    feature_set_dir.mkdir(parents=True, exist_ok=True)
    payloads = build_generated_feature_set_payloads(
        market=market,
        selected_sets=result.selected_feature_sets,
        selector_config=result.selector_config,
        window_contract=result.window_contract,
        window_grid_scorecard=result.window_grid_scorecard,
        location_aware_variants=result.location_aware_variants,
    )
    generated_paths: dict[str, str] = {}
    for profile, payload in payloads.items():
        out_path = feature_set_dir / f"{market}_AUTO_V2_{profile}.json"
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        generated_paths[f"feature_set_{market}_{profile}"] = str(out_path.resolve())

    summary_path, summary_blocked = generate_exec_summary(run_dir)

    manifest = load_manifest(run_dir)
    manifest["updated_at_utc"] = utc_now_iso()
    manifest.setdefault("folds", {})[market] = [f.to_manifest() for f in folds]
    all_blocked = split_blocked + result.blocked_reasons + summary_blocked
    manifest["feature_selector_config"] = result.selector_config
    manifest["feature_sets"] = result.selected_feature_sets
    manifest["blocked_reasons"] = _merge_blocked(manifest.get("blocked_reasons"), all_blocked)
    manifest.setdefault("artifacts", {}).update(
        {
            "feature_scorecard": str(scorecard_path.resolve()),
            "feature_stability": str(stability_path.resolve()),
            "window_grid_scorecard": str(window_grid_path.resolve()),
            "feature_set_report": str(feature_set_report_path.resolve()),
            "exec_summary": str(summary_path.resolve()),
            **generated_paths,
        }
    )
    write_manifest(run_dir, manifest)

    print(f"run_id={run_id}")
    print(f"feature_scorecard={scorecard_path}")
    print(f"feature_stability={stability_path}")
    print(f"window_grid_scorecard={window_grid_path}")
    print(f"feature_set_report={feature_set_report_path}")
    for key in sorted(generated_paths):
        print(f"{key}={generated_paths[key]}")
    print(f"exec_summary={summary_path}")
    return 0


def cmd_window_grid(args: argparse.Namespace, config: ModelLabConfig) -> int:
    run_id = args.run_id or default_run_id()
    run_dir = ensure_run_dir(config, run_id)
    _ensure_manifest(config, run_dir, run_id)

    frames = build_frames(config)
    frame_artifacts = write_frames(frames, run_dir)
    _record_frame_build(run_dir, frames, frame_artifacts)

    markets = args.markets if args.markets else ["spread", "total", "ml"]
    blocked: list[str] = []
    fold_manifest: dict[str, list[dict[str, Any]]] = {}
    rows: list[pd.DataFrame] = []
    contract_by_market: dict[str, dict[str, Any]] = {}

    for market in markets:
        frame = _frame_for_market(frames, market)
        if frame.empty:
            blocked.append(f"window_grid_empty_frame:{market}")
            continue

        folds, split_blocked = build_rolling_folds(frame, config)
        blocked.extend(split_blocked)
        fold_manifest[market] = [f.to_manifest() for f in folds]
        if not folds:
            blocked.append(f"window_grid_no_folds:{market}")
            continue

        result = evaluate_window_grid(frame, folds, market=market, config=config)
        blocked.extend(result.blocked_reasons)
        contract_by_market[market] = result.window_contract
        rows.append(result.window_grid_scorecard.copy())

    scorecard = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    out_path = run_dir / "window_grid_scorecard.csv"
    scorecard.to_csv(out_path, index=False)

    manifest = load_manifest(run_dir)
    manifest["updated_at_utc"] = utc_now_iso()
    manifest["folds"] = {**manifest.get("folds", {}), **fold_manifest}
    manifest["window_grid_contract"] = contract_by_market
    manifest["blocked_reasons"] = _merge_blocked(manifest.get("blocked_reasons"), blocked)
    manifest.setdefault("artifacts", {})["window_grid_scorecard"] = str(out_path.resolve())
    write_manifest(run_dir, manifest)

    print(f"run_id={run_id}")
    print(f"window_grid_scorecard={out_path}")
    return 0


def cmd_select_features(args: argparse.Namespace, config: ModelLabConfig) -> int:
    run_id = args.run_id or default_run_id()
    run_dir = ensure_run_dir(config, run_id)
    _ensure_manifest(config, run_dir, run_id)

    markets = args.markets if args.markets else ["spread", "total", "ml"]
    result = select_features_from_run(run_dir, config, markets=markets)

    manifest = load_manifest(run_dir)
    manifest["updated_at_utc"] = utc_now_iso()
    manifest["feature_sets"] = result.selected_feature_sets
    manifest["location_aware_variants"] = result.location_aware_variants
    manifest["blocked_reasons"] = _merge_blocked(manifest.get("blocked_reasons"), result.blocked_reasons)
    manifest.setdefault("artifacts", {}).update(
        {
            "feature_set_report": str(result.feature_set_report_path.resolve()),
            **result.generated_paths,
        }
    )
    write_manifest(run_dir, manifest)

    print(f"run_id={run_id}")
    print(f"feature_set_report={result.feature_set_report_path}")
    for key in sorted(result.generated_paths):
        print(f"{key}={result.generated_paths[key]}")
    return 0


def cmd_edge_analyze(args: argparse.Namespace, config: ModelLabConfig) -> int:
    run_id = args.run_id or default_run_id()
    run_dir = ensure_run_dir(config, run_id)
    _ensure_manifest(config, run_dir, run_id)

    result = run_edge_analysis(
        run_dir,
        config,
        market=args.market,
        model_name=args.model,
        min_n=int(args.min_n),
        limit=int(args.limit) if args.limit is not None else None,
    )

    manifest = load_manifest(run_dir)
    manifest["updated_at_utc"] = utc_now_iso()
    edge_key = f"{args.market}:{args.model}"
    manifest.setdefault("edge_analyzer", {})[edge_key] = {
        "rows_analyzed": result.rows_analyzed,
        "run_manifest": str(result.run_manifest_path.resolve()),
        "blocked_segments": result.blocked_segments,
        "blocked_reasons": result.blocked_reasons,
    }
    manifest["blocked_reasons"] = _merge_blocked(manifest.get("blocked_reasons"), result.blocked_reasons)
    manifest.setdefault("artifacts", {}).update(
        {
            f"edge_bucket_report_{args.market}_{args.model}": str(result.edge_bucket_report_path.resolve()),
            f"segment_report_{args.market}_{args.model}": str(result.segment_report_path.resolve()),
            f"worst_misses_{args.market}_{args.model}": str(result.worst_misses_path.resolve()),
            f"edge_exec_summary_{args.market}_{args.model}": str(result.exec_summary_path.resolve()),
            "edge_analyzer_run_manifest": str(result.run_manifest_path.resolve()),
        }
    )
    write_manifest(run_dir, manifest)

    print(f"run_id={run_id}")
    print(f"edge_bucket_report={result.edge_bucket_report_path}")
    print(f"segment_report={result.segment_report_path}")
    print(f"worst_misses={result.worst_misses_path}")
    print(f"edge_exec_summary={result.exec_summary_path}")
    print(f"edge_analyzer_manifest={result.run_manifest_path}")
    return 0


def cmd_exec_summary(args: argparse.Namespace, config: ModelLabConfig) -> int:
    run_id = args.run_id or default_run_id()
    run_dir = ensure_run_dir(config, run_id)
    _ensure_manifest(config, run_dir, run_id)

    summary_path, summary_blocked = generate_exec_summary(run_dir)

    manifest = load_manifest(run_dir)
    manifest["updated_at_utc"] = utc_now_iso()
    manifest["blocked_reasons"] = _merge_blocked(manifest.get("blocked_reasons"), summary_blocked)
    manifest.setdefault("artifacts", {})["exec_summary"] = str(summary_path.resolve())
    write_manifest(run_dir, manifest)

    print(f"run_id={run_id}")
    print(f"exec_summary={summary_path}")
    return 0


def cmd_ensemble(args: argparse.Namespace, config: ModelLabConfig) -> int:
    run_id = args.run_id or default_run_id()
    run_dir = ensure_run_dir(config, run_id)
    _ensure_manifest(config, run_dir, run_id)

    predictions_by_model = load_all_available_predictions(config, DEFAULT_MODEL_NAMES)
    blocked: list[str] = []

    markets = args.markets if args.markets else ["spread", "total", "ml"]
    weights_payload = {
        "run_id": run_id,
        "generated_at_utc": utc_now_iso(),
        "constraints": {
            "weight_min": 0.0,
            "weight_sum": 1.0,
            "max_weight": float(args.max_weight),
        },
        "weights_by_market": {},
        "fold_summary": {},
        "blocked_reasons": [],
    }

    fold_manifest: dict[str, list[dict[str, Any]]] = {}

    config.max_weight = float(args.max_weight)

    for market in markets:
        dataset = build_market_dataset(predictions_by_model, market)
        folds, split_blocked = build_rolling_folds(dataset, config)
        blocked.extend(split_blocked)
        fold_manifest[market] = [f.to_manifest() for f in folds]

        result = evaluate_ensemble(market, dataset, folds, config)
        blocked.extend(result.blocked_reasons)

        if result.weights_by_market:
            weights_payload["weights_by_market"].update(result.weights_by_market)
        if not result.fold_results.empty:
            weights_payload["fold_summary"][market] = result.fold_results.to_dict(orient="records")

    weights_payload["blocked_reasons"] = sorted(set(blocked))

    output_path = run_dir / "ensemble_weights.json"
    output_path.write_text(json.dumps(weights_payload, indent=2), encoding="utf-8")

    manifest = load_manifest(run_dir)
    manifest["updated_at_utc"] = utc_now_iso()
    manifest["folds"] = {**manifest.get("folds", {}), **fold_manifest}
    manifest["blocked_reasons"] = _merge_blocked(manifest.get("blocked_reasons"), blocked)
    manifest.setdefault("artifacts", {})["ensemble_weights"] = str(output_path.resolve())
    write_manifest(run_dir, manifest)

    print(f"run_id={run_id}")
    print(f"ensemble_weights={output_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model Lab CLI")
    parser.add_argument("--run-id", default=None, help="Run identifier. Defaults to UTC timestamp.")
    parser.add_argument("--min-train-games", type=int, default=None, help="Override minimum train rows per fold.")
    parser.add_argument("--min-test-games", type=int, default=None, help="Override minimum test rows per fold.")
    parser.add_argument(
        "--date-test-window-days",
        type=int,
        default=None,
        help="Override date-based test window length in days.",
    )
    parser.add_argument(
        "--date-step-days",
        type=int,
        default=None,
        help="Override date-based fold step length in days.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-frames", help="Build spread/total/ml frames for Model Lab")

    sub.add_parser("score-models", help="Score existing model outputs on rolling forward folds")

    p_feature = sub.add_parser("feature-signal", help="Run univariate/permutation/drop-one feature tests")
    p_feature.add_argument("--market", choices=["spread", "total", "ml"], default="spread")
    p_feature.add_argument("--random-seed", type=int, default=42)
    p_feature.add_argument("--max-features", type=int, default=None)

    p_window = sub.add_parser("window-grid", help="Evaluate fixed rolling-window feature combos on rolling folds")
    p_window.add_argument("--run-id", default=None, help="Run identifier. Defaults to UTC timestamp.")
    p_window.add_argument("--markets", nargs="*", choices=["spread", "total", "ml"], default=None)

    p_select = sub.add_parser("select-features", help="Generate AUTO_V2 feature sets from run scorecards")
    p_select.add_argument("--run-id", default=None, help="Run identifier. Defaults to UTC timestamp.")
    p_select.add_argument("--markets", nargs="*", choices=["spread", "total", "ml"], default=None)

    p_edge = sub.add_parser("edge-analyze", help="Analyze edge buckets/segments on rolling test folds")
    p_edge.add_argument("--run-id", default=None, help="Run identifier. Defaults to UTC timestamp.")
    p_edge.add_argument("--market", required=True, choices=["spread", "total", "ml"])
    p_edge.add_argument("--model", required=True, choices=DEFAULT_MODEL_NAMES + ["ensemble"])
    p_edge.add_argument("--min-n", type=int, default=50, help="Minimum sample threshold for stable segments/buckets.")
    p_edge.add_argument("--limit", type=int, default=None, help="Optional row limit before fold construction.")

    p_ens = sub.add_parser("ensemble", help="Optimize and evaluate weighted ensemble on rolling folds")
    p_ens.add_argument("--markets", nargs="*", choices=["spread", "total", "ml"], default=None)
    p_ens.add_argument("--max-weight", type=float, default=0.5)

    sub.add_parser("exec-summary", help="Write EXEC_SUMMARY.md from current run artifacts")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = ModelLabConfig()
    if args.min_train_games is not None:
        config.min_train_games = int(args.min_train_games)
    if args.min_test_games is not None:
        config.min_test_games = int(args.min_test_games)
    if args.date_test_window_days is not None:
        config.date_test_window_days = int(args.date_test_window_days)
    if args.date_step_days is not None:
        config.date_step_days = int(args.date_step_days)

    if args.command == "build-frames":
        return cmd_build_frames(args, config)
    if args.command == "score-models":
        return cmd_score_models(args, config)
    if args.command == "feature-signal":
        return cmd_feature_signal(args, config)
    if args.command == "window-grid":
        return cmd_window_grid(args, config)
    if args.command == "select-features":
        return cmd_select_features(args, config)
    if args.command == "edge-analyze":
        return cmd_edge_analyze(args, config)
    if args.command == "ensemble":
        return cmd_ensemble(args, config)
    if args.command == "exec-summary":
        return cmd_exec_summary(args, config)

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
