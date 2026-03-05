from __future__ import annotations

import argparse
import json
import shutil
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from model_lab.config import DEFAULT_MODEL_NAMES, ModelLabConfig, dataframe_nan_report
from model_lab.data_builder import FrameBuildResult, build_frames, write_frames
from model_lab.edge_analyzer import run_edge_analysis
from model_lab.model_wrappers import load_predictions
from pipeline.advanced_metrics.compute_runner import compute_features


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SMOKE_RUNS_DIR = DATA_DIR / "smoke_runs"
FEATURE_MANIFEST = DATA_DIR / "feature_runs" / "feature_run_manifest.json"


class SmokeFailure(RuntimeError):
    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _run_id() -> str:
    return _utc_now().strftime("%Y%m%dT%H%M%SZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (pd.Int64Dtype, pd.Float64Dtype)):
        return str(value)
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _safe_copy(src: Path, dst: Path, missing_inputs: list[str]) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    else:
        missing_inputs.append(str(src))


def _trim_frame(df: pd.DataFrame, limit: int) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "game_datetime_utc" in out.columns:
        out["game_datetime_utc"] = pd.to_datetime(out["game_datetime_utc"], utc=True, errors="coerce")
        out = out.sort_values(["game_datetime_utc", "game_id"], na_position="last")
    elif "game_id" in out.columns:
        out = out.sort_values(["game_id"])
    return out.tail(limit).reset_index(drop=True)


def _feature_stage(
    *,
    run_dir: Path,
    limit: int,
    rebuild_features: bool,
    missing_inputs: list[str],
    blocked_modules: list[dict[str, str]],
) -> dict[str, Any]:
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    team_path = DATA_DIR / "team_game_metrics.csv"
    matchup_path = DATA_DIR / "matchup_metrics.csv"
    team_df: pd.DataFrame | None = None
    matchup_df: pd.DataFrame | None = None
    mode = "built"

    existing_ready = (
        not rebuild_features
        and team_path.exists()
        and matchup_path.exists()
        and team_path.stat().st_size > 0
        and matchup_path.stat().st_size > 0
    )
    if existing_ready:
        mode = "loaded_existing"
        team_df = pd.read_csv(team_path, low_memory=False)
        matchup_df = pd.read_csv(matchup_path, low_memory=False)
    else:
        try:
            team_df, matchup_df = compute_features(limit_games=limit, rebuild=rebuild_features)
        except Exception as exc:
            mode = "loaded_existing"
            blocked_modules.append(
                {
                    "module": "pipeline.advanced_metrics.compute_runner",
                    "reason": f"compute_features_failed: {exc}",
                }
            )
            if team_path.exists():
                team_df = pd.read_csv(team_path, low_memory=False)
            if matchup_path.exists():
                matchup_df = pd.read_csv(matchup_path, low_memory=False)

    if team_df is None or matchup_df is None:
        missing = []
        if team_df is None:
            missing.append(str(team_path))
        if matchup_df is None:
            missing.append(str(matchup_path))
        raise SmokeFailure(
            "Feature stage failed: could not build features and fallback files were missing. "
            f"Missing inputs: {', '.join(missing)}",
            exit_code=2,
        )

    _safe_copy(team_path, feature_dir / "team_game_metrics.csv", missing_inputs)
    _safe_copy(matchup_path, feature_dir / "matchup_metrics.csv", missing_inputs)
    _safe_copy(FEATURE_MANIFEST, feature_dir / "feature_run_manifest.json", missing_inputs)

    return {
        "status": "ok",
        "module": "pipeline.advanced_metrics.compute_runner.compute_features",
        "mode": mode,
        "artifacts": [
            str(feature_dir / "team_game_metrics.csv"),
            str(feature_dir / "matchup_metrics.csv"),
            str(feature_dir / "feature_run_manifest.json"),
        ],
        "row_counts": {
            "team_game_metrics": int(len(team_df)),
            "matchup_metrics": int(len(matchup_df)),
        },
        "nan_report": {
            "team_game_metrics": dataframe_nan_report(team_df),
            "matchup_metrics": dataframe_nan_report(matchup_df),
        },
    }


def _model_lab_stage(
    *,
    run_dir: Path,
    limit: int,
    blocked_modules: list[dict[str, str]],
) -> tuple[dict[str, Any], ModelLabConfig]:
    model_lab_dir = run_dir / "model_lab"
    model_lab_dir.mkdir(parents=True, exist_ok=True)

    config = ModelLabConfig(repo_root=ROOT)
    frames = build_frames(config)
    trimmed = FrameBuildResult(
        spread_frame=_trim_frame(frames.spread_frame, limit),
        total_frame=_trim_frame(frames.total_frame, limit),
        ml_frame=_trim_frame(frames.ml_frame, limit),
        blocked_reasons=list(frames.blocked_reasons),
        source_paths=dict(frames.source_paths),
        frame_row_counts={},
        nan_report={},
    )
    trimmed.frame_row_counts = {
        "spread_frame": int(len(trimmed.spread_frame)),
        "total_frame": int(len(trimmed.total_frame)),
        "ml_frame": int(len(trimmed.ml_frame)),
    }
    trimmed.nan_report = {
        "spread_frame": dataframe_nan_report(trimmed.spread_frame),
        "total_frame": dataframe_nan_report(trimmed.total_frame),
        "ml_frame": dataframe_nan_report(trimmed.ml_frame),
    }

    frame_artifacts = write_frames(trimmed, model_lab_dir)
    if trimmed.blocked_reasons:
        for reason in sorted(set(trimmed.blocked_reasons)):
            blocked_modules.append({"module": "model_lab.data_builder.build_frames", "reason": reason})

    if all(v == 0 for v in trimmed.frame_row_counts.values()):
        raise SmokeFailure(
            "Model Lab frame build produced zero rows for spread/total/ml. "
            f"Blocked reasons: {', '.join(sorted(set(trimmed.blocked_reasons))) or 'none'}",
            exit_code=2,
        )

    return (
        {
            "status": "ok",
            "module": "model_lab.data_builder.build_frames",
            "artifacts": list(frame_artifacts.values()),
            "row_counts": trimmed.frame_row_counts,
            "blocked_reasons": sorted(set(trimmed.blocked_reasons)),
            "nan_report": trimmed.nan_report,
            "source_paths": trimmed.source_paths,
        },
        config,
    )


def _build_baseline_predictions(edge_dir: Path, limit: int) -> Path:
    base_sources = [
        edge_dir / "spread_frame.csv",
        edge_dir / "total_frame.csv",
        edge_dir / "ml_frame.csv",
    ]
    base_df = pd.DataFrame()
    for src in base_sources:
        if src.exists():
            df = pd.read_csv(src, low_memory=False)
            if not df.empty:
                base_df = df
                break
    if base_df.empty:
        raise SmokeFailure(
            "Unable to create baseline predictions: all edge input frames are empty.",
            exit_code=2,
        )

    out = pd.DataFrame()
    out["game_id"] = base_df.get("game_id")
    out["event_id"] = base_df.get("event_id", base_df.get("game_id"))
    out["game_datetime_utc"] = base_df.get("game_datetime_utc")
    out["game_date"] = base_df.get("game_date")
    out["home_team_id"] = base_df.get("home_team_id")
    out["away_team_id"] = base_df.get("away_team_id")
    out["neutral_site"] = base_df.get("neutral_site", 0)
    out["actual_margin"] = pd.to_numeric(base_df.get("actual_margin"), errors="coerce")
    out["actual_total"] = pd.to_numeric(base_df.get("actual_total"), errors="coerce")
    out["home_won"] = pd.to_numeric(base_df.get("home_won"), errors="coerce")
    out["spread_line"] = pd.to_numeric(base_df.get("spread_line"), errors="coerce")
    out["total_line"] = pd.to_numeric(base_df.get("total_line"), errors="coerce")
    out["home_ml"] = pd.to_numeric(base_df.get("home_ml"), errors="coerce")
    out["away_ml"] = pd.to_numeric(base_df.get("away_ml"), errors="coerce")
    out["fourfactors_spread"] = out["spread_line"].fillna(0.0)
    out["fourfactors_total"] = out["total_line"].fillna(out["actual_total"]).fillna(0.0)
    out["fourfactors_conf"] = 0.5
    out = out.tail(limit).reset_index(drop=True)

    baseline_path = edge_dir / "baseline_backtest_results_latest.csv"
    out.to_csv(baseline_path, index=False)
    return baseline_path


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(name)).strip("_") or "model"


def _write_edge_fallback_outputs(edge_dir: Path, market: str, model_name: str, reason: str) -> list[str]:
    output_dir = edge_dir / "edge_analyzer" / market / _safe_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    edge_bucket_path = output_dir / "edge_bucket_report.csv"
    segment_path = output_dir / "segment_report.csv"
    worst_path = output_dir / "worst_misses.csv"
    summary_path = output_dir / "EDGE_EXEC_SUMMARY.md"
    manifest_path = edge_dir / "edge_analyzer" / "run_manifest.json"

    pd.DataFrame(
        [
            {
                "run_id": edge_dir.parent.name,
                "market": market,
                "model_name": model_name,
                "bucket_label": "BLOCKED",
                "n": 0,
                "graded_n": 0,
                "hit_rate": None,
                "roi": None,
                "avg_edge": None,
                "avg_abs_error": None,
                "clv_mean": None,
                "small_sample_flag": True,
                "status": "BLOCKED",
                "blocked_reason": reason,
                "missing_columns": "",
            }
        ]
    ).to_csv(edge_bucket_path, index=False)
    pd.DataFrame(
        [
            {
                "run_id": edge_dir.parent.name,
                "market": market,
                "model_name": model_name,
                "segment_family": "all",
                "segment_value": "all",
                "n": 0,
                "graded_n": 0,
                "hit_rate": None,
                "roi": None,
                "avg_edge": None,
                "avg_abs_error": None,
                "clv_mean": None,
                "small_sample_flag": True,
                "status": "BLOCKED",
                "blocked_reason": reason,
                "missing_columns": "",
            }
        ]
    ).to_csv(segment_path, index=False)
    pd.DataFrame(columns=["run_id", "market", "model_name"]).to_csv(worst_path, index=False)
    summary_path.write_text(
        "# Edge Analyzer Executive Summary\n\n"
        f"- BLOCKED: `{market}` market edge analysis failed.\n"
        f"- Reason: `{reason}`\n",
        encoding="utf-8",
    )
    _write_json(
        manifest_path,
        {
            "run_id": edge_dir.parent.name,
            "market": market,
            "model_name": model_name,
            "rows_analyzed": 0,
            "blocked_reasons": [reason],
            "nan_report": {},
            "artifacts": {
                "edge_bucket_report": str(edge_bucket_path),
                "segment_report": str(segment_path),
                "worst_misses": str(worst_path),
                "edge_exec_summary": str(summary_path),
            },
        },
    )
    return [str(edge_bucket_path), str(segment_path), str(worst_path), str(summary_path), str(manifest_path)]


def _edge_stage(
    *,
    run_dir: Path,
    limit: int,
    min_n: int,
    market: str,
    config: ModelLabConfig,
    missing_inputs: list[str],
    blocked_modules: list[dict[str, str]],
) -> dict[str, Any]:
    edge_dir = run_dir / "edge"
    edge_dir.mkdir(parents=True, exist_ok=True)

    for name in ("spread_frame.csv", "total_frame.csv", "ml_frame.csv"):
        _safe_copy(run_dir / "model_lab" / name, edge_dir / name, missing_inputs)

    selected_model: str | None = None
    for model_name in DEFAULT_MODEL_NAMES:
        pred_df = load_predictions(config, model_name)
        if not pred_df.empty:
            selected_model = model_name
            break

    edge_config = config
    baseline_path: Path | None = None
    if selected_model is None:
        baseline_path = _build_baseline_predictions(edge_dir, limit)
        selected_model = "FourFactors"
        blocked_modules.append(
            {
                "module": "model_lab.model_wrappers.load_predictions",
                "reason": "No existing model prediction source found; baseline backtest predictions were generated.",
            }
        )
        edge_config = ModelLabConfig(
            repo_root=ROOT,
            backtest_results_path=baseline_path,
            matchup_metrics_path=config.matchup_metrics_path,
            team_game_metrics_path=config.team_game_metrics_path,
            results_graded_path=config.results_graded_path,
            results_log_path=config.results_log_path,
            backtest_training_path=config.backtest_training_path,
            market_closing_path=config.market_closing_path,
            market_lines_path=config.market_lines_path,
            games_path=config.games_path,
            predictions_combined_path=config.predictions_combined_path,
            ensemble_latest_path=config.ensemble_latest_path,
        )

    markets = ["spread", "total", "ml"] if market == "all" else [market]
    results: dict[str, Any] = {"artifacts": [], "blocked_reasons": {}, "row_counts": {}, "nan_report": {}}

    for market_name in markets:
        frame_path = edge_dir / f"{market_name}_frame.csv"
        if not frame_path.exists():
            blocked_modules.append(
                {
                    "module": "model_lab.edge_analyzer.run_edge_analysis",
                    "reason": f"missing_frame_for_market:{market_name}",
                }
            )
            results["blocked_reasons"][market_name] = [f"missing_frame:{frame_path}"]
            continue

        frame_df = pd.read_csv(frame_path, low_memory=False)
        if frame_df.empty:
            blocked_modules.append(
                {
                    "module": "model_lab.edge_analyzer.run_edge_analysis",
                    "reason": f"empty_frame_for_market:{market_name}",
                }
            )
            results["blocked_reasons"][market_name] = [f"empty_frame:{frame_path}"]
            continue

        try:
            edge_result = run_edge_analysis(
                run_dir=edge_dir,
                config=edge_config,
                market=market_name,
                model_name=selected_model,
                min_n=min_n,
                limit=limit,
            )
        except Exception as exc:
            reason = f"exception:{exc}"
            blocked_modules.append(
                {
                    "module": "model_lab.edge_analyzer.run_edge_analysis",
                    "reason": f"{market_name}:{reason}",
                }
            )
            results["blocked_reasons"][market_name] = [reason]
            results["artifacts"].extend(_write_edge_fallback_outputs(edge_dir, market_name, selected_model, reason))
            results["row_counts"][market_name] = 0
            results["nan_report"][market_name] = {}
            continue
        results["artifacts"].extend(
            [
                str(edge_result.edge_bucket_report_path),
                str(edge_result.segment_report_path),
                str(edge_result.worst_misses_path),
                str(edge_result.exec_summary_path),
                str(edge_result.run_manifest_path),
            ]
        )
        results["blocked_reasons"][market_name] = list(edge_result.blocked_reasons)
        results["row_counts"][market_name] = int(edge_result.rows_analyzed)

        if edge_result.run_manifest_path.exists():
            manifest = json.loads(edge_result.run_manifest_path.read_text(encoding="utf-8"))
            results["nan_report"][market_name] = manifest.get("nan_report", {})

        for reason in edge_result.blocked_reasons:
            blocked_modules.append(
                {
                    "module": "model_lab.edge_analyzer.run_edge_analysis",
                    "reason": f"{market_name}:{reason}",
                }
            )

    if not results["artifacts"]:
        raise SmokeFailure(
            "Edge analyzer did not run for any market. Check model_lab frames and prediction sources.",
            exit_code=2,
        )

    results["status"] = "ok"
    results["module"] = "model_lab.edge_analyzer.run_edge_analysis"
    results["model_name"] = selected_model
    if baseline_path is not None:
        results["baseline_predictions"] = str(baseline_path)
    return results


def run_smoke(limit: int, market: str, min_n: int, rebuild_features: bool) -> tuple[int, Path]:
    started = _utc_now()
    run_id = _run_id()
    run_dir = SMOKE_RUNS_DIR / run_id
    manifest_path = run_dir / "manifest.json"
    missing_inputs: list[str] = []
    blocked_modules: list[dict[str, str]] = []

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": None,
        "status": "failed",
        "what_ran": {},
        "artifacts_produced": [],
        "row_counts": {},
        "missing_inputs": [],
        "blocked_modules": [],
        "nan_report": {},
        "failure_message": None,
    }

    try:
        feature_info = _feature_stage(
            run_dir=run_dir,
            limit=limit,
            rebuild_features=rebuild_features,
            missing_inputs=missing_inputs,
            blocked_modules=blocked_modules,
        )
        manifest["what_ran"]["features"] = feature_info
        manifest["artifacts_produced"].extend(feature_info["artifacts"])
        manifest["row_counts"].update(feature_info["row_counts"])
        manifest["nan_report"]["features"] = feature_info["nan_report"]

        model_lab_info, config = _model_lab_stage(
            run_dir=run_dir,
            limit=limit,
            blocked_modules=blocked_modules,
        )
        manifest["what_ran"]["model_lab"] = model_lab_info
        manifest["artifacts_produced"].extend(model_lab_info["artifacts"])
        manifest["row_counts"].update(model_lab_info["row_counts"])
        manifest["nan_report"]["model_lab"] = model_lab_info["nan_report"]

        edge_info = _edge_stage(
            run_dir=run_dir,
            limit=limit,
            min_n=min_n,
            market=market,
            config=config,
            missing_inputs=missing_inputs,
            blocked_modules=blocked_modules,
        )
        manifest["what_ran"]["edge"] = edge_info
        manifest["artifacts_produced"].extend(edge_info["artifacts"])
        manifest["row_counts"]["edge_rows"] = edge_info["row_counts"]
        manifest["nan_report"]["edge"] = edge_info["nan_report"]

        manifest["status"] = "ok"
    except SmokeFailure as exc:
        manifest["failure_message"] = str(exc)
        rc = exc.exit_code
    except Exception as exc:  # pragma: no cover - defensive fallback
        manifest["failure_message"] = f"Unhandled smoke error: {exc}"
        blocked_modules.append({"module": "pipeline.smoke", "reason": traceback.format_exc()})
        rc = 1
    else:
        rc = 0
    finally:
        finished = _utc_now()
        manifest["finished_at_utc"] = finished.isoformat()
        manifest["missing_inputs"] = sorted(set(missing_inputs))
        manifest["blocked_modules"] = sorted(
            [asdict_row for asdict_row in blocked_modules],
            key=lambda x: (x.get("module", ""), x.get("reason", "")),
        )
        _write_json(manifest_path, manifest)

    return rc, manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fast end-to-end smoke pipeline.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum rows/games to process.")
    parser.add_argument(
        "--market",
        choices=["spread", "total", "ml", "all"],
        default="all",
        help="Markets to run through edge analyzer.",
    )
    parser.add_argument("--min-n", type=int, default=25, help="Minimum sample size threshold for edge buckets.")
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Force feature recompute instead of cache reuse.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rc, manifest_path = run_smoke(
        limit=max(1, int(args.limit)),
        market=args.market,
        min_n=max(1, int(args.min_n)),
        rebuild_features=bool(args.rebuild_features),
    )
    if rc == 0:
        print(f"[OK] Smoke run completed: {manifest_path}")
    else:
        print(f"[FAIL] Smoke run failed ({rc}). Manifest: {manifest_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
