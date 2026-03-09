from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.model_version import compute_model_version
from .artifacts import resolve_run_dir, write_json
from .backtest_outputs import BacktestOutputError, write_backtest_outputs
from .context import RunContext
from .evaluation import write_evaluation_outputs
from .integrity import run_integrity_gate, write_integrity_report
from .update_policy import evaluate_update_eligibility

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RUN_HISTORY = DATA_DIR / "run_evaluations.csv"
LAST_UPDATE = DATA_DIR / "last_update.json"
DECISIONS = ROOT / "DECISIONS_NEEDED.md"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _append_run_history(row: dict) -> None:
    df = pd.DataFrame([row])
    if RUN_HISTORY.exists():
        existing = pd.read_csv(RUN_HISTORY)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(RUN_HISTORY, index=False)


def cmd_audit(_: argparse.Namespace) -> int:
    report = {
        "entrypoints": {
            "predict": ["espn_prediction_runner.py", "cbb_ensemble.py"],
            "derived_csv": ["build_derived_csvs.py", "build_backtest_csvs.py"],
            "backtest": ["cbb_backtester.py", "build_backtest_csvs.py", "cbb_results_tracker.py"],
            "evaluation": ["evaluation/model_accuracy_weekly.py", "evaluation/predictions_graded.py"],
            "recommendations": ["enrichment/predictions_with_context.py", "data/csv/bet_recs.csv"],
            "training": ["optimize_weights.py", "models/weight_optimizer.py"],
        },
        "identity_keys": ["game_id", "event_id", "home_team_id", "away_team_id", "game_datetime_utc"],
        "implicit_latest_dependencies": [
            "data/predictions_latest.csv",
            "data/predictions_combined_latest.csv",
            "data/ensemble_predictions_latest.csv",
        ],
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "current_state_map.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not DECISIONS.exists():
        DECISIONS.write_text(
            "# Decisions Needed\n\n"
            "1. Canonical identity key: `game_id` vs `event_id` appears mixed across scripts.\n"
            "2. CLV source-of-truth: whether to use close in `results_log_graded.csv` or raw `market_lines.csv`.\n"
            "3. ROI denominator convention: per-bet unit risked vs stake-weighted, currently mixed.\n",
            encoding="utf-8",
        )
    print(f"[OK] Wrote {artifacts / 'current_state_map.json'}")
    return 0


def _run_subprocess(cmd: list[str]) -> int:
    p = subprocess.run(cmd, cwd=ROOT)
    return p.returncode


def cmd_run(args: argparse.Namespace) -> int:
    if args.date:
        as_of = pd.Timestamp(args.date)
        if as_of.tzinfo is None:
            as_of = as_of.tz_localize("UTC")
        else:
            as_of = as_of.tz_convert("UTC")
    else:
        as_of = pd.Timestamp(_now_utc())
    model_version_snapshot = compute_model_version(DATA_DIR)
    model_version = model_version_snapshot.get("model_version_hash", "unknown")
    ctx = RunContext.build(as_of=as_of.to_pydatetime(), model_version=model_version, feature_version="v1")
    run_dir = resolve_run_dir(ctx.run_id)

    integrity = run_integrity_gate(data_dir=DATA_DIR, as_of=as_of, mode=args.mode)
    backtest_meta = None
    integrity_path = run_dir / "manifest" / "integrity.json"
    write_integrity_report(integrity_path, integrity, ctx.to_dict())
    if not integrity.ok:
        print(f"[FAIL] Integrity gate failed. See {integrity_path}")
        return 2

    if args.mode == "predict":
        cmd = [sys.executable, "espn_prediction_runner.py"]
        if args.date:
            cmd.extend(["--date", args.date.replace("-", "")])
        rc = _run_subprocess(cmd)
    else:
        cmd = [sys.executable, "build_backtest_csvs.py"]
        if args.start:
            cmd.extend(["--since", args.start])
        rc = _run_subprocess(cmd)
        if rc == 0 and (DATA_DIR / "results_log_graded.csv").exists():
            graded = pd.read_csv(DATA_DIR / "results_log_graded.csv", low_memory=False)
            try:
                backtest_meta = write_backtest_outputs(run_dir, graded)
            except BacktestOutputError as exc:
                DECISIONS.write_text(
                    "# Decisions Needed\n\n"
                    "Backtest output generation was halted due to ambiguity:\n\n"
                    f"- {exc}\n",
                    encoding="utf-8",
                )
                raise

            summary = write_evaluation_outputs(DATA_DIR)
            (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
            for eval_name in ["evaluation.csv", "evaluation.json", "rolling_metrics.csv"]:
                src = DATA_DIR / eval_name
                if src.exists():
                    (run_dir / "evaluation" / eval_name).write_bytes(src.read_bytes())

            pred_src = DATA_DIR / "predictions_combined_latest.csv"
            if pred_src.exists():
                (run_dir / "predictions" / "predictions.csv").write_bytes(pred_src.read_bytes())

            run_tag = "UNKNOWN"
            if RUN_HISTORY.exists() and not pd.read_csv(RUN_HISTORY).empty:
                baseline = pd.read_csv(RUN_HISTORY).iloc[-1].to_dict()
            else:
                baseline = {
                    "roi_ats": summary.get("roi_ats") or 0.0,
                    "brier_score": summary.get("brier_score") or 0.0,
                    "spread_mae": summary.get("spread_mae") or 0.0,
                    "total_mae": summary.get("total_mae") or 0.0,
                }
            from .update_policy import tag_run

            run_tag, signals = tag_run(summary, baseline)
            _append_run_history(
                {
                    "run_id": ctx.run_id,
                    "as_of": ctx.as_of,
                    "model_version": ctx.model_version,
                    "feature_version": ctx.feature_version,
                    "run_tag": run_tag,
                    "signals": json.dumps(signals),
                    "graded_games": int(pd.read_csv(DATA_DIR / "results_log_graded.csv").shape[0]),
                    **summary,
                }
            )

    manifest = {
        "run_context": ctx.to_dict(),
        "mode": args.mode,
        "command": cmd,
        "return_code": rc,
        "artifact_run_dir": str(run_dir),
        "model_version": model_version_snapshot,
        "backtest_outputs": backtest_meta if args.mode == "backtest" and rc == 0 else None,
    }
    write_json(manifest, run_dir / "manifest" / "run_manifest.json")
    return rc


def cmd_update_check(args: argparse.Namespace) -> int:
    status = evaluate_update_eligibility(
        run_history_path=RUN_HISTORY,
        last_update_path=LAST_UPDATE,
        force_update=args.force_update,
        override_reason=args.override_reason,
    )
    print(json.dumps(status, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m pipeline")
    sub = parser.add_subparsers(dest="cmd", required=True)

    audit = sub.add_parser("audit")
    audit.set_defaults(func=cmd_audit)

    run = sub.add_parser("run")
    run.add_argument("--mode", choices=["predict", "backtest"], required=True)
    run.add_argument("--date", help="as_of date YYYY-MM-DD")
    run.add_argument("--start", help="backtest start date")
    run.add_argument("--end", help="backtest end date")
    run.set_defaults(func=cmd_run)

    upd = sub.add_parser("update-check")
    upd.add_argument("--force-update", action="store_true")
    upd.add_argument("--override-reason")
    upd.set_defaults(func=cmd_update_check)
    return parser


def cli_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
