#!/usr/bin/env python3
"""Write a deterministic run manifest for perplexity model workflow outputs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_bool(raw: str) -> bool | None:
    value = (raw or "").strip().lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def _csv_summary(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "exists": True,
        "absolute_path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "rows": None,
        "cols": None,
        "read_error": None,
    }
    if path.stat().st_size == 0:
        out["rows"] = 0
        out["cols"] = 0
        return out

    try:
        df = pd.read_csv(path, dtype=str, low_memory=False)
        out["rows"] = int(len(df))
        out["cols"] = int(len(df.columns))
    except Exception as exc:  # pragma: no cover
        out["read_error"] = str(exc)
    return out


def _file_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "absolute_path": str(path.resolve()),
            "size_bytes": 0,
            "rows": None,
            "cols": None,
            "read_error": None,
        }
    if path.suffix.lower() == ".csv":
        return _csv_summary(path)
    return {
        "exists": True,
        "absolute_path": str(path.resolve()),
        "size_bytes": int(path.stat().st_size),
        "rows": None,
        "cols": None,
        "read_error": None,
    }


def _normalize_exit_code(raw: str) -> int | None:
    value = (raw or "").strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _contract_status(preflight_allow: bool | None, exit_code: int | None) -> str:
    if preflight_allow is False:
        return "SKIPPED_PRECHECK"
    if exit_code is None:
        return "NOT_RUN"
    if exit_code == 0:
        return "SUCCESS"
    return "FAILED"


def _canonical_key(df: pd.DataFrame) -> pd.Series:
    if "event_id" in df.columns:
        event = df["event_id"].astype(str).str.strip()
    else:
        event = pd.Series("", index=df.index)
    if "game_id" in df.columns:
        game = df["game_id"].astype(str).str.strip()
    else:
        game = pd.Series("", index=df.index)
    key = event.where(event != "", game)
    key = key.str.replace(r"\.0$", "", regex=True).str.strip()
    return key


def _safe_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _eligible_slate(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "completed" in out.columns:
        completed = out["completed"].astype(str).str.lower()
        state = out.get("state", pd.Series("", index=out.index)).astype(str).str.lower()
        active = out[(completed != "true") | (state != "post")].copy()
        if not active.empty:
            out = active
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-manifest", type=Path, required=True)
    parser.add_argument("--contract-log", type=Path, required=True)
    parser.add_argument("--days-back", type=str, required=True)
    parser.add_argument("--run-mode", type=str, required=True)
    parser.add_argument("--preflight-allow", type=str, default="")
    parser.add_argument("--preflight-reason", type=str, default="")
    parser.add_argument("--contract-exit-code", type=str, default="")
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    preflight_allow = _parse_bool(args.preflight_allow)
    contract_exit_code = _normalize_exit_code(args.contract_exit_code)

    generated_files: dict[str, Any] = {}
    for file_str in args.files:
        rel = str(Path(file_str)).replace("\\", "/")
        generated_files[rel] = _file_summary(Path(file_str))

    games_df = _safe_csv(Path("data/games.csv"))
    preds_df = _safe_csv(Path("data/predictions_joint_latest.csv"))
    d1_report_df = _safe_csv(Path("debug/d1_filter_report.csv"))

    eligible_games = int(_canonical_key(_eligible_slate(games_df)).nunique()) if not games_df.empty else 0
    predicted_games = 0
    if not preds_df.empty:
        pred_key = _canonical_key(preds_df)
        margin = pd.to_numeric(preds_df.get("pred_margin"), errors="coerce")
        predicted_games = int(pred_key[margin.notna()].nunique())
    missing_predictions_count = max(0, eligible_games - predicted_games)

    d1_filtered_out_count = 0
    if not d1_report_df.empty and "filter_reason" in d1_report_df.columns:
        d1_filtered_out_count = int((d1_report_df["filter_reason"].astype(str) != "KEEP_D1_VS_D1").sum())

    contract_stage_status: dict[str, Any] = {}
    produced_files: dict[str, Any] = {}
    if args.contract_manifest.exists() and args.contract_manifest.stat().st_size > 0:
        try:
            contract_manifest = json.loads(args.contract_manifest.read_text(encoding="utf-8"))
            for stage in contract_manifest.get("stages", []):
                stage_name = str(stage.get("name", "")).strip()
                if not stage_name:
                    continue
                contract_stage_status[stage_name] = {
                    "status": "SUCCESS",
                    "started_at_utc": stage.get("started_at_utc"),
                    "finished_at_utc": stage.get("finished_at_utc"),
                }
                for out in stage.get("outputs", []):
                    rel = str(out.get("path", "")).replace("\\", "/")
                    if rel:
                        produced_files[rel] = out
        except Exception as exc:  # pragma: no cover
            contract_stage_status["manifest_parse_error"] = {"status": "FAILED", "error": str(exc)}

    manifest: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "workflow": "cbb_perplexity_models",
        "context": {
            "days_back": args.days_back,
            "run_mode": args.run_mode,
        },
        "pipeline_stage_status": {
            "preflight": {
                "allow_run": preflight_allow,
                "reason": args.preflight_reason,
            },
            "contract_runner": {
                "status": _contract_status(preflight_allow, contract_exit_code),
                "exit_code": contract_exit_code,
                "contract_manifest_exists": args.contract_manifest.exists(),
                "contract_log_exists": args.contract_log.exists(),
            },
            "contract_stages": contract_stage_status,
        },
        "counts": {
            "eligible_games": eligible_games,
            "predicted_games": predicted_games,
            "missing_predictions_count": missing_predictions_count,
            "d1_filtered_out_count": d1_filtered_out_count,
        },
        "produced_files": produced_files,
        "generated_files": generated_files,
        "artifacts": {
            "contract_manifest": _file_summary(args.contract_manifest),
            "contract_runner_log": _file_summary(args.contract_log),
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote run manifest: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
