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
        },
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
