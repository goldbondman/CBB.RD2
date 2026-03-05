#!/usr/bin/env python3
"""Generate consistent CSV integrity reports from JSON/YAML manifests."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class CheckResult:
    path: str
    required: bool
    exists: bool
    rows: int | None
    columns: int | None
    min_rows: int
    missing_required_columns: list[str]
    null_rates_pct: dict[str, float]
    oldest_date_utc: str | None
    newest_date_utc: str | None
    status: str
    message: str


DEFAULT_DATE_CANDIDATES = [
    "game_datetime_utc",
    "captured_at_utc",
    "pulled_at_utc",
    "generated_at_utc",
    "updated_at",
    "game_date",
    "date",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)

    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("YAML manifest requires PyYAML to be installed") from exc
        return yaml.safe_load(text)

    raise RuntimeError(f"Unsupported manifest type: {path.suffix}")


def _date_range(df: pd.DataFrame, candidates: list[str]) -> tuple[str | None, str | None]:
    for col in candidates:
        if col in df.columns:
            ts = pd.to_datetime(df[col], errors="coerce", utc=True).dropna()
            if not ts.empty:
                return ts.min().isoformat(), ts.max().isoformat()
    return None, None


def _null_rates(df: pd.DataFrame, key_columns: list[str]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for col in key_columns:
        if col in df.columns:
            rates[col] = round(float(df[col].isna().mean()) * 100.0, 2)
    return rates


def _check_one(spec: dict[str, Any], repo_root: Path) -> CheckResult:
    rel_path = str(spec.get("path", "")).strip()
    if not rel_path:
        raise RuntimeError("Manifest file check is missing 'path'")

    required = bool(spec.get("required", True))
    min_rows = int(spec.get("min_rows", 1))
    required_columns = [str(c) for c in spec.get("required_columns", [])]
    key_columns = [str(c) for c in spec.get("key_columns", [])]
    date_columns = [str(c) for c in spec.get("date_columns", [])]
    date_candidates = date_columns or DEFAULT_DATE_CANDIDATES

    target = repo_root / rel_path
    if not target.exists():
        status = "BLOCKED" if required else "OPTIONAL_MISSING"
        msg = f"missing file: {rel_path}"
        return CheckResult(
            path=rel_path,
            required=required,
            exists=False,
            rows=None,
            columns=None,
            min_rows=min_rows,
            missing_required_columns=[],
            null_rates_pct={},
            oldest_date_utc=None,
            newest_date_utc=None,
            status=status,
            message=msg,
        )

    df = pd.read_csv(target, dtype=str, low_memory=False)
    rows = int(len(df))
    cols = int(len(df.columns))
    missing_cols = [c for c in required_columns if c not in df.columns]
    oldest, newest = _date_range(df, date_candidates)
    null_rates = _null_rates(df, key_columns)

    errors: list[str] = []
    if rows < min_rows:
        errors.append(f"rows={rows} < min_rows={min_rows}")
    if missing_cols:
        errors.append(f"missing_required_columns={missing_cols}")

    if errors:
        status = "BLOCKED" if required else "OPTIONAL_FAILED"
        message = "; ".join(errors)
    else:
        status = "OK"
        message = "passed"

    return CheckResult(
        path=rel_path,
        required=required,
        exists=True,
        rows=rows,
        columns=cols,
        min_rows=min_rows,
        missing_required_columns=missing_cols,
        null_rates_pct=null_rates,
        oldest_date_utc=oldest,
        newest_date_utc=newest,
        status=status,
        message=message,
    )


def _markdown_summary(
    workflow_name: str,
    run_id: str,
    generated_at_utc: str,
    results: list[CheckResult],
    blocked: list[CheckResult],
) -> str:
    lines = [
        f"# Integrity Summary: {workflow_name}",
        "",
        f"- run_id: `{run_id}`",
        f"- generated_at_utc: `{generated_at_utc}`",
        f"- total_checks: `{len(results)}`",
        f"- blocked_checks: `{len(blocked)}`",
        "",
        "## File Checks",
        "",
        "| file | required | status | rows | cols | missing_required_cols | oldest_date_utc | newest_date_utc |",
        "|---|---:|---|---:|---:|---|---|---|",
    ]

    for r in results:
        missing_cols = ", ".join(r.missing_required_columns) if r.missing_required_columns else ""
        rows = "" if r.rows is None else str(r.rows)
        cols = "" if r.columns is None else str(r.columns)
        lines.append(
            f"| `{r.path}` | `{r.required}` | `{r.status}` | `{rows}` | `{cols}` | `{missing_cols}` | "
            f"`{r.oldest_date_utc or ''}` | `{r.newest_date_utc or ''}` |"
        )

    if blocked:
        lines.extend(["", "## Blocked Items", ""])
        for r in blocked:
            lines.append(f"- `{r.path}`: {r.message}")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True, help="JSON/YAML contract manifest path")
    parser.add_argument("--workflow-name", type=str, default="", help="Override workflow name")
    parser.add_argument("--run-id", type=str, default="", help="Run id (defaults to UTC timestamp)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest
    manifest = _load_manifest(manifest_path)

    workflow_name = args.workflow_name.strip() or str(manifest.get("workflow", "")).strip()
    if not workflow_name:
        raise RuntimeError("Workflow name is required (manifest.workflow or --workflow-name)")
    run_id = args.run_id.strip() or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    file_specs = manifest.get("files", [])
    if not isinstance(file_specs, list) or not file_specs:
        raise RuntimeError("Manifest must include a non-empty 'files' list")

    results = [_check_one(spec, repo_root) for spec in file_specs]
    blocked = [r for r in results if r.status == "BLOCKED"]
    generated_at = _utc_now()

    output_dir = repo_root / "data" / "integrity_reports" / workflow_name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "workflow": workflow_name,
        "run_id": run_id,
        "generated_at_utc": generated_at,
        "checks": [
            {
                "path": r.path,
                "required": r.required,
                "exists": r.exists,
                "rows": r.rows,
                "columns": r.columns,
                "min_rows": r.min_rows,
                "missing_required_columns": r.missing_required_columns,
                "null_rates_pct": r.null_rates_pct,
                "oldest_date_utc": r.oldest_date_utc,
                "newest_date_utc": r.newest_date_utc,
                "status": r.status,
                "message": r.message,
            }
            for r in results
        ],
    }

    json_path = output_dir / "integrity_report.json"
    md_path = output_dir / "INTEGRITY_EXEC_SUMMARY.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_summary(workflow_name, run_id, generated_at, results, blocked), encoding="utf-8")

    print(f"[INTEGRITY] report_json={json_path}")
    print(f"[INTEGRITY] report_summary={md_path}")

    if blocked:
        for item in blocked:
            print(f"[BLOCKED] {item.path}: {item.message}")
        return 1

    print("[INTEGRITY] all required checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
