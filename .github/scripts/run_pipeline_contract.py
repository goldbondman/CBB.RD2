#!/usr/bin/env python3
"""Execute dependency-aware workflow stages from JSON contracts."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DATE_COLUMNS = [
    "game_datetime_utc",
    "captured_at_utc",
    "pulled_at_utc",
    "updated_at",
    "game_date",
]


@dataclass
class StageResult:
    name: str
    command: str
    started_at_utc: str
    finished_at_utc: str
    outputs: list[dict[str, Any]]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _required_cols(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(v).strip() for v in raw if str(v).strip()]
    if isinstance(raw, str):
        return [v.strip() for v in raw.split(",") if v.strip()]
    return []


def _csv_summary(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, dtype=str, low_memory=False)
    out: dict[str, Any] = {
        "path": str(path),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
    }

    for col in DATE_COLUMNS:
        if col in df.columns:
            ts = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()
            if not ts.empty:
                out["min_date_utc"] = ts.min().isoformat()
                out["max_date_utc"] = ts.max().isoformat()
            break

    key_rates: dict[str, float] = {}
    for col in ["game_id", "event_id", "team_id", "game_datetime_utc"]:
        if col in df.columns:
            key_rates[col] = round(float(df[col].isna().mean()) * 100.0, 2)
    if key_rates:
        out["null_rates_pct"] = key_rates
    return out


def _validate_one_spec(repo_root: Path, spec: Any, stage_name: str, kind: str) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(spec, str):
        spec = {"path": spec}
    if not isinstance(spec, dict):
        return None, f"[{stage_name}] invalid {kind} spec: {spec!r}"

    rel = str(spec.get("path", "")).strip()
    if not rel:
        return None, f"[{stage_name}] invalid {kind} spec missing path"

    path = repo_root / rel
    if not path.exists():
        return None, f"[{stage_name}] missing required {kind}: {rel}"
    if path.is_file() and path.stat().st_size == 0:
        return None, f"[{stage_name}] empty {kind}: {rel}"

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, dtype=str, low_memory=False)
        min_rows = int(spec.get("min_rows", 1))
        if len(df) < min_rows:
            return None, f"[{stage_name}] {kind} {rel} has {len(df)} rows; expected >= {min_rows}"
        missing = [c for c in _required_cols(spec.get("required_columns")) if c not in df.columns]
        if missing:
            return None, f"[{stage_name}] {kind} {rel} missing required columns: {missing}"

    max_age_hours = spec.get("max_age_hours")
    if max_age_hours is not None:
        age_h = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0
        if age_h > float(max_age_hours):
            return None, f"[{stage_name}] {kind} {rel} stale: age_hours={age_h:.2f}, max_age_hours={max_age_hours}"

    return {"path": rel, "size_bytes": path.stat().st_size}, None


def _validate_specs(repo_root: Path, specs: list[Any], stage_name: str, kind: str) -> list[dict[str, Any]]:
    errors: list[str] = []
    ok_items: list[dict[str, Any]] = []
    for spec in specs:
        info, err = _validate_one_spec(repo_root, spec, stage_name, kind)
        if err:
            errors.append(err)
        elif info:
            ok_items.append(info)
    if errors:
        joined = "\n".join(errors)
        raise RuntimeError(f"[{stage_name}] {kind} validation failed:\n{joined}")
    return ok_items


def _toposort(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {str(stage["name"]): stage for stage in stages}
    pending = set(by_name.keys())
    ordered: list[dict[str, Any]] = []

    while pending:
        progressed = False
        for name in sorted(list(pending)):
            deps = [str(d) for d in by_name[name].get("depends_on", [])]
            unknown = [d for d in deps if d not in by_name]
            if unknown:
                raise RuntimeError(f"[{name}] unknown dependency references: {unknown}")
            if all(dep not in pending for dep in deps):
                ordered.append(by_name[name])
                pending.remove(name)
                progressed = True
        if not progressed:
            raise RuntimeError(f"cyclic dependencies detected for stages: {sorted(pending)}")

    return ordered


def _run_stage(repo_root: Path, stage: dict[str, Any], context: dict[str, str], dry_run: bool) -> StageResult:
    name = str(stage["name"])
    command_tpl = str(stage.get("command", "")).strip()
    if not command_tpl:
        raise RuntimeError(f"[{name}] missing command")
    command = command_tpl.format(**context)

    _validate_specs(repo_root, list(stage.get("requires", [])), name, "input")

    started = _utc_now()
    if dry_run:
        print(f"[DRY-RUN] {name}: {command}")
    else:
        proc = subprocess.run(shlex.split(command), cwd=repo_root)
        if proc.returncode != 0:
            raise RuntimeError(f"[{name}] command failed with exit code {proc.returncode}: {command}")

    outputs: list[dict[str, Any]] = []
    output_infos = _validate_specs(repo_root, list(stage.get("produces", [])), name, "output")
    for info in output_infos:
        out_path = repo_root / info["path"]
        record = info.copy()
        if out_path.suffix.lower() == ".csv":
            record.update(_csv_summary(out_path))
        outputs.append(record)

    finished = _utc_now()
    return StageResult(
        name=name,
        command=command,
        started_at_utc=started,
        finished_at_utc=finished,
        outputs=outputs,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--job", choices=["update", "rankings"], required=True)
    parser.add_argument("--days-back", type=str, default="3")
    parser.add_argument("--game-type", type=str, default="regular")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    contract_path = args.contract if args.contract.is_absolute() else repo_root / args.contract
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest

    contract = _load_json(contract_path)
    stages = [s for s in contract.get("stages", []) if str(s.get("job", "update")) == args.job]
    if not stages:
        raise RuntimeError(f"No contract stages found for job={args.job}")

    ordered = _toposort(stages)
    context = {
        "days_back": str(args.days_back),
        "game_type": str(args.game_type),
    }

    results: list[StageResult] = []
    for stage in ordered:
        print(f"[STAGE] {stage['name']}")
        result = _run_stage(repo_root, stage, context, args.dry_run)
        results.append(result)
        print(f"[OK] {result.name}")

    manifest = {
        "contract_name": contract.get("name", contract_path.name),
        "contract_version": contract.get("version"),
        "job": args.job,
        "generated_at_utc": _utc_now(),
        "dry_run": bool(args.dry_run),
        "context": context,
        "stages": [
            {
                "name": r.name,
                "command": r.command,
                "started_at_utc": r.started_at_utc,
                "finished_at_utc": r.finished_at_utc,
                "outputs": r.outputs,
            }
            for r in results
        ],
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
