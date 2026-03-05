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


def _norm_rel_path(path: str) -> str:
    return str(Path(path)).replace("\\", "/").strip()


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

    rel = _norm_rel_path(str(spec.get("path", "")).strip())
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
        allow_empty_when_no_games = bool(spec.get("allow_empty_when_no_games", False))
        if kind == "output" and len(df) == 0:
            if not allow_empty_when_no_games:
                return None, (
                    f"[{stage_name}] {kind} {rel} is empty (0 rows). "
                    "This is blocked to avoid silent empty outputs."
                )
            games_path = repo_root / "data" / "games.csv"
            if not games_path.exists():
                return None, (
                    f"[{stage_name}] {kind} {rel} is empty and allow_empty_when_no_games=true, "
                    "but data/games.csv is missing so 'no games' cannot be verified."
                )
            games_df = pd.read_csv(games_path, dtype=str, low_memory=False)
            if len(games_df) > 0:
                return None, (
                    f"[{stage_name}] {kind} {rel} is empty while data/games.csv has {len(games_df)} rows. "
                    "Empty output is only allowed when there are truly no games."
                )
        missing = [c for c in _required_cols(spec.get("required_columns")) if c not in df.columns]
        if missing:
            return None, f"[{stage_name}] {kind} {rel} missing required columns: {missing}"

    max_age_hours = spec.get("max_age_hours")
    if max_age_hours is not None:
        age_h = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600.0
        if age_h > float(max_age_hours):
            return None, f"[{stage_name}] {kind} {rel} stale: age_hours={age_h:.2f}, max_age_hours={max_age_hours}"

    return {"path": rel, "size_bytes": path.stat().st_size}, None


def _producer_stage_map(stages: list[dict[str, Any]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for stage in stages:
        sname = str(stage.get("name", "")).strip()
        for spec in list(stage.get("produces", [])):
            if isinstance(spec, str):
                rel = _norm_rel_path(spec)
            elif isinstance(spec, dict):
                rel = _norm_rel_path(str(spec.get("path", "")).strip())
            else:
                continue
            if not rel:
                continue
            out.setdefault(rel, [])
            if sname and sname not in out[rel]:
                out[rel].append(sname)
    return out


def _append_producer_hints(
    *,
    stage_name: str,
    errors_with_specs: list[tuple[str, Any]],
    producer_map: dict[str, list[str]] | None,
) -> list[str]:
    if not producer_map:
        return [err for err, _ in errors_with_specs]
    hinted: list[str] = []
    for err, spec in errors_with_specs:
        rel = None
        if isinstance(spec, str):
            rel = _norm_rel_path(spec)
        elif isinstance(spec, dict):
            rel = _norm_rel_path(str(spec.get("path", "")).strip())
        producers = [p for p in producer_map.get(rel or "", []) if p != stage_name]
        if producers:
            hinted.append(
                f"{err}\n  -> expected producer stage(s): {sorted(producers)} "
                f"(ensure dependency order reaches {stage_name})"
            )
        else:
            hinted.append(err)
    return hinted


def _validate_specs(
    repo_root: Path,
    specs: list[Any],
    stage_name: str,
    kind: str,
    *,
    producer_map: dict[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    errors_with_specs: list[tuple[str, Any]] = []
    ok_items: list[dict[str, Any]] = []
    for spec in specs:
        info, err = _validate_one_spec(repo_root, spec, stage_name, kind)
        if err:
            errors_with_specs.append((err, spec))
        elif info:
            ok_items.append(info)
    if errors_with_specs:
        if kind == "input":
            errors = _append_producer_hints(
                stage_name=stage_name,
                errors_with_specs=errors_with_specs,
                producer_map=producer_map,
            )
        else:
            errors = [err for err, _ in errors_with_specs]
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


def _validate_dependency_outputs(
    repo_root: Path,
    stage: dict[str, Any],
    stage_lookup: dict[str, dict[str, Any]],
) -> None:
    stage_name = str(stage.get("name", ""))
    deps = [str(d) for d in stage.get("depends_on", [])]
    for dep_name in deps:
        dep = stage_lookup.get(dep_name)
        if not dep:
            continue
        for produced_spec in list(dep.get("produces", [])):
            _, err = _validate_one_spec(
                repo_root=repo_root,
                spec=produced_spec,
                stage_name=stage_name,
                kind=f"dependency_output:{dep_name}",
            )
            if err:
                rel = produced_spec if isinstance(produced_spec, str) else produced_spec.get("path")
                raise RuntimeError(
                    f"[{stage_name}] missing dependency output '{rel}' from stage '{dep_name}'. "
                    f"Expected stage order: {dep_name} -> {stage_name}. Detail: {err}"
                )


def _run_stage(
    repo_root: Path,
    stage: dict[str, Any],
    context: dict[str, str],
    dry_run: bool,
    *,
    producer_map: dict[str, list[str]],
    stage_lookup: dict[str, dict[str, Any]],
) -> StageResult:
    name = str(stage["name"])
    command_tpl = str(stage.get("command", "")).strip()
    if not command_tpl:
        raise RuntimeError(f"[{name}] missing command")
    command = command_tpl.format(**context)

    _validate_dependency_outputs(repo_root, stage, stage_lookup)
    _validate_specs(
        repo_root,
        list(stage.get("requires", [])),
        name,
        "input",
        producer_map=producer_map,
    )

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


def _select_stage_subset(job_stages: list[dict[str, Any]], raw_stage_names: str | None) -> list[dict[str, Any]]:
    if not raw_stage_names:
        return job_stages
    requested = [s.strip() for s in str(raw_stage_names).split(",") if s.strip()]
    if not requested:
        return job_stages

    by_name = {str(s.get("name", "")).strip(): s for s in job_stages}
    unknown = [s for s in requested if s not in by_name]
    if unknown:
        raise RuntimeError(f"Unknown --stages values: {unknown}; available: {sorted(by_name)}")

    selected: set[str] = set()

    def add_with_deps(stage_name: str) -> None:
        if stage_name in selected:
            return
        stage = by_name[stage_name]
        for dep in [str(d) for d in stage.get("depends_on", [])]:
            if dep not in by_name:
                raise RuntimeError(f"[{stage_name}] dependency '{dep}' missing from selected job stages")
            add_with_deps(dep)
        selected.add(stage_name)

    for name in requested:
        add_with_deps(name)
    return [s for s in job_stages if str(s.get("name", "")).strip() in selected]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--job", choices=["update", "rankings"], required=True)
    parser.add_argument("--days-back", type=str, default="3")
    parser.add_argument("--game-type", type=str, default="regular")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stages", type=str, default="", help="Optional comma-separated stage names to run with dependency closure")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    contract_path = args.contract if args.contract.is_absolute() else repo_root / args.contract
    manifest_path = args.manifest if args.manifest.is_absolute() else repo_root / args.manifest

    contract = _load_json(contract_path)
    job_stages = [s for s in contract.get("stages", []) if str(s.get("job", "update")) == args.job]
    if not job_stages:
        raise RuntimeError(f"No contract stages found for job={args.job}")
    stages = _select_stage_subset(job_stages, args.stages)
    producer_map = _producer_stage_map(job_stages)
    stage_lookup = {str(s.get("name", "")).strip(): s for s in stages}

    ordered = _toposort(stages)
    context = {
        "days_back": str(args.days_back),
        "game_type": str(args.game_type),
    }

    results: list[StageResult] = []
    for stage in ordered:
        print(f"[STAGE] {stage['name']}")
        result = _run_stage(
            repo_root,
            stage,
            context,
            args.dry_run,
            producer_map=producer_map,
            stage_lookup=stage_lookup,
        )
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
