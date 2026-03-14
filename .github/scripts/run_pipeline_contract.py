#!/usr/bin/env python3
"""Execute dependency-aware workflow stages from JSON contracts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Read a CSV file and return (rows, column_names) using stdlib csv."""
    with path.open(newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        cols = list(reader.fieldnames or [])
    return rows, cols


def _parse_dt(val: str) -> datetime | None:
    """Parse a date string to a UTC-aware datetime; returns None on failure."""
    if not val or not val.strip():
        return None
    s = val.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _numeric_non_null_rate(rows: list[dict[str, str]], col: str) -> float:
    """Return fraction of rows where col can be parsed as a finite float."""
    if not rows:
        return 0.0
    count = 0
    for row in rows:
        v = (row.get(col) or "").strip()
        if v:
            try:
                if math.isfinite(float(v)):
                    count += 1
            except ValueError:
                pass
    return count / len(rows)


def _null_rate_pct(rows: list[dict[str, str]], col: str) -> float:
    """Return percentage of rows where col is empty/null."""
    if not rows:
        return 0.0
    null_count = sum(1 for r in rows if not (r.get(col) or "").strip())
    return round(null_count / len(rows) * 100.0, 2)


def _to_float_or_zero(val: str) -> float:
    """Parse val as float; return 0.0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _spec_path(spec: Any) -> str:
    if isinstance(spec, str):
        return _norm_rel_path(spec)
    if isinstance(spec, dict):
        return _norm_rel_path(str(spec.get("path", "")).strip())
    return ""


def _stage_output_status(repo_root: Path, stage: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stage_name = str(stage.get("name", "")).strip()
    for spec in list(stage.get("produces", [])):
        rel = _spec_path(spec)
        if not rel:
            continue
        path = repo_root / rel
        exists = path.exists()
        size_bytes = int(path.stat().st_size) if exists else 0
        _, err = _validate_one_spec(repo_root, spec, stage_name, "output")
        rows.append(
            {
                "path": rel,
                "exists": bool(exists),
                "size_bytes": size_bytes,
                "valid": err is None,
                "error": err,
            }
        )
    return rows


def _write_contract_missing_outputs(repo_root: Path, payload: dict[str, Any]) -> Path:
    out_path = repo_root / "debug" / "contract_missing_outputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


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
    rows, cols = _read_csv(path)
    out: dict[str, Any] = {
        "path": str(path),
        "rows": len(rows),
        "columns": len(cols),
    }

    for col in DATE_COLUMNS:
        if col in cols:
            parsed = [_parse_dt(r.get(col, "")) for r in rows]
            valid_dates = [d for d in parsed if d is not None]
            if valid_dates:
                out["min_date_utc"] = min(valid_dates).isoformat()
                out["max_date_utc"] = max(valid_dates).isoformat()
            break

    key_rates: dict[str, float] = {}
    for col in ["game_id", "event_id", "team_id", "game_datetime_utc"]:
        if col in cols:
            key_rates[col] = _null_rate_pct(rows, col)
    if key_rates:
        out["null_rates_pct"] = key_rates
    return out


def _feature_coverage_summary(repo_root: Path) -> dict[str, Any]:
    data_dir = repo_root / "data"
    summary: dict[str, Any] = {"market": {}, "team_snapshot": {}}

    market_path = data_dir / "market_lines_latest_by_game.csv"
    if not market_path.exists():
        market_path = data_dir / "market_lines_latest.csv"
    if market_path.exists() and market_path.stat().st_size > 0:
        mrows, mcols = _read_csv(market_path)
        spread_col = next((c for c in ["spread_line", "home_spread_current", "spread"] if c in mcols), None)
        total_col = next((c for c in ["total_line", "total_current", "over_under"] if c in mcols), None)
        spread_rate = _numeric_non_null_rate(mrows, spread_col) if spread_col else 0.0
        total_rate = _numeric_non_null_rate(mrows, total_col) if total_col else 0.0
        summary["market"] = {
            "path": str(market_path.relative_to(repo_root)),
            "rows": len(mrows),
            "spread_non_null_rate": round(spread_rate, 6),
            "total_non_null_rate": round(total_rate, 6),
        }
    else:
        summary["market"] = {"path": str(market_path.relative_to(repo_root)), "rows": 0, "missing": True}

    snap_path = data_dir / "team_snapshot.csv"
    if snap_path.exists() and snap_path.stat().st_size > 0:
        srows, scols = _read_csv(snap_path)
        feat_cols = [c for c in ["efg_pct", "efg_pct_l10", "pace", "pace_l10", "form_rating", "net_rtg_std_l10"] if c in scols]
        if feat_cols and srows:
            row_rate = sum(1 for r in srows if any((r.get(c) or "").strip() for c in feat_cols)) / len(srows)
        else:
            row_rate = 0.0
        summary["team_snapshot"] = {
            "path": str(snap_path.relative_to(repo_root)),
            "rows": len(srows),
            "present_feature_columns": feat_cols,
            "non_null_feature_row_rate": round(row_rate, 6),
        }
    else:
        summary["team_snapshot"] = {"path": str(snap_path.relative_to(repo_root)), "rows": 0, "missing": True}

    return summary


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
        rows, cols = _read_csv(path)
        min_rows = int(spec.get("min_rows", 1))
        if len(rows) < min_rows:
            return None, f"[{stage_name}] {kind} {rel} has {len(rows)} rows; expected >= {min_rows}"
        allow_empty = bool(spec.get("allow_empty", False))
        allow_empty_when_no_games = bool(spec.get("allow_empty_when_no_games", False))
        if kind == "output" and len(rows) == 0:
            if allow_empty:
                pass
            elif not allow_empty_when_no_games:
                return None, (
                    f"[{stage_name}] {kind} {rel} is empty (0 rows). "
                    "This is blocked to avoid silent empty outputs."
                )
            else:
                # Prefer upcoming-slate signals (if available) over historical games.csv.
                schedule_path = repo_root / "data" / "plumbing" / "games_schedule_master.csv"
                no_games_context: bool | None = None
                if schedule_path.exists():
                    schedule_rows, _ = _read_csv(schedule_path)
                    no_games_context = len(schedule_rows) == 0
                else:
                    summary_path = repo_root / "data" / "actionable" / "daily_card_summary.csv"
                    if summary_path.exists():
                        summary_rows, summary_cols = _read_csv(summary_path)
                        if "games_on_card" in summary_cols and summary_rows:
                            no_games_context = all(
                                _to_float_or_zero(r.get("games_on_card", "")) <= 0
                                for r in summary_rows
                            )

                if no_games_context is None:
                    games_path = repo_root / "data" / "games.csv"
                    if not games_path.exists():
                        return None, (
                            f"[{stage_name}] {kind} {rel} is empty and allow_empty_when_no_games=true, "
                            "but no upcoming-slate indicator or data/games.csv is available to verify context."
                        )
                    games_rows, _ = _read_csv(games_path)
                    no_games_context = len(games_rows) == 0

                if not no_games_context:
                    context_hint = (
                        "data/plumbing/games_schedule_master.csv"
                        if schedule_path.exists()
                        else ("data/actionable/daily_card_summary.csv" if (repo_root / "data" / "actionable" / "daily_card_summary.csv").exists() else "data/games.csv")
                    )
                    return None, (
                        f"[{stage_name}] {kind} {rel} is empty while {context_hint} indicates games are available. "
                        "Empty output is only allowed when there are truly no games."
                    )
        missing = [c for c in _required_cols(spec.get("required_columns")) if c not in cols]
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
                rel = _spec_path(produced_spec)
                target = repo_root / rel if rel else repo_root
                exists = target.exists() if rel else False
                size_bytes = int(target.stat().st_size) if exists and target.is_file() else 0
                raise RuntimeError(
                    f"[{stage_name}] missing dependency output '{rel}' from stage '{dep_name}'. "
                    f"exists={exists} size_bytes={size_bytes}. "
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
            output_status = _stage_output_status(repo_root, stage)
            formatted = "\n".join(
                f"  - {row['path']}: exists={row['exists']} size_bytes={row['size_bytes']} valid={row['valid']}"
                for row in output_status
            ) or "  - (no declared outputs)"
            raise RuntimeError(
                f"[{name}] command failed with exit code {proc.returncode}: {command}\n"
                f"Declared outputs status:\n{formatted}"
            )

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
    current_stage: dict[str, Any] | None = None
    try:
        for stage in ordered:
            current_stage = stage
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
    except Exception as exc:
        failed_name = str((current_stage or {}).get("name", "unknown"))
        failed_stage = stage_lookup.get(failed_name, current_stage or {})
        failed_outputs = _stage_output_status(repo_root, failed_stage) if failed_stage else []

        per_stage_outputs: dict[str, Any] = {}
        for s in ordered:
            sname = str(s.get("name", "")).strip()
            per_stage_outputs[sname] = _stage_output_status(repo_root, s)

        payload = {
            "generated_at_utc": _utc_now(),
            "contract": str(contract_path),
            "job": args.job,
            "failed_stage": failed_name,
            "error": str(exc),
            "failed_stage_requires": list((failed_stage or {}).get("requires", [])),
            "failed_stage_produces": list((failed_stage or {}).get("produces", [])),
            "failed_stage_outputs": failed_outputs,
            "all_stage_outputs": per_stage_outputs,
            "feature_coverage_summary": _feature_coverage_summary(repo_root),
        }
        debug_path = _write_contract_missing_outputs(repo_root, payload)

        print(f"[ERROR] stage failure: {failed_name}")
        print(f"[ERROR] details: {exc}")
        if failed_outputs:
            print(f"[INFO] required outputs for stage '{failed_name}':")
            for row in failed_outputs:
                print(
                    f"  - {row['path']}: exists={row['exists']} size_bytes={row['size_bytes']} valid={row['valid']}"
                )
                if not row["valid"]:
                    print(f"Missing required output: {row['path']} (stage: {failed_name})")
                    if row.get("error"):
                        print(f"  reason: {row['error']}")
        else:
            print(f"[INFO] stage '{failed_name}' declares no outputs or stage metadata was unavailable.")

        deps = [str(d) for d in (failed_stage.get("depends_on", []) if isinstance(failed_stage, dict) else [])]
        for dep_name in deps:
            dep_rows = per_stage_outputs.get(dep_name, [])
            if not dep_rows:
                continue
            print(f"[INFO] dependency stage output status '{dep_name}':")
            for row in dep_rows:
                print(
                    f"  - {row['path']}: exists={row['exists']} size_bytes={row['size_bytes']} valid={row['valid']}"
                )
                if not row["valid"]:
                    print(f"Missing required output: {row['path']} (stage: {dep_name})")
                    if row.get("error"):
                        print(f"  reason: {row['error']}")

        print("Look at: data/perplexity_contract_runner.log and debug/*")
        print(f"[INFO] wrote failure map: {debug_path}")

        # Always write the manifest even on failure so downstream jobs/artifacts
        # that check for its existence don't fail with a misleading "missing file" error.
        # Completed stages (if any) are included; failed + pending stages are absent.
        error_manifest = {
            "contract_name": contract.get("name", contract_path.name),
            "contract_version": contract.get("version"),
            "job": args.job,
            "generated_at_utc": _utc_now(),
            "dry_run": bool(args.dry_run),
            "status": "failed",
            "failed_stage": failed_name,
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
        manifest_path.write_text(json.dumps(error_manifest, indent=2), encoding="utf-8")
        print(f"[INFO] wrote failure manifest: {manifest_path}")
        return 1

    manifest = {
        "contract_name": contract.get("name", contract_path.name),
        "contract_version": contract.get("version"),
        "job": args.job,
        "generated_at_utc": _utc_now(),
        "dry_run": bool(args.dry_run),
        "context": context,
        "feature_coverage_summary": _feature_coverage_summary(repo_root),
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
