from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAG_PATH = ROOT / "model_lab" / "module_dag.yaml"

RUNTIME_CLASSES = {"FAST", "MED", "SLOW"}
STAGES = {"AUDIT", "BUILD", "EVAL", "DECISION", "REPORT"}
STATUSES = {"SUCCESS", "FAILED", "SKIPPED", "NOT_SELECTED"}


@dataclass
class ModuleSpec:
    name: str
    entrypoint: str
    requires: list[str]
    produces: list[str]
    runtime_class: str
    stage: str
    enabled: bool = True
    timeout_sec: int | None = None


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _load_dag(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"DAG manifest not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("DAG manifest must be a mapping.")
    return payload


def _parse_module(item: dict[str, Any]) -> ModuleSpec:
    if not isinstance(item, dict):
        raise ValueError("Each module entry must be a mapping.")

    name = str(item.get("name", "")).strip()
    entrypoint = str(item.get("entrypoint", "")).strip()
    requires = list(item.get("requires", []))
    produces = list(item.get("produces", []))
    runtime_class = str(item.get("runtime_class", "")).strip().upper()
    stage = str(item.get("stage", "")).strip().upper()
    enabled = bool(item.get("enabled", True))
    timeout_sec = item.get("timeout_sec")

    if not name:
        raise ValueError("Module entry missing non-empty 'name'.")
    if not entrypoint:
        raise ValueError(f"Module '{name}' missing non-empty 'entrypoint'.")
    if runtime_class not in RUNTIME_CLASSES:
        raise ValueError(f"Module '{name}' has invalid runtime_class '{runtime_class}'.")
    if stage not in STAGES:
        raise ValueError(f"Module '{name}' has invalid stage '{stage}'.")
    if timeout_sec is not None:
        timeout_sec = int(timeout_sec)

    return ModuleSpec(
        name=name,
        entrypoint=entrypoint,
        requires=[str(v) for v in requires],
        produces=[str(v) for v in produces],
        runtime_class=runtime_class,
        stage=stage,
        enabled=enabled,
        timeout_sec=timeout_sec,
    )


def _normalize_pattern(pattern: str) -> str:
    return pattern.replace("\\", "/").replace("{run_id}", "RUN_ID").replace("{limit}", "LIMIT")


def _static_prefix(pattern: str) -> str:
    chars: list[str] = []
    for ch in pattern:
        if ch in "*?[":
            break
        chars.append(ch)
    return "".join(chars)


def _patterns_overlap(a: str, b: str) -> bool:
    left = _normalize_pattern(a)
    right = _normalize_pattern(b)
    if left == right:
        return True

    left_prefix = _static_prefix(left)
    right_prefix = _static_prefix(right)
    if left_prefix and right_prefix and (left_prefix.startswith(right_prefix) or right_prefix.startswith(left_prefix)):
        return True

    return False


def _build_dependency_maps(
    modules: list[ModuleSpec],
) -> tuple[dict[str, set[str]], dict[str, dict[str, list[str]]]]:
    deps: dict[str, set[str]] = {m.name: set() for m in modules}
    req_producers: dict[str, dict[str, list[str]]] = {m.name: {} for m in modules}

    for consumer in modules:
        for req in consumer.requires:
            producers: list[str] = []
            for producer in modules:
                if producer.name == consumer.name:
                    continue
                if any(_patterns_overlap(req, out) for out in producer.produces):
                    producers.append(producer.name)
            producers = sorted(set(producers))
            if producers:
                deps[consumer.name].update(producers)
            req_producers[consumer.name][req] = producers

    return deps, req_producers


def _toposort(modules: list[ModuleSpec], deps: dict[str, set[str]]) -> tuple[list[str], bool]:
    names = sorted(m.name for m in modules)
    outgoing: dict[str, set[str]] = {name: set() for name in names}
    indegree: dict[str, int] = {name: 0 for name in names}

    for consumer, producers in deps.items():
        indegree[consumer] = len(producers)
        for producer in producers:
            outgoing.setdefault(producer, set()).add(consumer)

    queue = sorted([name for name in names if indegree[name] == 0])
    order: list[str] = []
    while queue:
        current = queue.pop(0)
        order.append(current)
        for nxt in sorted(outgoing.get(current, set())):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
                queue.sort()

    has_cycle = len(order) != len(names)
    if has_cycle:
        remaining = sorted(set(names) - set(order))
        order.extend(remaining)
    return order, has_cycle


def _profile_selection(
    payload: dict[str, Any],
    modules: list[ModuleSpec],
    profile: str,
) -> set[str]:
    profiles = payload.get("profiles", {})
    if profile not in profiles:
        raise ValueError(f"Unknown profile '{profile}'. Available: {sorted(profiles)}")
    cfg = profiles[profile] or {}

    stages = {str(v).upper() for v in cfg.get("stages", [])}
    runtimes = {str(v).upper() for v in cfg.get("runtime_classes", [])}
    include = {str(v) for v in cfg.get("include", [])}
    exclude = {str(v) for v in cfg.get("exclude", [])}

    selected: set[str] = set()
    for mod in modules:
        if mod.name in exclude:
            continue
        if mod.name in include:
            selected.add(mod.name)
            continue
        stage_ok = (not stages) or (mod.stage in stages)
        runtime_ok = (not runtimes) or (mod.runtime_class in runtimes)
        if stage_ok and runtime_ok:
            selected.add(mod.name)
    return selected


def _format_template(template: str, run_id: str, limit: int) -> str:
    return str(template).format(run_id=run_id, limit=limit)


def _contains_wildcard(path_pattern: str) -> bool:
    return any(ch in path_pattern for ch in "*?[")


def _resolve_pattern(path_pattern: str) -> str:
    if os.path.isabs(path_pattern):
        return path_pattern
    return str(ROOT / path_pattern)


def _matches_for_pattern(path_pattern: str) -> list[str]:
    resolved = _resolve_pattern(path_pattern)
    if _contains_wildcard(path_pattern):
        matches = glob.glob(resolved, recursive=True)
        return sorted(str(Path(m).resolve()) for m in matches)
    candidate = Path(resolved)
    if candidate.exists():
        return [str(candidate.resolve())]
    return []


def _entrypoint_missing_reason(command: str) -> str | None:
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        return f"invalid_entrypoint:{exc}"
    if not tokens:
        return "empty_entrypoint"

    exe = tokens[0]
    if exe == "python":
        if len(tokens) >= 3 and tokens[1] == "-m":
            module_name = tokens[2]
            if importlib.util.find_spec(module_name) is None:
                return f"missing_python_module:{module_name}"
            return None
        if len(tokens) >= 2:
            script = tokens[1]
            script_path = Path(script)
            if not script_path.is_absolute():
                script_path = ROOT / script_path
            if not script_path.exists():
                return f"missing_python_script:{script_path}"
            return None
        return None

    if exe.endswith(".py"):
        script_path = Path(exe)
        if not script_path.is_absolute():
            script_path = ROOT / script_path
        if not script_path.exists():
            return f"missing_python_script:{script_path}"
        return None

    if shutil.which(exe) is None:
        return f"missing_executable:{exe}"
    return None


def _render_summary(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Orchestrator Executive Summary")
    lines.append("")
    lines.append(f"- Run ID: `{payload.get('run_id')}`")
    lines.append(f"- Profile: `{payload.get('profile')}`")
    lines.append(f"- Limit: `{payload.get('limit')}`")
    lines.append(f"- Generated (UTC): `{payload.get('generated_at_utc')}`")
    lines.append(f"- DAG path: `{payload.get('dag_path')}`")
    lines.append("")
    lines.append("## Execution Order")
    for name in payload.get("execution_order", []):
        lines.append(f"- `{name}`")
    lines.append("")
    lines.append("## Module Results")

    results = payload.get("module_results", [])
    if not results:
        lines.append("- None.")
    else:
        for result in results:
            lines.append(
                f"- `{result.get('module')}`: `{result.get('status')}` "
                f"(rc={result.get('return_code')}, produced={result.get('produced_artifact_matches')})"
            )
            reason = result.get("skip_reason") or result.get("failure_reason")
            if reason:
                lines.append(f"  reason: `{reason}`")
    lines.append("")

    skipped = [r for r in results if r.get("status") == "SKIPPED"]
    if skipped:
        lines.append("## Skipped Modules")
        for result in skipped:
            lines.append(f"- `{result.get('module')}`: `{result.get('skip_reason')}`")
        lines.append("")

    failed = [r for r in results if r.get("status") == "FAILED"]
    if failed:
        lines.append("## Failed Modules")
        for result in failed:
            lines.append(f"- `{result.get('module')}`: `{result.get('failure_reason')}`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def cmd_run(args: argparse.Namespace) -> int:
    dag_path = Path(args.dag_path).resolve()
    payload = _load_dag(dag_path)
    modules = [_parse_module(item) for item in payload.get("modules", [])]
    if not modules:
        raise ValueError("No modules declared in DAG manifest.")

    module_by_name = {m.name: m for m in modules}
    deps, req_producers = _build_dependency_maps(modules)
    topo_order, has_cycle = _toposort(modules, deps)
    selected = _profile_selection(payload, modules, args.profile)

    run_root = (ROOT / "data" / "model_lab_runs" / args.run_id).resolve()
    orch_dir = run_root / "orchestrator"
    module_manifest_dir = orch_dir / "modules"
    module_manifest_dir.mkdir(parents=True, exist_ok=True)

    module_status: dict[str, str] = {
        name: ("NOT_SELECTED" if name not in selected else "PENDING")
        for name in module_by_name
    }
    results: list[dict[str, Any]] = []

    for module_name in topo_order:
        if module_name not in selected:
            continue

        mod = module_by_name[module_name]
        started_at = _utc_now()
        result: dict[str, Any] = {
            "module": mod.name,
            "entrypoint_template": mod.entrypoint,
            "runtime_class": mod.runtime_class,
            "stage": mod.stage,
            "enabled": mod.enabled,
            "status": "SKIPPED",
            "return_code": None,
            "started_at_utc": started_at.isoformat(),
            "finished_at_utc": None,
            "duration_sec": None,
            "skip_reason": "",
            "failure_reason": "",
            "command": "",
            "requires": [],
            "requires_check": [],
            "produces": [],
            "produces_check": [],
            "produced_artifact_matches": 0,
            "dependency_modules": sorted(deps.get(mod.name, set())),
            "missing_dependency_requirements": [],
            "missing_direct_requirements": [],
            "stdout_tail": "",
            "stderr_tail": "",
        }

        try:
            formatted_cmd = _format_template(mod.entrypoint, args.run_id, int(args.limit))
            requires_pairs = [(req, _format_template(req, args.run_id, int(args.limit))) for req in mod.requires]
            produces_pairs = [(prd, _format_template(prd, args.run_id, int(args.limit))) for prd in mod.produces]
        except Exception as exc:
            result["skip_reason"] = f"template_render_failed:{exc}"
            module_status[mod.name] = "SKIPPED"
            finished_at = _utc_now()
            result["finished_at_utc"] = finished_at.isoformat()
            result["duration_sec"] = round((finished_at - started_at).total_seconds(), 3)
            results.append(result)
            _write_json(module_manifest_dir / f"{mod.name}_run_manifest.json", result)
            continue

        result["command"] = formatted_cmd
        result["requires"] = [resolved for _, resolved in requires_pairs]
        result["produces"] = [resolved for _, resolved in produces_pairs]

        if not mod.enabled:
            result["skip_reason"] = "disabled_in_dag"
            module_status[mod.name] = "SKIPPED"
        else:
            missing_entry = _entrypoint_missing_reason(formatted_cmd)
            if missing_entry:
                result["skip_reason"] = missing_entry
                module_status[mod.name] = "SKIPPED"
            else:
                missing_dep: list[dict[str, Any]] = []
                missing_direct: list[str] = []
                requires_check: list[dict[str, Any]] = []

                for req_template, req_resolved in requires_pairs:
                    matches = _matches_for_pattern(req_resolved)
                    requires_check.append(
                        {
                            "template": req_template,
                            "resolved": req_resolved,
                            "match_count": len(matches),
                            "matches": matches,
                        }
                    )
                    if matches:
                        continue
                    producers = req_producers.get(mod.name, {}).get(req_template, [])
                    if producers:
                        missing_dep.append(
                            {
                                "resolved": req_resolved,
                                "template": req_template,
                                "candidate_producers": producers,
                                "producer_statuses": {p: module_status.get(p, "NOT_SELECTED") for p in producers},
                            }
                        )
                    else:
                        missing_direct.append(req_resolved)

                result["requires_check"] = requires_check
                result["missing_dependency_requirements"] = missing_dep
                result["missing_direct_requirements"] = missing_direct

                if missing_dep or missing_direct:
                    parts: list[str] = []
                    if missing_dep:
                        dep_strings = [
                            f"{m['resolved']} (producers={','.join(m['candidate_producers'])})"
                            for m in missing_dep
                        ]
                        parts.append("missing_dependency_artifacts: " + "; ".join(dep_strings))
                    if missing_direct:
                        parts.append("missing_direct_requirements: " + "; ".join(missing_direct))
                    result["skip_reason"] = " | ".join(parts)
                    module_status[mod.name] = "SKIPPED"
                else:
                    timeout_sec = int(mod.timeout_sec) if mod.timeout_sec else 1800
                    try:
                        cmd_parts = shlex.split(formatted_cmd)
                        if cmd_parts and cmd_parts[0] == "python":
                            cmd_parts[0] = sys.executable
                        proc = subprocess.run(
                            cmd_parts,
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                            timeout=timeout_sec,
                        )
                        result["return_code"] = int(proc.returncode)
                        result["stdout_tail"] = proc.stdout[-4000:] if proc.stdout else ""
                        result["stderr_tail"] = proc.stderr[-4000:] if proc.stderr else ""
                        if proc.returncode == 0:
                            result["status"] = "SUCCESS"
                            module_status[mod.name] = "SUCCESS"
                        else:
                            result["status"] = "FAILED"
                            result["failure_reason"] = f"nonzero_exit:{proc.returncode}"
                            module_status[mod.name] = "FAILED"
                    except subprocess.TimeoutExpired:
                        result["status"] = "FAILED"
                        result["failure_reason"] = f"timeout_after_sec:{timeout_sec}"
                        module_status[mod.name] = "FAILED"
                    except Exception as exc:
                        result["status"] = "FAILED"
                        result["failure_reason"] = f"execution_error:{exc}"
                        module_status[mod.name] = "FAILED"

        produces_check: list[dict[str, Any]] = []
        total_matches = 0
        for prd_template, prd_resolved in produces_pairs:
            matches = _matches_for_pattern(prd_resolved)
            produces_check.append(
                {
                    "template": prd_template,
                    "resolved": prd_resolved,
                    "match_count": len(matches),
                    "matches": matches,
                }
            )
            total_matches += len(matches)
        result["produces_check"] = produces_check
        result["produced_artifact_matches"] = int(total_matches)

        if result["status"] not in STATUSES:
            result["status"] = "SKIPPED"
            module_status[mod.name] = "SKIPPED"

        finished_at = _utc_now()
        result["finished_at_utc"] = finished_at.isoformat()
        result["duration_sec"] = round((finished_at - started_at).total_seconds(), 3)

        results.append(result)
        _write_json(module_manifest_dir / f"{mod.name}_run_manifest.json", result)

    execution_order = [name for name in topo_order if name in selected]
    selected_results = [row for row in results if row.get("module") in selected]
    failed_any = any(row.get("status") == "FAILED" for row in selected_results)
    failed_fast = any(
        row.get("status") == "FAILED" and module_by_name[row["module"]].runtime_class == "FAST"
        for row in selected_results
    )

    orchestrator_manifest = {
        "run_id": args.run_id,
        "profile": args.profile,
        "limit": int(args.limit),
        "generated_at_utc": _utc_iso(),
        "dag_path": str(dag_path),
        "dag_version": payload.get("version"),
        "topological_order_all_modules": topo_order,
        "execution_order": execution_order,
        "selected_modules": sorted(selected),
        "not_selected_modules": sorted(set(module_by_name) - selected),
        "has_cycle_in_dependency_graph": bool(has_cycle),
        "dependency_map": {k: sorted(v) for k, v in deps.items()},
        "module_status_map": module_status,
        "module_results": selected_results,
    }

    orch_manifest_path = orch_dir / "orchestrator_manifest.json"
    _write_json(orch_manifest_path, orchestrator_manifest)

    summary_path = orch_dir / "ORCHESTRATOR_EXEC_SUMMARY.md"
    _render_summary(summary_path, orchestrator_manifest)

    print(f"run_id={args.run_id}")
    print(f"profile={args.profile}")
    print(f"orchestrator_manifest={orch_manifest_path}")
    print(f"orchestrator_summary={summary_path}")

    if args.profile == "ci":
        return 1 if failed_fast else 0
    return 1 if failed_any else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Model Lab DAG orchestrator")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Execute modules from DAG with dependency-aware scheduling")
    run.add_argument("--run-id", required=True, help="Run identifier under data/model_lab_runs/<run_id>/")
    run.add_argument("--profile", required=True, choices=["ci", "nightly", "weekly"])
    run.add_argument("--limit", type=int, default=200, help="Limit forwarded to module commands when templated.")
    run.add_argument("--dag-path", default=str(DEFAULT_DAG_PATH), help="Path to module DAG yaml.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return cmd_run(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
