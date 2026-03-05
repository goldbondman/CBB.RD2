from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"
DOCS_AUDIT_DIR = ROOT / "docs" / "audit"
OUT_JSON = DOCS_AUDIT_DIR / "repo_audit.json"
OUT_MD = DOCS_AUDIT_DIR / "repo_audit.md"

ARTIFACT_EXTS = (".csv", ".parquet", ".json")
READ_METHODS = {"read_csv", "read_parquet", "read_json"}
WRITE_METHODS = {"to_csv", "to_parquet", "to_json"}


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _all_python_files() -> list[Path]:
    ignored = {".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache"}
    results: list[Path] = []
    for path in ROOT.rglob("*.py"):
        if any(part in ignored for part in path.parts):
            continue
        results.append(path)
    return sorted(results)


def _all_notebooks() -> list[Path]:
    return sorted(p for p in ROOT.rglob("*.ipynb") if ".ipynb_checkpoints" not in str(p))


def _module_name_from_path(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    return ".".join(rel.parts)


def _safe_read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _parse_workflow_python_commands(text: str) -> list[str]:
    cmds: list[str] = []
    for m in re.finditer(r"python\s+-m\s+([A-Za-z0-9_\.]+)", text):
        cmds.append(f"python -m {m.group(1)}")
    for m in re.finditer(r"python\s+([A-Za-z0-9_./-]+\.py)\b", text):
        cmds.append(f"python {m.group(1)}")
    return sorted(set(cmds))


def _discover_entrypoints(py_files: list[Path]) -> dict[str, Any]:
    workflow_entries: list[dict[str, Any]] = []
    for wf in sorted(WORKFLOWS_DIR.glob("*.yml")):
        text = _safe_read_text(wf)
        workflow_entries.append(
            {
                "workflow": str(wf.relative_to(ROOT)),
                "python_commands": _parse_workflow_python_commands(text),
            }
        )

    script_entrypoints: list[str] = []
    module_entrypoints: list[str] = []
    for path in py_files:
        text = _safe_read_text(path)
        rel = str(path.relative_to(ROOT))
        if "if __name__ == \"__main__\"" in text or "if __name__ == '__main__'" in text:
            if path.parent == ROOT:
                script_entrypoints.append(rel)
            module_entrypoints.append(_module_name_from_path(path))
        elif "argparse.ArgumentParser" in text and path.parent == ROOT:
            script_entrypoints.append(rel)

    make_targets: list[str] = []
    makefile = ROOT / "Makefile"
    if makefile.exists():
        for line in _safe_read_text(makefile).splitlines():
            if not line or line.startswith("\t") or line.startswith("#"):
                continue
            m = re.match(r"^([A-Za-z0-9_.-]+)\s*:", line)
            if m:
                make_targets.append(m.group(1))

    notebooks = [str(p.relative_to(ROOT)) for p in _all_notebooks()]
    return {
        "workflows": workflow_entries,
        "scripts": sorted(set(script_entrypoints)),
        "modules": sorted(set(module_entrypoints)),
        "notebooks": notebooks,
        "make_targets": sorted(set(make_targets)),
    }


def _extract_artifact_arg(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        value = node.value.strip()
        if value.endswith(ARTIFACT_EXTS):
            return value
    return None


def _discover_artifacts(py_files: list[Path]) -> dict[str, Any]:
    read_map: dict[str, set[str]] = defaultdict(set)
    write_map: dict[str, set[str]] = defaultdict(set)

    for path in py_files:
        rel = str(path.relative_to(ROOT))
        text = _safe_read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            if isinstance(node.func, ast.Attribute):
                method = node.func.attr
                if method in READ_METHODS and node.args:
                    artifact = _extract_artifact_arg(node.args[0])
                    if artifact:
                        read_map[artifact].add(rel)
                if method in WRITE_METHODS and node.args:
                    artifact = _extract_artifact_arg(node.args[0])
                    if artifact:
                        write_map[artifact].add(rel)

    artifacts: list[dict[str, Any]] = []
    keys = sorted(set(read_map.keys()) | set(write_map.keys()))
    for key in keys:
        ext = Path(key).suffix
        artifacts.append(
            {
                "artifact": key,
                "type": ext.lstrip("."),
                "producers": sorted(write_map.get(key, set())),
                "consumers": sorted(read_map.get(key, set())),
                "usage": {
                    "write_count": len(write_map.get(key, set())),
                    "read_count": len(read_map.get(key, set())),
                },
            }
        )
    return {"artifacts": artifacts}


def _discover_dependencies(py_files: list[Path]) -> dict[str, Any]:
    modules_by_name = {_module_name_from_path(p): str(p.relative_to(ROOT)) for p in py_files}
    inbound: dict[str, set[str]] = defaultdict(set)
    outbound: dict[str, set[str]] = defaultdict(set)

    for path in py_files:
        mod = _module_name_from_path(path)
        text = _safe_read_text(path)
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            target: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    if target in modules_by_name:
                        outbound[mod].add(target)
                        inbound[target].add(mod)
            elif isinstance(node, ast.ImportFrom):
                if node.level and path.parent != ROOT:
                    base_parts = list(_module_name_from_path(path).split(".")[:-1])
                    for _ in range(node.level - 1):
                        if base_parts:
                            base_parts.pop()
                    if node.module:
                        target = ".".join(base_parts + [node.module])
                    else:
                        target = ".".join(base_parts)
                else:
                    target = node.module

                if target and target in modules_by_name:
                    outbound[mod].add(target)
                    inbound[target].add(mod)

    dependency_edges = []
    for src, targets in sorted(outbound.items()):
        dependency_edges.append({"module": src, "imports": sorted(targets)})

    hotspots = []
    for mod_name in modules_by_name:
        refs = len(inbound.get(mod_name, set()))
        if refs > 0:
            hotspots.append(
                {
                    "module": mod_name,
                    "file": modules_by_name[mod_name],
                    "inbound_references": refs,
                }
            )
    hotspots.sort(key=lambda x: (-x["inbound_references"], x["module"]))
    return {"edges": dependency_edges, "hotspots": hotspots, "inbound": inbound}


def _discover_retire_and_quarantine(
    *,
    py_files: list[Path],
    entrypoints: dict[str, Any],
    dependencies: dict[str, Any],
    top_candidates: int,
) -> dict[str, Any]:
    workflow_text = "\n".join(_safe_read_text(p) for p in WORKFLOWS_DIR.glob("*.yml")).lower()
    entrypoint_modules = set(entrypoints.get("modules", []))
    entrypoint_scripts = set(entrypoints.get("scripts", []))

    retire: list[dict[str, Any]] = []
    inbound = dependencies.get("inbound", {})

    for path in py_files:
        rel = str(path.relative_to(ROOT))
        mod = _module_name_from_path(path)
        if rel.startswith("tests/") or rel.endswith("__init__.py"):
            continue

        basename = path.name.lower()
        is_entrypoint = mod in entrypoint_modules or rel in entrypoint_scripts
        in_workflow = basename in workflow_text or rel.lower() in workflow_text
        refs = len(inbound.get(mod, set()))
        if refs == 0 and not is_entrypoint and not in_workflow:
            retire.append(
                {
                    "path": rel,
                    "reason": "unreferenced_uninvoked",
                    "inbound_refs": refs,
                    "workflow_referenced": in_workflow,
                }
            )

    retire.sort(key=lambda x: x["path"])
    retire = retire[: max(1, top_candidates)]

    quarantine_groups: list[dict[str, Any]] = []
    group_patterns = {
        "feature_builders_overlap": [r".*feature.*\.py$", r"^pipeline/advanced_metrics/.*\.py$"],
        "model_runners_overlap": [r".*prediction.*\.py$", r".*ensemble.*\.py$", r".*backtest.*\.py$"],
        "market_lines_overlap": [r".*market_lines.*\.py$", r"^ingestion/historical_backfill\.py$"],
    }
    rel_paths = [str(p.relative_to(ROOT)).replace("\\", "/") for p in py_files]
    for reason, patterns in group_patterns.items():
        matched: set[str] = set()
        for rel in rel_paths:
            if rel.startswith("tests/"):
                continue
            if any(re.match(pat, rel) for pat in patterns):
                matched.add(rel)
        if len(matched) >= 2:
            quarantine_groups.append(
                {
                    "reason": reason,
                    "paths": sorted(matched)[: max(3, min(12, top_candidates))],
                    "note": "Potential overlapping legacy paths; quarantine before deletion.",
                }
            )

    return {"retire_candidates": retire, "quarantine_candidates": quarantine_groups}


def _build_markdown(report: dict[str, Any], top_candidates: int) -> str:
    lines: list[str] = []
    lines.append("# Repo Audit")
    lines.append("")
    lines.append(f"- Generated at: `{report['generated_at_utc']}`")
    lines.append("")

    lines.append("## Entrypoints")
    lines.append("")
    lines.append(f"- Workflow count: `{len(report['entrypoints']['workflows'])}`")
    lines.append(f"- Script entrypoints: `{len(report['entrypoints']['scripts'])}`")
    lines.append(f"- Module entrypoints: `{len(report['entrypoints']['modules'])}`")
    lines.append(f"- Notebook count: `{len(report['entrypoints']['notebooks'])}`")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Tracked artifacts: `{len(report['artifacts']['artifacts'])}`")
    lines.append("")
    lines.append("| Artifact | Producers | Consumers |")
    lines.append("|---|---:|---:|")
    for item in report["artifacts"]["artifacts"][:25]:
        lines.append(
            f"| `{item['artifact']}` | {item['usage']['write_count']} | {item['usage']['read_count']} |"
        )
    lines.append("")

    lines.append("## Dependency Hotspots")
    lines.append("")
    lines.append("| Module | Inbound Refs | File |")
    lines.append("|---|---:|---|")
    for hot in report["dependencies"]["hotspots"][:15]:
        lines.append(f"| `{hot['module']}` | {hot['inbound_references']} | `{hot['file']}` |")
    lines.append("")

    lines.append("## Retire Candidates")
    lines.append("")
    for row in report["candidates"]["retire_candidates"][:top_candidates]:
        lines.append(f"- `{row['path']}`: {row['reason']}")
    if not report["candidates"]["retire_candidates"]:
        lines.append("- None detected by static scan.")
    lines.append("")

    lines.append("## Quarantine Candidates")
    lines.append("")
    for group in report["candidates"]["quarantine_candidates"]:
        lines.append(f"- `{group['reason']}`: {', '.join(group['paths'])}")
    if not report["candidates"]["quarantine_candidates"]:
        lines.append("- None detected by overlap heuristics.")
    lines.append("")
    return "\n".join(lines)


def run_repo_audit(top_candidates: int) -> tuple[Path, Path]:
    py_files = _all_python_files()
    entrypoints = _discover_entrypoints(py_files)
    artifacts = _discover_artifacts(py_files)
    dependencies = _discover_dependencies(py_files)
    candidates = _discover_retire_and_quarantine(
        py_files=py_files,
        entrypoints=entrypoints,
        dependencies=dependencies,
        top_candidates=top_candidates,
    )

    report = {
        "generated_at_utc": _utc_iso(),
        "repo_root": str(ROOT),
        "entrypoints": entrypoints,
        "artifacts": artifacts,
        "dependencies": {
            "hotspots": dependencies["hotspots"],
            "edges": dependencies["edges"],
        },
        "candidates": candidates,
    }

    DOCS_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUT_MD.write_text(_build_markdown(report, top_candidates=top_candidates), encoding="utf-8")
    return OUT_JSON, OUT_MD


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate repo audit artifacts.")
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=20,
        help="Maximum retire candidates to include.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    top = max(1, int(args.top_candidates))
    out_json, out_md = run_repo_audit(top_candidates=top)
    print(f"[OK] Wrote {out_json}")
    print(f"[OK] Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
