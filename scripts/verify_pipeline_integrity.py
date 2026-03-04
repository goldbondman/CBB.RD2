#!/usr/bin/env python3
"""Post-fix verification suite for CBB pipeline integrity."""

from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys
from typing import Dict, List, Tuple

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

REQUIRED_CSVS = [
    DATA_DIR / "predictions_latest.csv",
    DATA_DIR / "predictions_combined_latest.csv",
    DATA_DIR / "ensemble_predictions_latest.csv",
    DATA_DIR / "market_lines.csv",
]

REQUIRED_ARTIFACT_UPLOADS = {
    "INFRA-espn-data",
    "INFRA-predictions-rolling",
    "INFRA-analytics-results",
    "INFRA-market-lines",
}

WORKFLOW_FILES = [
    WORKFLOWS_DIR / "update_espn_cbb.yml",
    WORKFLOWS_DIR / "cbb_predictions_rolling.yml",
    WORKFLOWS_DIR / "cbb_analytics.yml",
    WORKFLOWS_DIR / "market_lines.yml",
]

TOUCHED_PYTHON_FILES = [
    "pipeline_csv_utils.py",
    "config/schemas.py",
    "cbb_output_schemas.py",
    "espn_prediction_runner.py",
    "cbb_prediction_model.py",
    "cbb_ensemble.py",
    "cbb_results_tracker.py",
    "ingestion/market_lines.py",
    "enrichment/predictions_with_context.py",
    "espn_rankings.py",
]


def check_required_csvs() -> Tuple[bool, str]:
    missing = [str(p.relative_to(ROOT)) for p in REQUIRED_CSVS if not p.exists()]
    if missing:
        return False, f"Missing required CSVs: {missing}"

    empty = []
    for p in REQUIRED_CSVS:
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception as exc:
            return False, f"Failed reading {p.relative_to(ROOT)}: {exc}"
        if df.empty:
            empty.append(str(p.relative_to(ROOT)))
    if empty:
        return False, f"Required CSVs are empty: {empty}"
    return True, "Required CSVs exist and are non-empty"


def _workflow_text() -> str:
    parts = []
    for wf in WORKFLOWS_DIR.glob("*.yml"):
        parts.append(wf.read_text(encoding="utf-8"))
    return "\n".join(parts)


def check_artifact_upload_names() -> Tuple[bool, str]:
    text = _workflow_text()
    found = set(re.findall(r"upload-artifact@v\d+.*?name:\s*([^\n]+)", text, re.DOTALL))
    normalized = {f.strip().split()[0] for f in found}
    missing = sorted(REQUIRED_ARTIFACT_UPLOADS - normalized)
    if missing:
        return False, f"Missing artifact upload names: {missing}"
    return True, "All required artifact upload names are present"


def check_broken_downloads() -> Tuple[bool, str]:
    text = _workflow_text()
    uploads = {m.strip().split()[0] for m in re.findall(r"upload-artifact@v\d+.*?name:\s*([^\n]+)", text, re.DOTALL)}
    downloads = {m.strip().split()[0] for m in re.findall(r"download-artifact@v\d+.*?name:\s*([^\n]+)", text, re.DOTALL)}
    broken = sorted(downloads - uploads)
    if broken:
        return False, f"Broken artifact downloads found: {broken}"
    return True, "Zero broken artifact downloads"


def check_predictions_latest_columns_and_dtypes() -> Tuple[bool, str]:
    p = DATA_DIR / "predictions_latest.csv"
    if not p.exists():
        return False, "Missing data/predictions_latest.csv"
    df = pd.read_csv(p, low_memory=False)

    required_cols = ["event_id", "pred_spread", "pred_total", "model_confidence"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return False, f"Missing canonical columns in predictions_latest.csv: {missing}"

    bad_dtype = []
    for col in ["pred_spread", "pred_total", "model_confidence"]:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() == 0:
            bad_dtype.append(f"{col} (all non-numeric/null)")
    if bad_dtype:
        return False, f"Numeric dtype check failed: {bad_dtype}"

    return True, "Canonical columns and numeric dtypes are valid in predictions_latest.csv"


def check_conference_name_not_numeric_any_csv() -> Tuple[bool, str]:
    offenders: List[str] = []
    for p in sorted(DATA_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(p, low_memory=False)
        except Exception:
            continue
        if "conference_name" not in df.columns:
            continue
        vals = df["conference_name"].astype(str).str.strip()
        numeric_mask = vals.str.fullmatch(r"\d+")
        if numeric_mask.fillna(False).any():
            offenders.append(f"{p.name}:{int(numeric_mask.sum())}")
    if offenders:
        return False, f"Numeric conference_name values found in: {offenders}"
    return True, "No numeric conference_name values in CSV outputs"


def check_market_evaluated_positive_if_present() -> Tuple[bool, str]:
    p = DATA_DIR / "predictions_with_context.csv"
    if not p.exists():
        return True, "predictions_with_context.csv not present; check skipped"
    df = pd.read_csv(p, low_memory=False)
    if "market_evaluated" not in df.columns:
        return True, "market_evaluated column not present; check skipped"

    evaluated = pd.to_numeric(df["market_evaluated"], errors="coerce").fillna(0)
    if evaluated.sum() <= 0:
        return False, "market_evaluated <= 0 in predictions_with_context.csv"
    return True, "market_evaluated > 0 in predictions_with_context.csv"


def check_null_rate_non_regression() -> Tuple[bool, str]:
    baseline_path = DATA_DIR / "audit_baseline.json"
    if not baseline_path.exists():
        return False, "Missing baseline file data/audit_baseline.json"

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    regressions = []

    for csv_name, payload in baseline.items():
        if not csv_name.endswith(".csv"):
            continue
        null_rates: Dict[str, float] = payload.get("null_rate", {})
        current_path = DATA_DIR / csv_name
        if not current_path.exists():
            continue
        try:
            df = pd.read_csv(current_path, low_memory=False)
        except Exception:
            continue
        current_null = df.isna().mean().to_dict() if len(df.columns) else {}
        for col, base_rate in null_rates.items():
            if base_rate >= 0.5:
                continue
            if col not in current_null:
                regressions.append(f"{csv_name}:{col} missing")
                continue
            now = float(current_null[col])
            if now > float(base_rate):
                regressions.append(f"{csv_name}:{col} {base_rate:.3f}->{now:.3f}")

    if regressions:
        return False, "Null-rate regressions: " + "; ".join(regressions[:20])
    return True, "Null-rate non-regression check passed"


def check_concurrency_block_all_workflows() -> Tuple[bool, str]:
    missing = []
    for wf in WORKFLOW_FILES:
        if not wf.exists():
            missing.append(f"missing file {wf.relative_to(ROOT)}")
            continue
        text = wf.read_text(encoding="utf-8")
        if "concurrency:" not in text:
            missing.append(str(wf.relative_to(ROOT)))
    if missing:
        return False, f"Missing concurrency block in: {missing}"
    return True, "Concurrency block present in all 4 workflows"


def check_no_autostash_in_workflows() -> Tuple[bool, str]:
    offenders = []
    for wf in WORKFLOWS_DIR.glob("*.yml"):
        text = wf.read_text(encoding="utf-8").lower()
        if "autostash" in text:
            offenders.append(str(wf.relative_to(ROOT)))
    if offenders:
        return False, f"autostash found in workflows: {offenders}"
    return True, "No autostash usage in workflows"


def check_py_compile_touched_files() -> Tuple[bool, str]:
    missing_files = [p for p in TOUCHED_PYTHON_FILES if not (ROOT / p).exists()]
    if missing_files:
        return False, f"Touched Python files missing: {missing_files}"

    cmd = [sys.executable, "-m", "py_compile", *TOUCHED_PYTHON_FILES]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return False, f"py_compile failed: {detail}"
    return True, "py_compile passed on touched Python files"


def run() -> int:
    checks = [
        ("Required CSVs", check_required_csvs),
        ("Artifact upload names", check_artifact_upload_names),
        ("Broken artifact downloads", check_broken_downloads),
        ("Canonical columns + numeric dtypes", check_predictions_latest_columns_and_dtypes),
        ("conference_name sanity", check_conference_name_not_numeric_any_csv),
        ("market_evaluated coverage", check_market_evaluated_positive_if_present),
        ("Null-rate non-regression", check_null_rate_non_regression),
        ("Workflow concurrency", check_concurrency_block_all_workflows),
        ("No autostash", check_no_autostash_in_workflows),
        ("py_compile touched files", check_py_compile_touched_files),
    ]

    failures = 0
    for name, fn in checks:
        ok, message = fn()
        status = "PASS" if ok else "FAIL"
        print(f"{status} [{name}] {message}")
        if not ok:
            failures += 1

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(run())
# home/away splits
