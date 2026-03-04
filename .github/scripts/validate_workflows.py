"""
Validate all GitHub Actions workflows for known failure patterns.
Run: python .github/scripts/validate_workflows.py
"""
import re
import sys
from pathlib import Path

WORKFLOW_DIR = Path(".github/workflows")
REQUIRED_CONCURRENCY_GROUP = "cbb-data-pipeline"

# Artifact contract mapping: artifact_name -> {uploaded_by, downloaded_by}
ARTIFACT_CONTRACTS = {
    "INFRA-predictions-rolling": {
        "uploaded_by": "cbb_predictions_rolling.yml",
        "downloaded_by": ["market_lines.yml", "cbb_user_deliverables.yml"],
    },
    "INFRA-market-lines": {
        "uploaded_by": "market_lines.yml",
        "downloaded_by": ["cbb_analytics.yml"],
    },
    "INFRA-predictions-with-context": {
        "uploaded_by": "market_lines.yml",
        "downloaded_by": ["cbb_analytics.yml"],
    },
    "INFRA-espn-data": {
        "uploaded_by": "update_espn_cbb.yml",
        "downloaded_by": ["cbb_predictions_rolling.yml", "cbb_analytics.yml"],
    },
}
errors = []
warnings = []

# Audit each workflow file
for wf in WORKFLOW_DIR.glob("*.yml"):
    content = wf.read_text(encoding="utf-8", errors="replace")

    # Check 1: Concurrency group must be standardized
    if "concurrency:" in content:
        if f"group: {REQUIRED_CONCURRENCY_GROUP}" not in content:
            if REQUIRED_CONCURRENCY_GROUP not in content:
                errors.append(f"{wf.name}: wrong concurrency group")
    else:
        warnings.append(f"{wf.name}: no concurrency block")

    # Check 2: The YAML 'with:' block swap bug
    if "download-artifact" in content and "python-version" in content:
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "download-artifact" in line:
                nearby = "\n".join(lines[max(0, i-2):i+10])
                if "python-version" in nearby:
                    errors.append(
                        f"{wf.name}: python-version appears near "
                        f"download-artifact (likely swapped with: blocks)"
                    )

    # Check 3: Git push should have retry logic
    if "git push" in content and "for i in" not in content:
        warnings.append(f"{wf.name}: git push without retry loop")

    # Check 4: Upload steps should have if: always()
    for match in re.finditer(r"upload-artifact", content):
        start = content.rfind("\n      - name:", 0, match.start())
        if start == -1: start = content.rfind("\n    steps:", 0, match.start())
        block = content[start:match.start() + 200]
        if "if: always()" not in block and "if: " not in block:
            warnings.append(
                f"{wf.name}: upload-artifact step may be missing if: always()"
            )

# Check artifact contracts
uploads = {}
downloads = {}
for wf in WORKFLOW_DIR.glob("*.yml"):
    content = wf.read_text(encoding="utf-8", errors="replace")
    for name in ARTIFACT_CONTRACTS:
        # Use regex to find "name: <name>" and ensure it's not just a comment
        # We look for it preceded by some whitespace and 'name:'
        for match in re.finditer(r"^\s+name:\s+" + re.escape(name) + r"\s*$", content, re.MULTILINE):
            pos = match.start()
            start_search = max(0, pos - 400)
            end_search = pos + 100
            context_block = content[start_search:end_search]

            if "upload-artifact" in context_block:
                uploads[name] = wf.name
            if "download-artifact" in context_block:
                downloads.setdefault(name, []).append(wf.name)

for name, contract in ARTIFACT_CONTRACTS.items():
    if uploads.get(name) != contract["uploaded_by"]:
        errors.append(
            f"Artifact '{name}': expected upload from "
            f"{contract['uploaded_by']}, found {uploads.get(name, 'NONE')}"
        )

    for expected_downloader in contract["downloaded_by"]:
        if expected_downloader not in downloads.get(name, []):
            warnings.append(
                f"Artifact '{name}': expected download by "
                f"{expected_downloader}, but not found in file"
            )

# Final report
if errors:
    print("\nERRORS:")
    for e in errors:
        print(f"  - {e}")
if warnings:
    print("\nWARNINGS:")
    for w in warnings:
        print(f"  - {w}")

if not errors and not warnings:
    print("\nAll workflow checks passed")
elif not errors:
    print("\nWorkflow audit passed (with warnings)")

sys.exit(1 if errors else 0)
