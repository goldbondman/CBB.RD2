#!/usr/bin/env python3
"""Audit CSV row counts and render a markdown quality report.

Pattern matching for expectation rules uses shell-style globs (fnmatch),
matched against repo-relative POSIX paths (for example, ``data/*.csv``).
The first matching rule wins.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - fallback when yaml isn't installed
    yaml = None


DEFAULT_SMALL_ROW_THRESHOLD = 50
DEFAULT_EXPECTATIONS_PATH = Path("data/dq_expectations.yml")


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def parse_expectations_fallback(path: Path) -> dict:
    """Parse a minimal subset of YAML used by dq_expectations.yml."""
    raw = path.read_text(encoding="utf-8")
    out: dict = {"defaults": {}, "files": []}

    threshold_match = re.search(r"^\s*small_row_threshold:\s*(\d+)\s*$", raw, flags=re.MULTILINE)
    if threshold_match:
        out["defaults"]["small_row_threshold"] = int(threshold_match.group(1))

    file_chunks = re.split(r"^\s*-\s+pattern:\s*", raw, flags=re.MULTILINE)[1:]
    for chunk in file_chunks:
        lines = chunk.splitlines()
        if not lines:
            continue
        pattern = _strip_quotes(lines[0])
        min_rows_match = re.search(r"^\s*min_rows:\s*(-?\d+)\s*$", chunk, flags=re.MULTILINE)
        note_match = re.search(r"^\s*note:\s*(.+?)\s*$", chunk, flags=re.MULTILINE)

        expected: dict = {}
        if min_rows_match:
            expected["min_rows"] = int(min_rows_match.group(1))
        if note_match:
            expected["note"] = _strip_quotes(note_match.group(1))

        out["files"].append({"pattern": pattern, "expected": expected})

    return out


@dataclass
class ExpectationRule:
    pattern: str
    min_rows: int | None
    note: str | None


@dataclass
class FileAudit:
    path: str
    row_count: int
    suspicious_small_rows: bool
    expected_min_rows: int | None
    below_expected_min_rows: bool
    expectation_note: str | None
    small_rows_status: str


def load_expectations(path: Path) -> tuple[int, list[ExpectationRule]]:
    """Return (small_row_threshold, ordered_expectation_rules)."""
    if not path.exists() or path.stat().st_size == 0:
        return DEFAULT_SMALL_ROW_THRESHOLD, []

    if yaml is not None:
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = parse_expectations_fallback(path)

    defaults = raw.get("defaults") if isinstance(raw, dict) else {}
    small_row_threshold = DEFAULT_SMALL_ROW_THRESHOLD
    if isinstance(defaults, dict):
        value = defaults.get("small_row_threshold")
        if isinstance(value, int):
            small_row_threshold = value

    rules: list[ExpectationRule] = []
    files = raw.get("files") if isinstance(raw, dict) else []
    if not isinstance(files, list):
        return small_row_threshold, rules

    for entry in files:
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("pattern")
        expected = entry.get("expected")
        if not isinstance(pattern, str):
            continue
        min_rows = None
        note = None
        if isinstance(expected, dict):
            min_value = expected.get("min_rows")
            if isinstance(min_value, int):
                min_rows = min_value
            note_value = expected.get("note")
            if isinstance(note_value, str):
                note = note_value
        rules.append(ExpectationRule(pattern=pattern, min_rows=min_rows, note=note))

    return small_row_threshold, rules


def first_matching_rule(path: str, rules: list[ExpectationRule]) -> ExpectationRule | None:
    for rule in rules:
        if fnmatch(path, rule.pattern):
            return rule
    return None


def count_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        return len(pd.read_csv(path, low_memory=False))
    except Exception:
        return 0


def audit_csvs(data_dir: Path, expectations_path: Path) -> tuple[list[FileAudit], int]:
    small_row_threshold, rules = load_expectations(expectations_path)
    audits: list[FileAudit] = []

    for csv_path in sorted(data_dir.rglob("*.csv")):
        rel_path = csv_path.as_posix()
        row_count = count_rows(csv_path)
        suspicious_small_rows = row_count < small_row_threshold

        matched = first_matching_rule(rel_path, rules)
        expected_min_rows = matched.min_rows if matched else None
        note = matched.note if matched else None
        below_expected_min_rows = (
            expected_min_rows is not None and row_count < expected_min_rows
        )

        small_rows_status = "suspicious"
        if suspicious_small_rows:
            if expected_min_rows is not None and row_count >= expected_min_rows:
                small_rows_status = "small-but-expected"
        else:
            small_rows_status = "ok"

        audits.append(
            FileAudit(
                path=rel_path,
                row_count=row_count,
                suspicious_small_rows=suspicious_small_rows,
                expected_min_rows=expected_min_rows,
                below_expected_min_rows=below_expected_min_rows,
                expectation_note=note,
                small_rows_status=small_rows_status,
            )
        )

    return audits, small_row_threshold


def render_markdown(audits: list[FileAudit], small_row_threshold: int) -> str:
    lines = [
        "# CSV Quality Audit",
        "",
        f"- Small row threshold: `{small_row_threshold}`",
        "- Expectation pattern matching: first-match-wins shell-style glob (`fnmatch`) on repo-relative POSIX file paths.",
        "",
        "## File Summary",
        "",
        "| File | Row Count | suspicious_small_rows | expected_min_rows | below_expected_min_rows | small_rows_status | expectation_note |",
        "|---|---:|---|---:|---|---|---|",
    ]

    for item in audits:
        expected = "" if item.expected_min_rows is None else str(item.expected_min_rows)
        note = "" if item.expectation_note is None else item.expectation_note.replace("|", "\\|")
        lines.append(
            f"| {item.path} | {item.row_count} | {str(item.suspicious_small_rows).lower()} | {expected} | {str(item.below_expected_min_rows).lower()} | {item.small_rows_status} | {note} |"
        )

    below_expected = [a for a in audits if a.below_expected_min_rows]
    no_rule = [a for a in audits if a.expected_min_rows is None]

    lines.extend([
        "",
        "## Row Count Expectations",
        "",
        "### Files below expected minimum",
    ])

    if below_expected:
        for item in below_expected:
            lines.append(
                f"- `{item.path}`: {item.row_count} rows (expected min {item.expected_min_rows})"
            )
    else:
        lines.append("- None")

    lines.extend(["", "### Files with no expectation rule"])
    if no_rule:
        for item in no_rule:
            lines.append(f"- `{item.path}`")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory to recursively scan for CSV files.",
    )
    parser.add_argument(
        "--expectations",
        type=Path,
        default=DEFAULT_EXPECTATIONS_PATH,
        help="Path to row-count expectations YAML file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output markdown path. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audits, threshold = audit_csvs(args.data_dir, args.expectations)
    markdown = render_markdown(audits, threshold)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    else:
        print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
