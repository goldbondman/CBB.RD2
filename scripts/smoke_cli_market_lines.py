#!/usr/bin/env python3
"""Smoke-test market_lines CLI parser registration and defaults."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.market_lines import build_parser


def main() -> int:
    parser = build_parser(Path("data"))
    parser.parse_args(["--start-date", "2026-01-01", "--end-date", "2026-01-02"])
    parser.parse_args([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
