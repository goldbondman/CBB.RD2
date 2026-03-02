#!/usr/bin/env python3
"""Regenerate latest/closing market line views from snapshots."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.market_lines import regenerate_market_views


def main() -> None:
    latest_count, closing_count = regenerate_market_views(ROOT / "data")
    print(f"market_lines_latest rows: {latest_count}")
    print(f"market_lines_closing rows: {closing_count}")


if __name__ == "__main__":
    main()
