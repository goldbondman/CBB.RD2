"""Regression checks for resolve_date_range days_back handling."""

from datetime import date
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingestion.market_lines import resolve_date_range


def run() -> None:
    fixed_today = date(2026, 1, 15)

    start, end = resolve_date_range(None, None, None, today=fixed_today)
    assert start == fixed_today and end == fixed_today

    start, end = resolve_date_range(None, None, "100", today=fixed_today)
    assert start == date(2025, 10, 8)
    assert end == fixed_today

    try:
        resolve_date_range(None, None, 0, today=fixed_today)
        raise AssertionError("Expected ValueError for days_back=0")
    except ValueError as exc:
        assert "--days-back must be >= 1" in str(exc)

    start, end = resolve_date_range(None, None, "", today=fixed_today)
    assert start == fixed_today and end == fixed_today

    start, end = resolve_date_range("2026-01-10", None, None, today=fixed_today)
    assert start == date(2026, 1, 10) and end == date(2026, 1, 10)

    start, end = resolve_date_range(None, "2026-01-12", None, today=fixed_today)
    assert start == date(2026, 1, 12) and end == date(2026, 1, 12)

    try:
        resolve_date_range("2026-01-16", "2026-01-15", None, today=fixed_today)
        raise AssertionError("Expected ValueError when start_date > end_date")
    except ValueError as exc:
        assert "cannot be after" in str(exc)

    print("resolve_date_range regression checks passed")


if __name__ == "__main__":
    run()
