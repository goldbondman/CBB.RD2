from __future__ import annotations

import csv
import sys
from pathlib import Path

MASTER_PATH = Path("data/market_lines_master.csv")
STATE_PATH = Path("data/.market_master_rowcount")


def _current_rowcount(path: Path) -> int:
    if not path.exists():
        return 0

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _read_previous_count(path: Path) -> int:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("0\n", encoding="utf-8")
        return 0

    raw_value = path.read_text(encoding="utf-8").strip()
    if not raw_value:
        return 0

    try:
        return int(raw_value)
    except ValueError:
        return 0


def main() -> int:
    current_count = _current_rowcount(MASTER_PATH)
    previous_count = _read_previous_count(STATE_PATH)

    if current_count < previous_count:
        print("master shrank", file=sys.stderr)
        return 1

    STATE_PATH.write_text(f"{current_count}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
