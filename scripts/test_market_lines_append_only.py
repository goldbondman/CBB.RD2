"""Unit-ish check for append-only + dedupe market-lines merge logic."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingestion.market_lines import append_market_rows


def main() -> None:
    temp_dir = Path(".tmp_market_lines_test")
    temp_dir.mkdir(exist_ok=True)
    market_path = temp_dir / "market_lines.csv"

    existing = pd.DataFrame(
        [
            {
                "event_id": "g1",
                "game_id": "g1",
                "book": "dk",
                "capture_type": "pregame",
                "captured_at_utc": "2026-03-01T10:00:00Z",
                "pulled_at_utc": "2026-03-01T10:00:00Z",
                "home_spread_current": -4.5,
                "total_current": 145.5,
                "home_ml": -190,
                "away_ml": 165,
            },
            {
                "event_id": "g2",
                "game_id": "g2",
                "book": "fd",
                "capture_type": "pregame",
                "captured_at_utc": "2026-03-01T11:00:00Z",
                "pulled_at_utc": "2026-03-01T11:00:00Z",
                "home_spread_current": -1.5,
                "total_current": 139.5,
                "home_ml": -120,
                "away_ml": 105,
            },
        ]
    )
    existing.to_csv(market_path, index=False)

    new_rows = [
        {
            "event_id": "g1",
            "game_id": "g1",
            "book": "dk",
            "capture_type": "pregame",
            "captured_at_utc": "2026-03-01T10:00:00Z",
            "pulled_at_utc": "2026-03-01T10:00:00Z",
            "home_spread_current": -4.5,
            "total_current": 145.5,
            "home_ml": -190,
            "away_ml": 165,
        },
        {
            "event_id": "g3",
            "game_id": "g3",
            "book": "dk",
            "capture_type": "pregame",
            "captured_at_utc": "2026-03-01T12:00:00Z",
            "pulled_at_utc": "2026-03-01T12:00:00Z",
            "home_spread_current": -2.5,
            "total_current": 141.0,
            "home_ml": -135,
            "away_ml": 118,
        },
    ]

    before_rows = len(existing)
    inserted = append_market_rows(new_rows, market_path)
    merged = pd.read_csv(market_path)
    after_rows = len(merged)

    assert inserted == 1, f"expected 1 inserted row, got {inserted}"
    assert after_rows >= before_rows, f"append-only violated: {after_rows} < {before_rows}"
    assert after_rows == before_rows + 1, f"expected one unique appended row, got {after_rows - before_rows}"

    # Running the same payload again should keep rowcount stable due to dedupe.
    append_market_rows(new_rows, market_path)
    rerun = pd.read_csv(market_path)
    assert len(rerun) == after_rows, "dedupe failed: duplicate rows added on rerun"

    print("PASS: append-only and dedupe behavior verified")


if __name__ == "__main__":
    main()
