#!/usr/bin/env python3
"""Fast smoke test for run_capture() default call signature."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ingestion import market_lines


# Keep smoke test offline/fast.
market_lines.REQUEST_DELAY = 0
market_lines.fetch_espn_scoreboard = lambda _game_date: []
market_lines.fetch_action_network = lambda _game_date: []
market_lines.fetch_pinnacle_lines = lambda: []
market_lines.fetch_draftkings_lines = lambda: []

rows = market_lines.run_capture("pregame", Path("data"))
assert isinstance(rows, list), "run_capture should return a list"
print(f"smoke_ok rows={len(rows)}")
