#!/usr/bin/env python3
"""Stub: compare window configurations for spread performance."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

WINDOW_CONFIGS = [
    {"name": "L6/L11", "recent": 6, "baseline": 11, "is_production": True},
    {"name": "L5/L10", "recent": 5, "baseline": 10, "is_production": False},
    {"name": "L4/L8", "recent": 4, "baseline": 8, "is_production": False},
    {"name": "L4/L12", "recent": 4, "baseline": 12, "is_production": False},
]


def main() -> int:
    input_path = Path("data/internal/matchup_features.csv")
    out_dir = Path("data/internal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "window_backtest_report.csv"

    if not input_path.exists():
        print(f"[STUB] Missing input: {input_path}")
        return 1

    # Placeholder output contract for future implementation.
    rows = []
    for cfg in WINDOW_CONFIGS:
        rows.append(
            {
                "window_config": cfg["name"],
                "recent_window": cfg["recent"],
                "baseline_window": cfg["baseline"],
                "ats_hit_rate": None,
                "ats_hit_rate_last_50": None,
                "trend_signal_power": None,
                "is_production": cfg["is_production"],
                "promote_to_generate_picks": False,
                "status": "TODO_IMPLEMENT",
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"[STUB] Wrote placeholder report: {out_path}")
    print("[STUB] Next implementation: compute ATS hit rate, last-50 hit rate, and trend power per window config.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
