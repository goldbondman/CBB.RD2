from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enrichment.predictions_with_context import _build_market_lines_fallback


def test_build_market_lines_fallback_preserves_team_names(tmp_path: Path):
    market_path = tmp_path / "market_lines.csv"
    pred_df = pd.DataFrame(
        [
            {
                "event_id": "40123",
                "home_team": "Duke",
                "away_team": "UNC",
                "home_team_id": "150",
                "away_team_id": "153",
                "spread_line": -4.5,
                "total_line": 145.5,
                "home_ml": -190,
                "away_ml": 165,
            }
        ]
    )

    fallback = _build_market_lines_fallback(pred_df, market_path)

    assert not fallback.empty
    assert fallback.iloc[0]["home_team_name"] == "Duke"
    assert fallback.iloc[0]["away_team_name"] == "UNC"
    assert fallback.iloc[0]["home_team_id"] == "150"
    assert fallback.iloc[0]["away_team_id"] == "153"

    written = pd.read_csv(market_path, dtype=str)
    assert "home_team_name" in written.columns
    assert "away_team_name" in written.columns
