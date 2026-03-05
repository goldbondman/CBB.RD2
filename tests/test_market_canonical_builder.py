from pathlib import Path

import pandas as pd

from pipeline.market_canonical import build_market_canonical_tables, merge_market_lines


def test_market_canonical_builder_uses_games_fallback(tmp_path: Path):
    data_dir = tmp_path / "data"
    debug_dir = tmp_path / "debug"
    data_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        [
            {
                "event_id": "1001",
                "captured_at_utc": "2026-03-05T20:00:00Z",
                "home_spread_current": -4.5,
                "total_current": 145.5,
            }
        ]
    ).to_csv(data_dir / "market_lines_latest.csv", index=False)

    pd.DataFrame(
        [
            {"game_id": "1001", "home_team": "A", "away_team": "B", "spread": -4.0, "over_under": 144.0},
            {"game_id": "1002", "home_team": "C", "away_team": "D", "spread": -1.5, "over_under": 139.0},
        ]
    ).to_csv(data_dir / "games.csv", index=False)

    _, _, latest_by_game, audit, _ = build_market_canonical_tables(data_dir)

    assert audit["latest_by_game_rows"] == 2
    by_id = {str(r["event_id"]): r for _, r in latest_by_game.iterrows()}
    assert by_id["1001"]["source"] != "espn_games"
    assert by_id["1002"]["source"] == "espn_games"
    assert float(by_id["1002"]["spread_line"]) == -1.5

    pred = pd.DataFrame([{"event_id": "1002", "pred_spread": -2.0}])
    merged = merge_market_lines(
        pred,
        data_dir=data_dir,
        output_name="unit_test_output.csv",
        required_columns=["spread_line", "total_line"],
        debug_dir=debug_dir,
    )
    assert float(merged.iloc[0]["spread_line"]) == -1.5
    assert (debug_dir / "market_merge_coverage.csv").exists()

