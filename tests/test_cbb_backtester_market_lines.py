import pandas as pd

from cbb_backtester import BacktestEngine, BacktestConfig


def test_attach_market_lines_falls_back_to_market_lines_csv(tmp_path):
    records = pd.DataFrame(
        [
            {
                "game_id": "401000001",
                "actual_margin": 8.0,
                "actual_total": 150.0,
                "ens_spread": -5.5,
                "ens_total": 149.0,
                "cage_em_diff": 2.0,
            }
        ]
    )

    market_lines = pd.DataFrame(
        [
            {
                "event_id": "401000001",
                "captured_at_utc": "2026-01-01T00:00:00Z",
                "home_spread_current": -4.5,
                "total_current": 151.5,
            }
        ]
    )
    fallback_path = tmp_path / "market_lines.csv"
    market_lines.to_csv(fallback_path, index=False)

    engine = BacktestEngine(BacktestConfig())
    out = engine.attach_market_lines(
        records,
        closing_lines_path=tmp_path / "missing_closing.csv",
        fallback_lines_path=fallback_path,
    )

    row = out.iloc[0]
    assert row["home_market_spread"] == -4.5
    assert row["market_spread"] == -4.5
    assert row["market_total"] == 151.5
    assert row["actual_margin_ATS"] == 3.5
    assert row["pred_margin_ATS"] == 1.0
