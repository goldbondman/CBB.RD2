import pandas as pd

from pipeline.backtest_outputs import (
    build_backtest_model_summary,
    build_leaderboard,
    build_upset_picks_summary,
    write_backtest_outputs,
)


def _sample_game_level() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "1",
                "model_id": "M1",
                "segment": "HIGH_MAJOR",
                "su_result": "W",
                "ats_result": "W",
                "total_result": "L",
                "market_spread_home": 3.5,
                "picked_market_underdog_to_win_flag": True,
                "su_roi_u": 1.2,
                "ats_roi_u": 0.91,
                "total_roi_u": -1.0,
            },
            {
                "game_id": "2",
                "model_id": "M1",
                "segment": "MID_MAJOR",
                "su_result": "L",
                "ats_result": "PUSH",
                "total_result": "W",
                "market_spread_home": -2.0,
                "picked_market_underdog_to_win_flag": False,
                "su_roi_u": -1.0,
                "ats_roi_u": 0.0,
                "total_roi_u": 0.91,
            },
        ]
    )


def test_summary_columns_and_overall_segment():
    summary = build_backtest_model_summary(_sample_game_level())
    expected = [
        "model_id","segment","games",
        "su_w","su_l","su_push","su_win_pct",
        "ats_w","ats_l","ats_push","ats_win_pct",
        "total_w","total_l","total_push","total_win_pct",
        "su_roi_u","ats_roi_u","total_roi_u","overall_roi_u",
    ]
    assert list(summary.columns) == expected
    assert set(summary["segment"]) == {"OVERALL", "HIGH_MAJOR", "MID_MAJOR", "LOW_MAJOR"}


def test_upset_summary_columns():
    upsets = build_upset_picks_summary(_sample_game_level())
    expected = [
        "model_id","segment",
        "dogs_picked","games_dogs_picked_pct",
        "dogs_su_w","dogs_su_l","dogs_su_win_pct",
        "dogs_ats_w","dogs_ats_l","dogs_ats_push","dogs_ats_win_pct",
        "dogs_avg_spread","dogs_median_spread",
        "dogs_su_roi_u","dogs_ats_roi_u",
    ]
    assert list(upsets.columns) == expected


def test_write_backtest_outputs(tmp_path):
    graded = pd.DataFrame(
        [
            {
                "game_id": "10",
                "sub_model": "M2",
                "winner_correct": 1,
                "ats_result": "WIN",
                "ou_result": "LOSS",
                "conference_tier": "HIGH",
                "predicted_winner": "home",
                "spread_line": 2.5,
                "ats_roi": 0.909,
                "ou_roi": -1.0,
                "home_team": "A",
                "away_team": "B",
            }
        ]
    )
    meta = write_backtest_outputs(tmp_path, graded)
    assert (tmp_path / "backtest/summary/backtest_model_summary.csv").exists()
    assert (tmp_path / "backtest/summary/backtest_upset_picks_summary.csv").exists()
    assert meta["rows_summary"] > 0
    leaderboard = pd.read_csv(tmp_path / "backtest/summary/backtest_model_leaderboard.csv")
    assert not leaderboard.empty
