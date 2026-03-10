from dataclasses import replace

import pandas as pd

from cbb_prediction_model import ModelConfig, PerformanceVsExpectationAnalyzer
from espn_prediction_runner import OPP_HISTORY_WINDOW, build_team_game_list


def _make_row(
    team_id: str,
    opponent_id: str,
    event_id: str,
    game_dt: str,
    team: str,
    opponent: str,
    points_for: int,
    points_against: int,
) -> dict[str, object]:
    return {
        "team_id": team_id,
        "opponent_id": opponent_id,
        "event_id": event_id,
        "game_datetime_utc": game_dt,
        "team": team,
        "opponent": opponent,
        "points_for": points_for,
        "points_against": points_against,
        "neutral_site": False,
        "fgm": 30,
        "fga": 60,
        "tpm": 8,
        "tpa": 24,
        "ftm": 17,
        "fta": 22,
        "orb": 10,
        "drb": 24,
        "tov": 11,
        "pf": 16,
        "opp_fgm": 26,
        "opp_fga": 58,
        "opp_tpm": 7,
        "opp_tpa": 22,
        "opp_ftm": 15,
        "opp_fta": 20,
        "opp_orb": 9,
        "opp_drb": 23,
        "opp_tov": 12,
        "opp_pf": 17,
    }


def _sample_history_frame() -> pd.DataFrame:
    rows = [
        _make_row("B", "D", "1", "2025-01-01T00:00:00Z", "B", "D", 70, 80),
        _make_row("B", "E", "2", "2025-01-03T00:00:00Z", "B", "E", 68, 82),
        _make_row("C", "F", "3", "2025-01-02T00:00:00Z", "C", "F", 60, 55),
        _make_row("C", "G", "4", "2025-01-06T00:00:00Z", "C", "G", 58, 57),
        _make_row("A", "B", "5", "2025-01-05T00:00:00Z", "A", "B", 95, 85),
        _make_row("A", "C", "6", "2025-01-07T00:00:00Z", "A", "C", 88, 70),
    ]
    df = pd.DataFrame(rows)
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    return df


def test_transitive_opponent_history_is_strictly_prior_and_one_level_deep():
    df = _sample_history_frame()
    games = build_team_game_list(
        "A",
        df,
        cutoff_dt=pd.Timestamp("2025-01-08T00:00:00Z"),
        max_games=10,
    )

    assert len(games) == 2
    assert all(1 <= len(game.opponent_history) <= OPP_HISTORY_WINDOW for game in games)

    for game in games:
        for opp_game in game.opponent_history:
            assert opp_game.date < game.date
            assert opp_game.opponent_history == []


def test_transitive_history_changes_vs_expectation_signal():
    df = _sample_history_frame()
    game = build_team_game_list(
        "A",
        df,
        cutoff_dt=pd.Timestamp("2025-01-08T00:00:00Z"),
        max_games=10,
    )[0]

    analyzer = PerformanceVsExpectationAnalyzer(ModelConfig())
    with_history = analyzer.analyze_game(game, baseline_window=5)
    without_history = analyzer.analyze_game(replace(game, opponent_history=[]), baseline_window=5)

    assert with_history["baseline_confidence"] > 0.0
    assert without_history["baseline_confidence"] == 0.0
    assert with_history["off_eff_vs_exp"] != without_history["off_eff_vs_exp"]
