from pathlib import Path

import pandas as pd

from ides_of_march.data_steward import build_data_steward_frame


def test_data_steward_rolling_windows_are_pregame(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    games_rows = []
    team_rows = []
    for i in range(1, 8):
        event_id = str(i)
        dt = pd.Timestamp(f"2026-01-{i:02d}T20:00:00Z")
        games_rows.append(
            {
                "game_id": event_id,
                "event_id": event_id,
                "game_datetime_utc": dt.isoformat(),
                "home_team_id": "1",
                "away_team_id": "2",
                "home_team": "A",
                "away_team": "B",
                "completed": True,
                "home_score": 70 + i,
                "away_score": 65,
            }
        )
        for team_id, opp_id, side, net in [("1", "2", "home", float(i)), ("2", "1", "away", -float(i))]:
            team_rows.append(
                {
                    "event_id": event_id,
                    "game_datetime_utc": dt.isoformat(),
                    "team_id": team_id,
                    "opponent_id": opp_id,
                    "home_away": side,
                    "adj_net_rtg": net,
                    "efg_pct": 0.5,
                    "tov_pct": 0.18,
                    "orb_pct": 0.31,
                    "ftr": 0.28,
                    "ft_pct": 0.74,
                    "pace": 68.0,
                    "three_par": 0.36,
                    "drb_pct": 0.70,
                    "rest_days": 2,
                }
            )

    upcoming_dt = pd.Timestamp("2026-01-08T20:00:00Z")
    games_rows.append(
        {
            "game_id": "8",
            "event_id": "8",
            "game_datetime_utc": upcoming_dt.isoformat(),
            "home_team_id": "1",
            "away_team_id": "2",
            "home_team": "A",
            "away_team": "B",
            "completed": False,
            "home_score": None,
            "away_score": None,
        }
    )

    pd.DataFrame(games_rows).to_csv(data_dir / "games.csv", index=False)
    pd.DataFrame(team_rows).to_csv(data_dir / "team_game_weighted.csv", index=False)
    pd.DataFrame(
        [
            {
                "event_id": "8",
                "game_datetime_utc": upcoming_dt.isoformat(),
                "spread_line": -4.5,
                "total_line": 140.5,
                "line_source_used": "test",
            }
        ]
    ).to_csv(data_dir / "market_lines_latest_by_game.csv", index=False)

    result = build_data_steward_frame(
        data_dir=data_dir,
        as_of=pd.Timestamp("2026-01-08T00:00:00Z"),
        hours_ahead=30,
    )

    assert len(result.upcoming_games) == 1
    # Last 5 adj_em for team 1 before game 8 should average games 3..7 => 4.0
    val = float(result.upcoming_games.iloc[0]["home_Last5_AdjEM"])
    assert round(val, 4) == 4.0
