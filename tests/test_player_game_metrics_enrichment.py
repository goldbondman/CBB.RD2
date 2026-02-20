import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from espn_parsers import parse_summary
from espn_player_metrics import compute_player_metrics


def _summary_fixture_with_split_strings(event_id: str = "401"):
    return {
        "header": {
            "competitions": [
                {
                    "date": "2026-01-10T01:00Z",
                    "neutralSite": False,
                    "venue": {"fullName": "Test Arena"},
                    "status": {"type": {"completed": True, "state": "post"}, "period": 2},
                    "competitors": [
                        {
                            "homeAway": "home",
                            "score": "70",
                            "team": {"id": "10", "displayName": "Home", "conferenceId": "1", "conference": {"name": "ACC"}},
                            "records": [{"type": "total", "summary": "10-2"}],
                            "linescores": [{"value": "35"}, {"value": "35"}],
                        },
                        {
                            "homeAway": "away",
                            "score": "65",
                            "team": {"id": "20", "displayName": "Away", "conferenceId": "2", "conference": {"name": "SEC"}},
                            "records": [{"type": "total", "summary": "8-4"}],
                            "linescores": [{"value": "30"}, {"value": "35"}],
                        },
                    ],
                }
            ]
        },
        "boxscore": {
            "teams": [],
            "players": [
                {
                    "team": {"id": "10", "displayName": "Home"},
                    "statistics": [
                        {
                            "keys": ["MIN", "PTS", "FG", "3PT", "FT", "OREB", "DREB", "+/-"],
                            "athletes": [
                                {
                                    "athlete": {"id": "1001", "displayName": "Player A", "position": {"abbreviation": "G"}},
                                    "starter": True,
                                    "didNotPlay": False,
                                    "stats": ["33", "20", "7/13", "3/7", "3/4", "2", "4", "+8"],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
    }


def test_parse_summary_populates_raw_box_from_split_stat_strings():
    parsed = parse_summary(_summary_fixture_with_split_strings(), "401")
    player = parsed["players"][0]

    assert player["fgm"] == 7
    assert player["fga"] == 13
    assert player["tpm"] == 3
    assert player["tpa"] == 7
    assert player["fta"] == 4
    assert player["orb"] == 2
    assert player["drb"] == 4
    assert player["plus_minus"] == 8




def test_parse_summary_populates_stats_for_dict_payload_labels():
    summary = _summary_fixture_with_split_strings("402")
    summary["boxscore"]["players"][0]["statistics"][0] = {
        "labels": [
            "MIN",
            "PTS",
            "Field Goals",
            "Three Pointers",
            "Free Throws",
            "Offensive Rebounds",
            "Defensive Rebounds",
            "+/-",
        ],
        "athletes": [
            {
                "athlete": {"id": "1002", "displayName": "Player B", "position": {"abbreviation": "F"}},
                "starter": True,
                "didNotPlay": False,
                "stats": {
                    "MIN": "31",
                    "PTS": "18",
                    "Field Goals": "6/11",
                    "Three Pointers": "2/5",
                    "Free Throws": "4/4",
                    "Offensive Rebounds": "3",
                    "Defensive Rebounds": "5",
                    "+/-": "+6",
                },
            }
        ],
    }

    parsed = parse_summary(summary, "402")
    player = parsed["players"][0]

    assert player["fgm"] == 6
    assert player["fga"] == 11
    assert player["tpm"] == 2
    assert player["tpa"] == 5
    assert player["ftm"] == 4
    assert player["fta"] == 4
    assert player["orb"] == 3
    assert player["drb"] == 5


def test_compute_player_metrics_populates_raw_derived_and_last_windows():
    rows = []
    for i in range(1, 7):
        rows.append(
            {
                "event_id": str(500 + i),
                "game_datetime_utc": f"2026-01-{i:02d}T01:00:00Z",
                "team_id": "10",
                "athlete_id": "1001",
                "did_not_play": "False",
                "starter": "True",
                "min": str(30 + i),
                "pts": str(10 + i),
                "FG": f"{4+i}/{10+i}",
                "3PT": f"{1+i%3}/{3+i%4}",
                "FT": f"{2+i%2}/{3+i%2}",
                "ORB": str(1 + i % 2),
                "DRB": str(3 + i % 2),
                "ast": str(2 + i % 3),
                "stl": "1",
                "blk": "0",
                "tov": "2",
                "pf": "2",
                "plus_minus": str(i),
            }
        )

    player_df = pd.DataFrame(rows)
    team_logs_df = pd.DataFrame(
        [{"event_id": str(500 + i), "team_id": "10", "poss": 68 + i} for i in range(1, 7)]
    )

    out = compute_player_metrics(player_df, team_logs_df)

    for col in ["fgm", "fga", "tpm", "tpa", "fta", "orb", "drb", "plus_minus", "efg_pct", "three_pct", "fg_pct", "ft_pct"]:
        assert out[col].isna().mean() < 1.0, f"{col} should not be fully null"

    assert "last_5_pts" in out.columns
    assert "last_10_pts" in out.columns

    final_row = out.sort_values("game_datetime_utc").iloc[-1]
    assert pd.notna(final_row["pts_l5"])
    assert pd.notna(final_row["last_5_pts"])
