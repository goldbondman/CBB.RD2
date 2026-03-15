from pathlib import Path

import pandas as pd

from ides_of_march.data_steward import (
    _attach_wagertalk_source,
    _wt_name_matches,
    _wt_normalize_name,
    build_data_steward_frame,
)


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
    # wagertalk_historical_odds.csv absent → row count is 0
    assert result.audit["inputs"]["wagertalk_historical_odds_rows"] == 0


def test_wt_normalize_name():
    assert _wt_normalize_name("Ohio St.") == "ohio state"
    assert _wt_normalize_name("VCU") == "vcu"
    assert _wt_normalize_name("St. Bonaventure") == "state bonaventure"
    assert _wt_normalize_name("Texas A&M") == "texas a m"
    assert _wt_normalize_name("") == ""
    assert _wt_normalize_name("nan") == ""
    # Stop words are removed
    assert _wt_normalize_name("University of Kentucky") == "kentucky"


def test_attach_wagertalk_source_sets_source_and_fills_market(tmp_path: Path):
    """_attach_wagertalk_source tags matched rows and fills market columns."""
    import numpy as np

    hist = pd.DataFrame(
        [
            {
                "event_id": "1",
                "date": 20260115,
                "home_team": "Duke Blue Devils",
                "away_team": "North Carolina Tar Heels",
                "actual_margin": 5.0,
                "market_spread": float("nan"),
                "market_total": float("nan"),
            },
            {
                "event_id": "2",
                "date": 20260115,
                "home_team": "Kansas Jayhawks",
                "away_team": "Baylor Bears",
                "actual_margin": -3.0,
                "market_spread": float("nan"),
                "market_total": float("nan"),
            },
            {
                "event_id": "3",
                "date": 20260116,
                "home_team": "Duke Blue Devils",
                "away_team": "North Carolina Tar Heels",
                "actual_margin": 2.0,
                "market_spread": float("nan"),
                "market_total": float("nan"),
            },
        ]
    )

    wagertalk = pd.DataFrame(
        [
            {
                "game_date": "2026-01-15",
                "game_id": 1001,
                "home_team": "Duke",
                "away_team": "North Carolina",
                "consensus_spread": -4.5,
                "consensus_total": 153.0,
                "scraped_at_utc": "2026-01-15T05:00:00Z",
            },
        ]
    )

    result = _attach_wagertalk_source(hist, wagertalk)

    # Row 0: matched — should get source tag and market values
    assert result.loc[0, "historical_odds_source"] == "wagertalk_historical_odds"
    assert result.loc[0, "market_spread"] == -4.5
    assert result.loc[0, "market_total"] == 153.0

    # Row 1: unmatched (different teams) — source tag must be absent (NaN)
    src1 = result.loc[1, "historical_odds_source"]
    assert src1 != src1 or str(src1) in ("nan", "NaN", "")  # NaN check

    # Row 2: different date — not matched
    src2 = result.loc[2, "historical_odds_source"]
    assert src2 != src2 or str(src2) in ("nan", "NaN", "")  # NaN check

    # No extra rows introduced
    assert len(result) == len(hist)


def test_data_steward_populates_wagertalk_rows_in_audit(tmp_path: Path):
    """wagertalk_historical_odds_rows audit key reflects CSV row count."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Minimal games + team data (same as the rolling-windows test)
    games_rows = []
    team_rows = []
    for i in range(1, 5):
        event_id = str(i)
        dt = pd.Timestamp(f"2025-12-{i:02d}T20:00:00Z")
        games_rows.append(
            {
                "game_id": event_id,
                "event_id": event_id,
                "game_datetime_utc": dt.isoformat(),
                "home_team_id": "1",
                "away_team_id": "2",
                "home_team": "Alpha",
                "away_team": "Beta",
                "completed": True,
                "home_score": 70,
                "away_score": 65,
                "date": int(f"202512{i:02d}"),
            }
        )
        for team_id, opp_id, side, net in [("1", "2", "home", 1.0), ("2", "1", "away", -1.0)]:
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

    pd.DataFrame(games_rows).to_csv(data_dir / "games.csv", index=False)
    pd.DataFrame(team_rows).to_csv(data_dir / "team_game_weighted.csv", index=False)

    # Write a small wagertalk CSV with 3 rows (only 1 matching a game)
    pd.DataFrame(
        [
            {
                "game_date": "2025-12-01",
                "game_id": 9001,
                "home_team": "Alpha",
                "away_team": "Beta",
                "consensus_spread": -3.0,
                "consensus_total": 140.0,
                "scraped_at_utc": "2025-12-01T06:00:00Z",
            },
            {
                "game_date": "2025-11-20",
                "game_id": 9002,
                "home_team": "Gamma",
                "away_team": "Delta",
                "consensus_spread": 2.5,
                "consensus_total": 135.0,
                "scraped_at_utc": "2025-11-20T06:00:00Z",
            },
            {
                "game_date": "2025-11-21",
                "game_id": 9003,
                "home_team": "Epsilon",
                "away_team": "Zeta",
                "consensus_spread": float("nan"),
                "consensus_total": float("nan"),
                "scraped_at_utc": "2025-11-21T06:00:00Z",
            },
        ]
    ).to_csv(data_dir / "wagertalk_historical_odds.csv", index=False)

    # Create a minimal market file with a placeholder row so the market merge
    # doesn't fail on a zero-column empty DataFrame (the _combine_market_sources
    # helper returns the fallback copy when primary is empty, and the fallback
    # has no columns when its file is absent).
    pd.DataFrame(
        [{"event_id": "999", "game_datetime_utc": "2025-12-31T20:00:00Z", "spread_line": None, "total_line": None}]
    ).to_csv(data_dir / "market_lines_latest_by_game.csv", index=False)

    result = build_data_steward_frame(
        data_dir=data_dir,
        as_of=pd.Timestamp("2025-12-31T00:00:00Z"),
        hours_ahead=30,
    )

    # Audit must reflect all 3 wagertalk rows
    assert result.audit["inputs"]["wagertalk_historical_odds_rows"] == 3

    # The one matching game (Alpha vs Beta, 2025-12-01) gets tagged
    hist = result.historical_games
    matched = hist[hist.get("historical_odds_source", "") == "wagertalk_historical_odds"]
    assert len(matched) >= 1
    # And its market spread is back-filled from wagertalk
    assert float(matched.iloc[0]["market_spread"]) == -3.0

