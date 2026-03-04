import pandas as pd

from team_splits import get_team_splits


def test_get_team_splits_reads_latest_row_by_date(tmp_path):
    csv_path = tmp_path / "team_splits.csv"
    pd.DataFrame(
        [
            {
                "team_id": "10", "season": 2026, "date": "2026-01-01",
                "efg_home": 51.0, "efg_away": 49.0, "orb_home": 30.0, "orb_away": 28.0,
                "tov_home": 16.0, "tov_away": 15.0, "pace_home": 68.0, "pace_away": 67.0,
                "netrtg_home": 8.0, "netrtg_away": 6.0, "ftr_home": 35.0, "ftr_away": 34.0,
            },
            {
                "team_id": "20", "season": 2026, "date": "2026-01-01",
                "efg_home": 50.0, "efg_away": 47.0, "orb_home": 29.0, "orb_away": 27.0,
                "tov_home": 17.0, "tov_away": 14.0, "pace_home": 69.0, "pace_away": 66.0,
                "netrtg_home": 7.0, "netrtg_away": 3.0, "ftr_home": 33.0, "ftr_away": 31.0,
            },
        ]
    ).to_csv(csv_path, index=False)

    matchup = get_team_splits("2026-01-02", "10", "20", csv_path=str(csv_path))

    assert matchup is not None
    assert matchup.home_efg == 51.0
    assert matchup.away_efg == 47.0
    assert matchup.home_netrtg == 8.0
    assert matchup.away_netrtg == 3.0
