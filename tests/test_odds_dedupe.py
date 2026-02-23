import pandas as pd

from ingestion.espn_pipeline import _append_dedupe_write


def test_append_dedupe_preserves_existing_odds_when_new_row_is_null(tmp_path):
    path = tmp_path / "games.csv"

    existing = pd.DataFrame(
        [
            {
                "game_id": "123",
                "completed": "false",
                "pulled_at_utc": "2026-02-21T00:00:00Z",
                "spread": "-4.5",
                "over_under": "145.5",
                "odds_provider": "Draft Kings",
            }
        ]
    )
    existing.to_csv(path, index=False)

    new = pd.DataFrame(
        [
            {
                "game_id": "123",
                "completed": "true",
                "pulled_at_utc": "2026-02-22T00:00:00Z",
                "spread": None,
                "over_under": None,
                "odds_provider": None,
            }
        ]
    )

    out = _append_dedupe_write(path, new, unique_keys=["game_id"], persist=False)
    row = out.iloc[0]

    assert row["completed"] == "true"
    assert row["spread"] == "-4.5"
    assert row["over_under"] == "145.5"
    assert row["odds_provider"] == "Draft Kings"
