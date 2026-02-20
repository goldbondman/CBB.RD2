import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from espn_parsers import parse_summary
from espn_player_metrics import compute_player_metrics


FIXTURE = Path("tests/fixtures/summary_boxscore_fixture.json")


def test_parse_summary_populates_raw_player_box_fields():
    raw = json.loads(FIXTURE.read_text())
    parsed = parse_summary(raw, event_id="401")

    assert parsed is not None
    assert parsed["players"]

    row = parsed["players"][0]
    assert row["fgm"] == 7
    assert row["fga"] == 13
    assert row["tpm"] == 2
    assert row["tpa"] == 5
    assert row["fta"] == 6
    assert row["orb"] == 1
    assert row["drb"] == 4
    assert row["plus_minus"] == 8


def test_compute_player_metrics_adds_derived_and_rolling_features():
    raw = json.loads(FIXTURE.read_text())
    parsed = parse_summary(raw, event_id="401")
    base = parsed["players"][0].copy()

    rows = []
    for idx in range(6):
        r = base.copy()
        r["event_id"] = f"evt_{idx}"
        r["game_datetime_utc"] = f"2025-01-{idx+1:02d}T01:00:00Z"
        r["athlete_id"] = "9001"
        r["team_id"] = "100"
        r["did_not_play"] = False
        r["min"] = 30
        r["pts"] = 20 + idx
        rows.append(r)

    player_df = pd.DataFrame(rows)
    out = compute_player_metrics(player_df, pd.DataFrame())

    assert out["efg_pct"].notna().all()
    assert out["three_pct"].notna().all()
    assert out["fg_pct"].notna().all()
    assert out["ft_pct"].notna().all()

    last_row = out.sort_values("game_datetime_utc").iloc[-1]
    assert pd.notna(last_row["pts_l5"])
    assert pd.notna(last_row["pts_l10"])
