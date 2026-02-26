import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from espn_parsers import parse_summary, _parse_made_attempt, _map_player_stat_label
import espn_player_metrics
from espn_player_metrics import compute_player_metrics, add_role_split_metrics


FIXTURE = Path("tests/fixtures/summary_boxscore_fixture.json")


def test_parse_summary_populates_raw_player_box_fields():
    raw = json.loads(FIXTURE.read_text())
    parsed = parse_summary(raw, game_id="401")

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
    parsed = parse_summary(raw, game_id="401")
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


# ── _parse_made_attempt edge cases ────────────────────────────────────────────

def test_parse_made_attempt_standard():
    assert _parse_made_attempt("8-15") == (8, 15)
    assert _parse_made_attempt("0-0") == (0, 0)


def test_parse_made_attempt_slash():
    assert _parse_made_attempt("8/15") == (8, 15)


def test_parse_made_attempt_of():
    assert _parse_made_attempt("7 of 13") == (7, 13)


def test_parse_made_attempt_empty_and_none():
    assert _parse_made_attempt("") == (None, None)
    assert _parse_made_attempt(None) == (None, None)


def test_parse_made_attempt_dashes():
    assert _parse_made_attempt("-") == (None, None)
    assert _parse_made_attempt("--") == (None, None)


def test_parse_made_attempt_na():
    assert _parse_made_attempt("N/A") == (None, None)
    assert _parse_made_attempt("n/a") == (None, None)


# ── PLAYER_STAT_MAP alias coverage ────────────────────────────────────────────

def test_stat_map_fg_aliases():
    for label in ("fg", "fgm-a", "fgma", "fieldgoals", "field goals",
                  "field goal", "fg (ma)"):
        mapped = _map_player_stat_label(label)
        assert mapped == "_fg", f"label '{label}' mapped to {mapped!r}, expected '_fg'"


def test_stat_map_3pt_aliases():
    for label in ("3pt", "3pm-a", "3ptma", "3fgma", "3-pt", "3p",
                  "threepointers", "three pointers", "three point",
                  "3pt (ma)", "3fg"):
        mapped = _map_player_stat_label(label)
        assert mapped == "_3pt", f"label '{label}' mapped to {mapped!r}, expected '_3pt'"


def test_stat_map_ft_aliases():
    for label in ("ft", "ftm-a", "ftma", "freethrows", "free throws",
                  "free throw", "ft (ma)"):
        mapped = _map_player_stat_label(label)
        assert mapped == "_ft", f"label '{label}' mapped to {mapped!r}, expected '_ft'"


def test_stat_map_plus_minus_aliases():
    for label in ("+/-", "pm", "plusminus", "plus/minus"):
        mapped = _map_player_stat_label(label)
        assert mapped == "plus_minus", f"label '{label}' mapped to {mapped!r}"


def test_add_role_split_metrics_handles_string_booleans_for_starter_flag():
    df = pd.DataFrame([
        {"athlete_id": "1", "game_datetime_utc": "2025-01-01T00:00:00Z", "starter": "false", "pts": 10, "min": 20, "efg_pct": 50, "usage_rate": 20},
        {"athlete_id": "1", "game_datetime_utc": "2025-01-02T00:00:00Z", "starter": "true",  "pts": 12, "min": 22, "efg_pct": 52, "usage_rate": 22},
        {"athlete_id": "1", "game_datetime_utc": "2025-01-03T00:00:00Z", "starter": "false", "pts": 14, "min": 24, "efg_pct": 54, "usage_rate": 24},
        {"athlete_id": "1", "game_datetime_utc": "2025-01-04T00:00:00Z", "starter": "true",  "pts": 16, "min": 26, "efg_pct": 56, "usage_rate": 26},
    ])

    out = add_role_split_metrics(df).sort_values("game_datetime_utc").reset_index(drop=True)

    # At game 4, starter rolling window should only use prior starter rows (game 2 only).
    assert pd.isna(out.loc[3, "pts_starter_l5"])

    # At game 4, bench rolling window should average prior bench rows (games 1 and 3 => (10+14)/2).
    assert out.loc[3, "pts_bench_l5"] == 12.0


def test_write_player_splits_keeps_player_name_and_non_null_l5_rows(tmp_path):
    df = pd.DataFrame([
        {
            "athlete_id": "1",
            "player": "Starter One",
            "event_id": "evt_1",
            "team_id": "10",
            "game_datetime_utc": "2025-01-01T00:00:00Z",
            "pts_l5": pd.NA,
            "min_l5": pd.NA,
        },
        {
            "athlete_id": "1",
            "player": "Starter One",
            "event_id": "evt_2",
            "team_id": "10",
            "game_datetime_utc": "2025-01-02T00:00:00Z",
            "pts_l5": 14.0,
            "min_l5": 28.0,
        },
    ])

    rolling_out = tmp_path / "player_rolling_l5.csv"
    role_out = tmp_path / "player_role_splits.csv"

    old_rolling = espn_player_metrics.OUT_PLAYER_ROLLING_L5
    old_role = espn_player_metrics.OUT_PLAYER_ROLE_SPLITS
    try:
        espn_player_metrics.OUT_PLAYER_ROLLING_L5 = rolling_out
        espn_player_metrics.OUT_PLAYER_ROLE_SPLITS = role_out
        espn_player_metrics._write_player_splits(df)
    finally:
        espn_player_metrics.OUT_PLAYER_ROLLING_L5 = old_rolling
        espn_player_metrics.OUT_PLAYER_ROLE_SPLITS = old_role

    out = pd.read_csv(rolling_out)
    assert list(out["player"]) == ["Starter One"]
    assert out["pts_l5"].notna().all()
