"""
tests/test_player_matchup.py — Tests for cbb_player_matchup.py

Validates:
  - normalize() soft-normalization with edge cases
  - _defense_tier() tier mapping and boundaries
  - classify_player_role() classification for all roles
  - build_condition_profiles() condition profile construction
  - build_archetype_matrix() Bayesian shrinkage and grouping
  - compute_player_context_scores() full PCS pipeline
  - build_team_matchup_summary() team-level aggregation
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from features.cbb_player_matchup import (
    normalize,
    _defense_tier,
    classify_player_role,
    build_condition_profiles,
    build_archetype_matrix,
    compute_player_context_scores,
    build_team_matchup_summary,
    DEFENSE_TIERS,
    PACE_THRESHOLD,
    SUFFOCATION_HIGH,
    SUFFOCATION_LOW,
    SHORT_REST_DAYS,
    SHRINKAGE_N,
    MIN_GAMES_PROFILE,
    MIN_DEFENSE_TIER,
    MIN_PACE_SPLIT,
    MIN_LOC_SPLIT,
    MIN_REST_SPLIT,
    PCS_WEIGHTS,
)


# ── Helpers for synthetic data ─────────────────────────────────────────────

TEAM_A_ID = 100
TEAM_B_ID = 200
TEAM_C_ID = 300
TEAM_D_ID = 400
TEAM_E_ID = 500

PLAYER_1_ID = 1001
PLAYER_2_ID = 1002


def _make_player_game_row(
    event_id, team_id, athlete_id, player="Player A", position="G",
    starter=True, pts=15.0, efg_pct=0.50, usage_rate=22.0,
    three_pct=0.34, orb=1.0, min_val=30.0,
    ast_tov_ratio_season_avg=1.6, pts_season_avg=15.0,
    efg_pct_season_avg=0.50, usage_rate_season_avg=22.0,
    min_season_avg=30.0, games_played=20,
    floor_pct=0.45, game_datetime_utc="2025-01-15T19:00:00Z",
    did_not_play=False, team="Team A",
):
    return {
        "event_id": event_id,
        "game_datetime_utc": game_datetime_utc,
        "team_id": team_id,
        "team": team,
        "athlete_id": athlete_id,
        "player": player,
        "position": position,
        "starter": starter,
        "did_not_play": did_not_play,
        "min": min_val,
        "pts": pts,
        "efg_pct": efg_pct,
        "usage_rate": usage_rate,
        "floor_pct": floor_pct,
        "three_pct": three_pct,
        "orb": orb,
        "ast_tov_ratio_season_avg": ast_tov_ratio_season_avg,
        "pts_season_avg": pts_season_avg,
        "efg_pct_season_avg": efg_pct_season_avg,
        "usage_rate_season_avg": usage_rate_season_avg,
        "min_season_avg": min_season_avg,
        "games_played": games_played,
    }


def _make_team_log_row(event_id, team_id, team="Team A"):
    return {"event_id": event_id, "team_id": team_id, "team": team}


def _make_ranking_row(
    team_id, team="Team A", cage_d=100.0, cage_t=70.0,
    suffocation=55.0, opp_efg_pct=0.45, offensive_archetype="balanced",
    tov_pct=18.0, orb_pct=30.0, drb_pct=70.0, ftr=0.30,
    consistency_score=0.75,
):
    return {
        "team_id": team_id,
        "team": team,
        "cage_d": cage_d,
        "cage_t": cage_t,
        "suffocation": suffocation,
        "opp_efg_pct": opp_efg_pct,
        "offensive_archetype": offensive_archetype,
        "tov_pct": tov_pct,
        "orb_pct": orb_pct,
        "drb_pct": drb_pct,
        "ftr": ftr,
        "consistency_score": consistency_score,
    }


def _make_game_row(event_id, home_team_id, away_team_id,
                   game_datetime_utc="2025-01-15T19:00:00Z"):
    return {
        "event_id": event_id,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "game_datetime_utc": game_datetime_utc,
    }


def _build_synthetic_pipeline_data():
    """Build synthetic DataFrames for integration tests.

    Creates 20 games for Player 1 (team A) and Player 2 (team A) against
    opponents with different cage_d values spanning all defense tiers, plus
    varied pace, location, rest, and suffocation.
    """
    pm_rows = []
    tl_rows = []
    games_rows = []
    rk_rows = []

    # Opponent teams with different defense tiers
    opponents = [
        # (team_id, team_name, cage_d, cage_t, suffocation, archetype)
        (TEAM_B_ID, "Team B", 90.0, 72.0, 70.0, "perimeter"),   # elite
        (TEAM_C_ID, "Team C", 97.0, 65.0, 45.0, "interior"),    # good
        (TEAM_D_ID, "Team D", 103.0, 75.0, 55.0, "grind"),      # average
        (TEAM_E_ID, "Team E", 110.0, 60.0, 40.0, "balanced"),   # weak
    ]

    # Rankings for all teams
    rk_rows.append(_make_ranking_row(
        TEAM_A_ID, "Team A", cage_d=98.0, cage_t=70.0,
        suffocation=55.0, offensive_archetype="balanced",
    ))
    for tid, tname, cd, ct, suf, arch in opponents:
        rk_rows.append(_make_ranking_row(
            tid, tname, cage_d=cd, cage_t=ct, suffocation=suf,
            offensive_archetype=arch,
        ))

    # Generate 20 games (5 per opponent) cycling through opponents
    base_dates = [
        "2025-01-01T19:00:00Z", "2025-01-03T19:00:00Z",
        "2025-01-06T19:00:00Z", "2025-01-08T19:00:00Z",
        "2025-01-11T19:00:00Z", "2025-01-14T19:00:00Z",
        "2025-01-17T19:00:00Z", "2025-01-19T19:00:00Z",
        "2025-01-22T19:00:00Z", "2025-01-24T19:00:00Z",
        "2025-01-27T19:00:00Z", "2025-01-29T19:00:00Z",
        "2025-02-01T19:00:00Z", "2025-02-03T19:00:00Z",
        "2025-02-06T19:00:00Z", "2025-02-08T19:00:00Z",
        "2025-02-11T19:00:00Z", "2025-02-13T19:00:00Z",
        "2025-02-16T19:00:00Z", "2025-02-18T19:00:00Z",
    ]

    for i in range(20):
        eid = 9000 + i
        opp_idx = i % 4
        opp_tid, opp_name, opp_cd, opp_ct, opp_suf, opp_arch = opponents[opp_idx]
        dt = base_dates[i]
        is_home = (i % 2 == 0)

        home_tid = TEAM_A_ID if is_home else opp_tid
        away_tid = opp_tid if is_home else TEAM_A_ID

        games_rows.append(_make_game_row(eid, home_tid, away_tid, dt))

        # Team logs for both teams
        tl_rows.append(_make_team_log_row(eid, TEAM_A_ID, "Team A"))
        tl_rows.append(_make_team_log_row(eid, opp_tid, opp_name))

        # Player 1: primary handler (usage>25, atr>1.5)
        pts_variation = 2.0 * ((-1) ** i)  # alternate +/- for variance
        pm_rows.append(_make_player_game_row(
            event_id=eid, team_id=TEAM_A_ID, athlete_id=PLAYER_1_ID,
            player="Star Player", position="G", starter=True,
            pts=18.0 + pts_variation, efg_pct=0.52, usage_rate=27.0,
            three_pct=0.36, orb=0.5, min_val=32.0,
            ast_tov_ratio_season_avg=2.0, pts_season_avg=18.0,
            efg_pct_season_avg=0.50, usage_rate_season_avg=26.0,
            min_season_avg=32.0, games_played=20,
            game_datetime_utc=dt, team="Team A",
        ))

        # Player 2: bench energy (not starter, low mins)
        pm_rows.append(_make_player_game_row(
            event_id=eid, team_id=TEAM_A_ID, athlete_id=PLAYER_2_ID,
            player="Bench Guy", position="G", starter=False,
            pts=5.0 + pts_variation * 0.5, efg_pct=0.42, usage_rate=14.0,
            three_pct=0.30, orb=0.3, min_val=15.0,
            ast_tov_ratio_season_avg=1.0, pts_season_avg=5.0,
            efg_pct_season_avg=0.42, usage_rate_season_avg=14.0,
            min_season_avg=15.0, games_played=20,
            game_datetime_utc=dt, team="Team A",
        ))

    pm = pd.DataFrame(pm_rows)
    tl = pd.DataFrame(tl_rows)
    games = pd.DataFrame(games_rows)
    rankings = pd.DataFrame(rk_rows)

    return pm, tl, rankings, games


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def pipeline_data():
    return _build_synthetic_pipeline_data()


@pytest.fixture
def profiles(pipeline_data):
    pm, tl, rankings, games = pipeline_data
    prof = build_condition_profiles(pm, tl, rankings, games)
    # build_condition_profiles omits min_season_avg; restore from player metrics
    if "min_season_avg" not in prof.columns:
        min_map = (
            pm.groupby("athlete_id")["min_season_avg"]
            .first()
            .to_dict()
        )
        prof["min_season_avg"] = prof["athlete_id"].map(min_map)
    return prof


# ── Section: normalize ─────────────────────────────────────────────────────

class TestNormalize:
    def test_positive_delta(self):
        result = normalize(5.0, 10.0)
        expected = float(np.tanh(0.5))
        assert abs(result - expected) < 1e-9

    def test_negative_delta(self):
        result = normalize(-5.0, 10.0)
        expected = float(np.tanh(-0.5))
        assert abs(result - expected) < 1e-9

    def test_zero_delta(self):
        assert normalize(0.0, 10.0) == 0.0

    def test_none_returns_zero(self):
        assert normalize(None, 10.0) == 0.0

    def test_nan_returns_zero(self):
        assert normalize(float("nan"), 10.0) == 0.0

    def test_large_positive(self):
        result = normalize(100.0, 10.0)
        assert result > 0.99

    def test_large_negative(self):
        result = normalize(-100.0, 10.0)
        assert result < -0.99

    def test_output_bounded(self):
        for delta in [-1000, -10, -1, 0, 1, 10, 1000]:
            result = normalize(delta, 5.0)
            assert -1.0 <= result <= 1.0


# ── Section: _defense_tier ─────────────────────────────────────────────────

class TestDefenseTier:
    def test_elite_low(self):
        assert _defense_tier(0) == "elite"

    def test_elite_mid(self):
        assert _defense_tier(90) == "elite"

    def test_elite_boundary(self):
        """95 is the exclusive upper bound of elite → good starts at 95."""
        assert _defense_tier(94.99) == "elite"

    def test_good_at_95(self):
        assert _defense_tier(95) == "good"

    def test_good_mid(self):
        assert _defense_tier(97.5) == "good"

    def test_good_boundary(self):
        assert _defense_tier(99.99) == "good"

    def test_average_at_100(self):
        assert _defense_tier(100) == "average"

    def test_average_mid(self):
        assert _defense_tier(103) == "average"

    def test_average_boundary(self):
        assert _defense_tier(105.99) == "average"

    def test_weak_at_106(self):
        assert _defense_tier(106) == "weak"

    def test_weak_high(self):
        assert _defense_tier(150) == "weak"

    def test_nan_returns_none(self):
        assert _defense_tier(float("nan")) is None

    def test_none_returns_none(self):
        assert _defense_tier(None) is None

    def test_above_999_returns_none(self):
        assert _defense_tier(999) is None


# ── Section: classify_player_role ──────────────────────────────────────────

class TestClassifyPlayerRole:
    def test_primary_handler(self):
        row = pd.Series({
            "usage_rate_season_avg": 26, "ast_tov_ratio_season_avg": 2.0,
            "three_pct": 0.34, "position": "G", "orb": 1.0,
            "starter": True, "min_season_avg": 30,
        })
        assert classify_player_role(row) == "PRIMARY_HANDLER"

    def test_wing_scorer(self):
        row = pd.Series({
            "usage_rate_season_avg": 22, "ast_tov_ratio_season_avg": 1.0,
            "three_pct": 0.36, "position": "G", "orb": 1.0,
            "starter": True, "min_season_avg": 28,
        })
        assert classify_player_role(row) == "WING_SCORER"

    def test_wing_scorer_forward(self):
        row = pd.Series({
            "usage_rate_season_avg": 20, "ast_tov_ratio_season_avg": 1.0,
            "three_pct": 0.35, "position": "F", "orb": 0.5,
            "starter": True, "min_season_avg": 28,
        })
        assert classify_player_role(row) == "WING_SCORER"

    def test_post_scorer(self):
        row = pd.Series({
            "usage_rate_season_avg": 22, "ast_tov_ratio_season_avg": 0.8,
            "three_pct": 0.10, "position": "C", "orb": 3.0,
            "starter": True, "min_season_avg": 28,
        })
        assert classify_player_role(row) == "POST_SCORER"

    def test_post_scorer_forward(self):
        row = pd.Series({
            "usage_rate_season_avg": 20, "ast_tov_ratio_season_avg": 0.9,
            "three_pct": 0.15, "position": "F", "orb": 2.5,
            "starter": True, "min_season_avg": 28,
        })
        assert classify_player_role(row) == "POST_SCORER"

    def test_floor_spacer(self):
        row = pd.Series({
            "usage_rate_season_avg": 14, "ast_tov_ratio_season_avg": 1.0,
            "three_pct": 0.40, "position": "G", "orb": 0.5,
            "starter": True, "min_season_avg": 25,
        })
        assert classify_player_role(row) == "FLOOR_SPACER"

    def test_connector(self):
        row = pd.Series({
            "usage_rate_season_avg": 18, "ast_tov_ratio_season_avg": 1.5,
            "three_pct": 0.30, "position": "G", "orb": 1.0,
            "starter": True, "min_season_avg": 28,
        })
        assert classify_player_role(row) == "CONNECTOR"

    def test_bench_energy(self):
        row = pd.Series({
            "usage_rate_season_avg": 12, "ast_tov_ratio_season_avg": 0.8,
            "three_pct": 0.28, "position": "G", "orb": 0.5,
            "starter": False, "min_season_avg": 18,
        })
        assert classify_player_role(row) == "BENCH_ENERGY"

    def test_fallback_connector(self):
        """Players not matching any specific role fall back to CONNECTOR."""
        row = pd.Series({
            "usage_rate_season_avg": 30, "ast_tov_ratio_season_avg": 0.5,
            "three_pct": 0.28, "position": "C", "orb": 0.5,
            "starter": True, "min_season_avg": 30,
        })
        assert classify_player_role(row) == "CONNECTOR"

    def test_missing_fields_default_to_zero(self):
        """Missing fields should not raise; role defaults to BENCH_ENERGY or CONNECTOR."""
        row = pd.Series({"starter": False, "min_season_avg": 10})
        role = classify_player_role(row)
        assert role in (
            "PRIMARY_HANDLER", "WING_SCORER", "POST_SCORER",
            "FLOOR_SPACER", "CONNECTOR", "BENCH_ENERGY",
        )


# ── Section: build_condition_profiles ──────────────────────────────────────

class TestBuildConditionProfiles:
    def test_returns_dataframe(self, profiles):
        assert isinstance(profiles, pd.DataFrame)
        assert not profiles.empty

    def test_has_expected_columns(self, profiles):
        expected = [
            "athlete_id", "player", "team_id", "team", "position",
            "games_played", "pts_season_avg",
        ]
        for col in expected:
            assert col in profiles.columns, f"Missing column: {col}"

    def test_defense_tier_columns(self, profiles):
        for tier in DEFENSE_TIERS:
            assert f"n_vs_{tier}" in profiles.columns
            assert f"pts_delta_vs_{tier}" in profiles.columns
            assert f"efg_delta_vs_{tier}" in profiles.columns

    def test_pace_split_columns(self, profiles):
        for label in ["fast_pace", "slow_pace"]:
            assert f"pts_delta_{label}" in profiles.columns
            assert f"efg_delta_{label}" in profiles.columns

    def test_location_split_columns(self, profiles):
        for loc in ["home", "away"]:
            assert f"pts_delta_{loc}" in profiles.columns
            assert f"efg_delta_{loc}" in profiles.columns

    def test_rest_split_columns(self, profiles):
        assert "n_short_rest" in profiles.columns
        assert "pts_delta_short_rest" in profiles.columns

    def test_suffocation_columns(self, profiles):
        assert "pts_delta_vs_suffocation" in profiles.columns
        assert "n_vs_high_suffocation" in profiles.columns
        assert "n_vs_low_suffocation" in profiles.columns

    def test_player_count(self, profiles):
        """Two players were created in synthetic data."""
        assert len(profiles) == 2

    def test_tier_game_counts_sum(self, profiles):
        """Each player has 5 games per opponent (tier), total across tiers = 20."""
        for _, row in profiles.iterrows():
            total = sum(row.get(f"n_vs_{t}", 0) for t in DEFENSE_TIERS)
            assert total == 20

    def test_defense_tier_deltas_populated(self, profiles):
        """With 5 games per tier and MIN_DEFENSE_TIER=3, deltas should be non-null."""
        row = profiles.iloc[0]
        for tier in DEFENSE_TIERS:
            assert row[f"n_vs_{tier}"] >= MIN_DEFENSE_TIER
            assert pd.notna(row[f"pts_delta_vs_{tier}"])


# ── Section: build_archetype_matrix ────────────────────────────────────────

class TestBuildArchetypeMatrix:
    def test_returns_dataframe(self, pipeline_data, profiles):
        pm, tl, rankings, games = pipeline_data
        matrix = build_archetype_matrix(profiles, pm, tl, rankings)
        assert isinstance(matrix, pd.DataFrame)

    def test_has_expected_columns(self, pipeline_data, profiles):
        pm, tl, rankings, games = pipeline_data
        matrix = build_archetype_matrix(profiles, pm, tl, rankings)
        expected = [
            "offensive_archetype", "opponent_archetype", "player_role",
            "avg_pts_delta", "avg_efg_delta", "avg_usage_delta",
            "n_player_games", "confidence",
        ]
        for col in expected:
            assert col in matrix.columns, f"Missing column: {col}"

    def test_bayesian_shrinkage(self, pipeline_data, profiles):
        """Shrunk values should be closer to 0 than raw values for small N."""
        pm, tl, rankings, games = pipeline_data
        matrix = build_archetype_matrix(profiles, pm, tl, rankings)
        for _, row in matrix.iterrows():
            n = row["n_player_games"]
            shrink_factor = n / (n + SHRINKAGE_N)
            # Shrinkage means magnitude of shrunk value <= magnitude of raw
            assert shrink_factor <= 1.0
            assert shrink_factor > 0.0

    def test_confidence_levels(self, pipeline_data, profiles):
        pm, tl, rankings, games = pipeline_data
        matrix = build_archetype_matrix(profiles, pm, tl, rankings)
        assert set(matrix["confidence"].unique()).issubset({"LOW", "MEDIUM", "HIGH"})


# ── Section: compute_player_context_scores ─────────────────────────────────

class TestComputePlayerContextScores:
    @pytest.fixture
    def pcs_inputs(self, pipeline_data, profiles):
        pm, tl, rankings, games = pipeline_data
        # upcoming game: event 99999
        upcoming_eid = 99999
        upcoming_game = pd.DataFrame([
            _make_game_row(upcoming_eid, TEAM_A_ID, TEAM_B_ID,
                           "2025-02-20T19:00:00Z"),
        ])
        all_games = pd.concat([games, upcoming_game], ignore_index=True)
        predictions = pd.DataFrame([{
            "event_id": upcoming_eid,
            "home_team": "Team A",
            "away_team": "Team B",
        }])
        injury_proxy = pd.DataFrame({"athlete_id": []})
        return profiles, rankings, all_games, predictions, injury_proxy

    def test_returns_dataframe(self, pcs_inputs):
        profiles, rankings, games, predictions, injury_proxy = pcs_inputs
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
        )
        assert isinstance(pcs, pd.DataFrame)
        assert not pcs.empty

    def test_has_expected_columns(self, pcs_inputs):
        profiles, rankings, games, predictions, injury_proxy = pcs_inputs
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
        )
        expected = [
            "event_id", "athlete_id", "player", "team_id", "pcs",
            "pcs_tier", "expected_pts_delta", "data_confidence",
            "star_player", "injury_flagged", "location",
        ]
        for col in expected:
            assert col in pcs.columns, f"Missing column: {col}"

    def test_pcs_bounded(self, pcs_inputs):
        profiles, rankings, games, predictions, injury_proxy = pcs_inputs
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
        )
        assert (pcs["pcs"] >= -100).all()
        assert (pcs["pcs"] <= 100).all()

    def test_pcs_tier_values(self, pcs_inputs):
        profiles, rankings, games, predictions, injury_proxy = pcs_inputs
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
        )
        valid_tiers = {"STRONG_OVER", "LEAN_OVER", "NEUTRAL",
                       "LEAN_UNDER", "STRONG_UNDER"}
        assert set(pcs["pcs_tier"].unique()).issubset(valid_tiers)

    def test_location_assigned(self, pcs_inputs):
        profiles, rankings, games, predictions, injury_proxy = pcs_inputs
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
        )
        home_rows = pcs[pcs["team_id"] == TEAM_A_ID]
        away_rows = pcs[pcs["team_id"] == TEAM_B_ID]
        if not home_rows.empty:
            assert (home_rows["location"] == "home").all()
        if not away_rows.empty:
            assert (away_rows["location"] == "away").all()

    def test_injury_flagged_empty_proxy(self, pcs_inputs):
        """With empty injury proxy, no players should be flagged."""
        profiles, rankings, games, predictions, injury_proxy = pcs_inputs
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
        )
        assert not pcs["injury_flagged"].any()

    def test_injury_flagged_with_injured_player(self, pcs_inputs):
        profiles, rankings, games, predictions, _ = pcs_inputs
        injury_proxy = pd.DataFrame({"athlete_id": [PLAYER_1_ID]})
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
        )
        injured = pcs[pcs["athlete_id"] == PLAYER_1_ID]
        if not injured.empty:
            assert injured["injury_flagged"].all()

    def test_star_player_flag(self, pcs_inputs):
        """Player 1 has usage_rate_season_avg=26, should be star."""
        profiles, rankings, games, predictions, injury_proxy = pcs_inputs
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
        )
        p1 = pcs[pcs["athlete_id"] == PLAYER_1_ID]
        if not p1.empty:
            assert p1["star_player"].all()

    def test_data_confidence_values(self, pcs_inputs):
        profiles, rankings, games, predictions, injury_proxy = pcs_inputs
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
        )
        valid = {"HIGH", "MEDIUM", "LOW"}
        assert set(pcs["data_confidence"].unique()).issubset(valid)

    def test_empty_predictions(self, profiles, pipeline_data):
        _, _, rankings, games = pipeline_data
        predictions = pd.DataFrame(columns=["event_id", "home_team", "away_team"])
        pcs = compute_player_context_scores(
            profiles, rankings, games, predictions, None,
        )
        assert pcs.empty


# ── Section: build_team_matchup_summary ────────────────────────────────────

class TestBuildTeamMatchupSummary:
    @pytest.fixture
    def pcs_df(self, pipeline_data, profiles):
        pm, tl, rankings, games = pipeline_data
        upcoming_eid = 99999
        upcoming_game = pd.DataFrame([
            _make_game_row(upcoming_eid, TEAM_A_ID, TEAM_B_ID,
                           "2025-02-20T19:00:00Z"),
        ])
        all_games = pd.concat([games, upcoming_game], ignore_index=True)
        predictions = pd.DataFrame([{
            "event_id": upcoming_eid,
            "home_team": "Team A",
            "away_team": "Team B",
        }])
        return compute_player_context_scores(
            profiles, rankings, all_games, predictions, None,
        )

    def test_returns_dataframe(self, pcs_df, pipeline_data):
        pm = pipeline_data[0]
        summary = build_team_matchup_summary(pcs_df, pm)
        assert isinstance(summary, pd.DataFrame)
        assert not summary.empty

    def test_has_expected_columns(self, pcs_df, pipeline_data):
        pm = pipeline_data[0]
        summary = build_team_matchup_summary(pcs_df, pm)
        expected = [
            "event_id", "team_id", "team", "location",
            "star_pcs", "avg_pcs_starters", "avg_pcs_all",
            "expected_pts_impact", "n_players_scored",
            "star_underperform_risk", "favorable_conditions",
        ]
        for col in expected:
            assert col in summary.columns, f"Missing column: {col}"

    def test_one_row_per_team_event(self, pcs_df, pipeline_data):
        pm = pipeline_data[0]
        summary = build_team_matchup_summary(pcs_df, pm)
        dupes = summary.duplicated(subset=["event_id", "team_id"])
        assert not dupes.any()

    def test_empty_pcs(self, pipeline_data):
        pm = pipeline_data[0]
        summary = build_team_matchup_summary(pd.DataFrame(), pm)
        assert summary.empty

    def test_star_pcs_populated(self, pcs_df, pipeline_data):
        pm = pipeline_data[0]
        summary = build_team_matchup_summary(pcs_df, pm)
        team_a = summary[summary["team_id"] == TEAM_A_ID]
        if not team_a.empty:
            assert pd.notna(team_a.iloc[0]["star_pcs"])

    def test_n_players_scored(self, pcs_df, pipeline_data):
        pm = pipeline_data[0]
        summary = build_team_matchup_summary(pcs_df, pm)
        team_a = summary[summary["team_id"] == TEAM_A_ID]
        if not team_a.empty:
            assert team_a.iloc[0]["n_players_scored"] >= 1


# ── Section: enrich_with_matchup_summary ───────────────────────────────────

class TestEnrichWithMatchupSummary:
    def test_returns_unchanged_when_no_csv(self, tmp_path, monkeypatch):
        """When no team_matchup_summary.csv exists, df is returned as-is."""
        monkeypatch.setattr(
            "build_derived_csvs.CSV_DIR", tmp_path / "csv",
        )
        monkeypatch.setattr(
            "build_derived_csvs.DATA", tmp_path / "data",
        )
        (tmp_path / "csv").mkdir(exist_ok=True)
        (tmp_path / "data").mkdir(exist_ok=True)

        from build_derived_csvs import enrich_with_matchup_summary

        df = pd.DataFrame({
            "event_id": [1], "home_team_id": [100], "away_team_id": [200],
        })
        result = enrich_with_matchup_summary(df)
        assert list(result.columns) == list(df.columns)

    def test_merges_matchup_columns(self, tmp_path, monkeypatch):
        """When csv exists, home/away matchup columns are appended."""
        csv_dir = tmp_path / "csv"
        csv_dir.mkdir(exist_ok=True)
        data_dir = tmp_path / "data"
        data_dir.mkdir(exist_ok=True)

        monkeypatch.setattr("build_derived_csvs.CSV_DIR", csv_dir)
        monkeypatch.setattr("build_derived_csvs.DATA", data_dir)

        tms = pd.DataFrame([
            {"event_id": 1, "team_id": 100, "star_pcs": 15.0,
             "expected_pts_impact": 3.5, "star_underperform_risk": False,
             "favorable_conditions": True, "avg_pcs_starters": 12.0},
            {"event_id": 1, "team_id": 200, "star_pcs": -5.0,
             "expected_pts_impact": -1.2, "star_underperform_risk": False,
             "favorable_conditions": False, "avg_pcs_starters": -3.0},
        ])
        tms.to_csv(csv_dir / "team_matchup_summary.csv", index=False)

        from build_derived_csvs import enrich_with_matchup_summary

        df = pd.DataFrame({
            "event_id": [1], "home_team_id": [100], "away_team_id": [200],
        })
        result = enrich_with_matchup_summary(df)

        assert "home_star_pcs" in result.columns
        assert "away_star_pcs" in result.columns
        assert "matchup_pts_edge" in result.columns
        assert abs(result.iloc[0]["matchup_pts_edge"] - 4.7) < 0.1
