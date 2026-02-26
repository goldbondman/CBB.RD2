import pandas as pd

from models.player_matchup_overlay import PlayerOverlayConfig, build_player_overlay_predictions


def _base_row(**overrides):
    row = {
        "season": 2026,
        "game_id": "g1",
        "game_date": "2026-02-20",
        "team_id": 1,
        "opp_team_id": 2,
        "is_home": True,
        "expected_pace": 70,
        "rot_ftrate_l5": 0.32,
        "rot_stocks_per40_l10": 8.0,
        "rot_pf_per40_l5": 6.1,
        "rot_minshare_sd": 0.08,
        "rot_3par_l10": 0.41,
        "rot_3p_pct_sd": 0.06,
        "rot_to_rate_sd": 0.02,
        "top2_pused_share": 0.53,
        "top2_to_rate": 0.14,
        "opp_rot_stocks_per40_l10": 9.0,
        "opp_rot_pf_per40_l5": 7.3,
        "opp_rot_minshare_sd": 0.09,
    }
    row.update(overrides)
    return row


def test_missing_opponent_columns_yield_zero_adjustments():
    df = pd.DataFrame(
        [
            {
                "season": 2026,
                "game_id": "g1",
                "game_date": "2026-02-20",
                "team_id": 1,
                "opp_team_id": 2,
                "is_home": True,
            },
            {
                "season": 2026,
                "game_id": "g1",
                "game_date": "2026-02-20",
                "team_id": 2,
                "opp_team_id": 1,
                "is_home": False,
            },
        ]
    )

    out = build_player_overlay_predictions(df)
    assert (out["off_eff_adj_100"] == 0.0).all()
    assert (out["predicted_total_adjustment_pts"] == 0.0).all()
    assert out["predicted_spread_adjustment_pts_home"].fillna(0.0).eq(0.0).all()


def test_bidirectional_opponent_profile_changes_to_swing_and_efficiency():
    row_a = _base_row(game_id="g1", team_id=1, opp_team_id=2, is_home=True)
    row_b = _base_row(game_id="g2", team_id=1, opp_team_id=3, is_home=True, opp_rot_stocks_per40_l10=14.0)

    away_template = _base_row(team_id=2, opp_team_id=1, is_home=False)

    df = pd.DataFrame(
        [
            row_a,
            away_template,
            row_b,
            {**away_template, "game_id": "g2", "team_id": 3, "opp_team_id": 1},
        ]
    )

    out = build_player_overlay_predictions(df)
    home_rows = out[out["team_id"] == 1].set_index("game_id")

    assert home_rows.loc["g2", "to_swing"] > home_rows.loc["g1", "to_swing"]
    assert home_rows.loc["g2", "off_eff_adj_100"] < home_rows.loc["g1", "off_eff_adj_100"]


def test_game_level_aggregation_sign_convention_and_caps():
    cfg = PlayerOverlayConfig(
        k_ft_mean=30.0,
        k_to_mean=0.0,
        k_exec_mean=0.0,
        max_total_adj_pts=4.0,
        max_spread_adj_pts=2.0,
    )
    df = pd.DataFrame(
        [
            _base_row(game_id="g9", team_id=11, opp_team_id=22, is_home=True, rot_ftrate_l5=1.2, opp_rot_pf_per40_l5=12.0),
            _base_row(game_id="g9", team_id=22, opp_team_id=11, is_home=False, rot_ftrate_l5=0.1, opp_rot_pf_per40_l5=2.0),
        ]
    )

    out = build_player_overlay_predictions(df, cfg=cfg)
    home_row = out[out["is_home"]].iloc[0]

    assert home_row["predicted_total_adjustment_pts"] == 4.0
    assert home_row["predicted_spread_adjustment_pts_home"] == 2.0
    assert home_row["predicted_spread_adjustment_pts_home"] > 0
