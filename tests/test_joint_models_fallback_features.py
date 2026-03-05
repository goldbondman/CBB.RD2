import numpy as np

from model_lab import joint_models as jm


def test_joint_fallback_features_use_l10_columns():
    home = {
        "efg_pct_l10": 44.0,
        "tov_pct_l10": 17.0,
        "orb_pct_l10": 29.0,
        "ftr_l10": 28.0,
        "pace_l10": 70.5,
        "poss_l10": 70.5,
        "three_par_l10": 36.0,
        "ftm": 14.0,
        "fga": 55.0,
        "form_rating": -2.0,
        "net_rtg_std_l10": 8.0,
        "rest_days": 2.0,
    }
    away = {
        "efg_pct_l10": 52.0,
        "tov_pct_l10": 14.0,
        "orb_pct_l10": 31.0,
        "ftr_l10": 32.0,
        "pace_l10": 68.0,
        "poss_l10": 68.0,
        "three_par_l10": 33.0,
        "ftm": 16.0,
        "fga": 57.0,
        "form_rating": 3.5,
        "net_rtg_std_l10": 6.0,
        "rest_days": 3.0,
    }
    team_rows = {"home": home, "away": away}

    totals = jm.compute_totals_features({"game_id": "1"}, team_rows, {})
    spread = jm.compute_spread_features({"game_id": "1"}, team_rows, {})

    for key in ["posw", "sch", "wl", "tc_diff"]:
        assert np.isfinite(float(totals[key]))
    for key in ["odi_star_diff", "posw", "lns_diff", "vol_diff"]:
        assert np.isfinite(float(spread[key]))

