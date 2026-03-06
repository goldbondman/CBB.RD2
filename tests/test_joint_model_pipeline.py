import numpy as np

from model_lab import joint_models as jm


def test_joint_model_arithmetic_and_weights(monkeypatch):
    monkeypatch.setattr(
        jm,
        "compute_totals_features",
        lambda game_row, team_rows, advanced_metrics=None: {
            "posw": 2.0,
            "sch": 4.0,
            "wl": 1.0,
            "svi_avg": 0.0,
            "tc_diff": -1.0,
            "posw_sum": 2.0,
            "wl_sum": 1.0,
            "rfd_sum": 0.0,
            "alt_diff": 0.0,
        },
    )
    monkeypatch.setattr(
        jm,
        "compute_spread_features",
        lambda game_row, team_rows, advanced_metrics=None: {
            "odi_star_diff": 0.10,
            "sme_diff": 0.05,
            "posw": 0.02,
            "pxp_diff": 0.03,
            "lns_diff": 0.04,
            "vol_diff": -0.01,
            "away_efg_diff": 0.0,
            "rfd": 0.0,
        },
    )
    monkeypatch.setattr(jm, "_base_total", lambda team_rows: 140.0)

    out = jm.predict_game_joint(
        game_row={"game_id": "1"},
        team_rows={"home": {"team_id": "100"}, "away": {"team_id": "200"}},
        market_row={"spread_line": -3.5, "total_line": 145.5},
    )

    assert out["model_status"] == "OK"
    assert round(out["pred_total"], 1) == 148.1
    assert round(out["allocation_pct"], 4) == 0.0207
    assert round(out["pred_home_score"], 4) == round(148.1 * (0.5 + out["allocation_pct"]), 4)
    assert round(out["pred_away_score"], 4) == round(148.1 * (0.5 - out["allocation_pct"]), 4)
    assert round(out["pred_margin"], 4) == round(out["pred_home_score"] - out["pred_away_score"], 4)
    assert np.isfinite(out["spread_edge"])
    assert np.isfinite(out["total_edge"])
    assert round(out["pred_margin_raw"], 4) == round(out["pred_margin_final"], 4)
    assert isinstance(out.get("spread_calc_trace"), dict)


def test_joint_model_missing_inputs_blocked(monkeypatch):
    monkeypatch.setattr(
        jm,
        "compute_totals_features",
        lambda game_row, team_rows, advanced_metrics=None: {
            "posw": np.nan,
            "sch": np.nan,
            "wl": np.nan,
            "svi_avg": np.nan,
            "tc_diff": np.nan,
            "posw_sum": np.nan,
            "wl_sum": np.nan,
            "rfd_sum": np.nan,
            "alt_diff": np.nan,
        },
    )
    monkeypatch.setattr(
        jm,
        "compute_spread_features",
        lambda game_row, team_rows, advanced_metrics=None: {
            "odi_star_diff": np.nan,
            "sme_diff": np.nan,
            "posw": np.nan,
            "pxp_diff": np.nan,
            "lns_diff": np.nan,
            "vol_diff": np.nan,
            "away_efg_diff": np.nan,
            "rfd": np.nan,
        },
    )

    out = jm.predict_game_joint(
        game_row={"game_id": "1"},
        team_rows={"home": {"team_id": "100"}, "away": {"team_id": "200"}},
        market_row=None,
    )

    assert str(out["model_status"]).startswith("BLOCKED_MISSING_INPUT")
    assert np.isnan(out["pred_total"])
    assert np.isnan(out["pred_margin"])


def test_spread_calc_trace_matches_raw_margin(monkeypatch):
    monkeypatch.setattr(
        jm,
        "compute_totals_features",
        lambda game_row, team_rows, advanced_metrics=None: {
            "posw": 2.0,
            "sch": 4.0,
            "wl": 1.0,
            "svi_avg": 0.0,
            "tc_diff": -1.0,
            "posw_sum": 2.0,
            "wl_sum": 1.0,
            "rfd_sum": 0.0,
            "alt_diff": 0.0,
        },
    )
    monkeypatch.setattr(
        jm,
        "compute_spread_features",
        lambda game_row, team_rows, advanced_metrics=None: {
            "odi_star_diff": 0.10,
            "sme_diff": 0.05,
            "posw": 0.02,
            "pxp_diff": 0.03,
            "lns_diff": 0.04,
            "vol_diff": -0.01,
            "away_efg_diff": 0.0,
            "rfd": 0.0,
        },
    )
    monkeypatch.setattr(jm, "_base_total", lambda team_rows: 140.0)

    out = jm.predict_game_joint(
        game_row={"event_id": "1", "game_id": "1", "home_team": "H", "away_team": "A"},
        team_rows={"home": {"team_id": "100"}, "away": {"team_id": "200"}},
        market_row={"spread_line": -3.5, "total_line": 145.5},
    )
    trace = out["spread_calc_trace"]
    assert round(float(trace["pred_margin_raw"]), 6) == round(float(out["pred_margin_raw"]), 6)
    assert round(float(trace["pred_margin_final"]), 6) == round(float(out["pred_margin_final"]), 6)
    assert trace["perspective"].startswith("HOME")
    assert len(trace["feature_terms"]) > 0

