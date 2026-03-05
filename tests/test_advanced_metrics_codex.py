from datetime import date

from cbb_advanced_metrics_codex import (
    AdvancedMetricsCodex,
    GameInputs,
    ModelConfig,
    compute_odi_star,
    compute_pei,
    compute_posw,
)


def _sample_inputs() -> GameInputs:
    home_stats = {
        "efg_home_adj": 0.54,
        "tov_home_adj": 0.16,
        "orb_home_adj": 0.33,
        "ftr_home_adj": 0.32,
        "pace_home": 70.0,
        "ortg_home_adj": 112.0,
        "returning_minutes": 0.68,
        "rotation_stability": 0.72,
        "star_continuity": 0.70,
        "clutch_pct": 0.64,
        "bench_minutes": 0.34,
        "bench_ts_rel": 0.05,
        "bench_reb_rel": 0.04,
        "best_rotation_netrtg": 14.0,
        "worst_rotation_netrtg": -2.0,
        "backup_quality_key_positions": 0.80,
        "ffc_inverted": 0.20,
        "star_three_pa_rate": 0.38,
        "star_rim_rate": 0.34,
        "star_midrange_rate": 0.18,
        "star_post_ups": 0.10,
        "lineup_height_avg": 78.8,
        "three_pa_rate_home": 0.40,
        "ftm_home": 14.0,
        "fga_home": 58.0,
        "netrtg_adj_l8": [12, 9, 14, 11, 8, 10, 13, 7],
        "game_pace_l12": [69, 71, 68, 70, 72, 69, 70, 71, 68, 69, 70, 71],
        "team_season_pace_avg": 69.5,
        "ft_ppp_home": 0.23,
        "star_ts": 0.62,
        "team_ts": 0.58,
        "star_usage": 0.31,
    }
    away_stats = {
        "efg_away_adj": 0.51,
        "tov_away_adj": 0.18,
        "orb_away_adj": 0.29,
        "ftr_away_adj": 0.27,
        "pace_away": 67.0,
        "ortg_away_adj": 106.0,
        "returning_minutes": 0.57,
        "rotation_stability": 0.60,
        "star_continuity": 0.58,
        "clutch_pct": 0.52,
        "bench_minutes": 0.30,
        "bench_ts_rel": -0.02,
        "bench_reb_rel": -0.01,
        "best_rotation_netrtg": 8.0,
        "worst_rotation_netrtg": -7.0,
        "backup_quality_key_positions": 0.68,
        "ffc_inverted": 0.12,
        "star_three_pa_rate": 0.33,
        "star_rim_rate": 0.30,
        "star_midrange_rate": 0.25,
        "star_post_ups": 0.12,
        "lineup_height_avg": 78.1,
        "three_pa_rate_away": 0.35,
        "ftm_away": 11.0,
        "fga_away": 56.0,
        "netrtg_adj_l8": [3, -1, 5, 2, 0, 4, -2, 1],
        "game_pace_l12": [66, 67, 68, 65, 66, 67, 68, 66, 67, 65, 66, 67],
        "team_season_pace_avg": 66.5,
        "ft_ppp_away": 0.19,
        "star_ts": 0.57,
        "team_ts": 0.55,
        "star_usage": 0.28,
    }
    home_rotation = [
        {"minutes_share": 0.23, "tois": 1.4, "usage": 0.31, "ts": 0.62},
        {"minutes_share": 0.22, "tois": 1.2, "usage": 0.26, "ts": 0.60},
        {"minutes_share": 0.20, "tois": 1.0, "usage": 0.21, "ts": 0.58},
    ]
    away_rotation = [
        {"minutes_share": 0.24, "tois": 1.1, "usage": 0.29, "ts": 0.57},
        {"minutes_share": 0.21, "tois": 1.0, "usage": 0.24, "ts": 0.56},
        {"minutes_share": 0.19, "tois": 0.9, "usage": 0.20, "ts": 0.54},
    ]
    return GameInputs(
        home_team="Florida",
        away_team="Duke",
        game_date=date(2026, 3, 5),
        closing_spread=-3.5,
        closing_total=139.5,
        closing_ml_home=-150,
        closing_ml_away=130,
        home_team_stats=home_stats,
        away_team_stats=away_stats,
        home_rotation=home_rotation,
        away_rotation=away_rotation,
        days_rest_home=3,
        days_rest_away=1,
        back2back_home=False,
        back2back_away=False,
        elevation_home=184.0,
        elevation_away=540.0,
        cross_country_miles=900.0,
    )


def test_core_metric_math():
    odi = compute_odi_star(
        {"efg": 0.54, "tov": 0.16, "orb": 0.33, "ftr": 0.32},
        {"efg": 0.49, "tov": 0.19, "orb": 0.28, "ftr": 0.29},
        {"efg": 0.50, "tov": 0.18, "orb": 0.30, "ftr": 0.30},
    )
    pei = compute_pei(orb_off_adj=0.33, tov_off_adj=0.16, orb_def_adj=0.28, tov_def_adj=0.19)
    posw = compute_posw(pei=pei, pace_team=70.0, pace_opp=67.0)

    assert round(odi, 4) == 0.0420
    assert round(pei, 4) == 0.0800
    assert round(posw, 4) == 0.0548


def test_joint_model_reconciliation_identity():
    model = AdvancedMetricsCodex()
    out = model.predict_game(_sample_inputs())
    assert round(out["pred_home_score"] + out["pred_away_score"], 1) == out["pred_total"]
    assert round(out["pred_home_score"] - out["pred_away_score"], 1) == out["pred_margin"]


def test_allocation_is_clamped():
    cfg = ModelConfig(allocation_scale=2.0, allocation_cap=0.20)
    model = AdvancedMetricsCodex(config=cfg)
    out = model.predict_game(_sample_inputs())
    assert abs(out["allocation_pct"]) <= 0.20


def test_recommendation_thresholds():
    cfg = ModelConfig(spread_edge_threshold=1.5, total_edge_threshold=2.0, ml_edge_threshold=0.05)
    model = AdvancedMetricsCodex(config=cfg)
    recs = model.generate_bet_recommendations(
        spread_edge=2.1,
        total_edge=-2.5,
        ml_edge_home=0.07,
        closing_spread=-3.5,
        closing_total=139.5,
    )
    joined = " | ".join(recs)
    assert "SPREAD" in joined
    assert "TOTAL" in joined
    assert "ML" in joined
