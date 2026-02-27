import pandas as pd

import optimize_weights
from optimize_weights import _score_combo_walk_forward, run_search


def _combo() -> dict:
    return {
        "efg_weight": 0.2,
        "tov_weight": 0.2,
        "orb_weight": 0.2,
        "drb_weight": 0.2,
        "ftr_weight": 0.2,
        "three_par_weight": 0.0,
        "vs_exp_weight": 0.7,
        "eff_composite_weight": 0.6,
        "schedule_adjustment_factor": 0.5,
        "adj_pace_weight": 0.0,
        "pace_regression_factor": 0.0,
        "cage_prior_weight": 0.0,
        "decay_floor": 0.5,
        "decay_cliff": 0.75,
        "raw_weight": 0.3,
    }


def _df_with_weeks() -> pd.DataFrame:
    rows = []
    for week in range(8):
        for game in range(2):
            rows.append(
                {
                    "game_datetime_utc": pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=7 * week + game),
                    "model_efg_delta": 0.5,
                    "model_tov_delta": 0.2,
                    "model_orb_delta": 0.1,
                    "model_drb_delta": 0.3,
                    "model_ftr_delta": 0.4,
                    "model_tpar_delta": 0.0,
                    "_eff_edge": 1.0,
                    "_closing_line": -2.0,
                }
            )
    return pd.DataFrame(rows)


def test_score_combo_walk_forward_returns_folds_and_samples():
    score, n_games, folds = _score_combo_walk_forward(
        _df_with_weeks(),
        _combo(),
        use_fast_clv=True,
        min_train_weeks=4,
        test_window_weeks=1,
    )

    assert score != float("-inf")
    assert n_games > 0
    assert folds == 4


def test_run_search_robust_cv_reports_fold_count(monkeypatch):
    prior = _combo()

    monkeypatch.setattr(
        optimize_weights,
        "_build_search_grid",
        lambda _: {key: [prior[key]] for key in optimize_weights.SEARCH_PARAM_KEYS},
    )

    best, search_space_size, _ = run_search(
        _df_with_weeks(),
        prior,
        use_fast_clv=True,
        robust_cv=True,
        min_train_weeks=4,
        test_window_weeks=1,
        min_folds=3,
    )

    assert search_space_size == 1
    assert best["fold_count"] >= 3
