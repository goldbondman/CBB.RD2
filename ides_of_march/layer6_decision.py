from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .utils import logistic, margin_to_home_spread


DIRECT_WIN_FEATURES = [
    "adj_em_margin_l12",
    "efg_margin_l5",
    "to_margin_l5",
    "oreb_margin_l5",
    "ftr_margin_l5",
    "form_delta_diff",
    "rest_diff",
    "sos_diff",
]


@dataclass
class DirectWinModel:
    model: Pipeline
    calibrator: IsotonicRegression | None
    features: list[str]


def fit_direct_win_model(history_df: pd.DataFrame) -> DirectWinModel | None:
    if history_df.empty or "home_won" not in history_df.columns:
        return None

    features = [f for f in DIRECT_WIN_FEATURES if f in history_df.columns]
    if len(features) < 4:
        return None

    y = pd.to_numeric(history_df["home_won"], errors="coerce")
    x = history_df[features].apply(pd.to_numeric, errors="coerce")
    mask = y.notna() & x.notna().all(axis=1)
    if mask.sum() < 200 or y[mask].nunique() < 2:
        return None

    x_train = x[mask]
    y_train = y[mask].astype(int)

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=500, C=1.0)),
        ]
    )
    pipe.fit(x_train, y_train)

    p_raw = pipe.predict_proba(x_train)[:, 1]
    calibrator: IsotonicRegression | None = None
    if len(np.unique(p_raw)) >= 8 and len(y_train) >= 300:
        try:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            calibrator.fit(p_raw, y_train.to_numpy(dtype=float))
        except Exception:
            calibrator = None

    return DirectWinModel(model=pipe, calibrator=calibrator, features=features)


def _predict_direct_win(model: DirectWinModel | None, frame: pd.DataFrame) -> pd.Series:
    if model is None:
        return pd.Series(np.nan, index=frame.index)

    x = frame.reindex(columns=model.features).apply(pd.to_numeric, errors="coerce")
    x = x.fillna(0.0)
    p = model.model.predict_proba(x)[:, 1]
    if model.calibrator is not None:
        p = model.calibrator.predict(p)
    return pd.Series(np.clip(p, 0.01, 0.99), index=frame.index)


def apply_decision_layer(
    game_frame: pd.DataFrame,
    *,
    direct_win_model: DirectWinModel | None = None,
    mc_mode: str = "confidence_only",
) -> pd.DataFrame:
    out = game_frame.copy()

    out["projected_margin_home"] = pd.to_numeric(out.get("projected_margin_pre_mc"), errors="coerce").fillna(0.0)
    out["projected_spread"] = margin_to_home_spread(out["projected_margin_home"])

    out["win_prob_home_a"] = logistic(out["projected_margin_home"], scale=6.0)
    out["win_prob_home_b"] = _predict_direct_win(direct_win_model, out)

    out["win_prob_home"] = out["win_prob_home_a"]
    has_direct = out["win_prob_home_b"].notna()
    out.loc[has_direct, "win_prob_home"] = 0.5 * out.loc[has_direct, "win_prob_home_a"] + 0.5 * out.loc[has_direct, "win_prob_home_b"]

    if mc_mode == "blended" and "mc_home_win_prob" in out.columns:
        mc_p = pd.to_numeric(out.get("mc_home_win_prob"), errors="coerce")
        out["win_prob_home"] = 0.6 * out["win_prob_home"] + 0.4 * mc_p.fillna(out["win_prob_home"])

    out["win_prob_home"] = pd.to_numeric(out["win_prob_home"], errors="coerce").clip(0.01, 0.99)

    out["edge_home"] = pd.to_numeric(out.get("market_spread"), errors="coerce") - pd.to_numeric(out["projected_spread"], errors="coerce")
    out["ats_cover_prob_home"] = logistic(out["edge_home"], scale=4.5)

    if mc_mode == "blended" and "mc_home_cover_prob" in out.columns:
        mc_cover = pd.to_numeric(out.get("mc_home_cover_prob"), errors="coerce")
        out["ats_cover_prob_home"] = 0.6 * out["ats_cover_prob_home"] + 0.4 * mc_cover.fillna(out["ats_cover_prob_home"])

    if "mc_home_cover_prob" in out.columns and out["mc_home_cover_prob"].notna().any():
        out["ats_cover_prob_home"] = 0.7 * out["ats_cover_prob_home"] + 0.3 * pd.to_numeric(out["mc_home_cover_prob"], errors="coerce").fillna(out["ats_cover_prob_home"])

    out["ats_cover_prob_home"] = pd.to_numeric(out["ats_cover_prob_home"], errors="coerce").clip(0.01, 0.99)

    edge_abs = pd.to_numeric(out["edge_home"], errors="coerce").abs().fillna(0.0)
    stability = pd.to_numeric(out.get("base_model_stability"), errors="coerce").fillna(0.5)
    volatility = pd.to_numeric(out.get("mc_volatility"), errors="coerce").fillna(12.0)
    situ_boost = pd.to_numeric(out.get("situational_confidence_boost"), errors="coerce").fillna(0.0)
    win_strength = (pd.to_numeric(out["win_prob_home"], errors="coerce") - 0.5).abs()

    base_component = np.clip(edge_abs / 6.0, 0.0, 1.0) * 52.0
    stability_bonus = np.clip(stability, 0.0, 1.0) * 12.0
    variance_penalty = np.clip((volatility - 8.0) / 10.0, 0.0, 1.0) * 18.0
    win_bonus = np.clip(win_strength / 0.25, 0.0, 1.0) * 16.0

    out["confidence_score"] = (base_component + stability_bonus - variance_penalty + situ_boost + win_bonus).clip(0.0, 100.0)

    out["predicted_winner_side"] = np.where(out["win_prob_home"] >= 0.5, "HOME", "AWAY")
    out["predicted_ats_side"] = np.where(out["ats_cover_prob_home"] >= 0.5, "HOME", "AWAY")

    out["bet_recommendation"] = "PASS"
    recommend_mask = (
        edge_abs >= 1.5
    ) & (
        pd.to_numeric(out["confidence_score"], errors="coerce") >= 55.0
    ) & (
        pd.to_numeric(out["ats_cover_prob_home"], errors="coerce").sub(0.5).abs() >= 0.03
    )

    if mc_mode == "confidence_filter" and "mc_filter_pass" in out.columns:
        recommend_mask = recommend_mask & out["mc_filter_pass"].astype(bool)

    home_mask = recommend_mask & out["predicted_ats_side"].eq("HOME")
    away_mask = recommend_mask & out["predicted_ats_side"].eq("AWAY")
    out.loc[home_mask, "bet_recommendation"] = "HOME_SPREAD"
    out.loc[away_mask, "bet_recommendation"] = "AWAY_SPREAD"

    out["model_prob_source"] = np.where(has_direct, "blend_logistic_direct", "logistic_margin")
    return out


def build_bet_recs_frame(predictions: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "game_id",
        "event_id",
        "game_datetime_utc",
        "home_team",
        "away_team",
        "home_team_id",
        "away_team_id",
        "market_spread",
        "market_total",
        "projected_spread",
        "projected_margin_home",
        "edge_home",
        "win_prob_home",
        "ats_cover_prob_home",
        "confidence_score",
        "bet_recommendation",
        "line_source_used",
    ]
    out = predictions.reindex(columns=cols).copy()
    out = out[out["bet_recommendation"].astype(str) != "PASS"].copy()
    out = out.sort_values(["confidence_score", "edge_home"], ascending=[False, False], kind="mergesort")
    return out.reset_index(drop=True)
