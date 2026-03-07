#!/usr/bin/env python3
"""Agent 4: walk-forward train/validate models with Gate 4 enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

SPREAD_EDGE_MIN = 2.0
TOTAL_EDGE_MIN = 2.5


def _fit_binary_classifier(x_train: np.ndarray, y_train: pd.Series):
    classes = pd.Series(y_train).dropna().unique()
    if len(classes) < 2:
        constant = int(classes[0]) if len(classes) == 1 else 0
        clf = DummyClassifier(strategy="constant", constant=constant)
        clf.fit(x_train, np.full(len(x_train), constant))
        return clf
    clf = LogisticRegression(C=1.0, max_iter=500)
    clf.fit(x_train, y_train.astype(int))
    return clf


def _hit_rate(actual_col: str, pred_col: str, df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float((df[actual_col] == df[pred_col]).mean())


def main() -> int:
    df = pd.read_csv("data/internal/matchup_features.csv", low_memory=False)
    if df.empty:
        print("[STOP] matchup_features.csv is empty")
        return 1

    df["game_date"] = pd.to_datetime(df.get("game_date"), errors="coerce", utc=True)
    df = df.sort_values("game_date").reset_index(drop=True)
    df["is_upcoming"] = df["is_upcoming"].astype(str).str.lower().isin({"true", "1"})

    spread_features = [
        "netrtg_L11_diff",
        "netrtg_trend_diff",
        "odi_star_diff",
        "efg_matchup_diff",
        "pei_L6_matchup",
        "away_efg_L11_diff",
        "vol_L6_diff",
        "home_dummy",
    ]
    total_features = [
        "pace_L11_avg",
        "posw_L6",
        "sch_matchup",
        "wl_L11_sum",
        "efg_L6_avg",
        "tc_proxy_diff",
        "vol_L6_sum",
    ]
    spread_features = [f for f in spread_features if f in df.columns]
    total_features = [f for f in total_features if f in df.columns]

    completed = df[df["is_upcoming"] == False].copy()
    for col in ["actual_margin", "actual_total", "home_score", "away_score", "closing_spread", "closing_total"]:
        if col in completed.columns:
            completed[col] = pd.to_numeric(completed[col], errors="coerce")

    margin_col = None
    for c in ["actual_margin", "home_margin", "margin"]:
        if c in completed.columns and completed[c].notna().any():
            margin_col = c
            break
    total_col = None
    for c in ["actual_total", "final_total", "total_score"]:
        if c in completed.columns and completed[c].notna().any():
            total_col = c
            break

    if margin_col is None and {"home_score", "away_score"}.issubset(completed.columns):
        completed["_margin"] = completed["home_score"] - completed["away_score"]
        margin_col = "_margin"
    if total_col is None and {"home_score", "away_score"}.issubset(completed.columns):
        completed["_total"] = completed["home_score"] + completed["away_score"]
        total_col = "_total"

    if margin_col is None or total_col is None:
        print("[STOP] Could not identify actual margin/total columns")
        return 1

    completed = completed.dropna(subset=[margin_col, total_col, "closing_spread", "closing_total"])
    completed = completed.dropna(subset=spread_features + total_features, how="all")
    if len(completed) < 100:
        print(f"[STOP] Insufficient completed games for validation: {len(completed)}")
        return 1

    min_train = max(100, len(completed) // 3)
    if min_train >= len(completed):
        print(f"[STOP] Not enough rows after min_train split: rows={len(completed)} min_train={min_train}")
        return 1

    spread_preds: list[float] = []
    total_preds: list[float] = []
    spread_probs: list[float] = []
    total_probs: list[float] = []
    spread_actuals: list[float] = []
    total_actuals: list[float] = []

    for i in range(min_train, len(completed)):
        train = completed.iloc[:i]
        test = completed.iloc[i : i + 1]
        x_tr_s = train[spread_features].fillna(0.0)
        x_te_s = test[spread_features].fillna(0.0)
        x_tr_t = train[total_features].fillna(0.0)
        x_te_t = test[total_features].fillna(0.0)

        y_margin = train[margin_col].astype(float)
        y_total = train[total_col].astype(float)
        y_cover = (y_margin > train["closing_spread"].astype(float)).astype(int)
        y_over = (y_total > train["closing_total"].astype(float)).astype(int)

        sc_s = StandardScaler()
        sc_t = StandardScaler()
        x_tr_s_sc = sc_s.fit_transform(x_tr_s)
        x_te_s_sc = sc_s.transform(x_te_s)
        x_tr_t_sc = sc_t.fit_transform(x_tr_t)
        x_te_t_sc = sc_t.transform(x_te_t)

        ridge_s = Ridge(alpha=1.0).fit(x_tr_s_sc, y_margin)
        ridge_t = Ridge(alpha=1.0).fit(x_tr_t_sc, y_total)
        logit_s = _fit_binary_classifier(x_tr_s_sc, y_cover)
        logit_t = _fit_binary_classifier(x_tr_t_sc, y_over)

        spread_preds.append(float(ridge_s.predict(x_te_s_sc)[0]))
        total_preds.append(float(ridge_t.predict(x_te_t_sc)[0]))
        spread_probs.append(float(logit_s.predict_proba(x_te_s_sc)[0][1]))
        total_probs.append(float(logit_t.predict_proba(x_te_t_sc)[0][1]))
        spread_actuals.append(float(test[margin_col].iloc[0]))
        total_actuals.append(float(test[total_col].iloc[0]))

    val = completed.iloc[min_train:].copy()
    val["pred_margin"] = spread_preds
    val["pred_total"] = total_preds
    val["spread_prob"] = spread_probs
    val["total_prob"] = total_probs
    val["cover"] = (val[margin_col] > val["closing_spread"]).astype(int)
    val["pred_cover"] = (val["pred_margin"] > val["closing_spread"]).astype(int)
    val["over"] = (val[total_col] > val["closing_total"]).astype(int)
    val["pred_over"] = (val["pred_total"] > val["closing_total"]).astype(int)
    val["spread_edge"] = val["pred_margin"] - val["closing_spread"]
    val["total_edge"] = val["pred_total"] - val["closing_total"]

    fired_spread = val[val["spread_edge"].abs() >= SPREAD_EDGE_MIN].copy()
    fired_total = val[val["total_edge"].abs() >= TOTAL_EDGE_MIN].copy()

    spread_hr = _hit_rate("cover", "pred_cover", val)
    spread_hr_last100 = _hit_rate("cover", "pred_cover", val.iloc[-100:])
    spread_hr_last50 = _hit_rate("cover", "pred_cover", val.iloc[-50:])
    spread_fire_hr = _hit_rate("cover", "pred_cover", fired_spread) if len(fired_spread) else float("nan")

    total_hr = _hit_rate("over", "pred_over", val)
    total_hr_last100 = _hit_rate("over", "pred_over", val.iloc[-100:])
    total_fire_hr = _hit_rate("over", "pred_over", fired_total) if len(fired_total) else float("nan")

    spread_mae = float(mean_absolute_error(val[margin_col], val["pred_margin"]))
    total_mae = float(mean_absolute_error(val[total_col], val["pred_total"]))

    print("=" * 55)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("=" * 55)
    print(f"SPREAD MODEL ({len(val)} validation games)")
    print(f"  Overall ATS hit rate:       {spread_hr:.1%}")
    print(f"  Last 100 games:             {spread_hr_last100:.1%}")
    print(f"  Last 50 games:              {spread_hr_last50:.1%}")
    print(f"  Fire rate (edge >= 2pt):    {len(fired_spread)/len(val):.1%}  (N={len(fired_spread)})")
    if len(fired_spread):
        print(f"  Hit rate at fire threshold: {spread_fire_hr:.1%}")
    print(f"  MAE vs closing spread:      {spread_mae:.2f} pts")
    print()
    print(f"TOTALS MODEL ({len(val)} validation games)")
    print(f"  Overall O/U hit rate:       {total_hr:.1%}")
    print(f"  Last 100 games:             {total_hr_last100:.1%}")
    print(f"  Fire rate (edge >= 2.5pt):  {len(fired_total)/len(val):.1%}  (N={len(fired_total)})")
    if len(fired_total):
        print(f"  Hit rate at fire threshold: {total_fire_hr:.1%}")
    print(f"  MAE vs closing total:       {total_mae:.2f} pts")

    # Gate thresholds before final artifact write.
    if len(val) < 100:
        print(f"[STOP] Gate 4 failed: validation_games={len(val)} < 100")
        return 1
    if not (spread_hr > 0.50):
        print(f"[STOP] Gate 4 failed: spread_hit_rate={spread_hr:.1%} <= 50%")
        return 1

    x_all_s = completed[spread_features].fillna(0.0)
    x_all_t = completed[total_features].fillna(0.0)
    y_margin_all = completed[margin_col].astype(float)
    y_total_all = completed[total_col].astype(float)
    y_cover_all = (y_margin_all > completed["closing_spread"].astype(float)).astype(int)
    y_over_all = (y_total_all > completed["closing_total"].astype(float)).astype(int)

    sc_s_final = StandardScaler()
    sc_t_final = StandardScaler()
    x_s_sc = sc_s_final.fit_transform(x_all_s)
    x_t_sc = sc_t_final.fit_transform(x_all_t)

    final_ridge_s = Ridge(alpha=1.0).fit(x_s_sc, y_margin_all)
    final_ridge_t = Ridge(alpha=1.0).fit(x_t_sc, y_total_all)
    final_logit_s = _fit_binary_classifier(x_s_sc, y_cover_all)
    final_logit_t = _fit_binary_classifier(x_t_sc, y_over_all)

    print("\nSPREAD FEATURE COEFFICIENTS (standardized):")
    for feat, coef in sorted(zip(spread_features, final_ridge_s.coef_), key=lambda x: -abs(x[1])):
        flag = " <- TREND" if "trend" in feat else ""
        print(f"  {feat:<28} {coef:+.3f}{flag}")

    print("\nTOTALS FEATURE COEFFICIENTS (standardized):")
    for feat, coef in sorted(zip(total_features, final_ridge_t.coef_), key=lambda x: -abs(x[1])):
        print(f"  {feat:<28} {coef:+.3f}")

    low_signal_spread = [f for f, c in zip(spread_features, final_ridge_s.coef_) if abs(float(c)) < 0.05]
    if low_signal_spread:
        print(f"\n[NOTE] Low-signal spread features (|coef| < 0.05): {low_signal_spread}")

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_ridge_s, model_dir / "spread_ridge.pkl")
    joblib.dump(final_ridge_t, model_dir / "total_ridge.pkl")
    joblib.dump(final_logit_s, model_dir / "spread_logit.pkl")
    joblib.dump(final_logit_t, model_dir / "total_logit.pkl")
    joblib.dump(sc_s_final, model_dir / "spread_scaler.pkl")
    joblib.dump(sc_t_final, model_dir / "total_scaler.pkl")
    joblib.dump({"spread": spread_features, "total": total_features}, model_dir / "feature_lists.pkl")

    report = {
        "validation_games": int(len(val)),
        "spread_hit_rate": spread_hr,
        "spread_hit_rate_last_100": spread_hr_last100,
        "spread_hit_rate_last_50": spread_hr_last50,
        "spread_mae": spread_mae,
        "total_hit_rate": total_hr,
        "total_hit_rate_last_100": total_hr_last100,
        "total_mae": total_mae,
        "spread_fire_rate": len(fired_spread) / len(val),
        "total_fire_rate": len(fired_total) / len(val),
        "low_signal_spread_features": low_signal_spread,
    }
    out_dir = Path("data/internal")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    required_models = [
        model_dir / "spread_ridge.pkl",
        model_dir / "spread_logit.pkl",
        model_dir / "total_ridge.pkl",
        model_dir / "total_logit.pkl",
        model_dir / "spread_scaler.pkl",
        model_dir / "total_scaler.pkl",
    ]
    gate = {
        "all_models_saved": all(p.exists() for p in required_models),
        "feature_lists_saved": (model_dir / "feature_lists.pkl").exists(),
        "validation_games_ge_100": len(val) >= 100,
        "spread_hr_gt_50": spread_hr > 0.50,
    }
    print("\n=== GATE_4 RESULTS ===")
    for check, result in gate.items():
        print(f"  {'PASS' if result else 'FAIL'}  {check}")
    if not all(gate.values()):
        print("[STOP] Gate 4 failed")
        return 1

    print("[OK] Gate 4 passed. Models saved under models/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
