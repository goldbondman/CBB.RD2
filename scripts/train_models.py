#!/usr/bin/env python3
"""Agent 4: walk-forward train/validate models with Gate 4 enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

SPREAD_EDGE_MIN = 2.0
TOTAL_EDGE_MIN = 2.5

RIDGE_ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0]
LOGIT_C_GRID = [0.25, 0.5, 1.0, 2.0, 4.0]
RECENCY_DECAY = 0.995
MAX_TUNING_FOLDS = 120
MIN_CALIBRATION_SAMPLES = 50


def _fit_binary_classifier(
    x_train: np.ndarray,
    y_train: pd.Series,
    c_value: float,
    sample_weight: np.ndarray | None = None,
):
    classes = pd.Series(y_train).dropna().unique()
    if len(classes) < 2:
        constant = int(classes[0]) if len(classes) == 1 else 0
        clf = DummyClassifier(strategy="constant", constant=constant)
        clf.fit(x_train, np.full(len(x_train), constant), sample_weight=sample_weight)
        return clf
    clf = LogisticRegression(C=float(c_value), max_iter=500)
    clf.fit(x_train, y_train.astype(int), sample_weight=sample_weight)
    return clf


def _hit_rate(actual_col: str, pred_col: str, df: pd.DataFrame) -> float:
    if df.empty:
        return float("nan")
    return float((df[actual_col] == df[pred_col]).mean())


def _build_recency_weights(n_rows: int, decay: float = RECENCY_DECAY) -> np.ndarray:
    if n_rows <= 0:
        return np.array([], dtype=float)
    # Most recent row gets weight 1.0; older rows are exponentially downweighted.
    ages = np.arange(n_rows - 1, -1, -1, dtype=float)
    weights = np.power(float(decay), ages)
    mean = float(np.mean(weights))
    if mean > 0:
        weights = weights / mean
    return weights


def _build_feature_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    train_num = train_df[base_features].apply(pd.to_numeric, errors="coerce")
    test_num = test_df[base_features].apply(pd.to_numeric, errors="coerce")

    x_train = pd.DataFrame(index=train_df.index)
    x_test = pd.DataFrame(index=test_df.index)
    medians: dict[str, float] = {}

    for feat in base_features:
        tr_col = train_num[feat]
        te_col = test_num[feat]

        median = float(tr_col.median(skipna=True)) if tr_col.notna().any() else 0.0
        if not np.isfinite(median):
            median = 0.0
        medians[feat] = median

        x_train[feat] = tr_col.fillna(median)
        x_test[feat] = te_col.fillna(median)

        miss_col = f"{feat}__missing"
        x_train[miss_col] = tr_col.isna().astype(float)
        x_test[miss_col] = te_col.isna().astype(float)

    return x_train, x_test, medians


def _transform_feature_frame(
    df: pd.DataFrame,
    base_features: list[str],
    medians: dict[str, float],
    expected_cols: list[str],
) -> pd.DataFrame:
    num = df[base_features].apply(pd.to_numeric, errors="coerce")
    out = pd.DataFrame(index=df.index)

    for feat in base_features:
        col = num[feat]
        median = float(medians.get(feat, 0.0))
        if not np.isfinite(median):
            median = 0.0
        out[feat] = col.fillna(median)
        out[f"{feat}__missing"] = col.isna().astype(float)

    # Keep strict training-time column order.
    for col in expected_cols:
        if col not in out.columns:
            out[col] = 0.0
    return out[expected_cols]


def _tuning_split_points(n_rows: int, min_train: int) -> list[int]:
    n_eval = n_rows - min_train
    if n_eval <= 0:
        return []
    if n_eval <= MAX_TUNING_FOLDS:
        return list(range(min_train, n_rows))
    idx = np.linspace(min_train, n_rows - 1, num=MAX_TUNING_FOLDS, dtype=int)
    return sorted(set(int(i) for i in idx))


def _tune_ridge_alpha(
    completed: pd.DataFrame,
    base_features: list[str],
    target_col: str,
    min_train: int,
) -> float:
    split_points = _tuning_split_points(len(completed), min_train)
    best_alpha = 1.0
    best_mae = float("inf")

    for alpha in RIDGE_ALPHA_GRID:
        errors: list[float] = []
        for i in split_points:
            train = completed.iloc[:i]
            test = completed.iloc[i : i + 1]

            x_tr, x_te, _ = _build_feature_frames(train, test, base_features)
            sc = StandardScaler()
            x_tr_sc = sc.fit_transform(x_tr)
            x_te_sc = sc.transform(x_te)

            y_tr = train[target_col].astype(float)
            weights = _build_recency_weights(len(y_tr))

            model = Ridge(alpha=float(alpha))
            model.fit(x_tr_sc, y_tr, sample_weight=weights)

            pred = float(model.predict(x_te_sc)[0])
            actual = float(test[target_col].iloc[0])
            errors.append(abs(pred - actual))

        mae = float(np.mean(errors)) if errors else float("inf")
        if mae < best_mae:
            best_mae = mae
            best_alpha = float(alpha)

    return best_alpha


def _tune_logit_c(
    completed: pd.DataFrame,
    base_features: list[str],
    label_col: str,
    min_train: int,
) -> float:
    split_points = _tuning_split_points(len(completed), min_train)
    best_c = 1.0
    best_brier = float("inf")

    for c_val in LOGIT_C_GRID:
        briers: list[float] = []
        for i in split_points:
            train = completed.iloc[:i]
            test = completed.iloc[i : i + 1]

            x_tr, x_te, _ = _build_feature_frames(train, test, base_features)
            sc = StandardScaler()
            x_tr_sc = sc.fit_transform(x_tr)
            x_te_sc = sc.transform(x_te)

            y_tr = train[label_col].astype(int)
            y_te = int(test[label_col].iloc[0])
            weights = _build_recency_weights(len(y_tr))

            clf = _fit_binary_classifier(x_tr_sc, y_tr, c_value=float(c_val), sample_weight=weights)
            prob = float(clf.predict_proba(x_te_sc)[0][1])
            briers.append((prob - y_te) ** 2)

        brier = float(np.mean(briers)) if briers else float("inf")
        if brier < best_brier:
            best_brier = brier
            best_c = float(c_val)

    return best_c


def _fit_isotonic_calibrator(prob: pd.Series, label: pd.Series) -> IsotonicRegression | None:
    frame = pd.DataFrame({"p": pd.to_numeric(prob, errors="coerce"), "y": pd.to_numeric(label, errors="coerce")}).dropna()
    if len(frame) < MIN_CALIBRATION_SAMPLES:
        return None
    if frame["y"].nunique() < 2:
        return None
    if frame["p"].nunique() < 5:
        return None

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(frame["p"].to_numpy(dtype=float), frame["y"].to_numpy(dtype=float))
    return iso


def _apply_calibrator(model: IsotonicRegression | None, probs: pd.Series) -> np.ndarray:
    arr = pd.to_numeric(probs, errors="coerce").fillna(0.5).to_numpy(dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    if model is None:
        return arr
    out = model.predict(arr)
    return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)


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

    completed["_cover_label"] = (completed[margin_col] > completed["closing_spread"]).astype(int)
    completed["_over_label"] = (completed[total_col] > completed["closing_total"]).astype(int)

    min_train = max(100, len(completed) // 3)
    if min_train >= len(completed):
        print(f"[STOP] Not enough rows after min_train split: rows={len(completed)} min_train={min_train}")
        return 1

    print("[INFO] Tuning hyperparameters with time-series CV...")
    best_alpha_spread = _tune_ridge_alpha(completed, spread_features, margin_col, min_train)
    best_alpha_total = _tune_ridge_alpha(completed, total_features, total_col, min_train)
    best_c_spread = _tune_logit_c(completed, spread_features, "_cover_label", min_train)
    best_c_total = _tune_logit_c(completed, total_features, "_over_label", min_train)
    print(
        "[INFO] tuned params: "
        f"spread_alpha={best_alpha_spread} total_alpha={best_alpha_total} "
        f"spread_c={best_c_spread} total_c={best_c_total}"
    )

    spread_preds: list[float] = []
    total_preds: list[float] = []
    spread_probs: list[float] = []
    total_probs: list[float] = []

    for i in range(min_train, len(completed)):
        train = completed.iloc[:i]
        test = completed.iloc[i : i + 1]

        x_tr_s, x_te_s, _ = _build_feature_frames(train, test, spread_features)
        x_tr_t, x_te_t, _ = _build_feature_frames(train, test, total_features)

        y_margin = train[margin_col].astype(float)
        y_total = train[total_col].astype(float)
        y_cover = train["_cover_label"].astype(int)
        y_over = train["_over_label"].astype(int)

        weights = _build_recency_weights(len(train))

        sc_s = StandardScaler()
        sc_t = StandardScaler()
        x_tr_s_sc = sc_s.fit_transform(x_tr_s)
        x_te_s_sc = sc_s.transform(x_te_s)
        x_tr_t_sc = sc_t.fit_transform(x_tr_t)
        x_te_t_sc = sc_t.transform(x_te_t)

        ridge_s = Ridge(alpha=best_alpha_spread).fit(x_tr_s_sc, y_margin, sample_weight=weights)
        ridge_t = Ridge(alpha=best_alpha_total).fit(x_tr_t_sc, y_total, sample_weight=weights)
        logit_s = _fit_binary_classifier(x_tr_s_sc, y_cover, c_value=best_c_spread, sample_weight=weights)
        logit_t = _fit_binary_classifier(x_tr_t_sc, y_over, c_value=best_c_total, sample_weight=weights)

        spread_preds.append(float(ridge_s.predict(x_te_s_sc)[0]))
        total_preds.append(float(ridge_t.predict(x_te_t_sc)[0]))
        spread_probs.append(float(logit_s.predict_proba(x_te_s_sc)[0][1]))
        total_probs.append(float(logit_t.predict_proba(x_te_t_sc)[0][1]))

    val = completed.iloc[min_train:].copy()
    val["pred_margin"] = spread_preds
    val["pred_total"] = total_preds
    val["spread_prob"] = spread_probs
    val["total_prob"] = total_probs
    val["cover"] = val["_cover_label"].astype(int)
    val["pred_cover"] = (val["pred_margin"] > val["closing_spread"]).astype(int)
    val["over"] = val["_over_label"].astype(int)
    val["pred_over"] = (val["pred_total"] > val["closing_total"]).astype(int)
    val["spread_edge"] = val["pred_margin"] - val["closing_spread"]
    val["total_edge"] = val["pred_total"] - val["closing_total"]

    spread_calibrator = _fit_isotonic_calibrator(val["spread_prob"], val["cover"])
    total_calibrator = _fit_isotonic_calibrator(val["total_prob"], val["over"])
    val["spread_prob_cal"] = _apply_calibrator(spread_calibrator, val["spread_prob"])
    val["total_prob_cal"] = _apply_calibrator(total_calibrator, val["total_prob"])

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
    spread_brier_raw = float(np.mean((val["spread_prob"] - val["cover"]) ** 2))
    spread_brier_cal = float(np.mean((val["spread_prob_cal"] - val["cover"]) ** 2))
    total_brier_raw = float(np.mean((val["total_prob"] - val["over"]) ** 2))
    total_brier_cal = float(np.mean((val["total_prob_cal"] - val["over"]) ** 2))

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
    print(f"  Brier (raw/cal):            {spread_brier_raw:.4f} / {spread_brier_cal:.4f}")
    print()
    print(f"TOTALS MODEL ({len(val)} validation games)")
    print(f"  Overall O/U hit rate:       {total_hr:.1%}")
    print(f"  Last 100 games:             {total_hr_last100:.1%}")
    print(f"  Fire rate (edge >= 2.5pt):  {len(fired_total)/len(val):.1%}  (N={len(fired_total)})")
    if len(fired_total):
        print(f"  Hit rate at fire threshold: {total_fire_hr:.1%}")
    print(f"  MAE vs closing total:       {total_mae:.2f} pts")
    print(f"  Brier (raw/cal):            {total_brier_raw:.4f} / {total_brier_cal:.4f}")

    if len(val) < 100:
        print(f"[STOP] Gate 4 failed: validation_games={len(val)} < 100")
        return 1
    if not (spread_hr > 0.50):
        print(f"[STOP] Gate 4 failed: spread_hit_rate={spread_hr:.1%} <= 50%")
        return 1

    x_all_s_df, _, spread_medians = _build_feature_frames(completed, completed.iloc[0:0], spread_features)
    x_all_t_df, _, total_medians = _build_feature_frames(completed, completed.iloc[0:0], total_features)

    y_margin_all = completed[margin_col].astype(float)
    y_total_all = completed[total_col].astype(float)
    y_cover_all = completed["_cover_label"].astype(int)
    y_over_all = completed["_over_label"].astype(int)
    weights_all = _build_recency_weights(len(completed))

    sc_s_final = StandardScaler()
    sc_t_final = StandardScaler()
    x_s_sc = sc_s_final.fit_transform(x_all_s_df)
    x_t_sc = sc_t_final.fit_transform(x_all_t_df)

    final_ridge_s = Ridge(alpha=best_alpha_spread).fit(x_s_sc, y_margin_all, sample_weight=weights_all)
    final_ridge_t = Ridge(alpha=best_alpha_total).fit(x_t_sc, y_total_all, sample_weight=weights_all)
    final_logit_s = _fit_binary_classifier(x_s_sc, y_cover_all, c_value=best_c_spread, sample_weight=weights_all)
    final_logit_t = _fit_binary_classifier(x_t_sc, y_over_all, c_value=best_c_total, sample_weight=weights_all)

    print("\nSPREAD FEATURE COEFFICIENTS (standardized):")
    for feat, coef in sorted(zip(x_all_s_df.columns, final_ridge_s.coef_), key=lambda x: -abs(x[1])):
        flag = " <- TREND" if "trend" in feat else ""
        print(f"  {feat:<28} {coef:+.3f}{flag}")

    print("\nTOTALS FEATURE COEFFICIENTS (standardized):")
    for feat, coef in sorted(zip(x_all_t_df.columns, final_ridge_t.coef_), key=lambda x: -abs(x[1])):
        print(f"  {feat:<28} {coef:+.3f}")

    low_signal_spread = [f for f, c in zip(x_all_s_df.columns, final_ridge_s.coef_) if abs(float(c)) < 0.05]
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

    if spread_calibrator is not None:
        joblib.dump(spread_calibrator, model_dir / "spread_calibrator.pkl")
    elif (model_dir / "spread_calibrator.pkl").exists():
        (model_dir / "spread_calibrator.pkl").unlink()

    if total_calibrator is not None:
        joblib.dump(total_calibrator, model_dir / "total_calibrator.pkl")
    elif (model_dir / "total_calibrator.pkl").exists():
        (model_dir / "total_calibrator.pkl").unlink()

    feature_payload = {
        "spread": list(x_all_s_df.columns),
        "total": list(x_all_t_df.columns),
        "spread_base": spread_features,
        "total_base": total_features,
        "spread_medians": spread_medians,
        "total_medians": total_medians,
        "feature_engineering": "median_plus_missing_flags_v1",
        "recency_decay": RECENCY_DECAY,
        "ridge_alpha_spread": best_alpha_spread,
        "ridge_alpha_total": best_alpha_total,
        "logit_c_spread": best_c_spread,
        "logit_c_total": best_c_total,
    }
    joblib.dump(feature_payload, model_dir / "feature_lists.pkl")

    report = {
        "validation_games": int(len(val)),
        "spread_hit_rate": spread_hr,
        "spread_hit_rate_last_100": spread_hr_last100,
        "spread_hit_rate_last_50": spread_hr_last50,
        "spread_mae": spread_mae,
        "spread_brier_raw": spread_brier_raw,
        "spread_brier_cal": spread_brier_cal,
        "total_hit_rate": total_hr,
        "total_hit_rate_last_100": total_hr_last100,
        "total_mae": total_mae,
        "total_brier_raw": total_brier_raw,
        "total_brier_cal": total_brier_cal,
        "spread_fire_rate": len(fired_spread) / len(val),
        "total_fire_rate": len(fired_total) / len(val),
        "low_signal_spread_features": low_signal_spread,
        "ridge_alpha_spread": best_alpha_spread,
        "ridge_alpha_total": best_alpha_total,
        "logit_c_spread": best_c_spread,
        "logit_c_total": best_c_total,
        "has_spread_calibrator": bool(spread_calibrator is not None),
        "has_total_calibrator": bool(total_calibrator is not None),
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
