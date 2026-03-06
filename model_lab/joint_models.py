from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from pipeline.advanced_metrics import advanced_metrics_formulas as fm
from pipeline.market_canonical import load_latest_by_game, merge_market_lines

log = get_logger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DEBUG_DIR = REPO_ROOT / "debug"

GAMES_PATH = DATA_DIR / "games.csv"
WEIGHTED_PATH = DATA_DIR / "team_game_weighted.csv"
ADVANCED_PATH = DATA_DIR / "advanced_metrics.csv"
ROTATION_PATH = DATA_DIR / "rotation_features.csv"
MARKET_LATEST_PATH = DATA_DIR / "market_lines_latest.csv"
MARKET_CLOSING_PATH = DATA_DIR / "market_lines_closing.csv"

OUT_LATEST = DATA_DIR / "predictions_joint_latest.csv"
OUT_SNAPSHOTS = DATA_DIR / "predictions_joint_snapshots.csv"
OUT_SANITY = DEBUG_DIR / "sanity_outliers.csv"
OUT_MISSING = DEBUG_DIR / "missing_predictions_report.csv"
OUT_SCALE_CHECKS = DEBUG_DIR / "feature_scale_checks.json"
OUT_FEATURE_RANGE_AUDIT = DEBUG_DIR / "feature_range_audit.json"
OUT_PRED_MARGIN_DISTRIBUTION = DEBUG_DIR / "pred_margin_distribution.json"
OUT_PRED_MARGIN_CONSTANTS = DEBUG_DIR / "pred_margin_constants.csv"
OUT_SPREAD_CALC_SPEC = DEBUG_DIR / "spread_calculation_spec.md"
OUT_SPREAD_CALC_SAMPLES = DEBUG_DIR / "spread_calculation_samples.json"
OUT_PRED_FEATURE_JOIN_AUDIT = DEBUG_DIR / "pred_feature_join_audit.json"
OUT_JOIN_MISMATCH_SAMPLES = DEBUG_DIR / "join_mismatch_samples.csv"
OUT_BLOCKED_NUMERIC_PREDS = DEBUG_DIR / "blocked_rows_with_numeric_preds.csv"
TEAM_SNAPSHOT_PATH = DATA_DIR / "team_snapshot.csv"
TEAM_SNAPSHOT_AUDIT_PATH = DEBUG_DIR / "team_snapshot_audit.json"
BLOCKED_SAMPLES_PATH = DEBUG_DIR / "blocked_missing_input_samples.csv"
FEATURE_GATE_FAILURE_PATH = DEBUG_DIR / "feature_gate_failures.json"

MODEL_VERSION = "joint_v1.0"
TOTAL_BOUNDS = (110.0, 175.0)
ALLOCATION_BOUNDS = (-0.20, 0.20)
ALLOCATION_TANH_SCALE = 0.08
TOTAL_ADJUST_SCALE = 6.0
MAX_ABS_SPREAD = 60.0
SPREAD_EXTREME_THRESHOLD = 35.0
SPREAD_BLOCK_THRESHOLD = 60.0
SPREAD_INTERCEPT = 0.0
SPREAD_HCA_POINTS = 0.0
TOTAL_MIN = 110.0
TOTAL_MAX = 180.0
SPREAD_CONSTANTS_TO_AUDIT = (-70.0, -56.0, -44.0, 44.0, 56.0, 70.0)

SPREAD_WEIGHTS = {
    "odi_star_diff": 0.25,
    "sme_diff": 0.20,
    "posw": 0.15,
    "pxp_diff": 0.15,
    "lns_diff": 0.15,
    "vol_diff": 0.10,
}

TOTALS_WEIGHTS = {
    "posw": 0.25,
    "sch": 0.20,
    "wl": 0.20,
    "svi_avg": 0.20,
    "tc_diff": 0.15,
}

ULTIMATE_SPREAD_WEIGHTS = {
    "odi_star_diff": 0.22,
    "sme_diff": 0.18,
    "posw": 0.13,
    "pxp_diff": 0.12,
    "lns_diff": 0.12,
    "away_efg_diff": 0.10,
    "rfd": 0.07,
    "vol_diff": 0.06,
}

ULTIMATE_TOTALS_WEIGHTS = {
    "posw_sum": 0.22,
    "sch": 0.17,
    "wl_sum": 0.17,
    "svi_avg": 0.17,
    "tc_diff": 0.12,
    "alt_diff": 0.08,
    "rfd_sum": 0.07,
}

SPREAD_FEATURE_SCALES = {
    "odi_star_diff": 0.15,
    "sme_diff": 0.15,
    "posw": 0.35,
    "pxp_diff": 0.20,
    "lns_diff": 20.0,
    "vol_diff": 12.0,
    "away_efg_diff": 0.08,
    "rfd": 3.0,
}

FEATURE_ALIAS_REGISTRY: dict[str, list[str]] = {
    "lns_diff": ["lns_diff", "lines_diff"],
    "odi_star_diff": ["odi_star_diff", "odi_diff", "odi_star_delta"],
    "posw": ["posw", "posw_diff", "posw_sum"],
    "sch": ["sch", "SCH"],
    "tc_diff": ["tc_diff", "tc_delta"],
    "vol_diff": ["vol_diff", "vol_delta"],
    "wl": ["wl", "wl_sum"],
}

FEATURE_GATE_INPUTS: dict[str, list[str]] = {
    "lns_diff": ["form_rating", "net_rtg_l10", "adv_LNS"],
    "odi_star_diff": ["efg_pct", "efg_pct_l10", "tov_pct", "orb_pct", "ftr", "adv_ODI_star"],
    "posw": ["pace", "pace_l10", "poss", "orb_pct", "tov_pct", "adv_POSW"],
    "sch": ["pace", "pace_l10", "three_par", "three_par_l10", "ftr", "adv_SCH"],
    "tc_diff": ["pace", "pace_l10", "adv_TC"],
    "vol_diff": ["net_rtg_std_l10", "adv_VOL"],
    "wl": ["ftm", "ftr", "poss", "adv_WL"],
}


def _to_num(value: Any) -> float:
    x = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(x) if pd.notna(x) else np.nan


def _to_rate(value: Any) -> float:
    x = _to_num(value)
    if np.isnan(x):
        return np.nan
    if x > 1.5:
        return x / 100.0
    return x


def _canonical_id(value: Any) -> str:
    if value is None:
        return ""
    t = str(value).strip()
    if t.endswith(".0"):
        t = t[:-2]
    t = t.lstrip("0")
    return t or "0"


def canonical_game_key(df: pd.DataFrame) -> pd.Series:
    """Return canonical game key: event_id preferred, then game_id."""
    event = df["event_id"] if "event_id" in df.columns else pd.Series("", index=df.index)
    game = df["game_id"] if "game_id" in df.columns else pd.Series("", index=df.index)
    keys = event.where(event.astype(str).str.strip() != "", game)
    return keys.map(_canonical_id)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _normalize_key_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_id" not in out.columns and "game_id" in out.columns:
        out["event_id"] = out["game_id"]
    if "game_id" not in out.columns and "event_id" in out.columns:
        out["game_id"] = out["event_id"]
    if {"event_id", "game_id"} & set(out.columns):
        out["canonical_game_key"] = canonical_game_key(out)
        if "event_id" in out.columns:
            out["event_id"] = out["canonical_game_key"]
        if "game_id" in out.columns:
            out["game_id"] = out["canonical_game_key"]
    for col in ["event_id", "game_id", "team_id", "home_team_id", "away_team_id", "opponent_id"]:
        if col in out.columns:
            out[col] = out[col].map(_canonical_id)
    if "game_datetime_utc" in out.columns:
        out["game_datetime_utc"] = pd.to_datetime(out["game_datetime_utc"], utc=True, errors="coerce")
    return out


def _latest_team_snapshot(weighted: pd.DataFrame, advanced: pd.DataFrame, rotation: pd.DataFrame) -> dict[str, dict[str, Any]]:
    w = _normalize_key_cols(weighted)
    a = _normalize_key_cols(advanced)
    r = _normalize_key_cols(rotation)

    if not w.empty:
        w = w.sort_values(["team_id", "game_datetime_utc", "event_id"]).drop_duplicates("team_id", keep="last")
    if not a.empty:
        a = a.sort_values(["team_id", "game_datetime_utc", "event_id"]).drop_duplicates("team_id", keep="last")
    if not r.empty:
        if "event_id" not in r.columns and "game_id" in r.columns:
            r["event_id"] = r["game_id"]
        r = r.sort_values(["team_id", "event_id"]).drop_duplicates("team_id", keep="last")

    teams = sorted(set(w.get("team_id", pd.Series(dtype=str)).astype(str)) | set(a.get("team_id", pd.Series(dtype=str)).astype(str)) | set(r.get("team_id", pd.Series(dtype=str)).astype(str)))
    snapshots: dict[str, dict[str, Any]] = {}
    for team_id in teams:
        rec: dict[str, Any] = {"team_id": team_id}
        if not w.empty and team_id in set(w["team_id"]):
            rec.update(w.loc[w["team_id"] == team_id].iloc[0].to_dict())
        if not a.empty and team_id in set(a["team_id"]):
            rec.update({f"adv_{k}": v for k, v in a.loc[a["team_id"] == team_id].iloc[0].to_dict().items()})
        if not r.empty and team_id in set(r["team_id"]):
            rec.update({f"rot_{k}": v for k, v in r.loc[r["team_id"] == team_id].iloc[0].to_dict().items()})
        snapshots[team_id] = rec
    return snapshots


def _write_team_snapshot_artifact(snapshots: dict[str, dict[str, Any]]) -> pd.DataFrame:
    TEAM_SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(snapshots.values()) if snapshots else pd.DataFrame(columns=["team_id"])
    frame.to_csv(TEAM_SNAPSHOT_PATH, index=False)
    null_rates: dict[str, float] = {}
    for col in ["team_id", "efg_pct", "efg_pct_l10", "pace", "pace_l10", "form_rating", "net_rtg_std_l10"]:
        if col in frame.columns:
            null_rates[col] = round(float(frame[col].isna().mean()), 6)
    audit_payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "path": str(TEAM_SNAPSHOT_PATH),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "null_rates": null_rates,
    }
    _write_json(TEAM_SNAPSHOT_AUDIT_PATH, audit_payload)
    return frame


def _append_blocked_samples(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path = BLOCKED_SAMPLES_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        prev = pd.read_csv(path, low_memory=False)
        out = pd.concat([prev, new_df], ignore_index=True)
    else:
        out = new_df
    out.to_csv(path, index=False)


def _feature_presence_gate(snapshot_df: pd.DataFrame) -> None:
    coverage: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for feature_name, candidate_cols in FEATURE_GATE_INPUTS.items():
        present_cols = [c for c in candidate_cols if c in snapshot_df.columns]
        if present_cols:
            present_mask = snapshot_df[present_cols].apply(pd.to_numeric, errors="coerce").notna().any(axis=1)
            non_null_rows = int(present_mask.sum())
        else:
            non_null_rows = 0
        item = {
            "feature": feature_name,
            "candidate_columns": candidate_cols,
            "present_columns": present_cols,
            "non_null_rows": non_null_rows,
        }
        coverage.append(item)
        if non_null_rows == 0:
            failures.append(
                {
                    **item,
                    "expected_source_paths": "data/team_snapshot.csv|data/team_game_weighted.csv|data/advanced_metrics.csv",
                    "stage": "joint_models_predictions",
                }
            )
    if failures:
        payload = {
            "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
            "status": "blocked",
            "failures": failures,
            "coverage": coverage,
        }
        _write_json(FEATURE_GATE_FAILURE_PATH, payload)
        missing_names = ",".join(sorted(f["feature"] for f in failures))
        raise RuntimeError(
            "Feature Presence Gate failed for joint model. "
            f"Missing features with zero non-null coverage: {missing_names}. "
            f"See {FEATURE_GATE_FAILURE_PATH}."
        )

def _pick_num(row: dict[str, Any], candidates: list[str]) -> float:
    for col in candidates:
        value = _to_num(row.get(col))
        if np.isfinite(value):
            return value
    return np.nan


def _pick_rate(row: dict[str, Any], candidates: list[str]) -> float:
    for col in candidates:
        value = _to_rate(row.get(col))
        if np.isfinite(value):
            return value
    return np.nan


def _adv_metric(team_row: dict[str, Any], metric_col: str, acronym: str) -> float:
    status = str(team_row.get(f"adv_metric_status_{acronym}", ""))
    val = _to_num(team_row.get(f"adv_{metric_col}"))
    if status == "OK" and np.isfinite(val):
        return val
    return np.nan


def _team_fallback_features(team_row: dict[str, Any], opp_row: dict[str, Any]) -> dict[str, float]:
    efg = _pick_rate(team_row, ["efg_pct", "efg_pct_l10", "efg_pct_l5", "ha_efg_pct_l10", "efg"])
    tov = _pick_rate(team_row, ["tov_pct", "tov_pct_l10", "tov_pct_l5", "ha_tov_pct_l10", "tov"])
    orb = _pick_rate(team_row, ["orb_pct", "orb_pct_l10", "orb_pct_l5", "orb"])
    ftr = _pick_rate(team_row, ["ftr", "ftr_l10", "ftr_l5"])
    pace = _pick_num(team_row, ["pace", "pace_l10", "poss", "poss_l10", "adj_pace", "ha_pace_l10"])

    opp_efg = _pick_rate(opp_row, ["efg_pct", "efg_pct_l10", "efg_pct_l5", "ha_efg_pct_l10", "efg"])
    opp_tov = _pick_rate(opp_row, ["tov_pct", "tov_pct_l10", "tov_pct_l5", "ha_tov_pct_l10", "tov"])
    opp_orb = _pick_rate(opp_row, ["orb_pct", "orb_pct_l10", "orb_pct_l5", "orb"])
    opp_ftr = _pick_rate(opp_row, ["ftr", "ftr_l10", "ftr_l5"])
    opp_pace = _pick_num(opp_row, ["pace", "pace_l10", "poss", "poss_l10", "adj_pace", "ha_pace_l10"])

    odi = fm.odi_star(
        (efg - 0.505) - (opp_efg - 0.505) if np.isfinite(efg) and np.isfinite(opp_efg) else np.nan,
        (opp_tov - 0.18) - (tov - 0.18) if np.isfinite(tov) and np.isfinite(opp_tov) else np.nan,
        (orb - 0.30) - (opp_orb - 0.30) if np.isfinite(orb) and np.isfinite(opp_orb) else np.nan,
        (ftr - 0.28) - (opp_ftr - 0.28) if np.isfinite(ftr) and np.isfinite(opp_ftr) else np.nan,
    )

    pei = fm.pei(orb, opp_orb, opp_tov, tov) if np.isfinite(orb) and np.isfinite(opp_orb) and np.isfinite(opp_tov) and np.isfinite(tov) else np.nan
    projected_pace = (pace + opp_pace) / 2.0 if np.isfinite(pace) and np.isfinite(opp_pace) else np.nan
    posw = fm.posw(pei, projected_pace) if np.isfinite(pei) and np.isfinite(projected_pace) else np.nan

    ftm = _pick_num(team_row, ["ftm", "ftm_wtd_off_l10", "ftm_wtd_off_l5"])
    fga = _pick_num(team_row, ["fga", "fga_wtd_off_l10", "fga_wtd_off_l5"])
    ft_pts_per_fga = ftm / fga if np.isfinite(ftm) and np.isfinite(fga) and fga != 0 else np.nan
    three_pa = _pick_rate(team_row, ["three_par", "three_par_l10", "three_par_l5"])
    svi = fm.svi(
        fm.safe_zscore(efg, 0.505, 0.04),
        fm.safe_zscore(three_pa, 0.38, 0.08),
        fm.safe_zscore(ft_pts_per_fga, 0.22, 0.06),
    ) if np.isfinite(efg) and np.isfinite(three_pa) and np.isfinite(ft_pts_per_fga) else np.nan

    rotation_stability = 1.0 - _pick_num(team_row, ["rot_rot_minshare_sd", "rot_minshare_sd"])
    rotation_stability = float(np.clip(rotation_stability, 0.0, 1.0)) if np.isfinite(rotation_stability) else 0.5
    ics = fm.ics(0.5, rotation_stability, 0.5)
    bsi = fm.bsi(0.33, 0.0, 0.0)
    clutch = _to_rate(team_row.get("close_game_win_pct"))
    pxp = fm.pxp(ics, bsi, clutch if np.isfinite(clutch) else 0.5)

    lns = _pick_num(team_row, ["form_rating", "net_rtg_l10", "adj_net_rtg", "net_rtg"])
    if not np.isfinite(lns):
        lns = _to_num(team_row.get("net_rtg_l10"))

    vol = _pick_num(team_row, ["net_rtg_std_l10", "ha_net_rtg_l10"])

    pace_l10 = _pick_num(team_row, ["pace_l10", "poss_l10", "ha_pace_l10", "pace"])
    tc = fm.tc(fm.safe_zscore(abs(pace - pace_l10), 0.0, 4.0)) if np.isfinite(pace) and np.isfinite(pace_l10) else np.nan

    poss = _pick_num(team_row, ["poss", "pace", "poss_l10", "pace_l10"])
    ft_ppp = ftm / poss if np.isfinite(ftm) and np.isfinite(poss) and poss != 0 else np.nan
    opp_ftm = _pick_num(opp_row, ["ftm", "ftm_wtd_off_l10", "ftm_wtd_off_l5"])
    opp_poss = _pick_num(opp_row, ["poss", "pace", "poss_l10", "pace_l10"])
    opp_ft_ppp = opp_ftm / opp_poss if np.isfinite(opp_ftm) and np.isfinite(opp_poss) and opp_poss != 0 else np.nan
    wl = fm.wl(fm.safe_zscore(ft_ppp - opp_ft_ppp, 0.0, 0.08), fm.safe_zscore(ftr - opp_ftr, 0.0, 0.08)) if np.isfinite(ft_ppp) and np.isfinite(opp_ft_ppp) and np.isfinite(ftr) and np.isfinite(opp_ftr) else np.nan

    size = _pick_num(team_row, ["rot_rot_size", "rot_size"])
    if not np.isfinite(size):
        size = 8.0

    rest_days = _pick_num(team_row, ["rest_days", "home_rest_days", "away_rest_days"])
    b2b = 1.0 if np.isfinite(rest_days) and rest_days <= 1 else 0.0

    return {
        "odi_star": odi,
        "posw": posw,
        "svi": svi,
        "pxp": pxp,
        "lns": lns,
        "vol": vol,
        "tc": tc,
        "wl": wl,
        "pace": pace,
        "three_par": three_pa,
        "ftr": ftr,
        "size": size,
        "efg": efg,
        "rest_days": rest_days,
        "b2b": b2b,
        "travel_miles": _pick_num(team_row, ["estimated_travel_miles"]),
    }


def compute_totals_features(game_row: dict[str, Any], team_rows: dict[str, dict[str, Any]], advanced_metrics: dict[str, Any] | None = None) -> dict[str, float]:
    home, away = team_rows["home"], team_rows["away"]
    h = _team_fallback_features(home, away)
    a = _team_fallback_features(away, home)

    posw_h = _adv_metric(home, "POSW", "POSW")
    posw_a = _adv_metric(away, "POSW", "POSW")
    wl_h = _adv_metric(home, "WL", "WL")
    wl_a = _adv_metric(away, "WL", "WL")
    svi_h = _adv_metric(home, "SVI", "SVI")
    svi_a = _adv_metric(away, "SVI", "SVI")
    tc_h = _adv_metric(home, "TC", "TC")
    tc_a = _adv_metric(away, "TC", "TC")

    posw_sum = (posw_h if np.isfinite(posw_h) else h["posw"]) + (posw_a if np.isfinite(posw_a) else a["posw"])
    wl_sum = (wl_h if np.isfinite(wl_h) else h["wl"]) + (wl_a if np.isfinite(wl_a) else a["wl"])
    svi_avg = np.nanmean([svi_h if np.isfinite(svi_h) else h["svi"], svi_a if np.isfinite(svi_a) else a["svi"]])
    tc_diff = (tc_h if np.isfinite(tc_h) else h["tc"]) - (tc_a if np.isfinite(tc_a) else a["tc"])

    sch = fm.sch(h["pace"], a["pace"], h["three_par"], a["three_par"], h["ftr"], a["ftr"], h["size"], a["size"])

    rfd_sum = fm.rfd(h["rest_days"], a["rest_days"], a["b2b"] - h["b2b"]) if np.isfinite(h["rest_days"]) and np.isfinite(a["rest_days"]) else np.nan
    alt_diff = fm.alt(np.nan, np.nan, a["travel_miles"]) if np.isfinite(a["travel_miles"]) else np.nan
    return {
        "posw": posw_sum,
        "sch": sch,
        "wl": wl_sum,
        "svi_avg": svi_avg,
        "tc_diff": tc_diff,
        "posw_sum": posw_sum,
        "wl_sum": wl_sum,
        "rfd_sum": rfd_sum,
        "alt_diff": alt_diff,
    }


def compute_spread_features(game_row: dict[str, Any], team_rows: dict[str, dict[str, Any]], advanced_metrics: dict[str, Any] | None = None) -> dict[str, float]:
    home, away = team_rows["home"], team_rows["away"]
    h = _team_fallback_features(home, away)
    a = _team_fallback_features(away, home)

    odi_diff = (_adv_metric(home, "ODI_star", "ODI") if np.isfinite(_adv_metric(home, "ODI_star", "ODI")) else h["odi_star"]) - (
        _adv_metric(away, "ODI_star", "ODI") if np.isfinite(_adv_metric(away, "ODI_star", "ODI")) else a["odi_star"]
    )
    sme_h = _adv_metric(home, "SME", "SME")
    sme_a = _adv_metric(away, "SME", "SME")
    sme_diff = (sme_h if np.isfinite(sme_h) else 0.0) - (sme_a if np.isfinite(sme_a) else 0.0)
    posw_diff = (_adv_metric(home, "POSW", "POSW") if np.isfinite(_adv_metric(home, "POSW", "POSW")) else h["posw"]) - (
        _adv_metric(away, "POSW", "POSW") if np.isfinite(_adv_metric(away, "POSW", "POSW")) else a["posw"]
    )
    pxp_diff = (_adv_metric(home, "PXP", "PXP") if np.isfinite(_adv_metric(home, "PXP", "PXP")) else h["pxp"]) - (
        _adv_metric(away, "PXP", "PXP") if np.isfinite(_adv_metric(away, "PXP", "PXP")) else a["pxp"]
    )
    lns_diff = (_adv_metric(home, "LNS", "LNS") if np.isfinite(_adv_metric(home, "LNS", "LNS")) else h["lns"]) - (
        _adv_metric(away, "LNS", "LNS") if np.isfinite(_adv_metric(away, "LNS", "LNS")) else a["lns"]
    )
    vol_diff = (_adv_metric(home, "VOL", "VOL") if np.isfinite(_adv_metric(home, "VOL", "VOL")) else h["vol"]) - (
        _adv_metric(away, "VOL", "VOL") if np.isfinite(_adv_metric(away, "VOL", "VOL")) else a["vol"]
    )
    away_efg_diff = h["efg"] - a["efg"] if np.isfinite(h["efg"]) and np.isfinite(a["efg"]) else np.nan
    rfd = fm.rfd(h["rest_days"], a["rest_days"], a["b2b"] - h["b2b"]) if np.isfinite(h["rest_days"]) and np.isfinite(a["rest_days"]) else np.nan
    return {
        "odi_star_diff": odi_diff,
        "sme_diff": sme_diff,
        "posw": posw_diff,
        "pxp_diff": pxp_diff,
        "lns_diff": lns_diff,
        "vol_diff": vol_diff,
        "away_efg_diff": away_efg_diff,
        "rfd": rfd,
    }


def _weighted_signal_with_trace(
    features: dict[str, float],
    weights: dict[str, float],
    scales: dict[str, float] | None = None,
) -> tuple[float, list[str], float, list[dict[str, float | str]]]:
    missing: list[str] = []
    num = 0.0
    den = 0.0
    terms: list[dict[str, float | str]] = []
    for name, w in weights.items():
        aliases = FEATURE_ALIAS_REGISTRY.get(name, [name])
        v = np.nan
        for alias in aliases:
            candidate = _to_num(features.get(alias))
            if np.isfinite(candidate):
                v = candidate
                break
        if not np.isfinite(v):
            missing.append(name)
            continue
        scale = _to_num((scales or {}).get(name, 1.0))
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0
        normalized = v / scale
        term = normalized * w
        terms.append(
            {
                "name": name,
                "value": float(v),
                "coef": float(w),
                "scale": float(scale),
                "normalized_value": float(normalized),
                "term": float(term),
            }
        )
        num += term
        den += w
    if den <= 0:
        return np.nan, missing, 0.0, terms
    return num / den, missing, den, terms


def _weighted_signal(
    features: dict[str, float],
    weights: dict[str, float],
    scales: dict[str, float] | None = None,
) -> tuple[float, list[str], float]:
    signal, missing, coverage, _ = _weighted_signal_with_trace(features, weights, scales=scales)
    return signal, missing, coverage


def _spread_allocation_from_signal(signal: float) -> tuple[float, float, bool]:
    if not np.isfinite(signal):
        return np.nan, np.nan, False
    allocation_raw = float(np.tanh(signal) * ALLOCATION_TANH_SCALE)
    allocation_pct = float(np.clip(allocation_raw, ALLOCATION_BOUNDS[0], ALLOCATION_BOUNDS[1]))
    clipped = bool(allocation_pct != allocation_raw)
    return allocation_raw, allocation_pct, clipped


def _base_total(team_rows: dict[str, dict[str, Any]]) -> float:
    home = team_rows["home"]
    away = team_rows["away"]
    pace_h = _to_num(home.get("pace"))
    pace_a = _to_num(away.get("pace"))
    if not np.isfinite(pace_h) or not np.isfinite(pace_a):
        pace_h = _to_num(home.get("poss_l10"))
        pace_a = _to_num(away.get("poss_l10"))
    projected_pace = np.nanmean([pace_h, pace_a])

    ortg_h = _to_num(home.get("adj_ortg"))
    ortg_a = _to_num(away.get("adj_ortg"))
    if not np.isfinite(ortg_h):
        ortg_h = _to_num(home.get("ortg"))
    if not np.isfinite(ortg_a):
        ortg_a = _to_num(away.get("ortg"))
    if not np.isfinite(projected_pace):
        projected_pace = 68.0
    if not np.isfinite(ortg_h):
        ortg_h = 103.0
    if not np.isfinite(ortg_a):
        ortg_a = 103.0
    return float(projected_pace * ((ortg_h / 100.0) + (ortg_a / 100.0)))


def predict_game_joint(
    game_row: dict[str, Any],
    team_rows: dict[str, dict[str, Any]],
    advanced_metrics: dict[str, Any] | None = None,
    market_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    totals_features = compute_totals_features(game_row, team_rows, advanced_metrics)
    spread_features = compute_spread_features(game_row, team_rows, advanced_metrics)

    totals_signal, missing_totals, totals_cov, totals_terms = _weighted_signal_with_trace(totals_features, TOTALS_WEIGHTS)
    spread_signal, missing_spread, spread_cov, spread_terms = _weighted_signal_with_trace(
        spread_features,
        SPREAD_WEIGHTS,
        scales=SPREAD_FEATURE_SCALES,
    )
    legacy_spread_signal, _, _, _ = _weighted_signal_with_trace(spread_features, SPREAD_WEIGHTS)

    required_count = len(TOTALS_WEIGHTS) + len(SPREAD_WEIGHTS)
    missing_count = len(missing_totals) + len(missing_spread)
    feature_missing_rate = missing_count / required_count if required_count else 0.0

    model_status = "OK"
    if totals_cov < 0.5 or spread_cov < 0.5:
        names = sorted(set(missing_totals + missing_spread))
        model_status = f"BLOCKED_MISSING_INPUT:{','.join(names) if names else 'insufficient_feature_weight'}"
        blocked_row = {
            "event_id": _canonical_id(game_row.get("event_id") or game_row.get("game_id")),
            "game_id": _canonical_id(game_row.get("game_id") or game_row.get("event_id")),
            "home_team": game_row.get("home_team"),
            "away_team": game_row.get("away_team"),
            "model_status": model_status,
            "pred_total": np.nan,
            "pred_margin_raw": np.nan,
            "pred_margin_final": np.nan,
            "allocation_raw": np.nan,
            "allocation_pct": np.nan,
            "spread_signal_raw": spread_signal,
            "legacy_spread_signal_raw": legacy_spread_signal,
            "legacy_allocation_pct": np.nan,
            "legacy_pred_margin_raw": np.nan,
            "spread_signal_tanh": np.nan,
            "spread_terms": spread_terms,
            "spread_features": spread_features,
            "adjustments_applied": [],
            "missing_spread_features": missing_spread,
        }
        return {
            "pred_total": np.nan,
            "allocation_pct": np.nan,
            "pred_home_score": np.nan,
            "pred_away_score": np.nan,
            "pred_margin": np.nan,
            "pred_margin_raw": np.nan,
            "pred_margin_pre_guardrail": np.nan,
            "pred_margin_final": np.nan,
            "pred_total_ultimate": np.nan,
            "pred_margin_ultimate": np.nan,
            "feature_missing_rate": feature_missing_rate,
            "model_status": model_status,
            "pred_total_raw": np.nan,
            "allocation_raw": np.nan,
            "spread_signal_raw": spread_signal,
            "legacy_spread_signal_raw": legacy_spread_signal,
            "legacy_allocation_pct": np.nan,
            "legacy_pred_margin_raw": np.nan,
            "spread_signal_tanh": np.nan,
            "spread_terms": spread_terms,
            "totals_terms": totals_terms,
            "adjustments_applied": [],
            "total_clipped": False,
            "totals_features": totals_features,
            "spread_features": spread_features,
            "spread_calc_trace": build_spread_calc_trace(
                blocked_row,
                {
                    "model_name": MODEL_VERSION,
                    "model_version": MODEL_VERSION,
                    "weights": SPREAD_WEIGHTS,
                    "scales": SPREAD_FEATURE_SCALES,
                    "intercept": SPREAD_INTERCEPT,
                    "hca_points": SPREAD_HCA_POINTS,
                },
            ),
        }

    base_total = _base_total(team_rows)
    pred_total_raw = base_total + (totals_signal * TOTAL_ADJUST_SCALE)
    clipped_total = float(np.clip(pred_total_raw, TOTAL_BOUNDS[0], TOTAL_BOUNDS[1]))
    total_clipped = bool(clipped_total != pred_total_raw)
    if total_clipped:
        log.info(f"pred_total clipped game_id={game_row.get('game_id')} raw={pred_total_raw:.2f} clipped={clipped_total:.2f}")
    pred_total = clipped_total

    allocation_raw, allocation_pct, allocation_clipped = _spread_allocation_from_signal(spread_signal)
    spread_signal_tanh = float(np.tanh(spread_signal)) if np.isfinite(spread_signal) else np.nan
    legacy_allocation_pct = float(np.clip(legacy_spread_signal, ALLOCATION_BOUNDS[0], ALLOCATION_BOUNDS[1])) if np.isfinite(legacy_spread_signal) else np.nan
    legacy_pred_margin_raw = float(pred_total * 2.0 * legacy_allocation_pct) if np.isfinite(legacy_allocation_pct) else np.nan
    pred_margin_raw = (pred_total * 2.0 * allocation_pct) + SPREAD_INTERCEPT + SPREAD_HCA_POINTS
    pred_margin_pre_guardrail = pred_margin_raw
    pred_margin_final = pred_margin_raw
    adjustments_applied: list[str] = []
    if allocation_clipped:
        adjustments_applied.append(f"ALLOCATION_CLIPPED[{ALLOCATION_BOUNDS[0]},{ALLOCATION_BOUNDS[1]}]")

    if np.isfinite(pred_margin_raw) and abs(float(pred_margin_raw)) > SPREAD_BLOCK_THRESHOLD:
        model_status = f"BLOCKED_SPREAD_GUARDRAIL:abs(pred_margin_raw)>{SPREAD_BLOCK_THRESHOLD}"
        pred_margin_raw = np.nan
        pred_margin_final = np.nan
        adjustments_applied.append(f"BLOCKED_SPREAD_GUARDRAIL:{SPREAD_BLOCK_THRESHOLD}")

    if np.isfinite(pred_margin_final):
        pred_home_score = (pred_total / 2.0) + (pred_margin_final / 2.0)
        pred_away_score = (pred_total / 2.0) - (pred_margin_final / 2.0)
    else:
        pred_home_score = np.nan
        pred_away_score = np.nan

    u_totals, _, u_cov_tot = _weighted_signal(
        {
            "posw_sum": totals_features.get("posw_sum"),
            "sch": totals_features.get("sch"),
            "wl_sum": totals_features.get("wl_sum"),
            "svi_avg": totals_features.get("svi_avg"),
            "tc_diff": totals_features.get("tc_diff"),
            "alt_diff": totals_features.get("alt_diff"),
            "rfd_sum": totals_features.get("rfd_sum"),
        },
        ULTIMATE_TOTALS_WEIGHTS,
    )
    pred_total_ultimate = base_total + (u_totals * TOTAL_ADJUST_SCALE) if u_cov_tot > 0 else np.nan
    if np.isfinite(pred_total_ultimate):
        pred_total_ultimate = float(np.clip(pred_total_ultimate, TOTAL_BOUNDS[0], TOTAL_BOUNDS[1]))

    u_spread, _, u_cov_spread = _weighted_signal(spread_features, ULTIMATE_SPREAD_WEIGHTS, scales=SPREAD_FEATURE_SCALES)
    if u_cov_spread > 0:
        _, u_alloc_pct, _ = _spread_allocation_from_signal(u_spread)
        pred_margin_ultimate = float((u_alloc_pct * pred_total * 2.0) + SPREAD_INTERCEPT + SPREAD_HCA_POINTS)
    else:
        pred_margin_ultimate = np.nan

    market_status = "MISSING_LINES"
    spread_edge = np.nan
    total_edge = np.nan
    if market_row:
        spread_line = _to_num(market_row.get("spread_line"))
        total_line = _to_num(market_row.get("total_line"))
        if np.isfinite(spread_line) or np.isfinite(total_line):
            market_status = "OK"
            if np.isfinite(spread_line) and np.isfinite(pred_margin_final):
                spread_edge = pred_margin_final - spread_line
            if np.isfinite(total_line):
                total_edge = pred_total - total_line

    calc_row = {
        "event_id": _canonical_id(game_row.get("event_id") or game_row.get("game_id")),
        "game_id": _canonical_id(game_row.get("game_id") or game_row.get("event_id")),
        "home_team": game_row.get("home_team"),
        "away_team": game_row.get("away_team"),
        "model_status": model_status,
        "pred_total": pred_total,
        "pred_margin_raw": pred_margin_raw,
        "pred_margin_pre_guardrail": pred_margin_pre_guardrail,
        "pred_margin_final": pred_margin_final,
        "allocation_raw": allocation_raw,
        "allocation_pct": allocation_pct,
        "spread_signal_raw": spread_signal,
        "legacy_spread_signal_raw": legacy_spread_signal,
        "legacy_allocation_pct": legacy_allocation_pct,
        "legacy_pred_margin_raw": legacy_pred_margin_raw,
        "spread_signal_tanh": spread_signal_tanh,
        "spread_terms": spread_terms,
        "spread_features": spread_features,
        "adjustments_applied": adjustments_applied,
        "missing_spread_features": missing_spread,
    }
    spread_calc_trace = build_spread_calc_trace(
        calc_row,
        {
            "model_name": MODEL_VERSION,
            "model_version": MODEL_VERSION,
            "weights": SPREAD_WEIGHTS,
            "scales": SPREAD_FEATURE_SCALES,
            "intercept": SPREAD_INTERCEPT,
            "hca_points": SPREAD_HCA_POINTS,
        },
    )

    return {
        "pred_total": pred_total,
        "allocation_pct": allocation_pct,
        "pred_home_score": pred_home_score,
        "pred_away_score": pred_away_score,
        "pred_margin": pred_margin_final,
        "pred_margin_raw": pred_margin_raw,
        "pred_margin_pre_guardrail": pred_margin_pre_guardrail,
        "pred_margin_final": pred_margin_final,
        "pred_total_ultimate": pred_total_ultimate,
        "pred_margin_ultimate": pred_margin_ultimate,
        "spread_edge": spread_edge,
        "total_edge": total_edge,
        "market_status": market_status,
        "feature_missing_rate": feature_missing_rate,
        "model_status": model_status,
        "pred_total_raw": pred_total_raw,
        "allocation_raw": allocation_raw,
        "spread_signal_raw": spread_signal,
        "legacy_spread_signal_raw": legacy_spread_signal,
        "legacy_allocation_pct": legacy_allocation_pct,
        "legacy_pred_margin_raw": legacy_pred_margin_raw,
        "spread_signal_tanh": spread_signal_tanh,
        "spread_terms": spread_terms,
        "totals_terms": totals_terms,
        "adjustments_applied": adjustments_applied,
        "total_clipped": total_clipped,
        "totals_features": totals_features,
        "spread_features": spread_features,
        "spread_calc_trace": spread_calc_trace,
    }


def _build_market_lookup() -> dict[str, dict[str, Any]]:
    latest = load_latest_by_game(DATA_DIR, auto_build=True)
    if latest.empty:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for _, row in latest.iterrows():
        gid = _canonical_id(row.get("event_id") or row.get("game_id"))
        if not gid:
            continue
        out[gid] = {
            "spread_line": row.get("spread_line"),
            "total_line": row.get("total_line"),
            "line_source_used": row.get("line_source_used", row.get("source")),
            "line_timestamp_utc": row.get("line_timestamp_utc"),
            "market_status": row.get("market_status"),
        }
    return out


def _log_written_csv(path: Path, df: pd.DataFrame) -> None:
    log.info(f"csv_write path={path.resolve()} rows={len(df)} cols={len(df.columns)}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _series_distribution(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "p1": np.nan,
            "p5": np.nan,
            "p50": np.nan,
            "p95": np.nan,
            "p99": np.nan,
        }
    q = numeric.quantile([0.01, 0.05, 0.50, 0.95, 0.99]).to_dict()
    return {
        "count": int(numeric.notna().sum()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "p1": float(q.get(0.01, np.nan)),
        "p5": float(q.get(0.05, np.nan)),
        "p50": float(q.get(0.50, np.nan)),
        "p95": float(q.get(0.95, np.nan)),
        "p99": float(q.get(0.99, np.nan)),
    }


def build_spread_calc_trace(row: dict[str, Any], model_spec: dict[str, Any]) -> dict[str, Any]:
    terms = row.get("spread_terms", []) or []
    sorted_terms = sorted(terms, key=lambda item: abs(_to_num(item.get("term"))), reverse=True)
    return {
        "event_id": row.get("event_id"),
        "game_id": row.get("game_id"),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "model_name": model_spec.get("model_name"),
        "model_version": model_spec.get("model_version"),
        "perspective": "HOME (positive margin means home favored)",
        "formula": {
            "signal_raw": "sum((feature_value / feature_scale) * weight) / sum(weight_present)",
            "allocation_raw": f"tanh(signal_raw) * {ALLOCATION_TANH_SCALE}",
            "allocation_final": f"clip(allocation_raw, {ALLOCATION_BOUNDS[0]}, {ALLOCATION_BOUNDS[1]})",
            "pred_margin_raw": "pred_total * 2 * allocation_final + intercept + hca_points",
            "pred_margin_final": "pred_margin_raw unless blocked by guardrails",
            "legacy_reference_formula": "pred_total * 2 * clip(unscaled_signal_raw, -0.2, 0.2)",
        },
        "intercept": float(_to_num(model_spec.get("intercept"))),
        "hca_points": float(_to_num(model_spec.get("hca_points"))),
        "signal_raw": _to_num(row.get("spread_signal_raw")),
        "legacy_unscaled_signal_raw": _to_num(row.get("legacy_spread_signal_raw")),
        "legacy_allocation_pct": _to_num(row.get("legacy_allocation_pct")),
        "legacy_pred_margin_raw": _to_num(row.get("legacy_pred_margin_raw")),
        "signal_tanh": _to_num(row.get("spread_signal_tanh")),
        "allocation_raw": _to_num(row.get("allocation_raw")),
        "allocation_final": _to_num(row.get("allocation_pct")),
        "pred_total": _to_num(row.get("pred_total")),
        "pred_margin_raw": _to_num(row.get("pred_margin_raw")),
        "pred_margin_pre_guardrail": _to_num(row.get("pred_margin_pre_guardrail")),
        "pred_margin_final": _to_num(row.get("pred_margin_final")),
        "model_status": row.get("model_status"),
        "missing_spread_features": row.get("missing_spread_features", []),
        "feature_terms": terms,
        "top_abs_terms": sorted_terms[:5],
        "adjustments_applied": row.get("adjustments_applied", []),
    }


def _write_spread_calculation_spec() -> None:
    lines = [
        "# Spread Calculation Confirmation",
        "",
        f"Model: `{MODEL_VERSION}`",
        "Perspective: HOME (positive margin means home favored).",
        "",
        "## Exact Formula",
        "1. `signal_raw = sum((feature_i / scale_i) * weight_i) / sum(weight_i_present)`",
        f"2. `allocation_raw = tanh(signal_raw) * {ALLOCATION_TANH_SCALE}`",
        f"3. `allocation_final = clip(allocation_raw, {ALLOCATION_BOUNDS[0]}, {ALLOCATION_BOUNDS[1]})`",
        "4. `pred_margin_raw = pred_total * 2 * allocation_final + intercept + hca_points`",
        "5. `pred_margin_final = pred_margin_raw` unless a guardrail blocks the row.",
        "Legacy (pre-fix) reference: `pred_margin_legacy = pred_total * 2 * clip(unscaled_signal_raw, -0.2, 0.2)`.",
        "",
        "## Intercept / HCA",
        f"- intercept = {SPREAD_INTERCEPT}",
        f"- hca_points = {SPREAD_HCA_POINTS}",
        "",
        "## Feature Scales",
    ]
    for key in sorted(SPREAD_FEATURE_SCALES):
        lines.append(f"- `{key}` scale = {SPREAD_FEATURE_SCALES[key]}")
    lines.extend(
        [
            "",
            "## Percent Scaling Convention",
            "- Percent-like source columns are read as either `0-1` or `0-100`.",
            "- Normalization rule before metric construction: any value `> 1.5` is divided by `100`.",
            "",
            "## Notes",
            "- No hard-coded spread fallback constants are used.",
            "- BLOCKED rows keep `pred_margin` as blank/NaN.",
        ]
    )
    OUT_SPREAD_CALC_SPEC.parent.mkdir(parents=True, exist_ok=True)
    OUT_SPREAD_CALC_SPEC.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pred_margin_diagnostics(out: pd.DataFrame) -> None:
    margin_final = pd.to_numeric(out.get("pred_margin"), errors="coerce")
    margin_raw = pd.to_numeric(out.get("pred_margin_raw"), errors="coerce")
    margin_legacy = pd.to_numeric(out.get("legacy_pred_margin_raw"), errors="coerce")
    payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "model_version": MODEL_VERSION,
        "pred_margin_final": _series_distribution(margin_final),
        "pred_margin_raw": _series_distribution(margin_raw),
        "pred_margin_legacy_reference": _series_distribution(margin_legacy),
        "constant_values_checked": list(SPREAD_CONSTANTS_TO_AUDIT),
    }
    _write_json(OUT_PRED_MARGIN_DISTRIBUTION, payload)

    status = out.get("model_status", pd.Series("", index=out.index)).astype(str)
    records: list[dict[str, Any]] = []
    for constant in SPREAD_CONSTANTS_TO_AUDIT:
        for col in ("pred_margin", "pred_margin_raw"):
            values = pd.to_numeric(out.get(col), errors="coerce")
            match = values == float(constant)
            blocked = match & status.str.startswith("BLOCKED")
            records.append(
                {
                    "column": col,
                    "constant_value": float(constant),
                    "count_total": int(match.sum()),
                    "count_blocked": int(blocked.sum()),
                    "count_non_blocked": int((match & ~status.str.startswith("BLOCKED")).sum()),
                }
            )
    constants_df = pd.DataFrame(records)
    constants_df.to_csv(OUT_PRED_MARGIN_CONSTANTS, index=False)
    _log_written_csv(OUT_PRED_MARGIN_CONSTANTS, constants_df)


def _write_spread_calc_samples(traces: list[dict[str, Any]]) -> None:
    def _abs_val(item: dict[str, Any], key: str) -> float:
        return abs(_to_num(item.get(key))) if np.isfinite(_to_num(item.get(key))) else np.nan

    normal = [
        item
        for item in traces
        if str(item.get("model_status", "")).startswith("OK")
        and np.isfinite(_to_num(item.get("pred_margin_final")))
        and _abs_val(item, "pred_margin_final") <= SPREAD_EXTREME_THRESHOLD
    ]
    normal = sorted(normal, key=lambda item: (_abs_val(item, "pred_margin_final"), str(item.get("event_id"))))

    extreme = [
        item
        for item in traces
        if (
            np.isfinite(_to_num(item.get("pred_margin_final")))
            and _abs_val(item, "pred_margin_final") > SPREAD_EXTREME_THRESHOLD
        )
        or (
            np.isfinite(_to_num(item.get("pred_margin_raw")))
            and _abs_val(item, "pred_margin_raw") > SPREAD_EXTREME_THRESHOLD
        )
        or (
            np.isfinite(_to_num(item.get("legacy_pred_margin_raw")))
            and _abs_val(item, "legacy_pred_margin_raw") > SPREAD_EXTREME_THRESHOLD
        )
    ]
    extreme = sorted(
        extreme,
        key=lambda item: max(
            _abs_val(item, "pred_margin_raw") if np.isfinite(_abs_val(item, "pred_margin_raw")) else 0.0,
            _abs_val(item, "legacy_pred_margin_raw") if np.isfinite(_abs_val(item, "legacy_pred_margin_raw")) else 0.0,
        ),
        reverse=True,
    )

    blocked = [item for item in traces if str(item.get("model_status", "")).startswith("BLOCKED")]
    blocked = sorted(blocked, key=lambda item: str(item.get("event_id")))

    payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "model_version": MODEL_VERSION,
        "selection_rules": {
            "normal": "status=OK and abs(pred_margin_final)<=35, first 10 by abs margin",
            "extreme": "abs(pred_margin_raw)>35 or abs(pred_margin_final)>35 or abs(legacy_pred_margin_raw)>35, top 10 by max abs margin",
            "blocked": "status starts with BLOCKED, first 10",
        },
        "counts": {
            "total_traces": len(traces),
            "normal_candidates": len(normal),
            "extreme_candidates": len(extreme),
            "blocked_candidates": len(blocked),
        },
        "samples": {
            "normal": normal[:10],
            "extreme": extreme[:10],
            "blocked": blocked[:10],
        },
    }
    _write_json(OUT_SPREAD_CALC_SAMPLES, payload)


def _write_feature_range_audit(snapshot_df: pd.DataFrame) -> None:
    audit_cols = sorted(
        {
            "efg_pct",
            "efg_pct_l10",
            "orb_pct",
            "orb_pct_l10",
            "drb_pct",
            "drb_pct_l10",
            "tov_pct",
            "tov_pct_l10",
            "ftr",
            "ftr_l10",
            "pace",
            "pace_l10",
            "poss",
            "poss_l10",
            "form_rating",
            "net_rtg_l10",
            "adj_net_rtg",
            "net_rtg_std_l10",
            "rest_days",
        }
    )
    percent_like = {"efg_pct", "efg_pct_l10", "orb_pct", "orb_pct_l10", "drb_pct", "drb_pct_l10", "tov_pct", "tov_pct_l10", "ftr", "ftr_l10"}
    payload: dict[str, Any] = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "normalization_rule": "values > 1.5 are interpreted as 0-100 percentages and divided by 100",
        "columns": {},
        "warnings": [],
    }
    for col in audit_cols:
        if col not in snapshot_df.columns:
            continue
        raw = pd.to_numeric(snapshot_df[col], errors="coerce")
        entry: dict[str, Any] = {
            "raw": _series_distribution(raw),
            "raw_scale": _numeric_scale_summary(snapshot_df[col]),
        }
        if col in percent_like:
            normalized = snapshot_df[col].map(_to_rate)
            norm_series = pd.to_numeric(normalized, errors="coerce")
            entry["normalized"] = _series_distribution(norm_series)
            entry["post_normalization_out_of_bounds"] = int(((norm_series < 0) | (norm_series > 1.5)).sum())
            if entry["raw_scale"].get("scale") == "MIXED_0_1_AND_0_100":
                payload["warnings"].append(f"{col}: mixed raw percent scale detected; normalized with _to_rate")
        payload["columns"][col] = entry
    _write_json(OUT_FEATURE_RANGE_AUDIT, payload)


def _write_join_audit(active: pd.DataFrame, snapshots: dict[str, dict[str, Any]], market_lookup: dict[str, dict[str, Any]]) -> pd.DataFrame:
    event_source = active.get("event_id", pd.Series("", index=active.index)).astype(str).str.strip()
    game_source = active.get("game_id", pd.Series("", index=active.index)).astype(str).str.strip()
    source_event = int(((event_source != "") & (event_source != "0")).sum())
    source_game = int((((event_source == "") | (event_source == "0")) & (game_source != "") & (game_source != "0")).sum())

    duplicate_key_counts = (
        active["canonical_game_key"].value_counts().loc[lambda s: s > 1] if "canonical_game_key" in active.columns else pd.Series(dtype=int)
    )
    duplicate_keys = int(len(duplicate_key_counts))

    rows: list[dict[str, Any]] = []
    for _, g in active.iterrows():
        gid = _canonical_id(g.get("canonical_game_key") or g.get("event_id") or g.get("game_id"))
        home_id = _canonical_id(g.get("home_team_id"))
        away_id = _canonical_id(g.get("away_team_id"))
        home_present = home_id in snapshots
        away_present = away_id in snapshots
        market_present = gid in market_lookup
        rows.append(
            {
                "event_id": gid,
                "game_id": gid,
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "home_snapshot_present": home_present,
                "away_snapshot_present": away_present,
                "market_lookup_present": market_present,
                "join_issue": "" if (home_present and away_present and market_present) else "MISSING_LOOKUP_OR_SNAPSHOT",
            }
        )
    join_df = pd.DataFrame(rows)
    mismatch_df = join_df[join_df["join_issue"] != ""].copy()
    mismatch_df.to_csv(OUT_JOIN_MISMATCH_SAMPLES, index=False)
    _log_written_csv(OUT_JOIN_MISMATCH_SAMPLES, mismatch_df)

    payload = {
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "canonical_key_policy": "event_id preferred; fallback to game_id; both normalized to canonical_game_key",
        "active_rows": int(len(active)),
        "active_unique_keys": int(active["canonical_game_key"].nunique()) if "canonical_game_key" in active.columns else int(len(active)),
        "duplicate_key_rows": int(duplicate_key_counts.sum()) if duplicate_keys > 0 else 0,
        "duplicate_key_count": duplicate_keys,
        "event_id_key_source_rows": source_event,
        "game_id_fallback_rows": source_game,
        "snapshot_both_sides_match_rate": float(
            (
                join_df["home_snapshot_present"].astype(bool)
                & join_df["away_snapshot_present"].astype(bool)
            ).mean()
            if not join_df.empty
            else 0.0
        ),
        "market_lookup_match_rate": float(join_df["market_lookup_present"].astype(bool).mean() if not join_df.empty else 0.0),
        "mismatch_rows": int(len(mismatch_df)),
    }
    if duplicate_keys > 0:
        payload["duplicate_key_examples"] = duplicate_key_counts.head(10).to_dict()
    _write_json(OUT_PRED_FEATURE_JOIN_AUDIT, payload)
    return mismatch_df


def _numeric_scale_summary(series: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return {"present": False, "scale": "MISSING", "rows": 0, "violations": 0}

    le_one = int((numeric <= 1.5).sum())
    gt_one = int((numeric > 1.5).sum())
    if le_one > 0 and gt_one > 0:
        scale = "MIXED_0_1_AND_0_100"
    elif gt_one > 0:
        scale = "PCT_0_100"
    else:
        scale = "RATE_0_1"

    return {
        "present": True,
        "scale": scale,
        "rows": int(len(numeric)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "violations": int((numeric < 0).sum()),
    }


def _feature_scale_checks(weighted: pd.DataFrame) -> dict[str, Any]:
    checks: dict[str, Any] = {"warnings": [], "columns": {}}
    percent_cols = ["efg_pct", "orb_pct", "drb_pct", "tov_pct", "ftr", "three_par"]
    for col in percent_cols:
        if col in weighted.columns:
            checks["columns"][col] = _numeric_scale_summary(weighted[col])
            scale = checks["columns"][col]["scale"]
            if scale == "MIXED_0_1_AND_0_100":
                checks["warnings"].append(f"{col}: mixed percent scale detected (0-1 and 0-100)")

    if "net_rtg" in weighted.columns:
        vals = pd.to_numeric(weighted["net_rtg"], errors="coerce")
        high_abs = vals.abs() > 80
        checks["columns"]["net_rtg"] = {
            "present": True,
            "rows": int(vals.notna().sum()),
            "abs_gt_80": int(high_abs.sum()),
        }
        if int(high_abs.sum()) > 0:
            checks["warnings"].append(f"net_rtg: {int(high_abs.sum())} rows have |net_rtg| > 80")
    return checks


def run_joint_predictions(
    games_path: Path = GAMES_PATH,
    weighted_path: Path = WEIGHTED_PATH,
    advanced_path: Path = ADVANCED_PATH,
    out_latest: Path = OUT_LATEST,
    out_snapshots: Path = OUT_SNAPSHOTS,
) -> pd.DataFrame:
    games = _normalize_key_cols(_read_csv(games_path))
    weighted = _normalize_key_cols(_read_csv(weighted_path))
    advanced = _normalize_key_cols(_read_csv(advanced_path))
    rotation = _normalize_key_cols(_read_csv(ROTATION_PATH))

    if games.empty:
        raise ValueError("games.csv is missing or empty")
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    scale_checks = _feature_scale_checks(weighted)
    _write_json(OUT_SCALE_CHECKS, scale_checks)
    if scale_checks.get("warnings"):
        log.warning(f"feature_scale_checks warnings={scale_checks['warnings']}")

    snapshots = _latest_team_snapshot(weighted, advanced, rotation)
    snapshot_frame = _write_team_snapshot_artifact(snapshots)
    if snapshot_frame.empty:
        raise RuntimeError("team_snapshot artifact is empty; expected producer inputs data/team_game_weighted.csv and data/advanced_metrics.csv")
    _feature_presence_gate(snapshot_frame)
    _write_feature_range_audit(snapshot_frame)
    _write_spread_calculation_spec()
    market_lookup = _build_market_lookup()

    if "completed" in games.columns:
        active = games[(games["completed"].astype(str).str.lower() != "true") | (games["state"].astype(str).str.lower() != "post")].copy()
        if active.empty:
            active = games.copy()
    else:
        active = games.copy()
    active["canonical_game_key"] = canonical_game_key(active)
    active = active[active["canonical_game_key"] != "0"].copy()
    active = active.sort_values(["game_datetime_utc", "canonical_game_key"]).drop_duplicates("canonical_game_key", keep="last")
    _write_join_audit(active, snapshots, market_lookup)

    rows: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    blocked_samples: list[dict[str, Any]] = []
    calc_traces: list[dict[str, Any]] = []
    generated_at = pd.Timestamp.utcnow().isoformat()
    for _, g in active.iterrows():
        gid = _canonical_id(g.get("canonical_game_key") or g.get("event_id") or g.get("game_id"))
        home_id = _canonical_id(g.get("home_team_id"))
        away_id = _canonical_id(g.get("away_team_id"))
        team_rows = {"home": snapshots.get(home_id, {}), "away": snapshots.get(away_id, {})}
        market_row = market_lookup.get(gid, {})

        if not team_rows["home"] or not team_rows["away"]:
            missing_pieces: list[str] = []
            if not team_rows["home"]:
                missing_pieces.append("home_team_snapshot")
            if not team_rows["away"]:
                missing_pieces.append("away_team_snapshot")
            status_text = "BLOCKED_MISSING_INPUT:team_snapshot"
            pred = {
                "pred_total": np.nan,
                "allocation_pct": np.nan,
                "pred_home_score": np.nan,
                "pred_away_score": np.nan,
                "pred_margin": np.nan,
                "pred_margin_raw": np.nan,
                "pred_margin_pre_guardrail": np.nan,
                "pred_margin_final": np.nan,
                "pred_total_ultimate": np.nan,
                "pred_margin_ultimate": np.nan,
                "spread_edge": np.nan,
                "total_edge": np.nan,
                "market_status": "MISSING_LINES",
                "feature_missing_rate": 1.0,
                "model_status": status_text,
                "pred_total_raw": np.nan,
                "allocation_raw": np.nan,
                "spread_signal_raw": np.nan,
                "spread_signal_tanh": np.nan,
                "spread_terms": [],
                "totals_terms": [],
                "adjustments_applied": [],
                "total_clipped": False,
                "totals_features": {},
                "spread_features": {},
            }
            pred["spread_calc_trace"] = build_spread_calc_trace(
                {
                    "event_id": gid,
                    "game_id": gid,
                    "home_team": g.get("home_team"),
                    "away_team": g.get("away_team"),
                    "model_status": status_text,
                    "pred_total": np.nan,
                    "pred_margin_raw": np.nan,
                    "pred_margin_pre_guardrail": np.nan,
                    "pred_margin_final": np.nan,
                    "allocation_raw": np.nan,
                    "allocation_pct": np.nan,
                    "spread_signal_raw": np.nan,
                    "spread_signal_tanh": np.nan,
                    "spread_terms": [],
                    "spread_features": {},
                    "missing_spread_features": missing_pieces,
                    "adjustments_applied": ["BLOCKED_MISSING_INPUT:team_snapshot"],
                },
                {
                    "model_name": MODEL_VERSION,
                    "model_version": MODEL_VERSION,
                    "weights": SPREAD_WEIGHTS,
                    "scales": SPREAD_FEATURE_SCALES,
                    "intercept": SPREAD_INTERCEPT,
                    "hca_points": SPREAD_HCA_POINTS,
                },
            )
            blocked_samples.append(
                {
                    "model_name": MODEL_VERSION,
                    "event_id": gid,
                    "game_id": gid,
                    "missing_inputs": ",".join(missing_pieces),
                    "expected_source_paths": "data/team_snapshot.csv|data/team_game_weighted.csv|data/advanced_metrics.csv",
                    "stage": "joint_models_predictions",
                    "timestamp_utc": generated_at,
                }
            )
        else:
            pred = predict_game_joint(g.to_dict(), team_rows, {}, market_row)
            model_status = str(pred.get("model_status", ""))
            if model_status.startswith("BLOCKED_MISSING_INPUT:"):
                blocked_samples.append(
                    {
                        "model_name": MODEL_VERSION,
                        "event_id": gid,
                        "game_id": gid,
                        "missing_inputs": model_status.split(":", 1)[1],
                        "expected_source_paths": "data/team_snapshot.csv|data/team_game_weighted.csv|data/advanced_metrics.csv",
                        "stage": "joint_models_predictions",
                        "timestamp_utc": generated_at,
                    }
                )
            elif model_status.startswith("BLOCKED_"):
                blocked_samples.append(
                    {
                        "model_name": MODEL_VERSION,
                        "event_id": gid,
                        "game_id": gid,
                        "missing_inputs": model_status,
                        "expected_source_paths": "data/team_snapshot.csv|data/team_game_weighted.csv|data/advanced_metrics.csv",
                        "stage": "joint_models_predictions",
                        "timestamp_utc": generated_at,
                    }
                )
        calc_trace = pred.get("spread_calc_trace")
        if isinstance(calc_trace, dict):
            calc_traces.append(calc_trace)

        pred_spread = pred["pred_margin"]
        sanity_flag = ""
        sanity_flag_total = ""
        if np.isfinite(_to_num(pred_spread)) and abs(float(pred_spread)) > MAX_ABS_SPREAD:
            sanity_flag = "SPREAD_OUT_OF_RANGE"
        total_for_gate = _to_num(pred.get("pred_total_raw"))
        if np.isfinite(total_for_gate) and (float(total_for_gate) < TOTAL_MIN or float(total_for_gate) > TOTAL_MAX):
            sanity_flag_total = "TOTAL_OUT_OF_RANGE"
        is_official = (
            str(pred.get("model_status", "")).startswith("OK")
            and sanity_flag == ""
            and sanity_flag_total == ""
            and np.isfinite(_to_num(pred.get("pred_home_score")))
            and np.isfinite(_to_num(pred.get("pred_away_score")))
        )

        if sanity_flag or sanity_flag_total:
            sanity_rows.append(
                {
                    "event_id": gid,
                    "game_id": gid,
                    "home_team": g.get("home_team"),
                    "away_team": g.get("away_team"),
                    "pred_spread": pred_spread,
                    "pred_margin_raw": pred.get("pred_margin_raw"),
                    "pred_total": pred.get("pred_total"),
                    "pred_total_raw": pred.get("pred_total_raw"),
                    "allocation_pct": pred.get("allocation_pct"),
                    "spread_signal_raw": pred.get("spread_signal_raw"),
                    "sanity_flag": sanity_flag,
                    "sanity_flag_total": sanity_flag_total,
                    "odi_star_diff": pred.get("spread_features", {}).get("odi_star_diff"),
                    "sme_diff": pred.get("spread_features", {}).get("sme_diff"),
                    "posw": pred.get("spread_features", {}).get("posw"),
                    "pxp_diff": pred.get("spread_features", {}).get("pxp_diff"),
                    "lns_diff": pred.get("spread_features", {}).get("lns_diff"),
                    "vol_diff": pred.get("spread_features", {}).get("vol_diff"),
                    "sch": pred.get("totals_features", {}).get("sch"),
                    "wl": pred.get("totals_features", {}).get("wl"),
                    "svi_avg": pred.get("totals_features", {}).get("svi_avg"),
                    "tc_diff": pred.get("totals_features", {}).get("tc_diff"),
                }
            )

        rows.append(
            {
                "event_id": gid,
                "game_id": gid,
                "game_datetime_utc": g.get("game_datetime_utc"),
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "home_team_id": home_id,
                "away_team_id": away_id,
                "pred_total": pred["pred_total"],
                "pred_total_raw": pred.get("pred_total_raw"),
                "allocation_pct": pred["allocation_pct"],
                "allocation_raw": pred.get("allocation_raw"),
                "pred_home_score": pred["pred_home_score"],
                "pred_away_score": pred["pred_away_score"],
                "pred_margin": pred["pred_margin"],
                "pred_margin_raw": pred.get("pred_margin_raw"),
                "pred_margin_pre_guardrail": pred.get("pred_margin_pre_guardrail"),
                "pred_margin_final": pred.get("pred_margin_final", pred.get("pred_margin")),
                "pred_spread": pred_spread,
                "spread_signal_raw": pred.get("spread_signal_raw"),
                "legacy_spread_signal_raw": pred.get("legacy_spread_signal_raw"),
                "legacy_allocation_pct": pred.get("legacy_allocation_pct"),
                "legacy_pred_margin_raw": pred.get("legacy_pred_margin_raw"),
                "spread_signal_tanh": pred.get("spread_signal_tanh"),
                "sanity_flag": sanity_flag,
                "sanity_flag_total": sanity_flag_total,
                "is_official_prediction": is_official,
                "model_version": MODEL_VERSION,
                "generated_at_utc": generated_at,
                "feature_missing_rate": pred["feature_missing_rate"],
                "model_status": pred["model_status"],
                "market_status": pred.get("market_status", "MISSING_LINES"),
                "spread_line": market_row.get("spread_line"),
                "total_line": market_row.get("total_line"),
                "line_source_used": market_row.get("line_source_used"),
                "line_timestamp_utc": market_row.get("line_timestamp_utc"),
                "spread_edge": pred.get("spread_edge"),
                "total_edge": pred.get("total_edge"),
                "pred_total_ultimate": pred.get("pred_total_ultimate"),
                "pred_margin_ultimate": pred.get("pred_margin_ultimate"),
                "adjustments_applied": "|".join(pred.get("adjustments_applied", [])) if pred.get("adjustments_applied") else "",
                "blocked_stage": "joint_models_predictions" if str(pred.get("model_status", "")).startswith("BLOCKED_") else pd.NA,
                "blocked_expected_source_paths": "data/team_snapshot.csv|data/team_game_weighted.csv|data/advanced_metrics.csv" if str(pred.get("model_status", "")).startswith("BLOCKED_") else pd.NA,
                "blocked_missing_inputs": str(pred.get("model_status", "")).split(":", 1)[1] if str(pred.get("model_status", "")).startswith("BLOCKED_MISSING_INPUT:") else pd.NA,
            }
        )

    out = pd.DataFrame(rows).sort_values(["game_datetime_utc", "game_id"]).reset_index(drop=True)
    out = merge_market_lines(
        out,
        data_dir=DATA_DIR,
        output_name="predictions_joint_latest.csv",
        required_columns=["opening_spread", "closing_spread", "spread_line", "total_line", "line_source_used", "line_timestamp_utc", "market_status"],
        debug_dir=DEBUG_DIR,
    )
    spread_non_null = pd.to_numeric(out.get("spread_line"), errors="coerce").notna()
    total_non_null = pd.to_numeric(out.get("total_line"), errors="coerce").notna()
    out["line_status"] = "MISSING"
    out.loc[spread_non_null & total_non_null, "line_status"] = "OK"
    out.loc[spread_non_null ^ total_non_null, "line_status"] = "PARTIAL"
    out["line_missing_reason"] = pd.NA
    out.loc[out["line_status"] == "PARTIAL", "line_missing_reason"] = "PARTIAL"
    out.loc[(out["line_status"] == "MISSING") & out["line_source_used"].notna(), "line_missing_reason"] = "NO_ODDS_RETURNED"
    out.loc[(out["line_status"] == "MISSING") & out["line_source_used"].isna(), "line_missing_reason"] = "NO_MATCHING_GAME_ID"
    _append_blocked_samples(blocked_samples)
    _write_spread_calc_samples(calc_traces)
    _write_pred_margin_diagnostics(out)

    blocked_mask = out.get("model_status", pd.Series("", index=out.index)).astype(str).str.startswith("BLOCKED")
    blocked_numeric = out[
        blocked_mask
        & (
            pd.to_numeric(out.get("pred_margin"), errors="coerce").notna()
            | pd.to_numeric(out.get("pred_margin_final"), errors="coerce").notna()
        )
    ].copy()
    blocked_numeric.to_csv(OUT_BLOCKED_NUMERIC_PREDS, index=False)
    _log_written_csv(OUT_BLOCKED_NUMERIC_PREDS, blocked_numeric)
    if len(blocked_numeric) > 0:
        raise RuntimeError(
            "Blocked rows contain numeric pred_margin values. "
            f"See {OUT_BLOCKED_NUMERIC_PREDS}."
        )

    expected = active[
        [
            c
            for c in [
                "canonical_game_key",
                "event_id",
                "game_id",
                "game_datetime_utc",
                "home_team",
                "away_team",
                "home_team_id",
                "away_team_id",
            ]
            if c in active.columns
        ]
    ].copy()
    expected["canonical_game_key"] = expected["canonical_game_key"].map(_canonical_id)

    predicted = out[["game_id", "pred_margin", "model_status", "market_status"]].copy()
    predicted["canonical_game_key"] = predicted["game_id"].map(_canonical_id)
    coverage = expected.merge(
        predicted[["canonical_game_key", "pred_margin", "model_status", "market_status"]],
        on="canonical_game_key",
        how="left",
        indicator=True,
    )

    missing_rows: list[dict[str, Any]] = []
    for _, row in coverage.iterrows():
        reason = ""
        missing_cols = ""
        home_id = _canonical_id(row.get("home_team_id"))
        away_id = _canonical_id(row.get("away_team_id"))
        home_missing = home_id not in snapshots
        away_missing = away_id not in snapshots

        if str(row.get("_merge")) == "left_only":
            if home_missing or away_missing:
                reason = "MISSING_TEAM_SNAPSHOT"
                missing_cols = "home_snapshot,away_snapshot" if home_missing and away_missing else ("home_snapshot" if home_missing else "away_snapshot")
            else:
                reason = "ID_MISMATCH_EVENT_GAME_KEY"
        elif pd.isna(pd.to_numeric(row.get("pred_margin"), errors="coerce")):
            status = str(row.get("model_status", ""))
            if status.startswith("BLOCKED_MISSING_INPUT:"):
                reason = "MISSING_REQUIRED_FEATURES"
                missing_cols = status.split(":", 1)[1]
            elif status.startswith("BLOCKED_"):
                reason = "DROPPED_BY_MODEL_GATE"
            else:
                reason = "UNKNOWN_NO_NUMERIC_PREDICTION"

        if reason:
            missing_rows.append(
                {
                    "event_id": _canonical_id(row.get("event_id") or row.get("canonical_game_key")),
                    "game_id": _canonical_id(row.get("game_id") or row.get("canonical_game_key")),
                    "home_team": row.get("home_team"),
                    "away_team": row.get("away_team"),
                    "start_time": row.get("game_datetime_utc"),
                    "reason_bucket": reason,
                    "missing_columns_list": missing_cols,
                }
            )

    missing_report = pd.DataFrame(
        missing_rows,
        columns=[
            "event_id",
            "game_id",
            "home_team",
            "away_team",
            "start_time",
            "reason_bucket",
            "missing_columns_list",
        ],
    )
    missing_report.to_csv(OUT_MISSING, index=False)
    _log_written_csv(OUT_MISSING, missing_report)

    sanity_outliers = pd.DataFrame(sanity_rows)
    if sanity_outliers.empty:
        sanity_outliers = pd.DataFrame(
            columns=[
                "event_id",
                "game_id",
                "home_team",
                "away_team",
                "pred_spread",
                "pred_margin_raw",
                "pred_total",
                "pred_total_raw",
                "allocation_pct",
                "spread_signal_raw",
                "sanity_flag",
                "sanity_flag_total",
                "odi_star_diff",
                "sme_diff",
                "posw",
                "pxp_diff",
                "lns_diff",
                "vol_diff",
                "sch",
                "wl",
                "svi_avg",
                "tc_diff",
            ]
        )
    sanity_outliers.to_csv(OUT_SANITY, index=False)
    _log_written_csv(OUT_SANITY, sanity_outliers)
    if len(sanity_outliers) > 0:
        log.warning("sanity_outliers_detected count=%s path=%s", len(sanity_outliers), OUT_SANITY.resolve())

    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_latest, index=False)
    _log_written_csv(out_latest, out)

    if out_snapshots.exists() and out_snapshots.stat().st_size > 0:
        snap_prev = pd.read_csv(out_snapshots, low_memory=False)
        snap = pd.concat([snap_prev, out], ignore_index=True)
    else:
        snap = out.copy()
    snap.to_csv(out_snapshots, index=False)
    _log_written_csv(out_snapshots, snap)

    null_rate = out[["pred_total", "allocation_pct", "pred_margin"]].isna().mean().to_dict() if not out.empty else {}
    status_counts = out["model_status"].value_counts(dropna=False).to_dict() if "model_status" in out.columns else {}
    eligible_games = int(expected["canonical_game_key"].nunique()) if not expected.empty else 0
    predicted_games = int(out[pd.to_numeric(out["pred_margin"], errors="coerce").notna()]["game_id"].nunique()) if not out.empty else 0
    missing_predictions_count = max(0, eligible_games - predicted_games)
    spread_dist = _series_distribution(pd.to_numeric(out.get("pred_margin"), errors="coerce"))
    spread_raw_dist = _series_distribution(pd.to_numeric(out.get("pred_margin_raw"), errors="coerce"))
    log.info(
        "joint_predictions rows=%s null_rates=%s status_counts=%s eligible_games=%s predicted_games=%s missing_predictions=%s",
        len(out),
        null_rate,
        status_counts,
        eligible_games,
        predicted_games,
        missing_predictions_count,
    )
    log.info(
        "Spread Calculation Confirmation | formula=pred_margin_raw=pred_total*2*clip(tanh(signal_raw)*%.3f,[%.2f,%.2f])+intercept+hca | intercept=%.2f | hca=%.2f | perspective=HOME+",
        ALLOCATION_TANH_SCALE,
        ALLOCATION_BOUNDS[0],
        ALLOCATION_BOUNDS[1],
        SPREAD_INTERCEPT,
        SPREAD_HCA_POINTS,
    )
    log.info("Spread distribution final=%s raw=%s", spread_dist, spread_raw_dist)
    return out


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build sequential joint totals+spread predictions")
    p.add_argument("--games", default=str(GAMES_PATH))
    p.add_argument("--weighted", default=str(WEIGHTED_PATH))
    p.add_argument("--advanced", default=str(ADVANCED_PATH))
    p.add_argument("--out-latest", default=str(OUT_LATEST))
    p.add_argument("--out-snapshots", default=str(OUT_SNAPSHOTS))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    run_joint_predictions(
        games_path=Path(args.games),
        weighted_path=Path(args.weighted),
        advanced_path=Path(args.advanced),
        out_latest=Path(args.out_latest),
        out_snapshots=Path(args.out_snapshots),
    )


if __name__ == "__main__":
    main()
