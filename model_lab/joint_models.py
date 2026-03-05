from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from pipeline.advanced_metrics import advanced_metrics_formulas as fm

log = get_logger(__name__)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

GAMES_PATH = DATA_DIR / "games.csv"
WEIGHTED_PATH = DATA_DIR / "team_game_weighted.csv"
ADVANCED_PATH = DATA_DIR / "advanced_metrics.csv"
ROTATION_PATH = DATA_DIR / "rotation_features.csv"
MARKET_LATEST_PATH = DATA_DIR / "market_lines_latest.csv"
MARKET_CLOSING_PATH = DATA_DIR / "market_lines_closing.csv"

OUT_LATEST = DATA_DIR / "predictions_joint_latest.csv"
OUT_SNAPSHOTS = DATA_DIR / "predictions_joint_snapshots.csv"

MODEL_VERSION = "joint_v1.0"
TOTAL_BOUNDS = (110.0, 175.0)
ALLOCATION_BOUNDS = (-0.20, 0.20)
TOTAL_ADJUST_SCALE = 6.0

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


def _adv_metric(team_row: dict[str, Any], metric_col: str, acronym: str) -> float:
    status = str(team_row.get(f"adv_metric_status_{acronym}", ""))
    val = _to_num(team_row.get(f"adv_{metric_col}"))
    if status == "OK" and np.isfinite(val):
        return val
    return np.nan


def _team_fallback_features(team_row: dict[str, Any], opp_row: dict[str, Any]) -> dict[str, float]:
    efg = _to_rate(team_row.get("efg_pct"))
    tov = _to_rate(team_row.get("tov_pct"))
    orb = _to_rate(team_row.get("orb_pct"))
    ftr = _to_rate(team_row.get("ftr"))
    pace = _to_num(team_row.get("pace"))

    opp_efg = _to_rate(opp_row.get("efg_pct"))
    opp_tov = _to_rate(opp_row.get("tov_pct"))
    opp_orb = _to_rate(opp_row.get("orb_pct"))
    opp_ftr = _to_rate(opp_row.get("ftr"))
    opp_pace = _to_num(opp_row.get("pace"))

    odi = fm.odi_star(
        (efg - 0.505) - (opp_efg - 0.505) if np.isfinite(efg) and np.isfinite(opp_efg) else np.nan,
        (opp_tov - 0.18) - (tov - 0.18) if np.isfinite(tov) and np.isfinite(opp_tov) else np.nan,
        (orb - 0.30) - (opp_orb - 0.30) if np.isfinite(orb) and np.isfinite(opp_orb) else np.nan,
        (ftr - 0.28) - (opp_ftr - 0.28) if np.isfinite(ftr) and np.isfinite(opp_ftr) else np.nan,
    )

    pei = fm.pei(orb, opp_orb, opp_tov, tov) if np.isfinite(orb) and np.isfinite(opp_orb) and np.isfinite(opp_tov) and np.isfinite(tov) else np.nan
    projected_pace = (pace + opp_pace) / 2.0 if np.isfinite(pace) and np.isfinite(opp_pace) else np.nan
    posw = fm.posw(pei, projected_pace) if np.isfinite(pei) and np.isfinite(projected_pace) else np.nan

    ftm = _to_num(team_row.get("ftm"))
    fga = _to_num(team_row.get("fga"))
    ft_pts_per_fga = ftm / fga if np.isfinite(ftm) and np.isfinite(fga) and fga != 0 else np.nan
    three_pa = _to_rate(team_row.get("three_par"))
    svi = fm.svi(
        fm.safe_zscore(efg, 0.505, 0.04),
        fm.safe_zscore(three_pa, 0.38, 0.08),
        fm.safe_zscore(ft_pts_per_fga, 0.22, 0.06),
    ) if np.isfinite(efg) and np.isfinite(three_pa) and np.isfinite(ft_pts_per_fga) else np.nan

    rotation_stability = 1.0 - _to_num(team_row.get("rot_rot_minshare_sd"))
    rotation_stability = float(np.clip(rotation_stability, 0.0, 1.0)) if np.isfinite(rotation_stability) else 0.5
    ics = fm.ics(0.5, rotation_stability, 0.5)
    bsi = fm.bsi(0.33, 0.0, 0.0)
    clutch = _to_rate(team_row.get("close_game_win_pct"))
    pxp = fm.pxp(ics, bsi, clutch if np.isfinite(clutch) else 0.5)

    lns = _to_num(team_row.get("form_rating"))
    if not np.isfinite(lns):
        lns = _to_num(team_row.get("net_rtg_l10"))

    vol = _to_num(team_row.get("net_rtg_std_l10"))

    pace_l10 = _to_num(team_row.get("pace_l10"))
    tc = fm.tc(fm.safe_zscore(abs(pace - pace_l10), 0.0, 4.0)) if np.isfinite(pace) and np.isfinite(pace_l10) else np.nan

    ft_ppp = ftm / _to_num(team_row.get("poss")) if np.isfinite(ftm) and np.isfinite(_to_num(team_row.get("poss"))) and _to_num(team_row.get("poss")) != 0 else np.nan
    opp_ftm = _to_num(opp_row.get("ftm"))
    opp_poss = _to_num(opp_row.get("poss"))
    opp_ft_ppp = opp_ftm / opp_poss if np.isfinite(opp_ftm) and np.isfinite(opp_poss) and opp_poss != 0 else np.nan
    wl = fm.wl(fm.safe_zscore(ft_ppp - opp_ft_ppp, 0.0, 0.08), fm.safe_zscore(ftr - opp_ftr, 0.0, 0.08)) if np.isfinite(ft_ppp) and np.isfinite(opp_ft_ppp) and np.isfinite(ftr) and np.isfinite(opp_ftr) else np.nan

    size = _to_num(team_row.get("rot_rot_size"))
    if not np.isfinite(size):
        size = 8.0

    rest_days = _to_num(team_row.get("rest_days"))
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
        "travel_miles": _to_num(team_row.get("estimated_travel_miles")),
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


def _weighted_signal(features: dict[str, float], weights: dict[str, float]) -> tuple[float, list[str], float]:
    missing: list[str] = []
    num = 0.0
    den = 0.0
    for name, w in weights.items():
        v = _to_num(features.get(name))
        if not np.isfinite(v):
            missing.append(name)
            continue
        num += v * w
        den += w
    if den <= 0:
        return np.nan, missing, 0.0
    return num / den, missing, den


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

    totals_signal, missing_totals, totals_cov = _weighted_signal(totals_features, TOTALS_WEIGHTS)
    spread_signal, missing_spread, spread_cov = _weighted_signal(spread_features, SPREAD_WEIGHTS)

    required_count = len(TOTALS_WEIGHTS) + len(SPREAD_WEIGHTS)
    missing_count = len(missing_totals) + len(missing_spread)
    feature_missing_rate = missing_count / required_count if required_count else 0.0

    model_status = "OK"
    if totals_cov < 0.5 or spread_cov < 0.5:
        names = sorted(set(missing_totals + missing_spread))
        model_status = f"BLOCKED_MISSING_INPUT:{','.join(names) if names else 'insufficient_feature_weight'}"
        return {
            "pred_total": np.nan,
            "allocation_pct": np.nan,
            "pred_home_score": np.nan,
            "pred_away_score": np.nan,
            "pred_margin": np.nan,
            "pred_total_ultimate": np.nan,
            "pred_margin_ultimate": np.nan,
            "feature_missing_rate": feature_missing_rate,
            "model_status": model_status,
            "totals_features": totals_features,
            "spread_features": spread_features,
        }

    base_total = _base_total(team_rows)
    pred_total = base_total + (totals_signal * TOTAL_ADJUST_SCALE)
    clipped_total = float(np.clip(pred_total, TOTAL_BOUNDS[0], TOTAL_BOUNDS[1]))
    if clipped_total != pred_total:
        log.info(f"pred_total clipped game_id={game_row.get('game_id')} raw={pred_total:.2f} clipped={clipped_total:.2f}")
    pred_total = clipped_total

    allocation_pct = float(np.clip(spread_signal, ALLOCATION_BOUNDS[0], ALLOCATION_BOUNDS[1]))
    pred_home_score = pred_total * (0.5 + allocation_pct)
    pred_away_score = pred_total * (0.5 - allocation_pct)
    pred_margin = pred_home_score - pred_away_score

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

    u_spread, _, u_cov_spread = _weighted_signal(spread_features, ULTIMATE_SPREAD_WEIGHTS)
    pred_margin_ultimate = float(np.clip(u_spread, ALLOCATION_BOUNDS[0], ALLOCATION_BOUNDS[1]) * pred_total * 2.0) if u_cov_spread > 0 else np.nan

    market_status = "MISSING_LINES"
    spread_edge = np.nan
    total_edge = np.nan
    if market_row:
        spread_line = _to_num(market_row.get("spread_line"))
        total_line = _to_num(market_row.get("total_line"))
        if np.isfinite(spread_line) or np.isfinite(total_line):
            market_status = "OK"
            if np.isfinite(spread_line):
                spread_edge = pred_margin - spread_line
            if np.isfinite(total_line):
                total_edge = pred_total - total_line

    return {
        "pred_total": pred_total,
        "allocation_pct": allocation_pct,
        "pred_home_score": pred_home_score,
        "pred_away_score": pred_away_score,
        "pred_margin": pred_margin,
        "pred_total_ultimate": pred_total_ultimate,
        "pred_margin_ultimate": pred_margin_ultimate,
        "spread_edge": spread_edge,
        "total_edge": total_edge,
        "market_status": market_status,
        "feature_missing_rate": feature_missing_rate,
        "model_status": model_status,
        "totals_features": totals_features,
        "spread_features": spread_features,
    }


def _build_market_lookup() -> dict[str, dict[str, Any]]:
    for path in [MARKET_LATEST_PATH, MARKET_CLOSING_PATH]:
        df = _normalize_key_cols(_read_csv(path))
        if df.empty:
            continue
        spread_col = next((c for c in ["spread_line", "spread", "home_spread_current", "home_spread"] if c in df.columns), None)
        total_col = next((c for c in ["total_line", "over_under", "total_current", "total"] if c in df.columns), None)
        if spread_col is None and total_col is None:
            continue
        out: dict[str, dict[str, Any]] = {}
        for _, r in df.iterrows():
            gid = _canonical_id(r.get("game_id") or r.get("event_id"))
            if not gid:
                continue
            out[gid] = {
                "spread_line": r.get(spread_col) if spread_col else np.nan,
                "total_line": r.get(total_col) if total_col else np.nan,
            }
        if out:
            return out
    return {}


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
    snapshots = _latest_team_snapshot(weighted, advanced, rotation)
    market_lookup = _build_market_lookup()

    if "completed" in games.columns:
        active = games[(games["completed"].astype(str).str.lower() != "true") | (games["state"].astype(str).str.lower() != "post")].copy()
        if active.empty:
            active = games.copy()
    else:
        active = games.copy()
    active = active.sort_values(["game_datetime_utc", "game_id"]).drop_duplicates("game_id", keep="last")

    rows: list[dict[str, Any]] = []
    generated_at = pd.Timestamp.utcnow().isoformat()
    for _, g in active.iterrows():
        gid = _canonical_id(g.get("game_id") or g.get("event_id"))
        home_id = _canonical_id(g.get("home_team_id"))
        away_id = _canonical_id(g.get("away_team_id"))
        team_rows = {"home": snapshots.get(home_id, {}), "away": snapshots.get(away_id, {})}

        if not team_rows["home"] or not team_rows["away"]:
            pred = {
                "pred_total": np.nan,
                "allocation_pct": np.nan,
                "pred_home_score": np.nan,
                "pred_away_score": np.nan,
                "pred_margin": np.nan,
                "pred_total_ultimate": np.nan,
                "pred_margin_ultimate": np.nan,
                "spread_edge": np.nan,
                "total_edge": np.nan,
                "market_status": "MISSING_LINES",
                "feature_missing_rate": 1.0,
                "model_status": "BLOCKED_MISSING_INPUT:team_snapshot",
            }
        else:
            pred = predict_game_joint(g.to_dict(), team_rows, {}, market_lookup.get(gid))

        rows.append(
            {
                "game_id": gid,
                "game_datetime_utc": g.get("game_datetime_utc"),
                "home_team": g.get("home_team"),
                "away_team": g.get("away_team"),
                "pred_total": pred["pred_total"],
                "allocation_pct": pred["allocation_pct"],
                "pred_home_score": pred["pred_home_score"],
                "pred_away_score": pred["pred_away_score"],
                "pred_margin": pred["pred_margin"],
                "model_version": MODEL_VERSION,
                "generated_at_utc": generated_at,
                "feature_missing_rate": pred["feature_missing_rate"],
                "model_status": pred["model_status"],
                "market_status": pred.get("market_status", "MISSING_LINES"),
                "spread_edge": pred.get("spread_edge"),
                "total_edge": pred.get("total_edge"),
                "pred_total_ultimate": pred.get("pred_total_ultimate"),
                "pred_margin_ultimate": pred.get("pred_margin_ultimate"),
            }
        )

    out = pd.DataFrame(rows).sort_values(["game_datetime_utc", "game_id"]).reset_index(drop=True)
    out_latest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_latest, index=False)

    if out_snapshots.exists() and out_snapshots.stat().st_size > 0:
        snap_prev = pd.read_csv(out_snapshots, low_memory=False)
        snap = pd.concat([snap_prev, out], ignore_index=True)
    else:
        snap = out.copy()
    snap.to_csv(out_snapshots, index=False)

    null_rate = out[["pred_total", "allocation_pct", "pred_margin"]].isna().mean().to_dict() if not out.empty else {}
    status_counts = out["model_status"].value_counts(dropna=False).to_dict() if "model_status" in out.columns else {}
    log.info(f"joint_predictions rows={len(out)} null_rates={null_rate} status_counts={status_counts}")
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
