"""Build Advanced Metrics Codex table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config.logging_config import get_logger
from pipeline.advanced_metrics import advanced_metrics_formulas as f

log = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

TEAM_GAME_METRICS = DATA_DIR / "team_game_metrics.csv"
TEAM_GAME_WEIGHTED = DATA_DIR / "team_game_weighted.csv"
PLAYER_GAME_METRICS = DATA_DIR / "player_game_metrics.csv"
ROTATION_FEATURES = DATA_DIR / "rotation_features.csv"
SITUATIONAL_FEATURES = DATA_DIR / "situational_features.csv"
TRAVEL_FATIGUE = DATA_DIR / "team_travel_fatigue.csv"
GAMES = DATA_DIR / "games.csv"
DEFAULT_OUTPUT = DATA_DIR / "advanced_metrics.csv"

METRIC_SPECS: list[tuple[str, str]] = [
    ("ODI", "ODI_star"),
    ("PEI", "PEI"),
    ("POSW", "POSW"),
    ("SVI", "SVI"),
    ("PXP", "PXP"),
    ("LNS", "LNS"),
    ("USEF", "USEF"),
    ("DPC", "DPC"),
    ("FII", "FII"),
    ("SME", "SME"),
    ("SCH", "SCH"),
    ("VOL", "VOL"),
    ("TC", "TC"),
    ("WL", "WL"),
    ("RFD", "RFD"),
    ("GSR", "GSR"),
    ("ALT", "ALT"),
]


def _normalize_id(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.str.replace(r"\.0$", "", regex=True)
    out = out.str.lstrip("0")
    out = out.replace("", "0")
    return out


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "event_id" not in out.columns and "game_id" in out.columns:
        out["event_id"] = out["game_id"]
    if "game_id" not in out.columns and "event_id" in out.columns:
        out["game_id"] = out["event_id"]
    for col in ["event_id", "game_id", "team_id", "opponent_id", "home_team_id", "away_team_id"]:
        if col in out.columns:
            out[col] = _normalize_id(out[col])
    return out


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        log.warning(f"optional input missing: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # pragma: no cover
        log.warning(f"failed to read {path}: {exc}")
        return pd.DataFrame()


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _parse_bool(series: pd.Series) -> pd.Series:
    txt = series.astype(str).str.strip().str.lower()
    return txt.isin({"1", "true", "t", "yes", "y"})


def _first_present(df: pd.DataFrame, candidates: list[str], default: float = np.nan) -> pd.Series:
    out = pd.Series(default, index=df.index, dtype=float)
    for col in candidates:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            out = out.where(out.notna(), values)
    return out


def _wavg(frame: pd.DataFrame, value_col: str, weight_col: str) -> float:
    vv = pd.to_numeric(frame[value_col], errors="coerce")
    ww = pd.to_numeric(frame[weight_col], errors="coerce")
    den = ww.sum(min_count=1)
    if not np.isfinite(den) or den <= 0:
        return float(vv.mean())
    return float((vv * ww).sum(min_count=1) / den)


def _build_player_aggregates(player_df: pd.DataFrame) -> pd.DataFrame:
    if player_df.empty:
        return pd.DataFrame(columns=["event_id", "team_id"])

    p = _normalize_keys(player_df)
    required = {"event_id", "team_id", "athlete_id"}
    if not required.issubset(set(p.columns)):
        return pd.DataFrame(columns=["event_id", "team_id"])

    for col in ["min", "usage_rate", "ts_pct", "tpa", "fga", "reb", "plus_minus"]:
        if col not in p.columns:
            p[col] = np.nan
    if "starter" not in p.columns:
        p["starter"] = False
    if "game_datetime_utc" not in p.columns:
        p["game_datetime_utc"] = pd.NaT

    p = _safe_numeric(p, ["min", "usage_rate", "ts_pct", "tpa", "fga", "reb", "plus_minus"])
    p["starter_bool"] = _parse_bool(p["starter"])
    p["game_datetime_utc"] = pd.to_datetime(p["game_datetime_utc"], utc=True, errors="coerce")
    p["season"] = p["game_datetime_utc"].dt.year.where(p["game_datetime_utc"].dt.month < 10, p["game_datetime_utc"].dt.year + 1)

    rows: list[dict[str, object]] = []
    for (event_id, team_id), g in p.groupby(["event_id", "team_id"], dropna=False):
        g2 = g.copy()
        total_min = pd.to_numeric(g2["min"], errors="coerce").sum(min_count=1)
        if not np.isfinite(total_min) or total_min <= 0:
            total_min = np.nan
        g2["minutes_share"] = g2["min"] / total_min
        g2["player_tois"] = g2["usage_rate"] * g2["ts_pct"]

        top_usage = g2.sort_values("usage_rate", ascending=False).head(3)
        usage_sum = pd.to_numeric(top_usage["usage_rate"], errors="coerce").sum(min_count=1)
        if np.isfinite(usage_sum) and usage_sum > 0:
            weighted_ts_top = float((top_usage["ts_pct"] * top_usage["usage_rate"]).sum(min_count=1) / usage_sum)
        else:
            weighted_ts_top = float(pd.to_numeric(top_usage["ts_pct"], errors="coerce").mean())

        star = g2.sort_values("usage_rate", ascending=False).head(1)
        star_id = star["athlete_id"].iloc[0] if not star.empty else np.nan
        star_usage = float(pd.to_numeric(star["usage_rate"], errors="coerce").iloc[0]) if not star.empty else np.nan
        star_ts = float(pd.to_numeric(star["ts_pct"], errors="coerce").iloc[0]) if not star.empty else np.nan
        star_fga = float(pd.to_numeric(star["fga"], errors="coerce").iloc[0]) if not star.empty else np.nan
        star_tpa = float(pd.to_numeric(star["tpa"], errors="coerce").iloc[0]) if not star.empty else np.nan
        star_three_pa_rate = star_tpa / star_fga if np.isfinite(star_fga) and star_fga > 0 else np.nan

        bench = g2[~g2["starter_bool"]].copy()
        starters = g2[g2["starter_bool"]].copy()
        bench_minutes = float(pd.to_numeric(bench["min"], errors="coerce").sum(min_count=1) / total_min) if np.isfinite(total_min) and total_min > 0 else np.nan
        bench_ts = _wavg(bench, "ts_pct", "min") if not bench.empty else np.nan
        starter_ts = _wavg(starters, "ts_pct", "min") if not starters.empty else np.nan
        bench_ts_rel = bench_ts - starter_ts if np.isfinite(bench_ts) and np.isfinite(starter_ts) else np.nan
        bench_reb_per_min = _wavg(bench, "reb", "min") if not bench.empty else np.nan
        starter_reb_per_min = _wavg(starters, "reb", "min") if not starters.empty else np.nan
        bench_reb_rel = bench_reb_per_min - starter_reb_per_min if np.isfinite(bench_reb_per_min) and np.isfinite(starter_reb_per_min) else np.nan

        rows.append(
            {
                "event_id": event_id,
                "team_id": team_id,
                "lns_player": f.lns(g2["minutes_share"], g2["player_tois"]),
                "weighted_ts_top_usage_cluster": weighted_ts_top,
                "star_athlete_id": star_id,
                "star_usage": star_usage,
                "star_ts": star_ts,
                "star_three_pa_rate": star_three_pa_rate,
                "bench_minutes": bench_minutes,
                "bench_ts_rel": bench_ts_rel,
                "bench_reb_rel": bench_reb_rel,
                "best_rotation_netrtg": float(pd.to_numeric(g2["plus_minus"], errors="coerce").quantile(0.75)),
                "worst_rotation_netrtg": float(pd.to_numeric(g2["plus_minus"], errors="coerce").quantile(0.25)),
                "backup_quality_key_positions": float(pd.to_numeric(bench["plus_minus"], errors="coerce").mean()) if not bench.empty else np.nan,
                "rotation_stability": float(np.clip(1.0 - float(pd.to_numeric(g2["minutes_share"], errors="coerce").std(ddof=0)), 0.0, 1.0)),
                "team_ts_player": _wavg(g2, "ts_pct", "min"),
                "player_total_min": total_min,
                "game_datetime_utc": g2["game_datetime_utc"].dropna().min(),
                "season": g2["season"].dropna().iloc[0] if g2["season"].notna().any() else np.nan,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["team_id", "season", "game_datetime_utc", "event_id"]).reset_index(drop=True)
    out["returning_minutes"] = np.nan
    out["star_continuity"] = np.nan
    for (_, _), idx in out.groupby(["team_id", "season"], dropna=False).groups.items():
        sub = out.loc[idx].sort_values(["game_datetime_utc", "event_id"])
        prev_players: set[str] | None = None
        prev_star: str | None = None
        for ridx in sub.index:
            event = out.at[ridx, "event_id"]
            team = out.at[ridx, "team_id"]
            g = p[(p["event_id"] == event) & (p["team_id"] == team)].copy()
            cur_players = set(g["athlete_id"].astype(str))
            if prev_players is None:
                out.at[ridx, "returning_minutes"] = np.nan
                out.at[ridx, "star_continuity"] = np.nan
            else:
                cur_total = pd.to_numeric(g["min"], errors="coerce").sum(min_count=1)
                returning = pd.to_numeric(g.loc[g["athlete_id"].astype(str).isin(prev_players), "min"], errors="coerce").sum(min_count=1)
                if np.isfinite(cur_total) and cur_total > 0 and np.isfinite(returning):
                    out.at[ridx, "returning_minutes"] = float(returning / cur_total)
                else:
                    out.at[ridx, "returning_minutes"] = np.nan
                cur_star = str(out.at[ridx, "star_athlete_id"]) if pd.notna(out.at[ridx, "star_athlete_id"]) else None
                out.at[ridx, "star_continuity"] = float(cur_star == prev_star) if (cur_star and prev_star) else np.nan
            prev_players = cur_players
            prev_star = str(out.at[ridx, "star_athlete_id"]) if pd.notna(out.at[ridx, "star_athlete_id"]) else None
    return out


def _first_missing_col(df: pd.DataFrame, required_cols: list[str]) -> pd.Series:
    first_missing = pd.Series("", index=df.index, dtype=object)
    for col in required_cols:
        m = first_missing.eq("") & df[col].isna()
        first_missing.loc[m] = col
    first_missing = first_missing.replace("", "null_required")
    return first_missing.astype(str)


def _apply_metric(
    df: pd.DataFrame,
    *,
    acronym: str,
    metric_col: str,
    required_cols: list[str],
    compute_fn: Callable[[pd.DataFrame], pd.Series],
) -> pd.DataFrame:
    status_col = f"metric_status_{acronym}"
    out = df.copy()
    out[metric_col] = np.nan
    missing_cols = [c for c in required_cols if c not in out.columns]
    if missing_cols:
        out[status_col] = f"BLOCKED_MISSING_INPUT:missing_columns={','.join(missing_cols)}"
        return out

    valid = out[required_cols].notna().all(axis=1)
    if valid.any():
        vals = compute_fn(out.loc[valid].copy())
        out.loc[valid, metric_col] = pd.to_numeric(vals, errors="coerce")

    status = pd.Series("OK", index=out.index, dtype=object)
    blocked_reason = "BLOCKED_MISSING_INPUT:" + _first_missing_col(out, required_cols)
    status.loc[~valid] = blocked_reason.loc[~valid]
    status.loc[valid & out[metric_col].isna()] = "BLOCKED_MISSING_INPUT:compute_nan"
    out[status_col] = status
    return out


def _prepare_base_table() -> pd.DataFrame:
    team = _read_csv(TEAM_GAME_METRICS)
    if team.empty:
        return pd.DataFrame()
    team = _normalize_keys(team)
    if "game_datetime_utc" in team.columns:
        team["game_datetime_utc"] = pd.to_datetime(team["game_datetime_utc"], utc=True, errors="coerce")
        team["date"] = team["game_datetime_utc"].dt.date.astype("string")
    else:
        team["date"] = pd.Series(pd.NA, index=team.index, dtype="string")
    if "season" not in team.columns and "game_datetime_utc" in team.columns:
        dt = pd.to_datetime(team["game_datetime_utc"], utc=True, errors="coerce")
        team["season"] = dt.dt.year.where(dt.dt.month < 10, dt.dt.year + 1)
    return team


def _attach_optional_inputs(base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    weighted = _normalize_keys(_read_csv(TEAM_GAME_WEIGHTED))
    if not weighted.empty:
        keep = [c for c in ["event_id", "team_id", "adj_net_rtg", "ts_pct", "close_game_win_pct", "pace", "spread"] if c in weighted.columns]
        if {"event_id", "team_id"}.issubset(set(keep)):
            out = out.merge(weighted[keep], on=["event_id", "team_id"], how="left", suffixes=("", "_w"))

    rotation = _normalize_keys(_read_csv(ROTATION_FEATURES))
    if not rotation.empty:
        keep = [c for c in ["event_id", "game_id", "team_id", "rot_size", "top2_pused_share", "rot_minshare_sd"] if c in rotation.columns]
        if "event_id" not in keep and "game_id" in keep:
            rotation["event_id"] = rotation["game_id"]
            keep.append("event_id")
        if {"event_id", "team_id"}.issubset(set(keep)):
            out = out.merge(rotation[keep].drop(columns=["game_id"], errors="ignore"), on=["event_id", "team_id"], how="left")

    situ = _normalize_keys(_read_csv(SITUATIONAL_FEATURES))
    if not situ.empty:
        keep = [c for c in [
            "event_id", "game_id", "team_id", "home_rest_days", "away_rest_days", "rest_delta",
            "must_win_flag", "bubble_pressure_flag", "late_season_flag",
        ] if c in situ.columns]
        if "event_id" not in keep and "game_id" in keep:
            situ["event_id"] = situ["game_id"]
            keep.append("event_id")
        if {"event_id", "team_id"}.issubset(set(keep)):
            out = out.merge(situ[keep].drop(columns=["game_id"], errors="ignore"), on=["event_id", "team_id"], how="left")

    travel = _normalize_keys(_read_csv(TRAVEL_FATIGUE))
    if not travel.empty:
        keep = [c for c in ["event_id", "team_id", "rest_days", "estimated_travel_miles", "is_back_to_back"] if c in travel.columns]
        if {"event_id", "team_id"}.issubset(set(keep)):
            out = out.merge(travel[keep], on=["event_id", "team_id"], how="left")

    games = _normalize_keys(_read_csv(GAMES))
    if not games.empty:
        keep = [c for c in ["event_id", "game_id", "spread", "home_team_id", "away_team_id"] if c in games.columns]
        if "event_id" not in keep and "game_id" in keep:
            games["event_id"] = games["game_id"]
            keep.append("event_id")
        if "event_id" in keep:
            out = out.merge(games[keep].drop_duplicates("event_id").drop(columns=["game_id"], errors="ignore"), on="event_id", how="left", suffixes=("", "_g"))

    players = _build_player_aggregates(_read_csv(PLAYER_GAME_METRICS))
    if not players.empty:
        keep = [c for c in players.columns if c in {
            "event_id", "team_id", "lns_player", "weighted_ts_top_usage_cluster", "star_usage", "star_ts",
            "star_three_pa_rate", "bench_minutes", "bench_ts_rel", "bench_reb_rel", "best_rotation_netrtg",
            "worst_rotation_netrtg", "backup_quality_key_positions", "rotation_stability", "returning_minutes",
            "star_continuity", "team_ts_player",
        }]
        out = out.merge(players[keep], on=["event_id", "team_id"], how="left")
    return out


def _attach_opponent_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_cols = [c for c in [
        "event_id", "team_id", "poss", "3PA_rate", "FTr", "FT_pts_per_poss", "TOV%", "ORB%", "eFG",
        "NetRtg", "adj_net_rtg", "rest_days", "is_back_to_back", "estimated_travel_miles",
        "lineup_height_avg", "star_three_pa_rate", "spread",
    ] if c in out.columns]
    if {"event_id", "team_id"}.issubset(set(base_cols)):
        opp = out[base_cols].rename(columns={"team_id": "opponent_id"})
        rename_map = {c: f"opp_{c}" for c in base_cols if c not in {"event_id", "team_id"}}
        opp = opp.rename(columns=rename_map)
        out = out.merge(opp, on=["event_id", "opponent_id"], how="left")
    return out


def _compute_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    required_seed_cols = [
        "season", "eFG", "TOV%", "ORB%", "FTr", "3PA_rate", "FT_pts_per_FGA", "FT_pts_per_poss",
        "poss", "NetRtg", "adj_net_rtg", "opp_eFG", "opp_TOV%", "opp_ORB%", "opp_FTr", "opp_poss",
        "opp_3PA_rate", "opp_FT_pts_per_poss", "spread", "bench_minutes", "bench_ts_rel",
        "bench_reb_rel", "returning_minutes", "rotation_stability", "star_continuity",
        "close_game_win_pct", "lns_player", "star_usage", "star_ts", "team_ts_player",
        "weighted_ts_top_usage_cluster", "best_rotation_netrtg", "worst_rotation_netrtg",
        "backup_quality_key_positions", "FFC", "must_win_flag", "bubble_pressure_flag",
        "late_season_flag", "home_rest_days", "away_rest_days", "rest_days",
        "estimated_travel_miles", "lineup_height_avg", "opp_lineup_height_avg",
    ]
    for col in required_seed_cols:
        if col not in out.columns:
            out[col] = np.nan
    out = _safe_numeric(
        out,
        [
            "eFG", "TOV%", "ORB%", "FTr", "3PA_rate", "FT_pts_per_FGA", "FT_pts_per_poss", "poss", "NetRtg",
            "adj_net_rtg", "opp_eFG", "opp_TOV%", "opp_ORB%", "opp_FTr", "opp_poss", "opp_3PA_rate",
            "opp_FT_pts_per_poss", "spread", "bench_minutes", "bench_ts_rel", "bench_reb_rel", "returning_minutes",
            "rotation_stability", "star_continuity", "close_game_win_pct", "lns_player", "star_usage", "star_ts",
            "team_ts_player", "weighted_ts_top_usage_cluster", "best_rotation_netrtg", "worst_rotation_netrtg",
            "backup_quality_key_positions", "FFC", "must_win_flag", "bubble_pressure_flag", "late_season_flag",
            "home_rest_days", "away_rest_days", "rest_days", "estimated_travel_miles", "lineup_height_avg",
            "opp_lineup_height_avg",
        ],
    )

    out["season"] = pd.to_numeric(out.get("season"), errors="coerce")
    out["lg_eFG"] = out.groupby("season", dropna=False)["eFG"].transform("mean")
    out["lg_TOV"] = out.groupby("season", dropna=False)["TOV%"].transform("mean")
    out["lg_ORB"] = out.groupby("season", dropna=False)["ORB%"].transform("mean")
    out["lg_FTr"] = out.groupby("season", dropna=False)["FTr"].transform("mean")

    out["pace_team"] = _first_present(out, ["pace", "poss"])
    out["pace_opp"] = _first_present(out, ["opp_pace", "opp_poss"])
    out["projected_pace"] = (out["pace_team"] + out["pace_opp"]) / 2.0

    out["z_eFG"] = f.zscore_series(out["eFG"], by=out["season"])
    out["z_3PA_rate"] = f.zscore_series(out["3PA_rate"], by=out["season"])
    out["z_FT_pts_per_FGA"] = f.zscore_series(out["FT_pts_per_FGA"], by=out["season"])

    out["ft_ppp"] = _first_present(out, ["FT_pts_per_poss"])
    out["opp_ft_ppp"] = _first_present(out, ["opp_FT_pts_per_poss"])
    out["ft_ppp_diff"] = out["ft_ppp"] - out["opp_ft_ppp"]
    out["ftr_diff"] = out["FTr"] - out["opp_FTr"]
    out["z_ft_ppp_diff"] = f.zscore_series(out["ft_ppp_diff"], by=out["season"])
    out["z_ftr_diff"] = f.zscore_series(out["ftr_diff"], by=out["season"])

    out["clutch_pct"] = _first_present(out, ["close_game_win_pct"], default=np.nan)
    out["team_ts"] = _first_present(out, ["ts_pct", "team_ts_player"], default=np.nan)
    out["ICS"] = out.apply(
        lambda r: f.ics(r["returning_minutes"], r["rotation_stability"], r["star_continuity"])
        if pd.notna(r["returning_minutes"]) and pd.notna(r["rotation_stability"]) and pd.notna(r["star_continuity"])
        else np.nan,
        axis=1,
    )
    out["BSI"] = out.apply(
        lambda r: f.bsi(r["bench_minutes"], r["bench_ts_rel"], r["bench_reb_rel"])
        if pd.notna(r["bench_minutes"]) and pd.notna(r["bench_ts_rel"]) and pd.notna(r["bench_reb_rel"])
        else np.nan,
        axis=1,
    )

    if "FFC" in out.columns:
        out["ffc_inverted"] = -pd.to_numeric(out["FFC"], errors="coerce")
    else:
        out["ffc_inverted"] = out.apply(
            lambda r: -((0.4 * r["eFG"]) - (0.25 * r["TOV%"]) + (0.2 * r["ORB%"]) + (0.15 * r["FTr"]))
            if pd.notna(r["eFG"]) and pd.notna(r["TOV%"]) and pd.notna(r["ORB%"]) and pd.notna(r["FTr"])
            else np.nan,
            axis=1,
        )

    out["netrtg_adj"] = _first_present(out, ["adj_net_rtg", "NetRtg"], default=np.nan)
    sort_cols = [c for c in ["team_id", "season", "game_datetime_utc", "event_id"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)
    out["VOL_l8_std"] = out.groupby(["team_id", "season"], dropna=False)["netrtg_adj"].transform(
        lambda s: f.rolling_std(s, window=8, min_periods=2)
    )
    out["team_avg_pace"] = out.groupby(["team_id", "season"], dropna=False)["pace_team"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    out["pace_abs_dev"] = (out["pace_team"] - out["team_avg_pace"]).abs()
    out["pace_abs_dev_l12_mean"] = out.groupby(["team_id", "season"], dropna=False)["pace_abs_dev"].transform(
        lambda s: f.rolling_mean(s, window=12, min_periods=3)
    )
    out["z_pace_abs_dev_l12_mean"] = f.zscore_series(out["pace_abs_dev_l12_mean"], by=out["season"])

    home_flag = out["home_away"].astype(str).str.lower() == "home" if "home_away" in out.columns else pd.Series(False, index=out.index)
    out["days_rest_home"] = out["home_rest_days"] if "home_rest_days" in out.columns else np.where(home_flag, out.get("rest_days"), out.get("opp_rest_days"))
    out["days_rest_away"] = out["away_rest_days"] if "away_rest_days" in out.columns else np.where(home_flag, out.get("opp_rest_days"), out.get("rest_days"))
    out["b2b_home"] = np.where(home_flag, out.get("is_back_to_back"), out.get("opp_is_back_to_back"))
    out["b2b_away"] = np.where(home_flag, out.get("opp_is_back_to_back"), out.get("is_back_to_back"))
    out["b2b_home"] = _parse_bool(pd.Series(out["b2b_home"], index=out.index)).astype(float)
    out["b2b_away"] = _parse_bool(pd.Series(out["b2b_away"], index=out.index)).astype(float)
    out["back2back_penalty"] = out["b2b_away"] - out["b2b_home"]

    out["tournament_stage"] = _first_present(out, ["tournament_stage", "late_season_flag"], default=np.nan)
    if "elimination_risk" not in out.columns and {"must_win_flag", "bubble_pressure_flag"}.issubset(set(out.columns)):
        out["elimination_risk"] = out[["must_win_flag", "bubble_pressure_flag"]].max(axis=1, skipna=False)
    else:
        out["elimination_risk"] = _first_present(out, ["elimination_risk"], default=np.nan)
    out["spread_magnitude"] = out["spread"].abs() if "spread" in out.columns else np.nan

    out["cross_country_miles"] = np.where(home_flag, out.get("opp_estimated_travel_miles"), out.get("estimated_travel_miles"))
    out["elevation_home"] = _first_present(out, ["elevation_home"], default=np.nan)
    out["elevation_away"] = _first_present(out, ["elevation_away"], default=np.nan)
    return out


def _log_metric_health(df: pd.DataFrame) -> None:
    for acronym, metric_col in METRIC_SPECS:
        if metric_col not in df.columns:
            continue
        status_col = f"metric_status_{acronym}"
        null_rate = float(pd.to_numeric(df[metric_col], errors="coerce").isna().mean() * 100.0)
        status_counts = df[status_col].value_counts(dropna=False).to_dict() if status_col in df.columns else {}
        log.info(f"metric={acronym} null_rate_pct={null_rate:.2f} status_counts={status_counts}")


def _log_written_csv(path: Path, df: pd.DataFrame) -> None:
    log.info(f"csv_write path={path.resolve()} rows={len(df)} cols={len(df.columns)}")


def build_advanced_metrics(output_path: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    base = _prepare_base_table()
    if base.empty:
        log.warning("team_game_metrics input missing/empty; writing empty advanced_metrics.csv")
        empty_cols = ["game_id", "event_id", "team_id", "opponent_id", "date", "game_datetime_utc"]
        for acronym, metric_col in METRIC_SPECS:
            empty_cols.extend([metric_col, f"metric_status_{acronym}"])
        out = pd.DataFrame(columns=empty_cols)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(output_path, index=False)
        _log_written_csv(output_path, out)
        return out

    df = _attach_optional_inputs(base)
    df = _attach_opponent_context(df)
    df = _compute_derived_columns(df)

    df = _apply_metric(
        df,
        acronym="ODI",
        metric_col="ODI_star",
        required_cols=["eFG", "TOV%", "ORB%", "FTr", "opp_eFG", "opp_TOV%", "opp_ORB%", "opp_FTr", "lg_eFG", "lg_TOV", "lg_ORB", "lg_FTr"],
        compute_fn=lambda x: x.apply(
            lambda r: f.odi_star(
                (r["eFG"] - r["lg_eFG"]) - (r["opp_eFG"] - r["lg_eFG"]),
                (r["opp_TOV%"] - r["lg_TOV"]) - (r["TOV%"] - r["lg_TOV"]),
                (r["ORB%"] - r["lg_ORB"]) - (r["opp_ORB%"] - r["lg_ORB"]),
                (r["FTr"] - r["lg_FTr"]) - (r["opp_FTr"] - r["lg_FTr"]),
            ),
            axis=1,
        ),
    )
    df = _apply_metric(
        df,
        acronym="PEI",
        metric_col="PEI",
        required_cols=["ORB%", "opp_ORB%", "opp_TOV%", "TOV%"],
        compute_fn=lambda x: x.apply(lambda r: f.pei(r["ORB%"], r["opp_ORB%"], r["opp_TOV%"], r["TOV%"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="POSW",
        metric_col="POSW",
        required_cols=["PEI", "projected_pace"],
        compute_fn=lambda x: x.apply(lambda r: f.posw(r["PEI"], r["projected_pace"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="SVI",
        metric_col="SVI",
        required_cols=["z_eFG", "z_3PA_rate", "z_FT_pts_per_FGA"],
        compute_fn=lambda x: x.apply(lambda r: f.svi(r["z_eFG"], r["z_3PA_rate"], r["z_FT_pts_per_FGA"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="PXP",
        metric_col="PXP",
        required_cols=["ICS", "BSI", "clutch_pct"],
        compute_fn=lambda x: x.apply(lambda r: f.pxp(r["ICS"], r["BSI"], r["clutch_pct"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="LNS",
        metric_col="LNS",
        required_cols=["lns_player"],
        compute_fn=lambda x: x["lns_player"],
    )
    df["SIE"] = df.apply(
        lambda r: f.sie(r["star_ts"], r["team_ts"], r["star_usage"])
        if pd.notna(r.get("star_ts")) and pd.notna(r.get("team_ts")) and pd.notna(r.get("star_usage"))
        else np.nan,
        axis=1,
    )
    df = _apply_metric(
        df,
        acronym="USEF",
        metric_col="USEF",
        required_cols=["SIE", "weighted_ts_top_usage_cluster"],
        compute_fn=lambda x: x.apply(lambda r: f.usef(r["SIE"], r["weighted_ts_top_usage_cluster"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="DPC",
        metric_col="DPC",
        required_cols=["BSI", "best_rotation_netrtg", "worst_rotation_netrtg"],
        compute_fn=lambda x: x.apply(lambda r: f.dpc(r["BSI"], r["best_rotation_netrtg"], r["worst_rotation_netrtg"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="FII",
        metric_col="FII",
        required_cols=["DPC", "backup_quality_key_positions", "ffc_inverted"],
        compute_fn=lambda x: x.apply(lambda r: f.fii(r["DPC"], r["backup_quality_key_positions"], r["ffc_inverted"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="SME",
        metric_col="SME",
        required_cols=[
            "star_three_pa_rate",
            "star_rim_rate",
            "star_midrange_rate",
            "star_post_ups",
            "opp3p_allowed",
            "opp_rim_fg_allowed",
            "opp_mid_fg_allowed",
            "opp_post_ups_allowed",
        ],
        compute_fn=lambda x: x.apply(
            lambda r: f.sme(
                r["star_three_pa_rate"], r["star_rim_rate"], r["star_midrange_rate"], r["star_post_ups"],
                r["opp3p_allowed"], r["opp_rim_fg_allowed"], r["opp_mid_fg_allowed"], r["opp_post_ups_allowed"],
            ),
            axis=1,
        ),
    )
    df = _apply_metric(
        df,
        acronym="SCH",
        metric_col="SCH",
        required_cols=["pace_team", "pace_opp", "3PA_rate", "opp_3PA_rate", "FTr", "opp_FTr", "lineup_height_avg", "opp_lineup_height_avg"],
        compute_fn=lambda x: x.apply(
            lambda r: f.sch(
                r["pace_team"], r["pace_opp"], r["3PA_rate"], r["opp_3PA_rate"],
                r["FTr"], r["opp_FTr"], r["lineup_height_avg"], r["opp_lineup_height_avg"],
            ),
            axis=1,
        ),
    )
    df = _apply_metric(df, acronym="VOL", metric_col="VOL", required_cols=["VOL_l8_std"], compute_fn=lambda x: x["VOL_l8_std"])
    df = _apply_metric(
        df,
        acronym="TC",
        metric_col="TC",
        required_cols=["z_pace_abs_dev_l12_mean"],
        compute_fn=lambda x: x.apply(lambda r: f.tc(r["z_pace_abs_dev_l12_mean"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="WL",
        metric_col="WL",
        required_cols=["z_ft_ppp_diff", "z_ftr_diff"],
        compute_fn=lambda x: x.apply(lambda r: f.wl(r["z_ft_ppp_diff"], r["z_ftr_diff"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="RFD",
        metric_col="RFD",
        required_cols=["days_rest_home", "days_rest_away", "back2back_penalty"],
        compute_fn=lambda x: x.apply(lambda r: f.rfd(r["days_rest_home"], r["days_rest_away"], r["back2back_penalty"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="GSR",
        metric_col="GSR",
        required_cols=["tournament_stage", "elimination_risk", "spread_magnitude"],
        compute_fn=lambda x: x.apply(lambda r: f.gsr(r["tournament_stage"], r["elimination_risk"], r["spread_magnitude"]), axis=1),
    )
    df = _apply_metric(
        df,
        acronym="ALT",
        metric_col="ALT",
        required_cols=["elevation_home", "elevation_away", "cross_country_miles"],
        compute_fn=lambda x: x.apply(lambda r: f.alt(r["elevation_home"], r["elevation_away"], r["cross_country_miles"]), axis=1),
    )

    key_cols = [c for c in ["game_id", "event_id", "team_id", "opponent_id", "home_away", "date", "game_datetime_utc", "season"] if c in df.columns]
    metric_cols = [m for _, m in METRIC_SPECS if m in df.columns]
    status_cols = [f"metric_status_{a}" for a, _ in METRIC_SPECS if f"metric_status_{a}" in df.columns]
    out_cols = key_cols + metric_cols + status_cols
    out = df[out_cols].copy().sort_values([c for c in ["game_datetime_utc", "event_id", "team_id"] if c in out_cols]).reset_index(drop=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    _log_written_csv(output_path, out)
    _log_metric_health(out)
    log.info(f"advanced metrics output: {output_path} rows={len(out)} cols={len(out.columns)}")
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build advanced metrics Codex output table")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    build_advanced_metrics(Path(args.output))


if __name__ == "__main__":
    main()
