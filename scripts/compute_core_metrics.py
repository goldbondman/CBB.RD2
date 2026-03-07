#!/usr/bin/env python3
"""Agent 2: build leak-free L6/L11 core metrics with Gate 2 enforcement."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

L_RECENT = 6
L_BASELINE = 11
MIN_GAMES = 6

AUDIT_PATH = Path("data/internal/audit_report.json")
FALLBACK_HISTORY_PATH = Path("data/team_game_logs.csv")


def _pick_case_insensitive(cols: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in cols}
    for item in candidates:
        hit = lookup.get(item.lower())
        if hit:
            return hit
    return None


def _ensure_audit() -> dict[str, object]:
    if not AUDIT_PATH.exists():
        subprocess.run([sys.executable, "scripts/audit_data.py"], check=True)
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def _adequate_history(df: pd.DataFrame, team_col: str, min_games: int) -> bool:
    counts = df.groupby(team_col, dropna=False).size()
    return int((counts >= (min_games + 1)).sum()) >= 25


def _choose_primary_file(audit: dict[str, object], log_lines: list[str]) -> Path:
    candidates = [
        Path("data/advanced_metrics.csv"),
        Path("data/team_snapshot.csv"),
    ]
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError("No primary files found under data/")

    scored: list[tuple[int, Path]] = []
    for p in existing:
        key = p.stem
        rows = int(audit.get(key, {}).get("rows", 0))
        cols = len(audit.get(key, {}).get("cols", []))
        scored.append((rows * max(cols, 1), p))
    scored.sort(reverse=True)
    chosen = scored[0][1]

    probe = pd.read_csv(chosen, low_memory=False)
    team_col = _pick_case_insensitive(list(probe.columns), ["team_id", "team"])
    if team_col and _adequate_history(probe, team_col, MIN_GAMES):
        log_lines.append(f"[INFO] Primary source: {chosen} (sufficient history)")
        return chosen

    if FALLBACK_HISTORY_PATH.exists():
        log_lines.append(
            f"[WARN] {chosen} lacks rolling depth; using fallback history {FALLBACK_HISTORY_PATH}"
        )
        return FALLBACK_HISTORY_PATH

    log_lines.append(
        f"[WARN] {chosen} lacks rolling depth; fallback history missing, proceeding anyway"
    )
    return chosen


def _as_numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if not col or col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _is_away_value(val: object) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, (bool, np.bool_)):
        return not bool(val)
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val) == 0.0
    text = str(val).strip().lower()
    return text in {"away", "a", "0", "false", "road"}


def _safe_compute_foundations(df: pd.DataFrame, cols: list[str], log_lines: list[str]) -> pd.DataFrame:
    # Raw box-score mappings with flexible aliases.
    fga = _pick_case_insensitive(cols, ["fga", "FGA", "field_goals_attempted"])
    fgm = _pick_case_insensitive(cols, ["fgm", "FGM", "field_goals_made"])
    tpa = _pick_case_insensitive(cols, ["tpa", "3PA", "fg3a", "three_pa"])
    tpm = _pick_case_insensitive(cols, ["tpm", "3PM", "fg3m", "three_pm"])
    fta = _pick_case_insensitive(cols, ["fta", "FTA", "free_throw_attempts"])
    ftm = _pick_case_insensitive(cols, ["ftm", "FTM", "free_throws_made"])
    oreb = _pick_case_insensitive(cols, ["orb", "oreb", "OREB", "offensive_rebounds"])
    dreb = _pick_case_insensitive(cols, ["drb", "dreb", "DREB", "defensive_rebounds"])
    tov = _pick_case_insensitive(cols, ["tov", "to", "TO", "turnovers"])
    pts = _pick_case_insensitive(cols, ["points_for", "pts", "PTS", "points", "score"])
    opp_pts = _pick_case_insensitive(cols, ["points_against", "opp_points", "opp_pts"])

    opp_fga = _pick_case_insensitive(cols, ["opp_fga", "opp_FGA"])
    opp_oreb = _pick_case_insensitive(cols, ["opp_orb", "opp_oreb", "opp_OREB"])
    opp_tov = _pick_case_insensitive(cols, ["opp_tov", "opp_TO"])
    opp_fta = _pick_case_insensitive(cols, ["opp_fta", "opp_FTA"])
    opp_dreb = _pick_case_insensitive(cols, ["opp_dreb", "opp_drb", "opp_DREB"])

    pace_col = _pick_case_insensitive(cols, ["pace", "Pace"])
    efg_col = _pick_case_insensitive(cols, ["efg_pct", "eFG_pct", "efg"])
    off_eff_col = _pick_case_insensitive(cols, ["off_rtg", "offeff", "offensive_rating"])
    def_eff_col = _pick_case_insensitive(cols, ["def_rtg", "defeff", "defensive_rating"])
    net_col = _pick_case_insensitive(cols, ["net_rtg", "netrtg", "net_rating"])

    fga_s = _as_numeric(df, fga)
    fgm_s = _as_numeric(df, fgm)
    tpa_s = _as_numeric(df, tpa)
    tpm_s = _as_numeric(df, tpm)
    fta_s = _as_numeric(df, fta)
    ftm_s = _as_numeric(df, ftm)
    oreb_s = _as_numeric(df, oreb)
    tov_s = _as_numeric(df, tov)
    pts_s = _as_numeric(df, pts)
    opp_pts_s = _as_numeric(df, opp_pts)

    poss = fga_s - oreb_s + tov_s + (0.475 * fta_s)
    df["_POSS"] = poss

    if pts and fga and oreb and tov and fta:
        df["_OffEff"] = 100.0 * pts_s / poss.replace(0, np.nan)
        log_lines.append("[OK] _OffEff computed from raw box score")
    elif off_eff_col:
        df["_OffEff"] = _as_numeric(df, off_eff_col)
        log_lines.append(f"[MAP] _OffEff from {off_eff_col}")
    else:
        df["_OffEff"] = np.nan

    if opp_pts and opp_fga and opp_oreb and opp_tov and opp_fta:
        opp_poss = _as_numeric(df, opp_fga) - _as_numeric(df, opp_oreb) + _as_numeric(df, opp_tov) + (
            0.475 * _as_numeric(df, opp_fta)
        )
        df["_DefEff"] = 100.0 * opp_pts_s / opp_poss.replace(0, np.nan)
    elif def_eff_col:
        df["_DefEff"] = _as_numeric(df, def_eff_col)
    else:
        df["_DefEff"] = np.nan

    if "_OffEff" in df.columns and "_DefEff" in df.columns and df["_OffEff"].notna().any():
        df["_NetRtg"] = df["_OffEff"] - df["_DefEff"]
    elif net_col:
        df["_NetRtg"] = _as_numeric(df, net_col)
        log_lines.append(f"[MAP] _NetRtg from {net_col}")
    else:
        df["_NetRtg"] = np.nan

    if fgm and fga and tpm:
        df["_eFG"] = (fgm_s + (0.5 * tpm_s)) / fga_s.replace(0, np.nan)
    elif efg_col:
        df["_eFG"] = _as_numeric(df, efg_col)
        if df["_eFG"].max(skipna=True) and float(df["_eFG"].max(skipna=True)) > 1.5:
            df["_eFG"] = df["_eFG"] / 100.0
        log_lines.append(f"[MAP] _eFG from {efg_col}")
    else:
        df["_eFG"] = np.nan

    df["_TOV_pct"] = tov_s / poss.replace(0, np.nan)
    if opp_dreb:
        df["_ORB_pct"] = oreb_s / (oreb_s + _as_numeric(df, opp_dreb)).replace(0, np.nan)
    else:
        df["_ORB_pct"] = np.nan
    df["_FTr"] = fta_s / fga_s.replace(0, np.nan)
    df["_3PA_rate"] = tpa_s / fga_s.replace(0, np.nan)
    df["_FT_PPP"] = ftm_s / poss.replace(0, np.nan)

    if pace_col:
        df["_Pace"] = _as_numeric(df, pace_col)
    else:
        df["_Pace"] = poss
    return df


def _rolling_with_trend(team_df: pd.DataFrame, col: str) -> dict[str, pd.Series]:
    if col not in team_df.columns or team_df[col].isna().all():
        return {}
    shifted = team_df[col].shift(1)
    l11 = shifted.rolling(L_BASELINE, min_periods=MIN_GAMES).mean()
    l6 = shifted.rolling(L_RECENT, min_periods=MIN_GAMES).mean()
    return {
        f"{col}_L11": l11,
        f"{col}_L6": l6,
        f"{col}_trend": l6 - l11,
        f"{col}_vol_L6": shifted.rolling(L_RECENT, min_periods=MIN_GAMES).std(),
    }


def main() -> int:
    log_lines: list[str] = []
    audit = _ensure_audit()
    primary_file = _choose_primary_file(audit, log_lines)
    df_raw = pd.read_csv(primary_file, low_memory=False)
    raw_cols = list(df_raw.columns)

    date_col = _pick_case_insensitive(raw_cols, ["game_datetime_utc", "date", "game_date", "line_timestamp_utc"])
    if not date_col:
        raise ValueError("No date column found for rolling sort")
    team_col = _pick_case_insensitive(raw_cols, ["team_id", "team"])
    if not team_col:
        raise ValueError("No team identifier column found")
    ha_col = _pick_case_insensitive(raw_cols, ["home_away", "is_home", "location"])

    df_raw[date_col] = pd.to_datetime(df_raw[date_col], errors="coerce", utc=True)
    df_raw = df_raw.dropna(subset=[team_col, date_col]).copy()
    df_raw = _safe_compute_foundations(df_raw, raw_cols, log_lines)

    metric_cols = [
        "_NetRtg",
        "_OffEff",
        "_eFG",
        "_TOV_pct",
        "_ORB_pct",
        "_FTr",
        "_3PA_rate",
        "_FT_PPP",
        "_Pace",
    ]

    df_raw = df_raw.sort_values([team_col, date_col]).reset_index(drop=True)
    chunks: list[pd.DataFrame] = []
    teams = df_raw[team_col].dropna().unique()

    for team in teams:
        team_df = df_raw[df_raw[team_col] == team].copy()
        team_df = team_df.sort_values(date_col).reset_index(drop=True)
        new_cols: dict[str, pd.Series] = {}
        for col in metric_cols:
            new_cols.update(_rolling_with_trend(team_df, col))

        if ha_col and ha_col in team_df.columns:
            away_mask = team_df[ha_col].apply(_is_away_value)
            for base_col in ["_eFG", "_NetRtg"]:
                if base_col in team_df.columns:
                    shifted_away = team_df[base_col].where(away_mask).shift(1)
                    new_cols[f"{base_col}_away_L11"] = shifted_away.rolling(
                        L_BASELINE, min_periods=MIN_GAMES
                    ).mean()

        for key, series in new_cols.items():
            team_df[key] = series
        chunks.append(team_df)
        del team_df
        gc.collect()

    if not chunks:
        raise ValueError("No team chunks produced")
    df_metrics = pd.concat(chunks, ignore_index=True)

    # Lookahead assertion: shifted L6 must differ from non-shifted L6 for at least one team.
    sampled = df_metrics[df_metrics["_NetRtg"].notna()].copy()
    sample_team = None
    for team in teams:
        tdf = sampled[sampled[team_col] == team]
        if len(tdf) >= 12 and "_NetRtg_L6" in tdf.columns:
            sample_team = team
            break
    if sample_team is not None:
        st = sampled[sampled[team_col] == sample_team].sort_values(date_col)
        plain = st["_NetRtg"].rolling(L_RECENT, min_periods=MIN_GAMES).mean()
        shifted = st["_NetRtg_L6"]
        common = plain.notna() & shifted.notna()
        if common.any():
            same = np.isclose(plain[common].to_numpy(), shifted[common].to_numpy(), equal_nan=True)
            if bool(np.all(same)):
                raise ValueError("[CRITICAL] Lookahead detected: shifted rolling equals plain rolling.")
            log_lines.append(
                f"[OK] Lookahead assertion passed for {sample_team} ({int((~same).sum())} differing rows)"
            )

    out_dir = Path("data/internal")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "core_metrics.csv"
    df_metrics.to_csv(out_path, index=False)

    gate = {
        "has_rows": len(df_metrics) > 100,
        "has_L11_netrtg": "_NetRtg_L11" in df_metrics.columns,
        "has_L6_netrtg": "_NetRtg_L6" in df_metrics.columns,
        "has_trend": "_NetRtg_trend" in df_metrics.columns,
        "nan_rate_ok": (
            float(df_metrics["_NetRtg_L11"].isna().mean()) < 0.40 if "_NetRtg_L11" in df_metrics.columns else False
        ),
        "has_efg": "_eFG_L11" in df_metrics.columns or "_eFG_L6" in df_metrics.columns,
    }

    log_lines.append(f"[INFO] rows={len(df_metrics)} cols={len(df_metrics.columns)}")
    if "_NetRtg_L11" in df_metrics.columns:
        log_lines.append(f"[INFO] _NetRtg_L11 NaN rate={df_metrics['_NetRtg_L11'].isna().mean():.2%}")

    (out_dir / "metrics_build_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    print("=== GATE_2 RESULTS ===")
    for check, result in gate.items():
        print(f"  {'PASS' if result else 'FAIL'}  {check}")

    if not all(gate.values()):
        failed = [k for k, v in gate.items() if not v]
        print(f"[STOP] Gate 2 failed: {failed}")
        return 1

    print(f"[OK] Gate 2 passed. Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
