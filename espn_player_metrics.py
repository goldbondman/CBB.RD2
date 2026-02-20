"""
ESPN CBB Pipeline — Player Rolling Metrics
Per-game derived stats and L5/L10 rolling averages for player box scores.
Pure DataFrame transformations. No I/O.

Per-game derived metrics:
  efg_pct         Effective FG%
  ts_pct          True Shooting %
  fg_pct          FG%
  three_pct       3P%
  ft_pct          FT%
  usage_rate      Possession usage proxy: (FGA + 0.44*FTA + TOV) / team_poss
  ast_tov_ratio   AST / TOV
  pts_per_fga     Points per field goal attempt (efficiency of shot selection)
  floor_pct       % of possessions ending in pts/ast/stl/blk (positive play rate)

Rolling L5 and L10 (leak-free via shift(1)):
  All box score counting stats: pts, min, fgm, fga, tpm, tpa, ftm, fta,
                                 orb, drb, reb, ast, stl, blk, tov, pf
  All per-game derived metrics above
  Shooting variance: efg_std_l10, three_pct_std_l10

Output: player_game_metrics.csv
"""

import logging
import os
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

ROLLING_WINDOWS    = [5, 10]
MIN_PERIODS        = 3       # minimum games before reporting rolling values

RAW_FGM_MIN_NON_NULL = float(os.getenv("PLAYER_RAW_FGM_MIN_NON_NULL", "0.80"))
RAW_FGA_MIN_NON_NULL = float(os.getenv("PLAYER_RAW_FGA_MIN_NON_NULL", "0.80"))
RAW_TPM_MIN_NON_NULL = float(os.getenv("PLAYER_RAW_TPM_MIN_NON_NULL", "0.50"))
RAW_TPA_MIN_NON_NULL = float(os.getenv("PLAYER_RAW_TPA_MIN_NON_NULL", "0.50"))
RAW_ORB_MIN_NON_NULL = float(os.getenv("PLAYER_RAW_ORB_MIN_NON_NULL", "0.80"))
RAW_DRB_MIN_NON_NULL = float(os.getenv("PLAYER_RAW_DRB_MIN_NON_NULL", "0.80"))
RAW_PLUS_MINUS_MIN_NON_NULL = float(os.getenv("PLAYER_RAW_PLUS_MINUS_MIN_NON_NULL", "0.50"))

# All counting stats to roll
COUNTING_STATS = [
    "min", "pts",
    "fgm", "fga", "tpm", "tpa", "ftm", "fta",
    "orb", "drb", "reb",
    "ast", "stl", "blk", "tov", "pf",
    "plus_minus",
]

# Derived per-game metrics to roll
DERIVED_METRICS = [
    "efg_pct", "ts_pct", "fg_pct", "three_pct", "ft_pct",
    "usage_rate", "ast_tov_ratio", "pts_per_fga", "floor_pct",
]


# ── Safe column helper ────────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str, fill: float = np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(fill)
    return pd.Series(fill, index=df.index)


def _to_bool_series(s: pd.Series) -> pd.Series:
    """Robust bool parsing for CSV-loaded strings ('False' must stay False)."""
    true_tokens = {"1", "true", "t", "yes", "y"}
    return s.astype(str).str.strip().str.lower().isin(true_tokens)


def _parse_made_attempt_series(series: pd.Series):
    made = pd.Series(np.nan, index=series.index, dtype="float64")
    att = pd.Series(np.nan, index=series.index, dtype="float64")
    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" of ", "-", regex=False)
        .str.replace("/", "-", regex=False)
    )
    split = normalized.str.extract(r"^\s*(-?\d+)\s*-\s*(-?\d+)\s*$")
    ok = split[0].notna() & split[1].notna()
    made.loc[ok] = pd.to_numeric(split.loc[ok, 0], errors="coerce")
    att.loc[ok] = pd.to_numeric(split.loc[ok, 1], errors="coerce")
    return made, att


def _coalesce_numeric(df: pd.DataFrame, target: str, candidates: List[str]) -> None:
    if target not in df.columns:
        df[target] = np.nan
    dst = pd.to_numeric(df[target], errors="coerce")
    for c in candidates:
        if c in df.columns:
            src = pd.to_numeric(df[c], errors="coerce")
            dst = dst.where(dst.notna(), src)
    df[target] = dst


def _normalize_player_box_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize raw player box-score inputs from ESPN payload variants:
      - direct numeric cols (fgm/fga/tpm/tpa/ftm/fta/orb/drb/plus_minus)
      - alias cols (FGM/FGA/TPM/TPA/FTM/FTA/ORB/DRB)
      - combined split cols like FG/3PT/FT in 'made-attempted' format.
    """
    df = df.copy()

    _coalesce_numeric(df, "fgm", ["FGM"]) 
    _coalesce_numeric(df, "fga", ["FGA"]) 
    _coalesce_numeric(df, "tpm", ["TPM"]) 
    _coalesce_numeric(df, "tpa", ["TPA"]) 
    _coalesce_numeric(df, "ftm", ["FTM"]) 
    _coalesce_numeric(df, "fta", ["FTA"]) 
    _coalesce_numeric(df, "orb", ["ORB"]) 
    _coalesce_numeric(df, "drb", ["DRB"]) 
    _coalesce_numeric(df, "plus_minus", ["+/-", "plusMinus", "plusminus"]) 

    # Parse made-attempt text fallback from combined stat columns.
    for src_col, made_col, att_col in [
        ("FG", "fgm", "fga"),
        ("3PT", "tpm", "tpa"),
        ("FT", "ftm", "fta"),
    ]:
        if src_col in df.columns:
            made, att = _parse_made_attempt_series(df[src_col])
            df[made_col] = pd.to_numeric(df[made_col], errors="coerce").where(df[made_col].notna(), made)
            df[att_col] = pd.to_numeric(df[att_col], errors="coerce").where(df[att_col].notna(), att)

    for col in ["fgm", "fga", "tpm", "tpa", "ftm", "fta", "orb", "drb", "plus_minus"]:
        if col not in df.columns:
            df[col] = np.nan

    return df


def _add_last_window_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Add explicit last_5_/last_10_ aliases for *_l5/*_l10 rolling cols."""
    for col in list(df.columns):
        if col.endswith("_l5"):
            alias = f"last_5_{col[:-3]}"
            if alias not in df.columns:
                df[alias] = df[col]
        if col.endswith("_l10"):
            alias = f"last_10_{col[:-4]}"
            if alias not in df.columns:
                df[alias] = df[col]
    return df


def _log_player_metric_null_rates(stage: str, df: pd.DataFrame) -> None:
    target = [
        "fgm", "fga", "tpm", "tpa", "fta", "orb", "drb", "plus_minus",
        "efg_pct", "three_pct", "fg_pct", "ft_pct",
    ]
    present = [c for c in target if c in df.columns]
    if not present:
        log.info(f"{stage}: no target player metric columns present")
        return
    rates = (df[present].isna().mean() * 100).round(1).to_dict()
    key_cols = [c for c in ["event_id", "team_id", "athlete_id", "game_datetime_utc"] if c in df.columns]
    sample = df[key_cols].head(3).to_dict("records") if key_cols else []
    log.info(f"{stage}: rows={len(df)} null_rates(%)={rates} key_sample={sample}")


def _validate_player_metrics_enrichment(df: pd.DataFrame) -> None:
    if df.empty:
        return

    def _non_null_rate(col: str, mask: pd.Series) -> float:
        if col not in df.columns:
            return 0.0
        vals = df.loc[mask, col]
        if vals.empty:
            return 1.0
        token_null = vals.astype(str).str.strip().str.lower().isin({"", "nan", "none", "null", "nat", "<na>"})
        null_mask = vals.isna() | token_null
        return 1.0 - float(null_mask.mean())

    played_mask = ~_to_bool_series(df.get("did_not_play", pd.Series(False, index=df.index)))
    errors = []

    thresholds = {
        "fgm": RAW_FGM_MIN_NON_NULL,
        "fga": RAW_FGA_MIN_NON_NULL,
        "tpm": RAW_TPM_MIN_NON_NULL,
        "tpa": RAW_TPA_MIN_NON_NULL,
        "orb": RAW_ORB_MIN_NON_NULL,
        "drb": RAW_DRB_MIN_NON_NULL,
        "plus_minus": RAW_PLUS_MINUS_MIN_NON_NULL,
    }
    for col, threshold in thresholds.items():
        rate = _non_null_rate(col, played_mask)
        if rate <= 0.0:
            errors.append(f"player metrics enrichment failed: {col} is 100% null")
        elif rate < threshold:
            errors.append(f"player metrics enrichment below threshold: {col} < {threshold:.0%}")

    # At least one rolling last_5 and last_10 feature should populate for players with enough history.
    if "athlete_id" in df.columns:
        eligible_players = df.groupby("athlete_id").size()
        has_6_plus = eligible_players[eligible_players >= 6].index
        if len(has_6_plus) > 0:
            eligible = df["athlete_id"].isin(has_6_plus)
            l5_cols = [c for c in df.columns if c.endswith("_l5") or c.startswith("last_5_")]
            l10_cols = [c for c in df.columns if c.endswith("_l10") or c.startswith("last_10_")]
            if l5_cols:
                l5_any = (~df.loc[eligible, l5_cols].isna()).any().any()
                if not l5_any:
                    errors.append("player rolling enrichment failed: last_5 columns are 100% null for players with >=6 games")
            if l10_cols:
                l10_any = (~df.loc[eligible, l10_cols].isna()).any().any()
                if not l10_any:
                    errors.append("player rolling enrichment failed: last_10 columns are 100% null for players with >=6 games")

    if errors:
        raise ValueError(" | ".join(errors))


# ── Per-game derived metrics ──────────────────────────────────────────────────

def add_player_per_game_metrics(df: pd.DataFrame,
                                 team_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-game derived metrics to player box score rows.
    team_logs is used to get team-level possessions for usage rate.
    """
    df = df.copy()

    # ── Normalize blank counting columns to 0 ──────────────────────────────────
    # ESPN returns empty string (not "0") for players with 0 attempts.
    # Fill these in the dataframe so the raw columns show 0 in the output CSV
    # and so rolling windows in add_player_rolling() get clean numeric data.
    for _col_name in ["fgm", "fga", "tpm", "tpa", "ftm", "fta",
                      "orb", "drb", "plus_minus"]:
        if _col_name in df.columns:
            df[_col_name] = pd.to_numeric(df[_col_name], errors="coerce").fillna(0)

    fgm = _col(df, "fgm", 0); fga = _col(df, "fga", 0)
    tpm = _col(df, "tpm", 0); tpa = _col(df, "tpa", 0)
    ftm = _col(df, "ftm", 0); fta = _col(df, "fta", 0)
    pts = _col(df, "pts", 0)
    ast = _col(df, "ast", 0); tov = _col(df, "tov", 0)
    stl = _col(df, "stl", 0); blk = _col(df, "blk", 0)
    min_played = _col(df, "min", 0)

    fga_s = fga.replace(0, np.nan)
    fta_s = fta.replace(0, np.nan)

    # ── Shooting ──
    df["efg_pct"]    = ((fgm + 0.5 * tpm) / fga_s * 100).round(1)
    df["ts_pct"]     = (pts / (2 * (fga + 0.44 * fta)).replace(0, np.nan) * 100).round(1)
    df["fg_pct"]     = (fgm / fga_s * 100).round(1)
    df["three_pct"]  = (tpm / tpa.replace(0, np.nan) * 100).round(1)
    df["ft_pct"]     = (ftm / fta_s * 100).round(1)
    df["pts_per_fga"] = (pts / fga_s).round(2)

    # ── Playmaking ──
    df["ast_tov_ratio"] = (ast / tov.replace(0, np.nan)).round(2)

    # ── Usage rate ──
    # Requires team possessions — join from team_logs if available
    if not team_logs.empty and "poss" in team_logs.columns:
        team_poss = (team_logs[["event_id", "team_id", "poss"]]
                     .drop_duplicates()
                     .rename(columns={"poss": "_team_poss"}))
        df = df.merge(team_poss, on=["event_id", "team_id"], how="left")
        tp = pd.to_numeric(df.get("_team_poss", np.nan), errors="coerce").replace(0, np.nan)
        df["usage_rate"] = ((fga + 0.44 * fta + tov) / tp * 100).round(1)
        df = df.drop(columns=["_team_poss"], errors="ignore")
    else:
        # Approximate usage without team possessions
        # Rough: player usage ≈ player "actions" / 5 (5 players share possessions)
        df["usage_rate"] = ((fga + 0.44 * fta + tov) * 5 /
                            (fga + 0.44 * fta + tov + 1).replace(0, np.nan) * 20
                           ).round(1)

    # ── Floor % — % of minutes in "positive plays" ──
    # Proxy: (pts + ast*2 + stl + blk) / max(1, min * 2.5)
    # Not a true floor%, but captures positive contribution rate
    positive_plays = pts + ast * 2 + stl + blk
    df["floor_pct"] = (positive_plays / min_played.replace(0, np.nan)).round(2)

    # Null out derived metrics for DNP rows
    if "did_not_play" in df.columns:
        dnp = _to_bool_series(df["did_not_play"])
        for col in DERIVED_METRICS:
            if col in df.columns:
                df.loc[dnp, col] = np.nan

    return df


# ── Rolling averages ──────────────────────────────────────────────────────────

def add_player_rolling(df: pd.DataFrame,
                        windows: List[int] = ROLLING_WINDOWS) -> pd.DataFrame:
    """
    Add rolling averages for counting stats and derived metrics.
    Grouped by athlete_id, sorted by game_datetime_utc.
    Leak-free: shift(1) means game N reflects games 1..N-1.

    Only non-DNP appearances feed the rolling average.
    DNP games are excluded from the rolling window but don't reset it.
    """
    if "athlete_id" not in df.columns or "game_datetime_utc" not in df.columns:
        log.warning("add_player_rolling: missing athlete_id or game_datetime_utc")
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True,
                                    errors="coerce")
    df = df.sort_values(["athlete_id", "_sort_dt"])

    # Mark active game rows (played meaningful minutes)
    active = (
        ~_to_bool_series(df["did_not_play"]) &
        pd.to_numeric(df.get("min", 0), errors="coerce").gt(0)
    ) if "did_not_play" in df.columns else pd.Series(True, index=df.index)

    all_metrics = [m for m in COUNTING_STATS + DERIVED_METRICS if m in df.columns]

    # Counting stats: fill blank/NaN with 0 (0 FGM means they attempted none, not missing)
    # Derived metrics: keep NaN (NaN means DNP or insufficient data — don't impute)
    counting_set = set(COUNTING_STATS)

    for window in windows:
        for metric in all_metrics:
            col_name = f"{metric}_l{window}"
            is_counting = metric in counting_set
            df[col_name] = df.groupby("athlete_id")[metric].transform(
                lambda s, _is_counting=is_counting: (
                    pd.to_numeric(s, errors="coerce")
                    .fillna(0 if _is_counting else float("nan"))  # 0 for counts, NaN for rates
                    .where(active.reindex(s.index, fill_value=True))
                    .shift(1)
                    .rolling(window, min_periods=min(MIN_PERIODS, window))
                    .mean()
                    .round(2)
                )
            )

    # ── Shooting variance ──
    # Variance metrics: pts/min are counting (fillna 0), rates keep NaN
    for metric, col_name, fill_zero in [
        ("efg_pct",   "efg_std_l10",       False),  # rate — keep NaN for DNP
        ("three_pct", "three_pct_std_l10", False),  # rate — keep NaN for DNP
        ("pts",       "pts_std_l10",       True),   # count — 0 if didn't score
        ("min",       "min_std_l10",       True),   # count — 0 if DNP
    ]:
        if metric in df.columns:
            df[col_name] = df.groupby("athlete_id")[metric].transform(
                lambda s, _fill=fill_zero: (
                    pd.to_numeric(s, errors="coerce")
                    .fillna(0 if _fill else float("nan"))
                    .where(active.reindex(s.index, fill_value=True))
                    .shift(1)
                    .rolling(10, min_periods=MIN_PERIODS)
                    .std()
                    .round(2)
                )
            )

    # ── Season averages (expanding, leak-free) ──
    for metric in ["pts", "min", "efg_pct", "usage_rate", "ast_tov_ratio"]:
        if metric in df.columns:
            df[f"{metric}_season_avg"] = df.groupby("athlete_id")[metric].transform(
                lambda s: (
                    pd.to_numeric(s, errors="coerce")
                    .where(active.reindex(s.index, fill_value=True))
                    .shift(1)
                    .expanding(min_periods=MIN_PERIODS)
                    .mean()
                    .round(2)
                )
            )

    # ── Games played counter (active games before this one) ──
    df["_active_int"] = active.astype(int)
    df["games_played"] = df.groupby("athlete_id")["_active_int"].transform(
        lambda s: s.shift(1).expanding().sum().fillna(0).astype(int)
    )
    df = df.drop(columns=["_active_int"], errors="ignore")

    df = df.drop(columns=["_sort_dt"], errors="ignore")
    return df


# ── Starter/bench split rolling ───────────────────────────────────────────────

def add_role_split_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Separate rolling L5/L10 for starter games vs bench games.
    Starter and bench roles have very different expected production levels.
    """
    if "starter" not in df.columns:
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True,
                                    errors="coerce")
    df = df.sort_values(["athlete_id", "_sort_dt"])

    starter_flag = df["starter"].astype(bool)

    for metric in ["pts", "min", "efg_pct", "usage_rate"]:
        if metric not in df.columns:
            continue
        for role, mask in [("starter", starter_flag),
                           ("bench",   ~starter_flag)]:
            vals = pd.to_numeric(df[metric], errors="coerce").where(mask)
            df[f"{metric}_{role}_l5"] = df.groupby("athlete_id")[metric].transform(
                lambda s, m=mask: (
                    pd.to_numeric(s, errors="coerce")
                    .where(m.reindex(s.index, fill_value=False))
                    .shift(1)
                    .rolling(5, min_periods=2)
                    .mean()
                    .round(2)
                )
            )

    df = df.drop(columns=["_sort_dt"], errors="ignore")
    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_player_metrics(player_df: pd.DataFrame,
                            team_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full player metrics pipeline.
    Input:  player_game_logs DataFrame (full season history)
            team_logs_df: team_game_logs or team_game_metrics (for poss data)
    Output: player_game_metrics.csv
    """
    if player_df.empty:
        log.warning("compute_player_metrics: empty DataFrame")
        return player_df

    log.info(f"Computing player metrics for {len(player_df)} player-game rows "
             f"({player_df['athlete_id'].nunique()} unique players)")

    df = _normalize_player_box_columns(player_df)
    _log_player_metric_null_rates("player_metrics_after_normalize", df)

    df = add_player_per_game_metrics(df, team_logs_df)
    _log_player_metric_null_rates("player_metrics_after_derived", df)

    df = add_player_rolling(df, windows=ROLLING_WINDOWS)
    df = add_role_split_metrics(df)
    df = _add_last_window_aliases(df)
    _log_player_metric_null_rates("player_metrics_before_validate", df)

    _validate_player_metrics_enrichment(df)

    log.info("Player metrics complete")
    return df
