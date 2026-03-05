"""
ESPN CBB Pipeline: Strength of Schedule and Opponent-Adjusted Metrics
Requires team_game_metrics.csv (output of espn_metrics.py) as input.

Metrics computed:

STRENGTH OF SCHEDULE (3 windows per metric):
  opp_avg_ortg         Average ORTG of opponents faced
  opp_avg_drtg         Average DRTG of opponents faced
  opp_avg_net_rtg      Average NetRTG of opponents faced (primary SOS number)
  opp_avg_efg_pct      Average eFG% of opponents faced
  opp_avg_pace         Average pace of opponents faced

  Each computed for:
    _season   Full season (all prior games)
    _l5       Last 5 games
    _l10      Last 10 games

OPPONENT-CONTEXT METRICS (performance vs. what opponents allow):
  efg_vs_opp_allow      Team eFG% minus avg eFG% opponents allow to others
  orb_vs_opp_allow      Team ORB% minus avg ORB% opponents allow to others
  drb_vs_opp_allow      Team DRB% minus avg DRB% opponents allow to others
  reb_vs_opp_allow      Team REB% (total) vs opponent context
  tov_vs_opp_force      Team TOV% minus avg TOV% opponents force on others
  ftr_vs_opp_allow      Team FTR minus avg FTR opponents allow to others

  Each computed for:
    _season, _l5, _l10

ADJUSTED RATINGS:
  adj_ortg              ORTG adjusted for opponent defensive strength (drtg)
  adj_drtg              DRTG adjusted for opponent offensive strength (ortg)
  adj_net_rtg           adj_ortg - adj_drtg

ADJUSTED PACE:
  adj_pace              Pace normalized by opponent tempo context, scaled to league average pace
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from pandas.errors import MergeError

log = logging.getLogger(__name__)

# League average constants (D1-ish baselines; update each season if desired)
LEAGUE_AVG_ORTG = 103.0
LEAGUE_AVG_DRTG = 103.0
LEAGUE_AVG_EFG = 50.5
LEAGUE_AVG_ORB = 28.0
LEAGUE_AVG_DRB = 72.0
LEAGUE_AVG_TOV = 18.0
LEAGUE_AVG_FTR = 28.0

# Pace is computed dynamically from the dataset, with a safe fallback
LEAGUE_AVG_PACE_FALLBACK = 70.0

SOS_WINDOWS = [5, 10]  # rolling windows; season is always included
SOS_MAX_ROW_MULTIPLIER = float(os.getenv("SOS_MAX_ROW_MULTIPLIER", "1.25"))

DEBUG_DIR = Path("debug")
SOS_SIZE_AUDIT_PATH = DEBUG_DIR / "sos_size_audit.json"
SOS_DUP_SAMPLE_PATH = DEBUG_DIR / "sos_duplicate_input_samples.csv"
SOS_MERGE_VIOLATION_PATH = DEBUG_DIR / "sos_merge_violation_samples.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _rolling_mean(series: pd.Series, window: int) -> pd.Series:
    """Leak-free rolling mean: shift(1) so game N uses only prior games."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _expanding_mean(series: pd.Series) -> pd.Series:
    """Leak-free expanding (season-to-date) mean using only prior games."""
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.shift(1).expanding(min_periods=1).mean()


def _get_league_avg_pace(df: pd.DataFrame) -> float:
    """
    Compute league average pace from the dataset.
    This is used as a scaling constant, not a per-team feature.

    Returns a positive float, otherwise falls back to LEAGUE_AVG_PACE_FALLBACK.
    """
    pace = pd.to_numeric(df.get("pace", np.nan), errors="coerce")

    # Prefer a robust central tendency to damp outliers
    val = float(np.nanmedian(pace.values))
    if np.isfinite(val) and val > 0:
        return val

    # Fallback to mean if median is unusable
    val = float(np.nanmean(pace.values))
    if np.isfinite(val) and val > 0:
        return val

    return float(LEAGUE_AVG_PACE_FALLBACK)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_id_series(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"None": np.nan, "nan": np.nan, "NaN": np.nan, "": np.nan})
    s = s.str.replace(r"\.0$", "", regex=True)
    return s


def _jsonable_records(df: pd.DataFrame, max_rows: int = 5) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = []
    for row in df.head(max_rows).to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, (np.floating, np.integer)):
                clean[key] = float(value)
            elif pd.isna(value):
                clean[key] = None
            else:
                clean[key] = str(value)
        out.append(clean)
    return out


def _dup_report(df: pd.DataFrame, keys: list[str], label: str) -> dict[str, Any]:
    keys = [k for k in keys if k in df.columns]
    if not keys:
        msg = f"{label}: keys unavailable for duplicate audit"
        log.warning(msg)
        return {
            "label": label,
            "keys": keys,
            "rows": int(len(df)),
            "unique_rows": None,
            "duplicate_keys": None,
            "duplication_factor": None,
            "top_duplicates": [],
        }

    grouped = (
        df.groupby(keys, dropna=False)
        .size()
        .reset_index(name="_count")
        .sort_values("_count", ascending=False)
    )
    unique_rows = int(len(grouped))
    duplicate_keys = int((grouped["_count"] > 1).sum())
    duplication_factor = float(len(df) / max(unique_rows, 1))
    top_dupes = grouped[grouped["_count"] > 1].head(5)

    log.info(
        "%s rows=%s unique_by_keys=%s duplicate_keys=%s duplication_factor=%.4f",
        label,
        len(df),
        unique_rows,
        duplicate_keys,
        duplication_factor,
    )
    if not top_dupes.empty:
        log.warning("%s top duplicate keys: %s", label, top_dupes.to_dict(orient="records"))

    return {
        "label": label,
        "keys": keys,
        "rows": int(len(df)),
        "unique_rows": unique_rows,
        "duplicate_keys": duplicate_keys,
        "duplication_factor": duplication_factor,
        "top_duplicates": _jsonable_records(top_dupes),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolve_input_duplicates(df: pd.DataFrame, key_cols: list[str], audit: dict[str, Any]) -> pd.DataFrame:
    dup = _dup_report(df, key_cols, "compute_sos_metrics:input_key_audit")
    audit["input_dedup"] = dup
    if not dup.get("duplicate_keys"):
        return df

    tmp = df.copy()
    tmp["_row_order"] = np.arange(len(tmp))

    if "completed" in tmp.columns:
        completed = tmp["completed"].astype(str).str.strip().str.lower()
        tmp["_is_final"] = completed.isin({"true", "1", "yes", "y"}).astype(int)
    elif "state" in tmp.columns:
        state = tmp["state"].astype(str).str.strip().str.lower()
        tmp["_is_final"] = state.eq("post").astype(int)
    else:
        tmp["_is_final"] = 0

    sort_cols = key_cols + ["_is_final", "_row_order"]
    for col in ["updated_at", "pulled_at_utc", "captured_at_utc", "game_datetime_utc"]:
        if col in tmp.columns:
            parsed = pd.to_datetime(tmp[col], utc=True, errors="coerce")
            sort_name = f"_sort_{col}"
            tmp[sort_name] = parsed
            sort_cols.insert(-1, sort_name)

    dup_keys = (
        tmp.groupby(key_cols, dropna=False)
        .size()
        .reset_index(name="_dup_count")
        .query("_dup_count > 1")
        .sort_values("_dup_count", ascending=False)
    )
    dup_sample = dup_keys.head(100).merge(tmp, on=key_cols, how="left")
    SOS_DUP_SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    dup_sample.to_csv(SOS_DUP_SAMPLE_PATH, index=False)
    log.warning(
        "compute_sos_metrics: resolving %s duplicate keys using deterministic keep-last strategy; sample=%s",
        len(dup_keys),
        SOS_DUP_SAMPLE_PATH,
    )

    deduped = (
        tmp.sort_values(sort_cols)
        .drop_duplicates(subset=key_cols, keep="last")
        .drop(columns=[c for c in tmp.columns if c.startswith("_sort_") or c in {"_is_final", "_row_order"}], errors="ignore")
    )
    dedup_report = _dup_report(deduped, key_cols, "compute_sos_metrics:post_input_dedup")
    audit["input_dedup_result"] = dedup_report
    return deduped


def _enforce_row_guard(stage: str, pre_rows: int, post_rows: int, audit: dict[str, Any]) -> None:
    multiplier = float(post_rows / max(pre_rows, 1))
    entry = {
        "stage": stage,
        "pre_rows": int(pre_rows),
        "post_rows": int(post_rows),
        "multiplier": multiplier,
    }
    audit.setdefault("row_guards", []).append(entry)
    if multiplier > SOS_MAX_ROW_MULTIPLIER:
        raise RuntimeError(
            f"[{stage}] row explosion detected: pre_rows={pre_rows}, post_rows={post_rows}, "
            f"multiplier={multiplier:.4f}, guard={SOS_MAX_ROW_MULTIPLIER:.2f}"
        )


def _safe_many_to_one_merge(
    *,
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: list[str],
    right_on: list[str],
    stage: str,
    audit: dict[str, Any],
    match_probe_cols: list[str],
) -> pd.DataFrame:
    pre_rows = len(left)
    left_dup = _dup_report(left, left_on, f"{stage}:left_keys")
    right_dup = _dup_report(right, right_on, f"{stage}:right_keys")

    join_entry: dict[str, Any] = {
        "stage": stage,
        "left_on": left_on,
        "right_on": right_on,
        "left_dup_report": left_dup,
        "right_dup_report": right_dup,
        "left_dtypes": {k: str(left[k].dtype) for k in left_on if k in left.columns},
        "right_dtypes": {k: str(right[k].dtype) for k in right_on if k in right.columns},
    }

    try:
        merged = left.merge(
            right,
            how="left",
            left_on=left_on,
            right_on=right_on,
            validate="many_to_one",
        )
    except MergeError as exc:
        dup_keys = (
            right.groupby(right_on, dropna=False)
            .size()
            .reset_index(name="_dup_count")
            .query("_dup_count > 1")
            .sort_values("_dup_count", ascending=False)
            .head(200)
        )
        if not dup_keys.empty:
            sample = dup_keys.merge(right, on=right_on, how="left")
            SOS_MERGE_VIOLATION_PATH.parent.mkdir(parents=True, exist_ok=True)
            sample.to_csv(SOS_MERGE_VIOLATION_PATH, index=False)
        raise RuntimeError(
            f"[{stage}] merge validation failed ({exc}). Duplicate sample written to {SOS_MERGE_VIOLATION_PATH}."
        ) from exc

    post_rows = len(merged)
    _enforce_row_guard(stage, pre_rows, post_rows, audit)
    join_entry["pre_rows"] = int(pre_rows)
    join_entry["post_rows"] = int(post_rows)
    join_entry["multiplier"] = float(post_rows / max(pre_rows, 1))
    hit_col = next((c for c in match_probe_cols if c in merged.columns), None)
    if hit_col:
        join_entry["join_hit_rate"] = float(merged[hit_col].notna().mean())
    audit.setdefault("joins", []).append(join_entry)
    return merged


# ── Step 1: Build opponent season-to-date stat lookup ─────────────────────────

def _build_opponent_lookup(df: pd.DataFrame, audit: dict[str, Any]) -> pd.DataFrame:
    """
    For each team-game row, look up the opponent's rolling stats
    (season, L5, L10) as of that game date.

    Strategy:
      - Sort by game_datetime_utc
      - For each team compute expanding/rolling means with shift(1)
      - Join back to the original df on (opponent_id, game_datetime_utc)

    Returns a DataFrame indexed the same as df with opp_* columns added.
    """
    required = {"team_id", "game_datetime_utc", "opponent_id"}
    missing = required - set(df.columns)
    if missing:
        log.warning(f"_build_opponent_lookup: missing columns {missing} , skipping SOS")
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["team_id", "_sort_dt"])
    null_sort_dt_rate = float(df["_sort_dt"].isna().mean())
    audit["opponent_lookup_sort_dt_null_rate"] = null_sort_dt_rate
    log.info("_build_opponent_lookup: _sort_dt null_rate=%.4f", null_sort_dt_rate)

    # Metrics we want to pull from the opponent side
    opp_src_metrics = [
        "ortg", "drtg", "net_rtg", "efg_pct", "pace",
        # What opponents ALLOW/FORCE (for context metrics)
        "efg_pct",
        "orb_pct", "drb_pct", "tov_pct", "ftr",
    ]
    opp_src_metrics = list(dict.fromkeys(opp_src_metrics))
    present_metrics = [m for m in opp_src_metrics if m in df.columns]

    if not present_metrics:
        log.warning("_build_opponent_lookup: no SOS metrics found in input frame")
        return df

    # Guard against object/mixed dtypes causing DataError in expanding/rolling mean
    numeric_metrics = []
    dropped_metrics = []
    for metric in present_metrics:
        coerced = pd.to_numeric(df[metric], errors="coerce")
        has_numeric = coerced.notna().any()
        if has_numeric:
            df[metric] = coerced.astype("float64")
            numeric_metrics.append(metric)
        else:
            dropped_metrics.append(metric)

    if dropped_metrics:
        log.warning(
            "_build_opponent_lookup: dropping non-numeric SOS metrics %s",
            dropped_metrics,
        )

    if not numeric_metrics:
        log.error("_build_opponent_lookup: no numeric SOS metrics available")
        return df

    key_col = "_sort_dt"
    if "event_id" in df.columns:
        df["event_id"] = _canonical_id_series(df["event_id"])
        event_id_null_rate = float(df["event_id"].isna().mean())
        audit["opponent_lookup_event_id_null_rate"] = event_id_null_rate
        if event_id_null_rate == 0.0:
            key_col = "event_id"
    audit["opponent_lookup_join_key"] = key_col
    log.info("_build_opponent_lookup: join key=%s", key_col)

    team_stats = df[["team_id", key_col] + numeric_metrics].copy()

    rolled = team_stats.groupby("team_id")[numeric_metrics].transform(
        lambda s: _expanding_mean(s)
    )
    rolled.columns = [f"_opp_{c}_season" for c in numeric_metrics]
    team_stats = pd.concat([team_stats, rolled], axis=1)

    for w in SOS_WINDOWS:
        r = team_stats.groupby("team_id")[numeric_metrics].transform(
            lambda s, ww=w: _rolling_mean(s, ww)
        )
        r.columns = [f"_opp_{c}_l{w}" for c in numeric_metrics]
        team_stats = pd.concat([team_stats, r], axis=1)

    stat_cols = [c for c in team_stats.columns if c.startswith("_opp_")]
    lookup = team_stats[["team_id", key_col] + stat_cols].copy()
    lookup = lookup.rename(columns={"team_id": "opponent_id"})
    if key_col == "_sort_dt":
        lookup = lookup.rename(columns={"_sort_dt": "_opp_dt"})
        left_on = ["opponent_id", "_sort_dt"]
        right_on = ["opponent_id", "_opp_dt"]
    else:
        left_on = ["opponent_id", "event_id"]
        right_on = ["opponent_id", "event_id"]

    pre_dedupe = _dup_report(lookup, right_on, "_build_opponent_lookup:lookup_pre_dedupe")
    if pre_dedupe.get("duplicate_keys"):
        log.warning(
            "_build_opponent_lookup: collapsing %s duplicate lookup keys with mean aggregation",
            pre_dedupe["duplicate_keys"],
        )
        lookup = lookup.groupby(right_on, dropna=False, as_index=False)[stat_cols].mean()
    _dup_report(lookup, right_on, "_build_opponent_lookup:lookup_post_dedupe")

    df = _safe_many_to_one_merge(
        left=df,
        right=lookup,
        left_on=left_on,
        right_on=right_on,
        stage="_build_opponent_lookup",
        audit=audit,
        match_probe_cols=stat_cols,
    )
    df = df.drop(columns=["_opp_dt"], errors="ignore")

    return df


# ── Step 2: Build "what opponents allow/force" lookup ─────────────────────────

def _build_allowed_forced_lookup(df: pd.DataFrame, audit: dict[str, Any]) -> pd.DataFrame:
    """
    For each team, compute rolling averages of what THEIR OPPONENTS
    allowed/forced to OTHER opponents, giving us an opponent-context baseline.

    Example:
      Team A shot eFG%=55% vs Team B, but Team B typically allows 52%
      to opponents, so Team A is +3% above what Team B usually allows.

    We do this by:
      1. For each game row, find the opponent (Team B)
      2. Look at Team B's defensive view (what offenses did vs Team B)
      3. Average that prior to this game date

    Concretely: for each row (Team A vs Team B), the "opp_allows_efg" is the
    average eFG% that teams other than Team A shot against Team B, prior to
    this game date.
    """
    required = {"team_id", "opponent_id", "game_datetime_utc"}
    if not required.issubset(df.columns):
        log.warning("_build_allowed_forced_lookup: missing required columns , skipping")
        return df

    df = df.copy()
    if "_sort_dt" not in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    null_sort_dt_rate = float(df["_sort_dt"].isna().mean())
    audit["allowed_lookup_sort_dt_null_rate"] = null_sort_dt_rate
    log.info("_build_allowed_forced_lookup: _sort_dt null_rate=%.4f", null_sort_dt_rate)

    allow_metrics = {
        "efg_pct": "allows_efg",
        "orb_pct": "allows_orb",
        "drb_pct": "allows_drb",
        "tov_pct": "forces_tov",
        "ftr": "allows_ftr",
    }

    present = {k: v for k, v in allow_metrics.items() if k in df.columns}
    if not present:
        return df

    key_col = "_sort_dt"
    if "event_id" in df.columns:
        df["event_id"] = _canonical_id_series(df["event_id"])
        event_id_null_rate = float(df["event_id"].isna().mean())
        audit["allowed_lookup_event_id_null_rate"] = event_id_null_rate
        if event_id_null_rate == 0.0:
            key_col = "event_id"
    audit["allowed_lookup_join_key"] = key_col
    log.info("_build_allowed_forced_lookup: join key=%s", key_col)

    defense_view = df[["opponent_id", key_col, "team_id"] + list(present.keys())].copy()
    defense_view = defense_view.rename(columns={"opponent_id": "def_team_id", "team_id": "off_team_id"})
    defense_view = defense_view.sort_values(["def_team_id", key_col])

    for src_col, label in present.items():
        defense_view[f"_allow_{label}_season"] = defense_view.groupby("def_team_id")[src_col].transform(
            _expanding_mean
        )
        for w in SOS_WINDOWS:
            defense_view[f"_allow_{label}_l{w}"] = defense_view.groupby("def_team_id")[src_col].transform(
                lambda s, ww=w: _rolling_mean(s, ww)
            )

    allow_cols = [c for c in defense_view.columns if c.startswith("_allow_")]
    lookup = defense_view[["def_team_id", key_col] + allow_cols].copy()
    lookup = lookup.rename(columns={"def_team_id": "opponent_id"})
    if key_col == "_sort_dt":
        lookup = lookup.rename(columns={"_sort_dt": "_allow_dt"})
        left_on = ["opponent_id", "_sort_dt"]
        right_on = ["opponent_id", "_allow_dt"]
    else:
        left_on = ["opponent_id", "event_id"]
        right_on = ["opponent_id", "event_id"]

    pre_dedupe = _dup_report(lookup, right_on, "_build_allowed_forced_lookup:lookup_pre_dedupe")
    if pre_dedupe.get("duplicate_keys"):
        log.warning(
            "_build_allowed_forced_lookup: collapsing %s duplicate lookup keys with mean aggregation",
            pre_dedupe["duplicate_keys"],
        )
        lookup = lookup.groupby(right_on, dropna=False, as_index=False)[allow_cols].mean()
    _dup_report(lookup, right_on, "_build_allowed_forced_lookup:lookup_post_dedupe")

    df = _safe_many_to_one_merge(
        left=df,
        right=lookup,
        left_on=left_on,
        right_on=right_on,
        stage="_build_allowed_forced_lookup",
        audit=audit,
        match_probe_cols=allow_cols,
    )
    df = df.drop(columns=["_allow_dt"], errors="ignore")

    return df


# ── Step 3: Compute SOS and context metrics ───────────────────────────────────

def _add_sos_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename internal _opp_* columns to clean opp_* SOS columns.
    """
    rename_map = {}
    for suffix in ["season"] + [f"l{w}" for w in SOS_WINDOWS]:
        for metric, label in [
            ("ortg", "opp_avg_ortg"),
            ("drtg", "opp_avg_drtg"),
            ("net_rtg", "opp_avg_net_rtg"),
            ("efg_pct", "opp_avg_efg"),
            ("pace", "opp_avg_pace"),
        ]:
            src = f"_opp_{metric}_{suffix}"
            dst = f"{label}_{suffix}"
            if src in df.columns:
                rename_map[src] = dst

    df = df.rename(columns=rename_map)
    return df


def _add_context_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each window, compute team metric minus opponent's allowed/forced baseline.
    Positive means team performed above what that opponent typically allows/forces.
    Negative means below.
    """
    context_pairs = [
        ("efg_pct", "allows_efg", "efg_vs_opp"),
        ("orb_pct", "allows_orb", "orb_vs_opp"),
        ("drb_pct", "allows_drb", "drb_vs_opp"),
        ("tov_pct", "forces_tov", "tov_vs_opp"),
        ("ftr", "allows_ftr", "ftr_vs_opp"),
    ]

    for suffix in ["season"] + [f"l{w}" for w in SOS_WINDOWS]:
        for team_col, allow_base, out_base in context_pairs:
            allow_col = f"_allow_{allow_base}_{suffix}"
            out_col = f"{out_base}_{suffix}"
            if team_col in df.columns and allow_col in df.columns:
                team_vals = pd.to_numeric(df[team_col], errors="coerce")
                allow_vals = pd.to_numeric(df[allow_col], errors="coerce")
                df[out_col] = (team_vals - allow_vals).round(2)

    drop_cols = [c for c in df.columns if c.startswith("_allow_") or c.startswith("_opp_")]
    df = df.drop(columns=drop_cols, errors="ignore")

    return df


def _add_adjusted_pace(df: pd.DataFrame) -> pd.DataFrame:
    """
    True adjusted pace using opponent pace data.

    adj_pace = (team_pace / avg(team_pace, opp_avg_pace_season)) * league_avg_pace

    Accounts for opponent tempo:
      A team that plays fast against slow opponents is genuinely faster than one
      that looks fast mainly because opponents push pace.

    league_avg_pace is computed dynamically from the dataset (median pace),
    with a safe fallback.
    """
    if "pace" not in df.columns or "opp_avg_pace_season" not in df.columns:
        return df

    league_avg_pace = _get_league_avg_pace(df)

    team_pace = pd.to_numeric(df["pace"], errors="coerce")
    opp_pace = pd.to_numeric(df["opp_avg_pace_season"], errors="coerce")
    avg_pace = (team_pace + opp_pace) / 2

    df["adj_pace"] = (team_pace / avg_pace.replace(0, np.nan) * league_avg_pace).round(1)
    return df


def _add_adjusted_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute opponent-adjusted efficiency ratings.

    adj_ortg = ortg - (opp_avg_drtg_season - LEAGUE_AVG_DRTG)
      How well did you score, accounting for how tough the defense was?

    adj_drtg = drtg + (opp_avg_ortg_season - LEAGUE_AVG_ORTG)
      How well did you defend, accounting for how tough the offense was?

    adj_net_rtg = adj_ortg - adj_drtg
    """
    if "ortg" not in df.columns:
        return df

    ortg = pd.to_numeric(df.get("ortg", np.nan), errors="coerce")
    drtg = pd.to_numeric(df.get("drtg", np.nan), errors="coerce")
    opp_drtg = pd.to_numeric(df.get("opp_avg_drtg_season", np.nan), errors="coerce")
    opp_ortg = pd.to_numeric(df.get("opp_avg_ortg_season", np.nan), errors="coerce")

    df["adj_ortg"] = (ortg - (opp_drtg - LEAGUE_AVG_DRTG)).round(1)
    df["adj_drtg"] = (drtg + (opp_ortg - LEAGUE_AVG_ORTG)).round(1)
    df["adj_net_rtg"] = (df["adj_ortg"] - df["adj_drtg"]).round(1)

    return df


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_sos_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full SOS pipeline. Input should be the output of compute_all_metrics()
    (team_game_metrics.csv with per-game advanced stats).

    Requires columns:
      team_id, opponent_id, game_datetime_utc,
      ortg, drtg, net_rtg, efg_pct, pace,
      orb_pct, drb_pct, tov_pct, ftr

    Returns df with SOS, context, and adjusted rating columns appended.
    Output is written to team_game_sos.csv by the pipeline.
    """
    audit: dict[str, Any] = {
        "generated_at_utc": _utc_now(),
        "status": "started",
        "input_rows": int(len(df)),
        "input_columns": int(len(df.columns)),
        "join_key_columns_present": {
            "event_id": "event_id" in df.columns,
            "game_id": "game_id" in df.columns,
            "game_datetime_utc": "game_datetime_utc" in df.columns,
        },
        "join_key_dtypes": {c: str(df[c].dtype) for c in ["team_id", "opponent_id", "event_id", "game_id", "game_datetime_utc"] if c in df.columns},
    }

    try:
        if df.empty:
            log.warning("compute_sos_metrics: empty DataFrame , skipping")
            audit["status"] = "skipped_empty"
            return df

        missing = {"team_id", "opponent_id", "game_datetime_utc"} - set(df.columns)
        if missing:
            log.warning(f"compute_sos_metrics: missing required columns {missing} , skipping")
            audit["status"] = "skipped_missing_columns"
            audit["missing_required_columns"] = sorted(missing)
            return df

        df = df.copy()
        for col in ["team_id", "opponent_id", "event_id", "game_id"]:
            if col in df.columns:
                df[col] = _canonical_id_series(df[col])
        df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
        audit["game_datetime_utc_null_rate"] = float(df["game_datetime_utc"].isna().mean())

        input_unique_keys: dict[str, Any] = {}
        input_unique_keys["team_id+game_datetime_utc"] = _dup_report(
            df, ["team_id", "game_datetime_utc"], "compute_sos_metrics:key team_id+game_datetime_utc"
        )
        input_unique_keys["team_id+opponent_id+game_datetime_utc"] = _dup_report(
            df,
            ["team_id", "opponent_id", "game_datetime_utc"],
            "compute_sos_metrics:key team_id+opponent_id+game_datetime_utc",
        )
        if "game_id" in df.columns:
            input_unique_keys["team_id+game_id"] = _dup_report(
                df, ["team_id", "game_id"], "compute_sos_metrics:key team_id+game_id"
            )
            input_unique_keys["team_id+opponent_id+game_id"] = _dup_report(
                df,
                ["team_id", "opponent_id", "game_id"],
                "compute_sos_metrics:key team_id+opponent_id+game_id",
            )
        if "event_id" in df.columns:
            input_unique_keys["team_id+event_id"] = _dup_report(
                df, ["team_id", "event_id"], "compute_sos_metrics:key team_id+event_id"
            )
            input_unique_keys["team_id+opponent_id+event_id"] = _dup_report(
                df,
                ["team_id", "opponent_id", "event_id"],
                "compute_sos_metrics:key team_id+opponent_id+event_id",
            )
        audit["input_unique_keys"] = input_unique_keys

        event_id_ready = "event_id" in df.columns and float(df["event_id"].isna().mean()) == 0.0
        game_id_ready = "game_id" in df.columns and float(df["game_id"].isna().mean()) == 0.0
        dedupe_keys = (
            ["team_id", "event_id"] if event_id_ready else (
                ["team_id", "game_id"] if game_id_ready else ["team_id", "opponent_id", "game_datetime_utc"]
            )
        )
        df = _resolve_input_duplicates(df, dedupe_keys, audit)

        # Normalize expected numeric SOS inputs early to prevent object dtype
        # from breaking downstream rolling/expanding aggregations.
        numeric_sos_cols = [
            "ortg", "drtg", "net_rtg", "efg_pct", "tov_pct",
            "orb_pct", "drb_pct", "poss", "pace", "margin",
            "points_for", "points_against", "win", "win_pct",
            "h1_margin", "h2_margin", "ortg_l5", "drtg_l5",
            "net_rtg_l5", "ortg_l10", "drtg_l10", "net_rtg_l10", "ftr",
        ]
        for col in numeric_sos_cols:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors="coerce")
                log.debug("compute_sos_metrics: cast %s to numeric", col)

        log.info(f"Computing SOS metrics for {len(df)} rows")

        input_rows = len(df)
        df = _build_opponent_lookup(df, audit)
        df = _build_allowed_forced_lookup(df, audit)
        df = _add_sos_columns(df)
        df = _add_context_metrics(df)
        df = _add_adjusted_pace(df)
        df = _add_adjusted_ratings(df)

        df = df.drop(columns=["_sort_dt"], errors="ignore")
        _enforce_row_guard("compute_sos_metrics:final", input_rows, len(df), audit)

        audit["status"] = "success"
        audit["output_rows"] = int(len(df))
        audit["output_columns"] = int(len(df.columns))
        log.info("SOS metrics complete")
        return df
    except Exception as exc:
        audit["status"] = "failed"
        audit["error"] = str(exc)
        raise
    finally:
        try:
            _write_json(SOS_SIZE_AUDIT_PATH, audit)
            log.info("compute_sos_metrics: wrote size audit -> %s", SOS_SIZE_AUDIT_PATH)
        except Exception as write_exc:  # pragma: no cover
            log.warning("compute_sos_metrics: failed to write size audit (%s)", write_exc)
