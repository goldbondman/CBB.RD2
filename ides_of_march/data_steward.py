from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_DIR, DEFAULT_HOURS_AHEAD
from .utils import (
    canonical_id,
    home_spread_to_margin,
    normalize_dt,
    safe_read_csv,
    season_id_from_dt,
    to_rate,
)


@dataclass
class DataStewardResult:
    upcoming_games: pd.DataFrame
    historical_games: pd.DataFrame
    team_history: pd.DataFrame
    audit: dict[str, Any]


# Ordered candidate column names used when normalising market source files.
# The coalesce helper tries each name in turn and fills NaN from subsequent ones.
_SPREAD_CANDIDATES: list[str] = [
    "spread_line", "closing_spread", "spread", "home_spread_current",
    "pinnacle_spread", "draftkings_spread",
]
_TOTAL_CANDIDATES: list[str] = [
    "total_line", "closing_total", "over_under", "total_current",
]
_ML_HOME_CANDIDATES: list[str] = [
    "moneyline_home", "home_ml", "home_moneyline", "ml_home",
]
_ML_AWAY_CANDIDATES: list[str] = [
    "moneyline_away", "away_ml", "away_moneyline", "ml_away",
]
_WIN_PROB_HOME_CANDIDATES: list[str] = [
    "home_win_prob", "market_home_win_prob", "home_implied_prob",
]
_WIN_PROB_AWAY_CANDIDATES: list[str] = [
    "away_win_prob", "market_away_win_prob", "away_implied_prob",
]


def _pick_first(cols: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in cols}
    for c in candidates:
        found = lookup.get(c.lower())
        if found:
            return found
    return None


def _to_numeric(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col is None or col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _coalesce_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Coalesce numeric values across multiple candidate column names.

    Tries each column name in *candidates* in order. Columns that do not exist
    in *df* are skipped. The first column that exists initialises the result
    series; subsequent columns fill in remaining NaN positions. Returns a
    Series of NaN when no candidate column is present.

    Args:
        df: Source DataFrame.
        candidates: Column names to try in priority order.

    Returns:
        Numeric Series with as many non-null values as the candidate columns
        collectively provide.
    """
    lookup = {c.lower(): c for c in df.columns}
    result: pd.Series | None = None
    for c in candidates:
        found = lookup.get(c.lower())
        if found is None:
            continue
        values = pd.to_numeric(df[found], errors="coerce")
        if result is None:
            result = values
        else:
            result = result.where(result.notna(), values)
        if result.notna().all():
            break
    return result if result is not None else pd.Series(np.nan, index=df.index)


def _to_bool_completed(df: pd.DataFrame) -> pd.Series:
    if "completed" in df.columns:
        raw = df["completed"]
        return raw.astype(str).str.lower().isin({"true", "1", "yes"})
    if "state" in df.columns:
        return df["state"].astype(str).str.lower().isin({"post", "final"})
    if "status_desc" in df.columns:
        return df["status_desc"].astype(str).str.lower().str.contains("final", regex=False)
    return pd.Series(False, index=df.index)


def _bool_series(df: pd.DataFrame, candidates: list[str], *, default: bool = False) -> pd.Series:
    for col in candidates:
        if col in df.columns:
            return df[col].astype(str).str.lower().isin({"true", "1", "yes", "y"})
    return pd.Series(default, index=df.index, dtype=bool)


def _text_flag(df: pd.DataFrame, columns: list[str], patterns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(False, index=df.index, dtype=bool)
    text = pd.Series("", index=df.index, dtype=object)
    for col in columns:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str)
    txt = text.str.lower()
    out = pd.Series(False, index=df.index, dtype=bool)
    for p in patterns:
        out = out | txt.str.contains(p, regex=False)
    return out


def _derive_postseason_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out.get("game_datetime_utc"), utc=True, errors="coerce")
    month = dt.dt.month
    day = dt.dt.day

    # Neutral-site signal from any known venue/site columns.
    is_neutral = _bool_series(out, ["neutral_site", "neutral", "is_neutral"], default=False)
    if "site_type" in out.columns:
        is_neutral = is_neutral | out["site_type"].astype(str).str.lower().eq("neutral")

    text_cols = [
        c
        for c in ["phase", "round_name", "tournament_type", "status_desc", "venue", "odds_details"]
        if c in out.columns
    ]
    explicit_conf = _bool_series(out, ["conference_tournament_flag", "is_conference_tournament"], default=False)
    explicit_ncaa = _bool_series(out, ["ncaa_tournament_flag", "is_ncaa_tournament"], default=False)
    tag_conf = _text_flag(out, text_cols, ["conference tournament"])
    tag_ncaa = _text_flag(
        out,
        text_cols,
        ["ncaa tournament", "march madness", "first four", "sweet 16", "elite 8", "final four"],
    )

    # Conservative fallback when explicit tournament tags are absent.
    heuristic_ncaa = is_neutral & month.isin([3, 4]) & (day >= 15)
    heuristic_conf = is_neutral & month.eq(3) & (day < 15)

    out["is_neutral"] = is_neutral.astype(bool)
    out["is_ncaa_tournament"] = (explicit_ncaa | tag_ncaa | heuristic_ncaa).astype(bool)
    out["is_conference_tournament"] = (explicit_conf | tag_conf | heuristic_conf) & (~out["is_ncaa_tournament"])
    out["is_postseason"] = (out["is_conference_tournament"] | out["is_ncaa_tournament"]).astype(bool)

    # Home-court can only apply in explicit non-neutral non-postseason context.
    out["home_bonus_eligible"] = ((~out["is_neutral"]) & (~out["is_postseason"])).astype(bool)

    phase = np.where(
        out["is_ncaa_tournament"],
        "ncaa_tournament",
        np.where(out["is_conference_tournament"], "conference_tournament", "regular_season"),
    )
    out["phase"] = out.get("phase", pd.Series(phase, index=out.index)).fillna(pd.Series(phase, index=out.index))
    if "round_name" not in out.columns:
        out["round_name"] = np.nan
    return out


def _canonical_game_frame(games: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()
    if "event_id" not in out.columns and "game_id" in out.columns:
        out["event_id"] = out["game_id"]
    if "game_id" not in out.columns and "event_id" in out.columns:
        out["game_id"] = out["event_id"]
    out["event_id"] = out.get("event_id", pd.Series("", index=out.index)).map(canonical_id)
    out["game_id"] = out.get("game_id", pd.Series("", index=out.index)).map(canonical_id)
    out["event_id"] = out["event_id"].where(out["event_id"] != "", out["game_id"])
    out["game_id"] = out["game_id"].where(out["game_id"] != "", out["event_id"])
    out["game_datetime_utc"] = normalize_dt(out, "game_datetime_utc")
    for col in ["home_team_id", "away_team_id"]:
        if col in out.columns:
            out[col] = out[col].map(canonical_id)
    out = _derive_postseason_flags(out)
    return out


def _build_team_history(team_games: pd.DataFrame) -> pd.DataFrame:
    df = team_games.copy()
    cols = list(df.columns)

    if "event_id" not in df.columns and "game_id" in df.columns:
        df["event_id"] = df["game_id"]
    df["event_id"] = df.get("event_id", pd.Series("", index=df.index)).map(canonical_id)
    df["team_id"] = df.get("team_id", pd.Series("", index=df.index)).map(canonical_id)
    df["opponent_id"] = df.get("opponent_id", pd.Series("", index=df.index)).map(canonical_id)
    df["game_datetime_utc"] = normalize_dt(df, "game_datetime_utc")
    df = df[df["team_id"] != ""].copy()
    df = df[df["game_datetime_utc"].notna()].copy()

    adj_em_col = _pick_first(cols, ["adj_net_rtg", "adj_em", "net_rtg"])
    efg_col = _pick_first(cols, ["efg_pct", "efg", "fg_pct"])
    tov_col = _pick_first(cols, ["tov_pct"])
    orb_col = _pick_first(cols, ["orb_pct"])
    ftr_col = _pick_first(cols, ["ftr"])
    ft_col = _pick_first(cols, ["ft_pct"])
    pace_col = _pick_first(cols, ["pace", "adj_pace"])
    three_col = _pick_first(cols, ["three_par", "three_pa_rate"])
    drb_col = _pick_first(cols, ["drb_pct"])
    rest_col = _pick_first(cols, ["rest_days"])
    sos_col = _pick_first(cols, ["opp_avg_net_rtg_l10", "opp_avg_net_rtg_season"])

    df["adj_em"] = _to_numeric(df, adj_em_col)
    df["efg_pct"] = pd.Series(to_rate(df[efg_col] if efg_col else pd.Series(np.nan, index=df.index)), index=df.index)
    df["tov_pct"] = pd.Series(to_rate(df[tov_col] if tov_col else pd.Series(np.nan, index=df.index)), index=df.index)
    df["orb_pct"] = pd.Series(to_rate(df[orb_col] if orb_col else pd.Series(np.nan, index=df.index)), index=df.index)
    df["ftr"] = pd.Series(to_rate(df[ftr_col] if ftr_col else pd.Series(np.nan, index=df.index)), index=df.index)
    df["ft_pct"] = pd.Series(to_rate(df[ft_col] if ft_col else pd.Series(np.nan, index=df.index)), index=df.index)
    df["pace"] = _to_numeric(df, pace_col)
    df["three_par"] = pd.Series(to_rate(df[three_col] if three_col else pd.Series(np.nan, index=df.index)), index=df.index)
    df["drb_pct"] = pd.Series(to_rate(df[drb_col] if drb_col else pd.Series(np.nan, index=df.index)), index=df.index)
    df["ft_scoring_pressure"] = df["ftr"] * df["ft_pct"]
    df["sos_pre"] = _to_numeric(df, sos_col)

    df = df.sort_values(["team_id", "game_datetime_utc", "event_id"], kind="mergesort").reset_index(drop=True)
    df["season_id"] = season_id_from_dt(df["game_datetime_utc"])

    if rest_col and rest_col in df.columns:
        df["days_rest"] = _to_numeric(df, rest_col)
    else:
        diff = df.groupby(["team_id", "season_id"], dropna=False)["game_datetime_utc"].diff().dt.total_seconds() / 86400.0
        df["days_rest"] = diff.clip(lower=0.0)

    rolling_metrics = [
        "adj_em",
        "efg_pct",
        "tov_pct",
        "orb_pct",
        "ftr",
        "ft_pct",
        "ft_scoring_pressure",
        "pace",
        "three_par",
        "drb_pct",
    ]
    grouped = df.groupby(["team_id", "season_id"], dropna=False)
    for metric in rolling_metrics:
        shifted = grouped[metric].shift(1)
        df[f"{metric}_l5"] = shifted.groupby([df["team_id"], df["season_id"]]).rolling(5, min_periods=3).mean().reset_index(level=[0, 1], drop=True)
        df[f"{metric}_l12"] = shifted.groupby([df["team_id"], df["season_id"]]).rolling(12, min_periods=6).mean().reset_index(level=[0, 1], drop=True)

    df["Last5_AdjEM"] = df["adj_em_l5"]
    df["Last12_AdjEM"] = df["adj_em_l12"]
    df["Form_Delta"] = df["Last5_AdjEM"] - df["Last12_AdjEM"]

    if df["sos_pre"].isna().all():
        df["sos_pre"] = grouped["adj_em"].shift(1).groupby([df["team_id"], df["season_id"]]).rolling(12, min_periods=6).mean().reset_index(level=[0, 1], drop=True)

    return df


def _canonical_market(market: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return market
    out = market.copy()
    if "event_id" not in out.columns and "game_id" in out.columns:
        out["event_id"] = out["game_id"]
    out["event_id"] = out.get("event_id", pd.Series("", index=out.index)).map(canonical_id)
    out["game_datetime_utc"] = normalize_dt(out, "game_datetime_utc")
    out["market_spread"] = _coalesce_numeric(out, _SPREAD_CANDIDATES)
    out["market_total"] = _coalesce_numeric(out, _TOTAL_CANDIDATES)
    out["moneyline_home"] = _coalesce_numeric(out, _ML_HOME_CANDIDATES)
    out["moneyline_away"] = _coalesce_numeric(out, _ML_AWAY_CANDIDATES)
    out["market_home_win_prob"] = _coalesce_numeric(out, _WIN_PROB_HOME_CANDIDATES)
    out["market_away_win_prob"] = _coalesce_numeric(out, _WIN_PROB_AWAY_CANDIDATES)
    out["line_source_used"] = out.get("line_source_used", out.get("source", pd.Series("unknown", index=out.index)))
    keep = [
        "event_id",
        "game_datetime_utc",
        "market_spread",
        "market_total",
        "line_source_used",
        "moneyline_home",
        "moneyline_away",
        "market_home_win_prob",
        "market_away_win_prob",
    ]
    for col in keep:
        if col not in out.columns:
            out[col] = np.nan
    out = out[keep].copy()
    out = out.sort_values(["event_id", "game_datetime_utc"], kind="mergesort").drop_duplicates("event_id", keep="last")
    return out


def _combine_market_sources(primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    if primary.empty:
        return fallback.copy()
    if fallback.empty:
        return primary.copy()

    p = primary.set_index("event_id", drop=False).copy()
    f = fallback.set_index("event_id", drop=False).copy()
    fill_cols = [
        "game_datetime_utc",
        "market_spread",
        "market_total",
        "moneyline_home",
        "moneyline_away",
        "market_home_win_prob",
        "market_away_win_prob",
        "line_source_used",
    ]
    for col in fill_cols:
        if col not in p.columns:
            p[col] = np.nan
        if col not in f.columns:
            f[col] = np.nan
        p[col] = p[col].where(p[col].notna(), f[col])
    only_fallback = f.loc[~f.index.isin(p.index)]
    out = pd.concat([p, only_fallback], axis=0)
    out = out.reset_index(drop=True)
    out = out.sort_values(["event_id", "game_datetime_utc"], kind="mergesort").drop_duplicates("event_id", keep="last")
    return out


def _merge_team_asof(games: pd.DataFrame, team_history: pd.DataFrame, *, side: str) -> pd.DataFrame:
    team_col = f"{side}_team_id"
    base_cols = [
        "adj_em_l5",
        "adj_em_l12",
        "Last5_AdjEM",
        "Last12_AdjEM",
        "Form_Delta",
        "efg_pct_l5",
        "tov_pct_l5",
        "orb_pct_l5",
        "ftr_l5",
        "ft_pct_l5",
        "ft_scoring_pressure_l5",
        "pace_l5",
        "three_par_l5",
        "drb_pct_l5",
        "sos_pre",
        "days_rest",
    ]

    left = games[["event_id", "game_id", "game_datetime_utc", team_col]].copy()
    left = left.rename(columns={team_col: "team_id", "game_datetime_utc": "target_datetime"})
    left["team_id"] = left["team_id"].map(canonical_id)
    left = left[left["team_id"] != ""].copy()

    right = team_history[["team_id", "game_datetime_utc"] + [c for c in base_cols if c in team_history.columns]].copy()
    # merge_asof enforces monotonic order on the merge key itself.
    right = right.sort_values(["game_datetime_utc", "team_id"], kind="mergesort")
    left = left.sort_values(["target_datetime", "team_id"], kind="mergesort")

    merged = pd.merge_asof(
        left,
        right,
        left_on="target_datetime",
        right_on="game_datetime_utc",
        by="team_id",
        direction="backward",
        allow_exact_matches=False,
    )
    merged = merged.rename(columns={"team_id": team_col})
    rename_cols = {c: f"{side}_{c}" for c in base_cols if c in merged.columns}
    merged = merged.rename(columns=rename_cols)
    keep = ["event_id", "game_id", team_col] + list(rename_cols.values())
    return merged[keep]


def _build_game_feature_frame(games: pd.DataFrame, market: pd.DataFrame, team_history: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    g = games.copy()
    g = g[g["game_datetime_utc"].notna()].copy()
    g = g.sort_values("game_datetime_utc", kind="mergesort")

    home = _merge_team_asof(g, team_history, side="home")
    away = _merge_team_asof(g, team_history, side="away")

    out = g.merge(home, on=["event_id", "game_id", "home_team_id"], how="left")
    out = out.merge(away, on=["event_id", "game_id", "away_team_id"], how="left")
    out = out.merge(market, on="event_id", how="left", suffixes=("", "_mkt"))

    out["market_spread"] = pd.to_numeric(out.get("market_spread"), errors="coerce")
    out["market_total"] = pd.to_numeric(out.get("market_total"), errors="coerce")

    # Fall back to spread/over_under columns embedded in games.csv when market
    # lines are unavailable (e.g. conference-tournament games without posted lines).
    if "spread" in out.columns:
        fallback_spread = pd.to_numeric(out["spread"], errors="coerce")
        out["market_spread"] = out["market_spread"].where(out["market_spread"].notna(), fallback_spread)
    if "over_under" in out.columns:
        fallback_total = pd.to_numeric(out["over_under"], errors="coerce")
        out["market_total"] = out["market_total"].where(out["market_total"].notna(), fallback_total)

    # Preserve pre-derived venue/tournament flags from canonical game frame.
    out["is_neutral"] = out.get("is_neutral", pd.Series(False, index=out.index)).astype(bool)
    out["is_conference_tournament"] = out.get("is_conference_tournament", pd.Series(False, index=out.index)).astype(bool)
    out["is_ncaa_tournament"] = out.get("is_ncaa_tournament", pd.Series(False, index=out.index)).astype(bool)
    out["is_postseason"] = out.get("is_postseason", pd.Series(False, index=out.index)).astype(bool)
    out["home_bonus_eligible"] = out.get("home_bonus_eligible", ((~out["is_neutral"]) & (~out["is_postseason"]))).astype(bool)

    out["rest_diff"] = pd.to_numeric(out.get("home_days_rest"), errors="coerce") - pd.to_numeric(out.get("away_days_rest"), errors="coerce")
    out["form_delta_diff"] = pd.to_numeric(out.get("home_Form_Delta"), errors="coerce") - pd.to_numeric(out.get("away_Form_Delta"), errors="coerce")
    out["sos_diff"] = pd.to_numeric(out.get("home_sos_pre"), errors="coerce") - pd.to_numeric(out.get("away_sos_pre"), errors="coerce")

    out["adj_em_margin_l12"] = pd.to_numeric(out.get("home_adj_em_l12"), errors="coerce") - pd.to_numeric(out.get("away_adj_em_l12"), errors="coerce")
    out["efg_margin_l5"] = pd.to_numeric(out.get("home_efg_pct_l5"), errors="coerce") - pd.to_numeric(out.get("away_efg_pct_l5"), errors="coerce")
    out["to_margin_l5"] = pd.to_numeric(out.get("away_tov_pct_l5"), errors="coerce") - pd.to_numeric(out.get("home_tov_pct_l5"), errors="coerce")
    out["oreb_margin_l5"] = pd.to_numeric(out.get("home_orb_pct_l5"), errors="coerce") - pd.to_numeric(out.get("away_orb_pct_l5"), errors="coerce")
    out["ftr_margin_l5"] = pd.to_numeric(out.get("home_ftr_l5"), errors="coerce") - pd.to_numeric(out.get("away_ftr_l5"), errors="coerce")
    out["ft_scoring_pressure_margin_l5"] = pd.to_numeric(out.get("home_ft_scoring_pressure_l5"), errors="coerce") - pd.to_numeric(out.get("away_ft_scoring_pressure_l5"), errors="coerce")

    out["expected_pace"] = (
        pd.to_numeric(out.get("home_pace_l5"), errors="coerce")
        + pd.to_numeric(out.get("away_pace_l5"), errors="coerce")
    ) / 2.0
    out["expected_total"] = out["expected_pace"] * (
        (110.0 + pd.to_numeric(out.get("home_adj_em_l12"), errors="coerce") / 2.0)
        + (110.0 + pd.to_numeric(out.get("away_adj_em_l12"), errors="coerce") / 2.0)
    ) / 100.0

    out["active_as_of_utc"] = as_of
    return out


def _build_historical_labels(games: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    g = games.copy()
    g["home_score"] = pd.to_numeric(g.get("home_score"), errors="coerce")
    g["away_score"] = pd.to_numeric(g.get("away_score"), errors="coerce")
    g["actual_margin"] = g["home_score"] - g["away_score"]
    g["actual_total"] = g["home_score"] + g["away_score"]
    g["home_won"] = (g["actual_margin"] > 0).astype(float)
    g = g.merge(market[["event_id", "market_spread", "market_total"]], on="event_id", how="left")
    g["actual_home_covered"] = (g["actual_margin"] > home_spread_to_margin(g["market_spread"]))
    return g


def build_data_steward_frame(
    *,
    data_dir: Path = DATA_DIR,
    as_of: pd.Timestamp,
    hours_ahead: int = DEFAULT_HOURS_AHEAD,
    hours_back: int = 1,
) -> DataStewardResult:
    games = safe_read_csv(data_dir / "games.csv")
    team_games = safe_read_csv(data_dir / "team_game_weighted.csv")
    market_by_game = safe_read_csv(data_dir / "market_lines_latest_by_game.csv")
    market_latest = safe_read_csv(data_dir / "market_lines_latest.csv")

    audit: dict[str, Any] = {
        "inputs": {
            "games_rows": int(len(games)),
            "team_game_weighted_rows": int(len(team_games)),
            "market_rows": int(max(len(market_by_game), len(market_latest))),
        },
        "as_of": as_of.isoformat(),
        "hours_ahead": int(hours_ahead),
        "hours_back": int(hours_back),
    }

    if games.empty or team_games.empty:
        return DataStewardResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), audit)

    games_norm = _canonical_game_frame(games)
    market_norm = _combine_market_sources(_canonical_market(market_by_game), _canonical_market(market_latest))
    team_history = _build_team_history(team_games)

    completed = _to_bool_completed(games_norm)
    horizon_start = as_of - pd.Timedelta(hours=hours_back)
    horizon_end = as_of + pd.Timedelta(hours=hours_ahead)
    upcoming_games = games_norm[
        (~completed)
        & (games_norm["game_datetime_utc"] >= horizon_start)
        & (games_norm["game_datetime_utc"] <= horizon_end)
    ].copy()
    completed_games = games_norm[completed].copy()

    upcoming_frame = _build_game_feature_frame(upcoming_games, market_norm, team_history, as_of)
    historical_feature_frame = _build_game_feature_frame(completed_games, market_norm, team_history, as_of)
    labels = _build_historical_labels(completed_games, market_norm)
    historical_frame = historical_feature_frame.merge(
        labels[["event_id", "actual_margin", "actual_total", "home_won", "actual_home_covered"]],
        on="event_id",
        how="left",
    )

    audit["derived"] = {
        "upcoming_games": int(len(upcoming_frame)),
        "historical_games": int(len(historical_frame)),
        "team_history_rows": int(len(team_history)),
        "upcoming_window_start_utc": horizon_start.isoformat(),
        "upcoming_window_end_utc": horizon_end.isoformat(),
        "market_coverage_upcoming": float(upcoming_frame["market_spread"].notna().mean()) if len(upcoming_frame) else 0.0,
    }

    return DataStewardResult(
        upcoming_games=upcoming_frame,
        historical_games=historical_frame,
        team_history=team_history,
        audit=audit,
    )
