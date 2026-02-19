"""
ESPN CBB Pipeline — Injury Proxy Detection
Detects probable injuries and performance degradation from player box score
history without requiring official injury reports.

No I/O here — pure DataFrame transformations.
Output written to player_injury_proxy.csv by the pipeline.

Signal categories:

AVAILABILITY FLAGS
  dnp_flag             Player listed as DNP this game
  dnp_prev_game        Was DNP in immediately prior game
  games_missed_l14     Games missed (DNP or absent) in last 14 days
  sudden_absence       Active player (3+ games) who suddenly disappears from
                       box score entirely (not even listed as DNP)

MINUTES SIGNALS
  min_pct_of_season    This game minutes as % of season avg minutes
  min_drop_vs_l5       Minutes vs their own L5 average (z-score)
  min_drop_flag        Minutes dropped >30% vs L5 average
  min_drop_severe      Minutes dropped >50% vs L5 average (probable injury)
  starter_to_bench     Was a starter, now coming off bench

PERFORMANCE SIGNALS
  pts_z_l10            Points z-score vs player's own L10 average
  efg_z_l10            eFG% z-score vs player's own L10 average
  usage_z_l10          Usage proxy (FGA+FTA+TOV) z-score vs L10
  multi_stat_down      3+ stats simultaneously below their L10 average by >1 SD

COMPOSITE SCORES
  injury_proxy_score   0-100 composite score weighting all signals.
                       >50 = likely something going on
                       >75 = strong injury signal, worth flagging
  injury_proxy_flag    Binary: score > 50
  injury_proxy_severe  Binary: score > 75
"""

import logging
from typing import List

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
MIN_GAMES_FOR_BASELINE = 3     # minimum games before computing z-scores
MIN_DROP_FLAG          = 0.30  # 30% minutes drop = flag
MIN_DROP_SEVERE        = 0.50  # 50% minutes drop = severe
SCORE_WEIGHTS = {
    "dnp_flag":          30,
    "sudden_absence":    25,
    "min_drop_severe":   25,
    "min_drop_flag":     15,
    "multi_stat_down":   15,
    "starter_to_bench":  10,
    "dnp_prev_game":     10,
    "usage_z_l10":        5,   # partial — scaled by z-score magnitude
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _rolling_mean(s: pd.Series, w: int) -> pd.Series:
    return s.shift(1).rolling(w, min_periods=MIN_GAMES_FOR_BASELINE).mean()


def _rolling_std(s: pd.Series, w: int) -> pd.Series:
    return s.shift(1).rolling(w, min_periods=MIN_GAMES_FOR_BASELINE).std()


def _z_score(s: pd.Series, w: int) -> pd.Series:
    mean = _rolling_mean(s, w)
    std  = _rolling_std(s, w).replace(0, np.nan)
    return ((s - mean) / std).round(2)


# ── Step 1: Build full player appearance history ──────────────────────────────

def _build_appearance_history(df: pd.DataFrame,
                               games_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each player-game in player_game_logs, determine whether the player
    appeared or was absent. Uses games_df (team_game_logs or games.csv) to
    know which games each team played so we can detect silent absences
    (player not listed at all, vs. listed as DNP).

    Returns player_df with added columns:
      appeared, game_number (per player), games_since_last_appearance
    """
    if df.empty:
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["athlete_id", "_sort_dt"])

    # Mark actual appearances (played with real stats)
    df["appeared"] = (
        ~df["did_not_play"].astype(bool) &
        df["min"].notna() &
        (df["min"] > 0)
    ).astype(int)

    # Game number per player (1 = first game they appear in data)
    df["game_number"] = df.groupby("athlete_id").cumcount() + 1

    # Games since last actual appearance (for silent absence detection)
    def _games_since_last(group: pd.Series) -> pd.Series:
        result = []
        last_idx = None
        for i, val in enumerate(group.values):
            if last_idx is None:
                result.append(np.nan)
            else:
                result.append(i - last_idx)
            if val == 1:
                last_idx = i
        return pd.Series(result, index=group.index)

    df["games_since_appearance"] = df.groupby("athlete_id")["appeared"].transform(
        _games_since_last
    )

    return df


# ── Step 2: Per-player availability flags ─────────────────────────────────────

def add_availability_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "_sort_dt" not in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["athlete_id", "_sort_dt"])

    did_not_play = df["did_not_play"].astype(bool)
    appeared     = df.get("appeared", (~did_not_play).astype(int))

    # DNP this game
    df["dnp_flag"] = did_not_play.astype(int)

    # DNP in previous game
    df["dnp_prev_game"] = df.groupby("athlete_id")["dnp_flag"].transform(
        lambda s: s.shift(1).fillna(0).astype(int)
    )

    # Games missed in last 14 calendar days
    def _missed_l14(group: pd.DataFrame) -> pd.Series:
        result = []
        dts   = group["_sort_dt"].values
        dnps  = group["dnp_flag"].values
        for i in range(len(dts)):
            cutoff = dts[i] - pd.Timedelta(days=14)
            missed = sum(1 for j in range(i) if dts[j] >= cutoff and dnps[j] == 1)
            result.append(missed)
        return pd.Series(result, index=group.index)

    df["games_missed_l14"] = df.groupby("athlete_id", group_keys=False).apply(
        _missed_l14
    )

    # Sudden absence — player had 3+ consecutive appearances, then went missing
    # (games_since_appearance > 1 after being active)
    df["sudden_absence"] = (
        (df.get("games_since_appearance", pd.Series(np.nan, index=df.index)) > 1) &
        (df.get("game_number", pd.Series(0, index=df.index)) > 3)
    ).astype(int)

    # Starter → bench flag
    df["starter_to_bench"] = (
        (df.groupby("athlete_id")["starter"].transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).mean()) >= 0.67) &
        (~df["starter"].astype(bool))
    ).astype(int)

    return df


# ── Step 3: Minutes signals ───────────────────────────────────────────────────

def add_minutes_signals(df: pd.DataFrame) -> pd.DataFrame:
    if "min" not in df.columns:
        return df

    df = df.copy()
    if "_sort_dt" not in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["athlete_id", "_sort_dt"])

    minutes = pd.to_numeric(df["min"], errors="coerce")

    # Season average minutes (prior games only)
    df["min_season_avg"] = df.groupby("athlete_id")["min"].transform(
        lambda s: pd.to_numeric(s, errors="coerce")
        .shift(1).expanding(min_periods=MIN_GAMES_FOR_BASELINE).mean().round(1)
    )

    # L5 average minutes
    df["min_l5_avg"] = df.groupby("athlete_id")["min"].transform(
        lambda s: pd.to_numeric(s, errors="coerce")
        .shift(1).rolling(5, min_periods=MIN_GAMES_FOR_BASELINE).mean().round(1)
    )

    # Minutes as % of season average
    df["min_pct_of_season"] = (
        minutes / df["min_season_avg"].replace(0, np.nan) * 100
    ).round(1)

    # Minutes z-score vs L5
    df["min_z_l5"] = df.groupby("athlete_id")["min"].transform(
        lambda s: _z_score(pd.to_numeric(s, errors="coerce"), 5)
    )

    # Drop flags
    min_ratio = minutes / df["min_l5_avg"].replace(0, np.nan)
    df["min_drop_flag"]   = (min_ratio < (1 - MIN_DROP_FLAG)).astype(int)
    df["min_drop_severe"] = (min_ratio < (1 - MIN_DROP_SEVERE)).astype(int)

    # Only flag for players with established baselines (5+ games)
    has_baseline = df["game_number"].ge(5) if "game_number" in df.columns else pd.Series(True, index=df.index)
    df["min_drop_flag"]   = df["min_drop_flag"].where(has_baseline, 0)
    df["min_drop_severe"] = df["min_drop_severe"].where(has_baseline, 0)

    return df


# ── Step 4: Performance signals ───────────────────────────────────────────────

def add_performance_signals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "_sort_dt" not in df.columns:
        df["_sort_dt"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df = df.sort_values(["athlete_id", "_sort_dt"])

    has_baseline = (df["game_number"].ge(MIN_GAMES_FOR_BASELINE + 1)
                    if "game_number" in df.columns
                    else pd.Series(True, index=df.index))

    # Points z-score
    if "pts" in df.columns:
        df["pts_z_l10"] = df.groupby("athlete_id")["pts"].transform(
            lambda s: _z_score(pd.to_numeric(s, errors="coerce"), 10)
        ).where(has_baseline)

    # eFG% z-score (only for players who attempt shots)
    if "fgm" in df.columns and "fga" in df.columns:
        fga = pd.to_numeric(df["fga"], errors="coerce")
        fgm = pd.to_numeric(df["fgm"], errors="coerce")
        tpm = pd.to_numeric(df.get("tpm", 0), errors="coerce").fillna(0)
        df["_efg"] = ((fgm + 0.5 * tpm) / fga.replace(0, np.nan) * 100)
        df["efg_z_l10"] = df.groupby("athlete_id")["_efg"].transform(
            lambda s: _z_score(s, 10)
        ).where(has_baseline)
        df = df.drop(columns=["_efg"], errors="ignore")

    # Usage proxy z-score: FGA + 0.44*FTA + TOV
    def _safe_col(name):
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
        return pd.Series(0.0, index=df.index)

    usage_cols = {
        "fga": _safe_col("fga"),
        "fta": _safe_col("fta"),
        "tov": _safe_col("tov"),
    }
    df["_usage"] = usage_cols["fga"] + 0.44 * usage_cols["fta"] + usage_cols["tov"]
    df["usage_z_l10"] = df.groupby("athlete_id")["_usage"].transform(
        lambda s: _z_score(s, 10)
    ).where(has_baseline)
    df = df.drop(columns=["_usage"], errors="ignore")

    # Multi-stat simultaneous decline — 3+ stats below L10 mean by >1 SD
    stat_z_cols = [c for c in ["pts_z_l10", "efg_z_l10", "usage_z_l10",
                                "min_z_l5"] if c in df.columns]
    if len(stat_z_cols) >= 3:
        below_threshold = sum(
            (df[c] < -1).astype(int) for c in stat_z_cols
        )
        df["multi_stat_down"] = (below_threshold >= 3).astype(int).where(has_baseline, 0)
    else:
        df["multi_stat_down"] = 0

    return df


# ── Step 5: Composite injury proxy score ─────────────────────────────────────

def add_injury_proxy_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted composite score from all signals.
    Score is only meaningful for players with an established baseline.
    """
    df = df.copy()

    score = pd.Series(0.0, index=df.index)

    for col, weight in SCORE_WEIGHTS.items():
        if col not in df.columns:
            continue
        if col == "usage_z_l10":
            # Scale by z-score magnitude (more negative = higher signal)
            z = pd.to_numeric(df[col], errors="coerce").fillna(0)
            contribution = (z.clip(-3, 0) / -3 * weight).clip(0, weight)
        else:
            contribution = pd.to_numeric(df[col], errors="coerce").fillna(0) * weight

        score += contribution

    df["injury_proxy_score"]  = score.clip(0, 100).round(1)
    df["injury_proxy_flag"]   = (df["injury_proxy_score"] > 50).astype(int)
    df["injury_proxy_severe"] = (df["injury_proxy_score"] > 75).astype(int)

    return df


# ── Team-level injury impact ──────────────────────────────────────────────────

def compute_team_injury_impact(player_proxy_df: pd.DataFrame,
                                team_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate player injury signals to the team level.
    Produces one row per team per game with:

      players_flagged       Count of players with injury_proxy_flag = 1
      players_severe        Count of players with injury_proxy_severe = 1
      starters_flagged      Flagged players who are/were starters
      bench_flagged         Flagged players who are bench players
      team_injury_load      Weighted sum: (flagged_score * min_share)
                            Accounts for how much a flagged player contributes
      key_player_flag       1 if a player in the team's top-3 scorers is flagged
      minutes_at_risk_pct   % of prior-game team minutes played by flagged players
    """
    if player_proxy_df.empty or team_logs_df.empty:
        return pd.DataFrame()

    df  = player_proxy_df.copy()
    tdf = team_logs_df[["event_id", "team_id", "game_datetime_utc"]].drop_duplicates()

    # Get player's share of team minutes
    total_min = (df.groupby(["event_id", "team_id"])["min"]
                 .transform("sum")
                 .replace(0, np.nan))
    df["min_share"] = pd.to_numeric(df["min"], errors="coerce") / total_min

    # Weighted injury load
    df["_weighted_score"] = df["injury_proxy_score"] * df["min_share"].fillna(0)

    flagged = df[df["injury_proxy_flag"] == 1]

    # Per-team per-game aggregation
    agg = df.groupby(["event_id", "team_id"]).agg(
        players_flagged   = ("injury_proxy_flag",  "sum"),
        players_severe    = ("injury_proxy_severe", "sum"),
        starters_flagged  = pd.NamedAgg(
            column="injury_proxy_flag",
            aggfunc=lambda x: (x & df.loc[x.index, "starter"].astype(bool)).sum()
        ),
        team_injury_load  = ("_weighted_score", "sum"),
        minutes_at_risk_pct = pd.NamedAgg(
            column="min_share",
            aggfunc=lambda x: x[df.loc[x.index, "injury_proxy_flag"] == 1].sum() * 100
        ),
    ).reset_index()

    # Key player flag: is one of the team's top-3 scorers flagged?
    top_scorers = (
        df[df["appeared"] == 1]
        .groupby(["team_id", "athlete_id"])["pts"]
        .apply(lambda s: pd.to_numeric(s, errors="coerce").mean())
        .reset_index()
        .sort_values(["team_id", "pts"], ascending=[True, False])
        .groupby("team_id")
        .head(3)
    )
    top_scorer_ids = set(top_scorers["athlete_id"].astype(str))

    key_flag = (
        flagged[flagged["athlete_id"].astype(str).isin(top_scorer_ids)]
        .groupby(["event_id", "team_id"])
        .size()
        .gt(0)
        .astype(int)
        .reset_index()
        .rename(columns={0: "key_player_flag"})
    )

    agg = agg.merge(key_flag, on=["event_id", "team_id"], how="left")
    agg["key_player_flag"] = agg["key_player_flag"].fillna(0).astype(int)
    agg["team_injury_load"] = agg["team_injury_load"].round(2)
    agg["minutes_at_risk_pct"] = agg["minutes_at_risk_pct"].round(1)

    # Join game datetime for sorting
    agg = agg.merge(tdf, on=["event_id", "team_id"], how="left")

    return agg


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_injury_proxy(player_df: pd.DataFrame,
                          team_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full injury proxy pipeline.
    Input:  player_game_logs DataFrame (full history)
    Output: player-level proxy df (for player_injury_proxy.csv)
    Also returns team-level impact separately via compute_team_injury_impact().
    """
    if player_df.empty:
        log.warning("compute_injury_proxy: empty player DataFrame")
        return player_df

    log.info(f"Computing injury proxy for {len(player_df)} player-game rows "
             f"({player_df['athlete_id'].nunique()} unique players)")

    df = _build_appearance_history(player_df, team_logs_df)
    df = add_availability_flags(df)
    df = add_minutes_signals(df)
    df = add_performance_signals(df)
    df = add_injury_proxy_score(df)

    # Drop internal columns
    df = df.drop(columns=["_sort_dt"], errors="ignore")

    log.info(f"  Flagged: {df['injury_proxy_flag'].sum():.0f} player-games "
             f"({df['injury_proxy_severe'].sum():.0f} severe)")
    return df
