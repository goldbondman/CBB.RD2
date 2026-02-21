#!/usr/bin/env python3
"""
cbb_player_matchup.py
=====================
Player-level matchup analysis for the CBB prediction pipeline.

Sections:
  A — Player performance condition profiles
  B — Positional archetype matchup model
  C — Per-game Player Context Scores (PCS)
  D — Team-level matchup summary

Outputs (written to data/csv/ and data/):
  player_condition_profiles.csv
  player_archetype_matchup_matrix.csv
  player_context_scores.csv
  team_matchup_summary.csv

CLI:
  python cbb_player_matchup.py [--build-profiles-only] [--games PATH]
                                [--min-games N] [--min-minutes N]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import traceback
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from espn_config import (
    CSV_DIR,
    OUT_PLAYER_METRICS,
    OUT_RANKINGS,
    OUT_TEAM_LOGS,
    OUT_GAMES,
    OUT_PLAYER_PROXY,
    OUT_PREDICTIONS_COMBINED,
)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA     = pathlib.Path("data")
CSV_OUT  = DATA / "csv"
CSV_OUT.mkdir(parents=True, exist_ok=True)

NOW_ISO = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

# ── Constants ──────────────────────────────────────────────────────────────
DEFENSE_TIERS = {
    "elite":   (0,   95),
    "good":    (95,  100),
    "average": (100, 106),
    "weak":    (106, 999),
}

OFFENSIVE_ARCHETYPES = ["perimeter", "interior", "grind", "balanced"]

MIN_GAMES_PROFILE = 8
MIN_DEFENSE_TIER  = 3
MIN_PACE_SPLIT    = 4
MIN_LOC_SPLIT     = 4
MIN_REST_SPLIT    = 3
PACE_THRESHOLD    = 69
SUFFOCATION_HIGH  = 65
SUFFOCATION_LOW   = 50
SHORT_REST_DAYS   = 1.5
SHRINKAGE_N       = 10

PCS_WEIGHTS = {
    "defense":     0.35,
    "pace":        0.20,
    "home_away":   0.15,
    "rest":        0.15,
    "suffocation": 0.15,
}


# ── Helpers ────────────────────────────────────────────────────────────────

def _load(path: str | pathlib.Path, label: str = "") -> Optional[pd.DataFrame]:
    """Load a CSV; return None with a warning if missing or empty."""
    p = pathlib.Path(path)
    if not p.exists() or p.stat().st_size < 10:
        print(f"[WARN] {label or p.name}: source file missing or empty — {p}")
        return None
    try:
        df = pd.read_csv(p, low_memory=False)
        if df.empty:
            print(f"[WARN] {label or p.name}: loaded but empty")
            return None
        return df
    except Exception as exc:
        print(f"[WARN] {label or p.name}: failed to load — {exc}")
        return None


def _write(df: pd.DataFrame, stem: str) -> None:
    """Write df to data/csv/<stem> and data/<stem>; print status."""
    csv_path  = CSV_OUT / stem
    root_path = DATA / stem
    try:
        df["generated_at"] = NOW_ISO
        df.to_csv(csv_path, index=False)
        df.to_csv(root_path, index=False)
        print(f"[OK]   {stem}: {len(df)} rows")
    except Exception as exc:
        print(f"[WARN] {stem}: write failed — {exc}")
        traceback.print_exc()


def _to_num(df: pd.DataFrame, col: str) -> pd.Series:
    """Coerce column to numeric, returning NaN for failures."""
    return pd.to_numeric(df[col], errors="coerce")


def normalize(delta: float, scale: float) -> float:
    """Soft-normalize to roughly [-1, 1] using tanh."""
    if delta is None or np.isnan(delta):
        return 0.0
    return float(np.tanh(delta / scale))


def _defense_tier(cage_d: float) -> Optional[str]:
    """Return defense tier label for a cage_d value."""
    if pd.isna(cage_d):
        return None
    for tier, (lo, hi) in DEFENSE_TIERS.items():
        if lo <= cage_d < hi:
            return tier
    return None


def classify_player_role(row: pd.Series) -> str:
    """Classify a player into a positional archetype role."""
    usage  = row.get("usage_rate_season_avg", 0) or 0
    atr    = row.get("ast_tov_ratio_season_avg", 0) or 0
    tpct   = row.get("three_pct", 0) or 0
    pos    = str(row.get("position", "")).upper()
    orb    = row.get("orb", 0) or 0
    starter = row.get("starter", False)
    mins   = row.get("min_season_avg", 0) or 0

    if usage > 25 and atr > 1.5:
        return "PRIMARY_HANDLER"
    if 18 <= usage <= 28 and tpct > 0.33 and pos in ("G", "F"):
        return "WING_SCORER"
    if usage > 18 and orb >= 2 and pos in ("F", "C") and tpct < 0.25:
        return "POST_SCORER"
    if usage < 18 and tpct > 0.35 and mins > 20:
        return "FLOOR_SPACER"
    if 15 <= usage <= 22 and 1.2 <= atr <= 2.0:
        return "CONNECTOR"
    if not starter and mins < 22:
        return "BENCH_ENERGY"
    return "CONNECTOR"


# ── Data Loading ───────────────────────────────────────────────────────────

def load_pipeline_data() -> dict:
    """Load all required pipeline CSVs. Returns dict of DataFrames."""
    data = {}

    pm = _load(OUT_PLAYER_METRICS, "player_game_metrics")
    if pm is not None:
        pm = pm.loc[:, ~pm.columns.duplicated()]
    data["player_metrics"] = pm

    data["rankings"] = _load(OUT_RANKINGS, "cbb_rankings")
    data["team_logs"] = _load(OUT_TEAM_LOGS, "team_game_logs")
    data["games"] = _load(OUT_GAMES, "games")
    data["injury_proxy"] = _load(OUT_PLAYER_PROXY, "player_injury_proxy")

    return data


def _enrich_with_opponent(pm: pd.DataFrame, team_logs: pd.DataFrame,
                          rankings: pd.DataFrame) -> pd.DataFrame:
    """Join player metrics with opponent team info and rankings."""
    tl = team_logs[["event_id", "team_id", "team"]].copy()
    tl.rename(columns={"team_id": "opp_team_id", "team": "opp_team"}, inplace=True)

    # For each player-game row, find the opponent by matching event_id
    # where opponent team_id differs from the player's team_id
    pm = pm.copy()
    for col in ["event_id", "team_id"]:
        pm[col] = pd.to_numeric(pm[col], errors="coerce")
    tl["event_id"] = pd.to_numeric(tl["event_id"], errors="coerce")
    tl["opp_team_id"] = pd.to_numeric(tl["opp_team_id"], errors="coerce")

    merged = pm.merge(tl, on="event_id", how="left")
    merged = merged[merged["team_id"] != merged["opp_team_id"]].copy()

    # Drop duplicates that may arise from multiple opponent rows
    merged = merged.drop_duplicates(
        subset=["event_id", "athlete_id", "team_id"], keep="first"
    )

    # Join opponent rankings
    rk = rankings.copy()
    rk["team_id"] = pd.to_numeric(rk["team_id"], errors="coerce")
    rk_cols = ["team_id", "cage_d", "cage_t", "suffocation", "opp_efg_pct",
               "offensive_archetype", "tov_pct", "orb_pct", "drb_pct",
               "ftr", "consistency_score"]
    rk_cols = [c for c in rk_cols if c in rk.columns]
    rk = rk[rk_cols].rename(columns=lambda c: f"opp_{c}" if c != "team_id" else "opp_team_id")
    for col in ["opp_cage_d", "opp_cage_t", "opp_suffocation"]:
        if col in rk.columns:
            rk[col] = pd.to_numeric(rk[col], errors="coerce")

    merged = merged.merge(rk, on="opp_team_id", how="left")
    return merged


# ── Section A — Player Condition Profiles ──────────────────────────────────

def build_condition_profiles(pm: pd.DataFrame, team_logs: pd.DataFrame,
                             rankings: pd.DataFrame,
                             games: pd.DataFrame) -> pd.DataFrame:
    """Build per-player condition profiles across opponent tiers."""
    enriched = _enrich_with_opponent(pm, team_logs, rankings)

    # Coerce numeric columns
    for col in ["pts", "efg_pct", "usage_rate", "min",
                "pts_season_avg", "efg_pct_season_avg",
                "usage_rate_season_avg", "min_season_avg",
                "games_played", "three_pct", "orb",
                "ast_tov_ratio_season_avg"]:
        if col in enriched.columns:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce")

    if "starter" in enriched.columns:
        enriched["starter"] = enriched["starter"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

    # Filter to players with sufficient history
    enriched = enriched[enriched["games_played"] >= MIN_GAMES_PROFILE].copy()
    print(f"[MATCHUP] Building profiles for {len(enriched):,} player-game records...")

    players = enriched.groupby("athlete_id").first()[
        ["player", "team_id", "team", "position",
         "pts_season_avg", "efg_pct_season_avg",
         "usage_rate_season_avg", "min_season_avg", "games_played"]
    ].reset_index()
    print(f"[MATCHUP] {len(players):,} players with sufficient history (>= {MIN_GAMES_PROFILE} games)")

    # Assign defense tier per game
    if "opp_cage_d" in enriched.columns:
        enriched["opp_def_tier"] = enriched["opp_cage_d"].apply(_defense_tier)

    # Home/away from games.csv
    if games is not None:
        gm = games[["event_id", "home_team_id", "away_team_id"]].copy()
        for c in ["event_id", "home_team_id", "away_team_id"]:
            gm[c] = pd.to_numeric(gm[c], errors="coerce")
        enriched = enriched.merge(gm, on="event_id", how="left")
        enriched["location"] = np.where(
            enriched["team_id"] == enriched["home_team_id"], "home",
            np.where(enriched["team_id"] == enriched["away_team_id"], "away", None)
        )

    # Short rest: gap between consecutive team games <= 1.5 days
    enriched = _compute_short_rest(enriched)

    profiles = []
    for aid, grp in enriched.groupby("athlete_id"):
        row = grp.iloc[0]
        profile = {
            "athlete_id": aid,
            "player": row.get("player"),
            "team_id": row.get("team_id"),
            "team": row.get("team"),
            "position": row.get("position"),
            "games_played": row.get("games_played"),
            "pts_season_avg": row.get("pts_season_avg"),
            "efg_pct_season_avg": row.get("efg_pct_season_avg"),
            "usage_rate_season_avg": row.get("usage_rate_season_avg"),
        }

        # Defense tier splits
        for tier in DEFENSE_TIERS:
            tier_games = grp[grp["opp_def_tier"] == tier] if "opp_def_tier" in grp.columns else pd.DataFrame()
            n = len(tier_games)
            profile[f"n_vs_{tier}"] = n
            if n >= MIN_DEFENSE_TIER:
                profile[f"pts_delta_vs_{tier}"] = tier_games["pts"].mean() - row["pts_season_avg"]
                profile[f"efg_delta_vs_{tier}"] = tier_games["efg_pct"].mean() - row["efg_pct_season_avg"]
                profile[f"usage_delta_vs_{tier}"] = tier_games["usage_rate"].mean() - row["usage_rate_season_avg"]
            else:
                profile[f"pts_delta_vs_{tier}"] = None
                profile[f"efg_delta_vs_{tier}"] = None
                profile[f"usage_delta_vs_{tier}"] = None

        # Pace splits
        if "opp_cage_t" in grp.columns:
            fast = grp[grp["opp_cage_t"] > PACE_THRESHOLD]
            slow = grp[grp["opp_cage_t"] <= PACE_THRESHOLD]
            profile["n_fast_pace"] = len(fast)
            profile["n_slow_pace"] = len(slow)
            if len(fast) >= MIN_PACE_SPLIT:
                profile["pts_delta_fast_pace"] = fast["pts"].mean() - row["pts_season_avg"]
                profile["efg_delta_fast_pace"] = fast["efg_pct"].mean() - row["efg_pct_season_avg"]
            else:
                profile["pts_delta_fast_pace"] = None
                profile["efg_delta_fast_pace"] = None
            if len(slow) >= MIN_PACE_SPLIT:
                profile["pts_delta_slow_pace"] = slow["pts"].mean() - row["pts_season_avg"]
                profile["efg_delta_slow_pace"] = slow["efg_pct"].mean() - row["efg_pct_season_avg"]
            else:
                profile["pts_delta_slow_pace"] = None
                profile["efg_delta_slow_pace"] = None
        else:
            for k in ["pts_delta_fast_pace", "efg_delta_fast_pace",
                       "pts_delta_slow_pace", "efg_delta_slow_pace",
                       "n_fast_pace", "n_slow_pace"]:
                profile[k] = None

        # Home/away splits
        if "location" in grp.columns:
            for loc in ["home", "away"]:
                loc_games = grp[grp["location"] == loc]
                profile[f"n_{loc}"] = len(loc_games)
                if len(loc_games) >= MIN_LOC_SPLIT:
                    profile[f"pts_delta_{loc}"] = loc_games["pts"].mean() - row["pts_season_avg"]
                    profile[f"efg_delta_{loc}"] = loc_games["efg_pct"].mean() - row["efg_pct_season_avg"]
                else:
                    profile[f"pts_delta_{loc}"] = None
                    profile[f"efg_delta_{loc}"] = None
        else:
            for loc in ["home", "away"]:
                profile[f"n_{loc}"] = None
                profile[f"pts_delta_{loc}"] = None
                profile[f"efg_delta_{loc}"] = None

        # Short rest splits
        rest_games = grp[grp["short_rest"] == True] if "short_rest" in grp.columns else pd.DataFrame()
        profile["n_short_rest"] = len(rest_games)
        if len(rest_games) >= MIN_REST_SPLIT:
            profile["pts_delta_short_rest"] = rest_games["pts"].mean() - row["pts_season_avg"]
            profile["efg_delta_short_rest"] = rest_games["efg_pct"].mean() - row["efg_pct_season_avg"]
            profile["min_delta_short_rest"] = rest_games["min"].mean() - row["min_season_avg"]
        else:
            profile["pts_delta_short_rest"] = None
            profile["efg_delta_short_rest"] = None
            profile["min_delta_short_rest"] = None

        # Suffocation splits
        if "opp_suffocation" in grp.columns:
            high_suf = grp[grp["opp_suffocation"] > SUFFOCATION_HIGH]
            low_suf = grp[grp["opp_suffocation"] < SUFFOCATION_LOW]
            profile["n_vs_high_suffocation"] = len(high_suf)
            profile["n_vs_low_suffocation"] = len(low_suf)
            if len(high_suf) >= MIN_DEFENSE_TIER and len(low_suf) >= MIN_DEFENSE_TIER:
                profile["pts_delta_vs_suffocation"] = (
                    high_suf["pts"].mean() - low_suf["pts"].mean()
                )
                profile["usage_delta_vs_suffocation"] = (
                    high_suf["usage_rate"].mean() - low_suf["usage_rate"].mean()
                )
            else:
                profile["pts_delta_vs_suffocation"] = None
                profile["usage_delta_vs_suffocation"] = None
        else:
            profile["pts_delta_vs_suffocation"] = None
            profile["usage_delta_vs_suffocation"] = None
            profile["n_vs_high_suffocation"] = None
            profile["n_vs_low_suffocation"] = None

        # Variance
        elite_games = grp[grp["opp_def_tier"] == "elite"] if "opp_def_tier" in grp.columns else pd.DataFrame()
        profile["pts_std_vs_elite"] = elite_games["pts"].std() if len(elite_games) >= MIN_DEFENSE_TIER else None
        profile["pts_std_overall"] = grp["pts"].std() if len(grp) >= 2 else None

        profiles.append(profile)

    result = pd.DataFrame(profiles)
    n_with_splits = result.dropna(subset=["pts_delta_vs_elite"], how="all").shape[0] if "pts_delta_vs_elite" in result.columns else 0
    print(f"[MATCHUP] {n_with_splits} players with defense-tier splits computed")
    return result


def _compute_short_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Flag games where a team played within 1.5 days of its previous game."""
    df = df.copy()
    df["short_rest"] = False

    if "game_datetime_utc" not in df.columns:
        return df

    df["_gdt"] = pd.to_datetime(df["game_datetime_utc"], errors="coerce")
    for _, tgrp in df.groupby("team_id"):
        sorted_dates = tgrp[["event_id", "_gdt"]].drop_duplicates("event_id").sort_values("_gdt")
        if len(sorted_dates) < 2:
            continue
        sorted_dates["_gap"] = sorted_dates["_gdt"].diff().dt.total_seconds() / 86400
        short_events = sorted_dates.loc[sorted_dates["_gap"] <= SHORT_REST_DAYS, "event_id"]
        df.loc[
            (df["team_id"].isin(tgrp["team_id"].unique())) &
            (df["event_id"].isin(short_events)),
            "short_rest",
        ] = True

    df.drop(columns=["_gdt"], inplace=True, errors="ignore")
    return df


# ── Section B — Positional Archetype Matchup Model ────────────────────────

def build_archetype_matrix(enriched_profiles: pd.DataFrame,
                           pm: pd.DataFrame, team_logs: pd.DataFrame,
                           rankings: pd.DataFrame) -> pd.DataFrame:
    """Build role × offensive_archetype × opponent_archetype matrix."""
    enriched = _enrich_with_opponent(pm, team_logs, rankings)

    for col in ["pts", "efg_pct", "usage_rate", "pts_season_avg",
                "efg_pct_season_avg", "usage_rate_season_avg",
                "min_season_avg", "games_played", "three_pct",
                "orb", "ast_tov_ratio_season_avg"]:
        if col in enriched.columns:
            enriched[col] = pd.to_numeric(enriched[col], errors="coerce")
    if "starter" in enriched.columns:
        enriched["starter"] = enriched["starter"].astype(str).str.lower().isin(
            ["true", "1", "yes"]
        )

    enriched = enriched[enriched["games_played"] >= MIN_GAMES_PROFILE].copy()

    # Classify player roles
    enriched["player_role"] = enriched.apply(classify_player_role, axis=1)

    # Compute deltas
    enriched["pts_delta"] = enriched["pts"] - enriched["pts_season_avg"]
    enriched["efg_delta"] = enriched["efg_pct"] - enriched["efg_pct_season_avg"]
    enriched["usage_delta"] = enriched["usage_rate"] - enriched["usage_rate_season_avg"]

    # Get team offensive archetype
    rk = rankings.copy()
    rk["team_id"] = pd.to_numeric(rk["team_id"], errors="coerce")
    if "offensive_archetype" in rk.columns:
        arch_map = rk.set_index("team_id")["offensive_archetype"].to_dict()
        enriched["offensive_archetype"] = enriched["team_id"].map(arch_map)
    else:
        enriched["offensive_archetype"] = None

    # Opponent archetype
    if "opp_offensive_archetype" in enriched.columns:
        enriched.rename(columns={"opp_offensive_archetype": "opponent_archetype"}, inplace=True)
    else:
        enriched["opponent_archetype"] = None

    # Suffocation-based delta
    if "opp_suffocation" in enriched.columns:
        enriched["opp_suffocation"] = pd.to_numeric(enriched["opp_suffocation"], errors="coerce")
        enriched["pts_delta_vs_suffocation"] = np.where(
            enriched["opp_suffocation"] > SUFFOCATION_HIGH,
            enriched["pts_delta"],
            np.nan,
        )
    else:
        enriched["pts_delta_vs_suffocation"] = np.nan

    # Group by role × offensive_archetype × opponent_archetype
    group_cols = ["offensive_archetype", "opponent_archetype", "player_role"]
    enriched_valid = enriched.dropna(subset=group_cols)

    rows = []
    for keys, grp in enriched_valid.groupby(group_cols):
        off_arch, opp_arch, role = keys
        n = len(grp)
        avg_pts = grp["pts_delta"].mean()
        avg_efg = grp["efg_delta"].mean()
        avg_usage = grp["usage_delta"].mean()
        avg_suf = grp["pts_delta_vs_suffocation"].mean()

        # Bayesian shrinkage
        shrink = n / (n + SHRINKAGE_N)
        avg_pts_shrunk = avg_pts * shrink
        avg_efg_shrunk = avg_efg * shrink
        avg_usage_shrunk = avg_usage * shrink
        avg_suf_shrunk = (avg_suf * shrink) if not np.isnan(avg_suf) else None

        if n < 15:
            confidence = "LOW"
        elif n <= 30:
            confidence = "MEDIUM"
        else:
            confidence = "HIGH"

        rows.append({
            "offensive_archetype": off_arch,
            "opponent_archetype": opp_arch,
            "player_role": role,
            "avg_pts_delta": round(avg_pts_shrunk, 3) if not np.isnan(avg_pts_shrunk) else None,
            "avg_efg_delta": round(avg_efg_shrunk, 4) if not np.isnan(avg_efg_shrunk) else None,
            "avg_usage_delta": round(avg_usage_shrunk, 3) if not np.isnan(avg_usage_shrunk) else None,
            "avg_pts_delta_vs_suffocation": round(avg_suf_shrunk, 3) if avg_suf_shrunk is not None else None,
            "n_player_games": n,
            "confidence": confidence,
        })

    matrix = pd.DataFrame(rows)
    populated = len(matrix)
    possible = len(OFFENSIVE_ARCHETYPES) ** 2 * 6  # 4 archetypes × 4 opp × 6 roles
    print(f"[MATCHUP] Archetype matrix: {populated} populated cells (of {possible} possible)")
    return matrix


# ── Section C — Per-game Player Context Scores ─────────────────────────────

def compute_player_context_scores(
    profiles: pd.DataFrame,
    rankings: pd.DataFrame,
    games_df: pd.DataFrame,
    predictions: pd.DataFrame,
    injury_proxy: Optional[pd.DataFrame],
    min_games: int = 5,
    min_minutes: int = 15,
) -> pd.DataFrame:
    """Compute PCS for every player in upcoming games."""
    # Identify upcoming games from predictions
    if "event_id" not in predictions.columns:
        print("[WARN] predictions missing event_id — cannot compute PCS")
        return pd.DataFrame()

    upcoming_events = predictions["event_id"].unique()

    # Determine home/away team for each upcoming event
    event_teams = []
    if games_df is not None and "event_id" in games_df.columns:
        gm = games_df[["event_id", "home_team_id", "away_team_id"]].copy()
        for c in gm.columns:
            gm[c] = pd.to_numeric(gm[c], errors="coerce")
        gm = gm[gm["event_id"].isin(pd.to_numeric(pd.Series(upcoming_events), errors="coerce"))]
        for _, r in gm.iterrows():
            event_teams.append({"event_id": int(r["event_id"]),
                                "team_id": int(r["home_team_id"]),
                                "location": "home"})
            event_teams.append({"event_id": int(r["event_id"]),
                                "team_id": int(r["away_team_id"]),
                                "location": "away"})
    event_teams = pd.DataFrame(event_teams) if event_teams else pd.DataFrame(
        columns=["event_id", "team_id", "location"]
    )

    # Build opponent map: event_id + team_id -> opp_team_id
    opp_map = {}
    for _, r in event_teams.iterrows():
        eid = r["event_id"]
        tid = r["team_id"]
        partner = event_teams[(event_teams["event_id"] == eid) & (event_teams["team_id"] != tid)]
        if len(partner) > 0:
            opp_map[(eid, tid)] = int(partner.iloc[0]["team_id"])

    # Rankings lookup
    rk = rankings.copy()
    for col in ["team_id", "cage_d", "cage_t", "suffocation"]:
        if col in rk.columns:
            rk[col] = pd.to_numeric(rk[col], errors="coerce")
    rk_lookup = rk.set_index("team_id")

    # Injury set
    injured_ids = set()
    if injury_proxy is not None and "athlete_id" in injury_proxy.columns:
        injured_ids = set(pd.to_numeric(injury_proxy["athlete_id"], errors="coerce").dropna().astype(int))

    # Filter eligible players
    prof = profiles.copy()
    for col in ["games_played", "min_season_avg", "usage_rate_season_avg", "athlete_id", "team_id"]:
        if col in prof.columns:
            prof[col] = pd.to_numeric(prof[col], errors="coerce")

    prof = prof[
        (prof["games_played"] >= min_games) &
        (prof["min_season_avg"] >= min_minutes)
    ].copy()

    # Compute days_rest per team for upcoming games
    rest_days = _compute_upcoming_rest(games_df, upcoming_events)

    scores = []
    insufficient_count = 0

    for _, prow in prof.iterrows():
        aid = prow["athlete_id"]
        tid = prow["team_id"]
        if pd.isna(tid):
            continue
        tid = int(tid)

        # Find upcoming events for this team
        team_events = event_teams[event_teams["team_id"] == tid]
        if team_events.empty:
            continue

        for _, erow in team_events.iterrows():
            eid = int(erow["event_id"])
            loc = erow["location"]
            opp_tid = opp_map.get((eid, tid))
            if opp_tid is None:
                continue

            # Get opponent rankings
            if opp_tid not in rk_lookup.index:
                continue
            opp = rk_lookup.loc[opp_tid]
            opp_cage_d = opp.get("cage_d", np.nan) if isinstance(opp, pd.Series) else np.nan
            opp_cage_t = opp.get("cage_t", np.nan) if isinstance(opp, pd.Series) else np.nan
            opp_suf = opp.get("suffocation", np.nan) if isinstance(opp, pd.Series) else np.nan

            opp_tier = _defense_tier(opp_cage_d)
            pace_label = "fast" if (not pd.isna(opp_cage_t) and opp_cage_t > PACE_THRESHOLD) else "slow"

            # Gather deltas
            score = 0.0
            components = 0
            raw_pts_delta = 0.0
            raw_efg_delta = 0.0
            raw_usage_delta = 0.0

            # 1. Defense tier (35%)
            pts_d_tier = prow.get(f"pts_delta_vs_{opp_tier}") if opp_tier else None
            efg_d_tier = prow.get(f"efg_delta_vs_{opp_tier}") if opp_tier else None
            usage_d_tier = prow.get(f"usage_delta_vs_{opp_tier}") if opp_tier else None
            if pts_d_tier is not None and not pd.isna(pts_d_tier):
                score += PCS_WEIGHTS["defense"] * normalize(pts_d_tier, 10)
                raw_pts_delta += pts_d_tier
                components += 1
            if efg_d_tier is not None and not pd.isna(efg_d_tier):
                raw_efg_delta += efg_d_tier
            if usage_d_tier is not None and not pd.isna(usage_d_tier):
                raw_usage_delta += usage_d_tier

            # 2. Pace (20%)
            pts_d_pace = prow.get(f"pts_delta_{pace_label}_pace")
            efg_d_pace = prow.get(f"efg_delta_{pace_label}_pace")
            if pts_d_pace is not None and not pd.isna(pts_d_pace):
                score += PCS_WEIGHTS["pace"] * normalize(pts_d_pace, 8)
                raw_pts_delta += pts_d_pace
                components += 1
            if efg_d_pace is not None and not pd.isna(efg_d_pace):
                raw_efg_delta += efg_d_pace

            # 3. Home/away (15%)
            pts_d_loc = prow.get(f"pts_delta_{loc}")
            efg_d_loc = prow.get(f"efg_delta_{loc}")
            if pts_d_loc is not None and not pd.isna(pts_d_loc):
                score += PCS_WEIGHTS["home_away"] * normalize(pts_d_loc, 6)
                raw_pts_delta += pts_d_loc
                components += 1
            if efg_d_loc is not None and not pd.isna(efg_d_loc):
                raw_efg_delta += efg_d_loc

            # 4. Rest (15%, only if short rest)
            dr = rest_days.get(tid, 99)
            pts_d_rest = prow.get("pts_delta_short_rest")
            if dr <= SHORT_REST_DAYS and pts_d_rest is not None and not pd.isna(pts_d_rest):
                score += PCS_WEIGHTS["rest"] * normalize(pts_d_rest, 6)
                raw_pts_delta += pts_d_rest
                components += 1

            # 5. Suffocation (15%, only if opponent suffocation > 65)
            pts_d_suf = prow.get("pts_delta_vs_suffocation")
            if not pd.isna(opp_suf) and opp_suf > SUFFOCATION_HIGH:
                if pts_d_suf is not None and not pd.isna(pts_d_suf):
                    score += PCS_WEIGHTS["suffocation"] * normalize(pts_d_suf, 8)
                    raw_pts_delta += pts_d_suf
                    components += 1

            # Scale to -100 to +100
            pcs = max(-100, min(100, score * 100))

            # PCS tier
            if pcs > 25:
                tier = "STRONG_OVER"
            elif pcs > 10:
                tier = "LEAN_OVER"
            elif pcs >= -10:
                tier = "NEUTRAL"
            elif pcs >= -25:
                tier = "LEAN_UNDER"
            else:
                tier = "STRONG_UNDER"

            # Confidence
            if components >= 4:
                data_confidence = "HIGH"
            elif components >= 2:
                data_confidence = "MEDIUM"
            else:
                data_confidence = "LOW"
                insufficient_count += 1

            is_star = (prow.get("usage_rate_season_avg", 0) or 0) > 25
            is_injured = int(aid) in injured_ids if not pd.isna(aid) else False

            scores.append({
                "event_id": eid,
                "athlete_id": int(aid) if not pd.isna(aid) else None,
                "player": prow.get("player"),
                "team_id": tid,
                "team": prow.get("team"),
                "position": prow.get("position"),
                "location": loc,
                "opp_team_id": opp_tid,
                "opp_def_tier": opp_tier,
                "opp_cage_d": opp_cage_d,
                "opp_cage_t": opp_cage_t,
                "opp_suffocation": opp_suf,
                "pcs": round(pcs, 1),
                "pcs_tier": tier,
                "expected_pts_delta": round(raw_pts_delta, 2),
                "expected_efg_delta": round(raw_efg_delta, 4),
                "expected_usage_delta": round(raw_usage_delta, 3),
                "n_components": components,
                "data_confidence": data_confidence,
                "star_player": is_star,
                "injury_flagged": is_injured,
                "pts_season_avg": prow.get("pts_season_avg"),
                "usage_rate_season_avg": prow.get("usage_rate_season_avg"),
                "games_played": prow.get("games_played"),
            })

    result = pd.DataFrame(scores)
    if not result.empty:
        result = result.sort_values("pcs", key=lambda s: s.abs(), ascending=False)

    if insufficient_count > 0:
        print(f"[WARN] {insufficient_count} players had insufficient data — using neutral PCS")

    return result


def _compute_upcoming_rest(games_df: Optional[pd.DataFrame],
                           upcoming_events) -> dict:
    """Return dict of team_id -> days since last game for upcoming events."""
    rest = {}
    if games_df is None or "game_datetime_utc" not in games_df.columns:
        return rest

    gm = games_df.copy()
    gm["game_datetime_utc"] = pd.to_datetime(gm["game_datetime_utc"], errors="coerce")
    for col in ["event_id", "home_team_id", "away_team_id"]:
        if col in gm.columns:
            gm[col] = pd.to_numeric(gm[col], errors="coerce")

    upcoming_set = set(pd.to_numeric(pd.Series(list(upcoming_events)), errors="coerce").dropna())
    upcoming_gm = gm[gm["event_id"].isin(upcoming_set)]
    past_gm = gm[~gm["event_id"].isin(upcoming_set)]

    for _, row in upcoming_gm.iterrows():
        for tid_col in ["home_team_id", "away_team_id"]:
            tid = row.get(tid_col)
            if pd.isna(tid):
                continue
            tid = int(tid)
            if tid in rest:
                continue
            # Find most recent past game for this team
            team_past = past_gm[
                (past_gm["home_team_id"] == tid) | (past_gm["away_team_id"] == tid)
            ]
            if team_past.empty:
                rest[tid] = 99
                continue
            latest = team_past["game_datetime_utc"].max()
            gap = (row["game_datetime_utc"] - latest).total_seconds() / 86400
            rest[tid] = gap

    return rest


# ── Section D — Team-level Matchup Summary ─────────────────────────────────

def build_team_matchup_summary(pcs_df: pd.DataFrame,
                               pm: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PCS to team level per event."""
    if pcs_df.empty:
        return pd.DataFrame()

    # Get starter info from player metrics — use latest game per player
    starter_map = {}
    if pm is not None and "athlete_id" in pm.columns and "starter" in pm.columns:
        pm_copy = pm.loc[:, ~pm.columns.duplicated()].copy()
        pm_copy["starter"] = pm_copy["starter"].astype(str).str.lower().isin(["true", "1", "yes"])
        pm_copy["min_season_avg"] = pd.to_numeric(pm_copy["min_season_avg"], errors="coerce")
        pm_copy["athlete_id"] = pd.to_numeric(pm_copy["athlete_id"], errors="coerce")
        # Latest game per player
        if "game_datetime_utc" in pm_copy.columns:
            pm_copy = pm_copy.sort_values("game_datetime_utc")
        latest = pm_copy.groupby("athlete_id").last()
        starter_map = latest["starter"].to_dict()

    pcs = pcs_df.copy()
    pcs["is_starter"] = pcs["athlete_id"].map(starter_map).fillna(False)

    rows = []
    for (eid, tid), grp in pcs.groupby(["event_id", "team_id"]):
        min_avg = grp["min_season_avg"].fillna(0) if "min_season_avg" in grp.columns else pd.Series(0, index=grp.index)
        starters = grp[(grp["is_starter"] == True) & (min_avg >= 20)]
        if starters.empty:
            starters = grp[grp["is_starter"] == True]

        usage_col = grp["usage_rate_season_avg"].dropna()
        star_row = grp.loc[usage_col.idxmax()] if not usage_col.empty else None
        star_pcs = star_row["pcs"] if star_row is not None else None
        is_star = star_row["star_player"] if star_row is not None else False

        avg_pcs_starters = starters["pcs"].mean() if not starters.empty else None
        avg_pcs_all = grp["pcs"].mean()

        n_strong_over = int((grp["pcs_tier"] == "STRONG_OVER").sum())
        n_strong_under = int((grp["pcs_tier"] == "STRONG_UNDER").sum())

        exp_pts_impact = starters["expected_pts_delta"].sum() if not starters.empty else 0.0

        bench = grp[grp["is_starter"] == False]
        roster_depth = int((bench["pcs"] >= -10).sum()) if not bench.empty else 0

        star_underperform = bool(star_pcs is not None and star_pcs < -20 and is_star)
        favorable = bool(avg_pcs_starters is not None and avg_pcs_starters > 15)
        unfavorable = bool(avg_pcs_starters is not None and avg_pcs_starters < -15)

        rows.append({
            "event_id": eid,
            "team_id": tid,
            "team": grp.iloc[0].get("team"),
            "location": grp.iloc[0].get("location"),
            "star_pcs": round(star_pcs, 1) if star_pcs is not None else None,
            "avg_pcs_starters": round(avg_pcs_starters, 1) if avg_pcs_starters is not None else None,
            "avg_pcs_all": round(avg_pcs_all, 1) if avg_pcs_all is not None else None,
            "n_players_strong_over": n_strong_over,
            "n_players_strong_under": n_strong_under,
            "expected_pts_impact": round(exp_pts_impact, 2),
            "roster_depth_score": roster_depth,
            "star_underperform_risk": star_underperform,
            "favorable_conditions": favorable,
            "unfavorable_conditions": unfavorable,
            "n_players_scored": len(grp),
        })

    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Player matchup analysis for CBB prediction pipeline"
    )
    parser.add_argument(
        "--build-profiles-only", action="store_true",
        help="Only build condition profiles and archetype matrix (Sections A & B)",
    )
    parser.add_argument(
        "--games", type=str, default=str(OUT_PREDICTIONS_COMBINED),
        help="Path to predictions CSV for upcoming games",
    )
    parser.add_argument(
        "--min-games", type=int, default=5,
        help="Minimum games for PCS eligibility",
    )
    parser.add_argument(
        "--min-minutes", type=int, default=15,
        help="Minimum season-average minutes for PCS eligibility",
    )
    args = parser.parse_args()

    try:
        data = load_pipeline_data()
        pm = data["player_metrics"]
        rankings = data["rankings"]
        team_logs = data["team_logs"]
        games = data["games"]
        injury_proxy = data["injury_proxy"]

        if pm is None or rankings is None or team_logs is None:
            print("[WARN] Required source data missing — cannot run matchup analysis")
            sys.exit(0)

        # ── Section A ──────────────────────────────────────────────────
        profiles = build_condition_profiles(pm, team_logs, rankings, games)
        _write(profiles, "player_condition_profiles.csv")

        # ── Section B ──────────────────────────────────────────────────
        matrix = build_archetype_matrix(profiles, pm, team_logs, rankings)
        _write(matrix, "player_archetype_matchup_matrix.csv")

        if args.build_profiles_only:
            print("[MATCHUP] --build-profiles-only: skipping PCS and team summary")
            return

        # ── Section C ──────────────────────────────────────────────────
        predictions = _load(args.games, "predictions")
        if predictions is None:
            print("[WARN] No predictions file — skipping PCS")
            return

        pcs_df = compute_player_context_scores(
            profiles, rankings, games, predictions, injury_proxy,
            min_games=args.min_games,
            min_minutes=args.min_minutes,
        )

        n_games = pcs_df["event_id"].nunique() if not pcs_df.empty else 0
        n_scores = len(pcs_df)
        print(f"[MATCHUP] Today's slate: {n_games} games, {n_scores} player context scores generated")

        _write(pcs_df, "player_context_scores.csv")

        # ── Section D ──────────────────────────────────────────────────
        team_summary = build_team_matchup_summary(pcs_df, pm)
        _write(team_summary, "team_matchup_summary.csv")

        if not team_summary.empty:
            n_teams = len(team_summary)
            print(f"[OK]   team_matchup_summary.csv: {n_teams} rows ({n_games} games × 2 teams)")

    except Exception as exc:
        print(f"[WARN] Matchup analysis failed — {exc}")
        traceback.print_exc()
        sys.exit(0)


if __name__ == "__main__":
    main()
