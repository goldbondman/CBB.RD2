"""
ESPN CBB Pipeline — Tournament Composite Metrics
Designed for Conference Tournaments & NCAA March Madness R1/R2.
Pure DataFrame transformations. No I/O. No play-by-play required.

Input:  team_game_sos.csv  (output of espn_sos.py — richest available data)
        player_game_metrics.csv  (for star reliance section)

Team-level composite scores (one row per team-game, pre-tournament snapshot):
  tournament_dna_score          Weighted composite: how well does the profile
                                match historical tournament survivor fingerprints
  suffocation_rating            Defensive quality score (lower opp numbers = higher)
  momentum_quality_rating       Is the hot streak real or regression bait?
  star_reliance_risk            Roster fragility if top player struggles/fouls out
  offensive_identity_score      Clarity and execution of offensive system

Matchup-level outputs (one row per game, both teams joined):
  game_total_projection         Projected combined score (pts)
  total_confidence              Confidence band (±pts) based on variance flags
  underdog_winner_score         UWS 0–70 for the lower-seeded / higher-ML team
  uwp_upset_probability         Implied upset probability from UWS
  game_story                    Primary narrative label for the matchup

All columns are prefixed with `t_` to distinguish from raw box-score columns.

Pipeline position:
  espn_pipeline.py calls compute_tournament_metrics() after compute_sos_metrics()
  and compute_matchup_projections() after joining game opponents.
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
LEAGUE_AVG_ORTG       = 103.0
LEAGUE_AVG_DRTG       = 103.0
LEAGUE_AVG_PACE       = 70.0
LEAGUE_AVG_EFG        = 50.5
LEAGUE_AVG_TOV        = 18.0
LEAGUE_AVG_FTR        = 28.0

# Tournament environment multipliers (applied to raw pace×efficiency projection)
TOURN_MULTIPLIER = {
    "conf_tournament": 0.964,
    "ncaa_r1":         0.958,
    "ncaa_r2":         0.951,
}

# Slow-team drag: slower team's pace gets weighted heavier in the game
SLOW_TEAM_WEIGHT = 0.53
FAST_TEAM_WEIGHT = 0.47

# UWS thresholds
UWS_STRONG_ALERT = 55
UWS_LEGITIMATE   = 45
UWS_MILD_THREAT  = 35


# ── Safe column accessor ───────────────────────────────────────────────────────

def _col(df: pd.DataFrame, name: str, fill: float = np.nan) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(fill)
    return pd.Series(fill, index=df.index, dtype=float)


def _pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Percentile rank 0–100, higher = better unless ascending=False."""
    return series.rank(pct=True, ascending=ascending).mul(100).round(1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TEAM-LEVEL COMPOSITE SCORES
# Inputs: single team-game row from team_game_sos.csv
# ═══════════════════════════════════════════════════════════════════════════════

# ── 1A. Tournament DNA Index ──────────────────────────────────────────────────

def add_tournament_dna(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tournament DNA Score (0–100, higher = better tournament profile).

    Components & weights:
      28%  eFG% differential (team eFG% − opp eFG% allowed)
      22%  TOV rate differential (opp TOV% forced − team TOV%)
      15%  FTR differential (team FTR − opp FTR)
      18%  Away/neutral win% (road + neutral resilience)
      10%  Strength of schedule (opp avg net rating)
       7%  Close-game scoring margin (margin in games decided ≤10 pts)

    All components are Z-scored across the full df population then
    rescaled 0–100 so the output is cross-season comparable.
    """
    df = df.copy()

    # ── Component raw values ──

    # eFG% differential: our eFG% minus what we allow (context-adjusted if available)
    efg_off  = _col(df, "efg_pct")
    # Use opponent-context adjusted allow if SOS module ran, else raw drtg proxy
    efg_def  = _col(df, "efg_vs_opp_season")   # positive = shooting above what opp allows
    efg_diff = efg_off + efg_def.fillna(0)      # compound: shoot well AND suppress well

    # TOV differential: force more than you commit
    tov_forced   = _col(df, "tov_vs_opp_season")   # positive = forcing more TOs than opp avg
    tov_team     = _col(df, "tov_pct")
    tov_diff     = tov_forced.fillna(0) - tov_team  # higher = better (force more, commit fewer)

    # FTR differential: get to line more than opponent does
    ftr_team = _col(df, "ftr")
    ftr_diff = _col(df, "ftr_vs_opp_season").fillna(0) + (ftr_team - LEAGUE_AVG_FTR)

    # Away/neutral win% — best tournament environment proxy available
    away_wins   = _col(df, "away_wins",   0)
    away_losses = _col(df, "away_losses", 0)
    away_games  = (away_wins + away_losses).replace(0, np.nan)
    away_win_pct = (away_wins / away_games * 100)

    # SOS: use opponent average net rating (season). Higher opp net = harder schedule.
    sos_raw = _col(df, "opp_avg_net_rtg_season")

    # Close-game margin: average scoring margin in games decided ≤10 pts
    # Approximated from margin_capped rolling mean (already capped at ±15)
    # L10 capped margin as close-game proxy — teams with good close-game results
    # will have margin_capped_l10 meaningfully above zero despite the cap
    close_margin = _col(df, "margin_capped_l10")

    # ── Z-score each component within this dataset population ──
    def _zscore(s: pd.Series) -> pd.Series:
        mu, sigma = s.mean(), s.std()
        if pd.isna(sigma) or sigma == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma

    z_efg   = _zscore(efg_diff)
    z_tov   = _zscore(tov_diff)
    z_ftr   = _zscore(ftr_diff)
    z_away  = _zscore(away_win_pct)
    z_sos   = _zscore(sos_raw)
    z_close = _zscore(close_margin)

    # ── Weighted composite ──
    raw_score = (
        z_efg   * 0.28 +
        z_tov   * 0.22 +
        z_ftr   * 0.15 +
        z_away  * 0.18 +
        z_sos   * 0.10 +
        z_close * 0.07
    )

    # Rescale to 0–100 using min-max within population
    lo, hi = raw_score.min(), raw_score.max()
    rng = (hi - lo) if (hi - lo) != 0 else 1
    df["t_tournament_dna_score"] = ((raw_score - lo) / rng * 100).round(1)

    # ── Sub-component flags for readable output ──
    df["t_dna_efg_diff"]      = (efg_off  - LEAGUE_AVG_EFG).round(2)
    df["t_dna_tov_diff"]      = tov_diff.round(2)
    df["t_dna_away_win_pct"]  = away_win_pct.round(1)
    df["t_dna_sos_net_rtg"]   = sos_raw.round(2)

    return df


# ── 1B. Defensive Suffocation Rating ──────────────────────────────────────────

def add_suffocation_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive Suffocation Rating (0–100, higher = more suffocating defense).

    Components:
      30%  Opponent eFG% allowed        (lower = better)
      25%  Opponent TOV% forced         (higher = better)
      20%  Defensive rebound %          (higher = better → deny 2nd chances)
      15%  Opponent FTR allowed         (lower = better → foul discipline)
      10%  Opponent 3P% allowed         (lower = better → perimeter contest quality)

    Uses season-long averages. Where SOS context metrics exist, uses them to
    adjust for opponent offensive quality.
    """
    df = df.copy()

    # Raw defensive metrics — opponent-facing stats (lower = team defending better)
    opp_efg  = _col(df, "opp_avg_efg_season")      # from SOS module
    tov_forced = _col(df, "tov_pct")                # team's own forced TOV rate proxy
    # Better: use what opponents turn it over at vs this team
    # tov_vs_opp_season: positive means forcing more TOs than opponents avg allow
    tov_component = _col(df, "tov_vs_opp_season").fillna(tov_forced - LEAGUE_AVG_TOV)

    drb      = _col(df, "drb_pct")                  # defensive rebounding %
    opp_ftr  = _col(df, "ftr_vs_opp_season")        # negative = holding opp below their avg FTR
    opp_3pct = _col(df, "drtg")                     # use adj_drtg as holistic fallback

    # Prefer adj_drtg (opponent quality adjusted) over raw drtg
    adj_drtg_series = _col(df, "adj_drtg")
    drtg_component  = adj_drtg_series if not adj_drtg_series.isna().all() else _col(df, "drtg")

    # Z-score (flip sign for defensive stats where lower = better)
    def _zscore(s: pd.Series, flip: bool = False) -> pd.Series:
        mu, sigma = s.mean(), s.std()
        if pd.isna(sigma) or sigma == 0:
            z = pd.Series(0.0, index=s.index)
        else:
            z = (s - mu) / sigma
        return -z if flip else z

    z_opp_efg     = _zscore(opp_efg,        flip=True)   # lower opp eFG = better
    z_tov_forced  = _zscore(tov_component,  flip=False)  # higher forced TOV = better
    z_drb         = _zscore(drb,            flip=False)  # higher DRB = better
    z_opp_ftr     = _zscore(opp_ftr,        flip=True)   # lower opp FTR = better
    z_drtg        = _zscore(drtg_component, flip=True)   # lower drtg = better

    raw_score = (
        z_opp_efg    * 0.30 +
        z_tov_forced * 0.25 +
        z_drb        * 0.20 +
        z_opp_ftr    * 0.15 +
        z_drtg       * 0.10
    )

    lo, hi = raw_score.min(), raw_score.max()
    rng = (hi - lo) if (hi - lo) != 0 else 1
    df["t_suffocation_rating"] = ((raw_score - lo) / rng * 100).round(1)

    # Sub-components for drill-down
    df["t_suf_adj_drtg"]      = drtg_component.round(1)
    df["t_suf_drb_pct"]       = drb.round(1)
    df["t_suf_tov_forced_vs"] = tov_component.round(2)

    return df


# ── 1C. Momentum Quality Rating ───────────────────────────────────────────────

def add_momentum_quality_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Momentum Quality Rating (0–100, higher = hot streak is REAL, not regression bait).

    A high MQR = team is genuinely peaking. A low MQR despite a win streak =
    red flag for tournament regression.

    Components:
      25%  3P% sustainability  (L5 3P% vs season avg — gap flags regression risk)
      20%  Opponent quality trend  (opp net rtg L5 vs season — is the streak vs real teams?)
      20%  Defensive efficiency trend  (net_rtg L5 vs season avg net_rtg)
      20%  TOV rate trend  (improving TOV% in last 5 = disciplined execution)
      15%  Scoring margin trend  (L5 capped margin vs season capped margin)

    Note: negative direction on 3P% gap is intentional —
    a team shooting way above their average from 3 recently is PENALIZED.
    """
    df = df.copy()

    # 3P% sustainability — penalize outlier heaters
    three_pct_season = _col(df, "three_pct_l10")    # 10-game as season proxy
    three_pct_l5     = _col(df, "three_pct_l5")
    three_gap        = three_pct_l5 - three_pct_season  # positive gap = regression risk
    three_sustainability = -three_gap                    # flip: sustainable = near-zero gap

    # Opponent quality of recent schedule
    opp_net_season = _col(df, "opp_avg_net_rtg_season")
    opp_net_l5     = _col(df, "opp_avg_net_rtg_l5")
    opp_quality_trend = opp_net_l5 - opp_net_season    # positive = facing tougher opponents lately

    # Defensive trend: net_rtg L5 vs L10 (positive = improving)
    net_rtg_l5  = _col(df, "net_rtg_l5")
    net_rtg_l10 = _col(df, "net_rtg_l10")
    def_trend   = net_rtg_l5 - net_rtg_l10

    # TOV rate trend: lower L5 TOV% than L10 = improvement
    tov_l5  = _col(df, "tov_pct_l5")
    tov_l10 = _col(df, "tov_pct_l10")
    tov_trend = -(tov_l5 - tov_l10)                   # flip: lower TOV = better trend

    # Scoring margin trend
    margin_l5  = _col(df, "margin_capped_l5")
    margin_l10 = _col(df, "margin_capped_l10")
    margin_trend = margin_l5 - margin_l10

    def _zscore(s: pd.Series) -> pd.Series:
        mu, sigma = s.mean(), s.std()
        if pd.isna(sigma) or sigma == 0:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sigma

    z_three  = _zscore(three_sustainability)
    z_opp    = _zscore(opp_quality_trend)
    z_def    = _zscore(def_trend)
    z_tov    = _zscore(tov_trend)
    z_margin = _zscore(margin_trend)

    raw_score = (
        z_three  * 0.25 +
        z_opp    * 0.20 +
        z_def    * 0.20 +
        z_tov    * 0.20 +
        z_margin * 0.15
    )

    lo, hi = raw_score.min(), raw_score.max()
    rng = (hi - lo) if (hi - lo) != 0 else 1
    df["t_momentum_quality_rating"] = ((raw_score - lo) / rng * 100).round(1)

    # Flags for drill-down
    df["t_mom_three_gap"]     = three_gap.round(2)          # >5 = regression risk
    df["t_mom_opp_q_trend"]   = opp_quality_trend.round(2)  # positive = tougher opponents lately
    df["t_mom_net_rtg_trend"] = def_trend.round(2)
    df["t_mom_tov_trend"]     = (tov_l5 - tov_l10).round(2) # negative = improving

    # Regression bait flag: shooting way above average against weak competition
    df["t_regression_risk_flag"] = (
        (three_gap > 5.0) & (opp_quality_trend < 0)
    ).astype(int)

    return df


# ── 1D. Offensive Identity Score ─────────────────────────────────────────────

def add_offensive_identity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Offensive Identity Score — measures clarity and execution of offensive system.

    High score = team has a clear, dominant offensive identity executed consistently.
    Low score = jack-of-all-trades or inconsistent execution.

    Components:
      25%  Adjusted ORTG (efficiency adjusted for opponent defense)
      20%  eFG% vs what opponent typically allows (execution above baseline)
      20%  Assist rate proxy (AST/game — higher = system-based, ball-moving)
      20%  FTR (sustainable scoring source independent of shooting variance)
      15%  Shooting consistency (low efg_std_l10 = consistent execution)

    Identity archetype tags are also computed (perimeter, interior, balanced, grind).
    """
    df = df.copy()

    adj_ortg    = _col(df, "adj_ortg")
    efg_vs_opp  = _col(df, "efg_vs_opp_season")    # above/below what opp allows
    ast         = _col(df, "ast_l10")               # rolling assist count proxy
    ftr         = _col(df, "ftr")
    efg_std     = _col(df, "efg_std_l10")           # lower = more consistent
    three_par   = _col(df, "three_par")             # 3PA rate — archetype signal

    def _zscore(s: pd.Series, flip: bool = False) -> pd.Series:
        mu, sigma = s.mean(), s.std()
        if pd.isna(sigma) or sigma == 0:
            z = pd.Series(0.0, index=s.index)
        else:
            z = (s - mu) / sigma
        return -z if flip else z

    z_adj_ortg   = _zscore(adj_ortg)
    z_efg_vs_opp = _zscore(efg_vs_opp)
    z_ast        = _zscore(ast)
    z_ftr        = _zscore(ftr)
    z_consistency = _zscore(efg_std, flip=True)    # lower std = more consistent = better

    raw_score = (
        z_adj_ortg    * 0.25 +
        z_efg_vs_opp  * 0.20 +
        z_ast         * 0.20 +
        z_ftr         * 0.20 +
        z_consistency * 0.15
    )

    lo, hi = raw_score.min(), raw_score.max()
    rng = (hi - lo) if (hi - lo) != 0 else 1
    df["t_offensive_identity_score"] = ((raw_score - lo) / rng * 100).round(1)

    # ── Archetype tagging ──
    # Perimeter: high 3PA rate + high 3P%
    # Interior: low 3PA rate + high FTR
    # Balanced: moderate 3PA + high AST
    # Grind: low pace + low eFG variance
    three_pct = _col(df, "three_pct")
    pace      = _col(df, "pace")
    three_par_med = three_par.median()
    ftr_med       = ftr.median()
    three_pct_med = three_pct.median()
    pace_med      = pace.median()

    conditions = [
        (three_par > three_par_med) & (three_pct > three_pct_med),
        (three_par < three_par_med) & (ftr > ftr_med),
        (pace < pace_med) & (efg_std < efg_std.median()),
    ]
    choices = ["perimeter", "interior", "grind"]
    df["t_offensive_archetype"] = np.select(conditions, choices, default="balanced")

    df["t_oi_adj_ortg"]   = adj_ortg.round(1)
    df["t_oi_efg_vs_opp"] = efg_vs_opp.round(2)
    df["t_oi_efg_std"]    = efg_std.round(2)

    return df


# ── 1E. Star Reliance Risk ────────────────────────────────────────────────────

def add_star_reliance_risk(
    df: pd.DataFrame,
    player_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Star Reliance Risk Score (0–100, higher = MORE fragile / dangerous dependence).

    If player_df is provided (player_game_metrics.csv), uses actual player-level
    usage and scoring distribution. Otherwise falls back to team-box-score proxies.

    Components (player_df available):
      30%  Top player usage rate  (FGA + 0.44*FTA + TOV share of team poss)
      25%  Scoring distribution entropy  (how evenly spread is scoring?)
      25%  2nd + 3rd scorer share of team PPG
      20%  Win % when top scorer scores below their season average

    Components (box-score-only fallback):
      40%  AST rate (low AST = ISO-heavy = star dependent)
      30%  eFG% variance (high variance = dependent on one player getting hot)
      30%  Bench points % proxy (not directly available; use h2 margin trend)
    """
    df = df.copy()

    if player_df is not None and not player_df.empty:
        df = _star_reliance_from_players(df, player_df)
    else:
        df = _star_reliance_from_box(df)

    return df


def _star_reliance_from_players(
    df: pd.DataFrame,
    player_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute star reliance using player-level data."""

    # Ensure numeric
    player_df = player_df.copy()
    for c in ["pts", "fga", "fta", "tov", "usage_rate", "pts_season_avg"]:
        if c in player_df.columns:
            player_df[c] = pd.to_numeric(player_df[c], errors="coerce")

    # Latest game per player per team (pre-tournament snapshot)
    player_df["_sort_dt"] = pd.to_datetime(
        player_df.get("game_datetime_utc", ""), utc=True, errors="coerce"
    )
    latest_game = (
        player_df.sort_values("_sort_dt")
        .groupby(["team_id", "athlete_id"])
        .last()
        .reset_index()
    )

    # Per team: rank players by season avg pts, compute distribution metrics
    team_player_stats = []
    for team_id, grp in latest_game.groupby("team_id"):
        grp = grp.sort_values("pts_season_avg", ascending=False).reset_index(drop=True)
        if grp.empty:
            continue

        total_pts = grp["pts_season_avg"].sum()
        if total_pts == 0 or pd.isna(total_pts):
            continue

        top_share = grp["pts_season_avg"].iloc[0] / total_pts if total_pts > 0 else np.nan

        # 2nd + 3rd scorer share
        top2_3 = grp["pts_season_avg"].iloc[1:3].sum() / total_pts \
                 if len(grp) >= 3 else np.nan

        # Scoring distribution entropy (Shannon)
        shares = (grp["pts_season_avg"] / total_pts).clip(lower=1e-9)
        entropy = -(shares * np.log(shares)).sum()
        max_entropy = np.log(len(grp)) if len(grp) > 1 else 1
        norm_entropy = entropy / max_entropy  # 0 = one player, 1 = perfectly equal

        # Top player usage rate (L5 rolling avg)
        top_usage = grp["usage_rate_l5"].iloc[0] if "usage_rate_l5" in grp.columns else np.nan

        team_player_stats.append({
            "team_id":           team_id,
            "_top_scorer_share": top_share,
            "_2nd_3rd_share":    top2_3,
            "_scoring_entropy":  norm_entropy,
            "_top_usage":        top_usage,
        })

    if not team_player_stats:
        return _star_reliance_from_box(df)

    player_summary = pd.DataFrame(team_player_stats)

    df = df.merge(player_summary, on="team_id", how="left")

    # Compute raw star reliance score — high score = MORE reliant
    top_share = _col(df, "_top_scorer_share")
    second_third = _col(df, "_2nd_3rd_share")
    entropy = _col(df, "_scoring_entropy")
    top_usage = _col(df, "_top_usage")

    def _zscore(s: pd.Series, flip: bool = False) -> pd.Series:
        mu, sigma = s.mean(), s.std()
        if pd.isna(sigma) or sigma == 0:
            z = pd.Series(0.0, index=s.index)
        else:
            z = (s - mu) / sigma
        return -z if flip else z

    # High top_share and top_usage = high risk; high entropy and 2nd_3rd_share = low risk
    z_share   = _zscore(top_share,    flip=False)   # high = risky
    z_second  = _zscore(second_third, flip=True)    # high 2nd/3rd share = safer → flip
    z_entropy = _zscore(entropy,      flip=True)    # high entropy = safer → flip
    z_usage   = _zscore(top_usage,    flip=False)   # high usage = risky

    raw_risk = (
        z_share   * 0.30 +
        z_second  * 0.25 +
        z_entropy * 0.25 +
        z_usage   * 0.20
    )

    lo, hi = raw_risk.min(), raw_risk.max()
    rng = (hi - lo) if (hi - lo) != 0 else 1
    df["t_star_reliance_risk"] = ((raw_risk - lo) / rng * 100).round(1)
    df["t_star_top_share"]     = top_share.round(3)
    df["t_star_2nd3rd_share"]  = second_third.round(3)
    df["t_star_entropy"]       = entropy.round(3)
    df["t_star_top_usage"]     = top_usage.round(1)

    # Danger zone flag: top player usage >32% is historically the cutoff
    df["t_star_danger_flag"] = (top_usage > 32).astype(int)

    df = df.drop(columns=["_top_scorer_share", "_2nd_3rd_share",
                           "_scoring_entropy", "_top_usage"], errors="ignore")
    return df


def _star_reliance_from_box(df: pd.DataFrame) -> pd.DataFrame:
    """Box-score-only fallback for star reliance when player data unavailable."""
    df = df.copy()

    # AST rate: low assists = ISO-heavy = likely star dependent
    ast_l10 = _col(df, "ast_l10")
    # eFG variance: high variance = dependent on someone getting hot
    efg_std = _col(df, "efg_std_l10")
    # h2 margin trend: teams that fade in 2nd half often lack depth
    h2_margin = _col(df, "h2_margin_l10")

    def _zscore(s: pd.Series, flip: bool = False) -> pd.Series:
        mu, sigma = s.mean(), s.std()
        if pd.isna(sigma) or sigma == 0:
            z = pd.Series(0.0, index=s.index)
        else:
            z = (s - mu) / sigma
        return -z if flip else z

    z_ast    = _zscore(ast_l10,   flip=True)    # low AST = more risky → flip
    z_efg    = _zscore(efg_std,   flip=False)   # high variance = more risky
    z_h2     = _zscore(h2_margin, flip=True)    # fading 2nd halves = more risky → flip

    raw_risk = z_ast * 0.40 + z_efg * 0.30 + z_h2 * 0.30

    lo, hi = raw_risk.min(), raw_risk.max()
    rng = (hi - lo) if (hi - lo) != 0 else 1
    df["t_star_reliance_risk"]  = ((raw_risk - lo) / rng * 100).round(1)
    df["t_star_danger_flag"]    = (efg_std > efg_std.quantile(0.75)).astype(int)
    df["t_star_entropy"]        = np.nan   # not computable without player data
    df["t_star_top_usage"]      = np.nan
    df["t_star_top_share"]      = np.nan
    df["t_star_2nd3rd_share"]   = np.nan

    return df


# ── 1F. Master team tournament profile ───────────────────────────────────────

def compute_tournament_metrics(
    df: pd.DataFrame,
    player_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Full team-level tournament metrics pipeline.
    Input:  team_game_sos.csv (full season history, all teams)
    Output: same DataFrame with t_* columns appended.

    Call after compute_sos_metrics(). One row per team-game.
    Use the most recent row per team as the pre-tournament snapshot.
    """
    if df.empty:
        log.warning("compute_tournament_metrics: empty DataFrame — skipping")
        return df

    log.info(f"Computing tournament metrics for {len(df)} team-game rows")

    df = add_tournament_dna(df)
    df = add_suffocation_rating(df)
    df = add_momentum_quality_rating(df)
    df = add_offensive_identity_score(df)
    df = add_star_reliance_risk(df, player_df=player_df)

    # ── Composite tournament readiness score (meta-index) ──
    # Equal weight of DNA, Suffocation, Momentum; downweighted by Star Risk
    dna  = _col(df, "t_tournament_dna_score")
    suf  = _col(df, "t_suffocation_rating")
    mom  = _col(df, "t_momentum_quality_rating")
    oi   = _col(df, "t_offensive_identity_score")
    risk = _col(df, "t_star_reliance_risk")

    df["t_readiness_composite"] = (
        dna  * 0.30 +
        suf  * 0.28 +
        mom  * 0.20 +
        oi   * 0.15 +
        (100 - risk) * 0.07   # penalize star dependence
    ).round(1)

    log.info("Tournament metrics complete")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MATCHUP-LEVEL: GAME TOTALS PROJECTION
# Input: single row with both team season stats joined (home_ and away_ prefix)
# ═══════════════════════════════════════════════════════════════════════════════

def _team_poss_from_box(fga: float, orb: float, tov: float, fta: float) -> float:
    """Estimate possessions from box score components."""
    return max(fga - orb + tov + 0.44 * fta, 1.0)


def project_game_total(
    home_stats: dict,
    away_stats: dict,
    game_type: str = "ncaa_r1",
    home_seed: Optional[int] = None,
    away_seed: Optional[int] = None,
) -> dict:
    """
    Project the combined scoring total for a single matchup.

    Parameters
    ----------
    home_stats : dict
        Season-aggregate stats for the home/higher-seed team.
        Expected keys (L10 rolling): pace, ortg, drtg, efg_pct,
        tov_pct, ftr, three_pct, three_pct_l5, three_pct_l10,
        adj_ortg, adj_drtg, net_rtg_std_l10, efg_std_l10
    away_stats : dict
        Same keys for the away/lower-seed team.
    game_type : str
        One of 'conf_tournament', 'ncaa_r1', 'ncaa_r2'
    home_seed : int, optional
        NCAA seed (1–16) for the home team
    away_seed : int, optional
        NCAA seed (1–16) for the away team

    Returns
    -------
    dict with projection details and confidence flags
    """

    def _g(d: dict, k: str, default: float = np.nan) -> float:
        v = d.get(k, default)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    # ── Step 1: Adjusted Pace ──
    home_pace = _g(home_stats, "pace_l10", _g(home_stats, "pace", LEAGUE_AVG_PACE))
    away_pace = _g(away_stats, "pace_l10", _g(away_stats, "pace", LEAGUE_AVG_PACE))

    # Slower team drags the game tempo
    if home_pace <= away_pace:
        adj_pace = home_pace * SLOW_TEAM_WEIGHT + away_pace * FAST_TEAM_WEIGHT
    else:
        adj_pace = away_pace * SLOW_TEAM_WEIGHT + home_pace * FAST_TEAM_WEIGHT

    # ── Step 2: Opponent-adjusted offensive efficiency ──
    # Each team's expected points per possession vs this specific defense
    home_adj_ortg = _g(home_stats, "adj_ortg", _g(home_stats, "ortg_l10", LEAGUE_AVG_ORTG))
    away_adj_ortg = _g(away_stats, "adj_ortg", _g(away_stats, "ortg_l10", LEAGUE_AVG_ORTG))
    home_adj_drtg = _g(home_stats, "adj_drtg", _g(home_stats, "drtg_l10", LEAGUE_AVG_DRTG))
    away_adj_drtg = _g(away_stats, "adj_drtg", _g(away_stats, "drtg_l10", LEAGUE_AVG_DRTG))

    # DefRtg adjustment factor: national avg / team's defensive rating
    # >1.0 = weak defense inflates opponent scoring; <1.0 = elite defense suppresses it
    league_drtg = LEAGUE_AVG_DRTG
    home_def_factor = league_drtg / max(home_adj_drtg, 50.0)
    away_def_factor = league_drtg / max(away_adj_drtg, 50.0)

    # Each side's expected efficiency in this matchup
    home_expected_eff = home_adj_ortg * away_def_factor / 100.0
    away_expected_eff = away_adj_ortg * home_def_factor / 100.0

    # ── Step 3: Raw projected total ──
    home_pts_proj = adj_pace * home_expected_eff
    away_pts_proj = adj_pace * away_expected_eff
    raw_total     = home_pts_proj + away_pts_proj

    # ── Step 4: Tournament environment multiplier ──
    base_mult = TOURN_MULTIPLIER.get(game_type, TOURN_MULTIPLIER["ncaa_r1"])

    # Additional multipliers
    seed_gap = abs((home_seed or 8) - (away_seed or 8))
    # Double-digit vs 1/2 seed — top seed slows it down
    double_digit_matchup = (
        home_seed is not None and away_seed is not None and
        ((home_seed <= 2 and away_seed >= 10) or (away_seed <= 2 and home_seed >= 10))
    )
    high_seed_drag = 0.97 if double_digit_matchup else 1.0

    # Both top-50 tempo teams push pace
    both_fast = home_pace > 73 and away_pace > 73
    tempo_bonus = 1.02 if both_fast else 1.0

    final_mult = base_mult * high_seed_drag * tempo_bonus

    projected_total = round(raw_total * final_mult, 1)

    # ── Step 5: Risk flags and confidence adjustments ──
    flags        = []
    shade_adjust = 0.0

    # Regression risk: team shooting 5%+ above season avg from 3 recently
    home_3_gap = _g(home_stats, "three_pct_l5", 33.0) - _g(home_stats, "three_pct_l10", 33.0)
    away_3_gap = _g(away_stats, "three_pct_l5", 33.0) - _g(away_stats, "three_pct_l10", 33.0)
    if home_3_gap > 5.0 or away_3_gap > 5.0:
        flags.append("REGRESSION_RISK")
        shade_adjust -= 3.5

    # Defensive mismatch: one elite defense, one weak offense
    home_net = _g(home_stats, "adj_net_rtg", 0)
    away_net = _g(away_stats, "adj_net_rtg", 0)
    home_drtg_rank_proxy = home_adj_drtg    # lower = better
    away_drtg_rank_proxy = away_adj_drtg
    net_diff = abs(home_net - away_net)
    if net_diff > 12:   # significant talent gap
        flags.append("DEFENSIVE_MISMATCH")
        shade_adjust -= 4.0

    # Slow-game flag: both teams pace < 66
    if home_pace < 66 and away_pace < 66:
        flags.append("SLOW_GAME")
        shade_adjust -= 3.0

    # High-variance flag: either team has high eFG std — could go either way
    home_efg_std = _g(home_stats, "efg_std_l10", 4.0)
    away_efg_std = _g(away_stats, "efg_std_l10", 4.0)
    if home_efg_std > 6.0 or away_efg_std > 6.0:
        flags.append("HIGH_VARIANCE")
        # No shade adjustment — variance cuts both ways

    # Compute confidence band
    variance_factor = max(home_efg_std, away_efg_std, 3.0)
    confidence_band = round(3.0 + variance_factor * 0.5, 1)   # ± pts

    final_projected = round(projected_total + shade_adjust, 1)

    return {
        "game_total_projection":  final_projected,
        "total_raw":              round(raw_total, 1),
        "total_before_flags":     round(projected_total, 1),
        "total_shade_adjustment": round(shade_adjust, 1),
        "total_confidence_band":  confidence_band,
        "adj_pace_projected":     round(adj_pace, 1),
        "home_pts_projected":     round(home_pts_proj * final_mult, 1),
        "away_pts_projected":     round(away_pts_proj * final_mult, 1),
        "game_type":              game_type,
        "tourn_multiplier":       round(final_mult, 4),
        "total_flags":            "|".join(flags) if flags else "NONE",
        "total_flag_count":       len(flags),
        "regression_risk":        int("REGRESSION_RISK" in flags),
        "defensive_mismatch":     int("DEFENSIVE_MISMATCH" in flags),
        "slow_game_flag":         int("SLOW_GAME" in flags),
        "high_variance_flag":     int("HIGH_VARIANCE" in flags),
        "total_direction":        "UNDER" if shade_adjust < -2 else "PUSH" if shade_adjust == 0 else "OVER",
    }


def compute_game_totals_df(
    matchup_df: pd.DataFrame,
    game_type: str = "ncaa_r1",
) -> pd.DataFrame:
    """
    Vectorized wrapper for project_game_total.
    Expects matchup_df with home_ and away_ prefixed season-stat columns.

    home_pace, home_ortg_l10, home_adj_ortg, home_adj_drtg, etc.
    away_pace, away_ortg_l10, away_adj_ortg, away_adj_drtg, etc.
    Optional: home_seed, away_seed

    Returns matchup_df with all projection columns appended.
    """
    results = []
    for _, row in matchup_df.iterrows():
        home_stats = {k.replace("home_", ""): v for k, v in row.items()
                      if k.startswith("home_")}
        away_stats = {k.replace("away_", ""): v for k, v in row.items()
                      if k.startswith("away_")}
        proj = project_game_total(
            home_stats,
            away_stats,
            game_type=game_type,
            home_seed=row.get("home_seed"),
            away_seed=row.get("away_seed"),
        )
        results.append(proj)

    proj_df = pd.DataFrame(results)
    return pd.concat([matchup_df.reset_index(drop=True),
                      proj_df.reset_index(drop=True)], axis=1)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MATCHUP-LEVEL: UNDERDOG WINNER SCORE (UWS)
# Input: two team stat dicts (favorite, underdog) + seed data
# ═══════════════════════════════════════════════════════════════════════════════

# Historical upset rates by seed matchup (30-year NCAA tournament data)
SEED_UPSET_RATES = {
    (1, 16): 0.014,
    (2, 15): 0.087,
    (3, 14): 0.153,
    (4, 13): 0.212,
    (5, 12): 0.354,
    (6, 11): 0.371,
    (7, 10): 0.396,
    (8,  9): 0.490,
}


def _seed_score(fav_seed: Optional[int], dog_seed: Optional[int]) -> float:
    """Map seed matchup to UWS points (0–10)."""
    if fav_seed is None or dog_seed is None:
        return 5.0   # neutral default for unknown seeding / conference tourneys

    lo, hi = min(fav_seed, dog_seed), max(fav_seed, dog_seed)
    rate = SEED_UPSET_RATES.get((lo, hi), 0.25)

    # Scale: 50% upset rate = 10, 1% = 1
    score = round(1 + rate * 18, 1)
    return min(score, 10.0)


def compute_underdog_winner_score(
    favorite_stats: dict,
    underdog_stats: dict,
    fav_seed: Optional[int] = None,
    dog_seed: Optional[int] = None,
    game_type: str = "ncaa_r1",
) -> dict:
    """
    Underdog Winner Score (UWS) — 0 to 70.
    Higher score = stronger upset threat.

    Components (each 0–10):
      C1  Turnover Advantage     — underdog ball security vs fav forced TOV
      C2  3P Reliance Risk       — penalizes volatile perimeter-dependent offenses
      C3  Free Throw Rate Adv    — underdog getting to the line more than favorite
      C4  Defensive Profile      — underdog defense vs favorite's offensive style
      C5  Close-Game Résumé      — experience executing in tight games
      C6  Favorite Vulnerability — specific exploitable weaknesses in the favorite
      C7  Seed Value Gap         — historical base rate for this seed matchup

    Parameters
    ----------
    favorite_stats : dict  — season stats for the favored team
    underdog_stats : dict  — season stats for the underdog
    fav_seed       : int   — NCAA tournament seed (1–16) for favorite
    dog_seed       : int   — NCAA tournament seed (1–16) for underdog
    game_type      : str   — 'ncaa_r1', 'ncaa_r2', 'conf_tournament'
    """

    def _g(d: dict, k: str, default: float = np.nan) -> float:
        v = d.get(k, default)
        return float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else default

    scores = {}
    details = {}

    # ── C1: Turnover Advantage (0–10) ──────────────────────────────────────
    dog_tov    = _g(underdog_stats,  "tov_pct",  18.0)
    fav_forced = _g(favorite_stats,  "tov_vs_opp_season", 0.0)   # how much above avg they force
    # Underdog TOV% scoring
    if dog_tov < 14:
        c1 = 10.0
    elif dog_tov < 16:
        c1 = 7.0
    elif dog_tov < 18:
        c1 = 4.0
    else:
        c1 = 1.0
    # Bonus: if favorite forces TOs above average, underdog's ball security matters more
    if fav_forced > 2.0 and dog_tov < 16:
        c1 = min(c1 + 1.5, 10.0)
    scores["c1_tov_advantage"] = c1
    details["dog_tov_pct"] = dog_tov
    details["fav_tov_force_vs_avg"] = fav_forced

    # ── C2: 3P Reliance & Variance Risk (0–10) ─────────────────────────────
    dog_3par  = _g(underdog_stats, "three_par",  35.0)
    dog_3pct  = _g(underdog_stats, "three_pct",  33.0)
    dog_3std  = _g(underdog_stats, "three_pct_std_l10", 5.0)

    if dog_3par < 30:
        c2 = 10.0    # paint/mid-range dominant — low variance path to wins
    elif dog_3par < 36 and dog_3pct > 36:
        c2 = 7.0     # perimeter-heavy but skilled
    elif dog_3par < 36 and dog_3pct > 33:
        c2 = 5.0
    elif dog_3par > 38 and dog_3pct < 33:
        c2 = 2.0     # boom-or-bust
    elif dog_3par > 40 and dog_3pct < 33:
        c2 = 0.0     # lottery ticket
    else:
        c2 = 4.0
    # Penalize high shooting variance even if average is ok
    if dog_3std > 7.0:
        c2 = max(c2 - 1.5, 0.0)
    scores["c2_3p_variance_risk"] = c2
    details["dog_3par"] = dog_3par
    details["dog_3pct"] = dog_3pct
    details["dog_3pct_std"] = dog_3std

    # ── C3: Free Throw Rate Advantage (0–10) ────────────────────────────────
    dog_ftr = _g(underdog_stats, "ftr",     28.0)
    fav_ftr = _g(favorite_stats, "ftr",     28.0)
    dog_ftp = _g(underdog_stats, "ft_pct",  70.0)

    ftr_diff = dog_ftr - fav_ftr
    if ftr_diff >= 5:
        c3 = 10.0
    elif -2 <= ftr_diff < 5:
        c3 = 6.0
    elif -5 <= ftr_diff < -2:
        c3 = 3.0
    else:
        c3 = 1.0
    # Bonus: if underdog shoots FTs well (makes them count in crunch time)
    if dog_ftp > 73:
        c3 = min(c3 + 1.0, 10.0)
    scores["c3_ftr_advantage"] = c3
    details["dog_ftr"] = dog_ftr
    details["fav_ftr"] = fav_ftr
    details["dog_ft_pct"] = dog_ftp

    # ── C4: Defensive Profile vs Favorite's Offensive Style (0–10) ──────────
    dog_adj_drtg = _g(underdog_stats, "adj_drtg",    103.0)
    fav_3par     = _g(favorite_stats,  "three_par",    35.0)
    fav_3pct     = _g(favorite_stats,  "three_pct",    34.0)
    dog_opp_3pct = _g(underdog_stats, "drtg",         103.0)  # proxy for perimeter defense

    # Base score from underdog's adjusted defensive rating
    # Lower adj_drtg = better defense
    if dog_adj_drtg < 95:      # elite tier (~top 25 nationally)
        c4_base = 8.0
    elif dog_adj_drtg < 100:   # top 80 approx
        c4_base = 5.0
    elif dog_adj_drtg < 105:
        c4_base = 3.0
    else:
        c4_base = 0.0

    # Bonus: elite defense + perimeter-heavy favorite is the prime upset setup
    fav_perimeter_heavy = fav_3par > 37
    if c4_base >= 8.0 and fav_perimeter_heavy:
        c4 = min(c4_base + 2.0, 10.0)
    else:
        c4 = c4_base
    scores["c4_defensive_profile"] = c4
    details["dog_adj_drtg"] = dog_adj_drtg
    details["fav_3par"] = fav_3par

    # ── C5: Experience & Close-Game Résumé (0–10) ────────────────────────────
    dog_close_wpct = _g(underdog_stats, "close_game_win_pct",  0.5)
    dog_close_wpct_pct = dog_close_wpct * 100  # convert to 0–100 scale

    if dog_close_wpct_pct > 60:
        c5 = 10.0
    elif dog_close_wpct_pct > 50:
        c5 = 7.0
    elif dog_close_wpct_pct > 40:
        c5 = 4.0
    else:
        c5 = 1.0

    # Volume bonus: if close win pct is based on many games, it's more reliable
    # We use win_streak as a proxy for recent experience under pressure
    dog_streak = abs(_g(underdog_stats, "win_streak", 0))
    if dog_streak >= 3:
        c5 = min(c5 + 1.0, 10.0)

    scores["c5_close_game_resume"] = c5
    details["dog_close_win_pct"] = round(dog_close_wpct, 3)

    # ── C6: Favorite Vulnerability Index (0–10) ──────────────────────────────
    fav_tov_pct      = _g(favorite_stats, "tov_pct",           17.0)
    fav_3par_v       = _g(favorite_stats, "three_par",          35.0)
    fav_3pct_v       = _g(favorite_stats, "three_pct",          34.0)
    fav_net_rtg      = _g(favorite_stats, "adj_net_rtg",         5.0)
    fav_away_wins    = _g(favorite_stats, "away_wins",            8.0)
    fav_away_losses  = _g(favorite_stats, "away_losses",          3.0)
    fav_luck         = _g(favorite_stats, "luck_score",           0.0)
    fav_net_std      = _g(favorite_stats, "net_rtg_std_l10",      4.0)

    c6 = 0.0
    vuln_flags = []

    # Turnover-prone favorite
    if fav_tov_pct > 17:
        c6 += 3.0;  vuln_flags.append("FAV_TOV_PRONE")

    # Perimeter-dependent + cold shooter risk
    if fav_3par_v > 40 and fav_3pct_v < 34:
        c6 += 2.0;  vuln_flags.append("FAV_3P_VOLATILE")

    # Not actually dominant (narrow efficiency margin)
    if fav_net_rtg < 3.0:
        c6 += 2.0;  vuln_flags.append("FAV_THIN_MARGIN")

    # Hasn't been tested on neutral/road courts
    fav_away_games = fav_away_wins + fav_away_losses
    fav_away_wpct = fav_away_wins / max(fav_away_games, 1)
    if fav_away_wpct < 0.55 and fav_away_games >= 4:
        c6 += 3.0;  vuln_flags.append("FAV_ROAD_UNTESTED")

    # Overperforming their point differential (positive luck = regression candidate)
    if fav_luck > 0.05:
        c6 += 1.0;  vuln_flags.append("FAV_LUCKY_RECORD")

    c6 = min(c6, 10.0)
    scores["c6_favorite_vulnerability"] = c6
    details["fav_vulnerability_flags"] = "|".join(vuln_flags) if vuln_flags else "NONE"
    details["fav_tov_pct"] = fav_tov_pct
    details["fav_away_win_pct"] = round(fav_away_wpct, 3)
    details["fav_luck_score"] = fav_luck

    # ── C7: Seed Value Gap (0–10) ─────────────────────────────────────────────
    c7 = _seed_score(fav_seed, dog_seed)
    scores["c7_seed_gap"] = c7
    details["fav_seed"] = fav_seed
    details["dog_seed"] = dog_seed

    # ── Total UWS ──────────────────────────────────────────────────────────────
    total_uws = sum(scores.values())

    # ── Interpretation ─────────────────────────────────────────────────────────
    if total_uws >= UWS_STRONG_ALERT:
        alert_level = "STRONG_UPSET_ALERT"
        recommendation = "PICK_UNDERDOG"
    elif total_uws >= UWS_LEGITIMATE:
        alert_level = "LEGITIMATE_THREAT"
        recommendation = "UNDERDOG_COVERS"
    elif total_uws >= UWS_MILD_THREAT:
        alert_level = "MILD_THREAT"
        recommendation = "FAVORITE_SURVIVES"
    else:
        alert_level = "PAPER_TIGER"
        recommendation = "FAVORITE_ROLLS"

    # ── Implied upset probability (sigmoid mapped from UWS) ──────────────────
    # 35 UWS ≈ 15%, 45 ≈ 30%, 55 ≈ 45%, 65 ≈ 58%
    uws_normalized = (total_uws - 35) / 30.0   # center around "interesting" range
    upset_prob = round(1 / (1 + np.exp(-uws_normalized * 1.5)) * 0.75, 3)
    # Cap: even the best underdog profile rarely exceeds ~62% upset probability
    upset_prob = min(max(upset_prob, 0.04), 0.62)

    # ── Primary upset narrative ───────────────────────────────────────────────
    max_component = max(scores, key=scores.get)
    narrative_map = {
        "c1_tov_advantage":        "Ball-secure underdog neutralizes athletic pressure",
        "c2_3p_variance_risk":     "Interior/sustainable offense vs perimeter favorite",
        "c3_ftr_advantage":        "Getting to the line is the underdog's equalizer",
        "c4_defensive_profile":    "Elite defense smothers vulnerable favorite offense",
        "c5_close_game_resume":    "Battle-tested underdog built for one-possession games",
        "c6_favorite_vulnerability": "Favorite has exploitable structural weaknesses",
        "c7_seed_gap":             "Historical seed line is a proven upset window",
    }
    primary_narrative = narrative_map.get(max_component, "Balanced upset profile")

    return {
        "uws_total":              round(total_uws, 1),
        "uws_alert_level":        alert_level,
        "uws_recommendation":     recommendation,
        "uws_upset_probability":  upset_prob,
        "uws_primary_narrative":  primary_narrative,
        **scores,
        **details,
        "uws_game_type":          game_type,
    }


def compute_matchup_projections(
    matchup_df: pd.DataFrame,
    game_type: str = "ncaa_r1",
    favorite_col_prefix: str = "home_",
    underdog_col_prefix:  str = "away_",
) -> pd.DataFrame:
    """
    Vectorized wrapper: run both game total and UWS for every row in matchup_df.

    matchup_df must have columns prefixed by favorite_col_prefix and
    underdog_col_prefix for all needed stats, plus optional:
      home_seed, away_seed  (integer NCAA seeds)

    Returns matchup_df with all projection and UWS columns appended.
    """
    total_results = []
    uws_results   = []

    for _, row in matchup_df.iterrows():
        fav_stats = {k.replace(favorite_col_prefix, ""): v
                     for k, v in row.items()
                     if k.startswith(favorite_col_prefix)}
        dog_stats = {k.replace(underdog_col_prefix, ""): v
                     for k, v in row.items()
                     if k.startswith(underdog_col_prefix)}

        total_proj = project_game_total(
            fav_stats, dog_stats,
            game_type=game_type,
            home_seed=row.get(f"{favorite_col_prefix}seed"),
            away_seed=row.get(f"{underdog_col_prefix}seed"),
        )

        uws = compute_underdog_winner_score(
            fav_stats, dog_stats,
            fav_seed=row.get(f"{favorite_col_prefix}seed"),
            dog_seed=row.get(f"{underdog_col_prefix}seed"),
            game_type=game_type,
        )

        total_results.append(total_proj)
        uws_results.append(uws)

    totals_df = pd.DataFrame(total_results)
    uws_df    = pd.DataFrame(uws_results)

    return pd.concat(
        [matchup_df.reset_index(drop=True),
         totals_df.reset_index(drop=True),
         uws_df.reset_index(drop=True)],
        axis=1
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CONVENIENCE: BUILD PRE-TOURNAMENT SNAPSHOT
# ═══════════════════════════════════════════════════════════════════════════════

def build_pretournament_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """
    From the full season team_game_sos (or tournament metrics) DataFrame,
    extract the most recent row per team as their pre-tournament profile.

    Returns one row per team with all t_* composite scores, rolling metrics,
    adj ratings, and SOS metrics as of their last game.

    Use this to feed into compute_matchup_projections() after joining two teams.
    """
    if df.empty:
        return df

    df = df.copy()
    df["_sort_dt"] = pd.to_datetime(
        df.get("game_datetime_utc", ""), utc=True, errors="coerce"
    )
    snapshot = (
        df.sort_values("_sort_dt")
        .groupby("team_id")
        .last()
        .reset_index()
        .drop(columns=["_sort_dt"], errors="ignore")
    )

    log.info(f"Built pre-tournament snapshot: {len(snapshot)} teams")
    return snapshot


def build_matchup_row(
    fav_snapshot: pd.Series,
    dog_snapshot: pd.Series,
    fav_seed: Optional[int] = None,
    dog_seed: Optional[int] = None,
    game_type: str = "ncaa_r1",
) -> dict:
    """
    Convenience function: given two team snapshot Series, build a single
    matchup dict ready for project_game_total() and compute_underdog_winner_score().

    Returns a flat dict with fav_ and dog_ prefixed columns.
    """
    row = {"game_type": game_type}
    for k, v in fav_snapshot.items():
        row[f"fav_{k}"] = v
    for k, v in dog_snapshot.items():
        row[f"dog_{k}"] = v
    row["fav_seed"] = fav_seed
    row["dog_seed"] = dog_seed

    fav_stats = dict(fav_snapshot)
    dog_stats = dict(dog_snapshot)

    total = project_game_total(
        fav_stats, dog_stats,
        game_type=game_type,
        home_seed=fav_seed,
        away_seed=dog_seed,
    )
    uws = compute_underdog_winner_score(
        fav_stats, dog_stats,
        fav_seed=fav_seed,
        dog_seed=dog_seed,
        game_type=game_type,
    )
    return {**row, **total, **uws}
