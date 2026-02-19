#!/usr/bin/env python3
"""
ESPN CBB Pipeline — Team Rankings Engine
"CAGE RATINGS" — Composite Adjusted Grade Engine

Our answer to KenPom / Bart Torvik / Haslametrics.
Combines their foundational metrics with our proprietary composites from
the tournament module and pipeline to produce a single ranked CSV.

METRIC PHILOSOPHY
─────────────────────────────────────────────────────────────────────────────
KenPom     → AdjO, AdjD, AdjEM, AdjT, Luck, SOS              (we replicate)
Torvik     → BARTHAG, WAB, Quad records, RESUME               (we replicate)
Haslametrics → Blended adjusted efficiency + schedule context  (we replicate)
CAGE-only  → Suffocation, Momentum, Clutch, Floor/Ceiling,    (proprietary)
             Power Index, Efficiency Grade, Consistency,
             Star Risk, Offensive Identity, Trend Arrow

OUTPUT: data/cbb_rankings.csv — one row per D1 team, ranked by CAGE_EM
        data/cbb_rankings_YYYYMMDD.csv — dated snapshot

Inputs:
    data/team_pretournament_snapshot.csv  (one row per team, most recent game)
    data/team_game_weighted.csv           (full season, all games — for quads/WAB)

Usage:
    python espn_rankings.py
    python espn_rankings.py --output-dir data/ --top 25

Pipeline position: run after espn_pipeline.py + espn_tournament.py
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
TZ = ZoneInfo("America/Los_Angeles")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR          = Path("data")
CSV_SNAPSHOT      = DATA_DIR / "team_pretournament_snapshot.csv"
CSV_WEIGHTED      = DATA_DIR / "team_game_weighted.csv"
CSV_METRICS       = DATA_DIR / "team_game_metrics.csv"
CSV_LOGS          = DATA_DIR / "team_game_logs.csv"

OUT_RANKINGS      = DATA_DIR / "cbb_rankings.csv"
OUT_RANKINGS_CONF = DATA_DIR / "cbb_rankings_by_conference.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
LEAGUE_AVG_ORTG   = 103.0
LEAGUE_AVG_DRTG   = 103.0
PYTH_EXP          = 11.5      # Pythagorean exponent for CBB
BUBBLE_NET_RTG    = 0.0       # Bubble team definition (NET rating = 0)

# Quad thresholds (based on opponent adj_net_rtg, a proxy for NET ranking tiers)
QUAD_1_MIN_NET    =  8.0      # Top ~25% of D1 (elite opponents)
QUAD_2_MIN_NET    =  0.0      # Above average
QUAD_3_MIN_NET    = -8.0      # Below average
# Quad 4: < -8.0              # Bottom tier

# CAGE Power Index weights — tuned by quant team
POWER_INDEX_WEIGHTS = {
    "cage_em":          0.35,   # Adjusted efficiency margin (core)
    "barthag":          0.20,   # Win probability vs average D1
    "suffocation":      0.15,   # Defensive composite
    "momentum":         0.12,   # Trend-adjusted form
    "resume":           0.10,   # Strength of wins
    "clutch":           0.08,   # Close-game performance
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _to_num(df: pd.DataFrame, col: str, fill: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(fill)
    return pd.Series(fill, index=df.index, dtype=float)


def load_snapshot() -> pd.DataFrame:
    """
    Load team pre-tournament snapshot (one row per team).
    Falls back through the metric chain if snapshot isn't built yet.
    """
    if CSV_SNAPSHOT.exists() and CSV_SNAPSHOT.stat().st_size > 100:
        log.info(f"Loading snapshot from {CSV_SNAPSHOT.name}")
        df = pd.read_csv(CSV_SNAPSHOT, dtype=str, low_memory=False)
    else:
        # Build snapshot on-the-fly from the richest available game log
        for path in [CSV_WEIGHTED, CSV_METRICS, CSV_LOGS]:
            if path.exists() and path.stat().st_size > 100:
                log.info(f"Snapshot not found — building from {path.name}")
                df = pd.read_csv(path, dtype=str, low_memory=False)
                df["game_datetime_utc"] = pd.to_datetime(
                    df.get("game_datetime_utc", pd.NaT), utc=True, errors="coerce"
                )
                df = (
                    df.sort_values("game_datetime_utc")
                    .groupby("team_id")
                    .last()
                    .reset_index()
                )
                break
        else:
            raise FileNotFoundError(
                "No team data found. Run espn_pipeline.py first."
            )

    # Coerce all numeric columns
    str_cols = {
        "team_id", "team", "opponent_id", "opponent", "home_away",
        "conference", "conf_id", "opp_conference", "event_id", "game_id",
        "game_datetime_utc", "game_datetime_pst", "venue", "state",
        "source", "parse_version", "t_offensive_archetype",
        "home_team", "away_team", "home_team_id", "away_team_id",
    }
    for col in df.columns:
        if col not in str_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    log.info(f"Snapshot: {len(df)} teams loaded")
    return df


def load_game_log() -> pd.DataFrame:
    """
    Load full season game log for quad record / WAB computation.
    Returns empty DataFrame if unavailable (rankings still run, quads will be NaN).
    """
    for path in [CSV_WEIGHTED, CSV_METRICS, CSV_LOGS]:
        if path.exists() and path.stat().st_size > 100:
            log.info(f"Loading game log from {path.name} for quad/WAB analysis")
            df = pd.read_csv(path, dtype=str, low_memory=False)
            df["game_datetime_utc"] = pd.to_datetime(
                df.get("game_datetime_utc", pd.NaT), utc=True, errors="coerce"
            )
            for col in ["margin", "win", "opp_avg_net_rtg_season",
                        "adj_net_rtg", "adj_ortg", "adj_drtg",
                        "pythagorean_win_pct", "points_for", "points_against"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            return df
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — REPLICATION METRICS (KenPom / Torvik equivalents)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_barthag(adj_o: pd.Series, adj_d: pd.Series) -> pd.Series:
    """
    BARTHAG — probability of beating an average D1 team on a neutral court.
    Torvik's signature metric. Uses Pythagorean expectation with adjusted ratings.

    P(beat avg D1) = AdjO^EXP / (AdjO^EXP + AdjD^EXP)

    Average D1: AdjO = AdjD = 103.0
    So the formula compares team's efficiency to average opponent quality.
    Teams with AdjO >> AdjD → BARTHAG near 1.0 (elite)
    Teams with AdjO ≈ AdjD → BARTHAG near 0.5

    Unlike KenPom's SRS, this is bounded (0–1) and directly interpretable.
    """
    o = adj_o.clip(lower=60)
    d = adj_d.clip(lower=60)
    o_pow = o ** PYTH_EXP
    d_pow = d ** PYTH_EXP
    return (o_pow / (o_pow + d_pow)).round(4)


def compute_wab(game_log: pd.DataFrame) -> pd.Series:
    """
    WAB — Wins Above Bubble.
    Torvik's metric: how many more wins did this team get vs what a bubble
    team (BARTHAG ≈ 0.50, AdjEM ≈ 0) would have gotten against the SAME schedule?

    For each game:
        bubble_win_prob = P(AdjEM=0 team beats this opponent)
                        = f(opponent net rating) → sigmoid curve
        WAB contribution = actual_win - bubble_win_prob

    Higher WAB = beat better teams than a bubble team would have.
    A team that goes 25-5 vs a weak schedule may have lower WAB than
    a 20-10 team that faced elite opponents throughout.

    Implementation uses opponent's adj_net_rtg per game from the weighted log.
    """
    if game_log.empty:
        return pd.Series(dtype=float)

    required = {"team_id", "win", "opp_avg_net_rtg_season"}
    if not required.issubset(game_log.columns):
        log.warning("WAB: missing required columns in game log — returning NaN")
        return pd.Series(dtype=float)

    def _bubble_win_prob(opp_net: float) -> float:
        """P(bubble team wins) vs opponent with this net rating."""
        # Smooth sigmoid: 0 net → 50%, +10 net → ~27%, -10 net → ~73%
        return float(1.0 / (1.0 + np.exp(opp_net / 8.0)))

    wab_per_team = {}
    for team_id, grp in game_log.groupby("team_id"):
        grp = grp.dropna(subset=["win", "opp_avg_net_rtg_season"])
        if grp.empty:
            wab_per_team[team_id] = np.nan
            continue

        actual_wins  = grp["win"].sum()
        bubble_wins  = grp["opp_avg_net_rtg_season"].apply(_bubble_win_prob).sum()
        wab_per_team[team_id] = round(actual_wins - bubble_wins, 2)

    return pd.Series(wab_per_team, name="wab")


def compute_quad_records(game_log: pd.DataFrame) -> pd.DataFrame:
    """
    Quad records — wins/losses against tiered opponent quality.

    Our quad definitions (using opp_avg_net_rtg_season as opponent quality proxy):
        Quad 1: opp_net_rtg ≥  8.0  (elite, top ~25% of D1)
        Quad 2: opp_net_rtg  0–8.0  (above average)
        Quad 3: opp_net_rtg -8–0.0  (below average)
        Quad 4: opp_net_rtg < -8.0  (weak)

    Note: NCAA official Quads use location + NET rank. Our quads use
    schedule-adjusted opponent quality only (net-rank equivalent).

    Returns DataFrame indexed by team_id with columns:
        q1_w, q1_l, q2_w, q2_l, q3_w, q3_l, q4_w, q4_l
        q1_wpct, best_win_net (best win opponent quality)
        bad_loss_count (losses to Quad 4 opponents)
    """
    if game_log.empty:
        return pd.DataFrame()

    required = {"team_id", "win", "opp_avg_net_rtg_season"}
    if not required.issubset(game_log.columns):
        return pd.DataFrame()

    def _assign_quad(net: float) -> int:
        if net >= QUAD_1_MIN_NET:
            return 1
        elif net >= QUAD_2_MIN_NET:
            return 2
        elif net >= QUAD_3_MIN_NET:
            return 3
        else:
            return 4

    gl = game_log.dropna(subset=["win", "opp_avg_net_rtg_season"]).copy()
    gl["quad"] = gl["opp_avg_net_rtg_season"].apply(_assign_quad)
    gl["win"]  = pd.to_numeric(gl["win"], errors="coerce")

    records = []
    for team_id, grp in gl.groupby("team_id"):
        row = {"team_id": team_id}
        for q in [1, 2, 3, 4]:
            qg = grp[grp["quad"] == q]
            row[f"q{q}_w"] = int(qg["win"].sum())
            row[f"q{q}_l"] = int((1 - qg["win"]).sum())

        # Q1 win % (most important)
        q1_games = row["q1_w"] + row["q1_l"]
        row["q1_wpct"] = round(row["q1_w"] / q1_games, 3) if q1_games > 0 else np.nan

        # Best win: highest opponent net rating in wins
        wins = grp[grp["win"] == 1]
        row["best_win_net"] = round(wins["opp_avg_net_rtg_season"].max(), 1) if not wins.empty else np.nan

        # Bad losses: losses to Q4 opponents (embarrassing)
        row["bad_loss_count"] = int(grp[(grp["quad"] == 4) & (grp["win"] == 0)].shape[0])

        records.append(row)

    if not records:
        return pd.DataFrame(
            columns=[
                "q1_w", "q1_l", "q2_w", "q2_l", "q3_w", "q3_l", "q4_w", "q4_l",
                "q1_wpct", "best_win_net", "bad_loss_count",
            ],
            index=pd.Index([], name="team_id"),
        )

    return pd.DataFrame(records).set_index("team_id")


def compute_resume_score(df: pd.DataFrame, quad_df: pd.DataFrame) -> pd.Series:
    """
    RESUME — Composite quality-win score. Our version of Torvik's "resume" metric.

    Rewards:
      + Wins over elite opponents (Q1)
      + Overall Q1 win rate
      + Best win quality
      + Neutral site wins
      + Road wins

    Penalizes:
      − Bad losses (Q4)
      − Losses to Q3 opponents
      − Heavily negative SOS with mediocre record

    0–100 normalized score. 50 = bubble, >60 = comfortable, >75 = top-10 resume.
    """
    if quad_df.empty:
        return pd.Series(np.nan, index=df.index, name="resume_score")

    snap = df.set_index("team_id") if "team_id" in df.columns else df
    merged = snap.join(quad_df, how="left")

    # Raw components
    q1_w    = merged.get("q1_w",         pd.Series(0, index=merged.index)).fillna(0)
    q1_l    = merged.get("q1_l",         pd.Series(0, index=merged.index)).fillna(0)
    q1_wpct = merged.get("q1_wpct",      pd.Series(0, index=merged.index)).fillna(0)
    bad_l   = merged.get("bad_loss_count",pd.Series(0, index=merged.index)).fillna(0)
    best_w  = merged.get("best_win_net", pd.Series(0, index=merged.index)).fillna(0)
    sos     = _to_num(df if "team_id" not in df.columns else df.set_index("team_id"),
                      "opp_avg_net_rtg_season", 0).reindex(merged.index)
    away_w  = _to_num(merged, "away_wins",   0)
    away_l  = _to_num(merged, "away_losses", 1)
    away_wpct = away_w / (away_w + away_l + 1e-9)

    # Score
    raw = (
        q1_w    * 3.0   +    # Each Q1 win worth 3 points
        q1_wpct * 20.0  +    # Q1 win rate scaled to 20
        best_w  * 0.8   +    # Best win quality
        away_wpct * 15.0 +   # Road success
        sos     * 1.5   +    # Tougher schedule = more credit
        bad_l   * -6.0       # Penalize bad losses hard
    )

    # Normalize 0–100
    lo, hi = raw.min(), raw.max()
    rng = (hi - lo) if (hi - lo) > 0 else 1
    result = ((raw - lo) / rng * 100).round(1)
    result.index = merged.index
    return result.rename("resume_score")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PROPRIETARY CAGE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_consistency_score(df: pd.DataFrame) -> pd.Series:
    """
    CONSISTENCY — How reliable/predictable is this team game-to-game?

    Low variance = high consistency = easier to project (valuable in tournament).
    High variance = boom-or-bust = riskier pick even if average is elite.

    Built from:
    - net_rtg_std_l10   (net efficiency standard deviation, last 10 games)
    - efg_std_l10       (shooting variance, last 10)
    - margin_capped variance (indirect — how often are games close vs blowouts)

    Scores 0–100. 100 = machine-like consistency (e.g. Virginia under Tony Bennett).
    50 = average variance. <35 = wildly inconsistent / dangerous to pick.
    """
    net_std = _to_num(df, "net_rtg_std_l10",   fill=8.0)
    efg_std = _to_num(df, "efg_std_l10",        fill=5.0)
    # Capped margin std: high = inconsistent game scores
    # Approximate from (margin_l10 range) — we don't have raw std but std ≈ 1.5 * mean abs deviation
    # Use net_rtg_std as primary
    margin_std_proxy = net_std * 0.9   # Net rating std ≈ margin std in most cases

    raw_inconsistency = 0.5 * net_std + 0.3 * efg_std + 0.2 * margin_std_proxy

    # Invert: lower inconsistency = higher score
    lo, hi = raw_inconsistency.min(), raw_inconsistency.max()
    rng = (hi - lo) if (hi - lo) > 0 else 1
    score = 100 - ((raw_inconsistency - lo) / rng * 100)
    return score.round(1).rename("consistency_score")


def compute_floor_ceiling(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    """
    FLOOR / CEILING — Downside and upside range for team performance.

    Based on adjusted efficiency margin ± 1.5 standard deviations.
    Floor = what this team looks like on a bad night (but not catastrophic)
    Ceiling = what they can be on their best night

    Why 1.5σ? Captures ~87% of outcomes without extreme outliers dominating.

    Returns (floor_series, ceiling_series) in AdjEM units (pts/100 poss).
    """
    cage_em  = _to_num(df, "adj_net_rtg",    fill=0)
    net_std  = _to_num(df, "net_rtg_std_l10", fill=8.0)

    floor   = (cage_em - 1.5 * net_std).round(1).rename("floor_em")
    ceiling = (cage_em + 1.5 * net_std).round(1).rename("ceiling_em")
    return floor, ceiling


def compute_clutch_rating(df: pd.DataFrame) -> pd.Series:
    """
    CLUTCH_RATING — Tournament survival index based on close-game excellence.

    We define clutch as performance in games decided by ≤ 5 points.
    The pipeline computes close_game_win_pct (season-long).

    We augment with:
    - close_game_win_pct (primary)
    - h2_margin_l10 (second-half performance — do they close games?)
    - luck_score sign (positive luck = benefiting from close game breaks,
      negative luck = performing without the breaks — more impressive clutch)

    0–100 scale. 100 = elite closer. 50 = average. <40 = folds under pressure.
    """
    close_wpct = _to_num(df, "close_game_win_pct", fill=0.5)
    h2_margin  = _to_num(df, "h2_margin_l10",       fill=0.0)
    luck       = _to_num(df, "luck_score",           fill=0.0)

    # Luck adjustment: negative luck + high close win% = genuinely clutch
    # (winning close games despite not getting scoring margin "luck")
    luck_adj = -luck * 20   # penalize lucky teams, reward unlucky winners

    raw = (
        close_wpct * 60.0 +     # Primary signal
        h2_margin  * 1.5  +     # 2nd-half margin (closers win the 2nd)
        luck_adj               # Adjust out luck
    )

    lo, hi = raw.min(), raw.max()
    rng = (hi - lo) if (hi - lo) > 0 else 1
    score = ((raw - lo) / rng * 100).round(1)
    return score.rename("clutch_rating")


def compute_trend_arrow(df: pd.DataFrame) -> pd.Series:
    """
    TREND — Directional momentum indicator.

    Compares L5 net efficiency vs L10 net efficiency:
    ↑↑ SURGE    : L5 > L10 by 5+       (heating up significantly)
    ↑  UP       : L5 > L10 by 2–5
    →  FLAT     : within ±2
    ↓  DOWN     : L5 < L10 by 2–5
    ↓↓ SLIDE    : L5 < L10 by 5+       (dangerous in tournament)

    Encoded as: +2, +1, 0, -1, -2 (for sorting) + arrow label
    """
    net_l5  = _to_num(df, "net_rtg_l5",  fill=0)
    net_l10 = _to_num(df, "net_rtg_l10", fill=0)
    delta   = net_l5 - net_l10

    numeric = pd.cut(
        delta,
        bins=[-999, -5, -2, 2, 5, 999],
        labels=[-2, -1, 0, 1, 2],
    ).astype(float)

    labels = pd.cut(
        delta,
        bins=[-999, -5, -2, 2, 5, 999],
        labels=["↓↓ SLIDE", "↓ DOWN", "→ FLAT", "↑ UP", "↑↑ SURGE"],
    ).astype(str)

    return numeric.rename("trend_numeric"), labels.rename("trend_arrow")


def compute_home_road_delta(df: pd.DataFrame) -> pd.Series:
    """
    HOME_ROAD_DELTA — Gap between home and road performance.

    Large positive delta: team heavily dependent on home crowd — vulnerable
    on neutral courts. Small delta (or negative): road warriors, tournament-ready.

    KenPom accounts for this via his HCA correction. We expose it explicitly.
    """
    home_wins   = _to_num(df, "home_wins",   fill=0)
    home_losses = _to_num(df, "home_losses", fill=0)
    away_wins   = _to_num(df, "away_wins",   fill=0)
    away_losses = _to_num(df, "away_losses", fill=0)

    home_wpct = home_wins / (home_wins + home_losses + 1e-9)
    away_wpct = away_wins / (away_wins + away_losses + 1e-9)

    delta = ((home_wpct - away_wpct) * 100).round(1)
    return delta.rename("home_road_delta_pct")


def compute_power_index(
    df: pd.DataFrame,
    barthag: pd.Series,
    resume: pd.Series,
    clutch: pd.Series,
) -> pd.Series:
    """
    CAGE POWER INDEX — Our master composite ranking number (0–100).

    Combines adjusted efficiency (the engine), BARTHAG (win probability),
    defensive suffocation, recent momentum, resume quality, and clutch rating.

    Weights tuned by quant team through backtesting vs actual tournament outcomes.
    The exact weights are intentionally different from KenPom/Torvik — we weight
    momentum and clutch more heavily because this is optimized for predictive
    accuracy in single-elimination tournament contexts.

    100 = historically elite (2012 Kentucky tier)
    85+ = legitimate national title contender
    70–85 = tournament team, likely 5-seed or better
    55–70 = tournament team, first-round threat
    40–55 = bubble
    <40 = NIT / non-tournament
    """
    cage_em  = _to_num(df, "adj_net_rtg",        fill=0)
    suf      = _to_num(df, "t_suffocation_rating", fill=50)
    mom      = _to_num(df, "t_momentum_quality_rating", fill=50)

    # Normalize each component 0–100
    def _norm(s: pd.Series) -> pd.Series:
        lo, hi = s.min(), s.max()
        rng = (hi - lo) if (hi - lo) > 0 else 1
        return (s - lo) / rng * 100

    em_n      = _norm(cage_em)
    bthg_n    = (barthag.reindex(df.index) * 100)
    suf_n     = suf.reindex(df.index) if hasattr(suf, 'reindex') else suf
    mom_n     = mom.reindex(df.index) if hasattr(mom, 'reindex') else mom
    resume_n  = resume.reindex(df.index).fillna(50) if hasattr(resume, 'reindex') else resume.fillna(50)
    clutch_n  = clutch.reindex(df.index).fillna(50) if hasattr(clutch, 'reindex') else clutch.fillna(50)

    w = POWER_INDEX_WEIGHTS
    raw = (
        em_n     * w["cage_em"]      +
        bthg_n   * w["barthag"]      +
        suf_n    * w["suffocation"]  +
        mom_n    * w["momentum"]     +
        resume_n * w["resume"]       +
        clutch_n * w["clutch"]
    )

    lo, hi = raw.min(), raw.max()
    rng = (hi - lo) if (hi - lo) > 0 else 1
    return ((raw - lo) / rng * 100).round(1).rename("cage_power_index")


def assign_efficiency_grade(cage_em: pd.Series) -> pd.Series:
    """
    EFF_GRADE — Adjusted efficiency margin → human-readable letter grade.

    Graded on actual D1 distribution, not a curve:
    A+ : AdjEM ≥ 25      (national title tier — top 5 teams)
    A  : AdjEM 18–25     (elite — top 10–15)
    A- : AdjEM 12–18     (very good — top 25)
    B+ : AdjEM 7–12      (tournament team, 4–7 seed range)
    B  : AdjEM 2–7       (tournament bubble, 8–10 seed)
    B- : AdjEM 0–2       (bubble)
    C+ : AdjEM -3–0      (NIT contender)
    C  : AdjEM -7–-3     (mediocre)
    C- : AdjEM -12–-7    (poor)
    D  : AdjEM -18–-12   (bottom of conference)
    F  : AdjEM < -18     (historically bad)
    """
    def _grade(em: float) -> str:
        if pd.isna(em):
            return "N/A"
        if em >= 25:   return "A+"
        if em >= 18:   return "A"
        if em >= 12:   return "A-"
        if em >= 7:    return "B+"
        if em >= 2:    return "B"
        if em >= 0:    return "B-"
        if em >= -3:   return "C+"
        if em >= -7:   return "C"
        if em >= -12:  return "C-"
        if em >= -18:  return "D"
        return "F"

    return cage_em.apply(_grade).rename("eff_grade")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CONFERENCE RANKINGS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_conference_ranks(df: pd.DataFrame) -> pd.Series:
    """
    CONF_RANK — Rank within conference by CAGE_EM.
    Returns series indexed same as df.
    """
    if "conference" not in df.columns or "adj_net_rtg" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="conf_rank")

    df = df.copy()
    df["conf_rank"] = (
        df.groupby("conference")["adj_net_rtg"]
        .rank(ascending=False, method="min")
        .astype("Int64")
    )
    return df["conf_rank"]


def compute_conference_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conference-level strength table (separate CSV row).
    Returns one row per conference with avg/median AdjEM, BARTHAG, etc.
    """
    if "conference" not in df.columns:
        return pd.DataFrame()

    agg_cols = {
        "adj_net_rtg":          ["mean", "median", "max", "min"],
        "adj_ortg":             ["mean"],
        "adj_drtg":             ["mean"],
        "barthag":              ["mean", "max"],
        "cage_power_index":     ["mean"],
        "resume_score":         ["mean"],
        "team_id":              "count",
    }

    present_agg = {k: v for k, v in agg_cols.items() if k in df.columns}
    conf_df = df.groupby("conference").agg(present_agg)
    conf_df.columns = ["_".join(c).strip("_") for c in conf_df.columns]
    conf_df = conf_df.rename(columns={"team_id_count": "team_count"})
    conf_df = conf_df.sort_values(
        "adj_net_rtg_mean" if "adj_net_rtg_mean" in conf_df.columns else conf_df.columns[0],
        ascending=False
    )
    conf_df.index.name = "conference"
    return conf_df.reset_index()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — MASTER RANKING BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_rankings(
    snapshot: pd.DataFrame,
    game_log: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the full CAGE rankings table.

    Steps:
    1. Compute derived metrics (BARTHAG, WAB, Quads, Floor/Ceiling, etc.)
    2. Pull in tournament composites (t_* columns from espn_tournament.py)
    3. Compute CAGE Power Index
    4. Rank and format

    Returns one row per D1 team with ~50 columns.
    """
    df = snapshot.copy()

    # Ensure team_id is a column (not index)
    if "team_id" not in df.columns and df.index.name == "team_id":
        df = df.reset_index()

    log.info(f"Building rankings for {len(df)} teams")

    # ── 1. Replicate established metrics ──────────────────────────────────────

    # AdjEM family (already in pipeline as adj_net_rtg, adj_ortg, adj_drtg)
    df["cage_em"]    = _to_num(df, "adj_net_rtg",  fill=0).round(2)
    df["cage_o"]     = _to_num(df, "adj_ortg",      fill=LEAGUE_AVG_ORTG).round(1)
    df["cage_d"]     = _to_num(df, "adj_drtg",      fill=LEAGUE_AVG_DRTG).round(1)
    df["cage_t"]     = _to_num(df, "adj_pace",       fill=70.0).round(1)

    # If adj_* not populated yet, fall back to raw ratings
    if df["cage_em"].abs().max() < 0.1:
        log.warning("adj_net_rtg appears empty — using raw net_rtg as fallback")
        df["cage_em"] = _to_num(df, "net_rtg", fill=0).round(2)
        df["cage_o"]  = _to_num(df, "ortg",    fill=LEAGUE_AVG_ORTG).round(1)
        df["cage_d"]  = _to_num(df, "drtg",    fill=LEAGUE_AVG_DRTG).round(1)
        df["cage_t"]  = _to_num(df, "pace",    fill=70.0).round(1)

    # BARTHAG
    df["barthag"] = compute_barthag(df["cage_o"], df["cage_d"])

    # Luck (already in pipeline)
    df["luck"] = _to_num(df, "luck_score", fill=0).round(3)

    # SOS — use opp_avg_net_rtg_season as primary, fall back to opp_avg_ortg_season
    df["sos"] = _to_num(df, "opp_avg_net_rtg_season",
                        fill=_to_num(df, "opp_avg_ortg_season", 0) - LEAGUE_AVG_ORTG).round(2)

    # Expected win % (Pythagorean from pipeline)
    df["expected_win_pct"] = _to_num(df, "pythagorean_win_pct",
                                     fill=df["barthag"]).round(3)

    # Actual win %
    wins   = _to_num(df, "wins",   fill=0)
    losses = _to_num(df, "losses", fill=0)
    df["actual_win_pct"] = (wins / (wins + losses + 1e-9)).round(3)

    # ── 2. WAB from game log ──────────────────────────────────────────────────
    wab_series = compute_wab(game_log)
    if not wab_series.empty:
        df["wab"] = df["team_id"].astype(str).map(wab_series).round(2)
    else:
        # Approximate WAB from available data if game log missing
        df["wab"] = ((df["actual_win_pct"] - df["expected_win_pct"]) * (wins + losses)).round(2)

    # ── 3. Quad records from game log ────────────────────────────────────────
    quad_df = compute_quad_records(game_log)
    if not quad_df.empty:
        quad_df.index = quad_df.index.astype(str)
        df_indexed = df.set_index("team_id")
        df_indexed.index = df_indexed.index.astype(str)
        df = df_indexed.join(quad_df, how="left").reset_index().rename(
            columns={"index": "team_id"}
        )
        for q in [1, 2, 3, 4]:
            for s in ["w", "l"]:
                col = f"q{q}_{s}"
                if col in df.columns:
                    df[col] = df[col].fillna(0).astype(int)
        df["q1_record"] = df["q1_w"].astype(str) + "-" + df["q1_l"].astype(str)
        df["q2_record"] = df["q2_w"].astype(str) + "-" + df["q2_l"].astype(str)
        df["q3_record"] = df["q3_w"].astype(str) + "-" + df["q3_l"].astype(str)
        df["q4_record"] = df["q4_w"].astype(str) + "-" + df["q4_l"].astype(str)
    else:
        for col in ["q1_record", "q2_record", "q3_record", "q4_record",
                    "q1_wpct", "best_win_net", "bad_loss_count"]:
            df[col] = np.nan

    # ── 4. Resume score ───────────────────────────────────────────────────────
    resume = compute_resume_score(df, quad_df)
    df["resume_score"] = df["team_id"].astype(str).map(
        resume.reset_index().set_index("team_id")["resume_score"]
        if "team_id" in resume.reset_index().columns else resume
    ).fillna(50).round(1)

    # ── 5. Proprietary CAGE metrics ───────────────────────────────────────────

    df["consistency_score"]  = compute_consistency_score(df)
    df["clutch_rating"]      = compute_clutch_rating(df)
    df["home_road_delta_pct"] = compute_home_road_delta(df)

    floor_s, ceiling_s = compute_floor_ceiling(df)
    df["floor_em"]   = floor_s.values
    df["ceiling_em"] = ceiling_s.values

    trend_num, trend_arrow = compute_trend_arrow(df)
    df["trend_numeric"] = trend_num.values
    df["trend_arrow"]   = trend_arrow.values

    # ── 6. Pull tournament composites (from espn_tournament.py output) ────────
    # These are present if espn_tournament.py has run; gracefully NaN if not
    tourn_cols = {
        "t_tournament_dna_score":       "dna_score",
        "t_suffocation_rating":         "suffocation",
        "t_momentum_quality_rating":    "momentum",
        "t_offensive_identity_score":   "off_identity",
        "t_star_reliance_risk":         "star_risk",
        "t_readiness_composite":        "tourney_readiness",
        "t_offensive_archetype":        "offensive_archetype",
        "t_regression_risk_flag":       "regression_risk",
        "t_dna_efg_diff":               "dna_efg_diff",
        "t_dna_tov_diff":               "dna_tov_diff",
        "t_dna_away_win_pct":           "away_win_pct_tourn",
        "t_mom_three_gap":              "three_pct_gap",
        "t_star_danger_flag":           "star_danger",
    }
    for src, dst in tourn_cols.items():
        if src in df.columns:
            df[dst] = df[src]
        else:
            df[dst] = np.nan

    # Fallback: compute basic suffocation from raw stats if tournament module hasn't run
    if df["suffocation"].isna().all():
        opp_efg  = _to_num(df, "opp_avg_efg_season", LEAGUE_AVG_EFG := 50.5)
        drb_pct  = _to_num(df, "drb_pct",             72.0)
        cage_d_s = df["cage_d"]
        raw_suf = (100 - opp_efg) * 0.4 + drb_pct * 0.3 + (120 - cage_d_s) * 0.3
        lo, hi  = raw_suf.min(), raw_suf.max()
        df["suffocation"] = ((raw_suf - lo) / max(hi - lo, 1) * 100).round(1)

    # ── 7. CAGE Power Index ───────────────────────────────────────────────────
    df["cage_power_index"] = compute_power_index(
        df,
        barthag=df["barthag"],
        resume=df["resume_score"],
        clutch=df["clutch_rating"],
    )

    # ── 8. Efficiency grade ───────────────────────────────────────────────────
    df["eff_grade"] = assign_efficiency_grade(df["cage_em"])

    # ── 9. Conference rank ────────────────────────────────────────────────────
    df["conf_rank"] = compute_conference_ranks(df)

    # ── 10. Record string ─────────────────────────────────────────────────────
    df["record"]      = wins.astype(int).astype(str) + "-" + losses.astype(int).astype(str)
    df["home_record"] = (_to_num(df, "home_wins",  0).astype(int).astype(str) + "-" +
                         _to_num(df, "home_losses",0).astype(int).astype(str))
    df["away_record"] = (_to_num(df, "away_wins",  0).astype(int).astype(str) + "-" +
                         _to_num(df, "away_losses",0).astype(int).astype(str))

    # ── 11. Raw four-factor stats (for reference, like KenPom secondary table) ──
    df["efg_pct"]     = _to_num(df, "efg_pct",    fill=np.nan).round(1)
    df["opp_efg_pct"] = _to_num(df, "opp_avg_efg_season", fill=np.nan).round(1)
    df["tov_pct"]     = _to_num(df, "tov_pct",    fill=np.nan).round(1)
    df["orb_pct"]     = _to_num(df, "orb_pct",    fill=np.nan).round(1)
    df["drb_pct"]     = _to_num(df, "drb_pct",    fill=np.nan).round(1)
    df["ftr"]         = _to_num(df, "ftr",         fill=np.nan).round(2)
    df["three_par"]   = _to_num(df, "three_par",   fill=np.nan).round(1)
    df["three_pct"]   = _to_num(df, "three_pct",   fill=np.nan).round(1)
    df["ft_pct"]      = _to_num(df, "ft_pct",      fill=np.nan).round(1)
    df["close_wpct"]  = _to_num(df, "close_game_win_pct", fill=np.nan).round(3)

    # ── 12. L5 / L10 net for quick trend check ───────────────────────────────
    df["net_rtg_l5"]  = _to_num(df, "net_rtg_l5",  fill=np.nan).round(1)
    df["net_rtg_l10"] = _to_num(df, "net_rtg_l10", fill=np.nan).round(1)

    # ── 13. Rank by CAGE_EM (primary), BARTHAG (tiebreak) ────────────────────
    df = df.sort_values(["cage_em", "barthag"], ascending=[False, False])
    df["rank"] = range(1, len(df) + 1)
    df = df[["rank", *[c for c in df.columns if c != "rank"]]]

    # ── 14. Metadata ──────────────────────────────────────────────────────────
    df["updated_at"] = datetime.now(TZ).strftime("%Y-%m-%d %H:%M %Z")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CSV FORMATTER
# ═══════════════════════════════════════════════════════════════════════════════

# Canonical column order + display names (mirrors KenPom/Torvik layout logic)
RANKINGS_COLUMNS = [
    # Identity
    "rank",
    "team",
    "team_id",
    "conference",
    "conf_rank",
    "record",
    "home_record",
    "away_record",

    # Core efficiency (KenPom equivalents)
    "cage_em",          # AdjEM equivalent — primary sort
    "cage_o",           # AdjO
    "cage_d",           # AdjD
    "cage_t",           # AdjT (tempo)

    # Win probability (Torvik equivalents)
    "barthag",          # P(beat avg D1)
    "wab",              # Wins Above Bubble
    "expected_win_pct", # Pythagorean
    "actual_win_pct",

    # Schedule
    "luck",             # Actual win% - Pythagorean win%
    "sos",              # Strength of Schedule (opp avg net rtg)

    # Quad records (NCAA/Torvik style)
    "q1_record",        # vs elite opponents
    "q2_record",
    "q3_record",
    "q4_record",
    "q1_wpct",
    "best_win_net",
    "bad_loss_count",

    # CAGE proprietary composites
    "cage_power_index", # Master composite (0–100)
    "eff_grade",        # Letter grade
    "resume_score",     # Quality-wins composite
    "suffocation",      # Defensive composite
    "momentum",         # Hot-streak quality
    "clutch_rating",    # Close-game excellence
    "consistency_score",# Variance-based reliability
    "off_identity",     # Offensive system strength
    "tourney_readiness",# Overall tournament profile
    "star_risk",        # Fragility if star struggles
    "dna_score",        # Tournament DNA

    # Floor / Ceiling
    "floor_em",         # Worst realistic performance (AdjEM units)
    "ceiling_em",       # Best realistic performance

    # Trend
    "trend_arrow",
    "trend_numeric",

    # Style
    "offensive_archetype",
    "home_road_delta_pct",

    # Flags
    "regression_risk",
    "star_danger",
    "three_pct_gap",

    # Four factors (reference, like KenPom's secondary table)
    "efg_pct",
    "opp_efg_pct",
    "tov_pct",
    "orb_pct",
    "drb_pct",
    "ftr",
    "three_par",
    "three_pct",
    "ft_pct",
    "close_wpct",
    "net_rtg_l5",
    "net_rtg_l10",

    # Meta
    "updated_at",
]


def format_rankings_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Select and order columns for final CSV output."""
    available = [c for c in RANKINGS_COLUMNS if c in df.columns]
    out = df[available].copy()

    # Round floats that may have crept to too many decimals
    float_2 = ["cage_em", "cage_o", "cage_d", "cage_t", "sos", "wab",
                "floor_em", "ceiling_em", "net_rtg_l5", "net_rtg_l10"]
    float_1 = ["cage_power_index", "resume_score", "suffocation", "momentum",
                "clutch_rating", "consistency_score", "off_identity",
                "tourney_readiness", "dna_score", "star_risk",
                "home_road_delta_pct", "efg_pct", "opp_efg_pct", "tov_pct",
                "orb_pct", "drb_pct", "three_par", "three_pct", "ft_pct"]

    for col in float_2:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)
    for col in float_1:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)

    return out


def print_top_n(df: pd.DataFrame, n: int = 25) -> None:
    """Print top-N rankings table to stdout."""
    top = df.head(n)
    print()
    print("=" * 110)
    print(f"  CAGE RANKINGS — Top {n}   (Composite Adjusted Grade Engine)")
    print(f"  {datetime.now(TZ).strftime('%B %d, %Y  %I:%M %p %Z')}")
    print("=" * 110)
    print(
        f"{'RK':>3} {'TEAM':<28} {'CONF':<10} {'REC':<8} "
        f"{'CAGE_EM':>8} {'AdjO':>6} {'AdjD':>6} {'AdjT':>6} "
        f"{'BARTHAG':>8} {'WAB':>6} {'GRD':>4} {'PI':>5} "
        f"{'MOMO':>5} {'TREND':>9}"
    )
    print("-" * 110)

    for _, row in top.iterrows():
        def g(col, fmt="{:.1f}", default="—"):
            v = row.get(col)
            try:
                return fmt.format(float(v))
            except Exception:
                return str(default)

        print(
            f"{int(row.get('rank', 0)):>3} "
            f"{str(row.get('team',''))[:27]:<28} "
            f"{str(row.get('conference',''))[:9]:<10} "
            f"{str(row.get('record','')):<8} "
            f"{g('cage_em', '{:+.2f}'):>8} "
            f"{g('cage_o'):>6} "
            f"{g('cage_d'):>6} "
            f"{g('cage_t'):>6} "
            f"{g('barthag', '{:.4f}'):>8} "
            f"{g('wab', '{:+.1f}'):>6} "
            f"{str(row.get('eff_grade','')):<4} "
            f"{g('cage_power_index', '{:.0f}'):>5} "
            f"{g('momentum', '{:.0f}'):>5} "
            f"{str(row.get('trend_arrow',''))[:9]:>9}"
        )

    print("=" * 110)
    print(f"  CAGE_EM: Composite Adjusted Efficiency Margin (higher = better)")
    print(f"  BARTHAG: P(beat avg D1 team on neutral court)")
    print(f"  WAB: Wins Above Bubble | PI: Power Index (0-100) | MOMO: Momentum Score")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run(output_dir: Path = DATA_DIR, top_n: int = 25) -> Path:
    """Full rankings pipeline."""
    log.info("=" * 60)
    log.info("CAGE Rankings Engine — Building D1 Team Rankings")
    log.info("=" * 60)

    # Load data
    snapshot = load_snapshot()
    game_log = load_game_log()

    # Build rankings
    rankings = build_rankings(snapshot, game_log)
    formatted = format_rankings_csv(rankings)

    # Print top-N to stdout
    print_top_n(formatted, n=top_n)

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    out_latest = output_dir / "cbb_rankings.csv"
    out_dated  = output_dir / f"cbb_rankings_{datetime.now(TZ).strftime('%Y%m%d')}.csv"

    formatted.to_csv(out_latest, index=False)
    formatted.to_csv(out_dated,  index=False)
    log.info(f"Rankings written → {out_latest} ({len(formatted)} teams)")
    log.info(f"Dated snapshot   → {out_dated}")

    # Conference summary
    conf_table = compute_conference_strength(rankings)
    if not conf_table.empty:
        out_conf = output_dir / "cbb_rankings_by_conference.csv"
        conf_table.to_csv(out_conf, index=False)
        log.info(f"Conference table → {out_conf}")

    return out_latest


def main():
    parser = argparse.ArgumentParser(description="Build CAGE CBB team rankings")
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--top",        type=int,  default=25,
                        help="Print top-N teams to stdout")
    args = parser.parse_args()
    run(output_dir=args.output_dir, top_n=args.top)


if __name__ == "__main__":
    main()
