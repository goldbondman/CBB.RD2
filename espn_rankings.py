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
        return pd.DataFrame()

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
    # Build merge deterministically by aligned assignment so quad_df columns
    # always overwrite stale snapshot columns without pandas overlap errors.
    merged = snap.copy()
    quad_aligned = quad_df.reindex(merged.index)
    for col in quad_aligned.columns:
        merged[col] = quad_aligned[col]

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

    Uses nullable Int64 so teams with no conference (independents, NaN) get
    <NA> rather than crashing astype(int) — which was the original bug causing
    conf_rank to be missing for ALL teams whenever any team had a NaN conference.
    """
    # Need both columns; fall back gracefully if either missing
    rank_col = "cage_em" if "cage_em" in df.columns else "adj_net_rtg"
    if "conference" not in df.columns or rank_col not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Int64", name="conf_rank")

    # Drop rows with NaN conference from the ranking, then re-index back
    # so independents get <NA> rather than throwing ValueError
    try:
        ranks = (
            df.groupby("conference", dropna=True)[rank_col]
            .rank(ascending=False, method="min")
            .astype("Int64")   # nullable — NaN-conference rows stay <NA>
        )
        return ranks.rename("conf_rank")
    except Exception as exc:
        log.warning(f"compute_conference_ranks failed: {exc} — returning NaN")
        return pd.Series(pd.NA, index=df.index, dtype="Int64", name="conf_rank")


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
        # Re-runs can carry stale quad columns from prior snapshots. Remove
        # overlaps so fresh quad_df values always win and join stays stable.
        quad_overlap_cols = [c for c in quad_df.columns if c in df_indexed.columns]
        if quad_overlap_cols:
            df_indexed = df_indexed.drop(columns=quad_overlap_cols, errors="ignore")
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
    # Called after cage_em is set (step 1) so compute_conference_ranks uses cage_em.
    # conference column must survive the quad join — confirmed it does via reset_index.
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
    df.insert(0, "rank", range(1, len(df) + 1))

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

def write_data_dictionary(output_dir: Path = DATA_DIR) -> Path:
    """
    Write cbb_rankings_data_dictionary.csv alongside the rankings files.

    Two formats:
    - CSV: machine-readable, one row per column with name / category /
           description / range / source / notes
    - TXT: human-readable formatted version for quick reference

    Both are written every time rankings are generated so they stay in sync.
    """

    # ── Column definitions ────────────────────────────────────────────────────
    # Format: (column_name, category, description, typical_range, source, notes)
    COLUMNS = [
        # ── Identity ──────────────────────────────────────────────────────────
        ("rank",           "Identity", "Overall CAGE ranking, sorted by CAGE_EM then BARTHAG as tiebreaker.", "1 – 364", "Computed", "1 = best team in D1."),
        ("team",           "Identity", "Team name.", "—", "ESPN API", ""),
        ("team_id",        "Identity", "ESPN internal team identifier.", "—", "ESPN API", "Use for joins with other pipeline CSVs."),
        ("conference",     "Identity", "Athletic conference affiliation.", "—", "ESPN API", ""),
        ("conf_rank",      "Identity", "Rank within conference by CAGE_EM. 1 = best team in that conference.", "1 – ~18", "Computed", "NaN for independents with no conference."),
        ("record",         "Identity", "Overall win-loss record (season).", "e.g. 24-8", "Computed", ""),
        ("home_record",    "Identity", "Home win-loss record.", "e.g. 14-2", "Computed", ""),
        ("away_record",    "Identity", "Away win-loss record.", "e.g. 7-5", "Computed", ""),

        # ── Core Efficiency (KenPom equivalents) ──────────────────────────────
        ("cage_em",        "Core Efficiency", "CAGE Efficiency Margin — our AdjEM equivalent. Points per 100 possessions better than an average D1 team, adjusted for opponent quality and pace. Primary sort column.", "-25 to +30", "Computed from adj_net_rtg", "KenPom equivalent: AdjEM. Higher is better. A+: ≥25, A: 18–25, B+: 7–12, Bubble: 0–2."),
        ("cage_o",         "Core Efficiency", "CAGE Adjusted Offensive Rating — points scored per 100 possessions, adjusted for opponent defensive quality.", "95 – 125", "Computed from adj_ortg", "KenPom equivalent: AdjO. League average ≈ 103."),
        ("cage_d",         "Core Efficiency", "CAGE Adjusted Defensive Rating — points allowed per 100 possessions, adjusted for opponent offensive quality. Lower is better.", "85 – 115", "Computed from adj_drtg", "KenPom equivalent: AdjD. League average ≈ 103. Elite defenses: <92."),
        ("cage_t",         "Core Efficiency", "CAGE Adjusted Tempo — possessions per 40 minutes, pace-adjusted. Neither fast nor slow is inherently better.", "60 – 80", "Computed from adj_pace", "KenPom equivalent: AdjT. High = up-tempo, Low = slow and deliberate."),

        # ── Win Probability (Torvik equivalents) ──────────────────────────────
        ("barthag",        "Win Probability", "Probability of beating an average D1 team on a neutral court. Torvik's signature metric. Computed via Pythagorean expectation: AdjO^11.5 / (AdjO^11.5 + AdjD^11.5).", "0.000 – 1.000", "Computed from cage_o / cage_d", "Torvik equivalent: BARTHAG. Elite ≥0.90, Bubble ≈0.50, Bottom <0.20."),
        ("wab",            "Win Probability", "Wins Above Bubble — how many more wins this team earned vs what a bubble team (AdjEM=0) would have earned against the identical schedule.", "-5 to +12", "Computed from game log", "Torvik equivalent: WAB. Accounts for schedule difficulty. A 25-5 record vs cupcakes may have lower WAB than 20-10 vs elite schedule."),
        ("expected_win_pct","Win Probability","Pythagorean win percentage — expected win rate based on scoring efficiency, independent of actual results.", "0.000 – 1.000", "Pipeline", "High expected_win_pct with low actual_win_pct = unlucky team due for regression upward."),
        ("actual_win_pct", "Win Probability", "Actual season win percentage.", "0.000 – 1.000", "Computed", ""),

        # ── Schedule ──────────────────────────────────────────────────────────
        ("luck",           "Schedule", "Luck score — actual win% minus Pythagorean win%. Positive = winning more games than efficiency predicts (lucky). Negative = unlucky.", "-0.15 to +0.15", "Pipeline", "KenPom equivalent: Luck. Teams with high luck tend to regress toward their Pythagorean expectation. Used in RegressedEff ensemble model."),
        ("sos",            "Schedule", "Strength of Schedule — average opponent adjusted net rating this team has faced. Positive = tougher than average schedule.", "-10 to +10", "Pipeline (opp_avg_net_rtg_season)", "KenPom equivalent: SOS. Contextualizes record and efficiency ratings."),

        # ── Quad Records ──────────────────────────────────────────────────────
        ("q1_record",      "Quad Records", "Record vs Quad 1 opponents (opp net rating ≥ +8.0 — elite, top ~25% of D1).", "e.g. 4-3", "Computed from game log", "Most important quad for NCAA tournament seeding. Q1 wins are the gold standard."),
        ("q2_record",      "Quad Records", "Record vs Quad 2 opponents (opp net rating 0 to +8.0 — above average).", "e.g. 8-2", "Computed from game log", ""),
        ("q3_record",      "Quad Records", "Record vs Quad 3 opponents (opp net rating -8 to 0 — below average).", "e.g. 10-1", "Computed from game log", "Losses here are mildly concerning. Expected wins for tournament-caliber teams."),
        ("q4_record",      "Quad Records", "Record vs Quad 4 opponents (opp net rating < -8.0 — weak bottom tier).", "e.g. 6-0", "Computed from game log", "Losses to Q4 opponents are bad losses and damage tournament resume significantly."),
        ("q1_wpct",        "Quad Records", "Win percentage in Quad 1 games specifically.", "0.000 – 1.000", "Computed", "Most predictive single quad stat for NCAA seeding. Elite teams maintain >0.50 Q1 record."),
        ("best_win_net",   "Quad Records", "Best win quality — highest opponent adjusted net rating in a game this team won.", "-5 to +25", "Computed from game log", "Higher = better marquee win. Elite resume: best win net >15."),
        ("bad_loss_count", "Quad Records", "Number of losses to Quad 4 opponents (embarrassing losses).", "0 – 5+", "Computed", "Each bad loss significantly hurts tournament seeding and resume score."),

        # ── CAGE Proprietary Composites ───────────────────────────────────────
        ("cage_power_index","CAGE Composite", "Master composite ranking score combining efficiency, win probability, defense, momentum, resume, and clutch performance. Our single best overall team quality number.", "0 – 100", "Computed (weighted blend)", "100 = historically elite (2012 Kentucky tier). 85+ = title contender. 70–85 = top-5 seed. 55–70 = tournament team. 40–55 = bubble. <40 = NIT."),
        ("eff_grade",      "CAGE Composite", "Letter grade for adjusted efficiency margin. Calibrated to actual D1 distribution.", "A+ through F", "Computed from cage_em", "A+: ≥25. A: 18–25. A-: 12–18. B+: 7–12. B: 2–7. B-: 0–2. C+: -3–0. C: -7–-3. D: -18–-12. F: <-18."),
        ("resume_score",   "CAGE Composite", "Quality-wins composite on 0–100 scale. Rewards Q1 wins, win rate vs good teams, road success. Penalizes bad losses. Our version of Torvik's resume metric.", "0 – 100", "Computed", "50 = bubble-quality resume. >60 = comfortable. >75 = top-10 resume. Useful for committee-style seeding analysis."),
        ("suffocation",    "CAGE Composite", "Defensive composite — how completely this defense shuts down opponents. Combines opponent eFG%, defensive rebounding, and adjusted defensive rating.", "0 – 100", "Tournament module / fallback computed", "50 = average. >75 = elite shutdown defense. Low-scoring game predictor."),
        ("momentum",       "CAGE Composite", "Momentum Quality Rating — recent form weighted by opponent quality. Not just win streak: a 5-game win streak vs weak opponents scores lower than 3 wins vs top-25.", "0 – 100", "Tournament module", "50 = neutral. >65 = meaningfully hot. >80 = surging through elite competition."),
        ("clutch_rating",  "CAGE Composite", "Close-game excellence — performance in games decided by ≤5 points, adjusted for luck in those games.", "0 – 100", "Computed", "50 = average. >70 = consistently closes games. Important for single-elimination tournament prediction."),
        ("consistency_score","CAGE Composite","How predictable/reliable this team is game-to-game. Inverse of net efficiency and shooting variance.", "0 – 100", "Computed from std dev", "100 = machine-like (Virginia under Bennett). 50 = average. <35 = boom-or-bust — dangerous to pick in tournament."),
        ("off_identity",   "CAGE Composite", "Offensive Identity Score — how cohesive and defined the offensive system is. High = team executes a clear, repeatable offensive scheme.", "0 – 100", "Tournament module", "High identity teams are more consistent under tournament pressure."),
        ("tourney_readiness","CAGE Composite","Overall tournament readiness composite — aggregates schedule toughness, recent form, neutral-court performance, and defensive profile.", "0 – 100", "Tournament module", "Specifically calibrated for single-elimination. Best single number for tournament bracket analysis."),
        ("star_risk",      "CAGE Composite", "Star Reliance Risk — how dependent this team is on a single player's performance. High = fragile if the star has an off night.", "0 – 100", "Tournament module", "50 = balanced team. >70 = one player carries them — high variance in tournament. Best paired with consistency_score."),
        ("dna_score",      "CAGE Composite", "Tournament DNA Index — composite capturing historical behavioral markers that predict tournament success: road wins, close wins, elite opponent wins.", "0 – 100", "Tournament module", "The 'proven it matters' metric. A team with dna_score >70 has demonstrated performance specifically in tournament-like conditions."),

        # ── Floor / Ceiling ───────────────────────────────────────────────────
        ("floor_em",       "Range", "Downside performance estimate — team's CAGE_EM minus 1.5 standard deviations. What they look like on a bad night (captures ~87% of outcomes).", "-35 to +20", "Computed from cage_em ± 1.5σ", "In AdjEM units. Negative floor_em means on a bad night this team plays below average D1 level. Critical tournament risk metric."),
        ("ceiling_em",     "Range", "Upside performance estimate — team's CAGE_EM plus 1.5 standard deviations. Peak performance potential.", "-10 to +45", "Computed from cage_em ± 1.5σ", "The gap between floor_em and ceiling_em reflects variance/unpredictability. Wide gap = boom-or-bust team."),

        # ── Trend ─────────────────────────────────────────────────────────────
        ("trend_arrow",    "Trend", "Directional momentum indicator comparing last 5 games vs last 10 games net efficiency.", "↑↑ SURGE / ↑ UP / → FLAT / ↓ DOWN / ↓↓ SLIDE", "Computed from net_rtg_l5 vs net_rtg_l10", "↑↑ SURGE: L5 better by 5+ pts. ↑ UP: 2–5 pts. → FLAT: within ±2. ↓ DOWN: 2–5 worse. ↓↓ SLIDE: 5+ worse."),
        ("trend_numeric",  "Trend", "Numeric encoding of trend_arrow for sorting. +2=SURGE, +1=UP, 0=FLAT, -1=DOWN, -2=SLIDE.", "-2 to +2", "Computed", "Use for sorting/filtering. trend_arrow is for display."),

        # ── Style ─────────────────────────────────────────────────────────────
        ("offensive_archetype","Style","Offensive system classification based on pace, 3-point rate, and free throw generation.", "e.g. PACE-AND-SPACE, GRIND-IT-OUT, DRIBBLE-DRIVE", "Tournament module", "Useful for matchup analysis — some archetypes systematically exploit others."),
        ("home_road_delta_pct","Style","Percentage point gap between home and road win rates. Large positive = crowd-dependent, vulnerable on neutral courts.", "-20 to +40", "Computed from home/away records", "Tournament teams play on neutral courts. High home_road_delta = red flag. Negative or near-zero = road warrior, tournament-ready."),

        # ── Flags ─────────────────────────────────────────────────────────────
        ("regression_risk","Flags", "Binary flag: 1 = this team is shooting unsustainably well from 3-point range and is statistically due for a shooting correction.", "0 or 1", "Tournament module", "Teams with regression_risk=1 should be faded slightly in ensemble models. Three-point shooting is the most volatile CBB stat."),
        ("star_danger",    "Flags", "Binary flag: 1 = star player reliance is at a dangerous level AND team has shown fragility when the star underperforms.", "0 or 1", "Tournament module", "Combined star_risk + fragility indicator. More aggressive flag than star_risk alone."),
        ("three_pct_gap",  "Flags", "Recent 3-point percentage (L5) minus season 3-point percentage. Positive = currently running hot from three.", "-8 to +8 pct pts", "Tournament module", ">5 = potential regression candidate. Pairs with regression_risk flag."),

        # ── Four Factors (reference) ──────────────────────────────────────────
        ("efg_pct",        "Four Factors", "Effective Field Goal percentage — accounts for 3-pointers being worth 50% more than 2-pointers. (FGM + 0.5*3PM) / FGA.", "40 – 62%", "Pipeline", "Dean Oliver's #1 factor (weight: 40%). League average ≈ 50.5%. Elite offense: >55%."),
        ("opp_efg_pct",    "Four Factors", "Opponent Effective Field Goal percentage allowed — defensive eFG% suppression.", "40 – 62%", "Pipeline (season opponent avg)", "Lower is better. Elite defense: <46%. Combine with suffocation score for full defensive picture."),
        ("tov_pct",        "Four Factors", "Turnover percentage — turnovers per 100 possessions. Lower is better.", "12 – 25%", "Pipeline", "Oliver's factor #2 (weight: 25%). League average ≈ 18%. Elite: <15%."),
        ("orb_pct",        "Four Factors", "Offensive rebound percentage — share of available offensive rebounds captured.", "20 – 45%", "Pipeline", "Oliver's factor #3 (weight: 20%). High orb_pct = second-chance points machine."),
        ("drb_pct",        "Four Factors", "Defensive rebound percentage — share of available defensive rebounds captured.", "55 – 85%", "Pipeline", "Complement to orb_pct. Elite defense: drb_pct >75%."),
        ("ftr",            "Four Factors", "Free Throw Rate — free throw attempts per field goal attempt. Measures ability to get to the line.", "15 – 50%", "Pipeline", "Oliver's factor #4 (weight: 15%). High ftr + high ft_pct = free point machine."),
        ("three_par",      "Four Factors", "Three-Point Attempt Rate — share of field goal attempts from behind the arc.", "20 – 55%", "Pipeline", "High three_par teams live and die by 3-point shooting variance."),
        ("three_pct",      "Four Factors", "Three-point field goal percentage (season).", "28 – 42%", "Pipeline", "Most volatile major shooting stat. Use three_pct_gap to flag hot-shooting teams."),
        ("ft_pct",         "Four Factors", "Free throw percentage.", "62 – 82%", "Pipeline", "Matters most in close games and tournament play when fouls are more common."),
        ("close_wpct",     "Four Factors", "Win percentage in games decided by ≤5 points.", "0.000 – 1.000", "Pipeline", "Core input to clutch_rating. High close_wpct teams overperform in tournament settings."),

        # ── Rolling Windows ───────────────────────────────────────────────────
        ("net_rtg_l5",     "Rolling Windows", "Net efficiency rating (ortg minus drtg) averaged over last 5 games. Best indicator of current form.", "-20 to +30", "Pipeline", "Primary input to Momentum ensemble model. Recency-weighted."),
        ("net_rtg_l10",    "Rolling Windows", "Net efficiency rating averaged over last 10 games. Medium-term form baseline.", "-15 to +25", "Pipeline", "Used with net_rtg_l5 to compute trend_arrow. More stable than L5, less stable than season avg."),

        # ── Meta ─────────────────────────────────────────────────────────────
        ("updated_at",     "Meta", "Timestamp when this row was last computed.", "ISO datetime", "Computed", ""),
    ]

    # ── Write CSV version ─────────────────────────────────────────────────────
    dict_df = pd.DataFrame(
        COLUMNS,
        columns=["column", "category", "description", "typical_range", "source", "notes"]
    )
    csv_path = output_dir / "cbb_rankings_data_dictionary.csv"
    dict_df.to_csv(csv_path, index=False)
    log.info(f"Data dictionary (CSV) → {csv_path}")

    # ── Write TXT version (formatted for quick human reading) ─────────────────
    txt_lines = [
        "CAGE RANKINGS — DATA DICTIONARY",
        "Composite Adjusted Grade Engine | ESPN CBB Pipeline",
        "=" * 80,
        "",
        "This file describes every column in cbb_rankings.csv.",
        "Generated automatically alongside each rankings run.",
        "",
    ]

    current_cat = None
    for col, cat, desc, rng, src, notes in COLUMNS:
        if cat != current_cat:
            txt_lines += ["", f"{'─' * 80}", f"  {cat.upper()}", f"{'─' * 80}"]
            current_cat = cat
        txt_lines.append(f"\n  {col}")
        txt_lines.append(f"    {desc}")
        txt_lines.append(f"    Range : {rng}")
        txt_lines.append(f"    Source: {src}")
        if notes:
            txt_lines.append(f"    Notes : {notes}")

    txt_lines += [
        "",
        "=" * 80,
        "EQUIVALENCY GUIDE",
        "─" * 80,
        "  CAGE_EM       ↔  KenPom AdjEM  ↔  Torvik's NET-equivalent",
        "  CAGE_O        ↔  KenPom AdjO",
        "  CAGE_D        ↔  KenPom AdjD",
        "  CAGE_T        ↔  KenPom AdjT",
        "  BARTHAG       ↔  Torvik BARTHAG (direct replication)",
        "  WAB           ↔  Torvik WAB (direct replication)",
        "  RESUME_SCORE  ↔  Torvik Resume Composite",
        "  Quad Records  ↔  NCAA/Torvik Quad definitions (our quads use",
        "                    opponent net rating as proxy for NET rank tiers)",
        "",
        "CAGE-ONLY METRICS (no KenPom/Torvik equivalent):",
        "  CAGE_POWER_INDEX, SUFFOCATION, MOMENTUM, CLUTCH_RATING,",
        "  CONSISTENCY_SCORE, FLOOR_EM, CEILING_EM, DNA_SCORE,",
        "  TOURNEY_READINESS, OFF_IDENTITY, STAR_RISK, TREND_ARROW",
        "",
        "QUAD THRESHOLDS (opponent adjusted net rating):",
        "  Quad 1: opp_net ≥ +8.0   (elite — roughly top 75 teams by NET)",
        "  Quad 2: opp_net  0 – 8.0  (above average)",
        "  Quad 3: opp_net -8 – 0    (below average)",
        "  Quad 4: opp_net < -8.0   (weak — bottom tier)",
        "  Note: Official NCAA Quads also factor game location (home/road/neutral).",
        "  Our quads use opponent quality only as a location-agnostic equivalent.",
        "",
        f"Generated: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M %Z')}",
        "=" * 80,
    ]

    txt_path = output_dir / "cbb_rankings_data_dictionary.txt"
    txt_path.write_text("\n".join(txt_lines))
    log.info(f"Data dictionary (TXT) → {txt_path}")

    return csv_path


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

    # Data dictionary — human-readable metric descriptions alongside every output
    write_data_dictionary(output_dir)

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
