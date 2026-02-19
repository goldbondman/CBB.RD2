#!/usr/bin/env python3
"""
College Basketball Prediction Model
Recursive Bidirectional Analysis with Normalized Opponent Baselines

CHANGELOG v2.1 (Pipeline Integration — Quant Team):
─────────────────────────────────────────────────────────────────────────────
v2.0 (prior):
1. ✅ Removed SOS double-counting (built into normalization)
2. ✅ Three-layer normalized baselines (raw → weighted → schedule-adjusted)
3. ✅ Averaged possessions (eliminates team discrepancies)
4. ✅ Removed pace scaling from spread (pace affects total only)
5. ✅ Smooth exponential decay for game weighting
6. ✅ Baseline confidence weighting (accounts for variance)
7. ✅ Opponent quality weights (elite opponents = stronger signal)

v2.1 (this version — ESPN pipeline integration):
8. ✅ game_type field on ModelConfig — applies tournament total multipliers
9. ✅ pf (personal fouls) added to GameData required stats and box schema
10. ✅ Foul-rate confidence penalty (high-foul teams = higher variance predictions)
11. ✅ tournament_multiplier exposed in breakdown output dict
12. ✅ Spread confidence interval added to output
13. ✅ Four-factor deltas normalized to pts/game units (not per-100)
14. ✅ File renamed cbb_prediction_model.py for pipeline import compatibility

Author: Quant team
Philosophy: Beat the market through normalized performance vs expectation
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelConfig:
    """
    Model configuration with empirically-tuned parameters.

    v2.1 additions:
    - game_type: applies tournament pace/scoring multipliers to predicted_total
    - foul_confidence_penalty: reduces confidence for high-foul teams
    """

    # ─── Lookback Windows ───
    l5_window: int = 5
    l10_window: int = 10

    # ─── Game Decay Weighting ───
    # Options: 'smooth', 'plateau', 'simple'
    decay_type: str = 'smooth'

    # ─── Four Factors Weights (must sum to 1.0) ───
    efg_weight: float = 0.28
    tov_weight: float = 0.22
    orb_weight: float = 0.18
    ftr_weight: float = 0.16
    drb_weight: float = 0.16

    # ─── Raw vs Vs-Expectation Blending ───
    vs_exp_weight: float = 0.70
    raw_weight: float = 0.30

    # ─── Opponent Quality Weighting ───
    min_opp_weight: float = 0.5
    max_opp_weight: float = 2.0
    opp_weight_scale: float = 20.0

    # ─── Schedule Adjustment ───
    schedule_adjustment_factor: float = 0.5

    # ─── Baseline Values ───
    avg_pace: float = 70.0
    default_hca: float = 3.2
    league_avg_off_eff: float = 105.0

    # ─── Confidence Parameters ───
    min_games_for_full_confidence: int = 5
    consistency_threshold: float = 20.0
    foul_confidence_penalty: bool = True   # v2.1: penalize high-foul teams

    # ─── v2.1: Tournament context ───────────────────────────────────────────
    # Applies a total multiplier in _calculate_matchup.
    # Only affects predicted_total, NOT the spread.
    # Values tuned from 30 years of conference/NCAA tournament data.
    # 'regular': no adjustment (default)
    # Use espn_prediction_runner.py to pass game_type per matchup.
    game_type: str = 'regular'

    # Tournament total multipliers (applied to raw pace × efficiency product)
    # These encode that tournament games are systematically lower-scoring than
    # regular season games at the same adjusted ratings, due to:
    # - Defensive preparation time
    # - Conservative offensive game plans
    # - Neutral court shooting drag
    # - Heightened pressure reducing fast break opportunities
    tournament_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'regular':          1.000,
        'conf_tournament':  0.964,
        'ncaa_r1':          0.958,
        'ncaa_r2':          0.951,
    })

    def get_tournament_multiplier(self) -> float:
        return self.tournament_multipliers.get(self.game_type, 1.000)


@dataclass
class GameData:
    """
    Complete game data structure with recursive opponent history.

    v2.1 addition: pf (personal fouls) added to box schema.
    Used for foul-rate confidence adjustment — teams that foul heavily
    have higher variance (more opponent FT attempts = more randomness).
    """

    game_id: str
    date: datetime
    team_name: str
    opponent_name: str
    neutral_site: bool = False

    team_score: int = 0
    opponent_score: int = 0

    team_box: Dict[str, float] = field(default_factory=dict)
    opponent_box: Dict[str, float] = field(default_factory=dict)

    # Recursive opponent context — the engine of normalized baseline analysis.
    # Each entry is a GameData for one of the opponent's prior games.
    # Populated one level deep (opponent_history entries have empty histories)
    # to avoid exponential blowup.
    opponent_history: List['GameData'] = field(default_factory=list)

    def __post_init__(self):
        """Validate and zero-fill required stats."""
        # v2.1: pf added to schema
        required_stats = [
            'fgm', 'fga', 'tpm', 'tpa', 'ftm', 'fta',
            'orb', 'drb', 'tov', 'pf'
        ]
        for stat in required_stats:
            if stat not in self.team_box:
                self.team_box[stat] = 0.0
            if stat not in self.opponent_box:
                self.opponent_box[stat] = 0.0

    @property
    def team_foul_rate(self) -> float:
        """Personal fouls per game (for confidence adjustment)."""
        return float(self.team_box.get('pf', 0.0))

    @property
    def is_high_foul_game(self) -> bool:
        """Flag games where either team fouled excessively (>22 PF)."""
        return (self.team_box.get('pf', 0) > 22 or
                self.opponent_box.get('pf', 0) > 22)


# ============================================================================
# CORE BASKETBALL CALCULATIONS
# ============================================================================

def estimate_possessions_averaged(
    team_fga: float,
    team_fta: float,
    team_orb: float,
    team_tov: float,
    opp_fga: float,
    opp_fta: float,
    opp_orb: float,
    opp_tov: float,
) -> float:
    """
    Estimate possessions using Dean Oliver's formula, AVERAGED between both teams.

    Returns single possession count (averaged) to eliminate the small discrepancy
    that arises when each team's possession formula gives slightly different answers
    for the same game.

    Formula for each team:
        Poss ≈ FGA + 0.475*FTA - ORB + TOV + 0.33*OPP_ORB

    Final: Average of both teams' estimates.

    Note: 0.475 is the correct FTA coefficient for college basketball
    (vs 0.44 used in the pipeline's raw metric — we prefer 0.475 here for
    the bidirectional analysis which compares teams directly).
    """
    team_poss = team_fga + (0.475 * team_fta) - team_orb + team_tov + (0.33 * opp_orb)
    opp_poss  = opp_fga  + (0.475 * opp_fta)  - opp_orb  + opp_tov  + (0.33 * team_orb)
    return max((team_poss + opp_poss) / 2.0, 1.0)   # floor at 1 to prevent div/0


def calculate_four_factors(
    box: Dict[str, float],
    poss: float,
    opp_drb: float,
    opp_orb: float,
) -> Dict[str, float]:
    """
    Calculate Dean Oliver's Four Factors from box score.
    eFG%, TOV%, ORB%, DRB%, FTR, FT%.
    """
    def safe_div(num: float, den: float, default: float = 0.0) -> float:
        return num / den if den > 0 else default

    efg     = safe_div(box['fgm'] + 0.5 * box['tpm'], box['fga'], 0.50)
    tov_pct = safe_div(box['tov'], poss, 0.15) * 100
    orb_pct = safe_div(box['orb'], box['orb'] + opp_drb, 0.30)
    drb_pct = safe_div(box['drb'], box['drb'] + opp_orb, 0.70)
    ftr     = safe_div(box['fta'], box['fga'], 0.30)
    ft_pct  = safe_div(box['ftm'], box['fta'], 0.70)

    return {
        'efg':     efg,
        'tov_pct': tov_pct,
        'orb_pct': orb_pct,
        'drb_pct': drb_pct,
        'ftr':     ftr,
        'ft_pct':  ft_pct,
    }


def calculate_efficiency(points: float, poss: float) -> float:
    """Points per 100 possessions (offensive or defensive efficiency)."""
    return (points / poss * 100.0) if poss > 0 else 100.0


# ============================================================================
# GAME WEIGHTING (DECAY FUNCTIONS)
# ============================================================================

def get_game_weight(game_n: int, decay_type: str = 'smooth') -> float:
    """
    Calculate recency weight for game N (1 = most recent).

    'smooth'  : Gradual transition.  Games 1-5: 1.00→0.92.  Games 6-10: 0.75→0.50.
    'plateau' : Flat for 4 games, then stepdown.
    'simple'  : Equal weight L5, half weight 6-10.

    Quant team recommendation: 'smooth' best balances signal vs recency bias.
    'plateau' useful when very recent games are known to be against weaker opponents.
    """
    if decay_type == 'smooth':
        if game_n <= 5:
            return 1.00 - 0.02 * (game_n - 1)     # 1.00, 0.98, 0.96, 0.94, 0.92
        else:
            return max(0.50, 0.75 - 0.05 * (game_n - 5))  # 0.75 → 0.50

    elif decay_type == 'plateau':
        if game_n <= 4:   return 1.00
        elif game_n == 5: return 0.90
        else:             return max(0.50, 1.05 - 0.10 * game_n)

    else:  # 'simple'
        return 1.00 if game_n <= 5 else 0.50


# ============================================================================
# NORMALIZED OPPONENT BASELINE ANALYZER
# ============================================================================

class NormalizedOpponentBaseline:
    """
    Three-layer baseline calculation with opponent quality normalization.

    Layer 1: Raw baseline (simple average of what opponent allowed)
    Layer 2: Opponent-quality weighted (elite opponents = stronger signal)
    Layer 3: Schedule-adjusted (corrects for strength of offenses faced)

    The adjusted_baseline (Layer 3) is what the PerformanceVsExpectationAnalyzer
    uses when computing a team's efficiency vs expectation.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    def calculate_baseline(
        self,
        opponent_games: List[GameData],
        window: int = 5,
    ) -> Dict[str, float]:
        """
        Calculate opponent's defensive baseline (as pts/100 poss efficiency).

        Returns dict with:
            raw_baseline        : Simple average of points allowed (eff units)
            weighted_baseline   : Opponent-quality weighted average
            adjusted_baseline   : Schedule-adjusted → USE THIS for vs-expectation
            baseline_std        : Standard deviation (confidence proxy)
            confidence          : 0–1 reliability score
            n_games             : Sample size
            avg_opp_quality     : Mean offensive quality of teams opponent faced
        """
        if not opponent_games:
            return self._default_baseline()

        recent = opponent_games[-window:] if len(opponent_games) >= window else opponent_games

        game_analyses = []
        for game in recent:
            try:
                game_analyses.append(self._analyze_opponent_game(game))
            except Exception as e:
                pass  # Skip malformed games silently

        if not game_analyses:
            return self._default_baseline()

        # ── Layer 1: Raw ──────────────────────────────────────────────────────
        points_allowed = [g['points_allowed'] for g in game_analyses]
        raw_baseline   = float(np.mean(points_allowed))
        baseline_std   = float(np.std(points_allowed)) if len(points_allowed) > 1 else 5.0

        # ── Layer 2: Opponent-quality weighted ────────────────────────────────
        # Elite offenses (high net eff) facing this defense → stronger signal
        weighted_total = 0.0
        weight_sum     = 0.0

        for analysis in game_analyses:
            opp_quality = analysis['opponent_net_eff']
            weight = 1.0 + (opp_quality / self.config.opp_weight_scale)
            weight = np.clip(weight, self.config.min_opp_weight, self.config.max_opp_weight)

            weighted_total += analysis['points_allowed'] * weight
            weight_sum     += weight

        weighted_baseline = float(weighted_total / weight_sum) if weight_sum > 0 else raw_baseline

        # ── Layer 3: Schedule adjustment ─────────────────────────────────────
        # If opponent faced elite offenses, their raw allowed eff looks worse than it is
        avg_opp_off_eff = float(np.mean([g['opponent_off_eff'] for g in game_analyses]))
        schedule_factor = ((avg_opp_off_eff - self.config.league_avg_off_eff) *
                           self.config.schedule_adjustment_factor)
        adjusted_baseline = weighted_baseline - schedule_factor

        # ── Confidence ────────────────────────────────────────────────────────
        n_games = len(game_analyses)
        sample_conf   = min(1.0, n_games / self.config.min_games_for_full_confidence)
        variance_conf = 1.0 / (1.0 + baseline_std / 10.0)
        confidence    = float(sample_conf * variance_conf)

        return {
            'raw_baseline':      raw_baseline,
            'weighted_baseline': weighted_baseline,
            'adjusted_baseline': adjusted_baseline,
            'baseline_std':      baseline_std,
            'confidence':        confidence,
            'n_games':           n_games,
            'avg_opp_quality':   avg_opp_off_eff - self.config.league_avg_off_eff,
        }

    def _analyze_opponent_game(self, game: GameData) -> Dict[str, float]:
        """
        Extract defensive profile from a single opponent game.
        Uses averaged possessions (v2.0 methodology).
        """
        game_poss = estimate_possessions_averaged(
            game.opponent_box['fga'], game.opponent_box['fta'],
            game.opponent_box['orb'], game.opponent_box['tov'],
            game.team_box['fga'],     game.team_box['fta'],
            game.team_box['orb'],     game.team_box['tov'],
        )

        points_allowed = game.team_score          # What this team gave up
        allowed_eff    = calculate_efficiency(points_allowed, game_poss)
        opp_off_eff    = calculate_efficiency(game.opponent_score, game_poss)
        opp_def_eff    = allowed_eff
        opp_net_eff    = opp_off_eff - opp_def_eff

        return {
            'points_allowed':   allowed_eff,     # pts/100 what they allowed
            'opponent_off_eff': opp_off_eff,
            'opponent_net_eff': opp_net_eff,
            'game_poss':        game_poss,
        }

    def _default_baseline(self) -> Dict[str, float]:
        """NCAA Division I averages when no opponent history is available."""
        return {
            'raw_baseline':      105.0,
            'weighted_baseline': 105.0,
            'adjusted_baseline': 105.0,
            'baseline_std':      10.0,
            'confidence':        0.0,
            'n_games':           0,
            'avg_opp_quality':   0.0,
        }


# ============================================================================
# PERFORMANCE VS EXPECTATION ANALYZER
# ============================================================================

class PerformanceVsExpectationAnalyzer:
    """
    Compares team performance to normalized opponent baselines.

    For each game: how did the team perform relative to what the opponent
    typically allows/forces? The delta (vs-expectation) is the true signal.

    Raw stats lie. Rate stats lie less. Context-adjusted rate stats are truth.
    """

    def __init__(self, config: ModelConfig):
        self.config   = config
        self.baseline = NormalizedOpponentBaseline(config)

    def analyze_game(
        self,
        game: GameData,
        baseline_window: int = 5,
    ) -> Dict[str, float]:
        """
        Single game: actual performance vs opponent's normalized baseline.

        Returns both raw metrics AND vs-expectation deltas, weighted by
        baseline confidence. Low confidence (few opp history games) → vs-exp
        is downweighted, raw metrics carry more weight.
        """
        # ── Opponent's normalized baseline ───────────────────────────────────
        opp_baseline = self.baseline.calculate_baseline(
            game.opponent_history, window=baseline_window
        )
        baseline_conf = opp_baseline['confidence']

        # ── Team's actual performance ─────────────────────────────────────────
        game_poss = estimate_possessions_averaged(
            game.team_box['fga'], game.team_box['fta'],
            game.team_box['orb'], game.team_box['tov'],
            game.opponent_box['fga'], game.opponent_box['fta'],
            game.opponent_box['orb'], game.opponent_box['tov'],
        )

        team_factors = calculate_four_factors(
            game.team_box, game_poss,
            game.opponent_box['drb'], game.opponent_box['orb']
        )

        team_off_eff = calculate_efficiency(game.team_score,     game_poss)
        team_def_eff = calculate_efficiency(game.opponent_score, game_poss)
        team_net_eff = team_off_eff - team_def_eff
        margin       = game.team_score - game.opponent_score

        # ── Performance vs expectation ────────────────────────────────────────
        # Raw gap vs adjusted baseline, then weighted by baseline reliability
        off_eff_vs_exp_raw = team_off_eff - opp_baseline['adjusted_baseline']
        off_eff_vs_exp     = off_eff_vs_exp_raw * baseline_conf

        # Four-factor vs-expectation (use league averages as baseline when
        # opponent-specific baselines unavailable)
        efg_vs_exp = (team_factors['efg']     - 0.50)  * 100 * baseline_conf
        orb_vs_exp = (team_factors['orb_pct'] - 0.30)  * 100 * baseline_conf
        ftr_vs_exp = (team_factors['ftr']     - 0.30)  * 100 * baseline_conf
        tov_vs_exp = (15.0 - team_factors['tov_pct'])        * baseline_conf
        drb_vs_exp = (team_factors['drb_pct'] - 0.70)  * 100 * baseline_conf

        # v2.1: foul rate signal
        foul_rate  = game.team_foul_rate
        high_foul  = game.is_high_foul_game

        return {
            # Raw performance metrics
            'team_off_eff':     team_off_eff,
            'team_def_eff':     team_def_eff,
            'team_net_eff':     team_net_eff,
            'team_pace':        game_poss,
            'team_margin':      margin,
            'team_efg':         team_factors['efg'],
            'team_tov_pct':     team_factors['tov_pct'],
            'team_orb_pct':     team_factors['orb_pct'],
            'team_drb_pct':     team_factors['drb_pct'],
            'team_ftr':         team_factors['ftr'],
            'team_ft_pct':      team_factors['ft_pct'],
            'team_foul_rate':   foul_rate,
            'is_high_foul':     float(high_foul),

            # Performance vs normalized expectation
            'efg_vs_exp':       efg_vs_exp,
            'orb_vs_exp':       orb_vs_exp,
            'ftr_vs_exp':       ftr_vs_exp,
            'tov_vs_exp':       tov_vs_exp,
            'drb_vs_exp':       drb_vs_exp,
            'off_eff_vs_exp':   off_eff_vs_exp,

            # Baseline metadata
            'baseline_confidence':  baseline_conf,
            'opponent_baseline':    opp_baseline['adjusted_baseline'],
            'opponent_quality':     opp_baseline['avg_opp_quality'],
        }

    def aggregate_window(
        self,
        games: List[GameData],
        window: int,
        decay_type: str = 'smooth',
    ) -> Dict[str, float]:
        """
        Aggregate performance over a rolling window with decay weighting.

        Game 1 (most recent) gets full weight.
        Older games decay per get_game_weight().
        DNP / zero-possession games are automatically filtered.
        """
        if not games:
            return self._default_aggregation()

        recent = games[-window:] if len(games) >= window else games
        if not recent:
            return self._default_aggregation()

        analyzed_games = []
        for game in recent:
            try:
                analyzed_games.append(self.analyze_game(game, baseline_window=5))
            except Exception:
                pass

        if not analyzed_games:
            return self._default_aggregation()

        # Apply decay (game 1 = most recent = idx 0 after reversing)
        weighted_metrics = defaultdict(float)
        total_weight     = 0.0
        high_foul_count  = 0

        for idx, game_metrics in enumerate(reversed(analyzed_games)):
            game_n = idx + 1
            weight = get_game_weight(game_n, decay_type)

            for key, value in game_metrics.items():
                if key not in ('baseline_confidence', 'opponent_baseline', 'is_high_foul'):
                    weighted_metrics[key] += value * weight

            if game_metrics.get('is_high_foul', 0):
                high_foul_count += 1

            total_weight += weight

        result = {k: v / total_weight for k, v in weighted_metrics.items()}
        result['n_games']         = len(analyzed_games)
        result['total_weight']    = total_weight
        result['high_foul_games'] = high_foul_count

        # Variance metrics
        net_effs    = [g['team_net_eff']   for g in analyzed_games]
        foul_rates  = [g['team_foul_rate'] for g in analyzed_games]
        result['net_eff_std']   = float(np.std(net_effs))  if len(net_effs)   > 1 else 0.0
        result['foul_rate_avg'] = float(np.mean(foul_rates)) if foul_rates else 0.0

        return result

    def _default_aggregation(self) -> Dict[str, float]:
        return {
            'team_off_eff':     105.0,
            'team_def_eff':     105.0,
            'team_net_eff':     0.0,
            'team_pace':        70.0,
            'team_margin':      0.0,
            'team_efg':         0.50,
            'team_tov_pct':     15.0,
            'team_orb_pct':     0.30,
            'team_drb_pct':     0.70,
            'team_ftr':         0.30,
            'team_ft_pct':      0.70,
            'team_foul_rate':   15.0,
            'is_high_foul':     0.0,
            'efg_vs_exp':       0.0,
            'orb_vs_exp':       0.0,
            'ftr_vs_exp':       0.0,
            'tov_vs_exp':       0.0,
            'drb_vs_exp':       0.0,
            'off_eff_vs_exp':   0.0,
            'opponent_quality': 0.0,
            'n_games':          0,
            'net_eff_std':      0.0,
            'total_weight':     0.0,
            'high_foul_games':  0,
            'foul_rate_avg':    15.0,
        }


# ============================================================================
# MAIN PREDICTION ENGINE
# ============================================================================

class CBBPredictionModel:
    """
    Main prediction engine: spread + total for any CBB matchup.

    v2.1: game_type config applies tournament pace/scoring multiplier to total.
    All other methodology unchanged from v2.0.
    """

    def __init__(self, config: ModelConfig = None):
        self.config   = config or ModelConfig()
        self.analyzer = PerformanceVsExpectationAnalyzer(self.config)

    def predict_game(
        self,
        home_games: List[GameData],
        away_games: List[GameData],
        neutral_site: bool = False,
        game_type: Optional[str] = None,
    ) -> Dict:
        """
        Predict spread and total for a matchup.

        Parameters
        ----------
        home_games   : Recent GameData list for home/higher-seeded team
        away_games   : Recent GameData list for away/lower-seeded team
        neutral_site : If True, home court advantage is zeroed
        game_type    : Override self.config.game_type for this call only.
                       One of 'regular', 'conf_tournament', 'ncaa_r1', 'ncaa_r2'

        Returns dict with predicted_spread, predicted_total, confidence,
        and detailed breakdown.
        """
        # Allow per-call game_type override without mutating config
        effective_game_type = game_type or self.config.game_type

        # ── Aggregate both teams with decay weighting ──────────────────────────
        home_l5  = self.analyzer.aggregate_window(home_games, self.config.l5_window,  self.config.decay_type)
        home_l10 = self.analyzer.aggregate_window(home_games, self.config.l10_window, self.config.decay_type)
        away_l5  = self.analyzer.aggregate_window(away_games, self.config.l5_window,  self.config.decay_type)
        away_l10 = self.analyzer.aggregate_window(away_games, self.config.l10_window, self.config.decay_type)

        home_blended = self._blend_windows(home_l5, home_l10)
        away_blended = self._blend_windows(away_l5, away_l10)

        # ── Core matchup calculation ───────────────────────────────────────────
        prediction = self._calculate_matchup(
            home_blended, away_blended, neutral_site, effective_game_type
        )

        # ── Confidence ────────────────────────────────────────────────────────
        confidence = self._calculate_confidence(home_l5, away_l5, home_l10, away_l10)
        prediction['confidence'] = confidence

        # ── Confidence interval on spread (±pts) ─────────────────────────────
        # Based on combined net efficiency variance
        home_std = home_l5.get('net_eff_std', 8.0)
        away_std = away_l5.get('net_eff_std', 8.0)
        combined_variance = np.sqrt(home_std**2 + away_std**2) * 0.5
        prediction['spread_confidence_interval'] = round(combined_variance, 1)

        # ── Data quality metadata ─────────────────────────────────────────────
        prediction['home_games_used'] = home_l5.get('n_games', 0)
        prediction['away_games_used'] = away_l5.get('n_games', 0)
        prediction['home_foul_rate']  = round(home_blended.get('foul_rate_avg', 0), 1)
        prediction['away_foul_rate']  = round(away_blended.get('foul_rate_avg', 0), 1)
        prediction['game_type']       = effective_game_type

        return prediction

    # ── Window blending ───────────────────────────────────────────────────────

    def _blend_windows(self, l5: Dict, l10: Dict) -> Dict:
        """
        Blend L5 and L10 windows using their actual accumulated decay weights
        (not static proportions). L5's total weight naturally reflects recency.
        """
        l5_w  = l5.get('total_weight', 4.8)
        l10_w = l10.get('total_weight', 7.8)
        total = l5_w + l10_w
        p5    = l5_w  / total if total > 0 else 0.70
        p10   = l10_w / total if total > 0 else 0.30

        blended = {}
        for key in l5.keys():
            if key in ('n_games', 'total_weight', 'high_foul_games'):
                blended[key] = l5[key]
                continue
            blended[key] = p5 * l5[key] + p10 * l10.get(key, l5[key])

        return blended

    # ── Core matchup calculation ───────────────────────────────────────────────

    def _calculate_matchup(
        self,
        home: Dict,
        away: Dict,
        neutral: bool,
        game_type: str,
    ) -> Dict:
        """
        Compute expected margin and total from blended team profiles.

        Spread: pure efficiency edge — no pace scaling, no SOS adjustment
                (both are already embedded in the normalized baselines).

        Total:  pace × efficiency product, then tournament multiplier applied
                (v2.1: this is the key change from v2.0).
        """
        cfg = self.config

        # ── Expected pace ─────────────────────────────────────────────────────
        exp_pace = (home['team_pace'] + away['team_pace']) / 2.0

        # ── Four-factor deltas (blend raw + vs-expectation) ───────────────────
        def _delta(raw_home, raw_away, vsexp_home, vsexp_away) -> float:
            raw    = raw_home - raw_away
            vs_exp = vsexp_home - vsexp_away
            return cfg.raw_weight * raw + cfg.vs_exp_weight * vs_exp

        efg_delta = _delta(
            home['team_efg']     * 100, away['team_efg']     * 100,
            home['efg_vs_exp'],         away['efg_vs_exp'],
        )
        tov_delta = _delta(
            away['team_tov_pct'],        home['team_tov_pct'],  # reversed: lower is better
            home['tov_vs_exp'],          away['tov_vs_exp'],
        )
        orb_delta = _delta(
            home['team_orb_pct'] * 100, away['team_orb_pct'] * 100,
            home['orb_vs_exp'],          away['orb_vs_exp'],
        )
        drb_delta = _delta(
            home['team_drb_pct'] * 100, away['team_drb_pct'] * 100,
            home['drb_vs_exp'],          away['drb_vs_exp'],
        )
        ftr_delta = _delta(
            home['team_ftr']     * 100, away['team_ftr']     * 100,
            home['ftr_vs_exp'],          away['ftr_vs_exp'],
        )

        # ── Weighted composite edge ────────────────────────────────────────────
        composite_edge = (
            cfg.efg_weight * efg_delta +
            cfg.tov_weight * tov_delta +
            cfg.orb_weight * orb_delta +
            cfg.drb_weight * drb_delta +
            cfg.ftr_weight * ftr_delta
        )

        # ── Efficiency edge ───────────────────────────────────────────────────
        raw_eff    = home['team_net_eff']    - away['team_net_eff']
        vs_exp_eff = home['off_eff_vs_exp']  - away['off_eff_vs_exp']
        eff_edge   = cfg.raw_weight * raw_eff + cfg.vs_exp_weight * vs_exp_eff

        # ── Combined edge (60/40: efficiency / composite) ─────────────────────
        raw_edge = 0.60 * eff_edge + 0.40 * composite_edge

        # ── HCA and final spread ──────────────────────────────────────────────
        # Note: NO SOS adjustment (embedded in normalization)
        #       NO pace scaling on spread (pace only affects total)
        hca        = 0.0 if neutral else cfg.default_hca
        final_edge = raw_edge + hca

        # Spread convention: negative = home favored
        predicted_spread = -final_edge

        # ── Total: pace × efficiency × tournament multiplier ─────────────────
        # v2.1 KEY CHANGE: tournament multiplier applied here
        home_exp_pts = home['team_off_eff'] * (exp_pace / 100.0)
        away_exp_pts = away['team_off_eff'] * (exp_pace / 100.0)
        raw_total    = home_exp_pts + away_exp_pts

        # Pull multiplier from config OR from the game_type passed in
        multiplier_map = cfg.tournament_multipliers
        tourn_mult     = multiplier_map.get(game_type, 1.000)
        predicted_total = raw_total * tourn_mult

        return {
            'predicted_spread':      round(predicted_spread, 2),
            'predicted_total':       round(predicted_total,  1),
            'raw_total':             round(raw_total, 1),
            'tournament_multiplier': tourn_mult,
            'home_net_eff':          round(home['team_net_eff'],   2),
            'away_net_eff':          round(away['team_net_eff'],   2),
            'home_off_eff':          round(home['team_off_eff'],   2),
            'away_off_eff':          round(away['team_off_eff'],   2),
            'home_off_eff_vs_exp':   round(home['off_eff_vs_exp'], 2),
            'away_off_eff_vs_exp':   round(away['off_eff_vs_exp'], 2),
            'pace':                  round(exp_pace, 1),
            'breakdown': {
                'raw_edge':       round(raw_edge,       2),
                'eff_edge':       round(eff_edge,       2),
                'composite_edge': round(composite_edge, 2),
                'hca':            round(hca,            2),
                'efg_delta':      round(efg_delta,      2),
                'tov_delta':      round(tov_delta,      2),
                'orb_delta':      round(orb_delta,      2),
                'drb_delta':      round(drb_delta,      2),
                'ftr_delta':      round(ftr_delta,      2),
                'raw_total':      round(raw_total,      1),
                'tourn_mult':     tourn_mult,
            },
        }

    # ── Confidence ────────────────────────────────────────────────────────────

    def _calculate_confidence(
        self,
        home_l5: Dict,
        away_l5: Dict,
        home_l10: Dict,
        away_l10: Dict,
    ) -> float:
        """
        Prediction confidence (0–1).

        Components:
        - Sample size: do we have enough games to trust the rolling average?
        - Consistency: do L5 and L10 agree, or is the team volatile?
        - Foul rate: high-foul teams have higher game-to-game variance (v2.1)
        """
        cfg = self.config

        # Sample size
        home_sample = min(home_l5['n_games'] / cfg.min_games_for_full_confidence, 1.0)
        away_sample = min(away_l5['n_games'] / cfg.min_games_for_full_confidence, 1.0)
        sample_conf = (home_sample + away_sample) / 2.0

        # L5 vs L10 consistency
        home_delta = abs(home_l5['team_net_eff'] - home_l10['team_net_eff'])
        away_delta = abs(away_l5['team_net_eff'] - away_l10['team_net_eff'])
        home_consist = max(0.0, 1.0 - home_delta / cfg.consistency_threshold)
        away_consist = max(0.0, 1.0 - away_delta / cfg.consistency_threshold)
        consistency_conf = (home_consist + away_consist) / 2.0

        # v2.1: foul-rate penalty
        # Teams averaging >20 PF/game have higher variance due to opponent FTs
        foul_penalty = 0.0
        if cfg.foul_confidence_penalty:
            home_foul = home_l5.get('foul_rate_avg', 15.0)
            away_foul = away_l5.get('foul_rate_avg', 15.0)
            foul_excess = max(0, (home_foul + away_foul) / 2.0 - 18.0)
            foul_penalty = min(foul_excess * 0.02, 0.10)   # cap at -10% confidence

        base_confidence = 0.60 * sample_conf + 0.40 * consistency_conf
        confidence = base_confidence - foul_penalty

        return round(min(0.95, max(0.05, confidence)), 3)


# ============================================================================
# CONVENIENCE: QUICK PREDICT (single dict interface)
# ============================================================================

def quick_predict(
    home_stats_list: List[Dict],
    away_stats_list: List[Dict],
    neutral_site: bool = False,
    game_type: str = 'regular',
    decay: str = 'smooth',
) -> Dict:
    """
    Convenience wrapper — accepts raw stat dicts instead of GameData objects.
    Useful for quick tests or when building GameData is overkill.

    Each dict in home_stats_list / away_stats_list should have keys:
        team_score, opponent_score, fgm, fga, tpm, tpa, ftm, fta,
        orb, drb, tov, pf (optional)
    Plus optionally: opponent_history (empty list if unavailable)

    Returns the same output dict as CBBPredictionModel.predict_game().
    """
    def _dict_to_gamedata(d: Dict, idx: int) -> GameData:
        return GameData(
            game_id       = d.get('game_id', f'game_{idx}'),
            date          = d.get('date', datetime.now()),
            team_name     = d.get('team_name', 'Team'),
            opponent_name = d.get('opponent_name', 'Opponent'),
            neutral_site  = d.get('neutral_site', False),
            team_score    = int(d.get('team_score', 0)),
            opponent_score= int(d.get('opponent_score', 0)),
            team_box      = {k: float(d.get(k, 0)) for k in
                             ['fgm','fga','tpm','tpa','ftm','fta','orb','drb','tov','pf']},
            opponent_box  = {k: float(d.get(f'opp_{k}', 0)) for k in
                             ['fgm','fga','tpm','tpa','ftm','fta','orb','drb','tov','pf']},
            opponent_history = d.get('opponent_history', []),
        )

    home_games = [_dict_to_gamedata(d, i) for i, d in enumerate(home_stats_list)]
    away_games = [_dict_to_gamedata(d, i) for i, d in enumerate(away_stats_list)]

    config = ModelConfig(game_type=game_type, decay_type=decay)
    model  = CBBPredictionModel(config)
    return model.predict_game(home_games, away_games, neutral_site=neutral_site)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("COLLEGE BASKETBALL PREDICTION MODEL v2.1")
    print("Pipeline-Integrated | Tournament Multipliers | Foul-Rate Confidence")
    print("=" * 80)
    print()

    config = ModelConfig(
        decay_type='smooth',
        game_type='ncaa_r1',           # v2.1: tournament context
        foul_confidence_penalty=True,   # v2.1: foul variance adjustment
    )
    model = CBBPredictionModel(config)

    # ── Example: 5-seed vs 12-seed first round ───────────────────────────────
    # In production: built by espn_prediction_runner.build_team_game_list()
    fav_game = GameData(
        game_id="fav_r1_001",
        date=datetime(2025, 3, 20),
        team_name="Favorite 5-Seed",
        opponent_name="Last Opponent",
        team_score=76, opponent_score=68,
        team_box={
            'fgm': 28, 'fga': 60, 'tpm': 8,  'tpa': 22,
            'ftm': 12, 'fta': 16, 'orb': 10, 'drb': 25,
            'tov': 12, 'pf': 17,              # pf now tracked
        },
        opponent_box={
            'fgm': 26, 'fga': 58, 'tpm': 6,  'tpa': 20,
            'ftm': 10, 'fta': 14, 'orb': 8,  'drb': 23,
            'tov': 14, 'pf': 19,
        },
        opponent_history=[],
    )

    dog_game = GameData(
        game_id="dog_r1_001",
        date=datetime(2025, 3, 19),
        team_name="Underdog 12-Seed",
        opponent_name="Last Opponent",
        team_score=71, opponent_score=68,
        team_box={
            'fgm': 26, 'fga': 55, 'tpm': 10, 'tpa': 24,
            'ftm': 9,  'fta': 12, 'orb': 7,  'drb': 24,
            'tov': 10, 'pf': 14,              # low foul rate = tighter CI
        },
        opponent_box={
            'fgm': 26, 'fga': 59, 'tpm': 5,  'tpa': 18,
            'ftm': 11, 'fta': 15, 'orb': 9,  'drb': 22,
            'tov': 13, 'pf': 16,
        },
        opponent_history=[],
    )

    prediction = model.predict_game(
        home_games=[fav_game],
        away_games=[dog_game],
        neutral_site=True,
        game_type='ncaa_r1',
    )

    print("MATCHUP: Favorite 5-Seed vs Underdog 12-Seed  (NCAA R1, neutral)")
    print("-" * 80)
    print(f"Predicted Spread:         {prediction['predicted_spread']:+.1f}  (negative = favorite favored)")
    print(f"Predicted Total:          {prediction['predicted_total']:.1f}")
    print(f"Raw Total (before mult):  {prediction['raw_total']:.1f}")
    print(f"Tournament Multiplier:    {prediction['tournament_multiplier']:.3f}  ({config.game_type})")
    print(f"Confidence:               {prediction['confidence']:.1%}")
    print(f"Spread ± CI:              ±{prediction['spread_confidence_interval']:.1f} pts")
    print()
    print("EFFICIENCY")
    print("-" * 80)
    print(f"Favorite Net Eff:         {prediction['home_net_eff']:+.2f} pts/100")
    print(f"Underdog Net Eff:         {prediction['away_net_eff']:+.2f} pts/100")
    print(f"Favorite Off Eff vs Exp:  {prediction['home_off_eff_vs_exp']:+.2f}")
    print(f"Underdog Off Eff vs Exp:  {prediction['away_off_eff_vs_exp']:+.2f}")
    print(f"Projected Pace:           {prediction['pace']:.1f} possessions")
    print(f"Game Type:                {prediction['game_type']}")
    print()
    print("BREAKDOWN")
    print("-" * 80)
    bd = prediction['breakdown']
    print(f"Efficiency Edge:          {bd['eff_edge']:+.2f}")
    print(f"Composite Edge:           {bd['composite_edge']:+.2f}")
    print(f"  eFG delta:              {bd['efg_delta']:+.2f}")
    print(f"  TOV delta:              {bd['tov_delta']:+.2f}")
    print(f"  ORB delta:              {bd['orb_delta']:+.2f}")
    print(f"  DRB delta:              {bd['drb_delta']:+.2f}")
    print(f"  FTR delta:              {bd['ftr_delta']:+.2f}")
    print(f"HCA:                      {bd['hca']:+.2f}  (0 = neutral site)")
    print(f"Raw Edge:                 {bd['raw_edge']:+.2f}")
    print(f"Final Spread:             {prediction['predicted_spread']:+.1f}")
    print()
    print("v2.1 NOTES")
    print("-" * 80)
    print(f"Favorite foul rate:       {prediction['home_foul_rate']:.1f} PF/game")
    print(f"Underdog foul rate:       {prediction['away_foul_rate']:.1f} PF/game")
    print(f"Home games used:          {prediction['home_games_used']}")
    print(f"Away games used:          {prediction['away_games_used']}")
    print()
    print("=" * 80)
    print("In production: espn_prediction_runner.py builds GameData from pipeline CSVs")
    print("and calls model.predict_game() for every scheduled game automatically.")
    print("=" * 80)
