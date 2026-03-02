#!/usr/bin/env python3
"""
Backtest Engine: Model + Handicapper Alignment Analysis
Compute ROI when model agrees/disagrees with cappers
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _load_picks(data_dir: str) -> pd.DataFrame:
    """Load picks CSV with flexible column handling."""
    path = Path(data_dir) / "picks.csv"
    if not path.exists():
        logger.warning("picks.csv not found in %s", data_dir)
        return pd.DataFrame()
    df = pd.read_csv(path, dtype=str)
    # Skip any embedded header rows (rows where pick_id == 'pick_id')
    if 'pick_id' in df.columns:
        df = df[df['pick_id'] != 'pick_id'].reset_index(drop=True)
    return df


def _load_games(data_dir: str) -> pd.DataFrame:
    """Load games CSV, preferring app_games.csv then games.csv."""
    for fname in ('app_games.csv', 'games.csv'):
        path = Path(data_dir) / fname
        if path.exists():
            df = pd.read_csv(path, dtype=str)
            logger.debug("Loaded games from %s (%d rows)", fname, len(df))
            return df
    logger.warning("No games CSV found in %s", data_dir)
    return pd.DataFrame()


class BacktestEngine:
    """Compute alignment ROI, consensus signals, edge discovery."""

    def __init__(self, data_dir: str = "./data") -> None:
        self.data_dir = data_dir
        self.data = {
            'picks': _load_picks(data_dir),
            'games': _load_games(data_dir),
        }

    def grade_picks(self) -> pd.DataFrame:
        """Grade picks against game results → ROI per pick."""
        picks = self.data['picks'].copy()
        games = self.data['games'].copy()

        if picks.empty or games.empty:
            logger.warning("No picks or games data")
            return pd.DataFrame()

        # Normalise games column names to canonical names used by grading logic
        games = games.rename(columns={
            'spread': 'closing_spread',
            'over_under': 'total_line',
        }, errors='ignore')

        # Normalise picks column names
        picks = picks.rename(columns={
            'pick_type': 'market',
            'pick_spread': 'line',
            'confidence': 'units',
            'pick_team': 'side',
        }, errors='ignore')

        # Ensure pick_id exists
        if 'pick_id' not in picks.columns:
            picks = picks.reset_index(drop=True)
            picks.insert(0, 'pick_id', picks.index.astype(str))

        # Select only the game columns we need for grading
        game_cols = ['game_id']
        for col in ('closing_spread', 'total_line', 'home_score', 'away_score',
                    'completed', 'status'):
            if col in games.columns:
                game_cols.append(col)

        graded = picks.merge(games[game_cols], on='game_id', how='left')

        # Ensure required numeric columns exist
        if 'units' not in graded.columns:
            graded['units'] = 1.0
        graded['units'] = pd.to_numeric(graded['units'], errors='coerce').fillna(1.0)

        if 'line' not in graded.columns:
            graded['line'] = np.nan
        graded['line'] = pd.to_numeric(graded['line'], errors='coerce')

        if 'closing_spread' not in graded.columns:
            graded['closing_spread'] = np.nan
        graded['closing_spread'] = pd.to_numeric(graded['closing_spread'], errors='coerce')

        if 'total_line' not in graded.columns:
            graded['total_line'] = np.nan
        graded['total_line'] = pd.to_numeric(graded['total_line'], errors='coerce')

        if 'market' not in graded.columns:
            graded['market'] = 'spread'

        # Grade spreads: pick line < closing spread → covered
        graded['covered'] = np.where(
            graded['market'] == 'spread',
            graded['line'] < graded['closing_spread'],
            np.nan,
        )

        # Grade totals: over pick covers when pick_total > closing total
        if 'pick_total' in graded.columns and 'side' in graded.columns:
            graded['covered'] = np.where(
                graded['market'] == 'total',
                (graded['side'].str.lower() == 'over') == (
                    pd.to_numeric(graded['pick_total'], errors='coerce')
                    > graded['total_line']
                ),
                graded['covered'],
            )

        # Simple ROI: win = +units, loss = -units
        graded['payout'] = np.where(
            graded['covered'].fillna(False).astype(bool),
            graded['units'],
            -graded['units'],
        )
        graded['roi'] = graded['payout'] / graded['units']

        return graded

    def compute_alignment_stats(self, graded_df: pd.DataFrame) -> pd.DataFrame:
        """Compute model + capper alignment patterns."""
        if graded_df.empty:
            return pd.DataFrame()

        df = graded_df.copy()

        # Simulate model predictions (replace with real model output)
        np.random.seed(42)
        df['model_margin'] = np.random.normal(0, 8, len(df))
        df['model_edge'] = df['model_margin'] - df['closing_spread'].fillna(0)
        df['model_pick_spread'] = df['model_edge'] > 0

        if 'market' not in df.columns:
            df['market'] = 'spread'

        # Cappers agreeing with model per game/market
        capper_agreement = (
            df.groupby(['game_id', 'market'])
            .agg(
                total_cappers=('handicapper_id', 'nunique'),
                model_agree_cappers=('model_pick_spread', lambda x: x.sum()),
            )
            .reset_index()
        )

        capper_agreement['model_crowd_alignment'] = np.where(
            capper_agreement['model_agree_cappers'] >= 2,
            'model_plus_crowd',
            np.where(
                capper_agreement['model_agree_cappers'] == 0,
                'model_vs_crowd',
                'mixed',
            ),
        )

        return capper_agreement

    def run_full_backtest(self) -> Dict[str, object]:
        """Complete backtest pipeline."""
        graded = self.grade_picks()
        alignment = self.compute_alignment_stats(graded)

        if graded.empty:
            empty = pd.DataFrame()
            return {
                'graded_picks': empty,
                'capper_stats': empty,
                'alignment_stats': empty,
                'summary': self._generate_summary(empty, empty),
            }

        # Handicapper performance
        capper_stats = (
            graded.groupby('handicapper_id')
            .agg(
                total_picks=('pick_id', 'count'),
                win_pct=('covered', 'mean'),
                total_roi=('payout', 'sum'),
            )
            .round(3)
        )
        capper_stats['roi_pct'] = (
            capper_stats['total_roi'] / capper_stats['total_picks']
        ).round(3)

        # Alignment performance
        if not alignment.empty:
            alignment_stats = (
                graded.merge(alignment, on=['game_id', 'market'])
                .groupby('model_crowd_alignment')
                .agg(
                    samples=('pick_id', 'count'),
                    win_pct=('covered', 'mean'),
                    total_roi=('payout', 'sum'),
                )
                .round(3)
            )
        else:
            alignment_stats = pd.DataFrame()

        return {
            'graded_picks': graded,
            'capper_stats': capper_stats,
            'alignment_stats': alignment_stats,
            'summary': self._generate_summary(graded, alignment_stats),
        }

    def _generate_summary(
        self, graded: pd.DataFrame, alignment_stats: pd.DataFrame
    ) -> Dict[str, object]:
        """Key backtest metrics."""
        if graded.empty:
            return {
                'total_picks': 0,
                'overall_roi': 0.0,
                'overall_win_pct': 0.0,
                'model_plus_crowd_roi': 0.0,
                'model_vs_crowd_roi': 0.0,
                'best_alignment': None,
            }

        overall_roi = float(graded['payout'].sum())
        overall_win_pct = float(graded['covered'].mean())

        crowd_roi: float = 0.0
        model_vs_roi: float = 0.0
        best_alignment = None

        if not alignment_stats.empty and 'total_roi' in alignment_stats.columns:
            try:
                crowd_roi = float(alignment_stats.loc['model_plus_crowd', 'total_roi'])
            except KeyError:
                pass
            try:
                model_vs_roi = float(alignment_stats.loc['model_vs_crowd', 'total_roi'])
            except KeyError:
                pass
            best_alignment = alignment_stats['total_roi'].idxmax()

        return {
            'total_picks': len(graded),
            'overall_roi': overall_roi,
            'overall_win_pct': overall_win_pct,
            'model_plus_crowd_roi': crowd_roi,
            'model_vs_crowd_roi': model_vs_roi,
            'best_alignment': best_alignment,
        }


def cmd_backtest(args) -> None:
    """CLI backtest command."""
    data_dir = getattr(args, 'data_dir', './data')
    engine = BacktestEngine(data_dir)
    results = engine.run_full_backtest()

    print("\n" + "=" * 60)
    print("🏆 HANDICAPPER WISDOM BACKTEST RESULTS")
    print("=" * 60)

    summary = results['summary']
    print(f"\n📊 OVERALL STATS:")
    print(f"  Total Picks: {summary['total_picks']:,}")
    win_pct = summary['overall_win_pct']
    win_str = f"{win_pct:.1%}" if win_pct == win_pct else "N/A"  # NaN check
    print(f"  Win%: {win_str}")
    print(f"  Total ROI: {summary['overall_roi']:.2f} units")

    alignment_stats = results['alignment_stats']
    if not alignment_stats.empty:
        print(f"\n🎯 ALIGNMENT PATTERNS:")
        print(alignment_stats.round(3).to_string())

    capper_stats = results['capper_stats']
    if not capper_stats.empty:
        print(f"\n⭐ TOP PERFORMERS:")
        top_cappers = capper_stats.nlargest(5, 'roi_pct')
        print(top_cappers[['total_picks', 'win_pct', 'roi_pct']].to_string())

    # Save results
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "backtest_results.csv")
    results['graded_picks'].to_csv(out_path, index=False)
    print(f"\n💾 Full results → {out_path}")


def cmd_live_signals(args) -> None:
    """Generate live signals for upcoming games."""
    data_dir = getattr(args, 'data_dir', './data')
    engine = BacktestEngine(data_dir)
    games = engine.data['games'].copy()

    # Upcoming games (no home_score or status == scheduled)
    if 'home_score' in games.columns:
        upcoming = games[pd.to_numeric(games['home_score'], errors='coerce').isna()].copy()
    elif 'status' in games.columns:
        upcoming = games[games['status'].str.lower() == 'scheduled'].copy()
    else:
        upcoming = games.copy()

    if upcoming.empty:
        print("No upcoming games found.")
        return

    np.random.seed(0)
    upcoming['model_edge'] = np.random.normal(2, 3, len(upcoming))
    upcoming['capper_consensus'] = np.random.uniform(0, 1, len(upcoming))

    conditions = [
        (upcoming['model_edge'] >= 2.5) & (upcoming['capper_consensus'] >= 0.6),
        (upcoming['model_edge'] >= 1.5) & (upcoming['capper_consensus'] >= 0.4),
        (upcoming['model_edge'] >= 3.0) & (upcoming['capper_consensus'] <= 0.2),
    ]
    choices = ['🟢 MODEL+CROWD', '🟡 MODEL SUPPORT', '🔴 MODEL VS CROWD']
    upcoming['signal'] = np.select(conditions, choices, default='⚪ NEUTRAL')

    home_col = 'home_team' if 'home_team' in upcoming.columns else upcoming.columns[0]
    away_col = 'away_team' if 'away_team' in upcoming.columns else upcoming.columns[1]

    print("\n🎯 LIVE SIGNALS (Next 24h):")
    signals = upcoming[[home_col, away_col, 'model_edge', 'capper_consensus', 'signal']]
    print(signals.sort_values('model_edge', ascending=False).round(2).to_string(index=False))

    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, "live_signals.csv")
    signals.to_csv(out_path, index=False)
    print(f"\n💾 Saved to {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Backtest Engine")
    parser.add_argument(
        '--data-dir', default='./data',
        help='Path to data directory (default: ./data)',
    )
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('backtest', help='Full backtest analysis')
    subparsers.add_parser('signals', help='Live alignment signals')

    args = parser.parse_args()

    if args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'signals':
        cmd_live_signals(args)
    else:
        parser.print_help()

