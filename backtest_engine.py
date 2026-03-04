"""
Backtest engine for the CBB handicapper picks-tracking application.

Grades picks against game results and computes handicapper performance metrics.
Also generates live signals for upcoming/ungraded games.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np


class BacktestEngine:
    """Grade picks and generate live signals."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def _grade_spread(self, pick_line: float, actual_margin: float, side: str) -> str:
        """Return 'win', 'loss', or 'push' for a spread pick.

        pick_line is negative for the favourite (e.g. -4.5) and positive for
        the dog (e.g. +4.5).  actual_margin = home_score - away_score.
        """
        if side == 'home':
            result = actual_margin + pick_line
        else:
            result = -actual_margin + pick_line
        if result > 0:
            return 'win'
        elif result < 0:
            return 'loss'
        return 'push'

    def _grade_total(self, line: float, home_score: float, away_score: float, side: str) -> str:
        total = home_score + away_score
        if total > line:
            return 'win' if side in ('over', 'total') else 'loss'
        elif total < line:
            return 'loss' if side in ('over', 'total') else 'win'
        return 'push'

    # ------------------------------------------------------------------
    # Public commands
    # ------------------------------------------------------------------

    def cmd_backtest(self, args) -> pd.DataFrame:
        """Grade all picks and write backtest_results.csv."""
        picks = self._load('picks.csv')
        games = self._load('games.csv')
        handicappers = self._load('handicappers.csv')

        if picks.empty or games.empty:
            print("⚠️  Insufficient data for backtest (picks or games missing)")
            return pd.DataFrame()

        # Merge picks with games
        merged = picks.merge(games, on='game_id', how='left')

        results = []
        for _, row in merged.iterrows():
            home_score = row.get('home_score')
            away_score = row.get('away_score')

            # Skip games without results yet
            if pd.isna(home_score) or pd.isna(away_score):
                continue

            home_score = float(home_score)
            away_score = float(away_score)
            actual_margin = home_score - away_score

            market = str(row.get('market', '')).lower()
            side = str(row.get('side', '')).lower()
            line = row.get('line')
            units = float(row.get('units', 1.0))

            grade = None
            if market == 'spread' and not pd.isna(line):
                grade = self._grade_spread(float(line), actual_margin, side)
            elif market == 'total' and not pd.isna(line):
                grade = self._grade_total(float(line), home_score, away_score, side)
            elif market == 'moneyline':
                if side == 'home':
                    grade = 'win' if home_score > away_score else 'loss'
                else:
                    grade = 'win' if away_score > home_score else 'loss'

            if grade is None:
                continue

            unit_result = units if grade == 'win' else (-units if grade == 'loss' else 0.0)

            results.append({
                'pick_id': row.get('pick_id'),
                'handicapper_id': row.get('handicapper_id'),
                'game_id': row.get('game_id'),
                'market': market,
                'line': line,
                'units': units,
                'grade': grade,
                'unit_result': unit_result,
            })

        if not results:
            print("⚠️  No gradable picks found")
            return pd.DataFrame()

        results_df = pd.DataFrame(results)

        # Print summary
        total = len(results_df)
        wins = (results_df['grade'] == 'win').sum()
        win_pct = wins / total if total > 0 else 0
        total_units = results_df['unit_result'].sum()

        print(f"\n🏆 HANDICAPPER WISDOM BACKTEST RESULTS")
        print(f"📊 OVERALL STATS: {total} picks, {win_pct:.1%} win, {total_units:+.1f} units")

        # Per-handicapper breakdown
        if not handicappers.empty and 'handicapper_id' in handicappers.columns:
            for hid, grp in results_df.groupby('handicapper_id'):
                h_row = handicappers[handicappers['handicapper_id'] == hid]
                handle = h_row['handle'].values[0] if not h_row.empty and 'handle' in h_row.columns else f"H{hid}"
                h_wins = (grp['grade'] == 'win').sum()
                h_total = len(grp)
                h_units = grp['unit_result'].sum()
                h_win_pct = h_wins / h_total if h_total > 0 else 0
                print(f"  {handle}: {h_total} picks, {h_win_pct:.1%} win, {h_units:+.1f}u")

        # Save results
        out_path = self.data_dir / 'backtest_results.csv'
        results_df.to_csv(out_path, index=False)

        return results_df

    def cmd_live_signals(self, args) -> pd.DataFrame:
        """Print live signals for ungraded picks with line edge."""
        picks = self._load('picks.csv')
        games = self._load('games.csv')

        if picks.empty or games.empty:
            print("⚠️  No data for live signals")
            return pd.DataFrame()

        merged = picks.merge(games, on='game_id', how='left')

        # Only picks for games without final scores
        live = merged[merged['home_score'].isna() | merged['away_score'].isna()]

        if live.empty:
            print("ℹ️  No live/upcoming games found")
            return pd.DataFrame()

        print("\n🎯 LIVE SIGNALS:")
        signals = []
        for _, row in live.iterrows():
            home = row.get('home_team', '?')
            away = row.get('away_team', '?')
            market = row.get('market', '')
            line = row.get('line')
            units = row.get('units', 1.0)
            closing = row.get('closing_spread')

            edge = None
            if not pd.isna(line) and not pd.isna(closing):
                edge = abs(float(line) - float(closing))

            label = '🟢 LIVE' if edge and edge > 1.0 else '⚪ MONITOR'
            print(f"  {away} @ {home} | {market} {line} ({units}u) | {label}")

            signals.append({
                'game': f"{away} @ {home}",
                'market': market,
                'line': line,
                'units': units,
                'edge': edge,
                'signal': label,
            })

        return pd.DataFrame(signals)
