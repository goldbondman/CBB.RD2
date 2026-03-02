"""
Game mapper for the CBB handicapper picks-tracking application.

Maps raw picks (from raw_picks.csv) to games (from games.csv / app_games.csv)
and writes matched records to picks.csv.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from team_normalizer import CBBNormalizer


class GameMapper:
    """Map raw pick records to specific games."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.normalizer = CBBNormalizer()

    def batch_map_raw_picks(
        self,
        raw_picks: pd.DataFrame,
        games: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Map each successful raw pick to a game row.

        Parameters
        ----------
        raw_picks:
            DataFrame containing successful raw picks.
        games:
            DataFrame of known games.

        Returns
        -------
        list of dicts ready for picks.csv rows.
        """
        results: List[Dict[str, Any]] = []
        now = datetime.now().isoformat()

        for _, pick in raw_picks.iterrows():
            team_raw = str(pick.get('team_raw', ''))
            canonical = self.normalizer.normalize_or_raw(team_raw)
            market = pick.get('market', '')
            line = pick.get('line')
            units = pick.get('units', 1.0)

            game_id, side = self._find_game(canonical, market, games)
            mapping_status = 'ok' if game_id is not None else 'no_game_found'

            results.append({
                'raw_pick_id': pick.get('raw_pick_id'),
                'handicapper_id': pick.get('handicapper_id'),
                'game_id': game_id,
                'market': market,
                'side': side,
                'line': line,
                'units': units,
                'mapping_status': mapping_status,
                'created_at': now,
            })

        return results

    def _find_game(
        self,
        team: str,
        market: str,
        games: pd.DataFrame,
    ):
        """Return (game_id, side) for the best matching game, or (None, None)."""
        if games.empty or not team:
            return None, None

        team_lower = team.lower()

        for _, game in games.iterrows():
            home = str(game.get('home_team', '')).lower()
            away = str(game.get('away_team', '')).lower()

            if team_lower in home or home in team_lower:
                return int(game['game_id']), 'home'
            if team_lower in away or away in team_lower:
                return int(game['game_id']), 'away'

            # Handle "team1/team2" totals
            if '/' in team_lower:
                parts = [p.strip() for p in team_lower.split('/')]
                if any(p in home or home in p for p in parts) or \
                   any(p in away or away in p for p in parts):
                    return int(game['game_id']), 'total'

        return None, None

    def save_picks(
        self,
        mapping_results: List[Dict[str, Any]],
        picks_path: str | None = None,
    ) -> None:
        """Append mapping results to picks.csv."""
        path = Path(picks_path) if picks_path else self.data_dir / 'picks.csv'

        new_df = pd.DataFrame(mapping_results)
        if path.exists():
            existing = pd.read_csv(path)
            next_id = int(existing['pick_id'].max()) + 1 if not existing.empty and 'pick_id' in existing.columns else 1
        else:
            existing = pd.DataFrame()
            next_id = 1

        new_df.insert(0, 'pick_id', range(next_id, next_id + len(new_df)))

        combined = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
        combined.to_csv(path, index=False)
