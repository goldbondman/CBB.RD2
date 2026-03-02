"""
CBB Game Mapper
Maps raw picks (with team names) to game records in games.csv via fuzzy team matching.
"""

from __future__ import annotations

import pandas as pd
from typing import Optional

from team_normalizer import CBBNormalizer


class GameMapper:
    """Maps raw picks to games using team-name fuzzy matching."""

    def __init__(self, match_threshold: float = 0.6):
        self.normalizer = CBBNormalizer()
        self.match_threshold = match_threshold

    def map_raw_pick_to_game(
        self,
        pick: pd.Series,
        games_df: pd.DataFrame,
    ) -> dict:
        """Attempt to map *pick* to a row in *games_df*.

        Parameters
        ----------
        pick : pd.Series
            A row from ``raw_picks`` (must contain ``team_raw``).
        games_df : pd.DataFrame
            The games table (must contain ``game_id``, ``home_team``,
            ``away_team``).

        Returns
        -------
        dict
            Always contains ``game_id`` (``None`` if no match) and
            ``mapping_status`` (``'ok'``, ``'poor_team_match'``, or
            ``'no_match'``).
        """
        team_raw = pick.get("team_raw", "") or ""
        if not team_raw or games_df.empty:
            return {"game_id": None, "mapping_status": "no_match"}

        all_teams = list(games_df["home_team"]) + list(games_df["away_team"])
        best_team, score = self.normalizer.find_best_match(
            team_raw, all_teams, threshold=self.match_threshold
        )

        if best_team is None:
            # Try with a lower threshold and mark as poor match
            best_team, score = self.normalizer.find_best_match(
                team_raw, all_teams, threshold=0.4
            )
            if best_team is None:
                return {"game_id": None, "mapping_status": "no_match"}
            status = "poor_team_match"
        else:
            status = "ok"

        # Find the game containing best_team
        mask = (
            (games_df["home_team"] == best_team)
            | (games_df["away_team"] == best_team)
        )
        matched = games_df[mask]
        if matched.empty:
            return {"game_id": None, "mapping_status": "no_match"}

        game_id = int(matched.iloc[0]["game_id"])
        return {
            "game_id": game_id,
            "mapping_status": status,
            "matched_team": best_team,
            "match_score": round(score, 3),
        }
