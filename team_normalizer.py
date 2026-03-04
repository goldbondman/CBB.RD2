"""
CBB Team Name Normalizer
Provides fuzzy matching and normalization of raw team name strings
to canonical CBB team names.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Optional, Tuple


# Common nickname/alias mappings to help normalization
_ALIASES: dict[str, str] = {
    "unc": "North Carolina",
    "uconn": "UConn",
    "u conn": "UConn",
    "st john": "St. John's",
    "st johns": "St. John's",
    "fau": "Florida Atlantic",
    "florida atlanti": "Florida Atlantic",
    "gonzaga": "Gonzaga",
    "ill": "Illinois",
    "purdue": "Purdue",
    "ku": "Kansas",
    "uk": "Kentucky",
    "duke": "Duke",
    "k-state": "Kansas State",
    "kstate": "Kansas State",
    "iowa st": "Iowa State",
    "iowa state": "Iowa State",
    "ohio st": "Ohio State",
    "ohio state": "Ohio State",
    "mich st": "Michigan State",
    "michigan st": "Michigan State",
    "michigan state": "Michigan State",
    "wis": "Wisconsin",
    "wisc": "Wisconsin",
}


class CBBNormalizer:
    """Normalizes raw CBB team name strings to canonical team names."""

    def normalize_team(self, team_raw: str) -> str:
        """Return a cleaned/normalized version of *team_raw*.

        Strips punctuation and applies known alias mappings.
        """
        if not team_raw or not isinstance(team_raw, str):
            return team_raw

        cleaned = team_raw.strip()
        # Remove common suffixes like school mascots that appear in abbreviations
        cleaned = re.sub(r"\s+", " ", cleaned)

        lower = cleaned.lower()
        if lower in _ALIASES:
            return _ALIASES[lower]

        # Title-case as a baseline
        return cleaned

    def find_best_match(
        self,
        team_raw: str,
        valid_teams: list[str],
        threshold: float = 0.6,
    ) -> Tuple[Optional[str], float]:
        """Find the best fuzzy match for *team_raw* among *valid_teams*.

        Parameters
        ----------
        team_raw : str
            Raw team name string to match.
        valid_teams : list[str]
            Canonical team names to match against.
        threshold : float
            Minimum similarity ratio required (0–1).

        Returns
        -------
        tuple[str | None, float]
            ``(best_match, score)`` where *best_match* is ``None`` if no
            team meets *threshold*.
        """
        if not team_raw or not valid_teams:
            return None, 0.0

        normalized = self.normalize_team(team_raw).lower()

        best_match: Optional[str] = None
        best_score = 0.0

        for team in valid_teams:
            score = SequenceMatcher(None, normalized, team.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = team

        if best_score < threshold:
            return None, best_score

        return best_match, best_score
