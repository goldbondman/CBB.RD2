"""
Team name normalizer for the CBB picks-tracking application.

Provides :class:`CBBNormalizer`, which maps raw team name strings to canonical
D-I CBB team names using alias lookup and fuzzy matching.
"""

from __future__ import annotations

import difflib
import json
import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical D-I CBB team names used for fuzzy matching.
_CANONICAL_TEAMS: list[str] = [
    "Abilene Christian", "Air Force", "Akron", "Alabama", "Alabama A&M",
    "Alabama State", "Albany", "Alcorn State", "American", "Appalachian State",
    "Arizona", "Arizona State", "Arkansas", "Arkansas Pine Bluff", "Arkansas State",
    "Army", "Auburn", "Austin Peay", "Ball State", "Baylor",
    "Bellarmine", "Belmont", "Bethune-Cookman", "Binghamton", "Boise State",
    "Boston College", "Boston University", "Bowling Green", "Bradley", "Brigham Young",
    "Brown", "Bryant", "Bucknell", "Buffalo", "Butler",
    "Cal Poly", "California", "Campbell", "Canisius", "Central Arkansas",
    "Central Connecticut", "Central Florida", "Central Michigan", "Charleston", "Charlotte",
    "Chattanooga", "Chicago State", "Cincinnati", "Clemson", "Cleveland State",
    "Coastal Carolina", "Colgate", "Colorado", "Colorado State", "Columbia",
    "Connecticut", "Coppin State", "Cornell", "Creighton", "Dartmouth",
    "Davidson", "Dayton", "Delaware", "Delaware State", "Denver",
    "DePaul", "Detroit Mercy", "Drake", "Drexel", "Duke",
    "Duquesne", "East Carolina", "East Tennessee State", "Eastern Illinois", "Eastern Kentucky",
    "Eastern Michigan", "Eastern Washington", "Elon", "Evansville", "Fairfield",
    "Fairleigh Dickinson", "Florida", "Florida A&M", "Florida Atlantic", "Florida Gulf Coast",
    "Florida International", "Florida State", "Fordham", "Fresno State", "Furman",
    "Gardner-Webb", "George Mason", "George Washington", "Georgetown", "Georgia",
    "Georgia Southern", "Georgia State", "Georgia Tech", "Gonzaga", "Grambling State",
    "Grand Canyon", "Green Bay", "Hampton", "Hartford", "Harvard",
    "Hawaii", "High Point", "Hofstra", "Holy Cross", "Houston",
    "Howard", "Idaho", "Idaho State", "Illinois", "Illinois State",
    "Incarnate Word", "Indiana", "Indiana State", "Iona", "Iowa",
    "Iowa State", "IUPUI", "Jackson State", "Jacksonville", "Jacksonville State",
    "James Madison", "Kansas", "Kansas City", "Kansas State", "Kennesaw State",
    "Kent State", "Kentucky", "La Salle", "Lafayette", "Lamar",
    "Lehigh", "Liberty", "Lindenwood", "Little Rock", "Long Beach State",
    "Long Island", "Longwood", "Louisiana", "Louisiana Monroe", "Louisiana State",
    "Louisiana Tech", "Louisville", "Loyola Chicago", "Loyola Maryland", "Loyola Marymount",
    "Maine", "Manhattan", "Marist", "Marquette", "Marshall",
    "Maryland", "Massachusetts", "McNeese State", "Memphis", "Merrimack",
    "Miami FL", "Miami OH", "Michigan", "Michigan State", "Middle Tennessee",
    "Milwaukee", "Minnesota", "Mississippi", "Mississippi State", "Mississippi Valley State",
    "Missouri", "Missouri State", "Monmouth", "Montana", "Montana State",
    "Morehead State", "Morgan State", "Mount St. Mary's", "Murray State", "Navy",
    "Nebraska", "Nevada", "Nevada Las Vegas", "New Hampshire", "New Mexico",
    "New Mexico State", "New Orleans", "Niagara", "Nicholls", "Norfolk State",
    "North Alabama", "North Carolina", "North Carolina A&T", "North Carolina Central",
    "North Carolina State", "North Dakota", "North Dakota State", "North Florida",
    "North Texas", "Northeastern", "Northern Arizona", "Northern Colorado", "Northern Illinois",
    "Northern Iowa", "Northern Kentucky", "Northwestern", "Northwestern State", "Notre Dame",
    "Oakland", "Ohio", "Ohio State", "Oklahoma", "Oklahoma State",
    "Old Dominion", "Ole Miss", "Oral Roberts", "Oregon", "Oregon State",
    "Pacific", "Penn State", "Pennsylvania", "Pepperdine", "Pittsburgh",
    "Portland", "Portland State", "Prairie View A&M", "Presbyterian", "Princeton",
    "Providence", "Purdue", "Purdue Fort Wayne", "Queens", "Quinnipiac",
    "Radford", "Rhode Island", "Rice", "Richmond", "Rider",
    "Robert Morris", "Rutgers", "Sacramento State", "Sacred Heart", "Saint Francis PA",
    "Saint Joseph's", "Saint Louis", "Saint Mary's", "Saint Peter's", "Sam Houston",
    "Samford", "San Diego", "San Diego State", "San Francisco", "San Jose State",
    "Santa Barbara", "Seattle", "Seton Hall", "Siena", "South Alabama",
    "South Carolina", "South Carolina State", "South Dakota", "South Dakota State",
    "South Florida", "Southeast Missouri State", "Southeastern Louisiana", "Southern",
    "Southern Illinois", "Southern Indiana", "Southern Methodist", "Southern Mississippi",
    "Southern University", "Southern Utah", "St. Bonaventure", "St. Francis Brooklyn",
    "St. John's", "Stanford", "Stephen F. Austin", "Stetson", "Stony Brook",
    "Syracuse", "Tarleton State", "Temple", "Tennessee", "Tennessee State",
    "Tennessee Tech", "Texas", "Texas A&M", "Texas A&M Commerce", "Texas Christian",
    "Texas El Paso", "Texas Southern", "Texas State", "Texas Tech", "Toledo",
    "Towson", "Troy", "Tulane", "Tulsa", "UAB",
    "UC Davis", "UC Irvine", "UC Riverside", "UC San Diego", "UCLA",
    "UConn", "UMass Lowell", "UNC Asheville", "UNC Greensboro", "UNC Wilmington",
    "UNLV", "USC", "UT Arlington", "UT Martin", "Utah",
    "Utah State", "Utah Tech", "Utah Valley", "UTSA", "Valparaiso",
    "Vanderbilt", "Vermont", "Villanova", "Virginia", "Virginia Commonwealth",
    "Virginia Military Institute", "Virginia Tech", "Wagner", "Wake Forest", "Washington",
    "Washington State", "Weber State", "West Virginia", "Western Carolina", "Western Illinois",
    "Western Kentucky", "Western Michigan", "Wichita State", "William & Mary", "Winthrop",
    "Wisconsin", "Wofford", "Wright State", "Wyoming", "Xavier",
    "Yale", "Youngstown State",
]

# Lower-case lookup for O(1) exact matching.
_CANONICAL_LOWER: dict[str, str] = {t.lower(): t for t in _CANONICAL_TEAMS}


def _load_alias_map() -> dict[str, str]:
    """Load the global alias map from team_alias_map.json if it exists."""
    alias_path = Path(__file__).parent / "data" / "team_alias_map.json"
    if not alias_path.exists():
        return {}
    try:
        raw = json.loads(alias_path.read_text())
        return {k.lower(): v.lower() for k, v in raw.get("global", {}).items()}
    except Exception:
        return {}


class CBBNormalizer:
    """Normalize raw team name strings to canonical CBB team names.

    Uses a three-step approach:
    1. Exact match (case-insensitive) against the canonical team list.
    2. Alias lookup from ``data/team_alias_map.json``.
    3. Fuzzy match via :func:`difflib.get_close_matches`.

    Handles slash-separated multi-team strings (e.g. ``"UConn/Kentucky"``) by
    trying each part independently and returning the first match that exceeds
    the confidence threshold.
    """

    _FUZZY_CUTOFF = 0.6

    def __init__(self) -> None:
        self._alias_map = _load_alias_map()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize_team_name(self, team_raw: str) -> Tuple[Optional[str], float]:
        """Map *team_raw* to a canonical team name with a confidence score.

        Parameters
        ----------
        team_raw:
            Raw team name string, possibly slash-separated (e.g.
            ``"UConn/Kentucky"`` or ``"Illinois"``).

        Returns
        -------
        (canonical_name, confidence)
            *canonical_name* is ``None`` when no suitable match is found.
            *confidence* is in ``[0.0, 1.0]``.
        """
        if not team_raw or not isinstance(team_raw, str):
            return None, 0.0

        parts = [p.strip() for p in team_raw.split("/") if p.strip()]
        for part in parts:
            canonical, confidence = self._normalize_single(part)
            if canonical and confidence >= self._FUZZY_CUTOFF:
                return canonical, confidence

        # No part matched — return best effort from the first part.
        return self._normalize_single(parts[0]) if parts else (None, 0.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_single(self, team: str) -> Tuple[Optional[str], float]:
        """Normalize a single (non-slash) team name string."""
        key = team.lower().strip()

        # 1. Exact match.
        if key in _CANONICAL_LOWER:
            return _CANONICAL_LOWER[key], 1.0

        # 2. Alias lookup → exact match on resolved alias.
        alias_key = self._alias_map.get(key)
        if alias_key and alias_key in _CANONICAL_LOWER:
            return _CANONICAL_LOWER[alias_key], 0.95

        # 3. Fuzzy match against canonical list.
        matches = difflib.get_close_matches(
            team, _CANONICAL_TEAMS, n=1, cutoff=self._FUZZY_CUTOFF
        )
        if matches:
            ratio = difflib.SequenceMatcher(None, team.lower(), matches[0].lower()).ratio()
            return matches[0], round(ratio, 4)

        # Also try fuzzy match after alias resolution.
        if alias_key:
            alias_display = alias_key.title()
            matches = difflib.get_close_matches(
                alias_display, _CANONICAL_TEAMS, n=1, cutoff=self._FUZZY_CUTOFF
            )
            if matches:
                ratio = difflib.SequenceMatcher(
                    None, alias_display.lower(), matches[0].lower()
                ).ratio()
                return matches[0], round(ratio, 4)

        return None, 0.0
