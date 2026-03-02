"""
CBBNormalizer: normalize college-basketball team names to canonical forms.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Alias map: variant strings → canonical team name
# ---------------------------------------------------------------------------
_ALIASES: dict[str, str] = {
    # Illinois
    "illinois": "Illinois",
    "ill": "Illinois",
    "illini": "Illinois",
    # Wisconsin
    "wisconsin": "Wisconsin",
    "wisc": "Wisconsin",
    "badgers": "Wisconsin",
    # UConn / Connecticut
    "uconn": "UConn",
    "connecticut": "UConn",
    "huskies": "UConn",
    # Kentucky
    "kentucky": "Kentucky",
    "uk": "Kentucky",
    "wildcats": "Kentucky",
    # Duke
    "duke": "Duke",
    "blue devils": "Duke",
    # UNC / North Carolina
    "unc": "North Carolina",
    "north carolina": "North Carolina",
    "tar heels": "North Carolina",
    # Gonzaga
    "gonzaga": "Gonzaga",
    "zags": "Gonzaga",
    # Purdue
    "purdue": "Purdue",
    "boilermakers": "Purdue",
    # Houston
    "houston": "Houston",
    "cougars": "Houston",
    # Tennessee
    "tennessee": "Tennessee",
    "vols": "Tennessee",
}


def _key(name: str) -> str:
    """Lower-case and strip punctuation for lookup."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


class CBBNormalizer:
    """Normalize raw team-name strings to canonical CBB team names."""

    def __init__(self, extra_aliases: dict[str, str] | None = None) -> None:
        self._aliases = dict(_ALIASES)
        if extra_aliases:
            for raw, canonical in extra_aliases.items():
                self._aliases[_key(raw)] = canonical

    def normalize(self, raw_name: str) -> str:
        """Return the canonical name for *raw_name*, or *raw_name* unchanged.

        Parameters
        ----------
        raw_name:
            Team name as it appears in a tweet.

        Returns
        -------
        str
            Canonical team name if known; the original string otherwise.
        """
        return self._aliases.get(_key(raw_name), raw_name)
