"""
Tweet parser for the CBB handicapper picks-tracking application.

Parses raw tweet text into structured pick records for raw_picks.csv.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Dict, Any


class HandicapperParser:
    """Parse handicapper tweet text into structured pick records."""

    # Patterns for common pick formats
    _SPREAD_RE = re.compile(
        r'([A-Za-z][A-Za-z &\'\.]+?)\s*([+-]\d+(?:\.\d+)?)'
        r'(?:\s*\((\d+(?:\.\d+)?)u\))?',
        re.IGNORECASE,
    )
    _TOTAL_RE = re.compile(
        r'(?:OVER|UNDER|O/U|o/u)\s*(\d+(?:\.\d+)?)'
        r'(?:\s*\((\d+(?:\.\d+)?)u\))?',
        re.IGNORECASE,
    )
    _TOTAL_TEAM_RE = re.compile(
        r'([A-Za-z][A-Za-z &\'\.]+?)\s*/\s*([A-Za-z][A-Za-z &\'\.]+?)'
        r'\s+(?:OVER|UNDER|O/U)\s*(\d+(?:\.\d+)?)'
        r'(?:\s*\((\d+(?:\.\d+)?)u\))?',
        re.IGNORECASE,
    )
    _ML_RE = re.compile(
        r'([A-Za-z][A-Za-z &\'\.]+?)\s+ML\s*([+-]\d+)?'
        r'(?:\s*\((\d+(?:\.\d+)?)u\))?',
        re.IGNORECASE,
    )
    _FADE_RE = re.compile(r'\bFADE\b', re.IGNORECASE)
    _UNITS_RE = re.compile(r'\((\d+(?:\.\d+)?)u\)', re.IGNORECASE)

    def parse_tweet_to_raw_picks(
        self,
        text: str,
        handicapper_id: int,
        tweet_id: str,
        created_at: str,
    ) -> List[Dict[str, Any]]:
        """Parse a single tweet into a list of raw pick dicts.

        Each dict corresponds to one pick row in raw_picks.csv.
        """
        picks: List[Dict[str, Any]] = []
        now = datetime.now().isoformat()
        is_fade = bool(self._FADE_RE.search(text))

        # Try total (team1/team2 OVER/UNDER line) first
        total_team_match = self._TOTAL_TEAM_RE.search(text)
        if total_team_match:
            team_raw = f"{total_team_match.group(1).strip()}/{total_team_match.group(2).strip()}"
            line = float(total_team_match.group(3))
            units = float(total_team_match.group(4)) if total_team_match.group(4) else 1.0
            picks.append(self._make_pick(
                tweet_id, handicapper_id, 'total', team_raw, line, units,
                None, 'fade_detected' if is_fade else 'success', now,
            ))
            return picks

        # Try moneyline
        ml_match = self._ML_RE.search(text)
        if ml_match:
            team_raw = ml_match.group(1).strip()
            odds_str = ml_match.group(2)
            odds = float(odds_str) if odds_str else None
            units = float(ml_match.group(3)) if ml_match.group(3) else 1.0
            picks.append(self._make_pick(
                tweet_id, handicapper_id, 'moneyline', team_raw, None, units,
                odds, 'fade_detected' if is_fade else 'success', now,
            ))
            return picks

        # Try spread
        spread_match = self._SPREAD_RE.search(text)
        if spread_match:
            team_raw = spread_match.group(1).strip()
            line = float(spread_match.group(2))
            units_str = spread_match.group(3)
            if units_str is None:
                units_match = self._UNITS_RE.search(text)
                units = float(units_match.group(1)) if units_match else 1.0
            else:
                units = float(units_str)
            picks.append(self._make_pick(
                tweet_id, handicapper_id, 'spread', team_raw, line, units,
                None, 'fade_detected' if is_fade else 'success', now,
            ))
            return picks

        # Could not parse
        picks.append(self._make_pick(
            tweet_id, handicapper_id, None, None, None, None,
            None, 'parse_failed', now,
        ))
        return picks

    @staticmethod
    def _make_pick(
        tweet_id: str,
        handicapper_id: int,
        market,
        team_raw,
        line,
        units,
        odds,
        parse_status: str,
        parsed_at: str,
    ) -> Dict[str, Any]:
        return {
            'tweet_id': tweet_id,
            'handicapper_id': handicapper_id,
            'market': market,
            'team_raw': team_raw,
            'line': line,
            'units': units,
            'odds': odds,
            'parse_status': parse_status,
            'parsed_at': parsed_at,
        }
