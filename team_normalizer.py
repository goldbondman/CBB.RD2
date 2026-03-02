"""
Team name normalizer for the CBB handicapper picks-tracking application.

Maps raw team name strings (as typed in tweets) to canonical team names.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Dict


class CBBNormalizer:
    """Normalize raw team name strings to canonical CBB team names."""

    # Common abbreviations and aliases → canonical name
    _ALIASES: Dict[str, str] = {
        # Big Ten
        'illinois': 'Illinois', 'illini': 'Illinois',
        'wisconsin': 'Wisconsin', 'badgers': 'Wisconsin',
        'michigan': 'Michigan', 'wolverines': 'Michigan',
        'michigan state': 'Michigan State', 'msu': 'Michigan State', 'spartans': 'Michigan State',
        'ohio state': 'Ohio State', 'osu': 'Ohio State', 'buckeyes': 'Ohio State',
        'indiana': 'Indiana', 'hoosiers': 'Indiana',
        'purdue': 'Purdue', 'boilermakers': 'Purdue',
        'iowa': 'Iowa', 'hawkeyes': 'Iowa',
        'minnesota': 'Minnesota', 'gophers': 'Minnesota',
        'penn state': 'Penn State', 'nittany lions': 'Penn State',
        'nebraska': 'Nebraska', 'cornhuskers': 'Nebraska',
        'northwestern': 'Northwestern', 'wildcats nu': 'Northwestern',
        'rutgers': 'Rutgers',
        'maryland': 'Maryland', 'terps': 'Maryland',
        # Big East
        'uconn': 'UConn', 'connecticut': 'UConn', 'huskies': 'UConn',
        'villanova': 'Villanova', 'nova': 'Villanova',
        'marquette': 'Marquette', 'golden eagles': 'Marquette',
        'seton hall': 'Seton Hall', 'pirates': 'Seton Hall',
        'georgetown': 'Georgetown', 'hoyas': 'Georgetown',
        'st. johns': 'St. John\'s', 'st johns': 'St. John\'s',
        'butler': 'Butler', 'bulldogs butler': 'Butler',
        'depaul': 'DePaul',
        'providence': 'Providence', 'friars': 'Providence',
        'xavier': 'Xavier', 'musketeers': 'Xavier',
        'creighton': 'Creighton', 'bluejays': 'Creighton',
        # ACC
        'duke': 'Duke', 'blue devils': 'Duke',
        'unc': 'North Carolina', 'north carolina': 'North Carolina', 'tar heels': 'North Carolina',
        'nc state': 'NC State', 'wolfpack': 'NC State',
        'wake forest': 'Wake Forest', 'demon deacons': 'Wake Forest',
        'virginia': 'Virginia', 'cavaliers': 'Virginia',
        'virginia tech': 'Virginia Tech', 'hokies': 'Virginia Tech',
        'clemson': 'Clemson', 'tigers clemson': 'Clemson',
        'florida state': 'Florida State', 'seminoles': 'Florida State',
        'miami': 'Miami', 'hurricanes': 'Miami',
        'boston college': 'Boston College', 'eagles bc': 'Boston College',
        'louisville': 'Louisville', 'cardinals ul': 'Louisville',
        'notre dame': 'Notre Dame', 'fighting irish': 'Notre Dame',
        'georgia tech': 'Georgia Tech', 'yellow jackets': 'Georgia Tech',
        'pittsburgh': 'Pittsburgh', 'pitt': 'Pittsburgh', 'panthers': 'Pittsburgh',
        'syracuse': 'Syracuse', 'orange': 'Syracuse',
        # Big 12
        'kansas': 'Kansas', 'jayhawks': 'Kansas',
        'kansas state': 'Kansas State', 'wildcats ksu': 'Kansas State',
        'baylor': 'Baylor', 'bears': 'Baylor',
        'texas': 'Texas', 'longhorns': 'Texas',
        'tcu': 'TCU', 'horned frogs': 'TCU',
        'oklahoma': 'Oklahoma', 'sooners': 'Oklahoma',
        'oklahoma state': 'Oklahoma State', 'cowboys': 'Oklahoma State',
        'west virginia': 'West Virginia', 'mountaineers': 'West Virginia',
        'iowa state': 'Iowa State', 'cyclones': 'Iowa State',
        'texas tech': 'Texas Tech', 'red raiders': 'Texas Tech',
        # SEC
        'kentucky': 'Kentucky', 'wildcats uk': 'Kentucky',
        'alabama': 'Alabama', 'crimson tide': 'Alabama',
        'auburn': 'Auburn',
        'tennessee': 'Tennessee', 'vols': 'Tennessee',
        'arkansas': 'Arkansas', 'razorbacks': 'Arkansas',
        'lsu': 'LSU', 'tigers lsu': 'LSU',
        'mississippi state': 'Mississippi State', 'bulldogs msu': 'Mississippi State',
        'ole miss': 'Ole Miss', 'rebels': 'Ole Miss',
        'vanderbilt': 'Vanderbilt', 'commodores': 'Vanderbilt',
        'georgia': 'Georgia', 'bulldogs uga': 'Georgia',
        'florida': 'Florida', 'gators': 'Florida',
        'south carolina': 'South Carolina', 'gamecocks': 'South Carolina',
        'missouri': 'Missouri', 'tigers mou': 'Missouri',
        'texas a&m': 'Texas A&M', 'aggies': 'Texas A&M',
        # Other notable programs
        'gonzaga': 'Gonzaga', 'zags': 'Gonzaga', 'bulldogs g': 'Gonzaga',
        'houston': 'Houston', 'cougars': 'Houston',
        'memphis': 'Memphis',
        'cincinnati': 'Cincinnati', 'bearcats': 'Cincinnati',
        'wichita state': 'Wichita State', 'shockers': 'Wichita State',
        'dayton': 'Dayton', 'flyers': 'Dayton',
        'saint marys': 'Saint Mary\'s', "saint mary's": "Saint Mary's",
        'byu': 'BYU', 'cougars byu': 'BYU',
        'san diego state': 'San Diego State', 'sdsu': 'San Diego State', 'aztecs': 'San Diego State',
    }

    def normalize(self, raw: str) -> Optional[str]:
        """Return canonical team name or None if unrecognized."""
        if not raw:
            return None
        key = raw.strip().lower()
        key = re.sub(r'[^\w\s/&\']', '', key).strip()
        return self._ALIASES.get(key)

    def normalize_or_raw(self, raw: str) -> str:
        """Return canonical name if found, otherwise return raw stripped."""
        result = self.normalize(raw)
        return result if result is not None else raw.strip()
