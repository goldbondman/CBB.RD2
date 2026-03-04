import re
from typing import Dict, List, Optional, Tuple
from difflib import SequenceMatcher
import logging

import pandas as pd

logger = logging.getLogger(__name__)


class CBBNormalizer:
    """Standalone CBB team name normalizer - no external data required"""

    def __init__(self):
        # Comprehensive hardcoded mappings for top ~70 CBB teams
        self.team_mapping: Dict[str, str] = {
            # Power 5 + Big East + Top Mid-Majors
            'illinois': 'Illinois',
            'illini': 'Illinois',
            'fighting illini': 'Illinois',
            'uconn': 'UConn',
            'connecticut': 'UConn',
            'huskies': 'UConn',
            'kansas': 'Kansas',
            'ku': 'Kansas',
            'jayhawks': 'Kansas',
            'uk': 'Kentucky',
            'kentucky': 'Kentucky',
            'wildcats': 'Kentucky',
            'duke': 'Duke',
            'blue devils': 'Duke',
            'unc': 'North Carolina',
            'north carolina': 'North Carolina',
            'tar heels': 'North Carolina',
            'n carolina': 'North Carolina',
            'gonzaga': 'Gonzaga',
            'zags': 'Gonzaga',
            'bulldogs': 'Gonzaga',
            'houston': 'Houston',
            'uh': 'Houston',
            'cougars': 'Houston',
            'purdue': 'Purdue',
            'boilermakers': 'Purdue',
            'ucla': 'UCLA',
            'ucla bruins': 'UCLA',
            'arizona': 'Arizona',
            'arizona wildcats': 'Arizona',
            'bama': 'Alabama',
            'alabama': 'Alabama',
            'roll tide': 'Alabama',
            'osu': 'Ohio State',
            'ohio state': 'Ohio State',
            'buckeyes': 'Ohio State',
            'michigan state': 'Michigan State',
            'msu': 'Michigan State',
            'spartans': 'Michigan State',
            'michigan': 'Michigan',
            'wolverines': 'Michigan',
            'tennessee': 'Tennessee',
            'vols': 'Tennessee',
            'volunteers': 'Tennessee',
            'iowa state': 'Iowa State',
            'isu': 'Iowa State',
            'cyclones': 'Iowa State',
            'baylor': 'Baylor',
            'bears': 'Baylor',
            'creighton': 'Creighton',
            'bluejays': 'Creighton',
            'villanova': 'Villanova',
            'nova': 'Villanova',
            'st john': "St. John's",
            'st johns': "St. John's",
            'st. john': "St. John's",
            'st. johns': "St. John's",
            'red storm': "St. John's",
            'providence': 'Providence',
            'friars': 'Providence',
            'seton hall': 'Seton Hall',
            'pirates': 'Seton Hall',
            'xavier': 'Xavier',
            'musketeers': 'Xavier',
            'miami': 'Miami (FL)',
            'hurricanes': 'Miami (FL)',
            'rutgers': 'Rutgers',
            'scarlet knights': 'Rutgers',
            'syracuse': 'Syracuse',
            'orange': 'Syracuse',
            'marquette': 'Marquette',
            'golden eagles': 'Marquette',
            'dayton': 'Dayton',
            'flyers': 'Dayton',
            'memphis': 'Memphis',
            'tigers': 'Memphis',
            'wichita state': 'Wichita State',
            'shockers': 'Wichita State',
            'byu': 'BYU',
            'brigham young': 'BYU',
            'san diego state': 'San Diego State',
            'sdsu': 'San Diego State',
            'aztecs': 'San Diego State',
            'fau': 'Florida Atlantic',
            'florida atlantic': 'Florida Atlantic',
            'owls': 'Florida Atlantic',
        }

        # Common suffixes/phrases to strip.
        # Note: '\bst\.\s*' strips standalone "St." abbreviations (e.g. in
        # addresses).  Actual team names that begin with "St." (e.g. St. John's)
        # are pre-normalised before this list is applied – see _preprocess_name.
        self.junk_patterns = [
            r'\b(university|state|college|tech|university of)\b',
            r'\b(fighting|wild|red|golden|scarlet|blue|black|green)\s+\w+',
            r'\bst\.\s*',
            r'\bjr?\b',
            r'\bno\.\s*\d+\b',
        ]

    def normalize_team_name(self, team_raw: str) -> Tuple[Optional[str], float]:
        """
        Convert messy team name → canonical team name + confidence

        Input: "Illinois Fighting Illini" → Output: ("Illinois", 1.0)
        Input: "St Johns Red Storm" → Output: ("St. John's", 0.95)
        """
        if not team_raw or (isinstance(team_raw, float) and pd.isna(team_raw)):
            return None, 0.0

        # Step 1: Clean raw input
        cleaned = self._preprocess_name(team_raw.lower().strip())

        if not cleaned:
            return None, 0.0

        # Step 2: Exact mapping lookup
        if cleaned in self.team_mapping:
            return self.team_mapping[cleaned], 1.0

        # Step 3: Multi-word → single word mapping
        for pattern, canonical in self.team_mapping.items():
            if re.search(rf'\b{re.escape(pattern)}\b', cleaned):
                return canonical, 0.98

        # Step 4: Fuzzy matching (top 3 candidates)
        candidates = self._fuzzy_match(cleaned)
        if candidates:
            best_match, score = candidates[0]
            if score >= 0.75:  # Threshold for fuzzy matches
                return best_match, score

        logger.warning(f"Could not normalize team: '{team_raw}' → '{cleaned}'")
        return None, 0.0

    def _preprocess_name(self, name: str) -> Optional[str]:
        """Strip junk phrases, normalize punctuation"""
        cleaned = name

        # Pre-normalize "st. john" variants before junk stripping so the
        # \bst\.\s* pattern does not destroy the team prefix.
        cleaned = re.sub(r"\bst\.?\s+john", 'st johns', cleaned)

        # Remove common junk
        for pattern in self.junk_patterns:
            cleaned = re.sub(pattern, '', cleaned)

        # Normalize apostrophes, periods, extra spaces
        cleaned = re.sub(r'[^\w\s&/]', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()

        # Common abbreviations
        cleaned = cleaned.replace('n carolina', 'north carolina')

        return cleaned if cleaned else None

    def _fuzzy_match(self, cleaned: str, threshold: float = 0.75) -> List[Tuple[str, float]]:
        """Find best fuzzy matches from mapping"""
        candidates = []

        for pattern, canonical in self.team_mapping.items():
            # Simple word overlap + Levenshtein
            score = SequenceMatcher(None, cleaned, pattern).ratio()

            # Boost exact word matches; the 1.2× multiplier can push the score
            # above 1.0, so cap at 1.0 to keep confidence in [0, 1].
            if any(word in canonical.lower() for word in cleaned.split()):
                score *= 1.2
            score = min(score, 1.0)

            if score >= threshold:
                candidates.append((canonical, score))

        return sorted(candidates, key=lambda x: x[1], reverse=True)

    def batch_normalize(self, team_names: List[str]) -> pd.DataFrame:
        """Batch process list of team names → normalized + confidence"""
        results = []
        for team in team_names:
            canonical, confidence = self.normalize_team_name(team)
            results.append({
                'team_raw': team,
                'team_canonical': canonical,
                'confidence': confidence,
            })

        return pd.DataFrame(results)


# Test suite
def test_normalizer():
    """Comprehensive test of team normalization"""
    normalizer = CBBNormalizer()

    test_cases = [
        "Illinois Fighting Illini",
        "UConn Huskies",
        "Kansas Jayhawks",
        "Kentucky Wildcats",
        "Duke Blue Devils",
        "North Carolina Tar Heels",
        "Gonzaga Bulldogs",
        "St. John's Red Storm",
        "St Johns Red Storm",
        "Florida Atlantic Owls",
        "FAU",
        "Random Junk Team",
        "University of Miami Hurricanes",
    ]

    print("=== TEAM NORMALIZER TEST ===")
    results = []

    for team_raw in test_cases:
        canonical, confidence = normalizer.normalize_team_name(team_raw)
        result = f"{team_raw:<30} → {canonical or 'None':<12} ({confidence:.2f})"
        print(result)

        if canonical:
            results.append({'raw': team_raw, 'normalized': canonical, 'conf': confidence})

    print(f"\n✅ Successfully normalized {len(results)}/{len(test_cases)} teams")

    # Test fuzzy matching
    fuzzy_test = normalizer.normalize_team_name("St John Redstorm")
    print(f"\nFuzzy test - 'St John Redstorm': {fuzzy_test}")

    return normalizer


if __name__ == "__main__":
    test_normalizer()
