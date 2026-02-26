import re
import json
from pathlib import Path
from typing import Dict, List, Optional

def normalize_team_name(name: str) -> str:
    """
    Lowercase, remove punctuation, remove common suffixes/prefixes.
    Must handle abbreviations and punctuation.
    """
    if not name:
        return ""

    # Lowercase and handle common punctuation
    # Remove apostrophes without adding space
    cleaned = name.lower().replace("'", "")
    # Replace other non-alphanumeric with space
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", cleaned)

    # Common abbreviations and tokens to standardize or remove
    # Standardizing "st." to "state"
    cleaned = re.sub(r"\bst\b", "state", cleaned)

    # Tokens to remove (meaningless for identification)
    tokens_to_remove = {
        "university", "college", "of", "the", "at", "and",
    }

    tokens = [
        token for token in cleaned.split()
        if token not in tokens_to_remove
    ]

    return " ".join(tokens)

def slugify(canonical_name: str) -> str:
    """
    Creates a normalized slug from a canonical team name.
    """
    if not canonical_name:
        return ""
    return re.sub(r"\s+", "-", canonical_name.strip().lower())

class TeamCanonicalizer:
    def __init__(self, alias_map_path: str = "data/team_alias_map.json",
                 exceptions_path: str = "data/team_exceptions.json"):
        self.alias_map_path = Path(alias_map_path)
        self.exceptions_path = Path(exceptions_path)
        self.alias_map = self._load_json(self.alias_map_path)
        self.exceptions = self._load_json(self.exceptions_path)

    def _load_json(self, path: Path) -> Dict:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return {}
        return {}

    def get_canonical_name(self, raw_name: str, source: Optional[str] = None) -> str:
        """
        Resolves a raw name to a canonical name using aliases and exceptions.
        """
        norm_raw = normalize_team_name(raw_name)

        # 1. Check exceptions (high priority for collisions like Miami)
        if norm_raw in self.exceptions:
            return self.exceptions[norm_raw]

        # 2. Check source-specific aliases
        if source and source in self.alias_map.get("source_specific", {}):
            if norm_raw in self.alias_map["source_specific"][source]:
                return self.alias_map["source_specific"][source][norm_raw]

        # 3. Check global aliases
        if norm_raw in self.alias_map.get("global", {}):
            return self.alias_map["global"][norm_raw]

        return norm_raw

    def get_slug(self, raw_name: str, source: Optional[str] = None) -> str:
        canonical = self.get_canonical_name(raw_name, source)
        return slugify(canonical)
