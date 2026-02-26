import re
from typing import Any, Optional

def canonicalize_espn_game_id(game_id: Any) -> Optional[str]:
    """
    Enforces a single canonical key everywhere: espn_game_id stored as a digit-only string.
    Converts any event_id/game_id fields into this canonical format.
    """
    if game_id is None:
        return None

    # Convert to string and strip any leading/trailing whitespace
    gid_str = str(game_id).strip()

    # Handle float artifacts like "401234567.0"
    if gid_str.endswith(".0"):
        gid_str = gid_str[:-2]

    # Remove any non-digit characters
    gid_digits = re.sub(r"\D", "", gid_str)

    if not gid_digits:
        return None

    return gid_digits
