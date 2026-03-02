"""
HandicapperParser: parse sports-betting picks from tweet text into raw-pick dicts.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Regex patterns for individual pick segments
# ---------------------------------------------------------------------------

# Total: e.g. "UConn/Kentucky O142.5 (2u)" or "UConn/Kentucky OVER 142.5"
_TOTAL_RE = re.compile(
    r"^(?P<teams>.+?)\s+"
    r"(?:OVER|UNDER|O|U)\s*"
    r"(?P<total>\d+(?:\.\d+)?)"
    r"(?:\s*\((?P<units>\d+(?:\.\d+)?)u\))?$",
    re.IGNORECASE,
)

# Moneyline: e.g. "Purdue ML" or "Purdue ML +120 (1u)"
_ML_RE = re.compile(
    r"^(?P<team>.+?)\s+ML"
    r"(?:\s+(?P<odds>[+-]\d+))?"
    r"(?:\s*\((?P<units>\d+(?:\.\d+)?)u\))?$",
    re.IGNORECASE,
)

# Spread: e.g. "Illinois -3.5 (2u)"
_SPREAD_RE = re.compile(
    r"^(?P<team>.+?)\s+"
    r"(?P<line>[+-]\d+(?:\.\d+)?)"
    r"(?:\s*\((?P<units>\d+(?:\.\d+)?)u\))?$",
    re.IGNORECASE,
)


def _parse_segment(segment: str) -> dict[str, Any] | None:
    """Parse one pick segment into a raw-pick dict, or *None* if unparseable."""
    segment = segment.strip()

    m = _TOTAL_RE.match(segment)
    if m:
        return {
            "market": "total",
            "team_raw": m.group("teams").strip(),
            "line": float(m.group("total")),
            "units": float(m.group("units") or 1.0),
            "odds": None,
        }

    m = _ML_RE.match(segment)
    if m:
        return {
            "market": "moneyline",
            "team_raw": m.group("team").strip(),
            "line": None,
            "units": float(m.group("units") or 1.0),
            "odds": m.group("odds"),
        }

    m = _SPREAD_RE.match(segment)
    if m:
        return {
            "market": "spread",
            "team_raw": m.group("team").strip(),
            "line": float(m.group("line")),
            "units": float(m.group("units") or 1.0),
            "odds": None,
        }

    return None


class HandicapperParser:
    """Parse tweet text into a list of raw-pick records."""

    def __init__(self, data_dir: str | Path | None = None) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parent / "data" / "handicapper"
        self.data_dir = Path(data_dir)

    def parse_tweet_to_raw_picks(
        self,
        text: str,
        handicapper_id: int,
        tweet_id: str = "manual",
        created_at: str | None = None,
    ) -> list[dict[str, Any]]:
        """Parse *text* and return a list of raw-pick dicts.

        Parameters
        ----------
        text:
            The tweet body to parse.
        handicapper_id:
            ID of the handicapper who authored the tweet.
        tweet_id:
            Source tweet identifier (default ``"manual"``).
        created_at:
            ISO-8601 timestamp string; defaults to now.

        Returns
        -------
        list[dict]
            One dict per recognised pick, with keys ``market``, ``team_raw``,
            ``line``, ``units``, ``odds``, ``handicapper_id``, ``tweet_id``,
            ``parse_status``, ``parsed_at``.
        """
        if created_at is None:
            created_at = pd.Timestamp.now().isoformat()

        segments = [s.strip() for s in text.split("|")]
        raw_picks: list[dict[str, Any]] = []

        for segment in segments:
            parsed = _parse_segment(segment)
            if parsed is not None:
                parsed.update(
                    {
                        "handicapper_id": handicapper_id,
                        "tweet_id": tweet_id,
                        "parse_status": "success",
                        "parsed_at": created_at,
                    }
                )
                raw_picks.append(parsed)

        return raw_picks

    def save_raw_picks(self, raw_picks: list[dict[str, Any]]) -> None:
        """Append *raw_picks* to ``raw_picks.csv``.

        Parameters
        ----------
        raw_picks:
            List of raw-pick dicts as returned by :meth:`parse_tweet_to_raw_picks`.
        """
        if not raw_picks:
            return

        path = self.data_dir / "raw_picks.csv"
        new_df = pd.DataFrame(raw_picks)

        if path.exists():
            existing = pd.read_csv(path)
            if "raw_pick_id" in existing.columns and not existing.empty:
                next_id = int(existing["raw_pick_id"].max()) + 1
            else:
                next_id = 1
        else:
            existing = pd.DataFrame()
            next_id = 1

        new_df.insert(0, "raw_pick_id", range(next_id, next_id + len(new_df)))
        combined = pd.concat([existing, new_df], ignore_index=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False)
