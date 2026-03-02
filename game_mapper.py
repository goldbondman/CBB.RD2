"""
Game mapper for the CBB picks-tracking application.

Provides :class:`GameMapper`, which maps rows from ``raw_picks.csv`` to a
``game_id`` in ``games.csv`` using team-name normalisation and temporal
proximity logic.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

import pandas as pd

from data_loader import CSVDataManager
from team_normalizer import CBBNormalizer

logger = logging.getLogger(__name__)


class GameMapper:
    """Maps raw picks → game_id using team names + temporal logic."""

    def __init__(self) -> None:
        self.normalizer = CBBNormalizer()
        self.time_window_hours = 24  # Only map to same-day games

    # ------------------------------------------------------------------
    # Single-pick mapping
    # ------------------------------------------------------------------

    def map_raw_pick_to_game(
        self,
        raw_pick_row: pd.Series,
        games_df: pd.DataFrame,
    ) -> Dict[str, Union[int, str, None]]:
        """Map a single raw pick → game_id + mapping_status.

        Parameters
        ----------
        raw_pick_row:
            A ``pd.Series`` from ``raw_picks.csv``.
        games_df:
            DataFrame containing schedule/results (must have ``date``,
            ``home_team``, ``away_team``, and ``game_id`` columns).

        Returns
        -------
        dict with keys ``game_id`` and ``mapping_status`` (plus diagnostic
        fields on success or error).
        """
        pick_time = pd.to_datetime(
            raw_pick_row.get("parsed_at", raw_pick_row.get("created_at"))
        )
        team_raw = raw_pick_row.get("team_raw", "")

        if pd.isna(team_raw) or not team_raw:
            return {"game_id": None, "mapping_status": "no_team_name"}

        # Step 1: Normalize team name.
        canonical_team, confidence = self.normalizer.normalize_team_name(str(team_raw))
        if not canonical_team or confidence < 0.7:
            return {
                "game_id": None,
                "mapping_status": f"poor_team_match:{confidence:.2f}",
                "team_raw": team_raw,
                "team_canonical": canonical_team,
            }

        # Step 2: Find candidate games (same day + matching team).
        candidates = self._find_candidate_games(games_df, canonical_team, pick_time)

        if candidates.empty:
            return {
                "game_id": None,
                "mapping_status": "no_matching_game",
                "team_raw": team_raw,
                "team_canonical": canonical_team,
            }

        # Step 3: Pick closest future game (or most-recent past as fallback).
        best_game = self._select_best_match(candidates, pick_time)

        return {
            "game_id": int(best_game["game_id"]),
            "mapping_status": "ok",
            "team_raw": team_raw,
            "team_canonical": canonical_team,
            "tipoff_delta_minutes": best_game.get("tipoff_delta_minutes", 0),
        }

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------

    def _find_candidate_games(
        self,
        games_df: pd.DataFrame,
        canonical_team: str,
        pick_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """Filter *games_df* to viable candidates for *pick_time*."""
        if games_df.empty:
            return pd.DataFrame()

        # ±12-hour window centred on pick time.
        half = self.time_window_hours // 2
        start_window = pick_time - pd.Timedelta(hours=half)
        end_window = pick_time + pd.Timedelta(hours=half)

        candidates = games_df[
            (games_df["date"] >= start_window) & (games_df["date"] <= end_window)
        ].copy()

        if candidates.empty:
            return pd.DataFrame()

        # Normalise game team names for matching.
        candidates["home_team_norm"] = candidates["home_team"].apply(
            lambda x: self.normalizer.normalize_team_name(str(x))[0]
            if pd.notna(x)
            else None
        )
        candidates["away_team_norm"] = candidates["away_team"].apply(
            lambda x: self.normalizer.normalize_team_name(str(x))[0]
            if pd.notna(x)
            else None
        )

        # Match either home OR away team.
        team_match_mask = (candidates["home_team_norm"] == canonical_team) | (
            candidates["away_team_norm"] == canonical_team
        )

        return candidates[team_match_mask].copy()

    # ------------------------------------------------------------------
    # Best-match selection
    # ------------------------------------------------------------------

    def _select_best_match(
        self, candidates: pd.DataFrame, pick_time: pd.Timestamp
    ) -> pd.Series:
        """From *candidates*, return the closest future game (or best past fallback)."""
        if candidates.empty:
            return pd.Series(dtype=object)

        future = candidates[candidates["date"] > pick_time].copy()

        if future.empty:
            # Fallback: closest past game.
            past = candidates.copy()
            past["tipoff_delta_minutes"] = (
                pick_time - past["date"]
            ).dt.total_seconds() / 60
            past = past.sort_values("tipoff_delta_minutes")
            return past.iloc[0]

        future["tipoff_delta_minutes"] = (
            future["date"] - pick_time
        ).dt.total_seconds() / 60
        future = future.sort_values("tipoff_delta_minutes")
        return future.iloc[0]

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def batch_map_raw_picks(
        self, raw_picks_df: pd.DataFrame, games_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Map an entire raw_picks DataFrame and return a picks-ready DataFrame."""
        results = []

        for _, row in raw_picks_df.iterrows():
            mapping = self.map_raw_pick_to_game(row, games_df)
            for col in ("raw_pick_id", "handicapper_id", "market", "line", "units"):
                if col in row:
                    mapping[col] = row[col]
            results.append(mapping)

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_picks(
        self, mapping_results_df: pd.DataFrame, data_dir: str = "./data"
    ) -> pd.DataFrame:
        """Convert mapping results → picks.csv rows and persist them.

        Parameters
        ----------
        mapping_results_df:
            Output of :meth:`batch_map_raw_picks`.
        data_dir:
            Directory containing the app CSV files.

        Returns
        -------
        The new picks rows that were appended.
        """
        dm = CSVDataManager(data_dir)
        data = dm.load_app_data()

        # Generate pick_ids.
        next_pick_id = dm.get_next_id("picks")
        mapping_results_df = mapping_results_df.copy()
        mapping_results_df["pick_id"] = range(
            next_pick_id, next_pick_id + len(mapping_results_df)
        )

        # Select only picks.csv columns + clean up.
        keep_cols = [
            "pick_id", "raw_pick_id", "handicapper_id", "game_id",
            "market", "side", "line", "units", "mapping_status",
        ]
        available = [c for c in keep_cols if c in mapping_results_df.columns]
        picks_df = mapping_results_df[available].copy()

        if "side" in picks_df.columns:
            picks_df["side"] = picks_df["side"].fillna("unknown")

        # Append to existing picks.
        if not data["picks"].empty:
            data["picks"] = pd.concat(
                [data["picks"], picks_df], ignore_index=True
            )
        else:
            data["picks"] = picks_df

        dm.save_app_data(data)
        logger.info("Saved %d new picks to picks.csv", len(picks_df))
        return picks_df


# ---------------------------------------------------------------------------
# Test suite (standalone runner)
# ---------------------------------------------------------------------------

def test_game_mapper() -> "GameMapper":
    """Test mapping with sample data."""
    mapper = GameMapper()

    games_data = {
        "game_id": [123, 124, 125, 126],
        "date": pd.to_datetime([
            "2026-02-28 19:00:00",
            "2026-02-28 20:30:00",
            "2026-02-28 21:00:00",
            "2026-02-28 22:15:00",
        ]),
        "home_team": ["Illinois", "Kentucky", "Purdue", "Houston"],
        "away_team": ["Wisconsin", "UConn", "Gonzaga", "Tennessee"],
    }
    games_df = pd.DataFrame(games_data)

    test_picks = [
        pd.Series({
            "raw_pick_id": 1,
            "team_raw": "Illinois",
            "parsed_at": "2026-02-28 14:25:30",
        }),
        pd.Series({
            "raw_pick_id": 2,
            "team_raw": "UConn/Kentucky",
            "parsed_at": "2026-02-28 15:12:45",
        }),
        pd.Series({
            "raw_pick_id": 4,
            "team_raw": "Gonzaga",
            "parsed_at": "2026-02-28 18:22:15",
        }),
    ]

    print("=== GAME MAPPER TEST ===")
    for pick in test_picks:
        result = mapper.map_raw_pick_to_game(pick, games_df)
        print(f"Pick {pick['raw_pick_id']}: {pick['team_raw']}")
        print(f"  → {result}")
        print()

    print("Batch mapping test:")
    batch_results = mapper.batch_map_raw_picks(pd.DataFrame(test_picks), games_df)
    print(batch_results[["raw_pick_id", "game_id", "mapping_status"]].to_string(index=False))

    return mapper


if __name__ == "__main__":
    test_game_mapper()
