"""
GameMapper: map raw picks to scheduled games via team-name matching.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from team_normalizer import CBBNormalizer


def _norm(name: str) -> str:
    """Lower-case and strip whitespace for comparison."""
    return str(name).lower().strip()


class GameMapper:
    """Map raw picks to games using team-name normalisation."""

    def __init__(
        self,
        data_dir: str | Path | None = None,
        normalizer: CBBNormalizer | None = None,
    ) -> None:
        if data_dir is None:
            data_dir = Path(__file__).parent / "data" / "handicapper"
        self.data_dir = Path(data_dir)
        self._norm = normalizer or CBBNormalizer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def batch_map_raw_picks(
        self,
        raw_picks: pd.DataFrame,
        games: pd.DataFrame,
    ) -> pd.DataFrame:
        """Map every row in *raw_picks* to a game in *games*.

        Parameters
        ----------
        raw_picks:
            DataFrame with at least the columns ``raw_pick_id``,
            ``handicapper_id``, ``market``, ``team_raw``, ``line``, ``units``.
        games:
            DataFrame with at least the columns ``game_id``, ``home_team``,
            ``away_team``.

        Returns
        -------
        pd.DataFrame
            Copy of *raw_picks* with extra columns: ``game_id``, ``side``,
            ``mapping_status``, ``created_at``.
        """
        result = raw_picks.copy()
        game_ids: list[int | None] = []
        sides: list[str] = []
        statuses: list[str] = []

        for _, row in raw_picks.iterrows():
            game_id, side = self._find_game(row, games)
            if game_id is not None:
                game_ids.append(game_id)
                sides.append(side)
                statuses.append("ok")
            else:
                game_ids.append(None)
                sides.append("")
                statuses.append("no_match")

        result["game_id"] = game_ids
        result["side"] = sides
        result["mapping_status"] = statuses
        result["created_at"] = pd.Timestamp.now().isoformat()
        return result

    def save_picks(self, mapping_results: pd.DataFrame) -> None:
        """Save successfully-mapped picks to ``picks.csv``.

        Only rows with ``mapping_status == "ok"`` are written.

        Parameters
        ----------
        mapping_results:
            DataFrame returned by :meth:`batch_map_raw_picks`.
        """
        mapped = mapping_results[mapping_results["mapping_status"] == "ok"].copy()
        if mapped.empty:
            return

        path = self.data_dir / "picks.csv"

        # Determine next pick_id
        if path.exists():
            existing = pd.read_csv(path)
            next_id = int(existing["pick_id"].max()) + 1 if not existing.empty else 1
        else:
            existing = pd.DataFrame()
            next_id = 1

        out_cols = [
            "raw_pick_id",
            "handicapper_id",
            "game_id",
            "market",
            "side",
            "line",
            "units",
            "mapping_status",
            "created_at",
        ]
        picks = mapped[[c for c in out_cols if c in mapped.columns]].copy()
        picks.insert(0, "pick_id", range(next_id, next_id + len(picks)))

        combined = pd.concat([existing, picks], ignore_index=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(path, index=False)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_game(
        self,
        pick_row: pd.Series,
        games: pd.DataFrame,
    ) -> tuple[int | None, str]:
        """Return *(game_id, side)* for the best-matching game, or *(None, '')*."""
        team_raw = str(pick_row.get("team_raw", ""))
        # For totals the raw name may be "TeamA/TeamB"; extract both
        parts = [t.strip() for t in team_raw.replace("/", "|").split("|")]
        normalised = [_norm(self._norm.normalize(p)) for p in parts]

        for _, game in games.iterrows():
            home = _norm(str(game.get("home_team", "")))
            away = _norm(str(game.get("away_team", "")))

            for n in normalised:
                if n in (home, away):
                    side = "home" if n == home else "away"
                    market = str(pick_row.get("market", "")).lower()
                    if market == "total":
                        side = "over"
                    return int(game["game_id"]), side

        return None, ""
