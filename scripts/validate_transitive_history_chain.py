#!/usr/bin/env python3
"""Validate transitive opponent-history construction for prediction inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from espn_prediction_runner import OPP_HISTORY_WINDOW, build_team_game_list


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate transitive opponent-history chain.")
    parser.add_argument(
        "--input",
        default="data/team_game_weighted.csv",
        help="Path to team-game input CSV (default: data/team_game_weighted.csv)",
    )
    parser.add_argument(
        "--sample-teams",
        type=int,
        default=25,
        help="Number of recent teams to sample (default: 25)",
    )
    parser.add_argument(
        "--games-per-team",
        type=int,
        default=5,
        help="Number of recent games per sampled team to inspect (default: 5)",
    )
    parser.add_argument(
        "--warn-min-coverage",
        type=float,
        default=0.25,
        help="Warn if transitive coverage ratio is below this value (default: 0.25)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"[BLOCKED] Missing required input: {input_path}")

    df = pd.read_csv(input_path, dtype=str, low_memory=False)
    required = {"team_id", "opponent_id", "game_datetime_utc"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"[FAIL] Missing required columns in {input_path}: {', '.join(missing)}")

    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    base = df.dropna(subset=["team_id", "opponent_id", "game_datetime_utc"]).copy()
    if base.empty:
        raise SystemExit(f"[FAIL] No usable rows in {input_path} after datetime/team/opponent filtering")

    max_dt = base["game_datetime_utc"].max()
    cutoff = max_dt + pd.Timedelta(seconds=1)

    recent_ids = (
        base.sort_values("game_datetime_utc")
        .tail(max(args.sample_teams * args.games_per_team, 250))
        ["team_id"]
        .astype(str)
        .dropna()
        .drop_duplicates()
        .head(args.sample_teams)
        .tolist()
    )
    if not recent_ids:
        raise SystemExit("[FAIL] Could not derive any team_ids for transitive validation sample")

    checked_games = 0
    with_history = 0

    for team_id in recent_ids:
        games = build_team_game_list(
            team_id=str(team_id),
            all_data=base,
            cutoff_dt=cutoff,
            max_games=max(1, args.games_per_team),
        )
        for game in games:
            checked_games += 1
            if len(game.opponent_history) > OPP_HISTORY_WINDOW:
                raise SystemExit(
                    f"[FAIL] OPP_HISTORY_WINDOW exceeded for team_id={team_id} "
                    f"game_id={game.game_id}: {len(game.opponent_history)} > {OPP_HISTORY_WINDOW}"
                )
            if game.opponent_history:
                with_history += 1
            for opp_game in game.opponent_history:
                if not opp_game.date < game.date:
                    raise SystemExit(
                        f"[FAIL] Temporal leakage: opponent game {opp_game.game_id} "
                        f"({opp_game.date}) is not before parent {game.game_id} ({game.date})"
                    )
                if opp_game.opponent_history:
                    raise SystemExit(
                        f"[FAIL] Expected one-level transitive depth, found nested history in {opp_game.game_id}"
                    )

    if checked_games == 0:
        raise SystemExit("[FAIL] Sample produced zero parent games; cannot validate transitive chain")

    if with_history == 0:
        raise SystemExit(
            "[FAIL] No sampled games had transitive opponent history; "
            "this indicates opponent link or cutoff issues."
        )

    coverage = with_history / checked_games
    print(
        "[OK] transitive chain validated: "
        f"checked_games={checked_games} with_history={with_history} coverage={coverage:.3f}"
    )
    if coverage < args.warn_min_coverage:
        print(
            "[WARN] transitive coverage is below threshold "
            f"({coverage:.3f} < {args.warn_min_coverage:.3f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
