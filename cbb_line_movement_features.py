#!/usr/bin/env python3
"""Build one-row-per-game line movement features from capture-level market lines."""

# Input discovery findings from data/market_lines.csv (read before implementation):
# 1) capture_type unique values: opening, pregame
# 2) Home-team spread column: home_spread
# 3) Total line column: total_line
# 4) Capture timestamp column: captured_at_utc
# 5) Public betting % columns present: home_tickets_pct, away_tickets_pct (also money % columns exist)
# 6) Multiple rows are written per game_id (capture-level history), not one row with open/close columns.

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MARKET_LINES_PATH = Path("data/market_lines.csv")
GAMES_PATH = Path("data/games.csv")
OUTPUT_PATH = Path("data/line_movement_features.csv")

KEY_NUMBERS = [1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 5.5, 6, 6.5, 7, 7.5, 10, 10.5]


@dataclass
class Filters:
    days_back: Optional[int] = None
    season: Optional[int] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CBB line movement feature rows")
    parser.add_argument("--days-back", type=int, default=None, help="Restrict to last N days from games.csv")
    parser.add_argument("--season", type=int, default=None, help="Restrict to games in a season year")
    return parser.parse_args()


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _parse_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _is_explicit_pregame(capture_type: str) -> bool:
    if not capture_type:
        return False
    low = str(capture_type).lower()
    return any(token in low for token in ("pre", "open", "clos"))


def _exclude_capture_type(capture_type: str) -> bool:
    low = str(capture_type or "").lower()
    return any(token in low for token in ("live", "halftime", "final", "postgame"))


def _crossed_key_number(open_line: float, close_line: float) -> int:
    if pd.isna(open_line) or pd.isna(close_line) or open_line == close_line:
        return 0
    lo, hi = sorted([open_line, close_line])
    for key in KEY_NUMBERS:
        if lo < key < hi:
            return 1
    return 0


def _direction(value: float, pos: str, neg: str, threshold: float = 0.25) -> str:
    if pd.isna(value) or abs(value) < threshold:
        return "none"
    return pos if value > 0 else neg


def _steam_flag(deltas: pd.Series, overall_move: float, threshold: float) -> int:
    if pd.isna(overall_move) or overall_move == 0:
        return 0
    sign = 1 if overall_move > 0 else -1
    directional = deltas.dropna() * sign
    return int((directional >= threshold).any())


def _resolve_public_side(group: pd.DataFrame) -> tuple[Optional[int], Optional[str]]:
    if "home_tickets_pct" not in group.columns or "away_tickets_pct" not in group.columns:
        return None, None

    g = group.sort_values("captured_at_utc").copy()
    g["home_tickets_pct"] = _to_float(g["home_tickets_pct"])
    g["away_tickets_pct"] = _to_float(g["away_tickets_pct"])
    valid = g.dropna(subset=["home_tickets_pct", "away_tickets_pct"]).tail(1)
    if valid.empty:
        return None, None

    row = valid.iloc[0]
    if row["home_tickets_pct"] >= 60:
        return 1, "home"
    if row["away_tickets_pct"] >= 60:
        return 1, "away"
    return 0, None


def load_inputs(filters: Filters) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_csv(MARKET_LINES_PATH, dtype=str)
    games = pd.read_csv(GAMES_PATH, dtype=str)

    games["game_datetime_utc"] = _parse_ts(games.get("game_datetime_utc", pd.Series(dtype=str)))
    games["game_date"] = pd.to_datetime(games.get("date", pd.Series(dtype=str)), format="%Y%m%d", errors="coerce")

    if filters.season is not None:
        season_mask = games["game_datetime_utc"].dt.year.eq(filters.season) | games["game_date"].dt.year.eq(filters.season)
        games = games[season_mask].copy()

    if filters.days_back is not None:
        latest = games["game_date"].dropna().max()
        if pd.notna(latest):
            cutoff = latest - pd.Timedelta(days=filters.days_back)
            games = games[games["game_date"] >= cutoff].copy()

    if not games.empty:
        market = market[market["game_id"].isin(set(games["game_id"]))].copy()

    return market, games


def build_features(market: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame()

    market = market.copy()
    market["captured_at_utc"] = _parse_ts(market["captured_at_utc"])
    market["home_spread"] = _to_float(market["home_spread"])
    market["total_line"] = _to_float(market["total_line"])
    market = market.dropna(subset=["captured_at_utc"])

    game_meta = games[["game_id", "game_datetime_utc", "home_team_id", "away_team_id"]].drop_duplicates("game_id")
    merged = market.merge(game_meta, on="game_id", how="left", suffixes=("", "_game"))

    rows: list[dict] = []
    for game_id, group in merged.groupby("game_id", dropna=True):
        g = group.sort_values("captured_at_utc").copy()

        explicit_mask = g["capture_type"].fillna("").map(_is_explicit_pregame)
        non_excluded_mask = ~g["capture_type"].fillna("").map(_exclude_capture_type)

        if explicit_mask.any():
            pregame = g[explicit_mask & non_excluded_mask].copy()
        else:
            game_dt = g["game_datetime_utc"].iloc[0]
            if pd.notna(game_dt):
                pregame = g[g["captured_at_utc"] <= (game_dt - pd.Timedelta(hours=2))].copy()
            else:
                pregame = g[non_excluded_mask].copy()

        if pregame.empty:
            continue

        game_dt = pregame["game_datetime_utc"].iloc[0]
        if pd.notna(game_dt):
            close_pool = pregame[pregame["captured_at_utc"] <= game_dt]
            if close_pool.empty:
                close_pool = pregame
            close_capture = close_pool.iloc[-1]
        else:
            close_capture = pregame.iloc[-2] if len(pregame) >= 2 else pregame.iloc[-1]

        open_capture = pregame.iloc[0]
        open_spread = open_capture["home_spread"]
        close_spread = close_capture["home_spread"]
        open_total = open_capture["total_line"]
        close_total = close_capture["total_line"]

        spread_move = float(close_spread - open_spread) if pd.notna(open_spread) and pd.notna(close_spread) else np.nan
        total_move = float(close_total - open_total) if pd.notna(open_total) and pd.notna(close_total) else np.nan

        deltas_spread = pregame["home_spread"].diff()
        deltas_total = pregame["total_line"].diff()

        steam_flag = _steam_flag(deltas_spread, spread_move, threshold=1.5)
        total_steam_flag = _steam_flag(deltas_total, total_move, threshold=2.0)

        public_flag, public_side = _resolve_public_side(pregame)
        if public_flag is None:
            rlm_flag = None
            sharp_side = None
        elif public_flag == 0:
            rlm_flag = 0
            sharp_side = None
        else:
            moved_toward = _direction(spread_move, "home", "away")
            rlm_flag = int(public_side is not None and moved_toward not in ("none", public_side))
            sharp_side = moved_toward if rlm_flag == 1 else None

        n_captures = int(pregame["captured_at_utc"].nunique())
        first_cap = pregame["captured_at_utc"].iloc[0]
        last_cap = pregame["captured_at_utc"].iloc[-1]
        hours = float((last_cap - first_cap).total_seconds() / 3600.0) if n_captures > 1 else 0.0

        spread_move_abs = 0.0 if pd.isna(spread_move) else abs(spread_move)
        line_stability_score = float(np.clip(1 - (spread_move_abs / 7.0), 0.0, 1.0))

        if steam_flag == 1 or rlm_flag == 1:
            market_confidence = "sharp"
        elif line_stability_score >= 0.85:
            market_confidence = "stable"
        elif spread_move_abs >= 2.0 and steam_flag == 0:
            market_confidence = "volatile"
        elif n_captures <= 2:
            market_confidence = "thin"
        else:
            market_confidence = "normal"

        rows.append(
            {
                "game_id": game_id,
                "game_date": open_capture.get("game_date"),
                "home_team_id": str(open_capture.get("home_team_id") or ""),
                "away_team_id": str(open_capture.get("away_team_id") or ""),
                "open_home_spread": open_spread,
                "open_total": open_total,
                "open_source": open_capture.get("capture_type"),
                "open_captured_at": open_capture["captured_at_utc"],
                "close_home_spread": close_spread,
                "close_total": close_total,
                "close_captured_at": close_capture["captured_at_utc"],
                "spread_move": 0.0 if n_captures == 1 else spread_move,
                "spread_move_abs": 0.0 if n_captures == 1 else spread_move_abs,
                "spread_move_direction": _direction(0.0 if n_captures == 1 else spread_move, "home", "away"),
                "steam_move_flag": 0 if n_captures == 1 else steam_flag,
                "spread_crossed_key_number": _crossed_key_number(open_spread, close_spread),
                "total_move": 0.0 if n_captures == 1 else total_move,
                "total_move_direction": _direction(0.0 if n_captures == 1 else total_move, "over", "under"),
                "total_steam_flag": 0 if n_captures == 1 else total_steam_flag,
                "reverse_line_movement_flag": rlm_flag,
                "sharp_side": sharp_side,
                "n_captures": n_captures,
                "hours_of_movement": hours,
                "line_stability_score": line_stability_score,
                "market_confidence_flag": market_confidence,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["game_date", "game_id"], na_position="last").reset_index(drop=True)
    return out


def print_summary(df: pd.DataFrame) -> None:
    total_games = len(df)
    steam_pct = df["steam_move_flag"].eq(1).mean() * 100 if total_games else 0.0

    if df["reverse_line_movement_flag"].notna().any():
        rlm_pct = df["reverse_line_movement_flag"].eq(1).mean() * 100
        rlm_summary = f"{rlm_pct:.1f}%"
    else:
        none_rate = df["reverse_line_movement_flag"].isna().mean() * 100 if total_games else 100.0
        rlm_summary = f"None for {none_rate:.1f}% of games"

    mean_abs_move = pd.to_numeric(df["spread_move"], errors="coerce").abs().mean()

    print(f"games processed: {total_games}")
    print(f"steam_move_flag rate: {steam_pct:.1f}%")
    print(f"reverse_line_movement_flag rate: {rlm_summary}")
    print(f"mean abs(spread_move): {0.0 if pd.isna(mean_abs_move) else round(float(mean_abs_move), 3)}")


def main() -> None:
    args = parse_args()
    filters = Filters(days_back=args.days_back, season=args.season)

    market, games = load_inputs(filters)
    features = build_features(market, games)

    if features.empty:
        print("No valid pregame captures found; output file not written.")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(features)} rows to {OUTPUT_PATH}")
    print_summary(features)

    # Required join validation block.
    games_df = pd.read_csv("data/games.csv", dtype=str)
    lmf = pd.read_csv("data/line_movement_features.csv", dtype=str)
    shared = set(games_df["game_id"]).intersection(set(lmf["game_id"]))
    print(f"games with line movement data: {len(shared)} / {len(games_df)}")
    pct = len(shared) / max(len(games_df), 1)
    if pct < 0.3:
        print("[WARN] Low join rate — check game_id format mismatch")

    print(
        "NOTE: For build_training_data.py (separate pass), add a LEFT JOIN on line_movement_features by game_id "
        "after market_lines and select: spread_move, total_move, steam_move_flag, reverse_line_movement_flag, "
        "spread_crossed_key_number, line_stability_score, market_confidence_flag."
    )


if __name__ == "__main__":
    main()
