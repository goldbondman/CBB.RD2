# home/away splits
import logging
import math
from typing import List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def build_team_schedule_index(schedule_df: pd.DataFrame) -> pd.DataFrame:
    if schedule_df.empty:
        return pd.DataFrame(columns=["team_id", "opponent_id", "game_datetime_utc"])

    base = schedule_df.copy()
    home = pd.DataFrame({
        "team_id": base.get("home_team_id", ""),
        "opponent_id": base.get("away_team_id", ""),
        "game_datetime_utc": base.get("game_datetime_utc", ""),
    })
    away = pd.DataFrame({
        "team_id": base.get("away_team_id", ""),
        "opponent_id": base.get("home_team_id", ""),
        "game_datetime_utc": base.get("game_datetime_utc", ""),
    })
    out = pd.concat([home, away], ignore_index=True)
    out["team_id"] = out["team_id"].astype(str)
    out["opponent_id"] = out["opponent_id"].astype(str)
    out["game_datetime_utc"] = out["game_datetime_utc"].astype(str)
    out = out.sort_values(["team_id", "game_datetime_utc"]).reset_index(drop=True)
    return out


def detect_trap_game(team_id: str, game_date: str, schedule_df: pd.DataFrame, rankings_df: pd.DataFrame) -> dict:
    """Flag potential trap-game spot for a ranked team facing weak opposition."""
    if schedule_df.empty or rankings_df.empty:
        return {"trap_game_flag": False, "trap_game_reason": ""}

    team_games = schedule_df[schedule_df["team_id"].astype(str) == str(team_id)].sort_values("game_datetime_utc").reset_index(drop=True)
    mask = team_games["game_datetime_utc"].astype(str).str[:10] == str(game_date)[:10]
    game_idx = team_games[mask].index
    if len(game_idx) == 0:
        return {"trap_game_flag": False, "trap_game_reason": ""}

    pos = int(game_idx[0])
    prev_game = team_games.iloc[pos - 1] if pos > 0 else None
    next_game = team_games.iloc[pos + 1] if pos < len(team_games) - 1 else None

    rank_map = dict(zip(rankings_df["team_id"].astype(str), pd.to_numeric(rankings_df["cage_rank"], errors="coerce").fillna(999)))
    opp_rank = rank_map.get(str(team_games.iloc[pos].get("opponent_id", "")), 999)
    prev_rank = rank_map.get(str(prev_game.get("opponent_id", "")), 999) if prev_game is not None else 999
    next_rank = rank_map.get(str(next_game.get("opponent_id", "")), 999) if next_game is not None else 999
    team_rank = rank_map.get(str(team_id), 999)

    is_trap = bool(team_rank <= 40 and opp_rank > 100 and (prev_rank <= 40 or next_rank <= 40))
    return {
        "trap_game_flag": is_trap,
        "trap_game_reason": (
            f"Ranked team (#{int(team_rank)}) vs weak opp (#{int(opp_rank)}) between quality games"
            if is_trap else ""
        ),
    }


def detect_revenge_spot(home_team_id: str, away_team_id: str, game_date: str, results_df: pd.DataFrame, lookback_days: int = 45) -> dict:
    """Check if either side recently lost this same matchup and is in a revenge spot."""
    if results_df.empty:
        return {"revenge_flag": False, "revenge_team": "", "revenge_margin": None}

    cutoff = pd.Timestamp(game_date, tz="UTC") - pd.Timedelta(days=lookback_days)
    results = results_df.copy()
    results["dt"] = pd.to_datetime(results.get("game_datetime_utc"), utc=True, errors="coerce")
    recent = results[results["dt"] >= cutoff]

    home_lost = recent[(recent["home_team_id"].astype(str) == str(home_team_id)) &
                       (recent["away_team_id"].astype(str) == str(away_team_id)) &
                       (pd.to_numeric(recent["home_score"], errors="coerce") < pd.to_numeric(recent["away_score"], errors="coerce"))]
    away_lost = recent[(recent["home_team_id"].astype(str) == str(away_team_id)) &
                       (recent["away_team_id"].astype(str) == str(home_team_id)) &
                       (pd.to_numeric(recent["home_score"], errors="coerce") > pd.to_numeric(recent["away_score"], errors="coerce"))]

    if len(home_lost) > 0:
        r = home_lost.sort_values("dt").iloc[-1]
        return {"revenge_flag": True, "revenge_team": "home", "revenge_margin": int(float(r["away_score"]) - float(r["home_score"]))}
    if len(away_lost) > 0:
        r = away_lost.sort_values("dt").iloc[-1]
        return {"revenge_flag": True, "revenge_team": "away", "revenge_margin": int(float(r["home_score"]) - float(r["away_score"]))}

    return {"revenge_flag": False, "revenge_team": "", "revenge_margin": None}


def line_shopping_advisory(model_spread: float, closing_line: Optional[float]) -> str:
    """Flag near-threshold edges where shopping a half-point matters."""
    if closing_line is None:
        return ""
    edge = abs(model_spread - closing_line)
    if 2.0 <= edge <= 4.0:
        key_number = None
        for n in [1, 2, 3, 5, 6, 7]:
            if abs(closing_line) % n < 0.6:
                key_number = n
                break
        if key_number:
            return (f"LINE SHOP: edge {edge:.1f}pts — closing on key number {key_number}. "
                    f"Half point could be critical.")
    return ""


def model_total(team_a: dict, team_b: dict, log_actuals: bool = False) -> dict:
    """Dedicated totals model using pace + ortg/drtg interaction."""
    LEAGUE_AVG_PACE = 67.2
    LEAGUE_AVG_ORTG = 110.0

    def get_metric(team: dict, cols: List[str], default: float) -> float:
        for col in cols:
            val = team.get(col)
            if val is not None and pd.notna(val):
                try:
                    return float(val)
                except (TypeError, ValueError):
                    continue
        return default

    _home_pace = get_metric(team_a, ["pace", "adj_pace", "cage_t"], LEAGUE_AVG_PACE)
    _away_pace = get_metric(team_b, ["pace", "adj_pace", "cage_t"], LEAGUE_AVG_PACE)

    _home_ortg = get_metric(team_a, ["ortg", "adj_ortg", "cage_o"], LEAGUE_AVG_ORTG)
    _away_ortg = get_metric(team_b, ["ortg", "adj_ortg", "cage_o"], LEAGUE_AVG_ORTG)

    # [DIAG] Log first game metrics to verify fallback behavior
    if not hasattr(model_total, "_logged"):
        logger.info(
            "[DIAG] model_total first game metrics | home_pace: %.1f, away_pace: %.1f, home_ortg: %.1f, away_ortg: %.1f",
            _home_pace, _away_pace, _home_ortg, _away_ortg
        )
        model_total._logged = True

    projected_poss = round((_home_pace + _away_pace) / 2, 1)
    projected_total = round(((_home_ortg + _away_ortg) * projected_poss) / 100, 1)

    if log_actuals:
        logger.info(
            "model_total first call: home_pace=%.1f, away_pace=%.1f, home_ortg=%.1f, away_ortg=%.1f -> total=%.1f",
            _home_pace, _away_pace, _home_ortg, _away_ortg, projected_total
        )

    def _safe_int(value, default: int = 0) -> int:
        """Convert numeric-like values to int, treating NaN/None/empty as default."""
        if value is None or pd.isna(value):
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    _home_games = _safe_int(team_a.get("games_played") or team_a.get("game_number") or 0, default=0)
    _away_games = _safe_int(team_b.get("games_played") or team_b.get("game_number") or 0, default=0)
    _min_games = min(_home_games, _away_games)
    total_confidence_adj = round(min(1.0, 0.6 + (_min_games / 25.0) * 0.4), 3)

    score_a = (_home_ortg / 100) * projected_poss
    score_b = (_away_ortg / 100) * projected_poss

    return {
        "projected_total": projected_total,
        "projected_score_a": round(score_a, 1),
        "projected_score_b": round(score_b, 1),
        "projected_poss": projected_poss,
        "total_confidence_adj": total_confidence_adj,
    }
