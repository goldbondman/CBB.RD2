#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_PATH = DATA_DIR / "situational_features.csv"

CONF_TOURNAMENT_DATES = {
    "ACC": "2026-03-10",
    "Big 12": "2026-03-11",
    "Big Ten": "2026-03-11",
    "Big East": "2026-03-12",
    "SEC": "2026-03-12",
    "Pac-12": "2026-03-11",
    "American": "2026-03-07",
    "Mountain West": "2026-03-04",
    "Atlantic 10": "2026-03-07",
    "WCC": "2026-03-05",
}

RIVALRY_PAIRS = {
    frozenset(["duke", "north-carolina"]),
    frozenset(["kansas", "kansas-state"]),
    frozenset(["kentucky", "louisville"]),
    frozenset(["michigan", "michigan-state"]),
    frozenset(["ohio-state", "michigan"]),
    frozenset(["villanova", "georgetown"]),
    frozenset(["ucla", "usc"]),
    frozenset(["indiana", "purdue"]),
    frozenset(["florida", "florida-state"]),
    frozenset(["north-carolina", "nc-state"]),
    frozenset(["gonzaga", "saint-marys"]),
    frozenset(["syracuse", "georgetown"]),
    frozenset(["arizona", "arizona-state"]),
    frozenset(["arkansas", "missouri"]),
    frozenset(["alabama", "auburn"]),
}


def normalize_slug(team_name: str) -> str:
    s = str(team_name).lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"\s+", "-", s)
    return s




def canonical_rivalry_slug(team_name: str) -> str:
    slug = normalize_slug(team_name)
    patterns = [
        "north-carolina", "kansas-state", "kansas", "kentucky", "louisville",
        "michigan-state", "michigan", "ohio-state", "villanova", "georgetown",
        "ucla", "usc", "indiana", "purdue", "florida-state", "florida",
        "nc-state", "gonzaga", "saint-marys", "syracuse", "arizona-state",
        "arizona", "arkansas", "missouri", "alabama", "auburn", "duke",
    ]
    for ptn in sorted(patterns, key=len, reverse=True):
        if ptn in slug:
            return ptn
    parts = slug.split("-")
    return "-".join(parts[:-1]) if len(parts) > 1 else slug

def normalize_game_id(v: object) -> str:
    if pd.isna(v):
        return ""
    t = str(v).strip()
    return t[:-2] if t.endswith(".0") else t


def derive_season(dt: pd.Timestamp) -> int:
    if pd.isna(dt):
        return pd.NA
    return int(dt.year + 1) if dt.month >= 11 else int(dt.year)


def build_team_games(games: pd.DataFrame) -> pd.DataFrame:
    common = ["game_id", "game_datetime_utc", "game_date", "season", "neutral_site", "completed"]
    home = games[common + ["home_team_id", "away_team_id", "home_team", "away_team", "home_conference", "away_conference", "home_score", "away_score"]].copy()
    home.columns = common + ["team_id", "opponent_id", "team_name", "opponent_name", "conference", "opp_conference", "team_score", "opp_score"]
    home["home_away"] = "home"

    away = games[common + ["away_team_id", "home_team_id", "away_team", "home_team", "away_conference", "home_conference", "away_score", "home_score"]].copy()
    away.columns = common + ["team_id", "opponent_id", "team_name", "opponent_name", "conference", "opp_conference", "team_score", "opp_score"]
    away["home_away"] = "away"

    out = pd.concat([home, away], ignore_index=True)
    out["team_id"] = pd.to_numeric(out["team_id"], errors="coerce").astype("Int64")
    out["opponent_id"] = pd.to_numeric(out["opponent_id"], errors="coerce").astype("Int64")
    out["actual_margin"] = pd.to_numeric(out["team_score"], errors="coerce") - pd.to_numeric(out["opp_score"], errors="coerce")
    out["is_conference_game"] = np.where(
        out["conference"].notna() & out["opp_conference"].notna(),
        (out["conference"].astype(str) == out["opp_conference"].astype(str)).astype(int),
        pd.NA,
    )
    out = out.sort_values(["season", "team_id", "game_datetime_utc", "game_id"]).reset_index(drop=True)
    return out


def build_rankings_map(rankings_path: Path, weighted_path: Path, team_games: pd.DataFrame) -> tuple[dict[int, float], dict[int, float], set[int], dict[int, int]]:
    if rankings_path.exists():
        rankings = pd.read_csv(rankings_path, low_memory=False)
        rankings["team_id"] = pd.to_numeric(rankings.get("team_id"), errors="coerce").astype("Int64")
        rank_col = pd.to_numeric(rankings.get("rank"), errors="coerce")
        cage_em = pd.to_numeric(rankings.get("cage_em"), errors="coerce")
        ranking_map = dict(zip(rankings["team_id"].dropna().astype(int), rank_col))
        em_map = dict(zip(rankings["team_id"].dropna().astype(int), cage_em))
    else:
        weighted = pd.read_csv(weighted_path, low_memory=False)
        weighted["team_id"] = pd.to_numeric(weighted.get("team_id"), errors="coerce")
        weighted["game_datetime_utc"] = pd.to_datetime(weighted.get("game_datetime_utc"), utc=True, errors="coerce")
        weighted["net_rtg"] = pd.to_numeric(weighted.get("net_rtg"), errors="coerce")
        latest = weighted.sort_values("game_datetime_utc").dropna(subset=["team_id"]).drop_duplicates("team_id", keep="last")
        latest = latest.sort_values("net_rtg", ascending=False).reset_index(drop=True)
        latest["rank"] = np.arange(1, len(latest) + 1)
        ranking_map = dict(zip(latest["team_id"].astype(int), latest["rank"]))
        em_map = dict(zip(latest["team_id"].astype(int), latest["net_rtg"]))

    ordered = sorted(((tid, v) for tid, v in em_map.items() if pd.notna(v)), key=lambda x: x[1], reverse=True)
    em_rank = {tid: i + 1 for i, (tid, _) in enumerate(ordered)}
    top25 = {tid for tid, r in ranking_map.items() if pd.notna(r) and r <= 25}
    return ranking_map, em_map, top25, em_rank


def main() -> None:
    parser = argparse.ArgumentParser(description="Build team-game situational features.")
    parser.add_argument("--season", type=int)
    parser.add_argument("--days-back", type=int)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    games = pd.read_csv(DATA_DIR / "games.csv", low_memory=False)
    games["game_id"] = games.get("game_id", games.get("event_id")).map(normalize_game_id)
    games["game_datetime_utc"] = pd.to_datetime(games["game_datetime_utc"], utc=True, errors="coerce")
    games["game_date"] = games["game_datetime_utc"].dt.date
    games["season"] = games["game_datetime_utc"].map(derive_season)

    if args.season is not None:
        games = games[games["season"] == args.season].copy()
    if args.days_back is not None and not games.empty:
        cutoff = games["game_datetime_utc"].max() - pd.Timedelta(days=args.days_back)
        games = games[games["game_datetime_utc"] >= cutoff].copy()

    team_games = build_team_games(games)
    ranking_map, em_map, top25, em_rank = build_rankings_map(DATA_DIR / "cbb_rankings.csv", DATA_DIR / "team_game_weighted.csv", team_games)

    team_games["current_opp_rank"] = team_games["opponent_id"].map(lambda x: ranking_map.get(int(x)) if pd.notna(x) else np.nan)

    team_games["next_game_dt"] = team_games.groupby(["season", "team_id"])["game_datetime_utc"].shift(-1)
    team_games["next_opp_id"] = team_games.groupby(["season", "team_id"])["opponent_id"].shift(-1)
    team_games["next_gap_days"] = (team_games["next_game_dt"] - team_games["game_datetime_utc"]).dt.days
    has_next = team_games["next_game_dt"].notna() & (team_games["next_gap_days"] <= 10)
    team_games["next_opp_is_ranked"] = np.where(has_next, team_games["next_opp_id"].map(lambda x: int(x) in top25 if pd.notna(x) else False).astype(int), pd.NA)

    team_games["lookahead_magnitude"] = np.where(
        has_next,
        team_games["next_opp_id"].map(lambda x: em_map.get(int(x), np.nan) if pd.notna(x) else np.nan) - team_games["opponent_id"].map(lambda x: em_map.get(int(x), np.nan) if pd.notna(x) else np.nan),
        pd.NA,
    )

    team_games["lookahead_flag"] = np.where(
        has_next,
        ((team_games["next_opp_is_ranked"] == 1) & (team_games["current_opp_rank"].isna() | (pd.to_numeric(team_games["current_opp_rank"], errors="coerce") > 50))).astype(int),
        pd.NA,
    )

    team_games["prev_game_dt"] = team_games.groupby(["season", "team_id"])["game_datetime_utc"].shift(1)
    team_games["prev_opp_id"] = team_games.groupby(["season", "team_id"])["opponent_id"].shift(1)
    team_games["prev_margin"] = team_games.groupby(["season", "team_id"])["actual_margin"].shift(1)
    team_games["prev_gap_days"] = (team_games["game_datetime_utc"] - team_games["prev_game_dt"]).dt.days
    has_prev_10 = team_games["prev_game_dt"].notna() & (team_games["prev_gap_days"] <= 10)

    team_games["prev_opp_rank"] = team_games["prev_opp_id"].map(lambda x: ranking_map.get(int(x)) if pd.notna(x) else np.nan)
    team_games["prev_opp_was_ranked"] = np.where(has_prev_10, (pd.to_numeric(team_games["prev_opp_rank"], errors="coerce") <= 25).astype(int), 0)
    team_games["prev_game_won"] = np.where(has_prev_10, (pd.to_numeric(team_games["prev_margin"], errors="coerce") > 0).astype(int), pd.NA)
    team_games["prev_game_margin"] = np.where(has_prev_10, team_games["prev_margin"], pd.NA)

    curr_unranked = team_games["current_opp_rank"].isna() | (pd.to_numeric(team_games["current_opp_rank"], errors="coerce") > 25)
    team_games["letdown_flag"] = ((team_games["prev_opp_was_ranked"] == 1) & (team_games["prev_game_won"] == 1) & curr_unranked).astype(int)

    prev_inv = 1.0 / np.maximum(pd.to_numeric(team_games["prev_opp_rank"], errors="coerce").fillna(999), 1)
    curr_rank_for_ease = pd.to_numeric(team_games["current_opp_rank"], errors="coerce").fillna(125)
    curr_ease = np.maximum(0, (curr_rank_for_ease - 25) / 100)
    team_games["emotional_letdown_score"] = np.clip(pd.to_numeric(team_games["prev_game_margin"], errors="coerce").fillna(0) * prev_inv * curr_ease, 0, 1)

    prev_win = (pd.to_numeric(team_games["actual_margin"], errors="coerce") > 0).astype(int)
    team_games["consecutive_losses"] = 0
    team_games["consecutive_conf_losses"] = 0
    for (_, _), g in team_games.groupby(["season", "team_id"], sort=False):
        loss_run = 0
        conf_loss_run = 0
        idxs = g.index.tolist()
        for idx in idxs:
            team_games.at[idx, "consecutive_losses"] = loss_run
            team_games.at[idx, "consecutive_conf_losses"] = conf_loss_run
            is_loss = prev_win.loc[idx] == 0
            loss_run = loss_run + 1 if is_loss else 0
            conf_game = team_games.at[idx, "is_conference_game"]
            if pd.notna(conf_game) and int(conf_game) == 1:
                conf_loss_run = conf_loss_run + 1 if is_loss else 0

    team_games["prev_loss_margin"] = np.where(pd.to_numeric(team_games["prev_margin"], errors="coerce") < 0, team_games["prev_margin"], pd.NA)
    team_games["bounce_back_flag"] = ((team_games["consecutive_losses"] >= 2) & (pd.to_numeric(team_games["prev_loss_margin"], errors="coerce") < -10)).astype(int)

    team_games["must_win_flag"] = (team_games["consecutive_conf_losses"] >= 3).astype(int)

    # Prior matchup for revenge (game-level then expanded)
    game_level = games[["game_id", "season", "game_datetime_utc", "home_team_id", "away_team_id", "home_score", "away_score", "neutral_site"]].copy()
    game_level["home_team_id"] = pd.to_numeric(game_level["home_team_id"], errors="coerce").astype("Int64")
    game_level["away_team_id"] = pd.to_numeric(game_level["away_team_id"], errors="coerce").astype("Int64")
    game_level["pair_key"] = game_level.apply(lambda r: tuple(sorted((int(r["home_team_id"]), int(r["away_team_id"])))) if pd.notna(r["home_team_id"]) and pd.notna(r["away_team_id"]) else None, axis=1)
    game_level = game_level.sort_values(["season", "pair_key", "game_datetime_utc", "game_id"]).reset_index(drop=True)
    game_level["prior_game_id"] = game_level.groupby(["season", "pair_key"])["game_id"].shift(1)
    game_level["played_earlier_this_season"] = game_level["prior_game_id"].notna().astype(int)

    prior_lookup = game_level.set_index("game_id")[["home_score", "away_score"]]
    game_level["prior_home_score"] = game_level["prior_game_id"].map(prior_lookup["home_score"])
    game_level["prior_away_score"] = game_level["prior_game_id"].map(prior_lookup["away_score"])
    game_level["prior_matchup_margin"] = game_level["prior_home_score"] - game_level["prior_away_score"]
    game_level["prior_matchup_winner"] = np.where(
        game_level["played_earlier_this_season"] == 1,
        np.where(game_level["prior_matchup_margin"] > 0, "home", "away"),
        pd.NA,
    )

    revenge = game_level[["game_id", "played_earlier_this_season", "prior_matchup_winner", "prior_matchup_margin"]].copy()
    team_games = team_games.merge(revenge, on="game_id", how="left")

    team_games["revenge_flag"] = 0
    lost_prev_home = (team_games["home_away"] == "home") & (team_games["prior_matchup_winner"] == "away")
    lost_prev_away = (team_games["home_away"] == "away") & (team_games["prior_matchup_winner"] == "home")
    team_games.loc[lost_prev_home | lost_prev_away, "revenge_flag"] = 1
    team_games["revenge_margin"] = np.where(team_games["played_earlier_this_season"] == 1, team_games["prior_matchup_margin"].abs(), pd.NA)

    # Bubble + stakes
    team_games["team_cage_rank"] = team_games["team_id"].map(lambda x: em_rank.get(int(x)) if pd.notna(x) else np.nan)
    team_games["is_bubble_team"] = ((team_games["team_cage_rank"] >= 30) & (team_games["team_cage_rank"] <= 60)).astype(int)
    team_games["is_safe_team"] = (team_games["team_cage_rank"] <= 25).astype(int)

    game_day = pd.to_datetime(team_games["game_date"], errors="coerce")
    conf_date = pd.to_datetime(team_games["conference"].map(lambda c: CONF_TOURNAMENT_DATES.get(str(c), "2026-03-05")), errors="coerce")
    team_games["days_to_conf_tournament"] = np.where(game_day.dt.month >= 2, (conf_date - game_day).dt.days, pd.NA)
    team_games["late_season_flag"] = ((pd.to_numeric(team_games["days_to_conf_tournament"], errors="coerce") <= 14)).astype(int)
    team_games["bubble_pressure_flag"] = ((team_games["is_bubble_team"] == 1) & (team_games["late_season_flag"] == 1)).astype(int)

    # Rest/fatigue
    has_prev_30 = team_games["prev_game_dt"].notna() & ((team_games["game_datetime_utc"] - team_games["prev_game_dt"]).dt.days <= 30)
    team_games["rest_days"] = np.where(has_prev_30, (team_games["game_datetime_utc"] - team_games["prev_game_dt"]).dt.days, pd.NA)
    team_games["short_rest_flag"] = np.where(team_games["rest_days"].isna(), pd.NA, (pd.to_numeric(team_games["rest_days"], errors="coerce") <= 1).astype(int))
    team_games["extended_rest_flag"] = np.where(team_games["rest_days"].isna(), pd.NA, (pd.to_numeric(team_games["rest_days"], errors="coerce") >= 6).astype(int))
    team_games["optimal_rest_flag"] = np.where(team_games["rest_days"].isna(), pd.NA, pd.to_numeric(team_games["rest_days"], errors="coerce").isin([2, 3, 4]).astype(int))

    team_games["games_in_last_7_days"] = 0
    team_games["games_in_last_14_days"] = 0
    for (_, _), g in team_games.groupby(["season", "team_id"], sort=False):
        dts = g["game_datetime_utc"].tolist()
        idxs = g.index.tolist()
        for i, idx in enumerate(idxs):
            now = dts[i]
            prev_dts = dts[:i]
            team_games.at[idx, "games_in_last_7_days"] = sum((now - d).days <= 7 for d in prev_dts)
            team_games.at[idx, "games_in_last_14_days"] = sum((now - d).days <= 14 for d in prev_dts)

    sr = pd.to_numeric(team_games["short_rest_flag"], errors="coerce").fillna(0)
    team_games["fatigue_flag"] = ((team_games["games_in_last_7_days"] >= 3) | ((team_games["games_in_last_14_days"] >= 5) & (sr == 1))).astype(int)

    team_games["is_neutral_site"] = team_games["neutral_site"].fillna(False).astype(int)
    team_games["home_field_advantage_applicable"] = ((team_games["is_neutral_site"] == 0) & (team_games["home_away"] == "home")).astype(int)

    team_games["team_slug"] = team_games["team_name"].map(canonical_rivalry_slug)
    team_games["opp_slug"] = team_games["opponent_name"].map(canonical_rivalry_slug)
    team_games["is_rivalry_game"] = team_games.apply(lambda r: int(frozenset([r["team_slug"], r["opp_slug"]]) in RIVALRY_PAIRS), axis=1)

    team_games["situational_edge_score"] = (
        0.20 * team_games["revenge_flag"] * np.sign(pd.to_numeric(team_games["revenge_margin"], errors="coerce").fillna(0))
        + 0.15 * team_games["bounce_back_flag"]
        - 0.15 * team_games["letdown_flag"]
        - 0.10 * pd.to_numeric(team_games["lookahead_flag"], errors="coerce").fillna(0)
        + 0.15 * team_games["bubble_pressure_flag"]
        + 0.10 * team_games["must_win_flag"]
        - 0.10 * team_games["fatigue_flag"]
        - 0.10 * pd.to_numeric(team_games["extended_rest_flag"], errors="coerce").fillna(0)
        + 0.05 * team_games["is_rivalry_game"]
    )

    # home/away rest output
    rest_lookup = team_games[["game_id", "team_id", "rest_days"]].rename(columns={"team_id": "opponent_id", "rest_days": "away_rest_days"})
    team_games = team_games.merge(rest_lookup, on=["game_id", "opponent_id"], how="left")
    team_games["home_rest_days"] = np.where(team_games["home_away"] == "home", team_games["rest_days"], team_games["away_rest_days"])
    team_games["away_rest_days"] = np.where(team_games["home_away"] == "home", team_games["away_rest_days"], team_games["rest_days"])
    team_games["rest_delta"] = np.where(team_games["home_away"] == "home", team_games["home_rest_days"] - team_games["away_rest_days"], team_games["away_rest_days"] - team_games["home_rest_days"])

    out_cols = [
        "game_id", "team_id", "opponent_id", "game_date", "game_datetime_utc", "home_away", "season",
        "next_opp_is_ranked", "current_opp_rank", "lookahead_flag", "lookahead_magnitude",
        "prev_opp_was_ranked", "prev_game_won", "prev_game_margin", "letdown_flag", "emotional_letdown_score",
        "consecutive_losses", "prev_loss_margin", "bounce_back_flag",
        "played_earlier_this_season", "prior_matchup_winner", "prior_matchup_margin", "revenge_flag", "revenge_margin",
        "days_to_conf_tournament", "late_season_flag", "is_bubble_team", "is_safe_team", "bubble_pressure_flag",
        "consecutive_conf_losses", "must_win_flag",
        "home_rest_days", "away_rest_days", "rest_delta", "short_rest_flag", "extended_rest_flag", "optimal_rest_flag",
        "games_in_last_7_days", "games_in_last_14_days", "fatigue_flag",
        "is_neutral_site", "home_field_advantage_applicable", "is_conference_game", "is_rivalry_game", "situational_edge_score",
    ]
    for c in out_cols:
        if c not in team_games.columns:
            team_games[c] = pd.NA
    out = team_games[out_cols].copy()

    if args.report:
        print("SITUATIONAL FLAG FREQUENCIES (team-game level):")
        for flag in ["lookahead_flag", "letdown_flag", "bounce_back_flag", "revenge_flag", "bubble_pressure_flag", "must_win_flag", "fatigue_flag", "is_rivalry_game"]:
            pct = pd.to_numeric(out[flag], errors="coerce").mean()
            pct = 0 if pd.isna(pct) else pct
            print(f"  {flag:<22} {pct:.1%} of games")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
