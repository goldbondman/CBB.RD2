import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
PLAYER_LOGS_PATH = DATA_DIR / "player_game_logs.csv"
TEAM_INJURY_IMPACT_PATH = DATA_DIR / "team_injury_impact.csv"
OUTPUT_PATH = DATA_DIR / "player_availability_features.csv"


def gini(array: list[float]) -> float | None:
    array = sorted(array)
    n = len(array)
    if n == 0:
        return None
    total = sum(array)
    if total <= 0:
        return 0.0
    cumsum = 0.0
    for i, v in enumerate(array):
        cumsum += (2 * (i + 1) - n - 1) * v
    return cumsum / (n * total)


def _safe_usage_rate(df: pd.DataFrame) -> pd.Series:
    fga = pd.to_numeric(df.get("fga", 0), errors="coerce").fillna(0.0)
    fta = pd.to_numeric(df.get("fta", 0), errors="coerce").fillna(0.0)
    tov = pd.to_numeric(df.get("tov", 0), errors="coerce").fillna(0.0)
    return fga + 0.44 * fta + tov


def _prepare_player_logs(player_df: pd.DataFrame) -> pd.DataFrame:
    df = player_df.copy()
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], utc=True, errors="coerce")
    df["team_id"] = df["team_id"].astype("string")
    df["game_id"] = df["event_id"].astype("string")
    df["athlete_id"] = df["athlete_id"].astype("string")
    df["min"] = pd.to_numeric(df.get("min", 0), errors="coerce").fillna(0.0)

    dnp_raw = df.get("did_not_play", False)
    dnp = dnp_raw.astype(bool) if isinstance(dnp_raw, pd.Series) else pd.Series(False, index=df.index)
    df["eligible_rotation"] = (~dnp) & (df["min"] > 0)
    df["usage_rate"] = _safe_usage_rate(df)
    df["season"] = df["game_datetime_utc"].dt.year
    return df


def _compute_minutes_available(team_game: pd.DataFrame) -> pd.Series:
    prior5 = (
        team_game.groupby("team_id")["team_minutes"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    )
    return team_game["team_minutes"] / prior5


def build_player_availability_features(player_logs_path: Path, days_back: int | None = None) -> pd.DataFrame:
    player_df = pd.read_csv(player_logs_path)
    player_df = _prepare_player_logs(player_df)

    game_keys = ["game_id", "team_id", "game_datetime_utc", "season"]

    # Team-game minutes (actual, for backtesting)
    team_game = (
        player_df.groupby(game_keys, as_index=False)["min"]
        .sum()
        .rename(columns={"min": "team_minutes"})
        .sort_values(["team_id", "game_datetime_utc", "game_id"])
    )
    team_game["minutes_available_pct"] = _compute_minutes_available(team_game)

    # Season-long player usage baseline from prior games only
    player_usage_game = (
        player_df[player_df["eligible_rotation"]]
        .groupby(["team_id", "season", "athlete_id", "game_datetime_utc", "game_id"], as_index=False)["usage_rate"]
        .sum()
        .sort_values(["team_id", "season", "athlete_id", "game_datetime_utc", "game_id"])
    )
    player_usage_game["season_usage_prior"] = (
        player_usage_game.groupby(["team_id", "season", "athlete_id"])["usage_rate"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )

    # Expected stars per team-season (top-2 by prior season-average usage)
    games = (
        team_game[["team_id", "season", "game_datetime_utc", "game_id"]]
        .drop_duplicates()
        .sort_values(["team_id", "season", "game_datetime_utc", "game_id"])
    )

    star_candidates = player_usage_game[["team_id", "season", "game_datetime_utc", "game_id", "athlete_id", "season_usage_prior"]]
    stars_per_game = []
    for (team_id, season), group in games.groupby(["team_id", "season"], sort=False):
        group = group.sort_values(["game_datetime_utc", "game_id"])
        candidates = star_candidates[(star_candidates["team_id"] == team_id) & (star_candidates["season"] == season)]
        for row in group.itertuples(index=False):
            prior = candidates[candidates["game_datetime_utc"] < row.game_datetime_utc]
            if prior.empty:
                continue
            latest = prior.sort_values(["game_datetime_utc", "game_id"]).groupby("athlete_id", as_index=False).tail(1)
            latest = latest.dropna(subset=["season_usage_prior"])
            if latest.empty:
                continue
            top2 = latest.nlargest(2, "season_usage_prior")
            for _, r in top2.iterrows():
                stars_per_game.append(
                    {
                        "team_id": team_id,
                        "season": season,
                        "game_datetime_utc": row.game_datetime_utc,
                        "game_id": row.game_id,
                        "athlete_id": r["athlete_id"],
                        "expected_usage": r["season_usage_prior"],
                    }
                )

    stars_df = pd.DataFrame(stars_per_game)
    if not stars_df.empty:
        actual_usage = player_usage_game[["team_id", "season", "game_datetime_utc", "game_id", "athlete_id", "usage_rate"]]
        star_eval = stars_df.merge(
            actual_usage,
            on=["team_id", "season", "game_datetime_utc", "game_id", "athlete_id"],
            how="left",
        )
        star_eval["usage_rate"] = star_eval["usage_rate"].fillna(0.0)
        star_score = (
            star_eval.groupby(game_keys, as_index=False)[["usage_rate", "expected_usage"]]
            .sum()
            .rename(columns={"usage_rate": "star_usage_actual", "expected_usage": "star_usage_expected"})
        )
        star_score["star_availability_score"] = star_score["star_usage_actual"] / star_score["star_usage_expected"].replace(0, np.nan)
        star_score = star_score[game_keys + ["star_availability_score"]]
    else:
        star_score = pd.DataFrame(columns=game_keys + ["star_availability_score"])

    # Rotation sets
    rotation = (
        player_df[player_df["eligible_rotation"]]
        .sort_values(["team_id", "game_datetime_utc", "game_id", "min"], ascending=[True, True, True, False])
    )
    top7 = rotation.groupby(game_keys).head(7).groupby(game_keys)["athlete_id"].apply(set).reset_index(name="top7_set")
    top5 = rotation.groupby(game_keys).head(5).groupby(game_keys)["athlete_id"].apply(set).reset_index(name="top5_set")

    continuity_base = team_game[game_keys].merge(top7, on=game_keys, how="left").sort_values(["team_id", "game_datetime_utc", "game_id"])

    continuity_values = []
    for _, group in continuity_base.groupby("team_id", sort=False):
        sets = group["top7_set"].tolist()
        for i in range(len(group)):
            if i < 3 or not isinstance(sets[i], set) or len(sets[i]) == 0:
                continuity_values.append(None)
                continue
            overlaps = []
            cur = sets[i]
            for j in [i - 1, i - 2, i - 3]:
                prev = sets[j]
                if isinstance(prev, set) and len(prev) > 0:
                    overlaps.append(len(cur.intersection(prev)) / 7.0)
                else:
                    overlaps.append(0.0)
            continuity_values.append(float(np.mean(overlaps)))
    continuity_base["lineup_continuity_l3"] = continuity_values

    starter_base = team_game[game_keys].merge(top5, on=game_keys, how="left").sort_values(["team_id", "game_datetime_utc", "game_id"])
    new_starter = []
    for _, group in starter_base.groupby("team_id", sort=False):
        sets = group["top5_set"].tolist()
        for i in range(len(group)):
            if i < 2:
                new_starter.append(None)
                continue
            cur = sets[i] if isinstance(sets[i], set) else set()
            prev_union = set()
            if isinstance(sets[i - 1], set):
                prev_union |= sets[i - 1]
            if isinstance(sets[i - 2], set):
                prev_union |= sets[i - 2]
            new_starter.append(int(any(p not in prev_union for p in cur)) if cur else 0)
    starter_base["new_starter_flag"] = new_starter

    # Usage concentration on top-7 rotation
    usage_rotation = (
        rotation.assign(rank=rotation.groupby(game_keys)["min"].rank(method="first", ascending=False))
        .query("rank <= 7")
    )
    usage_conc = []
    for keys, g in usage_rotation.groupby(game_keys, sort=False):
        vals = g["usage_rate"].dropna().astype(float).tolist()
        total = sum(vals)
        usage_conc.append(
            {
                "game_id": keys[0],
                "team_id": keys[1],
                "game_datetime_utc": keys[2],
                "season": keys[3],
                "usage_gini": gini(vals),
                "top1_usage_share": (max(vals) / total) if total > 0 and vals else None,
            }
        )
    usage_conc_df = pd.DataFrame(usage_conc)

    # Injury impact merge
    if TEAM_INJURY_IMPACT_PATH.exists():
        impact_df = pd.read_csv(TEAM_INJURY_IMPACT_PATH)
        impact_df["game_id"] = impact_df["event_id"].astype("string")
        impact_df["team_id"] = impact_df["team_id"].astype("string")
        impact_df = impact_df.rename(columns={"team_injury_load": "injury_impact_delta"})
        impact_df = impact_df[["game_id", "team_id", "injury_impact_delta"]].drop_duplicates()
    else:
        impact_df = pd.DataFrame(columns=["game_id", "team_id", "injury_impact_delta"])

    out = team_game[game_keys + ["minutes_available_pct"]]
    out = out.merge(star_score, on=game_keys, how="left")
    out = out.merge(continuity_base[game_keys + ["lineup_continuity_l3"]], on=game_keys, how="left")
    out = out.merge(starter_base[game_keys + ["new_starter_flag"]], on=game_keys, how="left")
    out = out.merge(usage_conc_df, on=game_keys, how="left")
    out = out.merge(impact_df, on=["game_id", "team_id"], how="left")

    # Minimum prior games rules
    out = out.sort_values(["team_id", "game_datetime_utc", "game_id"]).reset_index(drop=True)
    prior_count = out.groupby("team_id").cumcount()
    out.loc[prior_count < 3, "minutes_available_pct"] = np.nan
    out.loc[prior_count < 3, "lineup_continuity_l3"] = np.nan
    out.loc[prior_count < 2, "new_starter_flag"] = np.nan
    for col in ["star_availability_score", "usage_gini", "top1_usage_share", "injury_impact_delta"]:
        out.loc[prior_count < 1, col] = np.nan

    if days_back is not None:
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_back)
        out = out[out["game_datetime_utc"] >= cutoff].copy()

    out["game_datetime_utc"] = out["game_datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out = out[
        [
            "game_id",
            "team_id",
            "game_datetime_utc",
            "season",
            "minutes_available_pct",
            "star_availability_score",
            "injury_impact_delta",
            "lineup_continuity_l3",
            "new_starter_flag",
            "usage_gini",
            "top1_usage_share",
        ]
    ]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build team-level player availability features")
    parser.add_argument("--days-back", type=int, default=None)
    args = parser.parse_args()

    features_df = build_player_availability_features(PLAYER_LOGS_PATH, days_back=args.days_back)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Total rows written: {len(features_df)}")
    print("Non-null rates by column:")
    non_null = features_df.notna().mean().sort_index()
    for col, val in non_null.items():
        print(f"  {col}: {val:.3f}")
    print(f"Count of games where star_availability_score < 0.7: {(features_df['star_availability_score'] < 0.7).sum()}")
    print(f"Count of games where new_starter_flag == 1: {(features_df['new_starter_flag'] == 1).sum()}")


if __name__ == "__main__":
    main()
