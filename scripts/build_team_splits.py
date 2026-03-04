#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

METRIC_MAP = {
    "efg": "efg_pct",
    "orb": "orb_pct",
    "tov": "tov_pct",
    "pace": "pace",
    "netrtg": "net_rtg",
    "ftr": "ftr",
}


def _season_from_date(dt: pd.Series) -> pd.Series:
    # College season label (Nov/Dec belong to next calendar year season)
    return dt.dt.year.where(dt.dt.month < 7, dt.dt.year + 1)


def build_team_splits(input_csv: Path, output_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, low_memory=False)

    required = {"team_id", "home_away", "game_datetime_utc", *METRIC_MAP.values()}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["game_datetime_utc"] = pd.to_datetime(df["game_datetime_utc"], errors="coerce", utc=True)
    df = df[df["game_datetime_utc"].notna() & df["team_id"].notna()].copy()
    df["home_away"] = df["home_away"].astype(str).str.lower()
    df = df[df["home_away"].isin(["home", "away"])].copy()
    df["date"] = df["game_datetime_utc"].dt.date
    df["season"] = _season_from_date(df["game_datetime_utc"])

    sort_cols = ["team_id", "season", "home_away", "game_datetime_utc"]
    if "event_id" in df.columns:
        sort_cols.append("event_id")
    df = df.sort_values(sort_cols)

    for out_name, src in METRIC_MAP.items():
        df[src] = pd.to_numeric(df[src], errors="coerce")
        roll = (
            df.groupby(["team_id", "season", "home_away"], sort=False)[src]
            .transform(lambda s: s.expanding(min_periods=5).mean())
        )
        df[f"{out_name}_split"] = roll

    home = df[df["home_away"] == "home"][["team_id", "season", "date", *[f"{k}_split" for k in METRIC_MAP]]].copy()
    away = df[df["home_away"] == "away"][["team_id", "season", "date", *[f"{k}_split" for k in METRIC_MAP]]].copy()

    home = home.rename(columns={f"{k}_split": f"{k}_home" for k in METRIC_MAP})
    away = away.rename(columns={f"{k}_split": f"{k}_away" for k in METRIC_MAP})

    timeline = (
        df[["team_id", "season", "date", "game_datetime_utc"]]
        .sort_values(["team_id", "season", "date", "game_datetime_utc"])
        .drop_duplicates(["team_id", "season", "date"], keep="last")
        .drop(columns=["game_datetime_utc"])
    )

    out = timeline.merge(home, on=["team_id", "season", "date"], how="left")
    out = out.merge(away, on=["team_id", "season", "date"], how="left")
    out = out.sort_values(["team_id", "season", "date"])

    split_cols = [f"{k}_{ha}" for k in METRIC_MAP for ha in ("home", "away")]
    out[split_cols] = out.groupby(["team_id", "season"], sort=False)[split_cols].ffill()

    # keep rows where both home/away split histories exist (>=5 games each side historically)
    required_pairs = [
        "efg_home", "efg_away", "orb_home", "orb_away", "tov_home", "tov_away",
        "pace_home", "pace_away", "netrtg_home", "netrtg_away", "ftr_home", "ftr_away",
    ]
    out = out[out[required_pairs].notna().all(axis=1)].copy()

    out = out[["team_id", "season", "date", *required_pairs]].copy()
    out.to_csv(output_csv, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build team home/away rolling split metrics")
    parser.add_argument("--input", default="data/team_game_metrics.csv")
    parser.add_argument("--output", default="team_splits.csv")
    args = parser.parse_args()

    out = build_team_splits(Path(args.input), Path(args.output))
    print(f"wrote {len(out)} rows to {args.output}")


if __name__ == "__main__":
    main()
