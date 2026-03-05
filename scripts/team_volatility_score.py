#!/usr/bin/env python3
"""Build team volatility scores from team-game tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SIZE = 10
MIN_PERIODS = 3
NET_WEIGHT = 0.4
PACE_WEIGHT = 0.3
MARGIN_WEIGHT = 0.3

OUTPUT_COLUMNS = [
    "event_id",
    "game_datetime_utc",
    "season",
    "team_id",
    "team",
    "home_away",
    "input_source_file",
    "net_metric_source",
    "pace_metric_source",
    "margin_metric_source",
    "net_rating_std_l10",
    "pace_or_possessions_std_l10",
    "scoring_margin_std_l10",
    "volatility_score",
    "volatility_tier",
    "window_size",
    "min_periods",
    "computed_at_utc",
]

SOURCE_PRIORITY = [
    "team_game_metrics.csv",
    "team_game_weighted.csv",
    "team_game_logs.csv",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _first_present(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _derive_season_from_datetime(dt: pd.Series) -> pd.Series:
    ts = pd.to_datetime(dt, utc=True, errors="coerce")
    # College season generally starts in the fall and is identified by spring year.
    return pd.Series(np.where(ts.dt.month >= 7, ts.dt.year + 1, ts.dt.year), index=dt.index)


@dataclass
class InputSpec:
    path: Path
    team_id_col: str
    team_name_col: str | None
    home_away_col: str | None
    event_id_col: str | None
    datetime_col: str
    season_col: str | None
    net_col: str | None
    pace_col: str
    possessions_col: str | None
    margin_col: str | None
    points_for_col: str | None
    points_against_col: str | None


def _build_input_spec(path: Path) -> tuple[InputSpec | None, list[str]]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    missing: list[str] = []

    team_id_col = _first_present(columns, ["team_id", "team"])
    if team_id_col is None:
        missing.append("team_id|team")

    datetime_col = _first_present(columns, ["game_datetime_utc"])
    if datetime_col is None:
        missing.append("game_datetime_utc")

    pace_col = _first_present(columns, ["pace", "possessions", "poss"])
    if pace_col is None:
        missing.append("pace|possessions|poss")

    margin_col = _first_present(columns, ["scoring_margin", "margin", "point_margin"])
    points_for_col = _first_present(columns, ["points_for"])
    points_against_col = _first_present(columns, ["points_against"])
    has_margin = margin_col is not None or (points_for_col is not None and points_against_col is not None)
    if not has_margin:
        missing.append("scoring_margin|margin|point_margin or points_for+points_against")

    net_col = _first_present(columns, ["net_rating", "net_rtg", "NetRtg", "adj_net_rtg"])
    possessions_col = _first_present(columns, ["possessions", "poss"])
    if net_col is None and (not has_margin or possessions_col is None):
        missing.append("net_rating/net_rtg/NetRtg or margin_per_poss inputs (margin + possessions/poss)")

    if missing:
        return None, missing

    return (
        InputSpec(
            path=path,
            team_id_col=team_id_col or "",
            team_name_col=_first_present(columns, ["team", "team_name"]),
            home_away_col=_first_present(columns, ["home_away"]),
            event_id_col=_first_present(columns, ["event_id", "game_id"]),
            datetime_col=datetime_col or "",
            season_col=_first_present(columns, ["season"]),
            net_col=net_col,
            pace_col=pace_col or "",
            possessions_col=possessions_col,
            margin_col=margin_col,
            points_for_col=points_for_col,
            points_against_col=points_against_col,
        ),
        [],
    )


def discover_input_spec(data_dir: Path) -> tuple[InputSpec | None, dict[str, list[str]]]:
    missing_by_file: dict[str, list[str]] = {}
    valid_specs: list[tuple[tuple[float, float, int, int], InputSpec]] = []
    for file_name in SOURCE_PRIORITY:
        path = data_dir / file_name
        if not path.exists():
            missing_by_file[str(path)] = ["<missing_file>"]
            continue
        try:
            spec, missing = _build_input_spec(path)
        except Exception:
            missing_by_file[str(path)] = ["<unreadable_csv>"]
            continue
        if spec is None:
            missing_by_file[str(path)] = missing
            continue
        score = _score_input_spec(spec, priority_index=SOURCE_PRIORITY.index(file_name))
        valid_specs.append((score, spec))

    if valid_specs:
        valid_specs.sort(key=lambda item: item[0], reverse=True)
        return valid_specs[0][1], missing_by_file
    return None, missing_by_file


def _score_input_spec(spec: InputSpec, *, priority_index: int) -> tuple[float, float, int, int]:
    usecols = [spec.team_id_col]
    if spec.season_col:
        usecols.append(spec.season_col)
    try:
        sample = pd.read_csv(spec.path, usecols=usecols, low_memory=False)
    except Exception:
        return (0.0, 0.0, 0, -priority_index)
    if sample.empty:
        return (0.0, 0.0, 0, -priority_index)

    team_col = spec.team_id_col
    sample[team_col] = sample[team_col].astype(str)
    if spec.season_col and spec.season_col in sample.columns:
        sizes = sample.groupby([team_col, spec.season_col], dropna=False).size()
    else:
        sizes = sample.groupby(team_col, dropna=False).size()
    pct_ge_10 = float((sizes >= 10).mean()) if not sizes.empty else 0.0
    pct_ge_4 = float((sizes >= 4).mean()) if not sizes.empty else 0.0
    return (pct_ge_10, pct_ge_4, int(len(sample)), -priority_index)


def _write_blocked_summary(output_md: Path, note: str, missing: dict[str, list[str]]) -> None:
    lines = [
        "# Exec Summary: team_volatility_score",
        "",
        "- status: `BLOCKED`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- note: {note}",
    ]
    if missing:
        lines.append("- missing files/columns:")
        for file_name, cols in missing.items():
            lines.append(f"  - `{file_name}`: `{', '.join(cols)}`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ok_summary(
    output_md: Path,
    output_csv: Path,
    output_df: pd.DataFrame,
    *,
    input_csv: Path,
    net_source: str,
    pace_source: str,
    margin_source: str,
) -> None:
    null_rates = {}
    for col in ["team_id", "season", "volatility_score", "volatility_tier"]:
        if col in output_df.columns:
            null_rates[col] = round(float(output_df[col].isna().mean()) * 100.0, 2)

    lines = [
        "# Exec Summary: team_volatility_score",
        "",
        "- status: `OK`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- input_csv: `{input_csv}`",
        f"- output_csv: `{output_csv}`",
        f"- rows: `{len(output_df)}`",
        f"- columns: `{len(output_df.columns)}`",
        f"- net_metric_source: `{net_source}`",
        f"- pace_metric_source: `{pace_source}`",
        f"- scoring_margin_source: `{margin_source}`",
        f"- weights: `net={NET_WEIGHT}, pace={PACE_WEIGHT}, margin={MARGIN_WEIGHT}`",
    ]
    if null_rates:
        lines.append("- key column null rates:")
        for col, rate in null_rates.items():
            lines.append(f"  - `{col}`: `{rate}%`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assign_tiers_by_season(df: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=df.index, dtype="object")
    for season_value, idx in df.groupby("season", dropna=False).groups.items():
        score = df.loc[idx, "volatility_score"].dropna()
        if score.empty:
            continue
        p33 = float(score.quantile(1.0 / 3.0))
        p67 = float(score.quantile(2.0 / 3.0))
        low_mask = df.loc[idx, "volatility_score"] <= p33
        med_mask = (df.loc[idx, "volatility_score"] > p33) & (df.loc[idx, "volatility_score"] <= p67)
        high_mask = df.loc[idx, "volatility_score"] > p67
        out.loc[idx[low_mask]] = "low"
        out.loc[idx[med_mask]] = "med"
        out.loc[idx[high_mask]] = "high"
    return out


def build_team_volatility_df(df: pd.DataFrame, spec: InputSpec) -> tuple[pd.DataFrame, str, str, str]:
    work = df.copy()
    work["game_datetime_utc"] = pd.to_datetime(work[spec.datetime_col], utc=True, errors="coerce")

    if spec.season_col and spec.season_col in work.columns:
        work["season"] = pd.to_numeric(work[spec.season_col], errors="coerce")
    else:
        work["season"] = _derive_season_from_datetime(work[spec.datetime_col])

    work["team_id"] = work[spec.team_id_col].astype(str)
    work["team"] = work[spec.team_name_col].astype(str) if spec.team_name_col else ""
    work["home_away"] = work[spec.home_away_col].astype(str) if spec.home_away_col else ""
    work["event_id"] = work[spec.event_id_col].astype(str) if spec.event_id_col else ""

    if spec.margin_col and spec.margin_col in work.columns:
        work["_margin_metric"] = pd.to_numeric(work[spec.margin_col], errors="coerce")
        margin_source = spec.margin_col
    else:
        points_for = pd.to_numeric(work[spec.points_for_col], errors="coerce") if spec.points_for_col else pd.Series(np.nan, index=work.index)
        points_against = (
            pd.to_numeric(work[spec.points_against_col], errors="coerce")
            if spec.points_against_col
            else pd.Series(np.nan, index=work.index)
        )
        work["_margin_metric"] = points_for - points_against
        margin_source = "derived_points_for_minus_points_against"

    work["_pace_metric"] = pd.to_numeric(work[spec.pace_col], errors="coerce")
    pace_source = spec.pace_col

    if spec.net_col and spec.net_col in work.columns:
        work["_net_metric"] = pd.to_numeric(work[spec.net_col], errors="coerce")
        net_source = spec.net_col
    else:
        poss = pd.to_numeric(work[spec.possessions_col], errors="coerce") if spec.possessions_col else pd.Series(np.nan, index=work.index)
        poss = poss.replace(0.0, np.nan)
        work["_net_metric"] = work["_margin_metric"] / poss
        net_source = f"derived_margin_per_poss_from_{margin_source}_and_{spec.possessions_col}"

    sort_cols = ["team_id", "season", "game_datetime_utc", "event_id"]
    work = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    group_keys = ["team_id", "season"]
    work["net_rating_std_l10"] = work.groupby(group_keys, dropna=False)["_net_metric"].transform(
        lambda s: s.shift(1).rolling(WINDOW_SIZE, min_periods=MIN_PERIODS).std(ddof=0)
    )
    work["pace_or_possessions_std_l10"] = work.groupby(group_keys, dropna=False)["_pace_metric"].transform(
        lambda s: s.shift(1).rolling(WINDOW_SIZE, min_periods=MIN_PERIODS).std(ddof=0)
    )
    work["scoring_margin_std_l10"] = work.groupby(group_keys, dropna=False)["_margin_metric"].transform(
        lambda s: s.shift(1).rolling(WINDOW_SIZE, min_periods=MIN_PERIODS).std(ddof=0)
    )

    has_all_components = work[
        ["net_rating_std_l10", "pace_or_possessions_std_l10", "scoring_margin_std_l10"]
    ].notna().all(axis=1)
    work["volatility_score"] = np.nan
    work.loc[has_all_components, "volatility_score"] = (
        NET_WEIGHT * work.loc[has_all_components, "net_rating_std_l10"]
        + PACE_WEIGHT * work.loc[has_all_components, "pace_or_possessions_std_l10"]
        + MARGIN_WEIGHT * work.loc[has_all_components, "scoring_margin_std_l10"]
    )
    work["volatility_tier"] = _assign_tiers_by_season(work)
    work["input_source_file"] = spec.path.name
    work["net_metric_source"] = net_source
    work["pace_metric_source"] = pace_source
    work["margin_metric_source"] = margin_source
    work["window_size"] = WINDOW_SIZE
    work["min_periods"] = MIN_PERIODS
    work["computed_at_utc"] = _utc_now()

    out = work[OUTPUT_COLUMNS].copy()
    out["game_datetime_utc"] = out["game_datetime_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return out, net_source, pace_source, margin_source


def run_team_volatility(
    *,
    data_dir: Path,
    output_csv: Path,
    output_md: Path,
    sample_limit: int | None,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    spec, missing = discover_input_spec(data_dir)
    if spec is None:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            note="No compatible team-game input file was found.",
            missing=missing,
        )
        return 1

    selected_columns = {
        spec.team_id_col,
        spec.datetime_col,
        spec.pace_col,
    }
    for opt in [
        spec.team_name_col,
        spec.home_away_col,
        spec.event_id_col,
        spec.season_col,
        spec.net_col,
        spec.possessions_col,
        spec.margin_col,
        spec.points_for_col,
        spec.points_against_col,
    ]:
        if opt:
            selected_columns.add(opt)

    input_df = pd.read_csv(spec.path, usecols=sorted(selected_columns), low_memory=False)
    output_df, net_source, pace_source, margin_source = build_team_volatility_df(input_df, spec)

    if sample_limit is not None and sample_limit >= 0:
        output_df = output_df.head(sample_limit).copy()

    output_df.to_csv(output_csv, index=False)
    _write_ok_summary(
        output_md,
        output_csv,
        output_df,
        input_csv=spec.path,
        net_source=net_source,
        pace_source=pace_source,
        margin_source=margin_source,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/teams/team_volatility.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/teams/team_volatility_exec_summary.md"))
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional output row limit (for smoke runs)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md

    rc = run_team_volatility(
        data_dir=data_dir,
        output_csv=output_csv,
        output_md=output_md,
        sample_limit=args.sample_limit,
    )
    print(json.dumps({"output_csv": str(output_csv), "output_md": str(output_md), "exit_code": rc}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
