#!/usr/bin/env python3
"""Compute opponent style similarity for each team-game row."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SIZE = 10
MIN_STYLE_GAMES = 3
RECENT_OPPONENT_WINDOW = 10
MIN_SHARED_DIMS = 3

SOURCE_PRIORITY = [
    "team_game_weighted.csv",
    "team_game_metrics.csv",
]

STYLE_FEATURE_ALIASES: dict[str, list[str]] = {
    "pace": ["pace", "possessions", "poss"],
    "efg": ["efg_pct", "eFG"],
    "three_pa_rate": ["three_par", "3PA_rate"],
    "orb_pct": ["orb_pct", "ORB%"],
    "tov_pct": ["tov_pct", "TOV%"],
    "ftr": ["ftr", "FTr"],
}

OUTPUT_COLUMNS = [
    "event_id",
    "game_datetime_utc",
    "season",
    "team_id",
    "team",
    "opponent_id",
    "opponent",
    "input_source_file",
    "style_features_used",
    "similarity_score",
    "prior_opponents_compared",
    "top1_opponent_id",
    "top1_opponent",
    "top1_similarity",
    "top2_opponent_id",
    "top2_opponent",
    "top2_similarity",
    "top3_opponent_id",
    "top3_opponent",
    "top3_similarity",
    "current_opponent_vector_source",
    "window_size",
    "min_style_games",
    "recent_opponent_window",
    "computed_at_utc",
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
    return pd.Series(np.where(ts.dt.month >= 7, ts.dt.year + 1, ts.dt.year), index=dt.index)


@dataclass
class InputSpec:
    path: Path
    event_col: str
    datetime_col: str
    team_id_col: str
    team_col: str | None
    opponent_id_col: str
    opponent_col: str | None
    season_col: str | None
    completed_col: str | None
    style_col_map: dict[str, str]


def _build_input_spec(path: Path) -> tuple[InputSpec | None, list[str]]:
    columns = list(pd.read_csv(path, nrows=0).columns)
    missing: list[str] = []

    event_col = _first_present(columns, ["event_id", "game_id"])
    datetime_col = _first_present(columns, ["game_datetime_utc"])
    team_id_col = _first_present(columns, ["team_id"])
    opponent_id_col = _first_present(columns, ["opponent_id"])

    if event_col is None:
        missing.append("event_id|game_id")
    if datetime_col is None:
        missing.append("game_datetime_utc")
    if team_id_col is None:
        missing.append("team_id")
    if opponent_id_col is None:
        missing.append("opponent_id")

    style_col_map: dict[str, str] = {}
    for canonical, aliases in STYLE_FEATURE_ALIASES.items():
        col = _first_present(columns, aliases)
        if col is not None:
            style_col_map[canonical] = col
    if len(style_col_map) < 3:
        missing.append("style_features (need at least 3 of pace/eFG/3PA/ORB/TOV/FTr)")

    if missing:
        return None, missing

    spec = InputSpec(
        path=path,
        event_col=event_col or "",
        datetime_col=datetime_col or "",
        team_id_col=team_id_col or "",
        team_col=_first_present(columns, ["team", "team_name"]),
        opponent_id_col=opponent_id_col or "",
        opponent_col=_first_present(columns, ["opponent", "opponent_name"]),
        season_col=_first_present(columns, ["season"]),
        completed_col=_first_present(columns, ["completed"]),
        style_col_map=style_col_map,
    )
    return spec, []


def discover_input_spec(data_dir: Path) -> tuple[InputSpec | None, dict[str, list[str]]]:
    missing_by_file: dict[str, list[str]] = {}
    valid: list[tuple[tuple[int, int, int], InputSpec]] = []
    for idx, name in enumerate(SOURCE_PRIORITY):
        path = data_dir / name
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
        rows = _safe_row_count(path)
        valid.append(((rows, len(spec.style_col_map), -idx), spec))
    if not valid:
        return None, missing_by_file
    valid.sort(key=lambda item: item[0], reverse=True)
    return valid[0][1], missing_by_file


def _safe_row_count(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
    except Exception:
        return 0


def _write_blocked_summary(output_md: Path, note: str, missing: dict[str, list[str]]) -> None:
    lines = [
        "# Exec Summary: opponent_style_similarity",
        "",
        "- status: `BLOCKED`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- note: {note}",
    ]
    if missing:
        lines.append("- missing files/columns:")
        for file_name, fields in missing.items():
            lines.append(f"  - `{file_name}`: `{', '.join(fields)}`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ok_summary(
    output_md: Path,
    output_csv: Path,
    output_df: pd.DataFrame,
    *,
    input_csv: Path,
    style_cols_used: dict[str, str],
) -> None:
    null_rates = {}
    for col in ["team_id", "opponent_id", "similarity_score"]:
        if col in output_df.columns:
            null_rates[col] = round(float(output_df[col].isna().mean()) * 100.0, 2)
    lines = [
        "# Exec Summary: opponent_style_similarity",
        "",
        "- status: `OK`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- input_csv: `{input_csv}`",
        f"- output_csv: `{output_csv}`",
        f"- rows: `{len(output_df)}`",
        f"- columns: `{len(output_df.columns)}`",
        f"- style_features_used: `{json.dumps(style_cols_used, sort_keys=True)}`",
        f"- window: `L{WINDOW_SIZE}`",
        f"- min_style_games: `{MIN_STYLE_GAMES}`",
        f"- recent_opponent_window: `{RECENT_OPPONENT_WINDOW}`",
    ]
    if null_rates:
        lines.append("- key column null rates:")
        for col, rate in null_rates.items():
            lines.append(f"  - `{col}`: `{rate}%`")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _coerce_bool(series: pd.Series) -> pd.Series:
    low = series.fillna("").astype(str).str.strip().str.lower()
    return low.isin({"1", "true", "t", "yes", "y"})


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < MIN_SHARED_DIMS:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 0.0:
        return float("nan")
    return float(np.dot(aa, bb) / denom)


def _normalize_style_columns(df: pd.DataFrame, style_cols: list[str]) -> list[str]:
    norm_cols: list[str] = []
    for col in style_cols:
        out_col = f"{col}__z"
        s = pd.to_numeric(df[col], errors="coerce")
        mu = float(s.mean(skipna=True)) if s.notna().any() else 0.0
        sigma = float(s.std(skipna=True, ddof=0)) if s.notna().any() else 0.0
        if sigma > 0:
            df[out_col] = (s - mu) / sigma
        else:
            df[out_col] = np.nan
        norm_cols.append(out_col)
    return norm_cols


def build_similarity_df(df: pd.DataFrame, spec: InputSpec) -> pd.DataFrame:
    work = df.copy()
    work["event_id"] = work[spec.event_col].astype(str)
    work["team_id"] = work[spec.team_id_col].astype(str)
    work["opponent_id"] = work[spec.opponent_id_col].astype(str)
    work["team"] = work[spec.team_col].astype(str) if spec.team_col else ""
    work["opponent"] = work[spec.opponent_col].astype(str) if spec.opponent_col else ""
    work["game_datetime_utc"] = pd.to_datetime(work[spec.datetime_col], utc=True, errors="coerce")
    if spec.season_col and spec.season_col in work.columns:
        work["season"] = pd.to_numeric(work[spec.season_col], errors="coerce")
    else:
        work["season"] = _derive_season_from_datetime(work[spec.datetime_col])
    work["computed_at_utc"] = _utc_now()

    if spec.completed_col and spec.completed_col in work.columns:
        work["_is_completed"] = _coerce_bool(work[spec.completed_col])
    else:
        work["_is_completed"] = False

    style_roll_cols: list[str] = []
    for canonical, source_col in spec.style_col_map.items():
        raw_col = f"_raw_{canonical}"
        roll_col = f"_style_{canonical}_l10"
        work[raw_col] = pd.to_numeric(work[source_col], errors="coerce")
        work[roll_col] = work.groupby(["team_id", "season"], dropna=False)[raw_col].transform(
            lambda s: s.shift(1).rolling(WINDOW_SIZE, min_periods=MIN_STYLE_GAMES).mean()
        )
        style_roll_cols.append(roll_col)

    norm_cols = _normalize_style_columns(work, style_roll_cols)
    work = work.sort_values(["team_id", "season", "game_datetime_utc", "event_id"], kind="mergesort").reset_index(drop=True)

    team_name_lookup = (
        work[["team_id", "team"]]
        .dropna(subset=["team_id"])
        .drop_duplicates(subset=["team_id"], keep="last")
        .set_index("team_id")["team"]
        .to_dict()
    )

    event_team_lookup: dict[tuple[str, str], np.ndarray] = {}
    for _, row in work.iterrows():
        key = (str(row["event_id"]), str(row["team_id"]))
        vec = np.array([row[c] for c in norm_cols], dtype=float)
        event_team_lookup[key] = vec

    history_by_team_season: dict[tuple[str, int], pd.DataFrame] = {}
    for (team_id, season), g in work.groupby(["team_id", "season"], dropna=False, sort=False):
        key = (str(team_id), int(season) if pd.notna(season) else -1)
        history_by_team_season[key] = g.sort_values(["game_datetime_utc", "event_id"], kind="mergesort")

    def resolve_opponent_vector(event_id: str, opponent_id: str, dt: pd.Timestamp, season: float) -> tuple[np.ndarray | None, str]:
        if not opponent_id or opponent_id == "nan":
            return None, "missing_opponent_id"
        key = (event_id, opponent_id)
        vec = event_team_lookup.get(key)
        if vec is not None and int(np.isfinite(vec).sum()) >= MIN_SHARED_DIMS:
            return vec, "event_match"

        season_key = int(season) if pd.notna(season) else -1
        opp_rows = history_by_team_season.get((opponent_id, season_key))
        if opp_rows is None or opp_rows.empty:
            return None, "missing_opponent_history"
        prior = opp_rows[opp_rows["game_datetime_utc"] < dt]
        if prior.empty:
            return None, "missing_opponent_prior_style"
        row = prior.iloc[-1]
        fallback = np.array([row[c] for c in norm_cols], dtype=float)
        if int(np.isfinite(fallback).sum()) < MIN_SHARED_DIMS:
            return None, "missing_opponent_prior_style"
        return fallback, "history_fallback"

    out_rows: list[dict[str, object]] = []
    for (team_id, season), g in work.groupby(["team_id", "season"], dropna=False, sort=False):
        history: list[dict[str, object]] = []
        g = g.sort_values(["game_datetime_utc", "event_id"], kind="mergesort")
        for _, row in g.iterrows():
            event_id = str(row["event_id"])
            dt = row["game_datetime_utc"]
            opponent_id = str(row["opponent_id"])
            opponent_name = str(row["opponent"]) if row["opponent"] is not None else ""
            current_vec, vec_source = resolve_opponent_vector(event_id, opponent_id, dt, row["season"])

            history_slice = history[-RECENT_OPPONENT_WINDOW:]
            sims: list[dict[str, object]] = []
            if current_vec is not None:
                per_opponent_best: dict[str, dict[str, object]] = {}
                for hist in history_slice:
                    sim = _cosine_similarity(current_vec, hist["vector"])
                    if not np.isfinite(sim):
                        continue
                    opp_key = str(hist["opponent_id"])
                    item = {
                        "opponent_id": opp_key,
                        "opponent": str(hist["opponent"]),
                        "similarity": float(sim),
                        "faced_dt": hist["faced_dt"],
                    }
                    prev = per_opponent_best.get(opp_key)
                    if prev is None or item["similarity"] > float(prev["similarity"]):
                        per_opponent_best[opp_key] = item
                sims = sorted(
                    per_opponent_best.values(),
                    key=lambda x: (float(x["similarity"]), x["faced_dt"]),
                    reverse=True,
                )

            top = sims[:3]
            similarity_score = float(np.mean([float(x["similarity"]) for x in sims])) if sims else float("nan")
            out_rows.append(
                {
                    "event_id": event_id,
                    "game_datetime_utc": dt.strftime("%Y-%m-%dT%H:%M:%SZ") if pd.notna(dt) else "",
                    "season": int(row["season"]) if pd.notna(row["season"]) else np.nan,
                    "team_id": str(row["team_id"]),
                    "team": str(row["team"]) if row["team"] is not None else "",
                    "opponent_id": opponent_id,
                    "opponent": opponent_name,
                    "input_source_file": spec.path.name,
                    "style_features_used": json.dumps(spec.style_col_map, sort_keys=True),
                    "similarity_score": similarity_score,
                    "prior_opponents_compared": int(len(sims)),
                    "top1_opponent_id": str(top[0]["opponent_id"]) if len(top) > 0 else "",
                    "top1_opponent": str(top[0]["opponent"]) if len(top) > 0 else "",
                    "top1_similarity": float(top[0]["similarity"]) if len(top) > 0 else np.nan,
                    "top2_opponent_id": str(top[1]["opponent_id"]) if len(top) > 1 else "",
                    "top2_opponent": str(top[1]["opponent"]) if len(top) > 1 else "",
                    "top2_similarity": float(top[1]["similarity"]) if len(top) > 1 else np.nan,
                    "top3_opponent_id": str(top[2]["opponent_id"]) if len(top) > 2 else "",
                    "top3_opponent": str(top[2]["opponent"]) if len(top) > 2 else "",
                    "top3_similarity": float(top[2]["similarity"]) if len(top) > 2 else np.nan,
                    "current_opponent_vector_source": vec_source,
                    "window_size": WINDOW_SIZE,
                    "min_style_games": MIN_STYLE_GAMES,
                    "recent_opponent_window": RECENT_OPPONENT_WINDOW,
                    "computed_at_utc": row["computed_at_utc"],
                }
            )

            raw_observed = any(pd.notna(row[f"_raw_{k}"]) for k in spec.style_col_map)
            if current_vec is not None and bool(raw_observed or row["_is_completed"]):
                history.append(
                    {
                        "opponent_id": opponent_id,
                        "opponent": opponent_name or team_name_lookup.get(opponent_id, ""),
                        "vector": current_vec,
                        "faced_dt": dt,
                    }
                )

    out = pd.DataFrame(out_rows, columns=OUTPUT_COLUMNS)
    return out


def run_opponent_style_similarity(
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
            note="No compatible team-game table found for style similarity.",
            missing=missing,
        )
        return 1

    usecols = {
        spec.event_col,
        spec.datetime_col,
        spec.team_id_col,
        spec.opponent_id_col,
    }
    if spec.team_col:
        usecols.add(spec.team_col)
    if spec.opponent_col:
        usecols.add(spec.opponent_col)
    if spec.season_col:
        usecols.add(spec.season_col)
    if spec.completed_col:
        usecols.add(spec.completed_col)
    for col in spec.style_col_map.values():
        usecols.add(col)

    df = pd.read_csv(spec.path, usecols=sorted(usecols), low_memory=False)
    output = build_similarity_df(df, spec)
    if sample_limit is not None and sample_limit >= 0:
        output = output.head(sample_limit).copy()
    output.to_csv(output_csv, index=False)
    _write_ok_summary(
        output_md,
        output_csv,
        output,
        input_csv=spec.path,
        style_cols_used=spec.style_col_map,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/matchups/opponent_style_similarity.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("data/matchups/opponent_style_similarity_exec_summary.md"),
    )
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional row limit for smoke runs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md

    rc = run_opponent_style_similarity(
        data_dir=data_dir,
        output_csv=output_csv,
        output_md=output_md,
        sample_limit=args.sample_limit,
    )
    print(json.dumps({"output_csv": str(output_csv), "output_md": str(output_md), "exit_code": rc}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
