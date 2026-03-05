#!/usr/bin/env python3
"""Evaluate rolling-window grid combos for spread prediction quality."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_COMBOS = [
    "4/8",
    "4/12",
    "4/8/12",
    "5/10",
    "6/11",
    "7/12",
]

WIN_PROFIT_UNITS = 100.0 / 110.0

OUTPUT_COLUMNS = [
    "window_combo",
    "windows",
    "sample_size",
    "wins",
    "losses",
    "pushes",
    "win_rate",
    "roi",
    "mae",
    "avg_abs_edge",
    "stability_score",
    "roi_std_by_slice",
    "time_slices",
    "excluded",
    "excluded_reason",
    "split_mode",
    "split_source",
    "results_source_file",
    "edge_definition",
    "mae_definition",
    "generated_at_utc",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _derive_season_from_datetime(dt: pd.Series) -> pd.Series:
    ts = pd.to_datetime(dt, utc=True, errors="coerce")
    return pd.Series(np.where(ts.dt.month >= 7, ts.dt.year + 1, ts.dt.year), index=dt.index)


def _pick_first(columns: list[str], options: list[str]) -> str | None:
    for col in options:
        if col in columns:
            return col
    return None


def _safe_row_count(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
    except Exception:
        return 0


def _write_blocked_summary(
    path: Path,
    *,
    reason: str,
    missing_files: list[str],
    missing_columns: dict[str, list[str]],
) -> None:
    lines = [
        "# Exec Summary: window_grid_evaluator",
        "",
        "- status: `BLOCKED`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- reason: {reason}",
    ]
    if missing_files:
        lines.append("- missing files:")
        for item in missing_files:
            lines.append(f"  - `{item}`")
    if missing_columns:
        lines.append("- missing columns:")
        for f, cols in missing_columns.items():
            lines.append(f"  - `{f}`: `{', '.join(cols)}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass
class ResultsSpec:
    path: Path
    event_col: str
    game_datetime_col: str
    actual_margin_col: str
    market_spread_col: str


def _discover_results_spec(data_dir: Path) -> tuple[ResultsSpec | None, list[str], dict[str, list[str]]]:
    candidates = [
        "results_log.csv",
        "results_log_graded.csv",
        "predictions_graded.csv",
    ]
    inspected: list[str] = []
    missing_cols: dict[str, list[str]] = {}
    valid: list[tuple[tuple[int, int], ResultsSpec]] = []

    for idx, name in enumerate(candidates):
        p = data_dir / name
        if not p.exists() or p.stat().st_size <= 1:
            inspected.append(f"{name}:missing_or_empty")
            continue
        try:
            cols = list(pd.read_csv(p, nrows=0, low_memory=False).columns)
        except Exception:
            inspected.append(f"{name}:unreadable")
            continue

        event_col = _pick_first(cols, ["event_id", "game_id"])
        dt_col = _pick_first(cols, ["game_datetime_utc", "game_date"])
        actual_margin_col = _pick_first(cols, ["actual_margin"])
        market_spread_col = _pick_first(cols, ["market_spread", "spread_line"])

        missing: list[str] = []
        if event_col is None:
            missing.append("event_id|game_id")
        if dt_col is None:
            missing.append("game_datetime_utc|game_date")
        if actual_margin_col is None:
            missing.append("actual_margin")
        if market_spread_col is None:
            missing.append("market_spread|spread_line")
        if missing:
            missing_cols[str(p)] = missing
            inspected.append(f"{name}:missing_required")
            continue

        rows = _safe_row_count(p)
        spec = ResultsSpec(
            path=p,
            event_col=event_col or "",
            game_datetime_col=dt_col or "",
            actual_margin_col=actual_margin_col or "",
            market_spread_col=market_spread_col or "",
        )
        valid.append(((rows, -idx), spec))
        inspected.append(f"{name}:valid:rows={rows}")

    if not valid:
        return None, inspected, missing_cols
    valid.sort(key=lambda x: x[0], reverse=True)
    return valid[0][1], inspected, missing_cols


def _load_pregame_split_features(data_dir: Path, required_windows: list[int]) -> tuple[pd.DataFrame, str, str]:
    # Required combos include windows (8,11) that are not guaranteed in precomputed split tables;
    # compute them leak-free from team_game_weighted net_rtg for deterministic coverage.
    source = data_dir / "team_game_weighted.csv"
    if not source.exists():
        raise FileNotFoundError(str(source))

    needed_cols = ["event_id", "game_datetime_utc", "team_id", "home_away", "net_rtg"]
    header = list(pd.read_csv(source, nrows=0, low_memory=False).columns)
    usecols = [c for c in needed_cols if c in header]
    missing = [c for c in needed_cols if c not in usecols]
    if missing:
        raise ValueError(f"missing_team_game_weighted_columns:{','.join(missing)}")

    raw = pd.read_csv(source, usecols=usecols, low_memory=False)
    raw["event_id"] = raw["event_id"].astype(str)
    raw["team_id"] = raw["team_id"].astype(str)
    raw["home_away"] = raw["home_away"].astype(str).str.lower().str.strip()
    raw = raw[raw["home_away"].isin(["home", "away"])].copy()
    raw["game_datetime_utc"] = pd.to_datetime(raw["game_datetime_utc"], utc=True, errors="coerce")
    raw["season"] = _derive_season_from_datetime(raw["game_datetime_utc"])
    raw["net_rtg"] = pd.to_numeric(raw["net_rtg"], errors="coerce")
    raw = raw.dropna(subset=["game_datetime_utc", "net_rtg"])
    raw = raw.sort_values(["team_id", "season", "home_away", "game_datetime_utc", "event_id"], kind="mergesort")

    for w in sorted(set(required_windows)):
        col = f"net_rtg_split_l{w}"
        raw[col] = raw.groupby(["team_id", "season", "home_away"], dropna=False)["net_rtg"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=w).mean()
        )

    home = raw[raw["home_away"] == "home"].copy()
    away = raw[raw["home_away"] == "away"].copy()

    home_cols = ["event_id", "team_id"] + [f"net_rtg_split_l{w}" for w in sorted(set(required_windows))]
    away_cols = home_cols.copy()
    home = home[home_cols].rename(columns={"team_id": "home_team_id", **{f"net_rtg_split_l{w}": f"home_l{w}" for w in set(required_windows)}})
    away = away[away_cols].rename(columns={"team_id": "away_team_id", **{f"net_rtg_split_l{w}": f"away_l{w}" for w in set(required_windows)}})

    event_df = home.merge(away, on="event_id", how="inner")
    event_df = event_df.drop_duplicates(subset=["event_id"], keep="last")
    return event_df, "computed_from_team_game_weighted", str(source)


def _stability_score(eval_df: pd.DataFrame) -> tuple[float, float, int]:
    if eval_df.empty:
        return float("nan"), float("nan"), 0
    dt = pd.to_datetime(eval_df["game_datetime_utc"], utc=True, errors="coerce")
    if dt.isna().all():
        return float("nan"), float("nan"), 0
    work = eval_df.copy()
    work["slice"] = dt.dt.to_period("M").astype(str)
    roi_by_slice = work.groupby("slice", dropna=True)["profit_units"].mean()
    if roi_by_slice.empty:
        return float("nan"), float("nan"), 0
    roi_std = float(roi_by_slice.std(ddof=0)) if len(roi_by_slice) > 1 else 0.0
    slice_penalty = min(1.0, len(roi_by_slice) / 4.0)
    stability = float((1.0 / (1.0 + max(roi_std, 0.0))) * slice_penalty)
    return stability, roi_std, int(len(roi_by_slice))


def _evaluate_combo(df: pd.DataFrame, combo: str, split_mode: str, split_source: str, results_source: str) -> dict[str, object]:
    windows = [int(x) for x in combo.split("/") if x.strip()]
    needed = [f"home_l{w}" for w in windows] + [f"away_l{w}" for w in windows]
    work = df.dropna(subset=needed + ["market_spread", "actual_margin", "game_datetime_utc"]).copy()

    if work.empty:
        return {
            "window_combo": combo,
            "windows": combo,
            "sample_size": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": np.nan,
            "roi": np.nan,
            "mae": np.nan,
            "avg_abs_edge": np.nan,
            "stability_score": np.nan,
            "roi_std_by_slice": np.nan,
            "time_slices": 0,
            "excluded": True,
            "excluded_reason": "insufficient_history_for_combo",
            "split_mode": split_mode,
            "split_source": split_source,
            "results_source_file": results_source,
            "edge_definition": "edge = pred_spread - market_spread (home-line convention)",
            "mae_definition": "mae = mean(abs(pred_home_margin - actual_margin))",
            "generated_at_utc": _utc_now(),
        }

    home_strength = work[[f"home_l{w}" for w in windows]].mean(axis=1)
    away_strength = work[[f"away_l{w}" for w in windows]].mean(axis=1)
    pred_home_margin = home_strength - away_strength
    pred_spread = -pred_home_margin

    work["pred_spread"] = pred_spread
    work["edge"] = work["pred_spread"] - work["market_spread"]
    work = work[work["edge"] != 0].copy()
    if work.empty:
        return {
            "window_combo": combo,
            "windows": combo,
            "sample_size": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": np.nan,
            "roi": np.nan,
            "mae": np.nan,
            "avg_abs_edge": np.nan,
            "stability_score": np.nan,
            "roi_std_by_slice": np.nan,
            "time_slices": 0,
            "excluded": True,
            "excluded_reason": "all_zero_edge",
            "split_mode": split_mode,
            "split_source": split_source,
            "results_source_file": results_source,
            "edge_definition": "edge = pred_spread - market_spread (home-line convention)",
            "mae_definition": "mae = mean(abs(pred_home_margin - actual_margin))",
            "generated_at_utc": _utc_now(),
        }

    work["cover_margin"] = work["actual_margin"] - work["market_spread"]
    prod = work["edge"] * work["cover_margin"]
    work["result"] = np.where(prod < 0, "win", np.where(prod > 0, "loss", "push"))
    work["profit_units"] = np.where(work["result"] == "win", WIN_PROFIT_UNITS, np.where(work["result"] == "loss", -1.0, 0.0))

    wins = int((work["result"] == "win").sum())
    losses = int((work["result"] == "loss").sum())
    pushes = int((work["result"] == "push").sum())
    denom = wins + losses
    win_rate = float(wins / denom) if denom > 0 else float("nan")
    roi = float(work["profit_units"].mean()) if not work.empty else float("nan")
    mae = float((pred_home_margin.loc[work.index] - work["actual_margin"]).abs().mean()) if not work.empty else float("nan")
    avg_abs_edge = float(work["edge"].abs().mean()) if not work.empty else float("nan")
    stability, roi_std, slice_count = _stability_score(work)

    return {
        "window_combo": combo,
        "windows": combo,
        "sample_size": int(len(work)),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": win_rate,
        "roi": roi,
        "mae": mae,
        "avg_abs_edge": avg_abs_edge,
        "stability_score": stability,
        "roi_std_by_slice": roi_std,
        "time_slices": slice_count,
        "excluded": False,
        "excluded_reason": "",
        "split_mode": split_mode,
        "split_source": split_source,
        "results_source_file": results_source,
        "edge_definition": "edge = pred_spread - market_spread (home-line convention)",
        "mae_definition": "mae = mean(abs(pred_home_margin - actual_margin))",
        "generated_at_utc": _utc_now(),
    }


def _write_ok_summary(
    path: Path,
    *,
    output_csv: Path,
    split_mode: str,
    split_source: str,
    results_source: Path,
    results_df: pd.DataFrame,
) -> None:
    excluded = results_df[results_df["excluded"] == True]
    lines = [
        "# Exec Summary: window_grid_evaluator",
        "",
        "- status: `OK`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- output_csv: `{output_csv}`",
        f"- rows: `{len(results_df)}`",
        f"- excluded_combos: `{len(excluded)}`",
        "- evaluated_combos:",
    ]
    for combo in WINDOW_COMBOS:
        cdf = results_df[results_df["window_combo"] == combo]
        if cdf.empty:
            lines.append(f"  - `{combo}`: `missing`")
            continue
        row = cdf.iloc[0]
        lines.append(
            f"  - `{combo}`: sample=`{int(row['sample_size'])}`, win_rate=`{row['win_rate']}`, roi=`{row['roi']}`, mae=`{row['mae']}`, stability=`{row['stability_score']}`, excluded=`{bool(row['excluded'])}`"
        )
    lines.extend(
        [
            f"- split_mode: `{split_mode}`",
            f"- split_source: `{split_source}`",
            f"- results_source_file: `{results_source}`",
            "- limit_note: Required windows include `8` and `11`; precomputed split windows were not guaranteed, so leak-free split windows were recomputed from `team_game_weighted.csv`.",
        ]
    )
    if not excluded.empty:
        lines.append("- excluded details:")
        for row in excluded.itertuples(index=False):
            lines.append(f"  - `{row.window_combo}`: `{row.excluded_reason}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_window_grid_evaluator(*, data_dir: Path, output_csv: Path, output_md: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    results_spec, inspected, missing = _discover_results_spec(data_dir)
    if results_spec is None:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            reason="No results table with required market/actual fields was found.",
            missing_files=[str(data_dir / x) for x in ["results_log.csv", "results_log_graded.csv", "predictions_graded.csv"]],
            missing_columns=missing,
        )
        return 1

    all_windows = sorted({int(x) for combo in WINDOW_COMBOS for x in combo.split("/") if x.strip()})
    try:
        split_df, split_mode, split_source = _load_pregame_split_features(data_dir, required_windows=all_windows)
    except FileNotFoundError:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            reason="Missing required feature table for split windows.",
            missing_files=[str(data_dir / "team_game_weighted.csv")],
            missing_columns={},
        )
        return 1
    except ValueError as exc:
        missing_info = str(exc).split(":", 1)[-1].split(",")
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            reason="Required columns missing from team_game_weighted.csv.",
            missing_files=[],
            missing_columns={str(data_dir / "team_game_weighted.csv"): missing_info},
        )
        return 1

    results_raw = pd.read_csv(results_spec.path, low_memory=False)
    eval_df = pd.DataFrame()
    eval_df["event_id"] = results_raw[results_spec.event_col].astype(str)
    eval_df["game_datetime_utc"] = pd.to_datetime(results_raw[results_spec.game_datetime_col], utc=True, errors="coerce")
    eval_df["market_spread"] = pd.to_numeric(results_raw[results_spec.market_spread_col], errors="coerce")
    eval_df["actual_margin"] = pd.to_numeric(results_raw[results_spec.actual_margin_col], errors="coerce")
    eval_df = eval_df.dropna(subset=["event_id"]).drop_duplicates(subset=["event_id"], keep="last")

    merged = eval_df.merge(split_df, on="event_id", how="inner")
    merged = merged.dropna(subset=["game_datetime_utc", "market_spread", "actual_margin"])

    rows: list[dict[str, object]] = []
    for combo in WINDOW_COMBOS:
        rows.append(
            _evaluate_combo(
                merged,
                combo=combo,
                split_mode=split_mode,
                split_source=split_source,
                results_source=str(results_spec.path),
            )
        )

    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(output_csv, index=False)
    _write_ok_summary(
        output_md,
        output_csv=output_csv,
        split_mode=split_mode,
        split_source=split_source,
        results_source=results_spec.path,
        results_df=out,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/analytics/window_grid_results.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/analytics/window_grid_exec_summary.md"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md

    rc = run_window_grid_evaluator(data_dir=data_dir, output_csv=output_csv, output_md=output_md)
    print(json.dumps({"output_csv": str(output_csv), "output_md": str(output_md), "exit_code": rc}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
