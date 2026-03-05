#!/usr/bin/env python3
"""Build segment-level ATS/total performance metrics with sample-size guards."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from edge_overconfidence_common import INPUT_CANDIDATES, build_dataset_spec, load_normalized_frame


EDGE_BINS = [0.0, 2.0, 4.0, 6.0, 8.0, float("inf")]
EDGE_LABELS = ["0-2", "2-4", "4-6", "6-8", "8+"]
DEFAULT_MIN_SAMPLE = 25
WIN_PROFIT_UNITS = 100.0 / 110.0

OUTPUT_COLUMNS = [
    "segment_name",
    "segment_value",
    "market_type",
    "sample_size",
    "wins",
    "losses",
    "pushes",
    "win_rate",
    "roi",
    "avg_edge",
    "avg_error",
    "min_sample",
    "excluded_by_min_sample",
    "excluded_reason",
    "edge_definition",
    "input_source_file",
    "generated_at_utc",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_row_count(path: Path) -> int:
    try:
        return max(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1, 0)
    except Exception:
        return 0


def _pick_first(columns: list[str], candidates: list[str]) -> str | None:
    for col in candidates:
        if col in columns:
            return col
    return None


def _clean_text(v: object) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _compute_similarity_tiers(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["similarity_score"] = pd.to_numeric(out["similarity_score"], errors="coerce")
    valid = out["similarity_score"].dropna()
    if valid.empty:
        out["similarity_tier"] = np.nan
        return out
    p33 = float(valid.quantile(1.0 / 3.0))
    p67 = float(valid.quantile(2.0 / 3.0))
    tiers = pd.Series(np.nan, index=out.index, dtype="object")
    score = out["similarity_score"]
    tiers.loc[score.notna() & (score <= p33)] = "low"
    tiers.loc[score.notna() & (score > p33) & (score <= p67)] = "med"
    tiers.loc[score.notna() & (score > p67)] = "high"
    out["similarity_tier"] = tiers
    return out


def _bucket_abs_edge(series: pd.Series) -> pd.Series:
    return pd.cut(
        pd.to_numeric(series, errors="coerce").abs(),
        bins=EDGE_BINS,
        labels=EDGE_LABELS,
        right=False,
        include_lowest=True,
    ).astype("string")


def _write_blocked_summary(
    path: Path,
    *,
    reason: str,
    missing_files: list[str],
    missing_columns: dict[str, list[str]],
    inspected: list[str],
) -> None:
    lines = [
        "# Exec Summary: segment_performance_explorer",
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
        for file_name, cols in missing_columns.items():
            lines.append(f"  - `{file_name}`: `{', '.join(cols)}`")
    if inspected:
        lines.append("- inspected inputs:")
        for item in inspected:
            lines.append(f"  - `{item}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _discover_source_file(data_dir: Path) -> tuple[Path | None, list[str], dict[str, list[str]]]:
    inspected: list[str] = []
    missing_cols: dict[str, list[str]] = {}
    ranked: list[tuple[tuple[int, int, int], Path]] = []

    optional_priority = [
        "home_team_id",
        "away_team_id",
        "home_team",
        "away_team",
        "home_conference",
        "away_conference",
    ]

    for idx, file_name in enumerate(INPUT_CANDIDATES):
        path = data_dir / file_name
        if not path.exists() or path.stat().st_size <= 1:
            inspected.append(f"{file_name}:missing_or_empty")
            continue

        try:
            spec, missing = build_dataset_spec(path)
        except Exception:
            inspected.append(f"{file_name}:unreadable")
            continue

        if spec is None:
            missing_cols[str(path)] = missing.get(str(path), [])
            inspected.append(f"{file_name}:invalid_required_columns")
            continue

        header = list(pd.read_csv(path, nrows=0, low_memory=False).columns)
        optional_score = sum(1 for col in optional_priority if col in header)
        rows = _safe_row_count(path)
        ranked.append(((optional_score, rows, -idx), path))
        inspected.append(f"{file_name}:valid:optional={optional_score}:rows={rows}")

    if not ranked:
        return None, inspected, missing_cols

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked[0][1], inspected, missing_cols


def _load_metadata_table(source_path: Path, event_col: str, game_col: str) -> pd.DataFrame:
    header = list(pd.read_csv(source_path, nrows=0, low_memory=False).columns)
    wanted = {
        event_col,
        game_col,
        "home_team_id",
        "away_team_id",
        "home_team",
        "away_team",
        "home_conference",
        "away_conference",
    }
    usecols = sorted(col for col in wanted if col in header)
    if not usecols:
        return pd.DataFrame(columns=["event_id", "game_id"])

    meta = pd.read_csv(source_path, usecols=usecols, low_memory=False)
    meta["event_id"] = meta[event_col].astype(str) if event_col in meta.columns else ""
    meta["game_id"] = meta[game_col].astype(str) if game_col in meta.columns else ""
    keep = [
        "event_id",
        "game_id",
        "home_team_id",
        "away_team_id",
        "home_team",
        "away_team",
        "home_conference",
        "away_conference",
    ]
    existing = [c for c in keep if c in meta.columns]
    meta = meta[existing].copy()
    meta = meta.drop_duplicates(subset=["event_id", "game_id"], keep="last")
    return meta


def _load_conference_lookup(data_dir: Path) -> pd.DataFrame:
    candidates = [
        "predictions_history.csv",
        "predictions_graded.csv",
        "predictions_latest.csv",
        "predictions_combined_latest.csv",
    ]
    tables: list[pd.DataFrame] = []
    for name in candidates:
        path = data_dir / name
        if not path.exists() or path.stat().st_size <= 1:
            continue
        try:
            header = list(pd.read_csv(path, nrows=0, low_memory=False).columns)
        except Exception:
            continue
        event_col = _pick_first(header, ["event_id", "game_id"])
        game_col = _pick_first(header, ["game_id", "event_id"])
        if event_col is None or game_col is None:
            continue
        if "home_conference" not in header or "away_conference" not in header:
            continue
        usecols = sorted({event_col, game_col, "home_conference", "away_conference"})
        try:
            t = pd.read_csv(path, usecols=usecols, low_memory=False)
        except Exception:
            continue
        t["event_id"] = t[event_col].astype(str)
        t["game_id"] = t[game_col].astype(str)
        t = t[["event_id", "game_id", "home_conference", "away_conference"]].copy()
        tables.append(t)
    if not tables:
        return pd.DataFrame(columns=["event_id", "game_id", "home_conference", "away_conference"])
    merged = pd.concat(tables, ignore_index=True)
    merged = merged.drop_duplicates(subset=["event_id", "game_id"], keep="last")
    return merged


def _load_volatility_table(data_dir: Path) -> tuple[pd.DataFrame, str]:
    path = data_dir / "teams" / "team_volatility.csv"
    if not path.exists():
        return pd.DataFrame(columns=["event_id", "team_id", "volatility_tier"]), "missing_file"
    try:
        df = pd.read_csv(path, usecols=["event_id", "team_id", "volatility_tier"], low_memory=False)
    except ValueError:
        return pd.DataFrame(columns=["event_id", "team_id", "volatility_tier"]), "missing_columns"
    df["event_id"] = df["event_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    df = df.drop_duplicates(subset=["event_id", "team_id"], keep="last")
    return df, "ok"


def _load_similarity_table(data_dir: Path) -> tuple[pd.DataFrame, str]:
    path = data_dir / "matchups" / "opponent_style_similarity.csv"
    if not path.exists():
        return pd.DataFrame(columns=["event_id", "team_id", "similarity_tier"]), "missing_file"
    try:
        df = pd.read_csv(path, usecols=["event_id", "team_id", "similarity_score"], low_memory=False)
    except ValueError:
        return pd.DataFrame(columns=["event_id", "team_id", "similarity_tier"]), "missing_columns"
    df["event_id"] = df["event_id"].astype(str)
    df["team_id"] = df["team_id"].astype(str)
    df = _compute_similarity_tiers(df)
    df = df[["event_id", "team_id", "similarity_tier"]]
    df = df.drop_duplicates(subset=["event_id", "team_id"], keep="last")
    return df, "ok"


def _load_context_overlay_table(data_dir: Path) -> tuple[pd.DataFrame, str]:
    path = data_dir / "context" / "context_overlay_latest.csv"
    if not path.exists():
        return pd.DataFrame(columns=["event_id", "game_id"]), "missing_file"
    try:
        header = list(pd.read_csv(path, nrows=0, low_memory=False).columns)
    except Exception:
        return pd.DataFrame(columns=["event_id", "game_id"]), "unreadable"

    event_col = _pick_first(header, ["event_id", "game_id"])
    game_col = _pick_first(header, ["game_id", "event_id"])
    if event_col is None or game_col is None:
        return pd.DataFrame(columns=["event_id", "game_id"]), "missing_id_columns"

    wanted_targets = {
        "game_type": ["game_type"],
        "ncaa_round": ["ncaa_round"],
        "fatigue_tier": ["fatigue_tier"],
        "neutral_tier": ["neutral_tier", "neutral_flag"],
        "fh_fast_start_tier": ["fh_fast_start_tier"],
        "sh_total_bias_tier": ["sh_total_bias_tier"],
    }
    col_map: dict[str, str] = {}
    for target, options in wanted_targets.items():
        found = _pick_first(header, options)
        if found:
            col_map[target] = found

    usecols = sorted(set([event_col, game_col] + list(col_map.values())))
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    out = pd.DataFrame()
    out["event_id"] = df[event_col].astype(str)
    out["game_id"] = df[game_col].astype(str)
    for target, src in col_map.items():
        out[target] = df[src]
    out = out.drop_duplicates(subset=["event_id", "game_id"], keep="last")
    return out, "ok"


def _season_phase(game_type: pd.Series) -> pd.Series:
    s = game_type.fillna("").astype(str).str.strip().str.lower()
    phase = pd.Series("regular", index=s.index, dtype="object")
    phase.loc[s.str.startswith("conf_tournament")] = "conf_tournament"
    phase.loc[s.str.startswith("ncaa_")] = "ncaa"
    return phase


def _neutral_scope(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=s.index, dtype="object")
    out.loc[s.isin({"1", "true", "yes", "y", "neutral"})] = "neutral"
    out.loc[s.isin({"0", "false", "no", "n", "non_neutral", "non-neutral"})] = "non_neutral"
    return out


def _compute_conference_scope(df: pd.DataFrame) -> pd.Series:
    if "home_conference" not in df.columns or "away_conference" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="object")
    h = df["home_conference"].map(_clean_text)
    a = df["away_conference"].map(_clean_text)
    out = pd.Series(np.nan, index=df.index, dtype="object")
    valid = (h != "") & (a != "")
    out.loc[valid & (h == a)] = "conference"
    out.loc[valid & (h != a)] = "non_conference"
    return out


def _spread_pick_rows(base: pd.DataFrame, meta: pd.DataFrame, vol: pd.DataFrame, sim: pd.DataFrame) -> pd.DataFrame:
    df = base.merge(meta, on=["event_id", "game_id"], how="left", suffixes=("", "_meta"))
    for col in ["home_team_id", "away_team_id", "home_team", "away_team", "home_conference", "away_conference"]:
        if col not in df.columns:
            df[col] = np.nan

    edge = pd.to_numeric(df["model_spread"], errors="coerce") - pd.to_numeric(df["market_spread"], errors="coerce")
    cover_margin = pd.to_numeric(df["actual_margin"], errors="coerce") - pd.to_numeric(df["market_spread"], errors="coerce")
    abs_error = (pd.to_numeric(df["model_spread"], errors="coerce") - pd.to_numeric(df["actual_margin"], errors="coerce")).abs()
    market_spread = pd.to_numeric(df["market_spread"], errors="coerce")

    work = pd.DataFrame(
        {
            "event_id": df["event_id"].astype(str),
            "game_id": df["game_id"].astype(str),
            "game_datetime_utc": df["game_datetime_utc"].astype(str),
            "home_team_id": df["home_team_id"].astype(str),
            "away_team_id": df["away_team_id"].astype(str),
            "home_team": df["home_team"].astype(str),
            "away_team": df["away_team"].astype(str),
            "home_conference": df["home_conference"],
            "away_conference": df["away_conference"],
            "edge_raw": edge,
            "abs_edge": edge.abs(),
            "abs_error": abs_error,
            "cover_margin": cover_margin,
            "market_spread": market_spread,
        }
    )
    work["market_type"] = "spread"
    work["pick_side"] = np.where(work["edge_raw"] < 0, "home", np.where(work["edge_raw"] > 0, "away", "none"))
    work = work[work["pick_side"] != "none"].copy()

    work["team_id"] = np.where(work["pick_side"] == "home", work["home_team_id"], work["away_team_id"])
    work["team"] = np.where(work["pick_side"] == "home", work["home_team"], work["away_team"])

    win_mask = (work["edge_raw"] * work["cover_margin"]) < 0
    loss_mask = (work["edge_raw"] * work["cover_margin"]) > 0
    work["result"] = np.where(win_mask, "win", np.where(loss_mask, "loss", "push"))
    work["profit_units"] = np.where(win_mask, WIN_PROFIT_UNITS, np.where(loss_mask, -1.0, 0.0))

    home_favorite = work["market_spread"] < 0
    home_dog = work["market_spread"] > 0
    pick_home = work["pick_side"] == "home"
    work["home_away_favorite_dog"] = np.where(
        pick_home & home_favorite,
        "home_favorite",
        np.where(
            pick_home & home_dog,
            "home_dog",
            np.where(
                (~pick_home) & home_favorite,
                "away_dog",
                np.where((~pick_home) & home_dog, "away_favorite", np.where(pick_home, "home_pickem", "away_pickem")),
            ),
        ),
    )
    work["edge_bucket"] = _bucket_abs_edge(work["abs_edge"])
    work["conference_scope"] = _compute_conference_scope(work)

    merge_cols = ["event_id", "team_id"]
    if not vol.empty:
        work = work.merge(vol, on=merge_cols, how="left")
    else:
        work["volatility_tier"] = np.nan
    if not sim.empty:
        work = work.merge(sim, on=merge_cols, how="left")
    else:
        work["similarity_tier"] = np.nan
    return work


def _total_pick_rows(base: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    df = base.merge(meta, on=["event_id", "game_id"], how="left")
    for col in ["home_conference", "away_conference"]:
        if col not in df.columns:
            df[col] = np.nan

    edge = pd.to_numeric(df["model_total"], errors="coerce") - pd.to_numeric(df["market_total"], errors="coerce")
    total_diff = pd.to_numeric(df["actual_total"], errors="coerce") - pd.to_numeric(df["market_total"], errors="coerce")
    abs_error = (pd.to_numeric(df["model_total"], errors="coerce") - pd.to_numeric(df["actual_total"], errors="coerce")).abs()

    work = pd.DataFrame(
        {
            "event_id": df["event_id"].astype(str),
            "game_id": df["game_id"].astype(str),
            "game_datetime_utc": df["game_datetime_utc"].astype(str),
            "home_conference": df["home_conference"],
            "away_conference": df["away_conference"],
            "edge_raw": edge,
            "abs_edge": edge.abs(),
            "abs_error": abs_error,
            "total_diff": total_diff,
        }
    )
    work["market_type"] = "total"
    work["pick_side"] = np.where(work["edge_raw"] > 0, "over", np.where(work["edge_raw"] < 0, "under", "none"))
    work = work[work["pick_side"] != "none"].copy()

    win_mask = (work["edge_raw"] * work["total_diff"]) > 0
    loss_mask = (work["edge_raw"] * work["total_diff"]) < 0
    work["result"] = np.where(win_mask, "win", np.where(loss_mask, "loss", "push"))
    work["profit_units"] = np.where(win_mask, WIN_PROFIT_UNITS, np.where(loss_mask, -1.0, 0.0))
    work["total_bucket"] = _bucket_abs_edge(work["abs_edge"])
    work["conference_scope"] = _compute_conference_scope(work)
    return work


def _aggregate_segment(
    df: pd.DataFrame,
    *,
    segment_name: str,
    value_col: str,
    min_sample: int,
    source_file: str,
    by_market_type: bool = False,
) -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    work = df.copy()
    value = work[value_col].map(_clean_text)
    work = work[(value != "") & (value != "nan")].copy()
    work["segment_value"] = value.loc[work.index]
    if work.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    group_cols = ["segment_value", "market_type"] if by_market_type else ["segment_value"]
    grouped = (
        work.groupby(group_cols, dropna=False)
        .agg(
            sample_size=("result", "size"),
            wins=("result", lambda s: int((s == "win").sum())),
            losses=("result", lambda s: int((s == "loss").sum())),
            pushes=("result", lambda s: int((s == "push").sum())),
            roi=("profit_units", "mean"),
            avg_edge=("abs_edge", "mean"),
            avg_error=("abs_error", "mean"),
        )
        .reset_index()
    )

    denom = grouped["wins"] + grouped["losses"]
    grouped["win_rate"] = np.where(denom > 0, grouped["wins"] / denom, np.nan)
    grouped["segment_name"] = segment_name
    if not by_market_type:
        grouped["market_type"] = str(work["market_type"].iloc[0]) if "market_type" in work.columns else ""
    grouped["min_sample"] = int(min_sample)
    grouped["excluded_by_min_sample"] = grouped["sample_size"] < int(min_sample)
    grouped["excluded_reason"] = np.where(grouped["excluded_by_min_sample"], "below_min_sample", "")
    grouped["edge_definition"] = "edge = model_line - market_line (absolute edge used for avg_edge)"
    grouped["input_source_file"] = source_file
    grouped["generated_at_utc"] = _utc_now()
    grouped = grouped[OUTPUT_COLUMNS]
    return grouped


def _build_summary(
    output_md: Path,
    *,
    source_file: Path,
    min_sample: int,
    result_df: pd.DataFrame,
    spread_rows: int,
    total_rows: int,
    unavailable_segments: list[str],
    inspected: list[str],
    context_status: str,
) -> None:
    included = result_df[~result_df["excluded_by_min_sample"]]
    excluded = result_df[result_df["excluded_by_min_sample"]]
    lines = [
        "# Exec Summary: segment_performance_explorer",
        "",
        "- status: `OK`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- input_source_file: `{source_file}`",
        f"- spread_pick_rows: `{spread_rows}`",
        f"- total_pick_rows: `{total_rows}`",
        f"- output_rows: `{len(result_df)}`",
        f"- min_sample: `{min_sample}`",
        f"- included_segments: `{len(included)}`",
        f"- excluded_segments: `{len(excluded)}`",
        f"- roi_assumption: `{WIN_PROFIT_UNITS:.6f} units on win, -1 on loss, 0 on push`",
        f"- context_overlay_status: `{context_status}`",
    ]
    if unavailable_segments:
        lines.append("- unavailable segment groups:")
        for item in unavailable_segments:
            lines.append(f"  - `{item}`")
    if not excluded.empty:
        lines.append("- excluded segment details (below min_sample):")
        for row in excluded.sort_values(["segment_name", "market_type", "segment_value"]).itertuples(index=False):
            lines.append(
                f"  - `{row.segment_name}` / `{row.market_type}` / `{row.segment_value}`: sample_size=`{row.sample_size}`"
            )
    lines.append("- inspected inputs:")
    for item in inspected:
        lines.append(f"  - `{item}`")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_segment_performance_explorer(
    *,
    data_dir: Path,
    output_csv: Path,
    output_md: Path,
    min_sample: int,
) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    source_file, inspected, missing_cols = _discover_source_file(data_dir)
    if source_file is None:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            reason="No input file with required model/market/actual columns was found.",
            missing_files=[str(data_dir / name) for name in INPUT_CANDIDATES],
            missing_columns=missing_cols,
            inspected=inspected,
        )
        return 1

    spec, missing = build_dataset_spec(source_file)
    if spec is None:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            reason=f"Input file {source_file.name} is missing required columns.",
            missing_files=[],
            missing_columns=missing,
            inspected=inspected,
        )
        return 1

    base = load_normalized_frame(spec)
    meta = _load_metadata_table(source_file, spec.event_col, spec.game_col)
    conf_lookup = _load_conference_lookup(data_dir)
    if "home_conference" not in meta.columns or "away_conference" not in meta.columns:
        meta = meta.merge(conf_lookup, on=["event_id", "game_id"], how="left", suffixes=("", "_lookup"))
        for col in ["home_conference", "away_conference"]:
            lookup_col = f"{col}_lookup"
            if col not in meta.columns and lookup_col in meta.columns:
                meta[col] = meta[lookup_col]
            elif col in meta.columns and lookup_col in meta.columns:
                left = meta[col].map(_clean_text)
                right = meta[lookup_col].map(_clean_text)
                fill_mask = left == ""
                meta.loc[fill_mask, col] = right.loc[fill_mask]

    vol_df, vol_status = _load_volatility_table(data_dir)
    sim_df, sim_status = _load_similarity_table(data_dir)
    context_df, context_status = _load_context_overlay_table(data_dir)

    spread = _spread_pick_rows(base, meta, vol_df, sim_df)
    total = _total_pick_rows(base, meta)
    if not context_df.empty:
        spread = spread.merge(context_df, on=["event_id", "game_id"], how="left")
        total = total.merge(context_df, on=["event_id", "game_id"], how="left")
        spread["season_phase"] = _season_phase(spread.get("game_type", pd.Series("", index=spread.index)))
        total["season_phase"] = _season_phase(total.get("game_type", pd.Series("", index=total.index)))
        spread["neutral_scope"] = _neutral_scope(spread.get("neutral_tier", pd.Series("", index=spread.index)))
        total["neutral_scope"] = _neutral_scope(total.get("neutral_tier", pd.Series("", index=total.index)))
    else:
        spread["season_phase"] = np.nan
        total["season_phase"] = np.nan
        spread["neutral_scope"] = np.nan
        total["neutral_scope"] = np.nan

    if spread.empty and total.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(output_csv, index=False)
        _write_blocked_summary(
            output_md,
            reason=f"{source_file.name} found but no eligible spread/total rows after numeric filtering.",
            missing_files=[],
            missing_columns={},
            inspected=inspected,
        )
        return 1

    unavailable_segments: list[str] = []
    segment_frames: list[pd.DataFrame] = []

    if not spread.empty:
        segment_frames.append(
            _aggregate_segment(
                spread,
                segment_name="home_away_favorite_dog",
                value_col="home_away_favorite_dog",
                min_sample=min_sample,
                source_file=str(source_file),
            )
        )
        segment_frames.append(
            _aggregate_segment(
                spread,
                segment_name="edge_bucket",
                value_col="edge_bucket",
                min_sample=min_sample,
                source_file=str(source_file),
            )
        )
        if "volatility_tier" in spread.columns and spread["volatility_tier"].notna().any():
            segment_frames.append(
                _aggregate_segment(
                    spread,
                    segment_name="volatility_tier",
                    value_col="volatility_tier",
                    min_sample=min_sample,
                    source_file=str(source_file),
                )
            )
        else:
            unavailable_segments.append(f"volatility_tier ({vol_status})")

        if "similarity_tier" in spread.columns and spread["similarity_tier"].notna().any():
            segment_frames.append(
                _aggregate_segment(
                    spread,
                    segment_name="similarity_tier",
                    value_col="similarity_tier",
                    min_sample=min_sample,
                    source_file=str(source_file),
                )
            )
        else:
            unavailable_segments.append(f"similarity_tier ({sim_status})")
    else:
        unavailable_segments.extend(
            ["home_away_favorite_dog (no_spread_rows)", "edge_bucket (no_spread_rows)", "volatility_tier (no_spread_rows)", "similarity_tier (no_spread_rows)"]
        )

    if not total.empty:
        segment_frames.append(
            _aggregate_segment(
                total,
                segment_name="total_bucket",
                value_col="total_bucket",
                min_sample=min_sample,
                source_file=str(source_file),
            )
        )
    else:
        unavailable_segments.append("total_bucket (no_total_rows)")

    both = pd.concat([spread, total], ignore_index=True, sort=False)
    if "conference_scope" in both.columns and both["conference_scope"].notna().any():
        segment_frames.append(
            _aggregate_segment(
                both,
                segment_name="conference_scope",
                value_col="conference_scope",
                min_sample=min_sample,
                source_file=str(source_file),
                by_market_type=True,
            )
        )
    else:
        unavailable_segments.append("conference_scope (missing_home_or_away_conference)")

    context_segment_cols = [
        ("season_phase", "season_phase"),
        ("ncaa_round", "ncaa_round"),
        ("fatigue_tier", "fatigue_tier"),
        ("neutral_scope", "neutral_scope"),
        ("fh_fast_start_tier", "fh_fast_start_tier"),
        ("sh_total_bias_tier", "sh_total_bias_tier"),
    ]
    for seg_name, col in context_segment_cols:
        if col in both.columns and both[col].notna().any():
            segment_frames.append(
                _aggregate_segment(
                    both,
                    segment_name=seg_name,
                    value_col=col,
                    min_sample=min_sample,
                    source_file=str(source_file),
                    by_market_type=True,
                )
            )
        else:
            unavailable_segments.append(f"{seg_name} (missing_context_column_or_values)")

    out = pd.concat([f for f in segment_frames if not f.empty], ignore_index=True) if segment_frames else pd.DataFrame(columns=OUTPUT_COLUMNS)
    if out.empty:
        out = pd.DataFrame(columns=OUTPUT_COLUMNS)
    else:
        out = out.sort_values(["segment_name", "market_type", "segment_value"], kind="mergesort").reset_index(drop=True)
    out.to_csv(output_csv, index=False)

    _build_summary(
        output_md,
        source_file=source_file,
        min_sample=min_sample,
        result_df=out,
        spread_rows=int(len(spread)),
        total_rows=int(len(total)),
        unavailable_segments=unavailable_segments,
        inspected=inspected,
        context_status=context_status,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/analytics/segment_performance.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/analytics/segment_exec_summary.md"))
    parser.add_argument("--min-sample", type=int, default=DEFAULT_MIN_SAMPLE)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md
    min_sample = max(int(args.min_sample), 1)

    rc = run_segment_performance_explorer(
        data_dir=data_dir,
        output_csv=output_csv,
        output_md=output_md,
        min_sample=min_sample,
    )
    print(json.dumps({"output_csv": str(output_csv), "output_md": str(output_md), "exit_code": rc}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
