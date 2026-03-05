#!/usr/bin/env python3
"""Build a unified context overlay from optional tournament/NCAA/halves/totals layers."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


OUTPUT_COLUMNS = [
    "event_id",
    "game_id",
    "game_datetime_utc",
    "game_type",
    "ncaa_round",
    "fatigue_tier",
    "neutral_tier",
    "leverage_tier",
    "familiarity_tier",
    "upset_profile_tier",
    "tempo_control_tier",
    "fh_fast_start_tier",
    "sh_total_bias_tier",
    "tournament_layers_active",
    "ncaa_layers_active",
    "generated_at_utc",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _pick_first(columns: list[str], options: list[str]) -> str | None:
    for col in options:
        if col in columns:
            return col
    return None


def _norm_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def _load_base_games(data_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    inspected: list[str] = []
    candidates = [
        data_dir / "games.csv",
        data_dir / "predictions_latest.csv",
        data_dir / "predictions_combined_latest.csv",
        data_dir / "results_log.csv",
    ]
    tables: list[pd.DataFrame] = []
    for p in candidates:
        if not p.exists() or p.stat().st_size <= 1:
            inspected.append(f"{p.name}:missing_or_empty")
            continue
        try:
            header = list(pd.read_csv(p, nrows=0, low_memory=False).columns)
        except Exception:
            inspected.append(f"{p.name}:unreadable")
            continue
        event_col = _pick_first(header, ["event_id", "game_id"])
        game_col = _pick_first(header, ["game_id", "event_id"])
        dt_col = _pick_first(header, ["game_datetime_utc", "game_datetime", "game_date"])
        game_type_col = _pick_first(header, ["game_type"])
        if event_col is None or game_col is None:
            inspected.append(f"{p.name}:missing_ids")
            continue
        usecols = [event_col, game_col]
        if dt_col:
            usecols.append(dt_col)
        if game_type_col:
            usecols.append(game_type_col)
        df = pd.read_csv(p, usecols=sorted(set(usecols)), low_memory=False)
        out = pd.DataFrame()
        out["event_id"] = _norm_id(df[event_col])
        out["game_id"] = _norm_id(df[game_col])
        out["game_datetime_utc"] = df[dt_col].astype(str) if dt_col else ""
        out["game_type"] = df[game_type_col].astype(str) if game_type_col else ""
        tables.append(out)
        inspected.append(f"{p.name}:loaded_rows={len(out)}")

    if not tables:
        return pd.DataFrame(columns=["event_id", "game_id", "game_datetime_utc", "game_type"]), inspected

    merged = pd.concat(tables, ignore_index=True)
    merged = merged.drop_duplicates(subset=["event_id", "game_id"], keep="last")
    return merged, inspected


def _load_optional_layer(data_dir: Path, rel_path: str, candidates: dict[str, list[str]]) -> tuple[pd.DataFrame, str]:
    p = data_dir / rel_path
    if not p.exists() or p.stat().st_size <= 1:
        return pd.DataFrame(columns=["event_id", "game_id"]), "missing_file"
    try:
        header = list(pd.read_csv(p, nrows=0, low_memory=False).columns)
    except Exception:
        return pd.DataFrame(columns=["event_id", "game_id"]), "unreadable"

    event_col = _pick_first(header, ["event_id", "game_id"])
    game_col = _pick_first(header, ["game_id", "event_id"])
    if event_col is None or game_col is None:
        return pd.DataFrame(columns=["event_id", "game_id"]), "missing_id_columns"

    col_map: dict[str, str] = {}
    for target, options in candidates.items():
        found = _pick_first(header, options)
        if found:
            col_map[target] = found

    usecols = sorted(set([event_col, game_col] + list(col_map.values())))
    df = pd.read_csv(p, usecols=usecols, low_memory=False)
    out = pd.DataFrame()
    out["event_id"] = _norm_id(df[event_col])
    out["game_id"] = _norm_id(df[game_col])
    for target, src in col_map.items():
        out[target] = df[src]
    out = out.drop_duplicates(subset=["event_id", "game_id"], keep="last")
    return out, "ok"


def _normalize_game_type(series: pd.Series) -> pd.Series:
    s = series.fillna("").astype(str).str.strip().str.lower()
    out = pd.Series("regular", index=s.index, dtype="object")
    out.loc[s.str.startswith("conf_tournament")] = "conf_tournament"
    out.loc[s.str.startswith("ncaa_")] = s.loc[s.str.startswith("ncaa_")]
    return out


def run_context_overlay(*, data_dir: Path, output_csv: Path, output_md: Path) -> int:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    base, inspected = _load_base_games(data_dir)
    if base.empty:
        empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
        empty.to_csv(output_csv, index=False)
        lines = [
            "# Exec Summary: context_layers/run_all",
            "",
            "- status: `BLOCKED`",
            f"- generated_at_utc: `{_utc_now()}`",
            "- reason: No base games/predictions/results table found with event_id/game_id.",
            "- inspected:",
        ]
        for item in inspected:
            lines.append(f"  - `{item}`")
        output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return 1

    module_status: dict[str, str] = {}
    layers: list[pd.DataFrame] = []

    layer_specs = [
        (
            "tournaments/fatigue_flags.csv",
            {"fatigue_tier": ["fatigue_tier", "rest_tier"]},
        ),
        (
            "tournaments/neutral_adjustments.csv",
            {"neutral_tier": ["neutral_tier", "neutral_adjustment_tier", "neutral_flag"]},
        ),
        (
            "tournaments/leverage_flags.csv",
            {"leverage_tier": ["leverage_tier"]},
        ),
        (
            "tournaments/familiarity_score.csv",
            {"familiarity_tier": ["familiarity_tier"]},
        ),
        (
            "ncaa/upset_profile.csv",
            {"upset_profile_tier": ["upset_profile_tier", "tier"]},
        ),
        (
            "totals/tempo_control_score.csv",
            {"tempo_control_tier": ["tempo_control_tier", "tier"]},
        ),
        (
            "halves/first_half_pace.csv",
            {"fh_fast_start_tier": ["fh_fast_start_tier", "tier"]},
        ),
        (
            "totals/second_half_total_factors.csv",
            {"sh_total_bias_tier": ["sh_total_bias_tier", "tier"]},
        ),
    ]

    for rel, col_candidates in layer_specs:
        layer, status = _load_optional_layer(data_dir, rel, col_candidates)
        module_status[rel] = status
        if not layer.empty:
            layers.append(layer)

    out = base.copy()
    for layer in layers:
        out = out.merge(layer, on=["event_id", "game_id"], how="left")

    out["game_type"] = _normalize_game_type(out.get("game_type", pd.Series("", index=out.index)))
    out["ncaa_round"] = np.where(
        out["game_type"].astype(str).str.startswith("ncaa_"),
        out["game_type"].astype(str).str.replace("ncaa_", "", regex=False),
        "",
    )
    out["tournament_layers_active"] = out["game_type"] == "conf_tournament"
    out["ncaa_layers_active"] = out["game_type"].astype(str).str.startswith("ncaa_")
    out["generated_at_utc"] = _utc_now()

    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    out = out[OUTPUT_COLUMNS].sort_values(["game_datetime_utc", "event_id", "game_id"], kind="mergesort")
    out.to_csv(output_csv, index=False)

    lines = [
        "# Exec Summary: context_layers/run_all",
        "",
        "- status: `OK`",
        f"- generated_at_utc: `{_utc_now()}`",
        f"- output_csv: `{output_csv}`",
        f"- rows: `{len(out)}`",
        "- module_status:",
    ]
    for rel, status in module_status.items():
        lines.append(f"  - `{rel}`: `{status}`")
    lines.append("- inspected base tables:")
    for item in inspected:
        lines.append(f"  - `{item}`")
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-csv", type=Path, default=Path("data/context/context_overlay_latest.csv"))
    parser.add_argument("--output-md", type=Path, default=Path("data/context/context_overlay_exec_summary.md"))
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = args.data_dir if args.data_dir.is_absolute() else repo_root / args.data_dir
    output_csv = args.output_csv if args.output_csv.is_absolute() else repo_root / args.output_csv
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md

    rc = run_context_overlay(data_dir=data_dir, output_csv=output_csv, output_md=output_md)
    print(json.dumps({"output_csv": str(output_csv), "output_md": str(output_md), "exit_code": rc}))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
