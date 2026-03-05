#!/usr/bin/env python3
"""Shared discovery and normalization helpers for edge QA modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


INPUT_CANDIDATES = [
    "results_log_graded.csv",
    "results_log.csv",
    "predictions_graded.csv",
    "predictions_with_context.csv",
    "predictions_combined_latest.csv",
]


@dataclass
class DatasetSpec:
    source_file: Path
    event_col: str
    game_col: str
    datetime_col: str
    confidence_col: str
    pred_spread_col: str
    market_spread_col: str
    actual_margin_col: str
    pred_total_col: str
    market_total_col: str
    actual_total_col: str


def pick_first(columns: Iterable[str], options: list[str]) -> str:
    cols = set(columns)
    for col in options:
        if col in cols:
            return col
    return ""


def _score_columns(cols: list[str]) -> int:
    score = 0
    if pick_first(cols, ["event_id", "game_id"]):
        score += 1
    if pick_first(cols, ["pred_spread", "predicted_spread", "ens_spread"]):
        score += 1
    if pick_first(cols, ["spread_line", "market_spread", "home_spread_current", "home_spread", "spread"]):
        score += 1
    if pick_first(cols, ["actual_margin"]):
        score += 1
    if pick_first(cols, ["pred_total", "predicted_total", "ens_total"]):
        score += 1
    if pick_first(cols, ["total_line", "market_total", "total_current", "total", "over_under"]):
        score += 1
    if pick_first(cols, ["actual_total"]):
        score += 1
    if pick_first(cols, ["model_confidence", "primary_confidence", "ens_confidence"]):
        score += 1
    return score


def discover_input_table(data_dir: Path) -> tuple[Path | None, list[str]]:
    inspected: list[str] = []
    best_path: Path | None = None
    best_score = -1
    best_rows = -1

    for file_name in INPUT_CANDIDATES:
        path = data_dir / file_name
        if not path.exists() or path.stat().st_size <= 1:
            inspected.append(f"{file_name}:missing_or_empty")
            continue
        try:
            header_df = pd.read_csv(path, nrows=0, low_memory=False)
        except Exception:
            inspected.append(f"{file_name}:unreadable")
            continue
        cols = list(header_df.columns)
        score = _score_columns(cols)
        try:
            rows = int(len(pd.read_csv(path, usecols=[cols[0]], low_memory=False)))
        except Exception:
            rows = 0
        inspected.append(f"{file_name}:score={score}:rows={rows}")
        if (score > best_score) or (score == best_score and rows > best_rows):
            best_score = score
            best_rows = rows
            best_path = path

    return best_path, inspected


def build_dataset_spec(path: Path) -> tuple[DatasetSpec | None, dict[str, list[str]]]:
    header = list(pd.read_csv(path, nrows=0, low_memory=False).columns)
    event_col = pick_first(header, ["event_id", "game_id"])
    game_col = pick_first(header, ["game_id", "event_id"])
    datetime_col = pick_first(header, ["game_datetime_utc", "predicted_at_utc", "captured_at_utc", "game_date"])
    confidence_col = pick_first(header, ["model_confidence", "primary_confidence", "ens_confidence"])
    pred_spread_col = pick_first(header, ["pred_spread", "predicted_spread", "ens_spread"])
    market_spread_col = pick_first(header, ["spread_line", "market_spread", "home_spread_current", "home_spread", "spread"])
    actual_margin_col = pick_first(header, ["actual_margin"])
    pred_total_col = pick_first(header, ["pred_total", "predicted_total", "ens_total"])
    market_total_col = pick_first(header, ["total_line", "market_total", "total_current", "total", "over_under"])
    actual_total_col = pick_first(header, ["actual_total"])

    missing: dict[str, list[str]] = {}
    required_for_both = {
        str(path): [
            x
            for x in [
                "event_id_or_game_id" if not event_col else "",
                "pred_spread/predicted_spread/ens_spread" if not pred_spread_col else "",
                "spread_line/market_spread/home_spread_current/home_spread/spread" if not market_spread_col else "",
                "actual_margin" if not actual_margin_col else "",
                "pred_total/predicted_total/ens_total" if not pred_total_col else "",
                "total_line/market_total/total_current/total/over_under" if not market_total_col else "",
                "actual_total" if not actual_total_col else "",
            ]
            if x
        ]
    }

    if required_for_both[str(path)]:
        missing = required_for_both
        return None, missing

    spec = DatasetSpec(
        source_file=path,
        event_col=event_col,
        game_col=game_col,
        datetime_col=datetime_col,
        confidence_col=confidence_col,
        pred_spread_col=pred_spread_col,
        market_spread_col=market_spread_col,
        actual_margin_col=actual_margin_col,
        pred_total_col=pred_total_col,
        market_total_col=market_total_col,
        actual_total_col=actual_total_col,
    )
    return spec, {}


def load_normalized_frame(spec: DatasetSpec) -> pd.DataFrame:
    df = pd.read_csv(spec.source_file, low_memory=False)
    out = pd.DataFrame()
    out["event_id"] = df[spec.event_col].astype(str)
    out["game_id"] = df[spec.game_col].astype(str)
    out["game_datetime_utc"] = df[spec.datetime_col].astype(str) if spec.datetime_col else ""
    out["confidence"] = pd.to_numeric(df[spec.confidence_col], errors="coerce") if spec.confidence_col else pd.NA
    out["model_spread"] = pd.to_numeric(df[spec.pred_spread_col], errors="coerce")
    out["market_spread"] = pd.to_numeric(df[spec.market_spread_col], errors="coerce")
    out["actual_margin"] = pd.to_numeric(df[spec.actual_margin_col], errors="coerce")
    out["model_total"] = pd.to_numeric(df[spec.pred_total_col], errors="coerce")
    out["market_total"] = pd.to_numeric(df[spec.market_total_col], errors="coerce")
    out["actual_total"] = pd.to_numeric(df[spec.actual_total_col], errors="coerce")
    return out
