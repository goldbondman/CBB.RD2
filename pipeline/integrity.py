from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class IntegrityResult:
    ok: bool
    errors: list[str]
    warnings: list[str]
    row_counts: dict[str, int]


def _require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> list[str]:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return [f"{label}: missing required columns: {missing}"]
    return []


def run_integrity_gate(
    *,
    data_dir: Path,
    as_of: pd.Timestamp,
    mode: str,
) -> IntegrityResult:
    errors: list[str] = []
    warnings: list[str] = []
    row_counts: dict[str, int] = {}

    required_files = ["games.csv", "market_lines.csv"]
    if mode == "backtest":
        required_files.extend(["predictions_combined_latest.csv", "results_log.csv"])
    else:
        required_files.extend(["team_game_weighted.csv"])

    for file_name in required_files:
        path = data_dir / file_name
        if not path.exists():
            errors.append(f"Missing required input file: {path}")

    if errors:
        return IntegrityResult(False, errors, warnings, row_counts)

    games = pd.read_csv(data_dir / "games.csv", low_memory=False)
    row_counts["games"] = len(games)
    errors.extend(_require_columns(games, ["game_id", "game_datetime_utc"], "games.csv"))

    if "game_datetime_utc" in games.columns:
        dt = pd.to_datetime(games["game_datetime_utc"], utc=True, errors="coerce")
        bad_ts = int((dt > as_of).sum())
        if bad_ts:
            errors.append(f"games.csv has {bad_ts} rows newer than as_of={as_of.isoformat()}")

    lines = pd.read_csv(data_dir / "market_lines.csv", low_memory=False)
    row_counts["market_lines"] = len(lines)
    errors.extend(_require_columns(lines, ["game_id", "pulled_at"], "market_lines.csv"))

    if "pulled_at" in lines.columns:
        pulled_at = pd.to_datetime(lines["pulled_at"], utc=True, errors="coerce")
        late_lines = int((pulled_at > as_of).sum())
        if late_lines:
            errors.append(f"market_lines.csv has {late_lines} rows newer than as_of={as_of.isoformat()}")

    if mode == "backtest":
        preds = pd.read_csv(data_dir / "predictions_combined_latest.csv", low_memory=False)
        row_counts["predictions_combined_latest"] = len(preds)
        errors.extend(_require_columns(preds, ["game_id"], "predictions_combined_latest.csv"))
        if "game_id" in preds.columns and preds["game_id"].duplicated().any():
            dups = int(preds["game_id"].duplicated().sum())
            errors.append(f"predictions_combined_latest.csv has duplicate game_id rows: {dups}")

        schedule_ids = set(games.get("game_id", pd.Series(dtype=str)).dropna().astype(str))
        pred_ids = set(preds.get("game_id", pd.Series(dtype=str)).dropna().astype(str))
        missing_in_games = sorted(pred_ids - schedule_ids)
        if missing_in_games:
            errors.append(
                "predictions_combined_latest.csv has game_id not in games.csv: "
                f"count={len(missing_in_games)}"
            )

    return IntegrityResult(ok=not errors, errors=errors, warnings=warnings, row_counts=row_counts)


def write_integrity_report(path: Path, result: IntegrityResult, run_context: dict) -> None:
    payload = {
        "ok": result.ok,
        "errors": result.errors,
        "warnings": result.warnings,
        "row_counts": result.row_counts,
        "run_context": run_context,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
