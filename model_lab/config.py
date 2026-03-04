from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_MODEL_NAMES = [
    "FourFactors",
    "AdjEfficiency",
    "Pythagorean",
    "Situational",
    "CAGERankings",
    "LuckRegression",
    "Variance",
    "HomeAwayForm",
]


@dataclass
class ModelLabConfig:
    repo_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = field(default_factory=lambda: Path("data"))
    runs_dir: Path = field(default_factory=lambda: Path("data") / "model_lab_runs")

    matchup_metrics_path: Path = field(default_factory=lambda: Path("data") / "matchup_metrics.csv")
    team_game_metrics_path: Path = field(default_factory=lambda: Path("data") / "team_game_metrics.csv")

    results_graded_path: Path = field(default_factory=lambda: Path("data") / "results_log_graded.csv")
    results_log_path: Path = field(default_factory=lambda: Path("data") / "results_log.csv")
    backtest_training_path: Path = field(default_factory=lambda: Path("data") / "backtest_training_data.csv")
    backtest_results_path: Path = field(default_factory=lambda: Path("data") / "backtest_results_latest.csv")

    market_closing_path: Path = field(default_factory=lambda: Path("data") / "market_lines_closing.csv")
    market_lines_path: Path = field(default_factory=lambda: Path("data") / "market_lines.csv")
    games_path: Path = field(default_factory=lambda: Path("data") / "games.csv")

    predictions_combined_path: Path = field(default_factory=lambda: Path("data") / "predictions_combined_latest.csv")
    ensemble_latest_path: Path = field(default_factory=lambda: Path("data") / "ensemble_predictions_latest.csv")

    min_train_games: int = 200
    min_test_games: int = 75
    date_test_window_days: int = 21
    date_step_days: int = 14

    default_odds: int = -110
    calibration_bins: int = 10

    max_weight: float = 0.5


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_run_dir(config: ModelLabConfig, run_id: str) -> Path:
    run_dir = (config.repo_root / config.runs_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def canonicalize_game_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = "".join(ch for ch in text if ch.isdigit())
    if not text:
        return ""
    return text.lstrip("0") or "0"


def derive_season_id(dt_series: pd.Series, season_series: pd.Series | None = None) -> pd.Series:
    if season_series is not None:
        season_vals = pd.to_numeric(season_series, errors="coerce")
    else:
        season_vals = pd.Series(pd.NA, index=dt_series.index, dtype="Float64")

    dt = pd.to_datetime(dt_series, utc=True, errors="coerce")
    derived = dt.dt.year.where(dt.dt.month < 10, dt.dt.year + 1)
    merged = season_vals.fillna(derived)
    return merged.astype("Int64")


def get_git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        sha = out.strip()
        return sha or None
    except Exception:
        return None


def dataframe_nan_report(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    if df.empty:
        return report

    null_count = df.isna().sum()
    null_rate = df.isna().mean()
    for col in df.columns:
        count = int(null_count[col])
        if count == 0:
            continue
        report[col] = {
            "null_count": float(count),
            "null_rate": float(round(null_rate[col], 6)),
        }
    return report


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_manifest(run_dir: Path, payload: dict[str, Any]) -> Path:
    path = run_dir / "run_manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def update_manifest(run_dir: Path, patch: dict[str, Any]) -> Path:
    current = load_manifest(run_dir)
    current.update(patch)
    return write_manifest(run_dir, current)
