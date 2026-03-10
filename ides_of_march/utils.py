from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def canonical_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    text = text.lstrip("0")
    return text or "0"


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def to_rate(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    return np.where(num.abs() > 1.5, num / 100.0, num)


def logistic(x: np.ndarray | pd.Series | float, scale: float = 6.0) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    scale = max(float(scale), 1e-6)
    return 1.0 / (1.0 + np.exp(-arr / scale))


def clip_series(series: pd.Series, lo: float, hi: float) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=lo, upper=hi)


def season_id_from_dt(ts: pd.Series) -> pd.Series:
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    return np.where(dt.dt.month >= 7, dt.dt.year + 1, dt.dt.year)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def sign_signal(value: float, eps: float = 1e-9) -> int:
    if not np.isfinite(value) or abs(value) <= eps:
        return 0
    return 1 if value > 0 else -1


def normalize_dt(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    return pd.to_datetime(df[col], utc=True, errors="coerce")


def margin_to_home_spread(projected_margin_home: pd.Series) -> pd.Series:
    return -pd.to_numeric(projected_margin_home, errors="coerce")


def home_spread_to_margin(spread_line_home: pd.Series) -> pd.Series:
    return -pd.to_numeric(spread_line_home, errors="coerce")


def pretty_exception(exc: Exception) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def expanding_windows(n_rows: int, min_train: int, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows <= min_train or min_train <= 0:
        return []
    start = min_train
    test_points = np.linspace(start, n_rows - 1, num=n_splits + 1, dtype=int)
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(len(test_points) - 1):
        train_end = int(test_points[i])
        test_end = int(test_points[i + 1])
        if test_end <= train_end:
            continue
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(train_end, test_end)
        if len(test_idx) == 0:
            continue
        windows.append((train_idx, test_idx))
    return windows


def maybe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return default
        return out
    except Exception:
        return default
