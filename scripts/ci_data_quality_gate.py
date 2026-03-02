#!/usr/bin/env python3
"""CI gate that enforces canonical CSV data quality thresholds."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd


CONFIG_PATH = Path("config/data_quality_gate.yml")
REPORT_PATH = Path("data/dq_report_v2.csv")


def _coerce_scalar(raw: str) -> Any:
    raw = raw.strip().strip("\'\"")
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing config: {path}")
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    config: dict[str, Any] = {"defaults": {}, "schedule": {}, "canonical_files": [], "market_lines": {"paths": []}}
    section = None
    current_item: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section = line[:-1].strip()
            current_item = None
            continue
        if section in {"defaults", "schedule"} and ":" in line:
            k, v = [x.strip() for x in line.split(":", 1)]
            config[section][k] = _coerce_scalar(v)
        elif section == "canonical_files":
            s = line.strip()
            if s.startswith("- "):
                current_item = {}
                config["canonical_files"].append(current_item)
                s = s[2:]
                if ":" in s:
                    k, v = [x.strip() for x in s.split(":", 1)]
                    current_item[k] = _coerce_scalar(v)
            elif current_item is not None and ":" in s:
                k, v = [x.strip() for x in s.split(":", 1)]
                current_item[k] = _coerce_scalar(v)
        elif section == "market_lines" and line.strip().startswith("- "):
            config["market_lines"]["paths"].append(_coerce_scalar(line.strip()[2:]))
    return config


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return False


def get_report_row(report_df: pd.DataFrame, pattern: str) -> pd.Series | None:
    matches = report_df[report_df["file_path"].astype(str).map(lambda p: fnmatch(p, pattern))]
    if matches.empty:
        return None
    # Prefer larger row_count if multiple snapshots match.
    return matches.sort_values("row_count", ascending=False).iloc[0]


def schedule_is_offseason(config: dict[str, Any]) -> bool:
    schedule_cfg = config.get("schedule", {}) if isinstance(config.get("schedule"), dict) else {}
    schedule_path = Path(schedule_cfg.get("path", "data/games.csv"))
    if not schedule_path.exists():
        return True

    try:
        sdf = pd.read_csv(schedule_path, low_memory=False)
    except Exception:
        return True

    date_col = schedule_cfg.get("date_column", "game_datetime_utc")
    if date_col not in sdf.columns:
        for candidate in ("game_datetime_utc", "date", "game_date"):
            if candidate in sdf.columns:
                date_col = candidate
                break
        else:
            return True

    datetimes = pd.to_datetime(sdf[date_col], errors="coerce", utc=True)
    if datetimes.dropna().empty:
        return True

    now = pd.Timestamp.now(tz="UTC")
    window_days = int(schedule_cfg.get("active_window_days", 45))
    min_games = int(schedule_cfg.get("min_games_in_window", 1))

    mask = (datetimes >= now - pd.Timedelta(days=window_days)) & (datetimes <= now + pd.Timedelta(days=window_days))
    return int(mask.sum()) < min_games


def run_gate() -> int:
    cfg = load_yaml(CONFIG_PATH)
    if not REPORT_PATH.exists():
        print(f"[FAIL] Missing report file: {REPORT_PATH}")
        return 1

    report_df = pd.read_csv(REPORT_PATH, low_memory=False)
    defaults = cfg.get("defaults", {}) if isinstance(cfg.get("defaults"), dict) else {}
    canonical = cfg.get("canonical_files", []) if isinstance(cfg.get("canonical_files"), list) else []

    default_null_pct = float(defaults.get("max_all_null_column_pct", 10.0))
    min_market_games = int(defaults.get("min_market_games_last_7d", 30))

    violations: list[dict[str, Any]] = []

    for item in canonical:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("path", "")).strip()
        if not pattern:
            continue

        row = get_report_row(report_df, pattern)
        if row is None:
            violations.append({"file": pattern, "reason": "missing_in_report", "value": "n/a", "threshold": "exists"})
            continue

        file_path = str(row["file_path"])
        row_count = int(row.get("row_count", 0))
        expected_min = int(item.get("expected_min_rows", 0))
        if row_count < expected_min:
            violations.append({"file": file_path, "reason": "below_expected_min_rows", "value": row_count, "threshold": expected_min})

        max_null_pct = float(item.get("max_all_null_column_pct", default_null_pct))
        all_null_pct = float(row.get("all_null_col_pct", 0.0))
        if all_null_pct > max_null_pct:
            violations.append({"file": file_path, "reason": "all_null_col_pct_exceeded", "value": all_null_pct, "threshold": max_null_pct})

        rolling_flag = _to_bool(row.get("rolling_broken_flag", False))
        if rolling_flag:
            violations.append({"file": file_path, "reason": "rolling_broken_flag", "value": True, "threshold": False})

    offseason = schedule_is_offseason(cfg)
    market_cfg = cfg.get("market_lines", {}) if isinstance(cfg.get("market_lines"), dict) else {}
    patterns = market_cfg.get("paths", ["data/market_lines_latest.csv", "data/market_lines_*.csv"])
    if not isinstance(patterns, list):
        patterns = ["data/market_lines_latest.csv", "data/market_lines_*.csv"]

    market_games_found = 0
    for pattern in patterns:
        row = get_report_row(report_df, str(pattern))
        if row is None:
            continue
        market_games_found = max(market_games_found, int(row.get("market_games_last_7d", 0)))

    if not offseason and market_games_found < min_market_games:
        violations.append(
            {
                "file": "market_lines_latest_or_snapshots",
                "reason": "market_games_last_7d_below_min",
                "value": market_games_found,
                "threshold": min_market_games,
            }
        )

    if violations:
        print("[FAIL] Data quality gate violations (top 20):")
        top = pd.DataFrame(violations).head(20)
        print(top.to_string(index=False))
        return 1

    print("[PASS] Data quality gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_gate())
