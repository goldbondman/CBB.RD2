#!/usr/bin/env python3
"""Generate CSV data quality report for CI gating."""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import pandas as pd


CONFIG_PATH = Path("config/data_quality_gate.yml")
OUTPUT_CSV = Path("data/dq_report_v2.csv")
OUTPUT_MD = Path("data/dq_report_v2.md")


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing config file: {path}")
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return raw if isinstance(raw, dict) else {}
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


def expected_min_for(file_path: str, canonical_files: list[dict[str, Any]]) -> int | None:
    for item in canonical_files:
        pattern = item.get("path")
        if isinstance(pattern, str) and fnmatch(file_path, pattern):
            value = item.get("expected_min_rows")
            if isinstance(value, int):
                return value
    return None


def detect_date_column(df: pd.DataFrame) -> str | None:
    for col in ("game_datetime_utc", "commence_time", "game_date", "date", "start_time"):
        if col in df.columns:
            return col
    return None


def build_report() -> int:
    config = load_config(CONFIG_PATH)
    canonical = config.get("canonical_files", []) if isinstance(config.get("canonical_files"), list) else []

    rows: list[dict[str, Any]] = []
    for csv_path in sorted(Path("data").glob("*.csv")):
        rel = csv_path.as_posix()
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "file_path": rel,
                    "read_ok": False,
                    "read_error": str(exc),
                    "row_count": 0,
                    "col_count": 0,
                    "all_null_cols": 0,
                    "all_null_col_pct": 0.0,
                    "rolling_col_count": 0,
                    "rolling_all_null_col_count": 0,
                    "rolling_broken_flag": True,
                    "expected_min_rows": expected_min_for(rel, canonical),
                    "below_expected_min_rows": False,
                    "market_games_last_7d": 0,
                }
            )
            continue

        row_count = int(len(df))
        col_count = int(len(df.columns))
        null_counts = df.isna().sum() if col_count else pd.Series(dtype=int)
        all_null_cols = int((null_counts == row_count).sum()) if row_count > 0 else int(col_count)
        all_null_col_pct = round((all_null_cols / col_count) * 100.0, 2) if col_count else 0.0

        rolling_cols = [c for c in df.columns if ("rolling" in c.lower()) or fnmatch(c.lower(), "*_l[0-9]*")]
        rolling_col_count = len(rolling_cols)
        rolling_all_null_col_count = 0
        if rolling_col_count:
            rolling_all_null_col_count = int((df[rolling_cols].isna().sum() == row_count).sum()) if row_count > 0 else rolling_col_count
        rolling_broken_flag = bool(rolling_col_count > 0 and rolling_all_null_col_count == rolling_col_count)

        market_games_last_7d = 0
        if "market_lines" in csv_path.stem.lower() and row_count > 0:
            date_col = detect_date_column(df)
            if date_col is not None:
                dts = pd.to_datetime(df[date_col], errors="coerce", utc=True)
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
                recent = df.loc[dts >= cutoff].copy()
                if not recent.empty:
                    if "game_id" in recent.columns:
                        market_games_last_7d = int(recent["game_id"].astype(str).nunique())
                    else:
                        keys = [c for c in ("external_game_id", "home_team", "away_team") if c in recent.columns]
                        if keys:
                            market_games_last_7d = int(recent[keys].astype(str).agg("|".join, axis=1).nunique())

        expected_min_rows = expected_min_for(rel, canonical)
        below_expected = bool(expected_min_rows is not None and row_count < expected_min_rows)
        rows.append(
            {
                "file_path": rel,
                "read_ok": True,
                "read_error": "",
                "row_count": row_count,
                "col_count": col_count,
                "all_null_cols": all_null_cols,
                "all_null_col_pct": all_null_col_pct,
                "rolling_col_count": rolling_col_count,
                "rolling_all_null_col_count": rolling_all_null_col_count,
                "rolling_broken_flag": rolling_broken_flag,
                "expected_min_rows": expected_min_rows,
                "below_expected_min_rows": below_expected,
                "market_games_last_7d": market_games_last_7d,
            }
        )

    report_df = pd.DataFrame(rows).sort_values(["file_path"]).reset_index(drop=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(OUTPUT_CSV, index=False)

    violations = report_df[
        (~report_df["read_ok"]) |
        (report_df["below_expected_min_rows"]) |
        (report_df["rolling_broken_flag"]) |
        (report_df["all_null_col_pct"] > 10.0)
    ]
    OUTPUT_MD.write_text(
        "# CSV Quality Audit v2\n\n"
        f"- Files audited: {len(report_df)}\n"
        f"- Violations: {len(violations)}\n\n"
        "## Top Violations\n\n"
        + (violations.head(20).to_csv(index=False) if not violations.empty else "None\n"),
        encoding="utf-8",
    )
    print(f"[OK] wrote {OUTPUT_CSV}")
    print(f"[OK] wrote {OUTPUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(build_report())
