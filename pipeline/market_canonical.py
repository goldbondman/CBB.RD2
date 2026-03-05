"""Canonical market-line builder and deterministic merge helpers."""

from __future__ import annotations

import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = [
    "event_id",
    "game_datetime_utc",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
    "book",
    "opening_spread",
    "closing_spread",
    "spread_line",
    "opening_total",
    "closing_total",
    "total_line",
    "moneyline_home",
    "moneyline_away",
    "source",
    "source_file",
    "source_priority",
    "captured_at_utc",
    "line_timestamp_utc",
    "market_status",
]

LATEST_BY_GAME_COLUMNS = CANONICAL_COLUMNS + ["line_source_used"]

MARKET_SOURCE_SPECS = [
    ("market_lines_latest", "market_lines_latest.csv", 0),
    ("odds_snapshot", "odds_snapshot.csv", 1),
    ("market_lines", "market_lines.csv", 2),
    ("market_lines_closing", "market_lines_closing.csv", 3),
    ("market_lines_master", "market_lines_master.csv", 4),
]


def canonicalize_event_id(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    text = text.lstrip("0")
    return text or "0"


def _pick_first_non_null(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    for col in candidates:
        if col in df.columns:
            out = out.where(out.notna(), df[col])
    return out


def _pick_first_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for col in candidates:
        if col in df.columns:
            out = out.fillna(pd.to_numeric(df[col], errors="coerce"))
    return out


def _status_from_lines(spread: pd.Series, total: pd.Series) -> pd.Series:
    status = pd.Series("MISSING", index=spread.index, dtype="object")
    both = spread.notna() & total.notna()
    partial = spread.notna() ^ total.notna()
    status.loc[both] = "OK"
    status.loc[partial] = "PARTIAL"
    return status


def _empty_market_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=CANONICAL_COLUMNS)


def _load_market_source(path: Path, source_name: str, source_priority: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    audit: dict[str, Any] = {
        "source_name": source_name,
        "path": str(path),
        "loaded": False,
        "rows": 0,
        "unique_event_ids": 0,
        "rows_with_any_line": 0,
        "key_column": None,
        "reason": "",
        "updated_at_utc": None,
    }
    if not path.exists() or path.stat().st_size == 0:
        audit["reason"] = "missing_or_empty"
        return _empty_market_frame(), audit

    audit["updated_at_utc"] = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    try:
        raw = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        audit["reason"] = f"read_error:{exc}"
        return _empty_market_frame(), audit

    if raw.empty:
        audit["reason"] = "empty_dataframe"
        return _empty_market_frame(), audit

    id_col = next((c for c in ["event_id", "game_id", "espn_game_id"] if c in raw.columns), None)
    if id_col is None:
        audit["reason"] = "missing_id_column:event_id|game_id|espn_game_id"
        return _empty_market_frame(), audit

    out = pd.DataFrame(index=raw.index)
    out["event_id"] = raw[id_col].map(canonicalize_event_id)
    out = out[out["event_id"] != ""].copy()
    if out.empty:
        audit["reason"] = "no_valid_event_ids"
        return _empty_market_frame(), audit

    out["game_datetime_utc"] = pd.to_datetime(
        _pick_first_non_null(raw.loc[out.index], ["game_datetime_utc", "game_time_utc", "date"]),
        errors="coerce",
        utc=True,
    )
    out["home_team_id"] = _pick_first_non_null(raw.loc[out.index], ["home_team_id"]).map(canonicalize_event_id)
    out["away_team_id"] = _pick_first_non_null(raw.loc[out.index], ["away_team_id"]).map(canonicalize_event_id)
    out["home_team_name"] = _pick_first_non_null(raw.loc[out.index], ["home_team_name", "home_team"])
    out["away_team_name"] = _pick_first_non_null(raw.loc[out.index], ["away_team_name", "away_team"])
    out["book"] = _pick_first_non_null(raw.loc[out.index], ["book"])
    out["opening_spread"] = _pick_first_numeric(raw.loc[out.index], ["opening_spread", "home_spread_open", "spread_open"])
    out["closing_spread"] = _pick_first_numeric(
        raw.loc[out.index],
        ["closing_spread", "home_spread_current", "home_spread", "spread_line", "market_spread", "spread"],
    )
    out["spread_line"] = _pick_first_numeric(raw.loc[out.index], ["spread_line"])
    out["spread_line"] = out["spread_line"].fillna(out["closing_spread"]).fillna(out["opening_spread"])
    out["opening_total"] = _pick_first_numeric(raw.loc[out.index], ["opening_total", "total_open"])
    out["closing_total"] = _pick_first_numeric(
        raw.loc[out.index],
        ["closing_total", "total_current", "total_line", "market_total", "over_under", "total"],
    )
    out["total_line"] = _pick_first_numeric(raw.loc[out.index], ["total_line", "market_total", "over_under", "total"])
    out["total_line"] = out["total_line"].fillna(out["closing_total"]).fillna(out["opening_total"])
    out["moneyline_home"] = _pick_first_numeric(raw.loc[out.index], ["moneyline_home", "home_ml"])
    out["moneyline_away"] = _pick_first_numeric(raw.loc[out.index], ["moneyline_away", "away_ml"])
    out["source"] = _pick_first_non_null(raw.loc[out.index], ["source"]).fillna(source_name)
    out["source_file"] = path.name
    out["source_priority"] = source_priority

    captured = pd.to_datetime(
        _pick_first_non_null(raw.loc[out.index], ["captured_at_utc", "pulled_at_utc", "snapshot_ts_utc", "created_at"]),
        errors="coerce",
        utc=True,
    )
    out["captured_at_utc"] = captured
    out["line_timestamp_utc"] = captured
    out["market_status"] = _status_from_lines(out["spread_line"], out["total_line"])

    for col in CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[CANONICAL_COLUMNS]

    audit["loaded"] = True
    audit["rows"] = int(len(out))
    audit["unique_event_ids"] = int(out["event_id"].nunique())
    audit["rows_with_any_line"] = int((out["spread_line"].notna() | out["total_line"].notna()).sum())
    audit["key_column"] = id_col
    audit["reason"] = "ok"
    return out, audit


def _build_games_fallback(data_dir: Path) -> pd.DataFrame:
    games_path = data_dir / "games.csv"
    if not games_path.exists() or games_path.stat().st_size == 0:
        return _empty_market_frame()
    games = pd.read_csv(games_path, low_memory=False)
    if games.empty:
        return _empty_market_frame()

    out = pd.DataFrame(index=games.index)
    out["event_id"] = _pick_first_non_null(games, ["event_id", "game_id"]).map(canonicalize_event_id)
    out = out[out["event_id"] != ""].copy()
    if out.empty:
        return _empty_market_frame()
    out["game_datetime_utc"] = pd.to_datetime(games.loc[out.index].get("game_datetime_utc"), errors="coerce", utc=True)
    out["home_team_id"] = _pick_first_non_null(games.loc[out.index], ["home_team_id"]).map(canonicalize_event_id)
    out["away_team_id"] = _pick_first_non_null(games.loc[out.index], ["away_team_id"]).map(canonicalize_event_id)
    out["home_team_name"] = _pick_first_non_null(games.loc[out.index], ["home_team_name", "home_team"])
    out["away_team_name"] = _pick_first_non_null(games.loc[out.index], ["away_team_name", "away_team"])
    out["book"] = pd.NA
    out["opening_spread"] = pd.to_numeric(games.loc[out.index].get("spread"), errors="coerce")
    out["closing_spread"] = pd.to_numeric(games.loc[out.index].get("spread"), errors="coerce")
    out["spread_line"] = pd.to_numeric(games.loc[out.index].get("spread"), errors="coerce")
    out["opening_total"] = pd.to_numeric(games.loc[out.index].get("over_under"), errors="coerce")
    out["closing_total"] = pd.to_numeric(games.loc[out.index].get("over_under"), errors="coerce")
    out["total_line"] = pd.to_numeric(games.loc[out.index].get("over_under"), errors="coerce")
    out["moneyline_home"] = pd.to_numeric(games.loc[out.index].get("home_ml"), errors="coerce")
    out["moneyline_away"] = pd.to_numeric(games.loc[out.index].get("away_ml"), errors="coerce")
    out["source"] = "espn_games"
    out["source_file"] = "games.csv"
    out["source_priority"] = 90
    out["captured_at_utc"] = pd.NaT
    out["line_timestamp_utc"] = pd.NaT
    out["market_status"] = _status_from_lines(out["spread_line"], out["total_line"])
    return out[CANONICAL_COLUMNS]


def _status_rank(series: pd.Series) -> pd.Series:
    return series.map({"OK": 0, "PARTIAL": 1, "MISSING": 2}).fillna(3).astype(int)


def build_market_canonical_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    source_audits: list[dict[str, Any]] = []
    source_frames: list[pd.DataFrame] = []
    for source_name, file_name, priority in MARKET_SOURCE_SPECS:
        frame, audit = _load_market_source(data_dir / file_name, source_name, priority)
        source_audits.append(audit)
        if not frame.empty:
            source_frames.append(frame)

    canonical = pd.concat(source_frames, ignore_index=True, sort=False) if source_frames else _empty_market_frame()
    if not canonical.empty:
        canonical["book"] = canonical["book"].astype("string").fillna("consensus")
        canonical = canonical.sort_values(
            ["event_id", "book", "line_timestamp_utc", "source_priority"],
            ascending=[True, True, False, True],
            kind="mergesort",
        )
        canonical = canonical.drop_duplicates(
            subset=["event_id", "book", "line_timestamp_utc", "spread_line", "total_line", "source_file"],
            keep="first",
        )

    if canonical.empty:
        latest_by_book = _empty_market_frame()
    else:
        canonical_book = canonical.copy()
        canonical_book["_status_rank"] = _status_rank(canonical_book["market_status"])
        canonical_book = canonical_book.sort_values(
            ["event_id", "book", "_status_rank", "source_priority", "line_timestamp_utc"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
        latest_by_book = canonical_book.groupby(["event_id", "book"], as_index=False).first()[CANONICAL_COLUMNS]

    if latest_by_book.empty:
        latest_by_game = pd.DataFrame(columns=LATEST_BY_GAME_COLUMNS)
    else:
        game_pick = latest_by_book.copy()
        game_pick["_status_rank"] = _status_rank(game_pick["market_status"])
        game_pick = game_pick.sort_values(
            ["event_id", "_status_rank", "source_priority", "line_timestamp_utc"],
            ascending=[True, True, True, False],
            kind="mergesort",
        )
        latest_by_game = game_pick.groupby("event_id", as_index=False).first()[CANONICAL_COLUMNS]
        latest_by_game["line_source_used"] = latest_by_game["source"]

    games_fallback = _build_games_fallback(data_dir)
    fallback_used = 0
    if not games_fallback.empty:
        if latest_by_game.empty:
            latest_by_game = games_fallback.copy()
            latest_by_game["line_source_used"] = latest_by_game["source"]
            fallback_used = len(latest_by_game)
        else:
            latest = latest_by_game.set_index("event_id")
            games_fallback = games_fallback.drop_duplicates("event_id", keep="last").set_index("event_id")
            common_ids = latest.index.intersection(games_fallback.index)
            for event_id in common_ids:
                spread_missing = pd.isna(latest.at[event_id, "spread_line"])
                total_missing = pd.isna(latest.at[event_id, "total_line"])
                if spread_missing and pd.notna(games_fallback.at[event_id, "spread_line"]):
                    latest.at[event_id, "spread_line"] = games_fallback.at[event_id, "spread_line"]
                    latest.at[event_id, "opening_spread"] = games_fallback.at[event_id, "opening_spread"]
                    latest.at[event_id, "closing_spread"] = games_fallback.at[event_id, "closing_spread"]
                    fallback_used += 1
                if total_missing and pd.notna(games_fallback.at[event_id, "total_line"]):
                    latest.at[event_id, "total_line"] = games_fallback.at[event_id, "total_line"]
                    latest.at[event_id, "opening_total"] = games_fallback.at[event_id, "opening_total"]
                    latest.at[event_id, "closing_total"] = games_fallback.at[event_id, "closing_total"]
                    fallback_used += 1
                if (spread_missing or total_missing) and latest.at[event_id, "line_source_used"] in {pd.NA, None, ""}:
                    latest.at[event_id, "line_source_used"] = "espn_games"

            missing_ids = games_fallback.index.difference(latest.index)
            if len(missing_ids) > 0:
                appended = games_fallback.loc[missing_ids].copy()
                appended["line_source_used"] = appended["source"]
                latest = pd.concat([latest, appended], axis=0)
                fallback_used += len(appended)
            latest_by_game = latest.reset_index()
            latest_by_game["market_status"] = _status_from_lines(latest_by_game["spread_line"], latest_by_game["total_line"])

    if latest_by_game.empty:
        missing_games = pd.DataFrame(columns=["event_id", "game_datetime_utc", "home_team_name", "away_team_name", "missing_reason"])
    else:
        missing_games = latest_by_game[latest_by_game["market_status"] != "OK"][
            ["event_id", "game_datetime_utc", "home_team_name", "away_team_name", "market_status"]
        ].rename(columns={"market_status": "missing_reason"})

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": source_audits,
        "canonical_rows": int(len(canonical)),
        "canonical_unique_event_ids": int(canonical["event_id"].nunique()) if not canonical.empty else 0,
        "latest_by_book_rows": int(len(latest_by_book)),
        "latest_by_game_rows": int(len(latest_by_game)),
        "latest_by_game_ok_rows": int((latest_by_game.get("market_status") == "OK").sum()) if not latest_by_game.empty else 0,
        "latest_by_game_partial_rows": int((latest_by_game.get("market_status") == "PARTIAL").sum()) if not latest_by_game.empty else 0,
        "latest_by_game_missing_rows": int((latest_by_game.get("market_status") == "MISSING").sum()) if not latest_by_game.empty else 0,
        "fallback_rows_used": int(fallback_used),
    }
    return canonical, latest_by_book, latest_by_game, audit, missing_games


def _write_market_inventory_md(data_dir: Path, debug_dir: Path, source_audits: list[dict[str, Any]]) -> None:
    lines = [
        "# Market Inventory",
        "",
        "| path | loaded | rows | unique_event_ids | key_column | updated_at_utc |",
        "|---|---:|---:|---:|---|---|",
    ]
    for audit in source_audits:
        lines.append(
            f"| {audit.get('path')} | {audit.get('loaded')} | {audit.get('rows')} | "
            f"{audit.get('unique_event_ids')} | {audit.get('key_column')} | {audit.get('updated_at_utc')} |"
        )
    (debug_dir / "market_inventory.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_market_canonical_outputs(data_dir: Path, debug_dir: Optional[Path] = None) -> dict[str, Any]:
    debug = debug_dir or (data_dir.parent / "debug")
    debug.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    canonical, latest_by_book, latest_by_game, audit, missing_games = build_market_canonical_tables(data_dir)
    canonical_path = data_dir / "market_lines_canonical.csv"
    latest_game_path = data_dir / "market_lines_latest_by_game.csv"
    latest_book_path = data_dir / "market_lines_latest_by_book.csv"
    audit_path = debug / "market_builder_audit.json"
    missing_path = debug / "market_missing_games.csv"

    canonical.to_csv(canonical_path, index=False)
    latest_by_game.to_csv(latest_game_path, index=False)
    latest_by_book.to_csv(latest_book_path, index=False)
    missing_games.to_csv(missing_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _write_market_inventory_md(data_dir, debug, audit.get("sources", []))

    return {
        "canonical_path": str(canonical_path),
        "latest_by_game_path": str(latest_game_path),
        "latest_by_book_path": str(latest_book_path),
        "audit_path": str(audit_path),
        "missing_path": str(missing_path),
        "audit": audit,
    }


def load_latest_by_game(data_dir: Path, auto_build: bool = True) -> pd.DataFrame:
    latest_path = data_dir / "market_lines_latest_by_game.csv"
    if latest_path.exists() and latest_path.stat().st_size > 0:
        df = pd.read_csv(latest_path, low_memory=False)
    elif auto_build:
        write_market_canonical_outputs(data_dir)
        if latest_path.exists() and latest_path.stat().st_size > 0:
            df = pd.read_csv(latest_path, low_memory=False)
        else:
            return pd.DataFrame(columns=LATEST_BY_GAME_COLUMNS)
    else:
        return pd.DataFrame(columns=LATEST_BY_GAME_COLUMNS)

    if "event_id" not in df.columns and "game_id" in df.columns:
        df["event_id"] = df["game_id"]
    if "event_id" in df.columns:
        df["event_id"] = df["event_id"].map(canonicalize_event_id)
    if "line_timestamp_utc" in df.columns:
        df["line_timestamp_utc"] = pd.to_datetime(df["line_timestamp_utc"], utc=True, errors="coerce")
    return df


def merge_market_lines(
    df: pd.DataFrame,
    *,
    data_dir: Path,
    output_name: str,
    required_columns: Optional[list[str]] = None,
    min_match_rate: Optional[float] = None,
    debug_dir: Optional[Path] = None,
) -> pd.DataFrame:
    debug = debug_dir or (data_dir.parent / "debug")
    debug.mkdir(parents=True, exist_ok=True)
    coverage_path = debug / "market_merge_coverage.csv"

    out = df.copy()
    if "event_id" not in out.columns:
        if "game_id" in out.columns:
            out["event_id"] = out["game_id"]
        else:
            out["event_id"] = pd.NA
    out["event_id"] = out["event_id"].map(canonicalize_event_id)

    latest = load_latest_by_game(data_dir, auto_build=True)
    if latest.empty:
        for col in ["opening_spread", "closing_spread", "spread_line", "opening_total", "closing_total", "total_line", "moneyline_home", "moneyline_away", "line_source_used", "line_timestamp_utc", "market_status"]:
            if col not in out.columns:
                out[col] = pd.NA
        match_rate = 0.0
    else:
        keep_cols = [
            "event_id",
            "opening_spread",
            "closing_spread",
            "spread_line",
            "opening_total",
            "closing_total",
            "total_line",
            "moneyline_home",
            "moneyline_away",
            "line_source_used",
            "line_timestamp_utc",
            "market_status",
            "source",
        ]
        available = [c for c in keep_cols if c in latest.columns]
        merged = out.merge(latest[available], on="event_id", how="left", suffixes=("", "_market"))
        for col in [c for c in keep_cols if c != "event_id"]:
            market_col = f"{col}_market"
            if market_col in merged.columns:
                if col in merged.columns:
                    merged[col] = merged[col].combine_first(merged[market_col])
                else:
                    merged[col] = merged[market_col]
                merged = merged.drop(columns=[market_col], errors="ignore")
        out = merged
        spread_present = pd.to_numeric(out.get("spread_line"), errors="coerce").notna()
        total_present = pd.to_numeric(out.get("total_line"), errors="coerce").notna()
        match_rate = float((spread_present | total_present).mean()) if len(out) else 0.0

    missing_required: list[str] = []
    required = required_columns or []
    for col in required:
        if col not in out.columns:
            missing_required.append(col)
            out[col] = pd.NA

    coverage_row = pd.DataFrame(
        [
            {
                "output_name": output_name,
                "rows": int(len(out)),
                "market_matched_rows": int((pd.to_numeric(out.get("spread_line"), errors="coerce").notna() | pd.to_numeric(out.get("total_line"), errors="coerce").notna()).sum()) if len(out) else 0,
                "match_rate": round(match_rate, 6),
                "missing_columns": ",".join(missing_required),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }
        ]
    )
    if coverage_path.exists() and coverage_path.stat().st_size > 0:
        prev = pd.read_csv(coverage_path, low_memory=False)
        coverage = pd.concat([prev, coverage_row], ignore_index=True)
    else:
        coverage = coverage_row
    coverage.to_csv(coverage_path, index=False)

    if min_match_rate is not None and match_rate < min_match_rate:
        raise RuntimeError(
            f"Market merge coverage below threshold for {output_name}: "
            f"match_rate={match_rate:.3f} required>={min_match_rate:.3f}"
        )

    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build canonical market-line tables and audits")
    parser.add_argument("--data-dir", default="data", help="Data directory containing market and games CSVs")
    parser.add_argument("--debug-dir", default="debug", help="Debug directory for audit outputs")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_dir = Path(args.data_dir)
    debug_dir = Path(args.debug_dir)
    result = write_market_canonical_outputs(data_dir=data_dir, debug_dir=debug_dir)
    audit = result.get("audit", {})
    print(
        f"[OK] canonical_rows={audit.get('canonical_rows', 0)} "
        f"latest_by_game_rows={audit.get('latest_by_game_rows', 0)} "
        f"latest_by_game_ok_rows={audit.get('latest_by_game_ok_rows', 0)}"
    )
    print(f"[OK] wrote {result.get('canonical_path')}")
    print(f"[OK] wrote {result.get('latest_by_game_path')}")


if __name__ == "__main__":
    main()
