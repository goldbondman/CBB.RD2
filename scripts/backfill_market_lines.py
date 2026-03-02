"""Backfill market line snapshots into data/market_lines_snapshots.csv.

This script reuses the same market normalization logic as ingestion.market_lines and
applies the same dedupe key shape used by the main market pipeline:
(event_id, capture_type, capture_hour).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingestion.market_lines import (
    REQUEST_DELAY,
    build_book_index,
    build_market_row,
    fetch_action_network,
    fetch_draftkings_lines,
    fetch_espn_scoreboard,
    fetch_pinnacle_lines,
    normalize_team_name,
    parse_action_network_game,
    parse_espn_event,
)
from pipeline_csv_utils import normalize_numeric_dtypes

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

SNAPSHOT_COLUMNS = [
    "event_id",
    "capture_type",
    "captured_at_utc",
    "pulled_at_utc",
    "snapshot_date",
    "verification_status",
    "verification_notes",
    "home_team_id",
    "away_team_id",
    "home_team_name",
    "away_team_name",
    "home_spread_open",
    "home_spread_current",
    "line_movement",
    "total_open",
    "total_current",
    "pinnacle_spread",
    "draftkings_spread",
    "home_tickets_pct",
    "away_tickets_pct",
    "home_money_pct",
    "away_money_pct",
    "steam_flag",
    "rlm_flag",
    "rlm_sharp_side",
    "rlm_note",
    "book_disagreement_flag",
    "book_spread_diff",
    "book_sharp_side",
    "book_note",
    "line_freeze_flag",
    "home_win_prob",
    "away_win_prob",
    "home_ats_wins",
    "home_ats_losses",
    "away_ats_wins",
    "away_ats_losses",
]


@dataclass
class BackfillSummary:
    days_processed: int = 0
    games_found: int = 0
    snapshots_fetched: int = 0
    new_rows_added: int = 0
    duplicates_skipped: int = 0


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def determine_date_range(days: int, start_date: str | None, end_date: str | None) -> tuple[date, date]:
    if start_date and end_date:
        start, end = _parse_date(start_date), _parse_date(end_date)
    elif start_date:
        start = _parse_date(start_date)
        end = date.today()
    elif end_date:
        end = _parse_date(end_date)
        start = end - timedelta(days=days - 1)
    else:
        end = date.today()
        start = end - timedelta(days=days - 1)

    if start > end:
        raise ValueError("start_date cannot be after end_date")
    return start, end


def iter_days(start: date, end: date) -> list[date]:
    span = (end - start).days
    return [start + timedelta(days=offset) for offset in range(span + 1)]


def bootstrap_snapshots(path: Path) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, dtype={"event_id": str}, low_memory=False)
        existing = normalize_numeric_dtypes(existing)
    else:
        existing = pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    for col in SNAPSHOT_COLUMNS:
        if col not in existing.columns:
            existing[col] = pd.NA
    return existing


def _capture_for_day(target_day: date, books: set[str]) -> tuple[int, list[dict]]:
    espn_events = fetch_espn_scoreboard(target_day)
    games_found = len(espn_events)

    an_games = fetch_action_network(target_day)
    time.sleep(REQUEST_DELAY)

    pinnacle_by_team: dict[tuple[str, str], dict] = {}
    if "pinnacle" in books:
        pinnacle_by_team = build_book_index(fetch_pinnacle_lines())
        time.sleep(REQUEST_DELAY)

    dk_by_team: dict[tuple[str, str], dict] = {}
    if "draftkings" in books:
        dk_by_team = build_book_index(fetch_draftkings_lines())
        time.sleep(REQUEST_DELAY)

    action_by_team: dict[tuple[str, str], dict] = {}
    for game in an_games:
        parsed = parse_action_network_game(game)
        if not parsed:
            continue
        key = (
            normalize_team_name(parsed.get("home_team_name", "")),
            normalize_team_name(parsed.get("away_team_name", "")),
        )
        action_by_team[key] = parsed

    rows: list[dict] = []
    for event in espn_events:
        parsed = parse_espn_event(event)
        if not parsed or not parsed.get("event_id"):
            continue

        espn_key = (
            normalize_team_name(parsed.get("home_team_name", "")),
            normalize_team_name(parsed.get("away_team_name", "")),
        )
        an_enrichment = action_by_team.get(espn_key) or action_by_team.get((espn_key[1], espn_key[0]))
        if an_enrichment:
            parsed["home_tickets_pct"] = an_enrichment.get("home_tickets_pct")
            parsed["away_tickets_pct"] = an_enrichment.get("away_tickets_pct")
            parsed["home_money_pct"] = an_enrichment.get("home_money_pct")
            parsed["away_money_pct"] = an_enrichment.get("away_money_pct")
            if parsed.get("home_spread_open") is None:
                parsed["home_spread_open"] = an_enrichment.get("home_spread_open")
            if parsed.get("total_open") is None:
                parsed["total_open"] = an_enrichment.get("total_open")

        team_key = (
            normalize_team_name(str(parsed.get("home_team_name", ""))),
            normalize_team_name(str(parsed.get("away_team_name", ""))),
        )
        pinn_match = pinnacle_by_team.get(team_key) or pinnacle_by_team.get((team_key[1], team_key[0]))
        dk_match = dk_by_team.get(team_key) or dk_by_team.get((team_key[1], team_key[0]))

        row = build_market_row(str(parsed["event_id"]), "backfill", parsed, pinn_match, dk_match, pd.DataFrame())
        row["snapshot_date"] = target_day.isoformat()
        row["home_team_id"] = parsed.get("home_team_id")
        row["away_team_id"] = parsed.get("away_team_id")
        row["home_team_name"] = parsed.get("home_team_name")
        row["away_team_name"] = parsed.get("away_team_name")
        row.setdefault("verification_status", "partial")
        row["verification_notes"] = (
            f"backfill_capture_for_{target_day.isoformat()}_from_current_line_sources"
        )
        rows.append(row)

    return games_found, rows


def _merge_rows(existing: pd.DataFrame, new_rows: list[dict]) -> tuple[pd.DataFrame, int, int]:
    df_new = pd.DataFrame(new_rows)
    if df_new.empty:
        return existing, 0, 0

    for col in SNAPSHOT_COLUMNS:
        if col not in df_new.columns:
            df_new[col] = pd.NA

    now = datetime.now(timezone.utc).isoformat()
    if "pulled_at_utc" not in df_new.columns:
        df_new["pulled_at_utc"] = now
    if "captured_at_utc" not in df_new.columns:
        df_new["captured_at_utc"] = df_new["pulled_at_utc"]

    existing = existing.copy()
    existing["event_id"] = existing["event_id"].astype(str)
    df_new["event_id"] = df_new["event_id"].astype(str)

    existing["capture_hour"] = pd.to_datetime(existing["captured_at_utc"], utc=True, errors="coerce").dt.floor("h").astype(str)
    df_new["capture_hour"] = pd.to_datetime(df_new["captured_at_utc"], utc=True, errors="coerce").dt.floor("h").astype(str)

    dedupe_existing = existing[["event_id", "capture_type", "capture_hour"]].apply(tuple, axis=1)
    dedupe_new = df_new[["event_id", "capture_type", "capture_hour"]].apply(tuple, axis=1)

    mask = ~dedupe_new.isin(dedupe_existing)
    deduped = df_new.loc[mask].copy()
    duplicates = int((~mask).sum())

    if deduped.empty:
        existing.drop(columns=["capture_hour"], inplace=True, errors="ignore")
        return existing, 0, duplicates

    merged = pd.concat([existing, deduped], ignore_index=True)
    merged.drop(columns=["capture_hour"], inplace=True, errors="ignore")
    merged = merged[[*SNAPSHOT_COLUMNS, *[c for c in merged.columns if c not in SNAPSHOT_COLUMNS]]]
    return merged, len(deduped), duplicates


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill market line snapshots")
    parser.add_argument("--days", type=int, default=100)
    parser.add_argument("--start_date", type=str, default=None)
    parser.add_argument("--end_date", type=str, default=None)
    parser.add_argument("--books", type=str, default="pinnacle,draftkings")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("data/market_lines_snapshots.csv"))
    args = parser.parse_args()

    books = {book.strip().lower() for book in args.books.split(",") if book.strip()}
    start, end = determine_date_range(args.days, args.start_date, args.end_date)
    days = iter_days(start, end)

    log.warning("Historical snapshots not available from this source")
    log.warning("Backfill captures only current lines available at run time for each processed date")

    existing = bootstrap_snapshots(args.output)
    summary = BackfillSummary()
    pending_rows: list[dict] = []

    for target_day in days:
        log.info("Processing %s", target_day.isoformat())
        games_found, rows = _capture_for_day(target_day, books)
        summary.days_processed += 1
        summary.games_found += games_found
        summary.snapshots_fetched += len(rows)
        pending_rows.extend(rows)

    merged, added, skipped = _merge_rows(existing, pending_rows)
    summary.new_rows_added = added
    summary.duplicates_skipped = skipped

    if args.dry_run:
        log.info("Dry-run enabled: no file writes performed")
    else:
        merged.to_csv(args.output, index=False)
        log.info("Wrote merged snapshots to %s", args.output)

    print("Backfill summary")
    print(f"days processed: {summary.days_processed}")
    print(f"games found: {summary.games_found}")
    print(f"snapshots fetched: {summary.snapshots_fetched}")
    print(f"new rows added: {summary.new_rows_added}")
    print(f"duplicates skipped: {summary.duplicates_skipped}")


if __name__ == "__main__":
    main()
