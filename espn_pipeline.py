"""
ESPN CBB Pipeline — Main Builder
Orchestrates scoreboard fetch → summary fetch → CSV write.
Keep this file focused on coordination only; logic lives in other modules.
"""

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from espn_config import (
    BASE_DIR, CSV_DIR, JSON_DIR,
    OUT_GAMES, OUT_TEAM_LOGS, OUT_PLAYER_LOGS, OUT_METRICS, OUT_SOS,
    DAYS_BACK, TZ, CHECKPOINT_FILE,
    SOURCE, PARSE_VERSION,
    FETCH_SLEEP, DRY_RUN,
)
from espn_client import fetch_scoreboard, fetch_summary
from espn_parsers import parse_scoreboard_event, parse_summary, summary_to_team_rows
from espn_metrics import compute_all_metrics
from espn_sos import compute_sos_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _load_checkpoint() -> Dict[str, Any]:
    p = Path(CHECKPOINT_FILE)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception as exc:
            log.warning(f"Could not load checkpoint ({exc}), starting fresh")
    return {}


def _save_checkpoint(data: Dict[str, Any]) -> None:
    try:
        Path(CHECKPOINT_FILE).write_text(json.dumps(data, indent=2))
    except Exception as exc:
        log.warning(f"Could not save checkpoint: {exc}")


def _clear_checkpoint() -> None:
    p = Path(CHECKPOINT_FILE)
    if p.exists():
        p.unlink()


# ── CSV helpers ───────────────────────────────────────────────────────────────

def _append_dedupe_write(
    path: Path,
    new_df: pd.DataFrame,
    unique_keys: List[str],
    sort_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Append new_df to existing CSV, deduplicate on unique_keys keeping the
    newest row (by pulled_at_utc then completeness), then write back.
    Always prefers newer/more-complete rows over stale ones.
    """
    if path.exists() and path.stat().st_size > 0:
        try:
            existing = pd.read_csv(path, dtype=str)
        except Exception as exc:
            log.warning(f"Could not read {path} ({exc}), overwriting")
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    combined = pd.concat([existing, new_df.astype(str)], ignore_index=True)

    if unique_keys:
        # Score rows: prefer completed=True, then newest pulled_at_utc
        if "completed" in combined.columns:
            combined["_completed_int"] = (
                combined["completed"].str.lower().isin(["true", "1", "yes"])
            ).astype(int)
        else:
            combined["_completed_int"] = 0

        if "pulled_at_utc" in combined.columns:
            combined["_pulled_ts"] = pd.to_datetime(
                combined["pulled_at_utc"], utc=True, errors="coerce"
            ).astype("int64", errors="ignore").fillna(0)
        else:
            combined["_pulled_ts"] = 0

        combined = (
            combined
            .sort_values(["_completed_int", "_pulled_ts"], ascending=[False, False])
            .drop_duplicates(subset=unique_keys, keep="first")
            .drop(columns=["_completed_int", "_pulled_ts"], errors="ignore")
        )

    if sort_cols:
        present = [c for c in sort_cols if c in combined.columns]
        if present:
            combined = combined.sort_values(present)

    if not DRY_RUN:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via temp file
        tmp = path.with_suffix(".tmp")
        combined.to_csv(tmp, index=False)
        tmp.replace(path)
    else:
        log.info(f"[DRY RUN] Would write {len(combined)} rows → {path}")

    return combined


def _save_raw_json(subdir: str, name: str, data: Any) -> None:
    """Save raw API response JSON for debugging/audit."""
    try:
        dest = JSON_DIR / subdir
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{name}.json").write_text(json.dumps(data, indent=2))
    except Exception as exc:
        log.warning(f"Could not save raw JSON ({exc})")


# ── Scoreboard pass ───────────────────────────────────────────────────────────

def build_games(days_back: int = DAYS_BACK) -> pd.DataFrame:
    """
    Fetch scoreboard for the past `days_back` days + today + tomorrow.
    Returns a DataFrame of all parsed game rows and writes games.csv.
    """
    now = datetime.now(TZ)
    dates = set()
    for i in range(days_back):
        dates.add((now - timedelta(days=i)).strftime("%Y%m%d"))
    dates.add(now.strftime("%Y%m%d"))
    dates.add((now + timedelta(days=1)).strftime("%Y%m%d"))  # catch late-night UTC games

    all_rows: List[Dict] = []
    for d in sorted(dates, reverse=True):
        try:
            raw = fetch_scoreboard(d)
            _save_raw_json("scoreboard", d, raw)
        except Exception as exc:
            log.error(f"Scoreboard fetch failed for {d}: {exc}")
            continue

        events = raw.get("events") or []
        day_rows = []
        for e in events:
            parsed = parse_scoreboard_event(e)
            if parsed:
                parsed["date"]          = d
                parsed["pulled_at_utc"] = datetime.now(TZ).isoformat()
                parsed["source"]        = SOURCE
                day_rows.append(parsed)
            else:
                log.warning(f"Could not parse scoreboard event {e.get('id','?')} on {d}")

        finals = sum(1 for r in day_rows if str(r.get("completed", "")).lower() == "true")
        log.info(f"Scoreboard {d}: {len(day_rows)} games, {finals} final")
        all_rows.extend(day_rows)

    if not all_rows:
        log.warning("No scoreboard games returned")
        return pd.DataFrame()

    df_new = pd.DataFrame(all_rows)
    df_all = _append_dedupe_write(
        OUT_GAMES,
        df_new,
        unique_keys=["game_id"],
        sort_cols=["date", "game_id"],
    )
    log.info(f"games.csv: {len(df_all)} total rows")
    return df_all


# ── Summary pass ──────────────────────────────────────────────────────────────

def build_team_and_player_logs(games_df: pd.DataFrame, days_back: int = DAYS_BACK) -> None:
    """
    For each completed game in the run window, fetch the ESPN summary and
    write team + player rows to their respective CSVs.
    """
    now = datetime.now(TZ)
    window = {(now - timedelta(days=i)).strftime("%Y%m%d") for i in range(days_back)}
    window.add(now.strftime("%Y%m%d"))

    # Only process games within our date window
    run_games = games_df[games_df["date"].astype(str).isin(window)].copy()
    game_ids  = run_games["game_id"].astype(str).unique().tolist()
    log.info(f"Games in run window: {len(game_ids)}")

    # Load checkpoint so we can resume interrupted runs
    checkpoint   = _load_checkpoint()
    processed    = set(map(str, checkpoint.get("processed_ids", [])))

    team_rows:   List[Dict] = []
    player_rows: List[Dict] = []
    failed:      List[str]  = []

    for i, gid in enumerate(game_ids, 1):
        if gid in processed:
            log.debug(f"Skipping {gid} (checkpoint)")
            continue

        try:
            raw    = fetch_summary(gid)
            _save_raw_json("summaries", gid, raw)
            parsed = parse_summary(raw, gid)

            if parsed is None:
                log.warning(f"parse_summary returned None for {gid}")
                failed.append(gid)
                continue

            now_iso = datetime.now(TZ).isoformat()
            hrow, arow = summary_to_team_rows(parsed)
            for row in (hrow, arow):
                row["pulled_at_utc"] = now_iso
                row["source"]        = SOURCE
                row["parse_version"] = PARSE_VERSION
            team_rows.extend([hrow, arow])

            for p in parsed.get("players", []):
                p["pulled_at_utc"] = now_iso
                p["source"]        = SOURCE
                p["parse_version"] = PARSE_VERSION
                player_rows.append(p)

            processed.add(gid)

        except Exception as exc:
            log.error(f"Failed to process game {gid}: {exc}")
            failed.append(gid)

        # Checkpoint every 25 games
        if i % 25 == 0:
            _save_checkpoint({"processed_ids": list(processed)})
            log.info(f"Progress: {i}/{len(game_ids)} games processed")

        time.sleep(FETCH_SLEEP)

    # ── Retry failed games once ──
    if failed:
        log.info(f"Retrying {len(failed)} failed games...")
        time.sleep(2.0)
        still_failed = []
        for gid in failed:
            if gid in processed:
                continue
            try:
                raw    = fetch_summary(gid)
                parsed = parse_summary(raw, gid)
                if parsed:
                    now_iso = datetime.now(TZ).isoformat()
                    hrow, arow = summary_to_team_rows(parsed)
                    for row in (hrow, arow):
                        row["pulled_at_utc"] = now_iso
                        row["source"]        = SOURCE
                        row["parse_version"] = PARSE_VERSION
                    team_rows.extend([hrow, arow])
                    for p in parsed.get("players", []):
                        p["pulled_at_utc"] = now_iso
                        p["source"]        = SOURCE
                        p["parse_version"] = PARSE_VERSION
                        player_rows.append(p)
                    processed.add(gid)
                    log.info(f"  Retry OK: {gid}")
                else:
                    still_failed.append(gid)
            except Exception as exc:
                log.error(f"  Retry failed {gid}: {exc}")
                still_failed.append(gid)
            time.sleep(0.25)
        failed = still_failed

    # ── Write CSVs ──
    if team_rows:
        df_team = pd.DataFrame(team_rows)
        df_all  = _append_dedupe_write(
            OUT_TEAM_LOGS,
            df_team,
            unique_keys=["event_id", "team_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id"],
        )
        log.info(f"team_game_logs.csv: {len(df_all)} total rows")

        # ── Compute advanced metrics + rolling windows on full history ──
        # Always runs on the full df_all (not just new rows) so rolling
        # windows are accurate across the entire season, not just this run.
        df_metrics = compute_all_metrics(df_all)
        df_metrics_out = _append_dedupe_write(
            OUT_METRICS,
            df_metrics,
            unique_keys=["event_id", "team_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id"],
        )
        log.info(f"team_game_metrics.csv: {len(df_metrics_out)} total rows")

        # ── Compute SOS + opponent-context metrics ──
        # Requires opponent_id column from summary_to_team_rows.
        df_sos = compute_sos_metrics(df_metrics_out)
        df_sos_out = _append_dedupe_write(
            OUT_SOS,
            df_sos,
            unique_keys=["event_id", "team_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id"],
        )
        log.info(f"team_game_sos.csv: {len(df_sos_out)} total rows")
    else:
        log.warning("No team rows to write")

    if player_rows:
        df_players = pd.DataFrame(player_rows)
        df_all_p   = _append_dedupe_write(
            OUT_PLAYER_LOGS,
            df_players,
            unique_keys=["event_id", "team_id", "athlete_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id", "athlete_id"],
        )
        log.info(f"player_game_logs.csv: {len(df_all_p)} total rows")
    else:
        log.warning("No player rows to write")

    # ── Reconciliation report ──
    log.info("=== Reconciliation ===")
    log.info(f"Total games in window:   {len(game_ids)}")
    log.info(f"Successfully processed:  {len(processed)}")
    log.info(f"Failed after retry:      {len(failed)}")
    if failed:
        log.warning(f"Still failing: {failed[:20]}")

    completion = len(processed) / max(1, len(game_ids))
    log.info(f"Completion rate: {completion*100:.1f}%")

    _clear_checkpoint()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(days_back: int = DAYS_BACK) -> None:
    log.info(f"=== ESPN CBB Pipeline | DAYS_BACK={days_back} | PARSE_VERSION={PARSE_VERSION} ===")
    games_df = build_games(days_back=days_back)
    if games_df.empty:
        log.error("No games from scoreboard — aborting")
        return
    build_team_and_player_logs(games_df, days_back=days_back)
    log.info("=== Run complete ===")


if __name__ == "__main__":
    run()
