"""
ESPN CBB Pipeline — Main Builder
Orchestrates scoreboard fetch → summary fetch → CSV write.
Keep this file focused on coordination only; logic lives in other modules.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from espn_config import (
    BASE_DIR, CSV_DIR, JSON_DIR,
    OUT_GAMES, OUT_TEAM_LOGS, OUT_PLAYER_LOGS, OUT_METRICS, OUT_SOS,
    OUT_PLAYER_PROXY, OUT_TEAM_INJURY,
    OUT_WEIGHTED, OUT_PLAYER_METRICS,
    OUT_TOURNAMENT_METRICS, OUT_TOURNAMENT_SNAPSHOT,
    OUT_RANKINGS, OUT_RANKINGS_CONF,
    DAYS_BACK, TZ, CHECKPOINT_FILE,
    SOURCE, PARSE_VERSION,
    FETCH_SLEEP, DRY_RUN,
)
from espn_client import fetch_scoreboard, fetch_summary
from espn_parsers import parse_scoreboard_event, parse_summary, summary_to_team_rows
from espn_metrics import compute_all_metrics
from espn_sos import compute_sos_metrics
from espn_injury_proxy import compute_injury_proxy, compute_team_injury_impact
from espn_weighted_metrics import compute_weighted_metrics
from espn_player_metrics import compute_player_metrics
from espn_tournament import compute_tournament_metrics, build_pretournament_snapshot
from espn_rankings import run as run_rankings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Required downstream integrity columns
REQUIRED_TEAM_COLUMNS = [
    "event_id", "team_id", "conference", "wins", "losses",
    "fgm", "fga", "ftm", "fta", "tpm", "tpa", "orb", "drb", "reb", "tov", "ast",
    "opp_fgm", "opp_fga", "opp_ftm", "opp_fta",
    "opp_tpm", "opp_tpa", "opp_orb", "opp_drb", "opp_tov",
]

REQUIRED_PLAYER_COLUMNS = [
    "event_id", "team_id", "athlete_id",
    "fgm", "fga", "ftm", "fta", "tpm", "tpa", "orb", "drb", "reb", "tov", "ast",
]

TARGET_NULL_GUARD_COLUMNS = [
    "home_conference", "away_conference", "home_rank", "away_rank",
    "spread", "over_under", "home_ml", "away_ml", "odds_provider", "odds_details",
    "home_h1", "home_h2", "away_h1", "away_h2",
    "home_wins", "home_losses", "away_wins", "away_losses",
    "conf_wins", "conf_losses", "win_pct",
    "h1_pts", "h2_pts", "h1_pts_against", "h2_pts_against",
]

PLAYER_TARGET_NULL_GUARD_COLUMNS = [
    "fgm", "fga", "tpm", "tpa", "fta", "orb", "drb", "plus_minus",
    "efg_pct", "three_pct", "fg_pct", "ft_pct",
]

HALF_SCORE_FINAL_MIN_NON_NULL = float(os.getenv("HALF_SCORE_FINAL_MIN_NON_NULL", "0.80"))
STANDINGS_MIN_NON_NULL = float(os.getenv("STANDINGS_MIN_NON_NULL", "0.80"))


def _scoreboard_team_context(games_df: pd.DataFrame) -> pd.DataFrame:
    """Build team-level context rows (wins/losses/conference) from games.csv rows."""
    if games_df.empty:
        return pd.DataFrame(columns=["event_id", "team_id"])

    g = games_df.copy()
    g["event_id"] = g.get("game_id", "").astype(str)

    shared = {
        "spread": g.get("spread", None),
        "over_under": g.get("over_under", None),
        "home_ml": g.get("home_ml", None),
        "away_ml": g.get("away_ml", None),
        "odds_provider": g.get("odds_provider", None),
        "odds_details": g.get("odds_details", None),
        "home_h1": g.get("home_h1", None),
        "home_h2": g.get("home_h2", None),
        "away_h1": g.get("away_h1", None),
        "away_h2": g.get("away_h2", None),
        "home_conference": g.get("home_conference", None),
        "away_conference": g.get("away_conference", None),
        "home_rank": g.get("home_rank", None),
        "away_rank": g.get("away_rank", None),
    }

    home = pd.DataFrame({
        "event_id": g.get("game_id", ""),
        "team_id": g.get("home_team_id", ""),
        "home_away": "home",
        "conference": g.get("home_conference", ""),
        "wins": g.get("home_wins", None),
        "losses": g.get("home_losses", None),
        "home_wins": g.get("home_wins", None),
        "home_losses": g.get("home_losses", None),
        "away_wins": g.get("away_wins", None),
        "away_losses": g.get("away_losses", None),
        "conf_wins": g.get("home_conf_wins", None),
        "conf_losses": g.get("home_conf_losses", None),
        "rank": g.get("home_rank", None),
        "h1_pts": g.get("home_h1", None),
        "h2_pts": g.get("home_h2", None),
        "h1_pts_against": g.get("away_h1", None),
        "h2_pts_against": g.get("away_h2", None),
        **shared,
    })
    away = pd.DataFrame({
        "event_id": g.get("game_id", ""),
        "team_id": g.get("away_team_id", ""),
        "home_away": "away",
        "conference": g.get("away_conference", ""),
        "wins": g.get("away_wins", None),
        "losses": g.get("away_losses", None),
        "home_wins": g.get("home_wins", None),
        "home_losses": g.get("home_losses", None),
        "away_wins": g.get("away_wins", None),
        "away_losses": g.get("away_losses", None),
        "conf_wins": g.get("away_conf_wins", None),
        "conf_losses": g.get("away_conf_losses", None),
        "rank": g.get("away_rank", None),
        "h1_pts": g.get("away_h1", None),
        "h2_pts": g.get("away_h2", None),
        "h1_pts_against": g.get("home_h1", None),
        "h2_pts_against": g.get("home_h2", None),
        **shared,
    })

    out = pd.concat([home, away], ignore_index=True)
    out["event_id"] = out["event_id"].astype(str)
    out["team_id"] = out["team_id"].astype(str)
    return out.drop_duplicates(["event_id", "team_id"], keep="last")


def _log_stage_null_rates(stage: str, df: pd.DataFrame, columns: List[str]) -> None:
    if df.empty:
        log.info(f"{stage}: 0 rows")
        return
    present = [c for c in columns if c in df.columns]
    if not present:
        log.info(f"{stage}: {len(df)} rows | none of target columns present")
        return
    null_rates = (df[present].isna().mean() * 100).round(1).to_dict()
    key_cols = [k for k in ["event_id", "team_id", "opponent_id", "game_datetime_utc"] if k in df.columns]
    sample = df[key_cols].head(3).to_dict("records") if key_cols else []
    log.info(f"{stage}: rows={len(df)} null_rates(%)={null_rates} key_sample={sample}")


def _log_player_stage_diagnostics(stage: str, df: pd.DataFrame) -> None:
    target_cols = [
        "fgm", "fga", "tpm", "tpa", "fta", "orb", "drb", "plus_minus",
        "efg_pct", "three_pct", "fg_pct", "ft_pct",
    ]
    _log_stage_null_rates(stage, df, target_cols)
    dtypes = {k: str(df[k].dtype) for k in ["event_id", "team_id", "athlete_id"] if k in df.columns}
    key_sample = df[[k for k in ["event_id", "team_id", "athlete_id"] if k in df.columns]].head(3).to_dict("records")
    suffix_collisions = [c for c in df.columns if c.endswith("_x") or c.endswith("_y")]
    log.info(f"{stage}: key_dtypes={dtypes} key_sample={key_sample} suffix_collisions={suffix_collisions[:10]}")


def _enrich_team_rows_from_scoreboard(df_team: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing team metadata from scoreboard context to avoid blank wins/losses/conference."""
    if df_team.empty:
        return df_team

    df = df_team.copy()
    ctx = _scoreboard_team_context(games_df)
    if ctx.empty:
        return df

    df["event_id"] = df.get("event_id", "").astype(str)
    df["team_id"] = df.get("team_id", "").astype(str)

    df = df.merge(ctx, on=["event_id", "team_id"], how="left", suffixes=("", "_sb"))

    fill_cols = [
        "conference", "wins", "losses", "home_wins", "home_losses", "away_wins", "away_losses",
        "conf_wins", "conf_losses", "rank", "spread", "over_under", "home_ml", "away_ml",
        "odds_provider", "odds_details", "home_h1", "home_h2", "away_h1", "away_h2",
        "home_conference", "away_conference", "home_rank", "away_rank",
        "h1_pts", "h2_pts", "h1_pts_against", "h2_pts_against",
    ]

    for col in fill_cols:
        sb_col = f"{col}_sb"
        if sb_col in df.columns:
            if col in df.columns:
                df[col] = df[col].where(df[col].notna() & (df[col].astype(str) != ""), df[sb_col])
            else:
                df[col] = df[sb_col]

    if "win_pct" in df.columns and {"wins", "losses"}.issubset(df.columns):
        wins = pd.to_numeric(df["wins"], errors="coerce")
        losses = pd.to_numeric(df["losses"], errors="coerce")
        calc = wins / (wins + losses)
        df["win_pct"] = df["win_pct"].where(df["win_pct"].notna(), calc.round(3))

    df = df.drop(columns=[c for c in df.columns if c.endswith("_sb")])
    return df


def _validate_team_log_enrichment(df: pd.DataFrame) -> None:
    """Hard guardrail: fail fast when key enrichment groups are completely missing."""
    if df.empty:
        return

    def _non_null_rate(col: str, mask: Optional[pd.Series] = None) -> float:
        if col not in df.columns:
            return 0.0
        subset = df[mask] if mask is not None else df
        if subset.empty:
            return 1.0

        vals = subset[col]
        null_like_tokens = {"", "nan", "none", "null", "nat", "<na>"}
        token_null = vals.astype(str).str.strip().str.lower().isin(null_like_tokens)
        null_mask = vals.isna() | token_null
        return 1.0 - float(null_mask.mean())

    errors: List[str] = []

    conf_home_rate = _non_null_rate("home_conference")
    conf_away_rate = _non_null_rate("away_conference")
    if conf_home_rate <= 0.0 or conf_away_rate <= 0.0:
        errors.append("conference enrichment failed: home/away conference are 100% null")

    final_mask = df.get("completed", pd.Series(False, index=df.index)).astype(str).str.lower().isin(["true", "1", "yes"])
    for half_col in ["home_h1", "home_h2", "away_h1", "away_h2"]:
        if _non_null_rate(half_col, mask=final_mask) < HALF_SCORE_FINAL_MIN_NON_NULL:
            errors.append(
                f"half-score enrichment below threshold for finals: {half_col} < {HALF_SCORE_FINAL_MIN_NON_NULL:.0%}"
            )

    for standings_col in ["home_wins", "home_losses"]:
        if _non_null_rate(standings_col) < STANDINGS_MIN_NON_NULL:
            errors.append(
                f"standings enrichment below threshold: {standings_col} < {STANDINGS_MIN_NON_NULL:.0%}"
            )

    odds_fields = ["spread", "over_under", "home_ml", "away_ml", "odds_provider", "odds_details"]
    odds_non_null = max((_non_null_rate(c) for c in odds_fields), default=0.0)
    if odds_non_null <= 0.0:
        errors.append("odds enrichment failed: all odds fields are 100% null")

    if errors:
        raise ValueError(" | ".join(errors))


def _assert_required_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


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
    persist: bool = True,
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

    if persist:
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
        _log_stage_null_rates("team_rows_raw_parse", df_team, TARGET_NULL_GUARD_COLUMNS)
        df_team = _enrich_team_rows_from_scoreboard(df_team, games_df)
        _log_stage_null_rates("team_rows_after_scoreboard_merge", df_team, TARGET_NULL_GUARD_COLUMNS)
        _assert_required_columns(df_team, REQUIRED_TEAM_COLUMNS, "team_rows")
        df_all  = _append_dedupe_write(
            OUT_TEAM_LOGS,
            df_team,
            unique_keys=["event_id", "team_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id"],
            persist=False,
        )
        _log_stage_null_rates("team_rows_before_validation", df_all, TARGET_NULL_GUARD_COLUMNS)
        _validate_team_log_enrichment(df_all)

        _append_dedupe_write(
            OUT_TEAM_LOGS,
            df_team,
            unique_keys=["event_id", "team_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id"],
            persist=True,
        )

        if OUT_TEAM_LOGS.exists() and OUT_TEAM_LOGS.stat().st_size > 0:
            _log_stage_null_rates(
                "team_rows_after_reload",
                pd.read_csv(OUT_TEAM_LOGS, dtype=str, low_memory=False),
                TARGET_NULL_GUARD_COLUMNS,
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

        # ── Opponent-weighted rolling metrics ──
        # Runs on SOS output so opponent quality weights are available.
        df_weighted = compute_weighted_metrics(df_sos_out)
        df_weighted_out = _append_dedupe_write(
            OUT_WEIGHTED,
            df_weighted,
            unique_keys=["event_id", "team_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id"],
        )
        log.info(f"team_game_weighted.csv: {len(df_weighted_out)} total rows")

        # ── Tournament composite metrics ──
        # Runs on weighted output so adj_ortg, adj_drtg, opp_avg_net_rtg,
        # efg_vs_opp, rolling windows are all available.
        # player_df is optional — uses player-level scoring distribution for
        # star reliance when available, falls back to box-score proxies.
        _player_df_for_tournament = (
            pd.read_csv(OUT_PLAYER_METRICS)
            if OUT_PLAYER_METRICS.exists() and OUT_PLAYER_METRICS.stat().st_size > 0
            else pd.DataFrame()
        )
        df_tournament = compute_tournament_metrics(
            df_weighted_out,
            player_df=_player_df_for_tournament if not _player_df_for_tournament.empty else None,
        )
        df_tournament_out = _append_dedupe_write(
            OUT_TOURNAMENT_METRICS,
            df_tournament,
            unique_keys=["event_id", "team_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id"],
        )
        log.info(f"team_tournament_metrics.csv: {len(df_tournament_out)} total rows")

        # ── Pre-tournament snapshot (one row per team = most recent game) ──
        # Primary input for matchup projections (game totals, UWS, rankings).
        # Rebuilt every run so it always reflects the latest completed game.
        df_snapshot = build_pretournament_snapshot(df_tournament_out)
        if not DRY_RUN:
            OUT_TOURNAMENT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            df_snapshot.to_csv(OUT_TOURNAMENT_SNAPSHOT, index=False)
        log.info(f"team_pretournament_snapshot.csv: {len(df_snapshot)} teams")

        # ── CAGE Rankings ──
        # Requires snapshot to be on disk — runs immediately after writing it.
        # Writes cbb_rankings.csv + cbb_rankings_by_conference.csv.
        try:
            from pathlib import Path as _Path
            run_rankings(output_dir=_Path("data"))
            log.info("cbb_rankings.csv updated")
        except Exception as exc:
            log.warning(f"Rankings generation failed (non-fatal): {exc}")

    else:
        log.warning("No team rows to write")

    if player_rows:
        df_players = pd.DataFrame(player_rows)
        _log_player_stage_diagnostics("player_rows:parsed", df_players)
        _assert_required_columns(df_players, REQUIRED_PLAYER_COLUMNS, "player_rows")
        df_all_p   = _append_dedupe_write(
            OUT_PLAYER_LOGS,
            df_players,
            unique_keys=["event_id", "team_id", "athlete_id"],
            sort_cols=["game_datetime_utc", "event_id", "team_id", "athlete_id"],
        )
        _log_player_stage_diagnostics("player_rows:post_dedupe", df_all_p)
        log.info(f"player_game_logs.csv: {len(df_all_p)} total rows")

        # ── Player rolling metrics ──
        team_logs_for_poss = df_metrics_out if team_rows else pd.DataFrame()
        df_player_metrics = compute_player_metrics(df_all_p, team_logs_for_poss)
        _log_player_stage_diagnostics("player_metrics:pre_write", df_player_metrics)
        df_player_metrics_out = _append_dedupe_write(
            OUT_PLAYER_METRICS,
            df_player_metrics,
            unique_keys=["event_id", "athlete_id"],
            sort_cols=["game_datetime_utc", "event_id", "athlete_id"],
        )
        _log_player_stage_diagnostics("player_metrics:post_write", df_player_metrics_out)

        if OUT_PLAYER_METRICS.exists() and OUT_PLAYER_METRICS.stat().st_size > 0:
            reloaded = pd.read_csv(OUT_PLAYER_METRICS)
            _log_player_stage_diagnostics("player_metrics:reload", reloaded)
        log.info(f"player_game_metrics.csv: {len(df_player_metrics_out)} total rows")

        # ── Injury proxy ──
        df_proxy = compute_injury_proxy(df_all_p, df_all if team_rows else pd.DataFrame())
        if not df_proxy.empty:
            df_proxy_out = _append_dedupe_write(
                OUT_PLAYER_PROXY,
                df_proxy,
                unique_keys=["event_id", "athlete_id"],
                sort_cols=["game_datetime_utc", "event_id", "athlete_id"],
            )
            log.info(f"player_injury_proxy.csv: {len(df_proxy_out)} total rows")

            # Team-level injury impact
            df_team_logs = pd.read_csv(OUT_TEAM_LOGS) if OUT_TEAM_LOGS.exists() else pd.DataFrame()
            df_impact = compute_team_injury_impact(df_proxy_out, df_team_logs)
            if not df_impact.empty:
                df_impact_out = _append_dedupe_write(
                    OUT_TEAM_INJURY,
                    df_impact,
                    unique_keys=["event_id", "team_id"],
                    sort_cols=["game_datetime_utc", "event_id", "team_id"],
                )
                log.info(f"team_injury_impact.csv: {len(df_impact_out)} total rows")
    else:
        log.warning("No player rows to write")

        # Keep required player artifacts present even when no boxscore player
        # data is available in this run window (e.g., pre-tip schedule slate).
        # This keeps downstream validators/pipelines deterministic.
        if not OUT_PLAYER_LOGS.exists():
            empty_player_logs = pd.DataFrame(columns=[
                "event_id", "game_datetime_utc", "game_datetime_pst",
                "team_id", "team", "home_away", "athlete_id", "player",
                "jersey", "position", "starter", "did_not_play",
                "min", "pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta",
                "orb", "drb", "reb", "ast", "stl", "blk", "tov", "pf", "plus_minus",
                "FGA", "FGM", "FTA", "FTM", "TPA", "TPM", "ORB", "DRB", "RB", "TO", "AST",
            ])
            OUT_PLAYER_LOGS.parent.mkdir(parents=True, exist_ok=True)
            empty_player_logs.to_csv(OUT_PLAYER_LOGS, index=False)
            log.info("player_game_logs.csv: 0 total rows")

        if not OUT_PLAYER_METRICS.exists():
            empty_player_metrics = pd.DataFrame(columns=[
                "event_id", "game_datetime_utc", "game_datetime_pst",
                "team_id", "team", "home_away", "athlete_id", "player",
                "jersey", "position", "starter", "did_not_play",
                "min", "pts", "fgm", "fga", "tpm", "tpa", "ftm", "fta",
                "orb", "drb", "reb", "ast", "stl", "blk", "tov", "pf", "plus_minus",
                "FGA", "FGM", "FTA", "FTM", "TPA", "TPM", "ORB", "DRB", "RB", "TO", "AST",
            ])
            OUT_PLAYER_METRICS.parent.mkdir(parents=True, exist_ok=True)
            empty_player_metrics.to_csv(OUT_PLAYER_METRICS, index=False)
            log.info("player_game_metrics.csv: 0 total rows")

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
