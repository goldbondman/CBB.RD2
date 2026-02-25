"""Build predictions_with_context.csv by adding latest market-line context."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from espn_config import (
    DATA_DIR,
    OUT_PREDICTIONS_COMBINED,
    OUT_PREDICTIONS_CONTEXT,
    OUT_GAMES,
    conference_id_to_name,
)
from models.alpha_evaluator import evaluate_alpha
from pipeline_csv_utils import (
    add_conference_name,
    normalize_column_names,
    normalize_numeric_dtypes,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

RECORDS_CACHE_PATH = DATA_DIR / "team_records.csv"


def _fetch_team_record(team_id: str) -> dict:
    """Fetch current team W-L record from ESPN team endpoint."""
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/"
        f"basketball/mens-college-basketball/teams/{team_id}"
    )
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        data = resp.json()
        team = data.get("team", {})
        record = (team.get("record") or {}).get("items", [{}])[0]
        stats = {s.get("name"): s.get("value") for s in record.get("stats", [])}
        return {
            "team_id": str(team_id),
            "wins": int(float(stats.get("wins", 0) or 0)),
            "losses": int(float(stats.get("losses", 0) or 0)),
        }
    except Exception as exc:  # noqa: BLE001
        log.debug("Team record fetch failed for %s: %s", team_id, exc)
        return {"team_id": str(team_id), "wins": 0, "losses": 0}




def _build_records_from_local_history() -> dict:
    """Fallback: derive team records from historical prediction snapshots."""
    records: dict[str, dict] = {}
    for csv_path in sorted(DATA_DIR.glob("predictions_*.csv")):
        try:
            hist = pd.read_csv(csv_path, dtype={"home_team_id": str, "away_team_id": str})
        except Exception:
            continue
        required = {"home_team_id", "away_team_id", "home_wins", "home_losses", "away_wins", "away_losses"}
        if not required.issubset(hist.columns):
            continue
        for _, row in hist.iterrows():
            htid = str(row.get("home_team_id", "")).strip()
            atid = str(row.get("away_team_id", "")).strip()
            if htid and htid.lower() != "nan":
                hw = int(float(row.get("home_wins", 0) or 0))
                hl = int(float(row.get("home_losses", 0) or 0))
                if hw > 0 or hl > 0:
                    records[htid] = {"team_id": htid, "wins": hw, "losses": hl}
            if atid and atid.lower() != "nan":
                aw = int(float(row.get("away_wins", 0) or 0))
                al = int(float(row.get("away_losses", 0) or 0))
                if aw > 0 or al > 0:
                    records[atid] = {"team_id": atid, "wins": aw, "losses": al}
    return records

def _enrich_win_loss_records(df: pd.DataFrame) -> pd.DataFrame:
    if "home_team_id" not in df.columns or "away_team_id" not in df.columns:
        log.warning("Skipping team record enrichment: missing home_team_id/away_team_id")
        return df

    if {"home_wins", "away_wins"}.issubset(df.columns):
        # A value of 0 or 1 is ambiguous — could be a binary game win flag.
        # Only skip enrichment if values clearly represent season totals
        # (i.e. at least one team has more than 1 win).
        max_wins = max(
            df["home_wins"].fillna(0).max(),
            df["away_wins"].fillna(0).max(),
        )
        if max_wins > 1:
            log.debug(
                "Skipping team record enrichment: "
                "home_wins max=%s (appears to be season totals already)", max_wins
            )
            return df
        elif max_wins == 1:
            log.info(
                "home_wins max=1 — treating as binary game flag, not season total. "
                "Overwriting with season W-L records."
            )
            # Drop the binary columns so the enrichment writes fresh season values
            df = df.drop(
                columns=["home_wins", "away_wins", "home_losses", "away_losses"],
                errors="ignore",
            )

    team_ids = sorted(
        {
            str(t).strip()
            for t in (df["home_team_id"].astype(str).tolist() + df["away_team_id"].astype(str).tolist())
            if str(t).strip() and str(t).strip().lower() != "nan"
        }
    )
    if not team_ids:
        log.warning("Skipping team record enrichment: no team IDs in predictions")
        return df

    existing_records = {}
    if RECORDS_CACHE_PATH.exists():
        try:
            cached = pd.read_csv(RECORDS_CACHE_PATH, dtype={"team_id": str})
            existing_records = {
                str(r["team_id"]): {
                    "team_id": str(r["team_id"]),
                    "wins": int(r.get("wins", 0) or 0),
                    "losses": int(r.get("losses", 0) or 0),
                }
                for _, r in cached.iterrows()
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read team records cache (%s): %s", RECORDS_CACHE_PATH, exc)

    records = dict(existing_records)

    # Local team-game derived records are authoritative for this repo run.
    local_records = _build_records_from_local_team_data()
    for tid, rec in local_records.items():
        records[tid] = rec

    history_records = _build_records_from_local_history()
    for tid, rec in history_records.items():
        if tid not in records or (
            int(records[tid].get("wins", 0)) == 0
            and int(records[tid].get("losses", 0)) == 0
        ):
            records[tid] = rec
    missing = [t for t in team_ids if t not in records]
    fetched = 0
    for tid in missing:
        records[tid] = _fetch_team_record(tid)
        fetched += 1
        time.sleep(0.2)

    if fetched or records:
        cache_df = pd.DataFrame(records.values()).sort_values("team_id")
        cache_df.to_csv(RECORDS_CACHE_PATH, index=False)

    df["home_wins"] = df["home_team_id"].astype(str).map(lambda t: records.get(str(t).strip(), {}).get("wins", 0))
    df["home_losses"] = df["home_team_id"].astype(str).map(lambda t: records.get(str(t).strip(), {}).get("losses", 0))
    df["away_wins"] = df["away_team_id"].astype(str).map(lambda t: records.get(str(t).strip(), {}).get("wins", 0))
    df["away_losses"] = df["away_team_id"].astype(str).map(lambda t: records.get(str(t).strip(), {}).get("losses", 0))

    nonzero = int(df["home_wins"].fillna(0).gt(0).sum())
    log.info("Team records enriched: %s/%s rows have non-zero home wins", nonzero, len(df))
    return df


def _normalize_conference_names(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["home_conference", "away_conference", "conference"]:
        if col in df.columns:
            df[col] = df[col].apply(conference_id_to_name)
    return df



def _build_market_lines_fallback(pred_df: pd.DataFrame, market_path: Path) -> pd.DataFrame:
    """
    Build minimal market_lines.csv from available local artifacts.

    Priority:
      1) predictions input itself (guaranteed event_id join key)
      2) games.csv snapshots (root or data/csv)
    """
    now_utc = pd.Timestamp.now("UTC").isoformat()

    # Source 1: predictions rows (guarantees event_id match)
    pred_market = pd.DataFrame({
        "event_id": pred_df.get("event_id", pd.Series(dtype=str)).astype(str).str.strip(),
        "home_spread_open": pd.to_numeric(pred_df.get("spread_line"), errors="coerce"),
        "home_spread_current": pd.to_numeric(pred_df.get("spread_line"), errors="coerce"),
        "spread": pd.to_numeric(pred_df.get("spread_line"), errors="coerce"),
        "over_under": pd.to_numeric(pred_df.get("total_line"), errors="coerce"),
        "home_ml": pd.to_numeric(pred_df.get("home_ml"), errors="coerce"),
        "away_ml": pd.to_numeric(pred_df.get("away_ml"), errors="coerce"),
        "line_movement": 0.0,
        "pinnacle_spread": pd.to_numeric(pred_df.get("spread_line"), errors="coerce"),
        "draftkings_spread": pd.to_numeric(pred_df.get("spread_line"), errors="coerce"),
        "home_tickets_pct": pd.NA,
        "home_money_pct": pd.NA,
        "steam_flag": False,
        "rlm_flag": False,
        "rlm_sharp_side": pd.NA,
        "book_disagreement_flag": False,
        "book_sharp_side": pd.NA,
        "book_spread_diff": 0.0,
        "line_freeze_flag": False,
        "captured_at_utc": now_utc,
        "capture_type": "fallback_predictions",
    })
    pred_market = pred_market[pred_market["event_id"].str.len() > 0]

    # Source 2: games.csv if available
    source_frames = [pred_market]
    sources = [OUT_GAMES, DATA_DIR / "csv" / OUT_GAMES.name]
    games_path = next((gp for gp in sources if gp.exists()), None)
    if games_path is not None:
        games = pd.read_csv(games_path, dtype={"game_id": str}, low_memory=False)
        if not games.empty:
            game_market = pd.DataFrame({
                "event_id": games.get("game_id", pd.Series(dtype=str)).astype(str).str.strip(),
                "home_spread_open": pd.to_numeric(games.get("spread"), errors="coerce"),
                "home_spread_current": pd.to_numeric(games.get("spread"), errors="coerce"),
                "spread": pd.to_numeric(games.get("spread"), errors="coerce"),
                "over_under": pd.to_numeric(games.get("over_under"), errors="coerce"),
                "home_ml": pd.to_numeric(games.get("home_ml"), errors="coerce"),
                "away_ml": pd.to_numeric(games.get("away_ml"), errors="coerce"),
                "line_movement": 0.0,
                "pinnacle_spread": pd.to_numeric(games.get("spread"), errors="coerce"),
                "draftkings_spread": pd.to_numeric(games.get("spread"), errors="coerce"),
                "home_tickets_pct": pd.NA,
                "home_money_pct": pd.NA,
                "steam_flag": False,
                "rlm_flag": False,
                "rlm_sharp_side": pd.NA,
                "book_disagreement_flag": False,
                "book_sharp_side": pd.NA,
                "book_spread_diff": 0.0,
                "line_freeze_flag": False,
                "captured_at_utc": now_utc,
                "capture_type": "fallback_games",
            })
            source_frames.append(game_market)

    fallback = pd.concat(source_frames, ignore_index=True)
    fallback = fallback.drop_duplicates(subset=["event_id"], keep="first")
    fallback = fallback[fallback["event_id"].isin(pred_df["event_id"].astype(str).str.strip())]

    if fallback.empty:
        log.warning("Fallback market builder could not create matching rows")
        return pd.DataFrame()

    market_path.parent.mkdir(parents=True, exist_ok=True)
    fallback.to_csv(market_path, index=False)
    log.warning("Built fallback market_lines.csv: %d rows", len(fallback))
    return fallback


def _build_records_from_local_team_data() -> dict:
    """Derive season W-L records from local team-game datasets."""
    records: dict[str, dict] = {}
    candidate_paths = [
        DATA_DIR / "team_game_weighted.csv",
        DATA_DIR / "team_game_metrics.csv",
        DATA_DIR / "csv" / "team_game_weighted.csv",
        DATA_DIR / "csv" / "team_game_metrics.csv",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, dtype={"team_id": str}, low_memory=False)
        except Exception:
            continue
        if "team_id" not in df.columns:
            continue

        if "win" in df.columns:
            win_flag = pd.to_numeric(df["win"], errors="coerce").fillna(0)
            grouped = (
                pd.DataFrame({"team_id": df["team_id"].astype(str), "win": win_flag})
                .groupby("team_id", as_index=False)
                .agg(wins=("win", "sum"), games=("win", "count"))
            )
            grouped["losses"] = grouped["games"] - grouped["wins"]
            for _, r in grouped.iterrows():
                tid = str(r["team_id"]).strip()
                if not tid:
                    continue
                records[tid] = {
                    "team_id": tid,
                    "wins": int(max(0, r["wins"])),
                    "losses": int(max(0, r["losses"])),
                }
            if records:
                return records

    return records



def _enrich_ats_records(df: pd.DataFrame) -> pd.DataFrame:
    """Fill ATS records from local team_season_summary snapshot when available."""
    summary_path = DATA_DIR / "team_season_summary.csv"
    if not summary_path.exists():
        return df
    try:
        summary = pd.read_csv(summary_path, dtype={"team_id": str}, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed reading %s: %s", summary_path, exc)
        return df

    required_cols = {"team_id", "ats_wins", "ats_losses"}
    if not required_cols.issubset(summary.columns):
        return df

    map_df = summary[["team_id", "ats_wins", "ats_losses"]].copy()
    map_df["team_id"] = map_df["team_id"].astype(str).str.strip()

    if "home_team_id" in df.columns:
        df["home_team_id"] = df["home_team_id"].astype(str).str.strip()
        home_map = map_df.rename(columns={
            "team_id": "home_team_id",
            "ats_wins": "home_ats_wins",
            "ats_losses": "home_ats_losses",
        })
        df = df.merge(home_map, on="home_team_id", how="left")

    if "away_team_id" in df.columns:
        df["away_team_id"] = df["away_team_id"].astype(str).str.strip()
        away_map = map_df.rename(columns={
            "team_id": "away_team_id",
            "ats_wins": "away_ats_wins",
            "ats_losses": "away_ats_losses",
        })
        df = df.merge(away_map, on="away_team_id", how="left")

    return df


def _enrich_win_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-implied win probabilities only when explicit fields are missing."""
    if "home_win_prob" not in df.columns:
        df["home_win_prob"] = pd.NA
    if "away_win_prob" not in df.columns:
        df["away_win_prob"] = pd.NA

    if "model_confidence" in df.columns and "pred_spread" in df.columns:
        conf = pd.to_numeric(df["model_confidence"], errors="coerce")
        spread = pd.to_numeric(df["pred_spread"], errors="coerce")
        home_implied = conf.where(spread < 0, 1.0 - conf)
        away_implied = 1.0 - home_implied

        df["home_win_prob"] = pd.to_numeric(df["home_win_prob"], errors="coerce").fillna(home_implied)
        df["away_win_prob"] = pd.to_numeric(df["away_win_prob"], errors="coerce").fillna(away_implied)

    if "win_prob_source" not in df.columns:
        df["win_prob_source"] = pd.NA
    null_source = df["win_prob_source"].isna() & df["home_win_prob"].notna() & df["away_win_prob"].notna()
    df.loc[null_source, "win_prob_source"] = "model_implied"
    return df

def build_predictions_with_context(
    predictions_path: Path = OUT_PREDICTIONS_COMBINED,
    out_path: Path = OUT_PREDICTIONS_CONTEXT,
) -> pd.DataFrame:

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"Missing predictions source file: {predictions_path}"
        )

    df = pd.read_csv(predictions_path, dtype={"event_id": str})
    df = normalize_numeric_dtypes(df)

    # Save model columns BEFORE normalize_column_names()
    _saved_cols = {}
    for c in ["pred_spread", "ens_ens_spread", "predicted_spread", "pred_total", "pred_home_score", "pred_away_score"]:
        if c in df.columns:
            _saved_cols[c] = df[c].copy()

    df = normalize_column_names(df)

    # Restore after normalization
    for c, val in _saved_cols.items():
        if c not in df.columns or df[c].isna().mean() > 0.5:
            df[c] = val.values
            log.debug("%s restored after normalize_column_names", c)

    log.info(
        "[DIAG] build_predictions_with_context | initial | pred_spread non-null: %d/%d (%.1f%% null)",
        int(df["pred_spread"].notna().sum()) if "pred_spread" in df.columns else 0,
        len(df),
        (df["pred_spread"].isna().mean() * 100) if "pred_spread" in df.columns else 100.0
    )

    if "event_id" not in df.columns:
        if "game_id" in df.columns:
            df["event_id"] = df["game_id"].astype(str).str.strip()
        else:
            raise ValueError(
                "predictions file missing required event_id/game_id column"
            )

    expected_market_cols = [
        "home_spread_open", "home_spread_current", "spread", "line_movement",
        "over_under", "home_ml", "away_ml", "home_win_prob", "away_win_prob",
        "pinnacle_spread", "draftkings_spread",
        "home_tickets_pct", "home_money_pct",
        "steam_flag", "rlm_flag", "rlm_sharp_side",
        "book_disagreement_flag", "book_sharp_side", "book_spread_diff",
        "line_freeze_flag",
    ]
    market_path = DATA_DIR / "market_lines.csv"
    market_merged = False

    market = pd.DataFrame()
    if market_path.exists():
        market = pd.read_csv(market_path, dtype={"event_id": str})
        market = normalize_numeric_dtypes(market)
        market = normalize_column_names(market)
    else:
        market = _build_market_lines_fallback(df, market_path)

    if not market.empty:
        # Protect model columns from market collision
        if "spread" in market.columns:
            market = market.rename(columns={"spread": "market_spread"})
        if "pred_spread" in market.columns:
            market = market.drop(columns=["pred_spread"])

        if "captured_at_utc" in market.columns:
            capture_order = {"closing": 0, "pregame": 1, "opening": 2}
            market["_cap_rank"] = (
                market.get("capture_type", pd.Series(dtype=str))
                .map(capture_order).fillna(9)
            )
            market_latest = (
                market
                .sort_values("_cap_rank")
                .groupby("event_id")
                .first()
                .reset_index()
                .drop(columns=["_cap_rank"], errors="ignore")
            )
            market_cols = ["event_id"] + expected_market_cols
            available = [c for c in market_cols if c in market_latest.columns]

            # Preserve model spreads before merge to handle potential market line collisions
            _pred_spread_backup = df["pred_spread"].copy() if "pred_spread" in df.columns else None
            _ens_spread_backup = df["ens_ens_spread"].copy() if "ens_ens_spread" in df.columns else None

            # Drop market columns from df before merge — but NEVER drop
            # model output columns even if they share a name with a market col.
            _protected = {"pred_spread", "ens_ens_spread", "predicted_spread",
                          "pred_total", "pred_home_score", "pred_away_score"}
            drop_cols = [
                c for c in expected_market_cols
                if c in df.columns and c not in _protected
            ]
            df = df.drop(columns=drop_cols, errors="ignore")

            df["event_id"] = df["event_id"].astype(str).str.strip()
            market_latest["event_id"] = (
                market_latest["event_id"].astype(str).str.strip()
            )

            before_rows = len(df)
            df = df.merge(
                market_latest[available], on="event_id", how="left"
            )

            log.info(
                "[DIAG] build_predictions_with_context | post-market-merge | pred_spread non-null: %d/%d (%.1f%% null)",
                int(df["pred_spread"].notna().sum()) if "pred_spread" in df.columns else 0,
                len(df),
                (df["pred_spread"].isna().mean() * 100) if "pred_spread" in df.columns else 100.0
            )

            # Restore model spreads if they were lost or partially nullified by merge
            if _pred_spread_backup is not None:
                df["pred_spread"] = _pred_spread_backup.values
            if _ens_spread_backup is not None:
                df["ens_ens_spread"] = _ens_spread_backup.values

            log.info(
                "Market merge diagnostic: pred_spread null rate = %.1f%%",
                df["pred_spread"].isna().mean() * 100 if "pred_spread" in df.columns else 100.0
            )

            market_signal_cols = [
                c for c in [
                    "home_spread_current", "home_spread_open", "line_movement",
                    "pinnacle_spread", "draftkings_spread", "home_tickets_pct",
                    "home_money_pct",
                ] if c in df.columns
            ]
            matched = int(df[market_signal_cols].notna().any(axis=1).sum()) if market_signal_cols else 0
            log.info(
                "Market lines merged: %d/%d games matched | steam=%d, RLM=%d, book_dis=%d",
                matched, before_rows,
                int(df.get("steam_flag", pd.Series(dtype=float))
                    .fillna(False).astype(bool).sum())
                if "steam_flag" in df.columns else 0,
                int(df.get("rlm_flag", pd.Series(dtype=float))
                    .fillna(False).astype(bool).sum())
                if "rlm_flag" in df.columns else 0,
                int(df.get("book_disagreement_flag", pd.Series(dtype=float))
                    .fillna(False).astype(bool).sum())
                if "book_disagreement_flag" in df.columns else 0,
            )
            if matched == 0:
                log.warning(
                    "ZERO market lines matched. Check event_id format. Market sample: %s | Pred sample: %s",
                    market_latest["event_id"].head(3).tolist(),
                    df["event_id"].head(3).tolist(),
                )
            else:
                market_merged = True
    else:
        log.warning("No market lines available after fallback: %s", market_path)

    sit_path = DATA_DIR / "team_situational.csv"
    if sit_path.exists():
        sit = pd.read_csv(sit_path, dtype={"event_id": str, "team_id": str})
        sit_cols = [
            "rest_days", "games_l7", "games_l14", "fatigue_index",
            "win_streak", "cover_streak", "close_win_pct_season",
            "close_game_win_pct", "scoring_consistency_l5",
        ]
        keep_cols = ["event_id", "team_id", "home_away", *[c for c in sit_cols if c in sit.columns]]
        sit = sit[keep_cols].copy()

        home_sit = sit[sit["home_away"] == "home"].copy()
        home_rename = {c: f"home_{c}" for c in sit_cols if c in home_sit.columns}
        home_sit = home_sit.rename(columns=home_rename).rename(columns={"team_id": "home_team_id"})
        home_keep = ["event_id", "home_team_id", *home_rename.values()]
        home_sit = home_sit[home_keep]

        away_sit = sit[sit["home_away"] == "away"].copy()
        away_rename = {c: f"away_{c}" for c in sit_cols if c in away_sit.columns}
        away_sit = away_sit.rename(columns=away_rename).rename(columns={"team_id": "away_team_id"})
        away_keep = ["event_id", "away_team_id", *away_rename.values()]
        away_sit = away_sit[away_keep]

        df["event_id"] = df["event_id"].astype(str).str.strip()
        df["home_team_id"] = df["home_team_id"].astype(str).str.strip()
        df["away_team_id"] = df["away_team_id"].astype(str).str.strip()
        home_sit["event_id"] = home_sit["event_id"].astype(str).str.strip()
        away_sit["event_id"] = away_sit["event_id"].astype(str).str.strip()
        home_sit["home_team_id"] = home_sit["home_team_id"].astype(str).str.strip()
        away_sit["away_team_id"] = away_sit["away_team_id"].astype(str).str.strip()

        df = df.merge(home_sit, on=["event_id", "home_team_id"], how="left")
        df = df.merge(away_sit, on=["event_id", "away_team_id"], how="left")
        log.info(
            "Situational features merged for %d/%d home, %d/%d away games",
            df.get("home_rest_days", pd.Series(dtype=float)).notna().sum(),
            len(df),
            df.get("away_rest_days", pd.Series(dtype=float)).notna().sum(),
            len(df),
        )
    else:
        log.warning("Situational file missing: %s", sit_path)

    # ── Team profile context merge ─────────────────────────────────
    _ctx_sources = [
        DATA_DIR / "team_pretournament_snapshot.csv",
        DATA_DIR / "team_game_weighted.csv",
        DATA_DIR / "csv" / "team_game_weighted.csv",
    ]
    _ctx_df = None
    _ctx_df_full = None
    for _src in _ctx_sources:
        if _src.exists() and _src.stat().st_size > 100:
            try:
                _ctx_df_full = pd.read_csv(_src, dtype={"team_id": str}, low_memory=False)
                _ctx_df = _ctx_df_full
                if "game_datetime_utc" in _ctx_df.columns:
                    _ctx_df = _ctx_df.sort_values("game_datetime_utc")
                _ctx_df = _ctx_df.drop_duplicates("team_id", keep="last")
                log.info(
                    "Team context source: %s (%d teams, cols: %s)",
                    _src.name, len(_ctx_df),
                    [c for c in _ctx_df.columns
                     if any(x in c for x in
                            ['momentum','luck','form','rtg_std','ha_net',
                             'net_eff','adj_net','cage_em'])][:8]
                )
                break
            except Exception as _exc:
                log.debug("Could not load %s: %s", _src, _exc)

    if _ctx_df is not None and "team_id" in _ctx_df.columns:
        # Map output field name -> ordered list of source column candidates
        _ctx_field_map = {
            "momentum_score":  [
                "t_momentum_quality_rating", "momentum_score",
                "momentum_rating", "momentum",
            ],
            "form_rating":     [
                "form_rating", "recent_form", "form_score",
                "t_form_rating",
            ],
            "luck_score":      ["luck_score", "luck"],
            "ha_net_rtg_l10":  [
                "ha_net_rtg_l10", "ha_net_eff_l10",
                "home_away_net_l10",
            ],
            "net_rtg_std_l10": [
                "net_rtg_std_l10", "net_eff_std_l10", "net_rtg_std",
            ],
        }

        # Log which candidates were actually found
        for field, candidates in _ctx_field_map.items():
            found = next(
                (c for c in candidates if c in _ctx_df.columns), None
            )
            if found:
                log.debug("Context field '%s' → source col '%s'", field, found)
            else:
                log.warning(
                    "Context field '%s' not found in %s. "
                    "Tried: %s. Available: %s",
                    field, _src.name, candidates,
                    [c for c in _ctx_df.columns
                     if any(x in c.lower() for x in
                            ['momentum','luck','form','std','ha_'])][:10]
                )

        # Build slim lookup frame
        slim_data = {"team_id": _ctx_df["team_id"].astype(str).str.strip()}
        _DERIVED_FIELDS = {"momentum_tier"}  # computed post-merge
        for field, candidates in _ctx_field_map.items():
            if field in _DERIVED_FIELDS:
                continue
            for cand in candidates:
                if cand in _ctx_df.columns:
                    slim_data[field] = _ctx_df[cand].values
                    break
            else:
                slim_data[field] = pd.NA

        for gp_col in ["games_played", "game_number",
                        "games_before", "n_games"]:
            if gp_col in _ctx_df.columns:
                slim_data["_games_played_season"] = pd.to_numeric(
                    _ctx_df[gp_col], errors="coerce"
                ).values
                break

        slim = pd.DataFrame(slim_data)

        # Drop pre-existing null collision columns before the context merge.
        # Coalesce _x variants if they have data.
        # Drop all _y variants for context fields.
        _ctx_cols = []
        for f in _ctx_field_map:
            _ctx_cols.extend([f"home_{f}", f"away_{f}"])

        for col in _ctx_cols:
            col_x = f"{col}_x"
            col_y = f"{col}_y"

            # Coalesce _x into clean name if clean is null or missing
            if col_x in df.columns:
                if col not in df.columns:
                    df = df.rename(columns={col_x: col})
                else:
                    df[col] = df[col].combine_first(df[col_x])
                    df = df.drop(columns=[col_x])

            # Drop clean if 100% null (to allow merge to fill it)
            if col in df.columns and df[col].isna().all():
                df = df.drop(columns=[col])

            # Drop _y unconditionally (they will be recreated by merge if conflict, then handled post-merge)
            if col_y in df.columns:
                df = df.drop(columns=[col_y])

        # Merge for home team
        home_slim = slim.rename(columns={
            "team_id": "home_team_id",
            "_games_played_season": "home_games_played_season",
            **{f: f"home_{f}" for f in _ctx_field_map},
        })
        df["home_team_id"] = df["home_team_id"].astype(str).str.strip()
        df = df.merge(home_slim, on="home_team_id", how="left")

        # Merge for away team
        away_slim = slim.rename(columns={
            "team_id": "away_team_id",
            "_games_played_season": "away_games_played_season",
            **{f: f"away_{f}" for f in _ctx_field_map},
        })
        df["away_team_id"] = df["away_team_id"].astype(str).str.strip()
        df = df.merge(away_slim, on="away_team_id", how="left")

        # Derive momentum_tier from momentum_score unconditionally
        # Verify no _x/_y collisions remain for context fields
        _collision_vars = []
        for f in _ctx_field_map:
            for side in ["home_", "away_"]:
                for suffix in ["_x", "_y"]:
                    if f"{side}{f}{suffix}" in df.columns:
                        _collision_vars.append(f"{side}{f}{suffix}")
        if _collision_vars:
            log.warning("Context merge left collision columns: %s", _collision_vars)

        # Derive momentum_tier from momentum_score
        for _side in ["home", "away"]:
            _tier_col = f"{_side}_momentum_tier"
            _score_col = f"{_side}_momentum_score"
            if _score_col in df.columns:
                df[_tier_col] = pd.cut(
                    pd.to_numeric(df[_score_col], errors="coerce"),
                _score = pd.to_numeric(df[_score_col], errors="coerce")
                df[_tier_col] = pd.cut(
                    _score,
                    bins=[-999, 40, 50, 60, 70, 999],
                    labels=["COLD", "NEUTRAL", "WARM", "HOT", "ELITE"],
                    right=True,
                ).astype(str).replace("nan", pd.NA)

        # Post-merge cleanup: handle any new _x/_y columns created by the merge
        for col in _ctx_cols:
            for suffix in ["_x", "_y"]:
                cs = f"{col}{suffix}"
                if cs in df.columns:
                    df[col] = df[col].combine_first(df[cs]) if col in df.columns else df[cs]
                    df = df.drop(columns=[cs])

        n_mom = int(df.get("home_momentum_score", pd.Series(dtype=float)).notna().sum())
        log.info(
            "[DIAG] build_predictions_with_context | post-context-merge | home_momentum_score: %d/%d non-null",
            n_mom, len(df)
        )

        # Fix games_used with season total when > 1
        for side in ["home", "away"]:
            gps_col = f"{side}_games_played_season"
            gu_col  = f"{side}_games_used"
            if gps_col in df.columns:
                _gps = pd.to_numeric(df[gps_col], errors="coerce")
                if _gps.max() > 1:
                    df[gu_col] = _gps.combine_first(
                        pd.to_numeric(df.get(gu_col), errors="coerce")
                    )
                    df = df.drop(columns=[gps_col], errors="ignore")

        # Additional fallback: count rows per team_id in weighted CSV
        # to fix home/away_games_used = 1 from single game rows
        if _ctx_df_full is not None and "team_id" in _ctx_df_full.columns:
            for _side, _id_col in [("home", "home_team_id"), ("away", "away_team_id")]:
                _gu_col = f"{_side}_games_used"
                if _id_col in df.columns:
                    _counts = (
                        _ctx_df_full.groupby("team_id")
                        .size()
                        .reset_index(name="_n_games")
                    )
                    _counts["team_id"] = _counts["team_id"].astype(str).str.strip()
                    _count_map = _counts.set_index("team_id")["_n_games"].to_dict()
                    _mapped = df[_id_col].astype(str).str.strip().map(_count_map)
                    if _mapped.max() > 1:
                        df[_gu_col] = _mapped.combine_first(
                            pd.to_numeric(df.get(_gu_col), errors="coerce")
                        )

        n_mom = int(df.get("home_momentum_score", pd.Series(dtype=float)).notna().sum())
        log.info(
            "Team profile context merged: %d/%d games have home_momentum_score",
            n_mom, len(df)
        )
    else:
        log.warning(
            "No team context source found. Checked: %s",
            [str(s) for s in _ctx_sources]
        )

    for col in expected_market_cols:
        if col not in df.columns:
            df[col] = pd.NA

    alpha_defaults = {
        "is_alpha": False,
        "edge_types": "",
        "alpha_reasoning": "",
        "kelly_fraction": 0.0,
        "kelly_units": 0.0,
        "kelly_multiplier": 1.0,
        "market_evaluated": False,
        "edge_pts": 0.0,
    }
    for col, default in alpha_defaults.items():
        if col not in df.columns:
            df[col] = default

    if market_merged:
        log.info(
            "Re-evaluating alpha for %d rows with market context...",
            len(df)
        )
        alpha_results = []
        for _, row in df.iterrows():
            # Robust extraction from potentially nullable Series
            ps = row.get("pred_spread")
            if pd.isna(ps):
                ps = row.get("ens_ens_spread")
            pred_spread = _safe_float(ps)

            sl = row.get("spread_line")
            if pd.isna(sl):
                sl = row.get("home_spread_current")
            spread_line = _safe_float(sl)
            confidence = _safe_float(row.get("model_confidence"), 0.55)

            trap = bool(row.get("trap_game_flag", False))
            reven = {
                "revenge_flag": bool(row.get("revenge_flag", False)),
                "revenge_team": str(row.get("revenge_team", "")),
                "revenge_margin": row.get("revenge_margin"),
            }

            mkt = {}
            for col in expected_market_cols:
                val = row.get(col)
                if not pd.isna(val):
                    mkt[col] = val

            alpha = evaluate_alpha(
                pred_spread=pred_spread or 0.0,
                spread_line=spread_line,
                model_confidence=confidence or 0.55,
                trap_for_favorite=trap,
                revenge_info=reven,
                market_context=mkt if mkt else None,
                game_id=str(row.get("event_id", "")),
                home_team=str(row.get("home_team", "")),
                away_team=str(row.get("away_team", "")),
            )
            alpha_results.append(alpha)

        alpha_df = pd.DataFrame(alpha_results)

        for col in ["is_alpha", "edge_types", "alpha_reasoning",
                    "kelly_fraction", "kelly_units",
                    "kelly_multiplier", "market_evaluated",
                    "edge_pts"]:
            if col in alpha_df.columns:
                df[col] = alpha_df[col].values

        n_alpha = int(df["is_alpha"].fillna(False).astype(bool).sum())
        n_steam_zero = int(
            (alpha_df.get("kelly_multiplier", pd.Series(dtype=float)) == 0.0).sum()
        )
        log.info(
            "Alpha re-evaluation complete: %d alpha games | %d zeroed by steam",
            n_alpha, n_steam_zero,
        )
    else:
        log.warning(
            "Market data not available — alpha not re-evaluated. "
            "Run: python -m ingestion.market_lines --mode all"
        )
        df["market_evaluated"] = False

    pred_col = (
        "pred_spread" if "pred_spread" in df.columns
        else "ens_ens_spread"
    )

    try:
        _pred = pd.to_numeric(df[pred_col], errors="coerce") \
            if pred_col in df.columns else pd.Series(dtype=float)

        # Build _line using fillna chaining — never use `or` with Series
        _line = pd.Series(dtype=float)
        for _lcol in ["spread_line", "home_spread_current"]:
            if _lcol in df.columns:
                _s = pd.to_numeric(df[_lcol], errors="coerce")
                if _line.empty:
                    _line = _s
                else:
                    _line = _line.fillna(_s)

        _open = pd.to_numeric(df["home_spread_open"], errors="coerce") \
            if "home_spread_open" in df.columns \
            else pd.Series(dtype=float)

        # Build _home_net and _away_net with same safe pattern
        _home_net = pd.Series(dtype=float)
        for _nc in ["home_net_eff", "home_adj_net_rtg", "home_net_rtg"]:
            if _nc in df.columns:
                _home_net = pd.to_numeric(df[_nc], errors="coerce")
                break

        _away_net = pd.Series(dtype=float)
        for _nc in ["away_net_eff", "away_adj_net_rtg", "away_net_rtg"]:
            if _nc in df.columns:
                _away_net = pd.to_numeric(df[_nc], errors="coerce")
                break

        # 1. spread_diff_vs_line: how far model is from market
        if len(_line) > 0 and len(_pred) > 0:
            df["spread_diff_vs_line"] = (_line - _pred).round(2)
        else:
            df["spread_diff_vs_line"] = pd.NA

        # 2. eff_edge: raw efficiency differential
        if len(_home_net) > 0 and len(_away_net) > 0:
            df["eff_edge"] = (_home_net - _away_net).round(2)
        else:
            df["eff_edge"] = pd.NA

        _close = pd.to_numeric(df["home_spread_current"], errors="coerce") \
            if "home_spread_current" in df.columns \
            else pd.Series(dtype=float)

        # 3. clv_vs_open: model vs opening line
        if len(_open) > 0 and _open.notna().any() and len(_pred) > 0:
            df["clv_vs_open"] = (_open - _pred).round(3)
        else:
            df["clv_vs_open"] = pd.NA

        # 4. clv_vs_close: model vs current/closing line
        if len(_close) > 0 and _close.notna().any() and len(_pred) > 0:
            df["clv_vs_close"] = (_close - _pred).round(3)
        else:
            df["clv_vs_close"] = pd.NA

        # 5. predicted_spread alias (required by schema)
        if len(_pred) > 0 and _pred.notna().any():
            df["predicted_spread"] = _pred

        # 6. pred_home_score / pred_away_score from spread + total
        if "pred_total" in df.columns and len(_pred) > 0:
            _total = pd.to_numeric(df["pred_total"], errors="coerce")
            # spread = away - home; total = home + away
            # home = (total - spread) / 2
            # away = (total + spread) / 2
            if _pred.notna().any() and _total.notna().any():
                df["pred_home_score"] = ((_total - _pred) / 2).round(1)
                df["pred_away_score"] = ((_total + _pred) / 2).round(1)

        log.info(
            "Computed context cols — "
            "spread_diff_vs_line: %d/%d | eff_edge: %d/%d | "
            "clv_vs_open: %d/%d | clv_vs_close: %d/%d | predicted_spread: %d/%d | "
            "pred_home_score: %d/%d",
            int(df["spread_diff_vs_line"].notna().sum()), len(df),
            int(df["eff_edge"].notna().sum()), len(df),
            int(df["clv_vs_open"].notna().sum()), len(df),
            int(df["clv_vs_close"].notna().sum()), len(df),
            int(df["predicted_spread"].notna().sum())
                if "predicted_spread" in df.columns else 0, len(df),
            int(df["pred_home_score"].notna().sum())
                if "pred_home_score" in df.columns else 0, len(df),
        )

        clv_open_null_rate = float(df["clv_vs_open"].isna().mean()) if "clv_vs_open" in df.columns else 1.0
        clv_close_null_rate = float(df["clv_vs_close"].isna().mean()) if "clv_vs_close" in df.columns else 1.0
        log.info(
            "CLV null rates — clv_vs_open: %.2f%% | clv_vs_close: %.2f%%",
            clv_open_null_rate * 100.0,
            clv_close_null_rate * 100.0,
        )

    except Exception as _exc:
        log.error(
            "Computed columns block failed: %s — "
            "spread_diff_vs_line/eff_edge/predicted_spread will be null",
            _exc,
            exc_info=True,
        )

    # Integrity normalization for downstream consumers.
    df = _normalize_conference_names(df)
    df = _enrich_win_loss_records(df)
    df = _enrich_ats_records(df)
    df = _enrich_win_probabilities(df)

    # Backward-compatible market aliases for downstream consumers.
    if "spread" not in df.columns:
        df["spread"] = pd.to_numeric(df.get("home_spread_current"), errors="coerce")
    else:
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(
            pd.to_numeric(df.get("home_spread_current"), errors="coerce")
        )

    if "over_under" not in df.columns:
        df["over_under"] = pd.to_numeric(df.get("total_line"), errors="coerce")
    else:
        df["over_under"] = pd.to_numeric(df["over_under"], errors="coerce").fillna(
            pd.to_numeric(df.get("total_line"), errors="coerce")
        )

    # Backward-compatible aggregate record aliases expected by some checks.
    if "wins" not in df.columns and "home_wins" in df.columns:
        df["wins"] = df["home_wins"]
    if "losses" not in df.columns and "home_losses" in df.columns:
        df["losses"] = df["home_losses"]

    df = add_conference_name(df)

    # Canonical single conference_name field for downstream DB/UI contracts.
    if "conference_name" not in df.columns:
        if "home_conference" in df.columns:
            df["conference_name"] = df["home_conference"]
        elif "conference" in df.columns:
            df["conference_name"] = df["conference"]
        else:
            df["conference_name"] = ""

    # Protect pred_spread from being overwritten by market spread.
    if "pred_spread" not in df.columns or df["pred_spread"].isna().all():
        if "ens_ens_spread" in df.columns:
            df["pred_spread"] = df["ens_ens_spread"]
            log.warning("pred_spread was null — recovered from ens_ens_spread")
        elif "predicted_spread" in df.columns:
            df["pred_spread"] = df["predicted_spread"]

    if "spread" in df.columns and "pred_spread" in df.columns:
        if df["pred_spread"].notna().any():
            if "market_spread" not in df.columns:
                df = df.rename(columns={"spread": "market_spread"})
        elif df["spread"].notna().any():
            pass

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # [DIAG] Final results logging
    null_spread = df["pred_spread"].isna().sum() if "pred_spread" in df.columns else len(df)
    null_pct = (null_spread / len(df)) * 100 if len(df) > 0 else 0
    n_mom = df["home_momentum_score"].notna().sum() if "home_momentum_score" in df.columns else 0
    unique_totals = df["projected_total"].nunique() if "projected_total" in df.columns else 0
    median_gu = int(df["home_games_used"].median()) if "home_games_used" in df.columns else 0

    log.info(
        "[DIAG] build_predictions_with_context | pred_spread: %d/%d non-null (%.1f%% null) | home_momentum_score: %d/%d non-null | projected_total unique: %d | home_games_used median: %d",
        len(df) - null_spread, len(df), null_pct,
        n_mom, len(df),
        unique_totals,
        median_gu
    )

    log.info(
        "Wrote predictions with context: %d rows -> %s",
        len(df), out_path
    )
    return df


def _safe_float(val, default=None):
    if pd.isna(val):
        return default
    try:
        f = float(val)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    build_predictions_with_context()
