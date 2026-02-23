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
    conference_id_to_name,
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

    # Keep existing non-zero values if they already exist.
    has_nonzero_records = False
    if {"home_wins", "away_wins"}.issubset(df.columns):
        has_nonzero_records = bool(df["home_wins"].fillna(0).gt(0).any() or df["away_wins"].fillna(0).gt(0).any())
    if has_nonzero_records:
        return df

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
    history_records = _build_records_from_local_history()
    for tid, rec in history_records.items():
        if tid not in records or (int(records[tid].get("wins", 0)) == 0 and int(records[tid].get("losses", 0)) == 0):
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


def build_predictions_with_context(
    predictions_path: Path = OUT_PREDICTIONS_COMBINED,
    out_path: Path = OUT_PREDICTIONS_CONTEXT,
) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions source file: {predictions_path}")

    df = pd.read_csv(predictions_path, dtype={"event_id": str})
    if "event_id" not in df.columns:
        if "game_id" in df.columns:
            df["event_id"] = df["game_id"].astype(str).str.strip()
        else:
            raise ValueError("predictions file missing required event_id/game_id column")
    else:
        df["event_id"] = df["event_id"].astype(str).str.strip()

    expected_market_cols = [
        "home_spread_open",
        "home_spread_current",
        "line_movement",
        "pinnacle_spread",
        "draftkings_spread",
        "home_tickets_pct",
        "home_money_pct",
        "steam_flag",
        "rlm_flag",
        "rlm_sharp_side",
        "book_disagreement_flag",
        "book_sharp_side",
        "line_freeze_flag",
    ]

    market_path = DATA_DIR / "market_lines.csv"
    if market_path.exists():
        market = pd.read_csv(market_path, dtype={"event_id": str})
        market["event_id"] = market["event_id"].astype(str).str.strip()
        if not market.empty:
            capture_order = {"closing": 0, "pregame": 1, "opening": 2}
            market["capture_rank"] = market.get("capture_type", pd.Series(index=market.index)).map(capture_order).fillna(9)
            market_latest = market.sort_values("capture_rank").groupby("event_id").first().reset_index()
            market_cols = [
                "event_id",
                "home_spread_open",
                "home_spread_current",
                "line_movement",
                "pinnacle_spread",
                "draftkings_spread",
                "home_tickets_pct",
                "home_money_pct",
                "steam_flag",
                "rlm_flag",
                "rlm_sharp_side",
                "book_disagreement_flag",
                "book_sharp_side",
                "line_freeze_flag",
            ]
            available = [c for c in market_cols if c in market_latest.columns]
            existing_mkt = [c for c in available if c != "event_id" and c in df.columns]
            df = df.drop(columns=existing_mkt, errors="ignore")
            before = len(df)
            df = df.merge(market_latest[available], on="event_id", how="left")
            matched = int(df["home_spread_current"].notna().sum()) if "home_spread_current" in df.columns else 0
            log.info("Market lines joined: %s/%s games matched (%s rows in market_lines.csv)", matched, before, len(market_latest))
            if matched == 0:
                log.warning(
                    "ZERO market line matches — check event_id format. Market sample=%s Pred sample=%s",
                    market["event_id"].head(3).tolist(),
                    df["event_id"].head(3).tolist(),
                )
    else:
        log.warning("Market lines file missing: %s (run: python -m ingestion.market_lines --mode all)", market_path)

    df = _normalize_conference_names(df)
    df = _enrich_win_loss_records(df)

    for col in expected_market_cols:
        if col not in df.columns:
            df[col] = pd.NA

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote predictions with context: %s rows -> %s", len(df), out_path)
    return df


if __name__ == "__main__":
    build_predictions_with_context()
