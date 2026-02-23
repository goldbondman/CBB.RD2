"""Build predictions_with_context.csv by adding latest market-line context."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

from espn_config import DATA_DIR, OUT_PREDICTIONS_COMBINED, OUT_PREDICTIONS_CONTEXT
from models.alpha_evaluator import evaluate_alpha

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
        raise FileNotFoundError(
            f"Missing predictions source file: {predictions_path}"
        )

    df = pd.read_csv(predictions_path, dtype={"event_id": str})
    if "event_id" not in df.columns:
        if "game_id" in df.columns:
            df["event_id"] = df["game_id"].astype(str).str.strip()
        else:
            raise ValueError(
                "predictions file missing required event_id/game_id column"
            )

    expected_market_cols = [
        "home_spread_open", "home_spread_current", "line_movement",
        "pinnacle_spread", "draftkings_spread",
        "home_tickets_pct", "home_money_pct",
        "steam_flag", "rlm_flag", "rlm_sharp_side",
        "book_disagreement_flag", "book_sharp_side", "line_freeze_flag",
    ]
    market_path = DATA_DIR / "market_lines.csv"
    market_merged = False

    if market_path.exists():
        market = pd.read_csv(market_path, dtype={"event_id": str})
        if not market.empty and "captured_at_utc" in market.columns:
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

            drop_cols = [c for c in expected_market_cols if c in df.columns]
            df = df.drop(columns=drop_cols, errors="ignore")

            df["event_id"] = df["event_id"].astype(str).str.strip()
            market_latest["event_id"] = (
                market_latest["event_id"].astype(str).str.strip()
            )

            before_rows = len(df)
            df = df.merge(
                market_latest[available], on="event_id", how="left"
            )
            matched = df["home_spread_current"].notna().sum()
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
        log.warning("market_lines.csv not found: %s", market_path)

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
            pred_spread = _safe_float(
                row.get("pred_spread") or row.get("ens_ens_spread")
            )
            spread_line = _safe_float(
                row.get("spread_line") or row.get("home_spread_current")
            )
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
                if val is not None and not (
                    isinstance(val, float) and pd.isna(val)
                ):
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
    if pred_col in df.columns and "home_spread_open" in df.columns:
        df["clv_vs_open"] = (
            pd.to_numeric(df[pred_col], errors="coerce") -
            pd.to_numeric(df["home_spread_open"], errors="coerce")
        ).round(3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info(
        "Wrote predictions with context: %d rows -> %s",
        len(df), out_path
    )
    return df


def _safe_float(val, default=None):
    try:
        f = float(val)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    build_predictions_with_context()
