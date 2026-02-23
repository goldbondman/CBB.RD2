"""Build predictions_with_context.csv by adding latest market-line context."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from espn_config import DATA_DIR, OUT_PREDICTIONS_COMBINED, OUT_PREDICTIONS_CONTEXT

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


def build_predictions_with_context(
    predictions_path: Path = OUT_PREDICTIONS_COMBINED,
    out_path: Path = OUT_PREDICTIONS_CONTEXT,
) -> pd.DataFrame:
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions source file: {predictions_path}")

    df = pd.read_csv(predictions_path, dtype={"event_id": str})
    if "event_id" not in df.columns:
        if "game_id" in df.columns:
            df["event_id"] = df["game_id"].astype(str)
        else:
            raise ValueError("predictions file missing required event_id/game_id column")

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
        if not market.empty and "captured_at_utc" in market.columns:
            market_latest = market.sort_values("captured_at_utc").groupby("event_id").last().reset_index()
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
            df = df.merge(market_latest[available], on="event_id", how="left")
            log.info(
                "Market lines merged: %s steam, %s RLM, %s book disagreement",
                int(df.get("steam_flag", pd.Series(dtype=float)).fillna(False).astype(bool).sum()) if "steam_flag" in df.columns else 0,
                int(df.get("rlm_flag", pd.Series(dtype=float)).fillna(False).astype(bool).sum()) if "rlm_flag" in df.columns else 0,
                int(df.get("book_disagreement_flag", pd.Series(dtype=float)).fillna(False).astype(bool).sum()) if "book_disagreement_flag" in df.columns else 0,
            )
    else:
        log.warning("Market lines file missing: %s", market_path)

    for col in expected_market_cols:
        if col not in df.columns:
            df[col] = pd.NA

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("Wrote predictions with context: %s rows -> %s", len(df), out_path)
    return df


if __name__ == "__main__":
    build_predictions_with_context()
