"""Build predictions_with_context.csv by adding latest market-line context."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from espn_config import DATA_DIR, OUT_PREDICTIONS_COMBINED, OUT_PREDICTIONS_CONTEXT
from models.alpha_evaluator import evaluate_alpha

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")


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
            df["event_id"] = df["game_id"].astype(str)
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
