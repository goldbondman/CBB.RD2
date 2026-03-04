from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .metrics import (
    brier_score,
    calibration_error,
    clv_spread,
    clv_total,
    hit_rate,
    mean_absolute_error,
    roi_units,
)


def _as_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values
    return pd.Series(values)


def _resolve_ml_odds(
    predicted_home: pd.Series,
    odds: Any,
    default_odds: int,
) -> pd.Series:
    if isinstance(odds, dict):
        home = pd.to_numeric(_as_series(odds.get("home_ml")), errors="coerce")
        away = pd.to_numeric(_as_series(odds.get("away_ml")), errors="coerce")
    elif isinstance(odds, pd.DataFrame):
        home = pd.to_numeric(_as_series(odds.get("home_ml")), errors="coerce")
        away = pd.to_numeric(_as_series(odds.get("away_ml")), errors="coerce")
    else:
        home = pd.Series(default_odds, index=predicted_home.index, dtype=float)
        away = pd.Series(default_odds, index=predicted_home.index, dtype=float)

    home = home.reindex(predicted_home.index).fillna(default_odds)
    away = away.reindex(predicted_home.index).fillna(default_odds)
    return pd.Series(np.where(predicted_home.astype(bool), home, away), index=predicted_home.index, dtype=float)


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    market_line: Any,
    odds: Any,
    *,
    market: str = "spread",
    default_odds: int = -110,
    line_open: Any = None,
    line_close: Any = None,
) -> dict[str, float]:
    """
    Generic evaluation across spread, total, and ml.

    Parameters
    ----------
    y_true:
        spread -> actual margin (home-away)
        total  -> actual total points
        ml     -> home_won (0/1)
    y_pred:
        spread -> predicted spread (negative = home favored)
        total  -> predicted total
        ml     -> predicted home win probability (or score converted to probability)
    market_line:
        spread/total line used for hit/ROI grading.
        ignored for ml.
    odds:
        spread/total: optional american odds series; defaults to -110 if missing.
        ml: dict/dataframe with home_ml and away_ml, or default -110.
    """
    yt = pd.to_numeric(_as_series(y_true), errors="coerce")
    yp = pd.to_numeric(_as_series(y_pred), errors="coerce")

    result: dict[str, float] = {
        "n": float(len(yt)),
        "graded_n": 0.0,
        "hit_rate": float("nan"),
        "roi": float("nan"),
        "clv_mean": float("nan"),
        "mae": float("nan"),
        "brier": float("nan"),
        "calibration_ece": float("nan"),
        "push_rate": float("nan"),
    }

    if market.lower() == "spread":
        line = pd.to_numeric(_as_series(market_line), errors="coerce")
        odds_s = pd.to_numeric(_as_series(odds), errors="coerce") if odds is not None else None

        mask = yt.notna() & yp.notna() & line.notna()
        if mask.sum() == 0:
            return result

        yt_m = yt[mask]
        yp_m = yp[mask]
        line_m = line[mask]

        pick_home = yp_m < line_m
        ats_actual = yt_m + line_m
        push = ats_actual == 0
        wins = np.where(
            push,
            0.5,
            np.where(
                (pick_home & (ats_actual > 0)) | (~pick_home & (ats_actual < 0)),
                1.0,
                0.0,
            ),
        )

        wins_s = pd.Series(wins, index=yt_m.index, dtype=float)
        result["graded_n"] = float(len(wins_s))
        result["push_rate"] = float((wins_s == 0.5).mean())
        result["hit_rate"] = hit_rate(wins_s[wins_s != 0.5])

        used_odds = odds_s[mask] if odds_s is not None else None
        result["roi"] = roi_units(wins_s, used_odds, default_odds=default_odds)

        result["mae"] = mean_absolute_error(yt_m, -yp_m)

        if line_open is not None and line_close is not None:
            opn = pd.to_numeric(_as_series(line_open), errors="coerce").reindex(yt_m.index)
            cls = pd.to_numeric(_as_series(line_close), errors="coerce").reindex(yt_m.index)
            clv = clv_spread(pick_home.astype(int), opn, cls)
            result["clv_mean"] = float(clv.mean(skipna=True))

        return result

    if market.lower() == "total":
        line = pd.to_numeric(_as_series(market_line), errors="coerce")
        odds_s = pd.to_numeric(_as_series(odds), errors="coerce") if odds is not None else None

        mask = yt.notna() & yp.notna() & line.notna()
        if mask.sum() == 0:
            return result

        yt_m = yt[mask]
        yp_m = yp[mask]
        line_m = line[mask]

        pick_over = yp_m > line_m
        push = yt_m == line_m
        wins = np.where(
            push,
            0.5,
            np.where(
                (pick_over & (yt_m > line_m)) | (~pick_over & (yt_m < line_m)),
                1.0,
                0.0,
            ),
        )

        wins_s = pd.Series(wins, index=yt_m.index, dtype=float)
        result["graded_n"] = float(len(wins_s))
        result["push_rate"] = float((wins_s == 0.5).mean())
        result["hit_rate"] = hit_rate(wins_s[wins_s != 0.5])

        used_odds = odds_s[mask] if odds_s is not None else None
        result["roi"] = roi_units(wins_s, used_odds, default_odds=default_odds)
        result["mae"] = mean_absolute_error(yt_m, yp_m)

        if line_open is not None and line_close is not None:
            opn = pd.to_numeric(_as_series(line_open), errors="coerce").reindex(yt_m.index)
            cls = pd.to_numeric(_as_series(line_close), errors="coerce").reindex(yt_m.index)
            clv = clv_total(pick_over.astype(int), opn, cls)
            result["clv_mean"] = float(clv.mean(skipna=True))

        return result

    if market.lower() == "ml":
        prob = yp.copy()
        if ((prob < 0.0) | (prob > 1.0)).any():
            prob = 1.0 / (1.0 + np.exp(-prob / 10.0))

        mask = yt.notna() & prob.notna()
        if mask.sum() == 0:
            return result

        yt_m = yt[mask].clip(0, 1)
        prob_m = prob[mask].clip(0, 1)

        pred_home = prob_m >= 0.5
        wins = (pred_home.astype(int) == yt_m.astype(int)).astype(float)

        result["graded_n"] = float(len(wins))
        result["hit_rate"] = float(wins.mean())
        result["brier"] = brier_score(yt_m, prob_m)
        result["calibration_ece"] = calibration_error(yt_m, prob_m)

        ml_odds = _resolve_ml_odds(pred_home, odds, default_odds=default_odds)
        result["roi"] = roi_units(wins, ml_odds, default_odds=default_odds)
        return result

    raise ValueError(f"Unsupported market: {market}")
