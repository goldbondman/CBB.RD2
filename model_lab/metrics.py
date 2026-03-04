from __future__ import annotations

import numpy as np
import pandas as pd


def hit_rate(outcomes: pd.Series) -> float:
    """Return mean of binary outcomes (1=win, 0=loss), excluding NaN."""
    if outcomes is None:
        return float("nan")
    s = pd.to_numeric(outcomes, errors="coerce")
    s = s[s.notna()]
    if s.empty:
        return float("nan")
    return float(s.mean())


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_pred, errors="coerce")
    mask = yt.notna() & yp.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((yt[mask] - yp[mask]).abs().mean())


def american_odds_profit(stake: float, odds: float, won: bool) -> float:
    if won:
        if odds > 0:
            return stake * (odds / 100.0)
        return stake * (100.0 / abs(odds))
    return -stake


def roi_units(
    outcomes: pd.Series,
    odds: pd.Series | None = None,
    default_odds: int = -110,
    stake: float = 1.0,
) -> float:
    """
    Compute ROI as profit / total_staked.
    outcomes must be: 1=win, 0=loss, 0.5=push, NaN=skip.
    """
    if outcomes is None:
        return float("nan")

    out = pd.to_numeric(outcomes, errors="coerce")
    if odds is None:
        odd_series = pd.Series(default_odds, index=out.index, dtype=float)
    else:
        odd_series = pd.to_numeric(odds, errors="coerce").fillna(default_odds)

    valid = out.notna()
    if valid.sum() == 0:
        return float("nan")

    profit = 0.0
    total_stake = 0.0
    for outcome, odd in zip(out[valid].tolist(), odd_series[valid].tolist()):
        total_stake += stake
        if outcome == 0.5:
            continue
        profit += american_odds_profit(stake, float(odd), bool(outcome == 1.0))

    if total_stake <= 0:
        return float("nan")
    return float(profit / total_stake)


def brier_score(y_true: pd.Series, y_prob: pd.Series) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_prob, errors="coerce")
    mask = yt.notna() & yp.notna()
    if mask.sum() == 0:
        return float("nan")
    ytv = yt[mask].clip(0, 1)
    ypv = yp[mask].clip(0, 1)
    return float(np.mean((ypv - ytv) ** 2))


def calibration_error(y_true: pd.Series, y_prob: pd.Series, n_bins: int = 10) -> float:
    yt = pd.to_numeric(y_true, errors="coerce")
    yp = pd.to_numeric(y_prob, errors="coerce")
    mask = yt.notna() & yp.notna()
    if mask.sum() == 0:
        return float("nan")

    ytv = yt[mask].clip(0, 1)
    ypv = yp[mask].clip(0, 1)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(ytv)
    ece = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            in_bin = (ypv >= lo) & (ypv <= hi)
        else:
            in_bin = (ypv >= lo) & (ypv < hi)
        n = int(in_bin.sum())
        if n == 0:
            continue
        acc = float(ytv[in_bin].mean())
        conf = float(ypv[in_bin].mean())
        ece += abs(acc - conf) * (n / total)
    return float(ece)


def clv_spread(
    side_home: pd.Series,
    open_line: pd.Series,
    close_line: pd.Series,
) -> pd.Series:
    """
    Positive values indicate better closing-line value for the chosen side.
    spread convention: negative means home favored.
    """
    home = pd.to_numeric(side_home, errors="coerce")
    opn = pd.to_numeric(open_line, errors="coerce")
    cls = pd.to_numeric(close_line, errors="coerce")
    mask = home.notna() & opn.notna() & cls.notna()
    out = pd.Series(np.nan, index=home.index, dtype=float)
    # Home side: line -3 to -5 is better for bettor (more negative close), so open-close positive.
    out.loc[mask & (home == 1)] = opn[mask & (home == 1)] - cls[mask & (home == 1)]
    # Away side benefits from movement toward away, inverse sign.
    out.loc[mask & (home == 0)] = cls[mask & (home == 0)] - opn[mask & (home == 0)]
    return out


def clv_total(
    side_over: pd.Series,
    open_total: pd.Series,
    close_total: pd.Series,
) -> pd.Series:
    """
    Positive values indicate better closing-line value for the chosen total side.
    """
    over = pd.to_numeric(side_over, errors="coerce")
    opn = pd.to_numeric(open_total, errors="coerce")
    cls = pd.to_numeric(close_total, errors="coerce")
    mask = over.notna() & opn.notna() & cls.notna()
    out = pd.Series(np.nan, index=over.index, dtype=float)
    out.loc[mask & (over == 1)] = cls[mask & (over == 1)] - opn[mask & (over == 1)]
    out.loc[mask & (over == 0)] = opn[mask & (over == 0)] - cls[mask & (over == 0)]
    return out
