from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .layer1_base_strength import apply_base_strength
from .layer2_context import apply_context_adjustments
from .layer3_situational import apply_situational_layer, discover_situational_rules
from .layer4_monte_carlo import apply_monte_carlo_layer
from .layer5_agreement import apply_agreement_layer, summarize_agreement_buckets
from .layer6_decision import DirectWinModel, apply_decision_layer, fit_direct_win_model


EDGE_BANDS: list[tuple[str, float, float | None]] = [
    ("0-1.5", 0.0, 1.5),
    ("1.51-3.5", 1.5, 3.5),
    ("3.6-5.5", 3.5, 5.5),
    ("5.6+", 5.5, None),
]

ATS_TOTALS_DEFAULT_ODDS = -110.0
KELLY_FRACTION = 0.25
TARGET_TOP_UNITS_PER_SEASON = 30
MONEYLINE_SIGMA = 11.0


@dataclass(frozen=True)
class BacktestArtifacts:
    scorecard: pd.DataFrame
    edge_band_summary: pd.DataFrame
    bet_ledger: pd.DataFrame
    kelly_summary: pd.DataFrame


# Edge-band bin boundaries and labels used consistently across evaluation helpers.
_EDGE_BAND_BINS: list[float] = [0.0, 1.5, 3.0, np.inf]
_EDGE_BAND_LABELS: list[str] = ["0-1.5", "1.5-3", "3+"]


@dataclass(frozen=True)
class Variant:
    backbone: str
    win_prob: str
    mc_mode: str
    situational_on: bool

    @property
    def id(self) -> str:
        sit = "sit_on" if self.situational_on else "sit_off"
        return f"bb_{self.backbone}|wp_{self.win_prob}|mc_{self.mc_mode}|{sit}"


@dataclass
class BacktestArtifacts:
    scorecard: pd.DataFrame
    edge_band_summary: pd.DataFrame
    bet_ledger: pd.DataFrame
    kelly_summary: pd.DataFrame


def variant_matrix() -> list[Variant]:
    backbones = ["A", "B"]
    win_probs = ["A", "B"]
    mc_modes = ["confidence_only", "confidence_filter", "blended"]
    sit = [False, True]
    return [Variant(*parts) for parts in itertools.product(backbones, win_probs, mc_modes, sit)]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col), errors="coerce")


def _edge_band_labels(edge_abs: pd.Series) -> pd.Series:
    edge = pd.to_numeric(edge_abs, errors="coerce").abs()
    out = pd.Series(np.nan, index=edge.index, dtype=object)
    out.loc[edge.notna() & (edge >= 0.0) & (edge <= 1.5)] = "0-1.5"
    out.loc[edge.notna() & (edge > 1.5) & (edge <= 3.5)] = "1.51-3.5"
    out.loc[edge.notna() & (edge > 3.5) & (edge <= 5.5)] = "3.6-5.5"
    out.loc[edge.notna() & (edge > 5.5)] = "5.6+"
    return out


def _home_cover_mask(margin_home: pd.Series, market_spread: pd.Series) -> pd.Series:
    cover_margin = pd.to_numeric(margin_home, errors="coerce") + pd.to_numeric(market_spread, errors="coerce")
    out = pd.Series(pd.NA, index=cover_margin.index, dtype="boolean")
    valid = cover_margin.notna()
    out.loc[valid] = cover_margin.loc[valid] > 0.0
    return out


def _predicted_home_cover_mask(frame: pd.DataFrame) -> pd.Series:
    out = pd.Series(pd.NA, index=frame.index, dtype="boolean")

    if "predicted_ats_side" in frame.columns:
        side = frame["predicted_ats_side"].astype("string").str.upper()
        out.loc[side.eq("HOME")] = True
        out.loc[side.eq("AWAY")] = False

    if "ats_cover_prob_home" in frame.columns:
        prob = _num(frame, "ats_cover_prob_home")
        prob_mask = out.isna() & prob.notna()
        out.loc[prob_mask] = prob.loc[prob_mask] >= 0.5

    projected_margin = _num(frame, "projected_margin_home")
    if projected_margin.isna().all() and "projected_margin_pre_mc" in frame.columns:
        projected_margin = _num(frame, "projected_margin_pre_mc")
    if projected_margin.isna().all() and "projected_spread" in frame.columns:
        projected_margin = -_num(frame, "projected_spread")

    market_spread = _num(frame, "market_spread")
    margin_mask = out.isna() & projected_margin.notna() & market_spread.notna()
    out.loc[margin_mask] = (projected_margin.loc[margin_mask] + market_spread.loc[margin_mask]) > 0.0
    return out


def _american_to_implied_prob(odds: pd.Series | np.ndarray | float) -> pd.Series:
    s = pd.to_numeric(odds, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    neg = s < 0
    pos = s > 0
    out.loc[neg] = (-s.loc[neg]) / ((-s.loc[neg]) + 100.0)
    out.loc[pos] = 100.0 / (s.loc[pos] + 100.0)
    return out.clip(0.01, 0.99)


def _prob_to_fair_american(prob: pd.Series | np.ndarray | float) -> pd.Series:
    p = pd.to_numeric(prob, errors="coerce").clip(0.01, 0.99)
    out = pd.Series(np.nan, index=p.index, dtype=float)
    fav = p >= 0.5
    dog = p < 0.5
    out.loc[fav] = -100.0 * p.loc[fav] / (1.0 - p.loc[fav])
    out.loc[dog] = 100.0 * (1.0 - p.loc[dog]) / p.loc[dog]
    return out


def _profit_per_unit_from_american(odds: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=odds.index, dtype=float)
    pos = odds > 0
    neg = odds < 0
    out.loc[pos] = odds.loc[pos] / 100.0
    out.loc[neg] = 100.0 / (-odds.loc[neg])
    return out


def _spread_to_home_win_prob(spread_home: pd.Series, sigma: float = MONEYLINE_SIGMA) -> pd.Series:
    margin = -pd.to_numeric(spread_home, errors="coerce")
    z = margin / max(float(sigma), 1e-9)
    return pd.Series(0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0))), index=spread_home.index).clip(0.01, 0.99)


def _kelly_fraction(prob: pd.Series, odds_american: pd.Series, fraction: float = KELLY_FRACTION) -> pd.Series:
    p = pd.to_numeric(prob, errors="coerce")
    odds = pd.to_numeric(odds_american, errors="coerce")
    b = _profit_per_unit_from_american(odds)
    q = 1.0 - p
    full = ((b * p) - q) / b
    out = (pd.to_numeric(full, errors="coerce") * max(float(fraction), 0.0)).clip(lower=0.0)
    return out.where(p.notna() & odds.notna(), np.nan).fillna(0.0)


def _annualized_count(count: int, dt: pd.Series) -> float:
    ts = pd.to_datetime(dt, utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    seasons = np.where(ts.dt.month >= 7, ts.dt.year + 1, ts.dt.year)
    n_seasons = max(1, int(pd.Series(seasons).nunique()))
    return float(count) / float(n_seasons)


def _apply_season_quantile_kelly_units(ledger: pd.DataFrame, *, target_top_count: int = TARGET_TOP_UNITS_PER_SEASON) -> pd.DataFrame:
    out = ledger.copy()
    out["recommended_stake_units"] = 0.0
    out["kelly_scale"] = np.nan
    out["season_scale_target_count"] = int(max(1, target_top_count))

    kelly_raw = pd.to_numeric(out.get("kelly_fraction_raw"), errors="coerce")
    mask = kelly_raw > 0
    if not mask.any():
        return out

    game_dt_raw = out["game_datetime_utc"] if "game_datetime_utc" in out.columns else pd.Series(pd.NaT, index=out.index)
    game_dt = pd.to_datetime(game_dt_raw, utc=True, errors="coerce")
    derived_season = pd.Series(
        np.where(game_dt.notna(), np.where(game_dt.dt.month >= 7, game_dt.dt.year + 1, game_dt.dt.year), np.nan),
        index=out.index,
    )
    season_raw = out["season"] if "season" in out.columns else pd.Series(np.nan, index=out.index)
    season_numeric = pd.to_numeric(season_raw, errors="coerce")
    season_key = season_numeric.where(season_numeric.notna(), derived_season).fillna(-1).astype(int)

    target = int(max(1, target_top_count))
    active_index = out.index[mask]
    for season in season_key.loc[active_index].unique():
        season_idx = active_index[season_key.loc[active_index] == season]
        active = kelly_raw.loc[season_idx].astype(float)
        if active.empty:
            continue

        k = target if len(active) >= target else 1
        top_idx = active.nlargest(k).index
        threshold = active.loc[top_idx].min()
        if not np.isfinite(threshold) or threshold <= 0:
            threshold = active.max()
        scale = 5.0 / threshold if np.isfinite(threshold) and threshold > 0 else 0.0

        out.loc[season_idx, "kelly_scale"] = scale
        raw_units = active * scale
        sized = raw_units.clip(lower=0.1, upper=4.9).round(1)
        sized.loc[top_idx] = 5.0
        out.loc[season_idx, "recommended_stake_units"] = sized
    return out


def _empty_backtest_artifacts() -> BacktestArtifacts:
    scorecard_cols = [
        "variant_id",
        "backbone",
        "win_prob_approach",
        "mc_mode",
        "situational_on",
        "sample_size",
        "spread_mae",
        "spread_rmse",
        "winner_accuracy",
        "ats_accuracy",
        "ats_win_pct_edge_gt_1",
        "ats_win_pct_edge_gt_2",
        "ats_win_pct_edge_gt_3",
        "totals_mae",
        "over_under_win_pct_all",
        "total_win_pct_edge_gt_1",
        "total_win_pct_edge_gt_2",
        "total_win_pct_edge_gt_3",
        "calibration_brier",
        "avg_spread_edge",
        "avg_total_edge",
        "ats_edge_buckets",
        "situational_bucket_perf",
        "agreement_bucket_perf",
        "roi_spread",
        "roi_total",
        "moneyline_win_pct_all",
        "roi_moneyline",
        "moneyline_proxy_usage_count",
        "moneyline_proxy_usage_rate",
        "five_unit_bets",
        "five_unit_annualized",
    ]
    edge_cols = [
        "variant_id",
        "market_type",
        "edge_band",
        "edge_unit",
        "bets_graded",
        "wins",
        "losses",
        "pushes",
        "win_rate",
        "units_risked",
        "pnl_units",
        "roi_pct",
        "avg_edge",
        "avg_stake_units",
        "avg_odds_american",
        "moneyline_proxy_usage_rate",
    ]
    ledger_cols = [
        "variant_id",
        "game_id",
        "event_id",
        "game_datetime_utc",
        "phase",
        "round_name",
        "market_type",
        "bet_side",
        "bet_outcome",
        "result",
        "market_line",
        "model_line",
        "market_odds_american",
        "odds_source",
        "pick_probability",
        "market_implied_probability",
        "edge",
        "edge_unit",
        "edge_band",
        "kelly_fraction_raw",
        "recommended_stake_units",
        "units_risked",
        "profit_per_unit",
        "pnl_units",
    ]
    kelly_cols = [
        "variant_id",
        "market_type",
        "bets_total",
        "bets_with_stake",
        "avg_kelly_fraction_raw",
        "avg_stake_units",
        "median_stake_units",
        "max_stake_units",
        "five_unit_bets",
        "five_unit_rate",
        "five_unit_annualized",
        "units_risked",
        "pnl_units",
        "roi_pct",
    ]
    return BacktestArtifacts(
        scorecard=pd.DataFrame(columns=scorecard_cols),
        edge_band_summary=pd.DataFrame(columns=edge_cols),
        bet_ledger=pd.DataFrame(columns=ledger_cols),
        kelly_summary=pd.DataFrame(columns=kelly_cols),
    )


def _apply_variant_totals_projection(frame: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    out = frame.copy()
    base_total = _num(out, "projected_total_ctx")
    expected_total = _num(out, "expected_total")

    if variant.backbone == "A":
        total = (0.6 * base_total.fillna(expected_total)) + (0.4 * expected_total.fillna(base_total))
    else:
        total = base_total.where(base_total.notna(), expected_total)

    to_sum = _num(out, "home_tov_pct_l5") + _num(out, "away_tov_pct_l5")
    oreb_sum = _num(out, "home_orb_pct_l5") + _num(out, "away_orb_pct_l5")
    ftsp_sum = _num(out, "home_ft_scoring_pressure_l5") + _num(out, "away_ft_scoring_pressure_l5")
    three_sum = _num(out, "home_three_par_l5") + _num(out, "away_three_par_l5")
    empty_possession_pressure = (to_sum - oreb_sum).fillna(0.0)

    total = total - (8.0 * empty_possession_pressure)
    total = total + (5.0 * (ftsp_sum.fillna(0.45) - 0.45))
    total = total + (2.5 * (three_sum.fillna(0.70) - 0.70))

    if variant.situational_on:
        situ = _num(out, "situational_score").fillna(0.0)
        total = total + np.clip(situ * 3.0, -2.5, 2.5)

    if variant.mc_mode == "blended":
        mc_mid = (_num(out, "mc_total_p10") + _num(out, "mc_total_p90")) / 2.0
        total = (0.7 * total) + (0.3 * mc_mid.fillna(total))
    elif variant.mc_mode == "confidence_filter":
        vol = _num(out, "mc_volatility").fillna(11.0)
        shrink = np.clip((vol - 10.5) / 8.0, 0.0, 0.25)
        total = ((1.0 - shrink) * total) + (shrink * 138.0)

    out["projected_total_ctx"] = total.where(total.notna(), expected_total).fillna(138.0).clip(110.0, 180.0)
    return out


def _apply_variant(
    ctx_train: pd.DataFrame,
    ctx_test: pd.DataFrame,
    variant: Variant,
    *,
    rulebook_sit: pd.DataFrame,
    direct_model_b: DirectWinModel | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a single variant to pre-computed base+context frames.

    Parameters
    ----------
    ctx_train:
        Training frame already processed by apply_base_strength + apply_context_adjustments.
    ctx_test:
        Test frame already processed by apply_base_strength + apply_context_adjustments.
    rulebook_sit:
        Pre-computed situational rulebook (from discover_situational_rules on ctx_train).
        Used when variant.situational_on is True; ignored otherwise.
    direct_model_b:
        Pre-fitted DirectWinModel for win_prob="B" variants (trained on the raw training data).
        Passed as None when win_prob != "B".
    """
    train = ctx_train.copy()
    test = ctx_test.copy()

    if variant.situational_on:
        rulebook = rulebook_sit
        train = apply_situational_layer(train, rulebook)
        test = apply_situational_layer(test, rulebook)
    else:
        rulebook = pd.DataFrame()
        for frame in (train, test):
            frame["situational_active_rules"] = ""
            frame["situational_score"] = 0.0
            frame["situational_spread_adjustment"] = 0.0
            frame["situational_confidence_boost"] = 0.0
            frame["situational_signal"] = 0

    for frame in (train, test):
        if variant.backbone == "A":
            frame["margin_ctx_blend"] = pd.to_numeric(frame.get("margin_ctx_a"), errors="coerce")
        else:
            frame["margin_ctx_blend"] = pd.to_numeric(frame.get("margin_ctx_b"), errors="coerce")

    train = apply_monte_carlo_layer(train, mode=variant.mc_mode, n_sims=250, fast_approx=True)
    test = apply_monte_carlo_layer(test, mode=variant.mc_mode, n_sims=350, fast_approx=True)
    train = _apply_variant_totals_projection(train, variant)
    test = _apply_variant_totals_projection(test, variant)

    direct_model = direct_model_b if variant.win_prob == "B" else None
    train = apply_decision_layer(train, direct_win_model=direct_model, mc_mode=variant.mc_mode)
    test = apply_decision_layer(test, direct_win_model=direct_model, mc_mode=variant.mc_mode)

    train = apply_agreement_layer(train)
    test = apply_agreement_layer(test)

    return test, rulebook


def _grade_ats(scored: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    df = scored.copy()
    if df.empty:
        return pd.DataFrame()

    market_spread = _num(df, "market_spread")
    margin = _num(df, "actual_margin")
    pred_side_home = _num(df, "ats_cover_prob_home").fillna(0.5) >= 0.5
    cover_margin = margin + market_spread
    home_cover = _home_cover_mask(margin, market_spread).fillna(False)

    result = pd.Series("NO_LINE", index=df.index, dtype=object)
    valid = market_spread.notna() & margin.notna()
    push = valid & cover_margin.eq(0.0)
    non_push = valid & ~push

    pred_correct = np.where(pred_side_home, home_cover, ~home_cover)
    result.loc[non_push] = np.where(pred_correct[non_push], "WIN", "LOSS")
    result.loc[push] = "PUSH"

    pick_probability = np.where(pred_side_home, _num(df, "ats_cover_prob_home"), 1.0 - _num(df, "ats_cover_prob_home"))
    edge_points = _num(df, "edge_home").abs()
    market_line = np.where(pred_side_home, market_spread, -market_spread)
    model_line = np.where(pred_side_home, _num(df, "projected_spread"), -_num(df, "projected_spread"))

    out = pd.DataFrame(
        {
            "variant_id": variant_id,
            "game_id": df.get("game_id"),
            "event_id": df.get("event_id", df.get("game_id")),
            "game_datetime_utc": df.get("game_datetime_utc"),
            "phase": df.get("phase", "all"),
            "round_name": df.get("round_name", "all"),
            "market_type": "ats",
            "bet_side": np.where(pred_side_home, "home", "away"),
            "bet_outcome": np.where(pred_side_home, "home_cover", "away_cover"),
            "result": result,
            "market_line": pd.to_numeric(market_line, errors="coerce"),
            "model_line": pd.to_numeric(model_line, errors="coerce"),
            "market_odds_american": ATS_TOTALS_DEFAULT_ODDS,
            "odds_source": "assumed_-110",
            "pick_probability": pd.to_numeric(pick_probability, errors="coerce").clip(0.01, 0.99),
            "market_implied_probability": _american_to_implied_prob(pd.Series(ATS_TOTALS_DEFAULT_ODDS, index=df.index)),
            "edge": edge_points,
            "edge_unit": "points",
        }
    )
    return out


def _grade_totals(scored: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    df = scored.copy()
    if df.empty:
        return pd.DataFrame()

    market_total = _num(df, "market_total")
    actual_total = _num(df, "actual_total")
    pred_total = _num(df, "projected_total_ctx")

    over_prob = (1.0 / (1.0 + np.exp(-(pred_total - market_total) / 6.0))).clip(0.01, 0.99)
    pick_over = over_prob >= 0.5
    total_diff = actual_total - market_total

    result = pd.Series("NO_LINE", index=df.index, dtype=object)
    valid = market_total.notna() & actual_total.notna()
    push = valid & total_diff.eq(0.0)
    non_push = valid & ~push
    over_hit = total_diff > 0.0
    pred_correct = np.where(pick_over, over_hit, ~over_hit)
    result.loc[non_push] = np.where(pred_correct[non_push], "WIN", "LOSS")
    result.loc[push] = "PUSH"

    pick_probability = np.where(pick_over, over_prob, 1.0 - over_prob)
    edge_points = (pred_total - market_total).abs()

    out = pd.DataFrame(
        {
            "variant_id": variant_id,
            "game_id": df.get("game_id"),
            "event_id": df.get("event_id", df.get("game_id")),
            "game_datetime_utc": df.get("game_datetime_utc"),
            "phase": df.get("phase", "all"),
            "round_name": df.get("round_name", "all"),
            "market_type": "total",
            "bet_side": np.where(pick_over, "over", "under"),
            "bet_outcome": np.where(pick_over, "over", "under"),
            "result": result,
            "market_line": market_total,
            "model_line": pred_total,
            "market_odds_american": ATS_TOTALS_DEFAULT_ODDS,
            "odds_source": "assumed_-110",
            "pick_probability": pd.to_numeric(pick_probability, errors="coerce").clip(0.01, 0.99),
            "market_implied_probability": _american_to_implied_prob(pd.Series(ATS_TOTALS_DEFAULT_ODDS, index=df.index)),
            "edge": edge_points,
            "edge_unit": "points",
        }
    )
    return out


def _grade_moneyline(scored: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    df = scored.copy()
    if df.empty:
        return pd.DataFrame()

    market_spread = _num(df, "market_spread")
    home_odds = _num(df, "moneyline_home")
    away_odds = _num(df, "moneyline_away")
    home_prob_from_odds = _american_to_implied_prob(home_odds)
    away_prob_from_odds = _american_to_implied_prob(away_odds)

    proxy_home_prob = _spread_to_home_win_prob(market_spread, sigma=MONEYLINE_SIGMA)
    proxy_away_prob = (1.0 - proxy_home_prob).clip(0.01, 0.99)
    proxy_home_odds = _prob_to_fair_american(proxy_home_prob)
    proxy_away_odds = _prob_to_fair_american(proxy_away_prob)

    market_prob_home = home_prob_from_odds.where(home_prob_from_odds.notna(), proxy_home_prob)
    market_prob_away = away_prob_from_odds.where(away_prob_from_odds.notna(), proxy_away_prob)
    market_odds_home = home_odds.where(home_odds.notna(), proxy_home_odds)
    market_odds_away = away_odds.where(away_odds.notna(), proxy_away_odds)

    model_prob_home = _num(df, "win_prob_home").clip(0.01, 0.99)
    model_prob_away = (1.0 - model_prob_home).clip(0.01, 0.99)

    edge_home = model_prob_home - market_prob_home
    edge_away = model_prob_away - market_prob_away
    pick_home = edge_home >= edge_away

    picked_market_prob = np.where(pick_home, market_prob_home, market_prob_away)
    picked_model_prob = np.where(pick_home, model_prob_home, model_prob_away)
    picked_edge = np.where(pick_home, edge_home, edge_away)
    picked_odds = np.where(pick_home, market_odds_home, market_odds_away)
    picked_odds_series = pd.Series(picked_odds, index=df.index)
    picked_side = np.where(pick_home, "home", "away")
    picked_outcome = np.where(pick_home, "home_win", "away_win")
    pick_won = np.where(pick_home, _num(df, "home_won") > 0.5, _num(df, "home_won") <= 0.5)

    result = pd.Series("NO_LINE", index=df.index, dtype=object)
    valid = pd.to_numeric(picked_odds_series, errors="coerce").notna() & _num(df, "home_won").notna()
    result.loc[valid] = np.where(pick_won[valid], "WIN", "LOSS")

    odds_source = np.where(home_odds.notna() & away_odds.notna(), "market_moneyline", "spread_proxy")

    out = pd.DataFrame(
        {
            "variant_id": variant_id,
            "game_id": df.get("game_id"),
            "event_id": df.get("event_id", df.get("game_id")),
            "game_datetime_utc": df.get("game_datetime_utc"),
            "phase": df.get("phase", "all"),
            "round_name": df.get("round_name", "all"),
            "market_type": "moneyline",
            "bet_side": picked_side,
            "bet_outcome": picked_outcome,
            "result": result,
            "market_line": pd.to_numeric(picked_odds_series, errors="coerce"),
            "model_line": _prob_to_fair_american(pd.Series(picked_model_prob, index=df.index)),
            "market_odds_american": pd.to_numeric(picked_odds_series, errors="coerce"),
            "odds_source": odds_source,
            "pick_probability": pd.to_numeric(picked_model_prob, errors="coerce").clip(0.01, 0.99),
            "market_implied_probability": pd.to_numeric(picked_market_prob, errors="coerce").clip(0.01, 0.99),
                "edge": np.abs(pd.to_numeric(picked_edge, errors="coerce")) * 100.0,
            "edge_unit": "pp",
        }
    )
    return out


def _grade_markets(scored: pd.DataFrame, variant_id: str) -> pd.DataFrame:
    ats = _grade_ats(scored, variant_id)
    totals = _grade_totals(scored, variant_id)
    moneyline = _grade_moneyline(scored, variant_id)
    out = pd.concat([ats, totals, moneyline], ignore_index=True, sort=False)
    if out.empty:
        return out

    out["edge"] = pd.to_numeric(out["edge"], errors="coerce").abs()
    out["edge_band"] = _edge_band_labels(out["edge"])
    out["profit_per_unit"] = _profit_per_unit_from_american(pd.to_numeric(out["market_odds_american"], errors="coerce"))
    out["kelly_fraction_raw"] = _kelly_fraction(
        pd.to_numeric(out["pick_probability"], errors="coerce").clip(0.01, 0.99),
        pd.to_numeric(out["market_odds_american"], errors="coerce"),
        fraction=KELLY_FRACTION,
    )
    out = _apply_season_quantile_kelly_units(out, target_top_count=TARGET_TOP_UNITS_PER_SEASON)
    out["units_risked"] = np.where(out["recommended_stake_units"] > 0.0, out["recommended_stake_units"], 0.0)
    out["pnl_units"] = np.where(
        out["result"].eq("WIN"),
        out["units_risked"] * out["profit_per_unit"],
        np.where(out["result"].eq("LOSS"), -out["units_risked"], 0.0),
    )
    out["bet_outcome"] = out["bet_outcome"].astype(str)
    out["market_type"] = out["market_type"].astype(str)
    return out


def _win_rate(results: pd.Series) -> float:
    graded = results.isin(["WIN", "LOSS"])
    if not graded.any():
        return np.nan
    return float(results[graded].eq("WIN").mean())


def _roi_pct(units_risked: pd.Series, pnl_units: pd.Series) -> float:
    risked = float(pd.to_numeric(units_risked, errors="coerce").fillna(0.0).sum())
    pnl = float(pd.to_numeric(pnl_units, errors="coerce").fillna(0.0).sum())
    if risked <= 0:
        return np.nan
    return (pnl / risked) * 100.0


def _summarize_edge_bands(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if ledger.empty:
        return pd.DataFrame()

    for (variant_id, market_type, edge_band), grp in ledger.groupby(["variant_id", "market_type", "edge_band"], observed=False, dropna=False):
        if pd.isna(edge_band):
            continue
        wins = int((grp["result"] == "WIN").sum())
        losses = int((grp["result"] == "LOSS").sum())
        pushes = int((grp["result"] == "PUSH").sum())
        graded = wins + losses + pushes
        units_risked = float(pd.to_numeric(grp["units_risked"], errors="coerce").fillna(0.0).sum())
        pnl_units = float(pd.to_numeric(grp["pnl_units"], errors="coerce").fillna(0.0).sum())
        roi = (pnl_units / units_risked * 100.0) if units_risked > 0 else np.nan
        rows.append(
            {
                "variant_id": variant_id,
                "market_type": market_type,
                "edge_band": edge_band,
                "edge_unit": grp["edge_unit"].mode(dropna=True).iloc[0] if grp["edge_unit"].notna().any() else np.nan,
                "bets_graded": graded,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate": _win_rate(grp["result"]),
                "units_risked": units_risked,
                "pnl_units": pnl_units,
                "roi_pct": roi,
                "avg_edge": float(pd.to_numeric(grp["edge"], errors="coerce").mean()),
                "avg_stake_units": float(pd.to_numeric(grp["recommended_stake_units"], errors="coerce").mean()),
                "avg_odds_american": float(pd.to_numeric(grp["market_odds_american"], errors="coerce").mean()),
                "moneyline_proxy_usage_rate": float(grp["odds_source"].eq("spread_proxy").mean()) if market_type == "moneyline" else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _summarize_kelly(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if ledger.empty:
        return pd.DataFrame()

    grouped = ledger.groupby(["variant_id", "market_type"], observed=False, dropna=False)
    for (variant_id, market_type), grp in grouped:
        units = pd.to_numeric(grp["recommended_stake_units"], errors="coerce").fillna(0.0)
        active = units > 0
        five_count = int((units >= 5.0).sum())
        rows.append(
            {
                "variant_id": variant_id,
                "market_type": market_type,
                "bets_total": int(len(grp)),
                "bets_with_stake": int(active.sum()),
                "avg_kelly_fraction_raw": float(pd.to_numeric(grp["kelly_fraction_raw"], errors="coerce").fillna(0.0).mean()),
                "avg_stake_units": float(units[active].mean()) if active.any() else 0.0,
                "median_stake_units": float(units[active].median()) if active.any() else 0.0,
                "max_stake_units": float(units.max()) if len(units) else 0.0,
                "five_unit_bets": five_count,
                "five_unit_rate": float(five_count / max(int(active.sum()), 1)) if active.any() else 0.0,
                "five_unit_annualized": _annualized_count(five_count, pd.to_datetime(grp["game_datetime_utc"], utc=True, errors="coerce")),
                "units_risked": float(pd.to_numeric(grp["units_risked"], errors="coerce").fillna(0.0).sum()),
                "pnl_units": float(pd.to_numeric(grp["pnl_units"], errors="coerce").fillna(0.0).sum()),
                "roi_pct": _roi_pct(grp["units_risked"], grp["pnl_units"]),
            }
        )
    return pd.DataFrame(rows)


def _compute_variant_metrics(scored: pd.DataFrame, ledger: pd.DataFrame, variant_id: str) -> dict[str, float | int | str]:
    if scored.empty:
        return {
            "sample_size": 0,
            "spread_mae": np.nan,
            "spread_rmse": np.nan,
            "winner_accuracy": np.nan,
            "ats_accuracy": np.nan,
            "ats_win_pct_edge_gt_1": np.nan,
            "ats_win_pct_edge_gt_2": np.nan,
            "ats_win_pct_edge_gt_3": np.nan,
            "totals_mae": np.nan,
            "over_under_win_pct_all": np.nan,
            "total_win_pct_edge_gt_1": np.nan,
            "total_win_pct_edge_gt_2": np.nan,
            "total_win_pct_edge_gt_3": np.nan,
            "calibration_brier": np.nan,
            "avg_spread_edge": np.nan,
            "avg_total_edge": np.nan,
            "ats_edge_buckets": "{}",
            "situational_bucket_perf": "{}",
            "agreement_bucket_perf": "{}",
            "roi_spread": np.nan,
            "roi_total": np.nan,
            "moneyline_win_pct_all": np.nan,
            "roi_moneyline": np.nan,
            "moneyline_proxy_usage_count": 0,
            "moneyline_proxy_usage_rate": 0.0,
            "five_unit_bets": 0,
            "five_unit_annualized": 0.0,
        }

    margin = _num(scored, "actual_margin")
    total = _num(scored, "actual_total")
    home_won = _num(scored, "home_won")
    pred_margin = _num(scored, "projected_margin_home")
    pred_total = _num(scored, "projected_total_ctx")
    pred_win = _num(scored, "win_prob_home")

    predicted_home_win = pred_win >= 0.5
    winner_accuracy = float((predicted_home_win == (home_won > 0.5)).mean()) if home_won.notna().any() else np.nan
    spread_mae = float((pred_margin - margin).abs().mean()) if margin.notna().any() else np.nan
    spread_rmse = float(np.sqrt(np.mean((pred_margin - margin) ** 2))) if margin.notna().any() else np.nan
    totals_mae = float((pred_total - total).abs().mean()) if total.notna().any() else np.nan
    brier = float(np.mean((pred_win - home_won) ** 2)) if home_won.notna().any() else np.nan

    ats = ledger[(ledger["variant_id"] == variant_id) & ledger["market_type"].eq("ats")].copy()
    totals = ledger[(ledger["variant_id"] == variant_id) & ledger["market_type"].eq("total")].copy()
    ml = ledger[(ledger["variant_id"] == variant_id) & ledger["market_type"].eq("moneyline")].copy()

    def _threshold_win_rate(frame: pd.DataFrame, threshold: float) -> float:
        if frame.empty:
            return np.nan
        hit = frame[pd.to_numeric(frame["edge"], errors="coerce") >= threshold]
        return _win_rate(hit["result"]) if not hit.empty else np.nan

    ats_edge = pd.to_numeric(ats["edge"], errors="coerce") if not ats.empty else pd.Series(dtype=float)
    ats_bucket_payload = {}
    if not ats.empty:
        for label, lo, hi in EDGE_BANDS:
            if hi is None:
                seg = ats[ats_edge > lo]
            elif label == "0-1.5":
                seg = ats[(ats_edge >= lo) & (ats_edge <= hi)]
            else:
                seg = ats[(ats_edge > lo) & (ats_edge <= hi)]
            ats_bucket_payload[label] = _win_rate(seg["result"]) if not seg.empty else np.nan

    situ_bucket = pd.cut(pd.to_numeric(scored.get("situational_score"), errors="coerce"), bins=[-np.inf, -0.02, 0.02, np.inf], labels=["negative", "neutral", "positive"])
    predicted_home_cover = _predicted_home_cover_mask(scored)
    market_spread = _num(scored, "market_spread")
    home_cover = _home_cover_mask(margin, market_spread)
    ats_correct = predicted_home_cover.eq(home_cover).where(predicted_home_cover.notna() & home_cover.notna()).astype("Float64")
    situ_perf = (
        pd.DataFrame({"bucket": situ_bucket, "ats_correct": ats_correct})
        .groupby("bucket", observed=False)["ats_correct"]
        .mean()
        .to_dict()
    )

    agreement = summarize_agreement_buckets(scored)
    agreement_payload = agreement.set_index("agreement_bucket")["ats_accuracy"].to_dict() if not agreement.empty else {}

    five_count = int((pd.to_numeric(ledger["recommended_stake_units"], errors="coerce") >= 5.0).sum()) if not ledger.empty else 0
    five_annual = _annualized_count(five_count, pd.to_datetime(ledger.get("game_datetime_utc"), utc=True, errors="coerce")) if not ledger.empty else 0.0

    ml_proxy_count = int(ml["odds_source"].eq("spread_proxy").sum()) if not ml.empty else 0
    ml_proxy_rate = float(ml["odds_source"].eq("spread_proxy").mean()) if not ml.empty else 0.0

    return {
        "sample_size": int(len(scored)),
        "spread_mae": spread_mae,
        "spread_rmse": spread_rmse,
        "winner_accuracy": winner_accuracy,
        "ats_accuracy": _win_rate(ats["result"]) if not ats.empty else np.nan,
        "ats_win_pct_edge_gt_1": _threshold_win_rate(ats, 1.0),
        "ats_win_pct_edge_gt_2": _threshold_win_rate(ats, 2.0),
        "ats_win_pct_edge_gt_3": _threshold_win_rate(ats, 3.0),
        "totals_mae": totals_mae,
        "over_under_win_pct_all": _win_rate(totals["result"]) if not totals.empty else np.nan,
        "total_win_pct_edge_gt_1": _threshold_win_rate(totals, 1.0),
        "total_win_pct_edge_gt_2": _threshold_win_rate(totals, 2.0),
        "total_win_pct_edge_gt_3": _threshold_win_rate(totals, 3.0),
        "calibration_brier": brier,
        "avg_spread_edge": float(pd.to_numeric(ats["edge"], errors="coerce").mean()) if not ats.empty else np.nan,
        "avg_total_edge": float(pd.to_numeric(totals["edge"], errors="coerce").mean()) if not totals.empty else np.nan,
        "ats_edge_buckets": json.dumps(ats_bucket_payload),
        "situational_bucket_perf": json.dumps(situ_perf),
        "agreement_bucket_perf": json.dumps(agreement_payload),
        "roi_spread": _roi_pct(ats["units_risked"], ats["pnl_units"]) if not ats.empty else np.nan,
        "roi_total": _roi_pct(totals["units_risked"], totals["pnl_units"]) if not totals.empty else np.nan,
        "moneyline_win_pct_all": _win_rate(ml["result"]) if not ml.empty else np.nan,
        "roi_moneyline": _roi_pct(ml["units_risked"], ml["pnl_units"]) if not ml.empty else np.nan,
        "moneyline_proxy_usage_count": ml_proxy_count,
        "moneyline_proxy_usage_rate": ml_proxy_rate,
        "five_unit_bets": five_count,
        "five_unit_annualized": five_annual,
    }


def run_variant_backtest(historical_games: pd.DataFrame) -> BacktestArtifacts:
    if historical_games.empty:
        return _empty_backtest_artifacts()

    df = historical_games.copy()
    df = df.sort_values("game_datetime_utc", kind="mergesort").reset_index(drop=True)
    min_train = max(120, int(len(df) * 0.7))
    if min_train >= len(df):
        min_train = max(1, len(df) - 1)
    windows = [(np.arange(0, min_train), np.arange(min_train, len(df)))]

    # Pre-compute invariant stages that do not depend on any variant parameter.
    # apply_base_strength and apply_context_adjustments are pure functions of the
    # input data, so we compute them once per (train, test) window instead of
    # once per variant (24×).  discover_situational_rules and fit_direct_win_model
    # are likewise independent of variant parameters and are pre-computed once.
    _precomputed: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DirectWinModel | None]] = []
    for train_idx, test_idx in windows:
        raw_train = df.iloc[train_idx].copy()
        raw_test = df.iloc[test_idx].copy()
        if raw_train.empty or raw_test.empty:
            _precomputed.append((raw_train, raw_test, pd.DataFrame(), None))
            continue
        ctx_train = apply_context_adjustments(apply_base_strength(raw_train))
        ctx_test = apply_context_adjustments(apply_base_strength(raw_test))
        rulebook_sit = discover_situational_rules(ctx_train)
        # fit_direct_win_model uses only DIRECT_WIN_FEATURES which are original input
        # columns never mutated by any layer, so raw_train and ctx_train are equivalent.
        direct_model_b = fit_direct_win_model(raw_train)
        _precomputed.append((ctx_train, ctx_test, rulebook_sit, direct_model_b))

    score_rows: list[dict[str, object]] = []
    ledgers: list[pd.DataFrame] = []
    for variant in variant_matrix():
        scored_parts: list[pd.DataFrame] = []
        for (ctx_train, ctx_test, rulebook_sit, direct_model_b) in _precomputed:
            if ctx_train.empty or ctx_test.empty:
                continue
            scored, _ = _apply_variant(ctx_train, ctx_test, variant, rulebook_sit=rulebook_sit, direct_model_b=direct_model_b)
            scored_parts.append(scored)

        scored_all = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
        ledger = _grade_markets(scored_all, variant.id)
        if not ledger.empty:
            ledgers.append(ledger)
        metrics = _compute_variant_metrics(scored_all, ledger, variant.id)
        score_rows.append(
            {
                "variant_id": variant.id,
                "backbone": variant.backbone,
                "win_prob_approach": variant.win_prob,
                "mc_mode": variant.mc_mode,
                "situational_on": bool(variant.situational_on),
                **metrics,
            }
        )

    scorecard = pd.DataFrame(score_rows)
    if not scorecard.empty:
        scorecard = scorecard.sort_values(["ats_accuracy", "winner_accuracy", "spread_mae"], ascending=[False, False, True], na_position="last").reset_index(drop=True)

    empty = _empty_backtest_artifacts()
    bet_ledger = pd.concat(ledgers, ignore_index=True, sort=False) if ledgers else empty.bet_ledger
    edge_band_summary = _summarize_edge_bands(bet_ledger)
    kelly_summary = _summarize_kelly(bet_ledger)

    return BacktestArtifacts(
        scorecard=scorecard if not scorecard.empty else empty.scorecard,
        edge_band_summary=edge_band_summary if not edge_band_summary.empty else empty.edge_band_summary,
        bet_ledger=bet_ledger,
        kelly_summary=kelly_summary if not kelly_summary.empty else empty.kelly_summary,
    )
