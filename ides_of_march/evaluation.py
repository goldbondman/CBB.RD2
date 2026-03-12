from __future__ import annotations

import itertools
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .layer1_base_strength import apply_base_strength
from .layer2_context import apply_context_adjustments
from .layer3_situational import apply_situational_layer, discover_situational_rules
from .layer4_monte_carlo import apply_monte_carlo_layer
from .layer5_agreement import apply_agreement_layer, summarize_agreement_buckets
from .layer6_decision import apply_decision_layer, fit_direct_win_model


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


def variant_matrix() -> list[Variant]:
    backbones = ["A", "B"]
    win_probs = ["A", "B"]
    mc_modes = ["confidence_only", "confidence_filter", "blended"]
    sit = [False, True]
    return [Variant(*parts) for parts in itertools.product(backbones, win_probs, mc_modes, sit)]


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame.get(col), errors="coerce")


def _apply_variant_totals_projection(frame: pd.DataFrame, variant: Variant) -> pd.DataFrame:
    out = frame.copy()
    base_total = _num(out, "projected_total_ctx")
    expected_total = _num(out, "expected_total")

    if variant.backbone == "A":
        total = (0.6 * base_total.fillna(expected_total)) + (0.4 * expected_total.fillna(base_total))
    else:
        total = base_total.where(base_total.notna(), expected_total)

    # Totals-specific possession retention / empty possession pressure.
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


def _apply_variant(train_df: pd.DataFrame, test_df: pd.DataFrame, variant: Variant) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = apply_base_strength(train_df)
    train = apply_context_adjustments(train)
    test = apply_base_strength(test_df)
    test = apply_context_adjustments(test)

    if variant.situational_on:
        rulebook = discover_situational_rules(train)
        train = apply_situational_layer(train, rulebook)
        test = apply_situational_layer(test, rulebook)
    else:
        rulebook = discover_situational_rules(train.iloc[0:0])
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

    direct_model = fit_direct_win_model(train) if variant.win_prob == "B" else None
    train = apply_decision_layer(train, direct_win_model=direct_model, mc_mode=variant.mc_mode)
    test = apply_decision_layer(test, direct_win_model=direct_model, mc_mode=variant.mc_mode)

    train = apply_agreement_layer(train)
    test = apply_agreement_layer(test)

    return test, rulebook


def _compute_metrics(scored: pd.DataFrame) -> dict[str, float | str]:
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
        }

    df = scored.copy()
    margin = pd.to_numeric(df.get("actual_margin"), errors="coerce")
    total = pd.to_numeric(df.get("actual_total"), errors="coerce")
    home_won = pd.to_numeric(df.get("home_won"), errors="coerce")
    market_spread = pd.to_numeric(df.get("market_spread"), errors="coerce")
    market_total = pd.to_numeric(df.get("market_total"), errors="coerce")

    pred_margin = pd.to_numeric(df.get("projected_margin_home"), errors="coerce")
    pred_total = pd.to_numeric(df.get("projected_total_ctx"), errors="coerce")
    pred_win = pd.to_numeric(df.get("win_prob_home"), errors="coerce")

    predicted_home_win = pred_win >= 0.5
    winner_accuracy = float((predicted_home_win == (home_won > 0.5)).mean()) if home_won.notna().any() else np.nan

    home_cover = (margin + market_spread) > 0
    predicted_home_cover = df.get("predicted_ats_side", pd.Series("HOME", index=df.index)).astype(str).eq("HOME")
    ats_accuracy = float((predicted_home_cover == home_cover).mean()) if margin.notna().any() and market_spread.notna().any() else np.nan

    spread_mae = float((pred_margin - margin).abs().mean()) if margin.notna().any() else np.nan
    spread_rmse = float(np.sqrt(np.mean((pred_margin - margin) ** 2))) if margin.notna().any() else np.nan
    brier = float(np.mean((pred_win - home_won) ** 2)) if home_won.notna().any() else np.nan

    edge_abs = pd.to_numeric(df.get("edge_home"), errors="coerce").abs()
    ats_correct = (predicted_home_cover == home_cover).astype(float)
    ats_win_pct_edge_gt_1 = float(ats_correct[edge_abs >= 1.0].mean()) if (edge_abs >= 1.0).any() else np.nan
    ats_win_pct_edge_gt_2 = float(ats_correct[edge_abs >= 2.0].mean()) if (edge_abs >= 2.0).any() else np.nan
    ats_win_pct_edge_gt_3 = float(ats_correct[edge_abs >= 3.0].mean()) if (edge_abs >= 3.0).any() else np.nan

    over_prob = (1.0 / (1.0 + np.exp(-(pred_total - market_total) / 6.0))).clip(0.01, 0.99)
    over_pick = over_prob >= 0.5
    actual_over = total > market_total
    ou_correct = (over_pick == actual_over).astype(float)
    totals_mae = float((pred_total - total).abs().mean()) if total.notna().any() else np.nan
    over_under_win_pct_all = float(ou_correct.mean()) if total.notna().any() and market_total.notna().any() else np.nan

    total_edge_abs = (pred_total - market_total).abs()
    total_win_pct_edge_gt_1 = float(ou_correct[total_edge_abs >= 1.0].mean()) if (total_edge_abs >= 1.0).any() else np.nan
    total_win_pct_edge_gt_2 = float(ou_correct[total_edge_abs >= 2.0].mean()) if (total_edge_abs >= 2.0).any() else np.nan
    total_win_pct_edge_gt_3 = float(ou_correct[total_edge_abs >= 3.0].mean()) if (total_edge_abs >= 3.0).any() else np.nan
    bucket = pd.cut(edge_abs, bins=[-np.inf, 1.5, 3.0, np.inf], labels=["0-1.5", "1.5-3", "3+"])
    edge_perf = (
        pd.DataFrame({"bucket": bucket, "ats_correct": (predicted_home_cover == home_cover).astype(float)})
        .groupby("bucket", observed=False)["ats_correct"]
        .mean()
        .to_dict()
    )

    situ_bucket = pd.cut(pd.to_numeric(df.get("situational_score"), errors="coerce"), bins=[-np.inf, -0.02, 0.02, np.inf], labels=["negative", "neutral", "positive"])
    situ_perf = (
        pd.DataFrame({"bucket": situ_bucket, "ats_correct": (predicted_home_cover == home_cover).astype(float)})
        .groupby("bucket", observed=False)["ats_correct"]
        .mean()
        .to_dict()
    )

    agreement = summarize_agreement_buckets(df)
    agreement_payload = agreement.set_index("agreement_bucket")["ats_accuracy"].to_dict()

    return {
        "sample_size": int(len(df)),
        "spread_mae": spread_mae,
        "spread_rmse": spread_rmse,
        "winner_accuracy": winner_accuracy,
        "ats_accuracy": ats_accuracy,
        "ats_win_pct_edge_gt_1": ats_win_pct_edge_gt_1,
        "ats_win_pct_edge_gt_2": ats_win_pct_edge_gt_2,
        "ats_win_pct_edge_gt_3": ats_win_pct_edge_gt_3,
        "totals_mae": totals_mae,
        "over_under_win_pct_all": over_under_win_pct_all,
        "total_win_pct_edge_gt_1": total_win_pct_edge_gt_1,
        "total_win_pct_edge_gt_2": total_win_pct_edge_gt_2,
        "total_win_pct_edge_gt_3": total_win_pct_edge_gt_3,
        "calibration_brier": brier,
        "avg_spread_edge": float(edge_abs.mean()) if edge_abs.notna().any() else np.nan,
        "avg_total_edge": float(total_edge_abs.mean()) if total_edge_abs.notna().any() else np.nan,
        "ats_edge_buckets": json.dumps(edge_perf),
        "situational_bucket_perf": json.dumps(situ_perf),
        "agreement_bucket_perf": json.dumps(agreement_payload),
    }


def _build_bet_ledger(variant_id: str, scored: pd.DataFrame) -> pd.DataFrame:
    """Build individual bet rows from a scored DataFrame for one variant."""
    _COLS = [
        "variant_id", "game_id", "game_datetime_utc",
        "market_type", "odds_source", "edge", "side", "bet_won",
    ]
    if scored.empty:
        return pd.DataFrame(columns=_COLS)

    s = scored.reset_index(drop=True)
    gid = s.get("game_id", pd.Series(dtype=str, index=s.index))
    gdt = s.get("game_datetime_utc", pd.Series(dtype=str, index=s.index))

    ms = pd.to_numeric(s.get("market_spread"), errors="coerce")
    mt = pd.to_numeric(s.get("market_total"), errors="coerce")
    pm = pd.to_numeric(s.get("projected_margin_home"), errors="coerce")
    pt = pd.to_numeric(s.get("projected_total_ctx"), errors="coerce")
    am = pd.to_numeric(s.get("actual_margin"), errors="coerce")
    at_ = pd.to_numeric(s.get("actual_total"), errors="coerce")
    wp = pd.to_numeric(s.get("win_prob_home"), errors="coerce")
    hw = pd.to_numeric(s.get("home_won"), errors="coerce")

    ledger_parts: list[pd.DataFrame] = []

    # ATS bets
    ats_mask = ms.notna() & pm.notna()
    if ats_mask.any():
        raw_edge = pm[ats_mask] - ms[ats_mask]
        predicted_home_cover = raw_edge > 0
        home_cover = (am[ats_mask] + ms[ats_mask]) > 0
        has_result = am[ats_mask].notna()
        bet_won = np.where(has_result, (predicted_home_cover == home_cover).astype(float), np.nan)
        ats = pd.DataFrame({
            "variant_id": variant_id,
            "game_id": gid[ats_mask].values,
            "game_datetime_utc": gdt[ats_mask].values,
            "market_type": "ats",
            "odds_source": "market",
            "edge": raw_edge.abs().round(4).values,
            "side": np.where(predicted_home_cover, "HOME", "AWAY"),
            "bet_won": bet_won,
        })
        ledger_parts.append(ats)

    # Total bets
    tot_mask = mt.notna() & pt.notna()
    if tot_mask.any():
        raw_total_edge = pt[tot_mask] - mt[tot_mask]
        predicted_over = raw_total_edge > 0
        actual_over = at_[tot_mask] > mt[tot_mask]
        has_result = at_[tot_mask].notna()
        bet_won = np.where(has_result, (predicted_over == actual_over).astype(float), np.nan)
        tot = pd.DataFrame({
            "variant_id": variant_id,
            "game_id": gid[tot_mask].values,
            "game_datetime_utc": gdt[tot_mask].values,
            "market_type": "total",
            "odds_source": "market",
            "edge": raw_total_edge.abs().round(4).values,
            "side": np.where(predicted_over, "OVER", "UNDER"),
            "bet_won": bet_won,
        })
        ledger_parts.append(tot)

    # Moneyline bets (proxied from spread model win probability)
    ml_mask = wp.notna()
    if ml_mask.any():
        ml_edge = (wp[ml_mask] - 0.5).abs().round(4)
        predicted_home_win = wp[ml_mask] >= 0.5
        actual_home_win = hw[ml_mask] > 0.5
        has_result = hw[ml_mask].notna()
        bet_won = np.where(has_result, (predicted_home_win == actual_home_win).astype(float), np.nan)
        ml = pd.DataFrame({
            "variant_id": variant_id,
            "game_id": gid[ml_mask].values,
            "game_datetime_utc": gdt[ml_mask].values,
            "market_type": "moneyline",
            "odds_source": "spread_proxy",
            "edge": ml_edge.values,
            "side": np.where(predicted_home_win, "HOME", "AWAY"),
            "bet_won": bet_won,
        })
        ledger_parts.append(ml)

    if not ledger_parts:
        return pd.DataFrame(columns=_COLS)
    return pd.concat(ledger_parts, ignore_index=True)[_COLS]


def _build_edge_band_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Aggregate bet_ledger into edge-band win-rate statistics."""
    _COLS = ["variant_id", "market_type", "edge_band", "bets_graded", "bets_won", "win_rate"]
    if ledger.empty:
        return pd.DataFrame(columns=_COLS)

    df = ledger[ledger["bet_won"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=_COLS)

    df["edge_band"] = pd.cut(
        pd.to_numeric(df["edge"], errors="coerce"),
        bins=_EDGE_BAND_BINS,
        labels=_EDGE_BAND_LABELS,
        right=True,
    )
    grp = df.groupby(["variant_id", "market_type", "edge_band"], observed=False)
    agg = grp.agg(
        bets_graded=("bet_won", "count"),
        bets_won=("bet_won", lambda x: int((pd.to_numeric(x, errors="coerce") == 1).sum())),
    ).reset_index()
    agg["win_rate"] = (
        agg["bets_won"] / agg["bets_graded"].clip(lower=1)
    ).round(4)
    return agg[_COLS]


def _build_kelly_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    """Compute Kelly criterion fractions per variant and market type."""
    _COLS = ["variant_id", "market_type", "kelly_fraction", "avg_edge", "win_rate", "bets_graded"]
    if ledger.empty:
        return pd.DataFrame(columns=_COLS)

    df = ledger[ledger["bet_won"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=_COLS)

    grp = df.groupby(["variant_id", "market_type"], observed=True)
    agg = grp.agg(
        bets_graded=("bet_won", "count"),
        bets_won=("bet_won", lambda x: int((pd.to_numeric(x, errors="coerce") == 1).sum())),
        avg_edge=("edge", lambda x: round(float(pd.to_numeric(x, errors="coerce").mean()), 4)),
    ).reset_index()
    agg["win_rate"] = (agg["bets_won"] / agg["bets_graded"].clip(lower=1)).round(4)
    # Kelly fraction using even-money (1:1) approximation: f* = 2p - 1, clipped at 0.
    # This is a simplification; typical sports-book ATS/totals odds are ~-110 (0.909:1),
    # but even-money provides a conservative upper-bound estimate for unit sizing.
    agg["kelly_fraction"] = (2 * agg["win_rate"] - 1).clip(lower=0).round(4)
    return agg[_COLS]


def run_variant_backtest(historical_games: pd.DataFrame) -> BacktestArtifacts:
    _SCORECARD_COLS = [
        "variant_id",
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
    ]
    _EMPTY_LEDGER_COLS = [
        "variant_id", "game_id", "game_datetime_utc",
        "market_type", "odds_source", "edge", "side", "bet_won",
    ]
    _EMPTY_EDGE_COLS = ["variant_id", "market_type", "edge_band", "bets_graded", "bets_won", "win_rate"]
    _EMPTY_KELLY_COLS = ["variant_id", "market_type", "kelly_fraction", "avg_edge", "win_rate", "bets_graded"]

    if historical_games.empty:
        return BacktestArtifacts(
            scorecard=pd.DataFrame(columns=_SCORECARD_COLS),
            edge_band_summary=pd.DataFrame(columns=_EMPTY_EDGE_COLS),
            bet_ledger=pd.DataFrame(columns=_EMPTY_LEDGER_COLS),
            kelly_summary=pd.DataFrame(columns=_EMPTY_KELLY_COLS),
        )

    df = historical_games.copy()
    df = df.sort_values("game_datetime_utc", kind="mergesort").reset_index(drop=True)
    min_train = max(120, int(len(df) * 0.7))
    if min_train >= len(df):
        min_train = max(120, len(df) - 1)
    windows = [(np.arange(0, min_train), np.arange(min_train, len(df)))]

    rows: list[dict[str, object]] = []
    ledger_parts: list[pd.DataFrame] = []
    for variant in variant_matrix():
        scored_parts: list[pd.DataFrame] = []
        for train_idx, test_idx in windows:
            train = df.iloc[train_idx].copy()
            test = df.iloc[test_idx].copy()
            if train.empty or test.empty:
                continue
            scored, _ = _apply_variant(train, test, variant)
            scored_parts.append(scored)

        scored_all = pd.concat(scored_parts, ignore_index=True) if scored_parts else pd.DataFrame()
        metrics = _compute_metrics(scored_all)
        rows.append(
            {
                "variant_id": variant.id,
                "backbone": variant.backbone,
                "win_prob_approach": variant.win_prob,
                "mc_mode": variant.mc_mode,
                "situational_on": bool(variant.situational_on),
                **metrics,
            }
        )
        if not scored_all.empty:
            ledger_parts.append(_build_bet_ledger(variant.id, scored_all))

    out = pd.DataFrame(rows)
    out = out.sort_values(["ats_accuracy", "winner_accuracy", "spread_mae"], ascending=[False, False, True], na_position="last")
    scorecard = out.reset_index(drop=True)

    ledger = pd.concat(ledger_parts, ignore_index=True) if ledger_parts else pd.DataFrame(columns=_EMPTY_LEDGER_COLS)
    edge_band_summary = _build_edge_band_summary(ledger)
    kelly_summary = _build_kelly_summary(ledger)

    return BacktestArtifacts(
        scorecard=scorecard,
        edge_band_summary=edge_band_summary,
        bet_ledger=ledger,
        kelly_summary=kelly_summary,
    )
