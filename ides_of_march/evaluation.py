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
            "winner_accuracy": np.nan,
            "ats_accuracy": np.nan,
            "calibration_brier": np.nan,
            "ats_edge_buckets": "{}",
            "situational_bucket_perf": "{}",
            "agreement_bucket_perf": "{}",
        }

    df = scored.copy()
    margin = pd.to_numeric(df.get("actual_margin"), errors="coerce")
    total = pd.to_numeric(df.get("actual_total"), errors="coerce")
    home_won = pd.to_numeric(df.get("home_won"), errors="coerce")
    market_spread = pd.to_numeric(df.get("market_spread"), errors="coerce")

    pred_margin = pd.to_numeric(df.get("projected_margin_home"), errors="coerce")
    pred_win = pd.to_numeric(df.get("win_prob_home"), errors="coerce")

    predicted_home_win = pred_win >= 0.5
    winner_accuracy = float((predicted_home_win == (home_won > 0.5)).mean()) if home_won.notna().any() else np.nan

    home_cover = (margin + market_spread) > 0
    predicted_home_cover = df.get("predicted_ats_side", pd.Series("HOME", index=df.index)).astype(str).eq("HOME")
    ats_accuracy = float((predicted_home_cover == home_cover).mean()) if margin.notna().any() and market_spread.notna().any() else np.nan

    spread_mae = float((pred_margin - margin).abs().mean()) if margin.notna().any() else np.nan
    brier = float(np.mean((pred_win - home_won) ** 2)) if home_won.notna().any() else np.nan

    edge_abs = pd.to_numeric(df.get("edge_home"), errors="coerce").abs()
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
        "winner_accuracy": winner_accuracy,
        "ats_accuracy": ats_accuracy,
        "calibration_brier": brier,
        "ats_edge_buckets": json.dumps(edge_perf),
        "situational_bucket_perf": json.dumps(situ_perf),
        "agreement_bucket_perf": json.dumps(agreement_payload),
    }


def run_variant_backtest(historical_games: pd.DataFrame) -> pd.DataFrame:
    if historical_games.empty:
        return pd.DataFrame(
            columns=[
                "variant_id",
                "sample_size",
                "spread_mae",
                "winner_accuracy",
                "ats_accuracy",
                "calibration_brier",
                "ats_edge_buckets",
                "situational_bucket_perf",
                "agreement_bucket_perf",
            ]
        )

    df = historical_games.copy()
    df = df.sort_values("game_datetime_utc", kind="mergesort").reset_index(drop=True)
    min_train = max(120, int(len(df) * 0.7))
    if min_train >= len(df):
        min_train = max(120, len(df) - 1)
    windows = [(np.arange(0, min_train), np.arange(min_train, len(df)))]

    rows: list[dict[str, object]] = []
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

    out = pd.DataFrame(rows)
    out = out.sort_values(["ats_accuracy", "winner_accuracy", "spread_mae"], ascending=[False, False, True], na_position="last")
    return out.reset_index(drop=True)
