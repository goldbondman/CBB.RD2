from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import sign_signal


BUCKETS = [
    "Base model only",
    "Situational signal only",
    "Monte Carlo signal only",
    "Base + Situational agree",
    "Base + Monte Carlo agree",
    "Base + Situational + Monte Carlo agree",
    "Base + Situational conflict",
    "Base + Monte Carlo conflict",
    "No actionable signal",
]


def assign_agreement_bucket(base: int, situ: int, mc: int) -> str:
    if base != 0 and situ == 0 and mc == 0:
        return "Base model only"
    if base == 0 and situ != 0 and mc == 0:
        return "Situational signal only"
    if base == 0 and situ == 0 and mc != 0:
        return "Monte Carlo signal only"
    if base != 0 and situ != 0 and mc == 0:
        return "Base + Situational agree" if base == situ else "Base + Situational conflict"
    if base != 0 and situ == 0 and mc != 0:
        return "Base + Monte Carlo agree" if base == mc else "Base + Monte Carlo conflict"
    if base != 0 and situ != 0 and mc != 0:
        if base == situ == mc:
            return "Base + Situational + Monte Carlo agree"
        if base == situ:
            return "Base + Situational agree"
        if base == mc:
            return "Base + Monte Carlo agree"
        if base != situ:
            return "Base + Situational conflict"
        if base != mc:
            return "Base + Monte Carlo conflict"
    return "No actionable signal"


def apply_agreement_layer(game_frame: pd.DataFrame) -> pd.DataFrame:
    out = game_frame.copy()

    edge_home = pd.to_numeric(out.get("edge_home"), errors="coerce").fillna(0.0)
    out["base_signal"] = edge_home.apply(sign_signal)
    out["situational_signal"] = pd.to_numeric(out.get("situational_signal"), errors="coerce").fillna(0).astype(int)
    out["mc_signal"] = pd.to_numeric(out.get("mc_signal"), errors="coerce").fillna(0).astype(int)

    out["agreement_bucket"] = [
        assign_agreement_bucket(int(b), int(s), int(m))
        for b, s, m in zip(out["base_signal"], out["situational_signal"], out["mc_signal"])
    ]
    return out


def summarize_agreement_buckets(scored_games: pd.DataFrame) -> pd.DataFrame:
    if scored_games.empty:
        return pd.DataFrame(
            {
                "agreement_bucket": BUCKETS,
                "sample_size": 0,
                "su_accuracy": np.nan,
                "ats_accuracy": np.nan,
                "avg_edge": np.nan,
                "confidence_mean": np.nan,
                "confidence_std": np.nan,
            }
        )

    df = scored_games.copy()

    if "su_correct" not in df.columns and {"predicted_winner_side", "actual_margin"}.issubset(df.columns):
        pred_home = df["predicted_winner_side"].astype(str).eq("HOME")
        df["su_correct"] = pred_home.eq(pd.to_numeric(df["actual_margin"], errors="coerce") > 0)

    if "ats_correct" not in df.columns and {"predicted_ats_side", "actual_margin", "market_spread"}.issubset(df.columns):
        home_cover = pd.to_numeric(df["actual_margin"], errors="coerce") + pd.to_numeric(df["market_spread"], errors="coerce") > 0
        pred_home_cover = df["predicted_ats_side"].astype(str).eq("HOME")
        df["ats_correct"] = pred_home_cover.eq(home_cover)

    grouped = df.groupby("agreement_bucket", dropna=False)
    out = grouped.agg(
        sample_size=("agreement_bucket", "size"),
        su_accuracy=("su_correct", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        ats_accuracy=("ats_correct", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        avg_edge=("edge_home", lambda s: float(pd.to_numeric(s, errors="coerce").abs().mean())),
        confidence_mean=("confidence_score", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        confidence_std=("confidence_score", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0))),
    ).reset_index()

    all_buckets = pd.DataFrame({"agreement_bucket": BUCKETS})
    out = all_buckets.merge(out, on="agreement_bucket", how="left")
    out["sample_size"] = out["sample_size"].fillna(0).astype(int)
    return out
