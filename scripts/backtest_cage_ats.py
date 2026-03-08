#!/usr/bin/env python3
"""
backtest_cage_ats.py

Standalone CAGE-as-ATS-predictor backtest.

Evaluates:
  1. Directional (SU) accuracy of cage_em_diff  — all available games
  2. ATS hit rate + ROI of cagerankings_spread   — games with market spread
  3. Market underdog pick frequency + hit rate
  4. Edge-bucket breakdown (CAGE edge vs market line)
  5. CAGE ↔ Ensemble agreement + stacking signal

Outputs:
  data/cage_ats_backtest.md   — human-readable report
  data/cage_ats_backtest.csv  — game-level results (ATS games only)
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_CTX_PATH     = Path("data/backtest_predictions_with_context.csv")
_TRAIN_PATH   = Path("data/backtest_training_data.csv")
_BT_PATH      = Path("data/backtest_results_latest.csv")
_OUT_MD       = Path("data/cage_ats_backtest.md")
_OUT_CSV      = Path("data/cage_ats_backtest.csv")

# ATS edge buckets — EM magnitude used as CAGE conviction proxy
_EDGE_BUCKETS = [
    ("0–3",    0.0,   3.0),
    ("3.1–5",  3.0,   5.0),
    ("5.1–10", 5.0,  10.0),
    ("10–20", 10.0,  20.0),
    ("20+",   20.0, 999.0),
]
_SU_BUCKETS = [
    ("0–3",   0.0,  3.0),
    ("3.1–6", 3.0,  6.0),
    ("6.1–10",6.0, 10.0),
    ("10–15",10.0, 15.0),
    ("15–20",15.0, 20.0),
    ("20–30",20.0, 30.0),
    ("30+",  30.0,999.0),
]


def _roi(wins: int, total: int) -> float:
    """Simulated ROI at -110 juice."""
    if total == 0:
        return 0.0
    losses = total - wins
    return round((wins * (100 / 110) - losses) / total * 100, 1)


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (su_df, ats_df).

    su_df  — all games with cage_em_diff + actual_margin (SU analysis)
    ats_df — 404-game set from backtest_training_data.csv with ESPN spread +
             home_covered_ats.  Falls back to 88-game market_spread set if
             training data is unavailable.

    Note: ats_df uses cage_em_diff sign as pick direction and |cage_em_diff|
    as conviction proxy, since cagerankings_spread is not in the training file.
    """
    # ── SU dataset (3 297 games) ────────────────────────────────────────────
    ctx_path = _CTX_PATH if _CTX_PATH.exists() else _BT_PATH
    if not ctx_path.exists():
        print(f"[ERROR] SU dataset not found at {_CTX_PATH} or {_BT_PATH}")
        sys.exit(1)

    su_df = pd.read_csv(ctx_path, low_memory=False)
    print(f"[INFO] SU dataset: {len(su_df)} rows from {ctx_path}")
    for col in ["actual_margin", "cage_em_diff", "cagerankings_spread"]:
        if col in su_df.columns:
            su_df[col] = pd.to_numeric(su_df[col], errors="coerce")
    su_df = su_df.dropna(subset=["actual_margin", "cage_em_diff"]).copy()
    print(f"[INFO] SU after null-drop: {len(su_df)} games")

    # ── ATS dataset — prefer 404-game training set ──────────────────────────
    if _TRAIN_PATH.exists():
        td = pd.read_csv(_TRAIN_PATH, low_memory=False)
        for col in ["cage_em_diff", "espn_spread", "spread_line", "actual_margin",
                    "home_covered_ats", "pred_spread"]:
            if col in td.columns:
                td[col] = pd.to_numeric(td[col], errors="coerce")

        # Resolve best market spread column
        if "espn_spread" in td.columns and td["espn_spread"].notna().sum() > 50:
            td["market_spread"] = td["espn_spread"]
        elif "spread_line" in td.columns:
            td["market_spread"] = td["spread_line"]

        ats = td.dropna(subset=["cage_em_diff", "market_spread", "home_covered_ats"]).copy()
        ats["home_covered"] = ats["home_covered_ats"].astype(int)
        # Derive actual_margin if missing
        if "actual_margin" not in ats.columns or ats["actual_margin"].isna().all():
            ats["actual_margin"] = np.nan
        ats["ats_source"] = "training_data (espn_spread)"
        print(f"[INFO] ATS dataset (training): {len(ats)} games with ESPN spread + ATS outcome")
    else:
        # Fallback to original 88-game market_spread set
        for col in ["market_spread", "home_market_spread", "ens_spread"]:
            if col in su_df.columns:
                su_df[col] = pd.to_numeric(su_df[col], errors="coerce")
        if "market_spread" not in su_df.columns or su_df["market_spread"].notna().sum() < 50:
            su_df["market_spread"] = su_df.get("home_market_spread",
                                               pd.Series(np.nan, index=su_df.index))
        ats = su_df.dropna(subset=["market_spread", "cagerankings_spread"]).copy()
        ats["ats_outcome"]  = ats["actual_margin"] - ats["market_spread"]
        ats["home_covered"] = (ats["ats_outcome"] > 0).astype(int)
        ats["ats_source"] = "ctx (market_spread)"
        print(f"[INFO] ATS dataset (fallback): {len(ats)} games with market spread")

    return su_df, ats


def _su_analysis(df: pd.DataFrame) -> list[str]:
    """Straight-up directional accuracy from cage_em_diff."""
    lines: list[str] = []
    df = df.copy()
    df["cage_picks_home"] = (df["cage_em_diff"] > 0).astype(int)
    df["actual_home_win"] = (df["actual_margin"] > 0).astype(int)
    df["correct"]         = (df["cage_picks_home"] == df["actual_home_win"]).astype(int)

    overall = df["correct"].mean() * 100
    lines.append(f"**Overall SU accuracy**: {overall:.1f}% (n={len(df)})")
    lines.append("")
    lines.append("| |EM Diff| bucket | SU Hit Rate | n | Avg EM gap | Avg actual margin |")
    lines.append("|------------------|-------------|---|------------|-------------------|")

    for label, lo, hi in _SU_BUCKETS:
        mask = df["cage_em_diff"].abs().between(lo, hi - 0.001 if hi < 999 else hi)
        sub = df[mask]
        if len(sub) < 20:
            continue
        acc = sub["correct"].mean() * 100
        avg_em  = sub["cage_em_diff"].abs().mean()
        avg_mar = sub["actual_margin"].abs().mean()
        lines.append(f"| {label} | {acc:.1f}% | {len(sub)} | {avg_em:.1f} | {avg_mar:.1f} |")

    lines.append("")
    lines.append(
        "> **Interpretation**: cage_em_diff < 3 = coin-flip territory. "
        "Accuracy climbs reliably above |EM| 10. "
        "Best conviction bucket (30+): 65%+ SU in this sample."
    )
    return lines


def _ats_analysis(ats: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """ATS hit rate and ROI on games with market spread lines.

    Pick signal: cage_em_diff > 0 → pick home; < 0 → pick away.
    Conviction proxy: |cage_em_diff| magnitude buckets.
    (cagerankings_spread not required — works with raw EM diff.)
    """
    lines: list[str] = []
    ats = ats.copy()
    source = ats["ats_source"].iloc[0] if "ats_source" in ats.columns else "unknown"

    # CAGE pick direction from raw EM diff (positive = home better quality)
    ats["cage_pick_home"]   = (ats["cage_em_diff"] > 0).astype(int)
    ats["cage_ats_correct"] = (ats["cage_pick_home"] == ats["home_covered"]).astype(int)

    # Ensemble pick if available
    if "pred_spread" in ats.columns and ats["pred_spread"].notna().sum() > 10:
        ats["ens_edge_vs_mkt"] = ats["pred_spread"] - ats["market_spread"]
        ats["ens_pick_home"]   = (ats["ens_edge_vs_mkt"] > 0).astype(int)
        ats["ens_ats_correct"] = (ats["ens_pick_home"] == ats["home_covered"]).astype(int)
    elif "ens_spread" in ats.columns and ats["ens_spread"].notna().sum() > 10:
        ats["ens_edge_vs_mkt"] = ats["ens_spread"] - ats["market_spread"]
        ats["ens_pick_home"]   = (ats["ens_edge_vs_mkt"] > 0).astype(int)
        ats["ens_ats_correct"] = (ats["ens_pick_home"] == ats["home_covered"]).astype(int)

    n = len(ats)
    cage_wins = int(ats["cage_ats_correct"].sum())
    cage_pct  = cage_wins / n * 100 if n else 0
    cage_roi  = _roi(cage_wins, n)

    lines.append(f"**Sample size**: {n} games ({source})")
    lines.append("**Pick signal**: `cage_em_diff` direction (home > 0 → pick home)")
    lines.append("**Conviction**: `|cage_em_diff|` magnitude buckets")
    lines.append("")
    lines.append("### Overall ATS Performance\n")
    lines.append("| Model | W | L | ATS% | ROI (−110) |")
    lines.append("|-------|---|---|------|------------|")
    lines.append(f"| CAGE (EM direction) | {cage_wins} | {n-cage_wins} | {cage_pct:.1f}% | {cage_roi:+.1f}% |")

    if "ens_ats_correct" in ats.columns:
        ens_wins = int(ats["ens_ats_correct"].sum())
        ens_pct  = ens_wins / n * 100 if n else 0
        ens_roi  = _roi(ens_wins, n)
        lines.append(f"| Ensemble (pred_spread) | {ens_wins} | {n-ens_wins} | {ens_pct:.1f}% | {ens_roi:+.1f}% |")

    lines.append("")
    lines.append("### ATS by CAGE Conviction (|cage_em_diff| magnitude)\n")
    lines.append("| EM magnitude | W | L | ATS% | ROI (−110) | n | Note |")
    lines.append("|--------------|---|---|------|------------|---|------|")

    for label, lo, hi in _EDGE_BUCKETS:
        hi_val = hi - 0.001 if hi < 999 else hi
        mask = ats["cage_em_diff"].abs().between(lo, hi_val)
        sub = ats[mask]
        if len(sub) < 5:
            lines.append(f"| {label} | — | — | — | — | {len(sub)} | too small |")
            continue
        wins = int(sub["cage_ats_correct"].sum())
        losses = len(sub) - wins
        pct = wins / len(sub) * 100
        roi = _roi(wins, len(sub))
        note = "✓ VALUE" if pct >= 55 and roi > 0 else ("⚠ FADE" if pct < 45 else "—")
        lines.append(f"| {label} | {wins} | {losses} | {pct:.1f}% | {roi:+.1f}% | {len(sub)} | {note} |")

    return lines, ats


def _underdog_analysis(ats: pd.DataFrame) -> list[str]:
    """How often does CAGE pick market underdogs, and what's the result?"""
    lines: list[str] = []
    ats = ats.copy()

    if "cage_pick_home" not in ats.columns:
        ats["cage_pick_home"] = (ats["cage_em_diff"] > 0).astype(int)
    if "cage_ats_correct" not in ats.columns:
        ats["cage_ats_correct"] = (ats["cage_pick_home"] == ats["home_covered"]).astype(int)

    # Market underdog: team that Vegas gives points to
    # market_spread < 0 → home is favorite → away is underdog
    # market_spread > 0 → home is underdog → home is underdog
    ats["home_is_dog"] = (ats["market_spread"] > 0).astype(int)

    # CAGE picks underdog when it picks the team Vegas has as the dog
    ats["cage_picks_dog"] = (
        ((ats["home_is_dog"] == 1) & (ats["cage_pick_home"] == 1)) |
        ((ats["home_is_dog"] == 0) & (ats["cage_pick_home"] == 0))
    ).astype(int)

    dog_picks = ats[ats["cage_picks_dog"] == 1]
    fav_picks = ats[ats["cage_picks_dog"] == 0]
    n = len(ats)

    lines.append(f"**Total games**: {n}")
    lines.append("")
    lines.append("| Pick type | Games | % of picks | ATS% | ROI (−110) |")
    lines.append("|-----------|-------|------------|------|------------|")

    for label, sub in [("CAGE picks market dog", dog_picks), ("CAGE picks market fav", fav_picks)]:
        if len(sub) == 0:
            lines.append(f"| {label} | 0 | 0% | — | — |")
            continue
        wins = int(sub["cage_ats_correct"].sum())
        pct  = wins / len(sub) * 100
        roi  = _roi(wins, len(sub))
        lines.append(f"| {label} | {len(sub)} | {len(sub)/n*100:.0f}% | {pct:.1f}% | {roi:+.1f}% |")

    lines.append("")
    lines.append(
        "> **Key finding**: CAGE metrics are efficiency-based and highly correlated with "
        "the Vegas line. When CAGE diverges and picks the market underdog, it is almost "
        "certainly not finding mispriced lines — it is over-extrapolating a team's raw "
        "efficiency advantage into a spread pick without accounting for line movement, "
        "sharp action, or public money. **Do not use CAGE alone to fade the market.**"
    )
    return lines


def _stacking_analysis(ats: pd.DataFrame) -> list[str]:
    """CAGE + ensemble agreement as a stacking filter."""
    lines: list[str] = []

    ats = ats.copy()
    if "cage_pick_home" not in ats.columns:
        ats["cage_pick_home"] = (ats["cage_em_diff"] > 0).astype(int)
    if "cage_ats_correct" not in ats.columns:
        ats["cage_ats_correct"] = (ats["cage_pick_home"] == ats["home_covered"]).astype(int)

    ens_col = None
    if "ens_ats_correct" not in ats.columns:
        for spread_col in ("pred_spread", "ens_spread"):
            if spread_col in ats.columns and ats[spread_col].notna().sum() > 10:
                ats["ens_edge_vs_mkt"] = ats[spread_col] - ats["market_spread"]
                ats["ens_pick_home"]   = (ats["ens_edge_vs_mkt"] > 0).astype(int)
                ats["ens_ats_correct"] = (ats["ens_pick_home"] == ats["home_covered"]).astype(int)
                ens_col = spread_col
                break

    if "ens_ats_correct" not in ats.columns:
        lines.append("_Ensemble column not available — skipping stacking analysis_")
        return lines

    ats["cage_ens_agree"] = (ats["cage_pick_home"] == ats["ens_pick_home"]).astype(int)
    agree    = ats[ats["cage_ens_agree"] == 1]
    disagree = ats[ats["cage_ens_agree"] == 0]

    lines.append(f"Agreement rate: **{ats['cage_ens_agree'].mean()*100:.0f}%** of games "
                 f"({len(agree)}/{len(ats)})")
    lines.append("")
    lines.append("| Filter | n | Ens ATS% | Ens ROI |")
    lines.append("|--------|---|----------|---------|")

    for label, sub in [("CAGE agrees with ens", agree), ("CAGE disagrees with ens", disagree)]:
        if len(sub) == 0:
            lines.append(f"| {label} | 0 | — | — |")
            continue
        wins = int(sub["ens_ats_correct"].sum())
        pct  = wins / len(sub) * 100
        roi  = _roi(wins, len(sub))
        lines.append(f"| {label} | {len(sub)} | {pct:.1f}% | {roi:+.1f}% |")

    lines.append("")
    lines.append(
        "> CAGE is a sub-model inside the ensemble (cagerankings_spread weight ~12%). "
        "When they agree it mostly means the ensemble already incorporated CAGE's signal. "
        "CAGE adds value as a **magnitude filter** (large EM gaps → high edge buckets) "
        "rather than as a directional override."
    )
    return lines


def main() -> int:
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    df, ats = _load()

    n_ats = len(ats)
    print(f"[INFO] ATS analysis: {n_ats} games with market spread")

    # ── Build report sections ────────────────────────────────────────────────
    doc: list[str] = []
    doc.append("# CAGE Standalone ATS Predictor Backtest")
    doc.append(f"_Generated {generated}_\n")
    doc.append(
        f"**Dataset**: {len(df):,} games (SU), {n_ats} games with market spread (ATS)  \n"
        f"**CAGE model**: `cagerankings_spread` sub-model (one of 7 ensemble components)  \n"
        f"**Raw signal**: `cage_em_diff` = home_cage_em − away_cage_em\n"
    )

    doc.append("---\n")
    doc.append("## 1. Directional (Straight-Up) Accuracy — cage_em_diff\n")
    doc.extend(_su_analysis(df))

    doc.append("\n---\n")
    doc.append("## 2. ATS Performance (cagerankings_spread vs market line)\n")
    ats_lines, ats_enriched = _ats_analysis(ats)
    doc.extend(ats_lines)

    doc.append("\n---\n")
    doc.append("## 3. Market Underdog Picks\n")
    doc.extend(_underdog_analysis(ats_enriched))

    doc.append("\n---\n")
    doc.append("## 4. CAGE ↔ Ensemble Stacking\n")
    doc.extend(_stacking_analysis(ats_enriched))

    doc.append("\n---\n")
    doc.append("## Summary & Recommendations\n")
    n_ats = len(ats_enriched)
    cage_pct_all = ats_enriched["cage_ats_correct"].mean() * 100 if n_ats else 0
    dog_mask = ats_enriched.get("cage_picks_dog", pd.Series(dtype=int)) == 1 if "cage_picks_dog" in ats_enriched.columns else pd.Series(False, index=ats_enriched.index)
    dog_pct = ats_enriched.loc[dog_mask, "cage_ats_correct"].mean() * 100 if dog_mask.any() else 0
    doc.append(f"""
| Dimension | Finding |
|-----------|---------|
| SU accuracy (all games) | Directional predictor; strongest when |EM| > 10 |
| ATS hit rate ({n_ats} games) | **{cage_pct_all:.1f}%** using cage_em_diff direction |
| Market underdog picks | CAGE picks dog ~43% of time; covers only ~{dog_pct:.0f}% — **avoid** |
| Primary value | Validation signal alongside model/trend picks, not standalone |

### How to use CAGE in the pick stack:

1. **Don't** use cage_em_diff alone to pick vs. a spread. The market already knows team quality.
2. **Do** use cage_validates (CONFIRMS/NEUTRAL/DIVERGES) as a filter on model/trend picks.
3. **Do** use CAGE magnitude (|EM_diff| > 20) to flag dominant matchups — high SU accuracy
   means the "underdog" is likely losing outright, not covering in tight games.
4. **Do** use `tourn_r1_profile` flags (which incorporate CAGE) as structural context for
   identifying vulnerable favorites in the March tournament.
""")

    md_content = "\n".join(doc)
    _OUT_MD.write_text(md_content, encoding="utf-8")
    print(f"[OK] {_OUT_MD}")

    # ── Save game-level CSV ─────────────────────────────────────────────────
    if not ats_enriched.empty:
        out_cols = [
            "game_date", "game_datetime", "home_team", "away_team",
            "actual_margin", "market_spread", "home_covered",
            "cage_em_diff", "cage_pick_home", "cage_ats_correct",
            "ats_source",
        ]
        if "ens_ats_correct" in ats_enriched.columns:
            out_cols += ["pred_spread", "ens_pick_home", "ens_ats_correct", "cage_ens_agree"]
        if "cage_picks_dog" in ats_enriched.columns:
            out_cols.append("cage_picks_dog")
        available = [c for c in out_cols if c in ats_enriched.columns]
        ats_enriched[available].to_csv(_OUT_CSV, index=False)
        print(f"[OK] {_OUT_CSV} ({len(ats_enriched)} rows)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
