#!/usr/bin/env python3
"""
backtest_kelly.py

Simulates fractional Kelly unit sizing on historical games where we have:
  • ESPN market spread   (espn_spread)
  • Model prediction     (pred_spread)
  • CAGE signal          (cage_em_diff)
  • Trend signal         (net_rtg_trend_delta)
  • ATS outcome          (home_covered_ats)

Tracks cumulative unit P/L, ROI, and frequency of each unit tier.
Answers: does Kelly weighting + signal stacking improve unit-weighted returns?

Outputs:
  data/backtest_kelly.md   — human-readable report
  data/backtest_kelly.csv  — game-level results with kelly_units + unit_pnl
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Import shared Kelly logic
sys.path.insert(0, str(Path(__file__).parent))
from kelly import kelly_units, is_elite_spot, EDGE_MIN

_TRAIN_PATH = Path("data/backtest_training_data.csv")
_OUT_MD     = Path("data/backtest_kelly.md")
_OUT_CSV    = Path("data/backtest_kelly.csv")

_CAGE_EM_MIN     = 3.0
_TREND_DELTA_MIN = 1.5
_JUICE           = 110.0


def _unit_pnl(units: float, covered: int) -> float:
    """Unit P&L at -110: win → +units*(100/110), loss → -units."""
    if units == 0.0:
        return 0.0
    if covered:
        return round(units * (100.0 / _JUICE), 4)
    return -units


def _roi_flat(wins: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round((wins * (100 / _JUICE) - (total - wins)) / total * 100, 1)


def _validate_model_sign(df: pd.DataFrame) -> int:
    """Determine which sign of (pred_spread - espn_spread) predicts home covers."""
    edge = df["pred_spread"] - df["espn_spread"]
    acc_pos = ((edge > 0) == (df["home_covered_ats"] == 1)).mean()
    acc_neg = ((edge < 0) == (df["home_covered_ats"] == 1)).mean()
    sign = 1 if acc_pos >= acc_neg else -1
    print(f"[INFO] Model sign: {sign:+d}  (acc_pos={acc_pos:.3f}, acc_neg={acc_neg:.3f})")
    return sign


def _validate_trend_sign(df: pd.DataFrame) -> int:
    """Determine which sign of net_rtg_trend_delta predicts home covers."""
    trend = df["net_rtg_trend_delta"]
    acc_pos = ((trend > 0) == (df["home_covered_ats"] == 1)).mean()
    acc_neg = ((trend < 0) == (df["home_covered_ats"] == 1)).mean()
    sign = 1 if acc_pos >= acc_neg else -1
    print(f"[INFO] Trend sign: {sign:+d}  (acc_pos={acc_pos:.3f}, acc_neg={acc_neg:.3f})")
    return sign


def _tier_label(u: float) -> str:
    if u == 0.0:
        return "0u (no bet)"
    if u <= 0.5:
        return "0.5u"
    if u <= 1.0:
        return "1u"
    if u <= 1.5:
        return "1.5u"
    if u <= 2.0:
        return "2u"
    if u <= 2.5:
        return "2.5u"
    if u <= 3.0:
        return "3u"
    if u <= 3.5:
        return "3.5u"
    if u <= 4.0:
        return "4u"
    if u <= 4.5:
        return "4.5u"
    return "5u"


def main() -> int:
    if not _TRAIN_PATH.exists():
        print(f"[ERROR] {_TRAIN_PATH} not found")
        return 1

    df = pd.read_csv(_TRAIN_PATH, low_memory=False)
    print(f"[INFO] Loaded {len(df)} rows")

    for col in ["espn_spread", "pred_spread", "cage_em_diff",
                "net_rtg_trend_delta", "home_covered_ats"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["espn_spread", "pred_spread", "cage_em_diff",
                            "net_rtg_trend_delta", "home_covered_ats"]).copy()
    print(f"[INFO] After null-drop: {len(df)} games with all signals")

    if len(df) < 30:
        print("[STOP] Too few games for Kelly backtest")
        return 1

    # ── Sign conventions ──────────────────────────────────────────────────────
    model_sign = _validate_model_sign(df)
    trend_sign = _validate_trend_sign(df)

    df["model_edge_raw"] = model_sign * (df["pred_spread"] - df["espn_spread"])
    df["trend_raw"]      = trend_sign * df["net_rtg_trend_delta"]
    df["cage_raw"]       = df["cage_em_diff"]   # positive = home is better quality

    # Model-favored team covers when model_edge_raw > 0 → home_covered = 1
    df["model_picks_home"] = df["model_edge_raw"] > 0
    df["model_correct"]    = df["model_picks_home"] == (df["home_covered_ats"] == 1)

    # Signal alignment flags
    df["cage_validates"] = df["cage_raw"].apply(
        lambda x: "CONFIRMS" if x >= _CAGE_EM_MIN else ("DIVERGES" if x <= -_CAGE_EM_MIN else "NEUTRAL")
    )
    # CAGE validates when it points same direction as model
    df["cage_confirms_model"] = (
        ((df["model_edge_raw"] > 0) & (df["cage_raw"] >= _CAGE_EM_MIN)) |
        ((df["model_edge_raw"] < 0) & (df["cage_raw"] <= -_CAGE_EM_MIN))
    )
    df["cage_diverges_model"] = (
        ((df["model_edge_raw"] > 0) & (df["cage_raw"] <= -_CAGE_EM_MIN)) |
        ((df["model_edge_raw"] < 0) & (df["cage_raw"] >= _CAGE_EM_MIN))
    )
    df["cage_val_str"] = df.apply(
        lambda r: "CONFIRMS" if r["cage_confirms_model"] else
                  ("DIVERGES" if r["cage_diverges_model"] else "NEUTRAL"),
        axis=1,
    )

    df["trend_aligns"] = (
        (df["model_edge_raw"] > 0) & (df["trend_raw"] >= _TREND_DELTA_MIN)
    ) | (
        (df["model_edge_raw"] < 0) & (df["trend_raw"] <= -_TREND_DELTA_MIN)
    )

    # ── Kelly units ───────────────────────────────────────────────────────────
    df["kelly_units"] = df.apply(
        lambda r: kelly_units(r["model_edge_raw"], r["cage_val_str"], r["trend_aligns"]),
        axis=1,
    )
    df["is_elite"] = df.apply(
        lambda r: is_elite_spot(r["model_edge_raw"], r["cage_val_str"], r["trend_aligns"]),
        axis=1,
    )

    # Unit P&L
    df["unit_pnl"] = df.apply(
        lambda r: _unit_pnl(r["kelly_units"], int(r["model_correct"])),
        axis=1,
    )
    df["cumulative_units"] = df["unit_pnl"].cumsum()

    # ── Summary stats ─────────────────────────────────────────────────────────
    bet_df    = df[df["kelly_units"] > 0].copy()
    elite_df  = df[df["is_elite"]].copy()
    n_total   = len(df)
    n_bets    = len(bet_df)
    n_elite   = len(elite_df)
    total_units_wagered = bet_df["kelly_units"].sum()
    total_unit_pnl      = bet_df["unit_pnl"].sum()
    unit_roi  = total_unit_pnl / total_units_wagered * 100 if total_units_wagered > 0 else 0

    flat_wins = int(bet_df["model_correct"].sum())
    flat_pct  = flat_wins / n_bets * 100 if n_bets > 0 else 0
    flat_roi  = _roi_flat(flat_wins, n_bets)

    # ── Build report ──────────────────────────────────────────────────────────
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc: list[str] = []
    doc.append("# Fractional Kelly Backtest: Model + CAGE + Trend")
    doc.append(f"_Generated {generated}_\n")
    doc.append(
        f"**Dataset**: {n_total} games with all signals  \n"
        f"**Bets placed**: {n_bets} / {n_total} (edge ≥ {EDGE_MIN} pts)  \n"
        f"**Elite spots (5u)**: {n_elite} total  \n"
        f"**Unit scale**: 0.5u – 5u (signal-weighted fractional Kelly)\n"
    )

    doc.append("---\n")
    doc.append("## 1. Overall Performance\n")
    doc.append("| Metric | Flat betting | Kelly-weighted |")
    doc.append("|--------|-------------|----------------|")
    doc.append(f"| Bets | {n_bets} | {n_bets} |")
    doc.append(f"| Win rate | {flat_pct:.1f}% | {flat_pct:.1f}% |")
    doc.append(f"| ROI | {flat_roi:+.1f}% | {unit_roi:+.1f}% |")
    doc.append(f"| Total units wagered | — | {total_units_wagered:.1f}u |")
    doc.append(f"| Net unit P&L | — | {total_unit_pnl:+.1f}u |")

    doc.append("\n---\n")
    doc.append("## 2. Performance by Kelly Unit Tier\n")
    doc.append("| Tier | Bets | Win% | Unit P&L | ROI | Avg edge |")
    doc.append("|------|------|------|----------|-----|----------|")

    for tier_val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        sub = bet_df[bet_df["kelly_units"] == tier_val]
        if len(sub) == 0:
            continue
        wins = int(sub["model_correct"].sum())
        pct  = wins / len(sub) * 100
        pnl  = sub["unit_pnl"].sum()
        roi  = pnl / (sub["kelly_units"].sum()) * 100
        avg_edge = sub["model_edge_raw"].abs().mean()
        label = _tier_label(tier_val)
        doc.append(f"| {label} | {len(sub)} | {pct:.1f}% | {pnl:+.1f}u | {roi:+.1f}% | {avg_edge:.1f} pts |")

    doc.append("\n---\n")
    doc.append("## 3. Elite Spots (5u plays)\n")
    doc.append(
        f"**Criteria**: edge ≥ 6 pts + CAGE CONFIRMS + Trend aligns  \n"
        f"**Count**: {n_elite} games ({n_elite/n_total*100:.1f}% of sample)\n"
    )
    if n_elite > 0:
        e_wins = int(elite_df["model_correct"].sum())
        e_pct  = e_wins / n_elite * 100
        e_pnl  = elite_df["unit_pnl"].sum()
        e_roi  = _roi_flat(e_wins, n_elite)
        doc.append(f"| Metric | Value |")
        doc.append(f"|--------|-------|")
        doc.append(f"| Win rate | {e_pct:.1f}% |")
        doc.append(f"| Unit P&L | {e_pnl:+.1f}u |")
        doc.append(f"| ROI (flat) | {e_roi:+.1f}% |")
        doc.append(f"| Avg edge | {elite_df['model_edge_raw'].abs().mean():.1f} pts |")
    else:
        doc.append("_No elite spots in this dataset._")

    doc.append("\n---\n")
    doc.append("## 4. Signal Stack Comparison (unit-weighted)\n")
    doc.append("| Cohort | Bets | Win% | Unit P&L | Unit ROI |")
    doc.append("|--------|------|------|----------|----------|")

    cohorts = [
        ("Model only (all edges)",       bet_df),
        ("+ CAGE CONFIRMS",              bet_df[bet_df["cage_confirms_model"]]),
        ("+ Trend aligns",               bet_df[bet_df["trend_aligns"]]),
        ("+ CAGE + Trend",               bet_df[bet_df["cage_confirms_model"] & bet_df["trend_aligns"]]),
        ("CAGE DIVERGES (red flag)",     bet_df[bet_df["cage_diverges_model"]]),
    ]
    for label, sub in cohorts:
        if len(sub) == 0:
            doc.append(f"| {label} | 0 | — | — | — |")
            continue
        wins = int(sub["model_correct"].sum())
        pct  = wins / len(sub) * 100
        pnl  = sub["unit_pnl"].sum()
        wag  = sub["kelly_units"].sum()
        roi  = pnl / wag * 100 if wag > 0 else 0
        doc.append(f"| {label} | {len(sub)} | {pct:.1f}% | {pnl:+.1f}u | {roi:+.1f}% |")

    doc.append("\n---\n")
    doc.append("## 5. Edge Tier × Signal Stack\n")
    doc.append("| Edge | n | All win% | All P&L | +Both signals win% | +Both P&L |")
    doc.append("|------|---|----------|---------|--------------------|-----------|")
    for lo, hi, label in [(2,4,"2–4 pts"),(4,6,"4–6 pts"),(6,8,"6–8 pts"),(8,999,"8+ pts")]:
        sub = bet_df[bet_df["model_edge_raw"].abs().between(lo, hi - 0.01 if hi < 999 else hi)]
        both = sub[sub["cage_confirms_model"] & sub["trend_aligns"]]
        if len(sub) < 3:
            doc.append(f"| {label} | {len(sub)} | — | — | — | — |")
            continue
        def _pct(s: pd.DataFrame) -> str:
            if len(s) < 3:
                return f"—(n={len(s)})"
            return f"{s['model_correct'].mean()*100:.1f}%"
        def _pnl(s: pd.DataFrame) -> str:
            if len(s) < 3:
                return "—"
            return f"{s['unit_pnl'].sum():+.1f}u"
        doc.append(f"| {label} | {len(sub)} | {_pct(sub)} | {_pnl(sub)} | {_pct(both)} | {_pnl(both)} |")

    doc.append("\n---\n")
    doc.append("## 6. Unit Sizing Rules (reference)\n")
    doc.append("""
| Edge (pts) | Base | +CAGE CONFIRMS | +Trend | Both | CAGE DIVERGES |
|------------|------|----------------|--------|------|---------------|
| 2–4        | 1u   | 1.5u           | 1.5u   | 2u   | 0.5u          |
| 4–6        | 2u   | 2.5u           | 2.5u   | 3u   | 1.5u          |
| 6–8        | 3u   | 3.5u           | 3.5u   | 4u   | 2.5u          |
| 8–10       | 4u   | 4.5u           | 4.5u   | 5u ★ | 3.5u         |
| 10+        | 4u   | 4.5u           | 4.5u   | 5u ★ | 3.5u         |

★ = Elite spot. 5u requires edge ≥ 6 pts + CAGE CONFIRMS + Trend aligns.
Max 5u. Min 0.5u (even with CAGE DIVERGES penalty applied).
""")

    md_text = "\n".join(doc)
    _OUT_MD.write_text(md_text, encoding="utf-8")
    print(f"[OK] {_OUT_MD}")

    # ── Game-level CSV ────────────────────────────────────────────────────────
    save_cols = [
        "game_date", "home_team", "away_team",
        "espn_spread", "pred_spread", "cage_em_diff", "net_rtg_trend_delta",
        "home_covered_ats", "model_edge_raw", "model_picks_home", "model_correct",
        "cage_val_str", "trend_aligns", "kelly_units", "is_elite",
        "unit_pnl", "cumulative_units",
    ]
    available = [c for c in save_cols if c in df.columns]
    df[available].to_csv(_OUT_CSV, index=False)
    print(f"[OK] {_OUT_CSV} ({len(df)} rows, {n_bets} bets, "
          f"{total_unit_pnl:+.1f}u P&L, {unit_roi:+.1f}% unit ROI)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
