#!/usr/bin/env python3
"""
backtest_alignment.py

3-way signal alignment study: Model vs Model+CAGE vs Model+Trend vs Model+CAGE+Trend.

Answers: When CAGE, Trend, and Model all agree, is ATS accuracy higher than
any two-signal or one-signal combination?

Signals:
  Model  — pred_spread vs espn_spread edge (sign validated empirically)
  CAGE   — cage_em_diff direction matches model pick direction
  Trend  — net_rtg_trend_delta direction matches model pick direction

Output:
  data/backtest_alignment.md   — human-readable report
  data/backtest_alignment.csv  — game-level results
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_TRAIN_PATH = Path("data/backtest_training_data.csv")
_OUT_MD     = Path("data/backtest_alignment.md")
_OUT_CSV    = Path("data/backtest_alignment.csv")

# Minimum edge thresholds to qualify as a "signal"
_MODEL_EDGE_MIN  = 1.0   # |pred_spread - espn_spread| > 1.0
_CAGE_EM_MIN     = 3.0   # |cage_em_diff| > 3 (above coin-flip zone)
_TREND_DELTA_MIN = 1.5   # |net_rtg_trend_delta| > 1.5 net-rtg/game


def _roi(wins: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round((wins * (100 / 110) - (total - wins)) / total * 100, 1)


def _fmt_row(label: str, sub: pd.DataFrame) -> str:
    n = len(sub)
    if n < 5:
        return f"| {label} | — | — | — | — | {n} |"
    wins = int(sub["home_covered_ats"].sum())
    pct  = wins / n * 100
    roi  = _roi(wins, n)
    note = "✓" if pct >= 55 else ("⚠" if pct < 45 else "")
    return f"| {label} | {wins} | {n-wins} | {pct:.1f}% | {roi:+.1f}% | {n} | {note} |"


def _validate_model_sign(df: pd.DataFrame) -> int:
    """Return +1 or -1 for the sign convention of pred_spread vs espn_spread.

    Tries both sign conventions and returns the one where model picks correct
    direction at > 50% (i.e. better than random).
    """
    # Convention A: model picks home when pred_spread - espn_spread > 0
    conv_a = (df["pred_spread"] - df["espn_spread"]).apply(np.sign)
    acc_a  = ((conv_a > 0) == (df["home_covered_ats"] == 1)).mean()

    # Convention B: model picks home when espn_spread - pred_spread > 0
    acc_b  = ((conv_a < 0) == (df["home_covered_ats"] == 1)).mean()

    sign = 1 if acc_a >= acc_b else -1
    best = max(acc_a, acc_b)
    print(f"[INFO] Model sign convention: {'+' if sign==1 else '-'}(pred-espn); "
          f"acc_a={acc_a:.3f} acc_b={acc_b:.3f} → using sign={sign} ({best:.3f} base accuracy)")
    return sign


def _validate_trend_sign(df: pd.DataFrame) -> int:
    """Return +1 or -1 for net_rtg_trend_delta sign convention."""
    conv = df["net_rtg_trend_delta"].apply(np.sign)
    acc_pos = ((conv > 0) == (df["home_covered_ats"] == 1)).mean()
    acc_neg = ((conv < 0) == (df["home_covered_ats"] == 1)).mean()
    sign = 1 if acc_pos >= acc_neg else -1
    best = max(acc_pos, acc_neg)
    print(f"[INFO] Trend sign: {'positive' if sign==1 else 'negative'} trend_delta → home cover; "
          f"acc_pos={acc_pos:.3f} acc_neg={acc_neg:.3f} → sign={sign} ({best:.3f})")
    return sign


def main() -> int:
    if not _TRAIN_PATH.exists():
        print(f"[ERROR] {_TRAIN_PATH} not found")
        return 1

    df = pd.read_csv(_TRAIN_PATH, low_memory=False)
    print(f"[INFO] Loaded {len(df)} rows from {_TRAIN_PATH}")

    for col in ["espn_spread", "pred_spread", "cage_em_diff",
                "net_rtg_trend_delta", "home_covered_ats", "actual_margin"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Require all four signals + outcome
    required = ["espn_spread", "pred_spread", "cage_em_diff",
                "net_rtg_trend_delta", "home_covered_ats"]
    df = df.dropna(subset=required).copy()
    print(f"[INFO] After null-drop: {len(df)} games with all signals")

    if len(df) < 50:
        print("[STOP] Insufficient data for alignment analysis")
        return 1

    # ── Sign validation ───────────────────────────────────────────────────────
    model_sign = _validate_model_sign(df)
    trend_sign = _validate_trend_sign(df)

    # ── Compute signal directions ─────────────────────────────────────────────
    # Raw edge: how much model diverges from market (home perspective)
    df["model_raw_edge"] = model_sign * (df["pred_spread"] - df["espn_spread"])
    # Positive = model likes home to cover; negative = model likes away

    df["cage_raw"]  = df["cage_em_diff"]          # positive = home better quality
    df["trend_raw"] = trend_sign * df["net_rtg_trend_delta"]  # positive = home trending better

    # Model has edge only when it exceeds the minimum threshold
    df["model_has_edge"] = df["model_raw_edge"].abs() >= _MODEL_EDGE_MIN
    df["model_picks_home"] = df["model_raw_edge"] > 0

    # Signals align with model when they point in the same direction
    df["cage_aligns"]  = (
        (df["cage_raw"].abs() >= _CAGE_EM_MIN) &
        (df["cage_raw"].apply(np.sign) == df["model_raw_edge"].apply(np.sign))
    )
    df["trend_aligns"] = (
        (df["trend_raw"].abs() >= _TREND_DELTA_MIN) &
        (df["trend_raw"].apply(np.sign) == df["model_raw_edge"].apply(np.sign))
    )

    # Outcome: did the model-favored team cover?
    df["model_correct"] = (df["model_picks_home"] == (df["home_covered_ats"] == 1))

    # ── Filter to games where model has an edge ───────────────────────────────
    base = df[df["model_has_edge"]].copy()
    n_base = len(base)
    print(f"[INFO] Games with model edge (>={_MODEL_EDGE_MIN}): {n_base}")

    # ── Alignment cohorts ─────────────────────────────────────────────────────
    m_only    = base
    m_cage    = base[base["cage_aligns"]]
    m_trend   = base[base["trend_aligns"]]
    m_c_t     = base[base["cage_aligns"] & base["trend_aligns"]]
    # Also: strict versions with larger thresholds
    m_cage_hi = base[base["cage_raw"].abs() >= 10.0]  # CAGE conviction high
    m_c_t_hi  = base[(base["cage_raw"].abs() >= 10.0) & base["trend_aligns"]]

    # ── Build report ─────────────────────────────────────────────────────────
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc: list[str] = []
    doc.append("# Signal Alignment Backtest: Model + CAGE + Trend")
    doc.append(f"_Generated {generated}_\n")
    doc.append(
        f"**Dataset**: {n_base} games with model edge (|edge| ≥ {_MODEL_EDGE_MIN} pts)  \n"
        f"**Source**: `backtest_training_data.csv` ({len(df)} games with all signals)  \n"
        f"**Signals**: Model (`pred_spread` vs `espn_spread`), CAGE (`cage_em_diff`), "
        f"Trend (`net_rtg_trend_delta`)  \n"
        f"**Sign convention**: model_sign={model_sign:+d}, trend_sign={trend_sign:+d}\n"
    )

    doc.append("---\n")
    doc.append("## 1. ATS Accuracy by Signal Alignment Cohort\n")
    doc.append(
        f"CAGE threshold: |cage_em_diff| ≥ {_CAGE_EM_MIN} + same direction as model.  \n"
        f"Trend threshold: |net_rtg_trend_delta| ≥ {_TREND_DELTA_MIN} + same direction as model.\n"
    )
    doc.append("| Cohort | W | L | ATS% | ROI (−110) | n | |")
    doc.append("|--------|---|---|------|------------|---|-|")
    doc.append(_fmt_row("Model alone",             m_only))
    doc.append(_fmt_row("Model + CAGE aligns",     m_cage))
    doc.append(_fmt_row("Model + Trend aligns",    m_trend))
    doc.append(_fmt_row("Model + CAGE + Trend",    m_c_t))
    doc.append(_fmt_row("Model + CAGE (|EM|≥10)",  m_cage_hi))
    doc.append(_fmt_row("Model + CAGE(≥10) + Trend", m_c_t_hi))

    doc.append("")
    doc.append("> **Interpretation**: Does stacking CAGE and/or Trend confirmation "
               "lift ATS% above Model alone?")

    doc.append("\n---\n")
    doc.append("## 2. CAGE Agreement Breakdown (all model-edge games)\n")
    doc.append("| CAGE status | W | L | ATS% | ROI | n | |")
    doc.append("|-------------|---|---|------|-----|---|-|")
    cage_confirms = base[base["cage_aligns"]]
    cage_neutral  = base[base["cage_raw"].abs() < _CAGE_EM_MIN]
    cage_diverges = base[~base["cage_aligns"] & (base["cage_raw"].abs() >= _CAGE_EM_MIN)]
    doc.append(_fmt_row("CAGE CONFIRMS model", cage_confirms))
    doc.append(_fmt_row("CAGE NEUTRAL (small EM)", cage_neutral))
    doc.append(_fmt_row("CAGE DIVERGES from model", cage_diverges))

    doc.append("\n---\n")
    doc.append("## 3. Trend Alignment Breakdown (all model-edge games)\n")
    doc.append("| Trend status | W | L | ATS% | ROI | n | |")
    doc.append("|--------------|---|---|------|-----|---|-|")
    trend_aligns  = base[base["trend_aligns"]]
    trend_neutral = base[base["trend_raw"].abs() < _TREND_DELTA_MIN]
    trend_against = base[~base["trend_aligns"] & (base["trend_raw"].abs() >= _TREND_DELTA_MIN)]
    doc.append(_fmt_row("Trend ALIGNS with model",   trend_aligns))
    doc.append(_fmt_row("Trend NEUTRAL (flat)",       trend_neutral))
    doc.append(_fmt_row("Trend OPPOSES model pick",   trend_against))

    doc.append("\n---\n")
    doc.append("## 4. Model Edge Magnitude × Alignment\n")
    doc.append("| Model edge | All | +CAGE | +Trend | +Both | n_all | n_both |")
    doc.append("|------------|-----|-------|--------|-------|-------|--------|")
    for lo, hi, label in [(1.0, 3.0, "1–3 pts"), (3.0, 6.0, "3–6 pts"),
                           (6.0, 10.0, "6–10 pts"), (10.0, 999.0, "10+ pts")]:
        sub = base[base["model_raw_edge"].abs().between(lo, hi - 0.001 if hi < 999 else hi)]
        if len(sub) < 5:
            doc.append(f"| {label} | — | — | — | — | {len(sub)} | — |")
            continue
        def pct(s: pd.DataFrame) -> str:
            if len(s) < 5:
                return f"—({len(s)})"
            return f"{s['model_correct'].mean()*100:.1f}%"
        doc.append(
            f"| {label} | {pct(sub)} | {pct(sub[sub['cage_aligns']])} | "
            f"{pct(sub[sub['trend_aligns']])} | "
            f"{pct(sub[sub['cage_aligns'] & sub['trend_aligns']])} | "
            f"{len(sub)} | {len(sub[sub['cage_aligns'] & sub['trend_aligns']])} |"
        )

    doc.append("\n---\n")
    doc.append("## 5. Key Takeaways\n")

    # Compute incremental lift
    base_acc = m_only["model_correct"].mean() * 100 if len(m_only) > 0 else 0
    cage_acc  = m_cage["model_correct"].mean() * 100 if len(m_cage) > 0 else 0
    trend_acc = m_trend["model_correct"].mean() * 100 if len(m_trend) > 0 else 0
    both_acc  = m_c_t["model_correct"].mean() * 100 if len(m_c_t) > 0 else 0
    doc.append(f"""
| Signal stack | ATS% | vs Model alone |
|---|---|---|
| Model alone | {base_acc:.1f}% | baseline |
| + CAGE aligns | {cage_acc:.1f}% | {cage_acc-base_acc:+.1f}pp |
| + Trend aligns | {trend_acc:.1f}% | {trend_acc-base_acc:+.1f}pp |
| + CAGE + Trend | {both_acc:.1f}% | {both_acc-base_acc:+.1f}pp |

Signal thresholds: Model edge ≥ {_MODEL_EDGE_MIN} pts, CAGE |EM| ≥ {_CAGE_EM_MIN}, Trend |delta| ≥ {_TREND_DELTA_MIN}
""")

    md_text = "\n".join(doc)
    _OUT_MD.write_text(md_text, encoding="utf-8")
    print(f"[OK] {_OUT_MD}")

    # ── Save game-level CSV ───────────────────────────────────────────────────
    save_cols = [
        "game_date", "home_team", "away_team",
        "espn_spread", "pred_spread", "cage_em_diff", "net_rtg_trend_delta",
        "home_covered_ats", "model_raw_edge", "cage_raw", "trend_raw",
        "model_picks_home", "cage_aligns", "trend_aligns", "model_correct",
    ]
    available = [c for c in save_cols if c in base.columns]
    base[available].to_csv(_OUT_CSV, index=False)
    print(f"[OK] {_OUT_CSV} ({len(base)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
