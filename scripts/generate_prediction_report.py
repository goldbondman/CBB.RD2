#!/usr/bin/env python3
"""
generate_prediction_report.py

Produces a rolling 30-hour prediction performance report from
backtest_results_latest.csv.  Emits:

  data/prediction_report_30h.md   — human-readable Markdown report
  data/prediction_report_30h.csv  — machine-readable line-by-line results

Metrics:
  - Win/Loss per game (model covered ATS yes/no)
  - Overall hit rate for the 30-hour window
  - Hit rate broken out by edge bucket:
      0–3     │ 3.1–5     │ 5.1–8     │ 8.1+
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ── Config ─────────────────────────────────────────────────────────────────
_RESULTS_PATH   = Path("data/backtest_results_latest.csv")
_OUT_MD         = Path("data/prediction_report_30h.md")
_OUT_CSV        = Path("data/prediction_report_30h.csv")
_WINDOW_HOURS   = 30

# Edge buckets: (label, lo_inclusive, hi_exclusive)
_EDGE_BUCKETS = [
    ("0–3",    0.0,  3.0),
    ("3.1–5",  3.0,  5.0),
    ("5.1–8",  5.0,  8.0),
    ("8.1+",   8.0, 999.0),
]


def _load_results() -> pd.DataFrame:
    """Load and coerce backtest_results_latest.csv."""
    if not _RESULTS_PATH.exists():
        print(f"[ERROR] {_RESULTS_PATH} not found")
        sys.exit(1)

    df = pd.read_csv(_RESULTS_PATH, low_memory=False)

    # ── Date column ────────────────────────────────────────────────────────
    date_col = next(
        (c for c in ["game_datetime", "game_datetime_utc", "game_date", "date"]
         if c in df.columns),
        None,
    )
    if date_col is None:
        print("[ERROR] No date column found in backtest_results_latest.csv")
        sys.exit(1)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df = df.rename(columns={date_col: "_game_dt"})

    # ── Required columns ──────────────────────────────────────────────────
    ens_col = next(
        (c for c in ["ens_spread", "pred_margin_ATS", "pred_home_spread"] if c in df.columns),
        None,
    )
    ats_col = next(
        (c for c in ["actual_margin_ATS", "home_cover_margin"] if c in df.columns),
        None,
    )
    conf_col = next(
        (c for c in ["ens_confidence", "spread_conf"] if c in df.columns), None
    )
    spread_col = next(
        (c for c in ["spread_line", "market_spread", "home_market_spread"] if c in df.columns),
        None,
    )

    if not ens_col:
        print("[ERROR] No ensemble spread column found")
        sys.exit(1)
    if not ats_col:
        print("[ERROR] No actual ATS margin column found")
        sys.exit(1)

    df[ens_col]  = pd.to_numeric(df[ens_col],  errors="coerce")
    df[ats_col]  = pd.to_numeric(df[ats_col],  errors="coerce")
    if spread_col:
        df[spread_col] = pd.to_numeric(df[spread_col], errors="coerce")

    df = df.dropna(subset=["_game_dt", ens_col, ats_col])

    # ── Compute edge (model disagreement vs Vegas) ────────────────────────
    # edge = |pred_margin_ATS| — how far the model is from the spread
    # If spread_line is available, edge = |ens_spread - spread_line|
    if spread_col and spread_col in df.columns:
        df["_edge"] = (df[ens_col] - df[spread_col]).abs()
    else:
        # Fallback: use |ens_spread| itself (how far from 0, i.e., pick side conviction)
        df["_edge"] = df[ens_col].abs()

    # ── Model pick direction ──────────────────────────────────────────────
    # ens_spread > 0 → model likes home; actual_margin_ATS > 0 → home covered
    df["_model_pick_home"] = (df[ens_col] > 0).astype(int)
    df["_home_covered"]    = (df[ats_col] > 0).astype(int)
    df["_correct"]         = (df["_model_pick_home"] == df["_home_covered"]).astype(int)

    # ── Confidence ────────────────────────────────────────────────────────
    if conf_col and conf_col in df.columns:
        df["_conf"] = df[conf_col].astype(str).str.strip().str.upper()
    else:
        df["_conf"] = "UNKNOWN"

    # Carry through human-readable columns
    df["_home_team"] = df.get("home_team", pd.Series("", index=df.index)).astype(str)
    df["_away_team"] = df.get("away_team", pd.Series("", index=df.index)).astype(str)
    df["_spread_line"] = df[spread_col].round(1) if spread_col else np.nan
    df["_ens_spread"]  = df[ens_col].round(2)
    df["_actual_ats"]  = df[ats_col].round(2)

    return df


def _filter_window(df: pd.DataFrame, hours: int = _WINDOW_HOURS) -> pd.DataFrame:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return df[df["_game_dt"] >= cutoff].copy()


def _hit_rate(correct: pd.Series) -> tuple[float, int]:
    n = len(correct)
    if n == 0:
        return 0.0, 0
    return round(correct.sum() / n * 100, 1), n


def _edge_bucket(edge: float) -> str:
    for label, lo, hi in _EDGE_BUCKETS:
        if lo <= edge < hi:
            return label
    return "8.1+"


def _result_str(correct: int, model_pick_home: int, home: str, away: str,
                spread_line, ens_spread: float, actual_ats: float) -> str:
    pick_team = home if model_pick_home else away
    opp_team  = away if model_pick_home else home
    result    = "WIN" if correct else "LOSS"

    spread_str = ""
    if not pd.isna(spread_line):
        sign = "+" if ens_spread > 0 else ""
        spread_str = f" [model {sign}{ens_spread:.1f} vs line {spread_line:.1f}]"

    ats_str = f"actual_ATS={actual_ats:+.1f}"
    return f"{result}: {pick_team} vs {opp_team}{spread_str} — {ats_str}"


def generate_report(df_window: pd.DataFrame, generated_at: str) -> tuple[str, pd.DataFrame]:
    """Build markdown report and CSV rows. Returns (markdown_str, csv_df)."""

    rows_out: list[dict] = []
    for _, row in df_window.sort_values("_game_dt").iterrows():
        edge_label = _edge_bucket(float(row["_edge"]))
        rows_out.append({
            "game_datetime_utc": str(row["_game_dt"])[:16],
            "away_team":         row["_away_team"],
            "home_team":         row["_home_team"],
            "spread_line":       row["_spread_line"],
            "model_pick":        row["_home_team"] if row["_model_pick_home"] else row["_away_team"],
            "model_edge":        round(float(row["_edge"]), 2),
            "edge_bucket":       edge_label,
            "confidence":        row["_conf"],
            "ens_spread":        row["_ens_spread"],
            "actual_margin_ats": row["_actual_ats"],
            "result":            "WIN" if row["_correct"] else "LOSS",
        })

    csv_df = pd.DataFrame(rows_out)

    # ── Aggregations ──────────────────────────────────────────────────────
    overall_rate, overall_n = _hit_rate(df_window["_correct"])

    bucket_stats: list[tuple[str, float, int]] = []
    for label, lo, hi in _EDGE_BUCKETS:
        mask = df_window["_edge"].between(lo, hi - 1e-9) if hi < 999 else (df_window["_edge"] >= lo)
        rate, n = _hit_rate(df_window.loc[mask, "_correct"])
        bucket_stats.append((label, rate, n))

    # ── Build Markdown ─────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append(f"# CBB Prediction Report — Last {_WINDOW_HOURS} Hours")
    lines.append(f"_Generated {generated_at} UTC_\n")

    lines.append(f"## Overall: **{overall_rate:.1f}%** ATS ({overall_n} games)\n")

    lines.append("## Hit Rate by Edge Bucket\n")
    lines.append("| Edge Bucket | Hit Rate | Games |")
    lines.append("|-------------|----------|-------|")
    for label, rate, n in bucket_stats:
        marker = " ← best" if rate == max(r for _, r, c in bucket_stats if c > 0) and n > 0 else ""
        lines.append(f"| {label} | {rate:.1f}% | {n}{marker} |")

    lines.append("\n## Game-by-Game Results\n")
    lines.append("| Time (UTC) | Away | Home | Line | Pick | Edge | Conf | Result |")
    lines.append("|------------|------|------|------|------|------|------|--------|")

    for _, row in df_window.sort_values("_game_dt").iterrows():
        pick_team = row["_home_team"] if row["_model_pick_home"] else row["_away_team"]
        result    = "✓ WIN" if row["_correct"] else "✗ LOSS"
        line_str  = f"{row['_spread_line']:+.1f}" if not pd.isna(row["_spread_line"]) else "—"
        dt_str    = str(row["_game_dt"])[:16].replace("T", " ")
        lines.append(
            f"| {dt_str} | {row['_away_team']} | {row['_home_team']} "
            f"| {line_str} | {pick_team} | {row['_edge']:.1f} "
            f"| {row['_conf']} | {result} |"
        )

    return "\n".join(lines), csv_df


def main() -> int:
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    df_all = _load_results()
    print(f"[INFO] Loaded {len(df_all)} total backtest records")

    df_window = _filter_window(df_all, _WINDOW_HOURS)
    print(f"[INFO] {len(df_window)} games in last {_WINDOW_HOURS} hours")

    if df_window.empty:
        print("[WARN] No completed games in the last 30 hours — writing empty report")
        _OUT_MD.write_text(
            f"# CBB Prediction Report — Last {_WINDOW_HOURS} Hours\n\n"
            f"_Generated {generated_at} UTC_\n\n"
            "No completed predictions found in this window.\n",
            encoding="utf-8",
        )
        pd.DataFrame(columns=[
            "game_datetime_utc", "away_team", "home_team", "spread_line",
            "model_pick", "model_edge", "edge_bucket", "confidence",
            "ens_spread", "actual_margin_ats", "result",
        ]).to_csv(_OUT_CSV, index=False)
        return 0

    md_content, csv_df = generate_report(df_window, generated_at)

    _OUT_MD.write_text(md_content, encoding="utf-8")
    csv_df.to_csv(_OUT_CSV, index=False)

    overall_rate, n = _hit_rate(df_window["_correct"])
    wins  = int(df_window["_correct"].sum())
    losses = n - wins
    print(f"[OK] Report: {n} games | {wins}-{losses} | {overall_rate:.1f}% ATS")
    print(f"[OK] Written → {_OUT_MD}, {_OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
