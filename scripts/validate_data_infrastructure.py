#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd


def normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.lstrip("0")
        .replace("", "0")
    )


def check_file(path, required_cols, pk_cols,
               critical_col=None, min_rows=1,
               min_nonnull=0.5):
    """
    Returns dict with:
      exists, row_count, missing_cols, pk_duplicates,
      critical_col_nonnull_rate, status (PASS/WARN/FAIL)
    """
    p = Path(path)
    out = {
        "path": path,
        "exists": p.exists() and p.stat().st_size > 0,
        "row_count": 0,
        "missing_cols": [],
        "pk_duplicates": None,
        "critical_col_nonnull_rate": None,
        "status": "FAIL",
    }
    if not out["exists"]:
        return out

    try:
        df = pd.read_csv(p, low_memory=False)
    except Exception:
        return out

    out["row_count"] = len(df)
    out["missing_cols"] = [c for c in required_cols if c not in df.columns]

    if all(c in df.columns for c in pk_cols):
        out["pk_duplicates"] = int(df.duplicated(subset=pk_cols).sum())

    if critical_col and critical_col in df.columns and len(df) > 0:
        out["critical_col_nonnull_rate"] = float(df[critical_col].notna().mean())

    fail = False
    warn = False
    if out["row_count"] < min_rows:
        fail = True
    if out["missing_cols"]:
        fail = True
    if out["pk_duplicates"] is not None and out["pk_duplicates"] > 0:
        fail = True
    if critical_col:
        if out["critical_col_nonnull_rate"] is None:
            fail = True
        elif out["critical_col_nonnull_rate"] < min_nonnull:
            warn = True

    out["status"] = "FAIL" if fail else ("WARN" if warn else "PASS")
    return out


CHECKS = [
    {
      'path': 'data/games.csv',
      'required_cols': ['game_id','game_date','home_team_id',
                        'away_team_id','completed',
                        'home_score','away_score'],
      'pk_cols': ['game_id'],
      'critical_col': 'completed',
      'min_rows': 100,
      'min_nonnull': 0.95,
    },
    {
      'path': 'data/team_game_weighted.csv',
      'required_cols': ['event_id','team_id',
                        'game_datetime_utc','net_rtg_l10',
                        'adj_ortg','adj_drtg','pace_l5'],
      'pk_cols': ['event_id','team_id'],
      'critical_col': 'net_rtg_l10',
      'min_rows': 200,
      'min_nonnull': 0.70,
    },
    {
      'path': 'data/predictions_combined_latest.csv',
      'required_cols': ['game_id','pred_spread','pred_total'],
      'pk_cols': ['game_id'],
      'critical_col': 'pred_spread',
      'min_rows': 1,
      'min_nonnull': 0.80,
    },
    {
      'path': 'data/predictions_history.csv',
      'required_cols': ['game_id','pred_spread','game_date'],
      'pk_cols': ['game_id'],
      'critical_col': 'pred_spread',
      'min_rows': 50,
      'min_nonnull': 0.80,
    },
    {
      'path': 'data/market_lines.csv',
      'required_cols': ['game_id','home_spread',
                        'total_line','capture_type'],
      'pk_cols': ['game_id','capture_type'],
      'critical_col': 'home_spread',
      'min_rows': 50,
      'min_nonnull': 0.70,
    },
    {
      'path': 'data/results_log.csv',
      'required_cols': ['game_id','pred_spread',
                        'actual_margin','market_spread',
                        'home_covered_ats'],
      'pk_cols': ['game_id'],
      'critical_col': 'home_covered_ats',
      'min_rows': 50,
      'min_nonnull': 0.90,
    },
    {
      'path': 'data/player_game_logs.csv',
      'required_cols': ['event_id','team_id','athlete_id',
                        'min','did_not_play'],
      'pk_cols': ['event_id','team_id','athlete_id'],
      'critical_col': 'min',
      'min_rows': 500,
      'min_nonnull': 0.80,
    },
]


def join_health():
    def get_ids(path, col):
        p = Path(path)
        if not p.exists() or p.stat().st_size == 0:
            return set()
        df = pd.read_csv(p, dtype=str, low_memory=False)
        if col not in df.columns:
            return set()
        return set(normalize_id(df[col]))

    joins = [
        ("games.csv", "data/games.csv", "game_id", "predictions_history.csv", "data/predictions_history.csv", "game_id"),
        ("games.csv", "data/games.csv", "game_id", "results_log.csv", "data/results_log.csv", "game_id"),
        ("games.csv", "data/games.csv", "game_id", "market_lines.csv", "data/market_lines.csv", "game_id"),
    ]
    rows = []
    for a_name, a_path, a_col, b_name, b_path, b_col in joins:
        a = get_ids(a_path, a_col)
        b = get_ids(b_path, b_col)
        denom = min(len(a), len(b)) if a and b else 0
        inter = len(a & b)
        pct = (inter / denom * 100) if denom else 0.0
        status = "PASS" if pct >= 50 else ("WARN" if pct >= 20 else "FAIL")
        rows.append((a_name, b_name, inter, denom, pct, status))
    return rows


def fmt_pct(v):
    if v is None:
        return "N/A"
    return f"{v*100:.1f}%"


if __name__ == "__main__":
    results = [check_file(**cfg) for cfg in CHECKS]

    print("┌─────────────────────────────────────────┬──────┬────────┬────────────┬──────────┐")
    print("│ File                                    │ Rows │ Pk_ok  │ Crit_nonnull│ Status  │")
    print("├─────────────────────────────────────────┼──────┼────────┼────────────┼──────────┤")
    for r in results:
        pk_ok = "N/A" if r["pk_duplicates"] is None else ("YES" if r["pk_duplicates"] == 0 else "NO")
        print(f"│ {r['path'][:39]:<39} │ {r['row_count']:>4} │ {pk_ok:<6} │ {fmt_pct(r['critical_col_nonnull_rate']):<10} │ {r['status']:<8} │")
    print("└─────────────────────────────────────────┴──────┴────────┴────────────┴──────────┘")

    print("\nJOIN HEALTH")
    j = join_health()
    for a, b, inter, denom, pct, status in j:
        print(f"{a} ∩ {b}: {inter:>4} / {denom:<4} ({pct:>5.1f}%)  {status}")

    statuses = [r["status"] for r in results] + [x[-1] for x in j]
    if "FAIL" in statuses:
        sys.exit(2)
    if "WARN" in statuses:
        sys.exit(1)
    sys.exit(0)
