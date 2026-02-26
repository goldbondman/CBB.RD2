import json
import sys
from pathlib import Path

import pandas as pd

DEFAULT_PATHS = [
    Path("data/team_game_metrics.csv"),
    Path("data/csv/team_game_metrics.csv"),
    Path("data/team_game_weighted.csv"),
    Path("data/csv/team_game_weighted.csv"),
]


def _resolve_metrics_path() -> Path:
    for path in DEFAULT_PATHS:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise FileNotFoundError(
        "Could not find team_game_metrics.csv in expected paths: "
        + ", ".join(str(p) for p in DEFAULT_PATHS)
    )


metrics_path = _resolve_metrics_path()
metrics = pd.read_csv(metrics_path, low_memory=False)
metrics["game_datetime_utc"] = pd.to_datetime(metrics["game_datetime_utc"], utc=True, errors="coerce")
metrics = metrics.sort_values(["team_id", "game_datetime_utc"])

leakage_issues = []

rolling_pairs = [
    ("net_rtg", "net_rtg_l5"),
    ("ortg", "ortg_l5"),
    ("drtg", "drtg_l5"),
    ("efg_pct", "efg_pct_l5"),
    ("tov_pct", "tov_pct_l5"),
]

for base_col, l5_col in rolling_pairs:
    pooled_frames = []
    for team_id, grp in metrics.groupby("team_id"):
        if len(grp) < 20:
            continue
        grp = grp.sort_values("game_datetime_utc").reset_index(drop=True)
        if base_col not in grp.columns or l5_col not in grp.columns:
            continue
        sub = grp[[base_col, l5_col]].apply(pd.to_numeric, errors="coerce")
        if sub[base_col].std() == 0 or sub[l5_col].std() == 0:
            continue
        pooled_frames.append(sub)

    if not pooled_frames:
        continue

    pooled = pd.concat(pooled_frames, ignore_index=True).dropna()
    if len(pooled) < 20:
        continue

    current_corr = pooled[base_col].corr(pooled[l5_col])
    prior_corr = pooled[base_col].shift(1).corr(pooled[l5_col])

    if pd.notna(current_corr) and pd.notna(prior_corr) and current_corr > prior_corr + 0.25:
        leakage_issues.append(
            {
                "column": l5_col,
                "current_corr": round(float(current_corr), 3),
                "prior_corr": round(float(prior_corr), 3),
                "rows_pooled": int(len(pooled)),
                "rule": "flag_if_current_gt_prior_plus_0.25",
            }
        )

with open("data/leakage_audit.json", "w", encoding="utf-8") as f:
    json.dump(leakage_issues, f, indent=2)

if leakage_issues:
    print(f"[FAIL] Potential feature leakage detected in {len(leakage_issues)} rolling features ({metrics_path}).")
    for issue in leakage_issues:
        print(
            f"  {issue['column']}: current_corr={issue['current_corr']:.3f} "
            f"prior_corr={issue['prior_corr']:.3f} rows={issue['rows_pooled']}"
        )
    sys.exit(1)

print(f"[OK] No feature leakage detected — pooled rolling checks passed ({metrics_path})")
sys.exit(0)
