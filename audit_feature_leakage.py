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

# Filter to teams with sufficient history.
valid_teams = metrics.groupby("team_id").filter(lambda x: len(x) >= 20)

for base_col, l5_col in rolling_pairs:
    if base_col not in valid_teams.columns or l5_col not in valid_teams.columns:
        continue

    # Correlation across all teams pooled.
    subset = valid_teams[[base_col, l5_col, "team_id"]].copy()
    subset[[base_col, l5_col]] = subset[[base_col, l5_col]].apply(pd.to_numeric, errors="coerce")
    subset = subset.dropna()

    if len(subset) < 100:
        continue

    current_corr = subset[base_col].corr(subset[l5_col])
    prior_base = subset.groupby("team_id")[base_col].shift(1)
    prior_corr = prior_base.corr(subset[l5_col])

    if pd.notna(current_corr) and pd.notna(prior_corr) and current_corr > prior_corr + 0.25:
        leakage_issues.append(
            {
                "column": l5_col,
                "current_corr": round(float(current_corr), 3),
                "prior_corr": round(float(prior_corr), 3),
                "diff": round(float(current_corr - prior_corr), 3),
                "rows_pooled": int(len(subset)),
                "rule": "flag_if_current_gt_prior_plus_0.25",
            }
        )

by_col = {}
for issue in leakage_issues:
    by_col.setdefault(issue["column"], []).append(issue)

output_path = Path("data/leakage_audit.json")
output_path.write_text(json.dumps(leakage_issues, indent=2), encoding="utf-8")

if by_col:
    print(f"[FAIL] Potential feature leakage detected in {len(by_col)} columns ({metrics_path}):")
    for col, issues in sorted(by_col.items()):
        avg_curr = sum(i["current_corr"] for i in issues) / len(issues)
        avg_prior = sum(i["prior_corr"] for i in issues) / len(issues)
        print(
            f"  {col}: {len(issues)} pooled checks - "
            f"avg current_corr={avg_curr:.3f} vs prior_corr={avg_prior:.3f}"
        )
    print(f"Details written to {output_path}")
    sys.exit(1)

print(f"[OK] No feature leakage detected - pooled rolling checks passed ({metrics_path})")
sys.exit(0)

