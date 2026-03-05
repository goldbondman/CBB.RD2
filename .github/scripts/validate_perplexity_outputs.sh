#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${1:-data/perplexity_contract_runner.log}"

declare -a OUTPUT_SPECS=(
  "data/advanced_metrics.csv|advanced_metrics_builder|python -m pipeline.advanced_metrics.build_advanced_metrics|event_id,team_id|0"
  "data/market_lines_latest_by_game.csv|market_builder|python -m pipeline.market_canonical --data-dir data --debug-dir debug|event_id,spread_line,total_line|0"
  "data/team_snapshot.csv|joint_models_predictions|python -m model_lab.joint_models|team_id|1"
  "data/predictions_joint_latest.csv|joint_models_predictions|python -m model_lab.joint_models|game_id,pred_total,pred_margin|0"
  "data/predictions_joint_snapshots.csv|joint_models_predictions|python -m model_lab.joint_models|game_id,generated_at_utc|0"
)

missing=0
for spec in "${OUTPUT_SPECS[@]}"; do
  IFS="|" read -r path producer command required_cols min_rows <<<"${spec}"
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Expected required output missing: ${path} (producer: ${producer}; command: ${command})"
    missing=1
    continue
  fi
  bash .github/scripts/check_csv.sh "${path}" "${min_rows}" "${required_cols}"
  python - "${path}" <<'PY'
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

path = Path(sys.argv[1])
df = pd.read_csv(path, dtype=str, low_memory=False)
print(f"[INFO] producer_write path={path.resolve()} rows={len(df)} cols={len(df.columns)}")
PY
done

if (( missing != 0 )); then
  if [[ -f "${LOG_PATH}" ]]; then
    echo "[INFO] Last 50 lines from ${LOG_PATH}:"
    tail -n 50 "${LOG_PATH}" || true
  else
    echo "[WARN] Producer log not found at ${LOG_PATH}"
  fi
  exit 1
fi

echo "[OK] Perplexity required outputs validated"
