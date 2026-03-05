#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: check_csv.sh <file> [min_rows] [required_cols_csv] [max_age_hours]" >&2
  exit 2
fi

FILE="$1"
MIN_ROWS="${2:-1}"
REQUIRED_COLS="${3:-}"
MAX_AGE_HOURS="${4:-}"

if [[ ! -f "$FILE" ]]; then
  echo "[ERROR] Missing required file: $FILE" >&2
  exit 1
fi

if [[ ! "$MIN_ROWS" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] min_rows must be numeric: $MIN_ROWS" >&2
  exit 1
fi

ROWS=$(( $(wc -l < "$FILE") - 1 ))
if (( ROWS < MIN_ROWS )); then
  echo "[ERROR] $FILE has $ROWS rows; requires >= $MIN_ROWS" >&2
  exit 1
fi

python - "$FILE" "$REQUIRED_COLS" "$MAX_AGE_HOURS" <<'PY'
from __future__ import annotations

import time
from pathlib import Path
import sys

import pandas as pd

path = Path(sys.argv[1])
required = [c.strip() for c in (sys.argv[2] or "").split(",") if c.strip()]
max_age_hours_raw = sys.argv[3]

df = pd.read_csv(path, dtype=str, low_memory=False)
if required:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[ERROR] {path} missing required columns: {missing}")

if max_age_hours_raw:
    max_age_hours = float(max_age_hours_raw)
    age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    if age_hours > max_age_hours:
        raise SystemExit(f"[ERROR] {path} stale: age_hours={age_hours:.2f}, max_age_hours={max_age_hours}")

print(f"[OK] {path} rows={len(df)} cols={len(df.columns)}")
PY
