import pandas as pd
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

def validate_market_data(file_path: str):
    if not os.path.exists(file_path):
        print(f"[ERROR] {file_path} not found")
        sys.exit(1)

    df = pd.read_csv(file_path)
    total_rows = len(df)

    if total_rows == 0:
        print(f"[ERROR] {file_path} is empty")
        sys.exit(1)

    # Filter to completed games older than 48 hours
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=48)

    df["game_datetime"] = pd.to_datetime(df["game_datetime"], utc=True)
    completed_old_df = df[df["game_datetime"] < cutoff]

    comp_old_rows = len(completed_old_df)
    if comp_old_rows == 0:
        print("[INFO] No completed games older than 48 hours to validate.")
        return

    missing_home_spread = completed_old_df["home_market_spread"].isna().sum()
    missing_market_total = completed_old_df["market_total"].isna().sum()

    pct_missing_spread = (missing_home_spread / comp_old_rows) * 100
    pct_missing_total = (missing_market_total / comp_old_rows) * 100

    print(f"=== Market Data Validation Summary ({file_path}) ===")
    print(f"Total Rows: {total_rows}")
    print(f"Completed Games (>48h old): {comp_old_rows}")
    print(f"Missing home_market_spread: {missing_home_spread} ({pct_missing_spread:.2f}%)")
    print(f"Missing market_total: {missing_market_total} ({pct_missing_total:.2f}%)")

    # Diagnostics: Top 20 missing
    missing_any = completed_old_df[completed_old_df["home_market_spread"].isna() | completed_old_df["market_total"].isna()]
    if not missing_any.empty:
        print("\n--- Top 20 Missing Market Fields ---")
        cols_to_show = ["game_id", "game_datetime", "home_team", "away_team", "home_market_spread", "market_total"]
        print(missing_any[cols_to_show].head(20).to_string(index=False))

    # Failure Threshold
    threshold = 5.0
    fail = False
    if pct_missing_spread > threshold:
        print(f"[FAIL] Missing home_market_spread exceeds {threshold}% threshold: {pct_missing_spread:.2f}%")
        fail = True
    if pct_missing_total > threshold:
        print(f"[FAIL] Missing market_total exceeds {threshold}% threshold: {pct_missing_total:.2f}%")
        fail = True

    if fail:
        if os.environ.get("ALLOW_MARKET_MISSING") == "1":
            print("[WARN] Validation failed, but ALLOW_MARKET_MISSING=1 is set. Proceeding...")
        else:
            print("[ERROR] Market data validation failed. Blocking build.")
            sys.exit(1)
    else:
        print("[PASS] Market data validation successful.")

if __name__ == "__main__":
    target_file = "data/backtest_results_latest.csv"
    validate_market_data(target_file)
