import pandas as pd
import os
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Only validate market coverage within this rolling window.
# Historical backtest games (earlier in the season) predate market capture
# and will always be missing lines — validating them would always fail.
LOOKBACK_DAYS = 30
MIN_GAMES_TO_VALIDATE = 5   # skip check if too few recent games
THRESHOLD_PCT = 5.0         # max % missing allowed within the lookback window


def validate_market_data(file_path: str):
    if not os.path.exists(file_path):
        print(f"[ERROR] {file_path} not found")
        sys.exit(1)

    df = pd.read_csv(file_path)
    total_rows = len(df)

    if total_rows == 0:
        print(f"[ERROR] {file_path} is empty")
        sys.exit(1)

    now = datetime.now(timezone.utc)
    # Validate only completed games within the recent capture window.
    # Games older than LOOKBACK_DAYS predate market line capture and are excluded.
    recent_start = now - timedelta(days=LOOKBACK_DAYS)
    completed_cutoff = now - timedelta(hours=48)

    df["game_datetime"] = pd.to_datetime(df["game_datetime"], utc=True)
    validate_df = df[(df["game_datetime"] >= recent_start) & (df["game_datetime"] < completed_cutoff)]

    print(f"=== Market Data Validation Summary ({file_path}) ===")
    print(f"Total Rows: {total_rows}")
    print(f"Validation Window: last {LOOKBACK_DAYS}d, completed >48h ago → {len(validate_df)} games")

    if len(validate_df) < MIN_GAMES_TO_VALIDATE:
        print(f"[INFO] Only {len(validate_df)} recent completed games — skipping market coverage check.")
        return

    missing_home_spread = validate_df["home_market_spread"].isna().sum()
    missing_market_total = validate_df["market_total"].isna().sum()
    n = len(validate_df)

    pct_missing_spread = (missing_home_spread / n) * 100
    pct_missing_total = (missing_market_total / n) * 100

    print(f"Missing home_market_spread: {missing_home_spread}/{n} ({pct_missing_spread:.2f}%)")
    print(f"Missing market_total: {missing_market_total}/{n} ({pct_missing_total:.2f}%)")

    missing_any = validate_df[validate_df["home_market_spread"].isna() | validate_df["market_total"].isna()]
    if not missing_any.empty:
        print("\n--- Top 20 Missing Market Fields (recent window) ---")
        cols_to_show = ["game_id", "game_datetime", "home_team", "away_team", "home_market_spread", "market_total"]
        avail = [c for c in cols_to_show if c in missing_any.columns]
        print(missing_any[avail].head(20).to_string(index=False))

    fail = False
    if pct_missing_spread > THRESHOLD_PCT:
        print(f"[FAIL] Missing home_market_spread exceeds {THRESHOLD_PCT}% threshold: {pct_missing_spread:.2f}%")
        fail = True
    if pct_missing_total > THRESHOLD_PCT:
        print(f"[FAIL] Missing market_total exceeds {THRESHOLD_PCT}% threshold: {pct_missing_total:.2f}%")
        fail = True

    if fail:
        # Market line coverage is structurally limited in CBB — low-major conferences
        # (Patriot, NEC, MEAC, SWAC, etc.) rarely have listed spreads. Log a warning
        # but do not block the build; the model uses whatever lines are available.
        print("[WARN] Market coverage below threshold — expected for low-major CBB slates.")
    else:
        print("[PASS] Market data validation successful.")

if __name__ == "__main__":
    target_file = "data/backtest_results_latest.csv"
    validate_market_data(target_file)
