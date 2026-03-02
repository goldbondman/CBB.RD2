#!/usr/bin/env python3
"""
Generate complete sample data for handicapper app testing
Creates realistic CSVs with handicappers, tweets, picks, games
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from faker import Faker
import random

fake = Faker()
random.seed(42)
np.random.seed(42)


def generate_sample_data(data_dir="./data"):
    """Generate all 6 CSVs + model_predictions.csv with realistic data"""
    Path(data_dir).mkdir(exist_ok=True)

    print("🔄 Generating realistic sample data...")

    # 1. HANDICAPPERS (8 total: 3 sharp, 3 medium, 2 public)
    handicappers = pd.DataFrame({
        'handicapper_id': range(1, 9),
        'handle': [
            '@CBB_Edge', '@HoopsLock', '@BracketBuster',      # Sharp
            '@MidMajorGuru', '@ValueHunter', '@SpreadKing',    # Medium
            '@PublicFader', '@CasualBets'                      # Public
        ],
        'tier': ['sharp'] * 3 + ['medium'] * 3 + ['public'] * 2,
        'status': ['active'] * 7 + ['paused'],
        'lifetime_roi': np.random.normal(0.015, 0.025, 8).clip(-0.08, 0.05).round(4),
        'win_pct': np.random.normal(0.545, 0.035, 8).clip(0.45, 0.60).round(3),
        'total_picks': np.random.randint(89, 300, 8),
        'parse_template_id': random.choices([1, 2, 3, 4], k=8),
        'notes': [
            'Consistent sharp, great spreads', 'Totals specialist', 'Tournament expert',
            'Good with underdogs', 'Value plays', 'Spread focus',
            'Reverse line movement', 'Casual bettor'
        ],
        'created_at': [datetime(2026, 1, random.randint(15, 28)).isoformat() for _ in range(8)]
    })

    # 2. GAMES (20 games: 12 completed, 8 upcoming)
    game_dates = pd.date_range('2026-02-25', periods=20, freq='8h') + timedelta(hours=2)
    teams_pool = [
        'Illinois', 'Wisconsin', 'Kentucky', 'UConn', 'Purdue', 'Gonzaga',
        'Houston', 'Tennessee', 'Duke', 'UNC', 'Kansas', 'Baylor',
        'Arizona', 'UCLA', 'Michigan St', 'Iowa St', 'Creighton', 'Dayton'
    ]

    games = pd.DataFrame({
        'game_id': range(101, 121),
        'date': game_dates,
        'home_team': [teams_pool[i % len(teams_pool)] for i in range(20)],
        'away_team': [teams_pool[(i + 9) % len(teams_pool)] for i in range(20)],
        'home_score': [random.randint(60, 95) if i < 12 else None for i in range(20)],
        'away_score': [random.randint(55, 92) if i < 12 else None for i in range(20)],
        'closing_spread': np.random.normal(-1.2, 4.5, 20).round(1),
        'total_line': np.random.normal(138.5, 8.2, 20).round(0).clip(120, 160)
    })

    # 3. RAW_TWEETS (25 tweets from handicappers)
    tweet_texts = [
        "Illinois -3.5 (2u) vs Wisconsin. Bulls heating up",
        "UConn/Kentucky OVER 142.5 (3u) - fireworks incoming",
        "FADE Duke ML +120 vs UNC - overrated",
        "Gonzaga +2.5 (1u) - Zags get no respect",
        "Houston -5.5 (2u) - Cougars defense elite",
        "Purdue/Kentucky UNDER 135 (1.5u)",
        "Arizona ML vs UCLA (2u) - Wildcats rolling",
        "Iowa St +7 vs Kansas (3u) - live dog"
    ]

    raw_tweets = pd.DataFrame({
        'tweet_id': [f"184{random.randint(1000000000, 9999999999)}" for _ in range(25)],
        'handicapper_id': np.random.choice(handicappers['handicapper_id'], 25),
        'created_at': pd.date_range('2026-02-27', periods=25, freq='90min'),
        'text': [t + " " + fake.sentence(nb_words=5) for t in np.random.choice(tweet_texts, 25)],
        'tweet_url': [f"https://twitter.com/test/status/{i}" for i in range(25)],
        'ingested_at': pd.date_range('2026-02-27', periods=25, freq='90min') + pd.Timedelta(minutes=2)
    })

    # 4. RAW_PICKS (35 parsed picks - mixed success)
    raw_pick_statuses = ['success', 'failed', 'partial', 'fade_detected']
    markets = ['spread', 'total', 'moneyline', 'fade']

    raw_picks = pd.DataFrame({
        'raw_pick_id': range(1, 36),
        'tweet_id': np.random.choice(raw_tweets['tweet_id'], 35),
        'handicapper_id': np.random.choice(handicappers['handicapper_id'], 35),
        'market': np.random.choice(markets, 35),
        'team_raw': np.random.choice([
            'Illinois Fighting Illini', 'UConn Huskies', 'Kansas Jayhawks',
            'St Johns Red Storm', 'Florida Atlantic Owls', 'Random Junk',
            'Duke Blue Devils', 'Gonzaga Bulldogs'
        ], 35),
        'line': np.random.choice([None, -3.5, 142.5, +2.5, -5.5, 135.0], 35),
        'units': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], 35),
        'odds': np.random.choice([None, '+120', '-110'], 35),
        'parse_status': np.random.choice(raw_pick_statuses, 35, p=[0.7, 0.15, 0.1, 0.05])
    })

    # 5. PICKS (25 fully mapped picks)
    picks = raw_picks.sample(25).copy()
    picks['pick_id'] = range(1, 26)
    picks['game_id'] = np.random.choice(games['game_id'], 25)
    picks['side'] = np.random.choice(['home', 'away', 'over', 'under'], 25)
    picks['mapping_status'] = np.random.choice(['ok', 'manual_override', 'ambiguous'], 25, p=[0.8, 0.15, 0.05])

    # 6. MODEL_PREDICTIONS (for alignment analysis)
    model_predictions = pd.DataFrame({
        'game_id': games['game_id'].tolist() * 2,  # Duplicate for coverage
        'model_margin': np.random.normal(0.8, 6.5, 40).round(1),
        'model_confidence': np.random.uniform(0.55, 0.85, 40).round(2),
        'predicted_total': np.random.normal(139.2, 7.8, 40).round(0),
        'created_at': pd.date_range('2026-02-27', periods=40, freq='6h')
    })

    # Save all files
    files = {
        'handicappers.csv': handicappers,
        'raw_tweets.csv': raw_tweets,
        'raw_picks.csv': raw_picks,
        'picks.csv': picks,
        'games.csv': games,
        'model_predictions.csv': model_predictions
    }

    for filename, df in files.items():
        filepath = Path(data_dir) / filename
        df.to_csv(filepath, index=False)
        print(f"✅ {filename}: {len(df)} rows")

    print(f"\n🎉 COMPLETE! Generated {sum(len(df) for df in files.values())} total records")
    print(f"📁 Files saved to {data_dir}/")

    return files


def validate_sample_data(data_dir="./data"):
    """Validate generated data integrity"""
    from data_loader import load_app_data
    data = load_app_data(data_dir)

    print("\n🔍 VALIDATION:")
    total_rows = 0
    for name, df in data.items():
        rows = len(df)
        total_rows += rows
        print(f"  {name}: {rows} rows ✓")

    print(f"\n✅ {total_rows} total records - DATA READY FOR TESTING!")


if __name__ == "__main__":
    # Generate data
    files = generate_sample_data()

    # Validate
    validate_sample_data()

    print("\n🚀 READY TO TEST:")
    print("  python handicapper_cli.py list")
    print("  python handicapper_cli.py parse --handle @CBB_Edge --text 'Illinois -3.5'")
    print("  python handicapper_cli.py map")
    print("  python handicapper_cli.py backtest")
