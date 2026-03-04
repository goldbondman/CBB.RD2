#!/usr/bin/env python3
"""
Batch Tweet Processor - COMPLETE PIPELINE
1. Fetch tweets (mock API for now)
2. Parse → raw_picks.csv
3. Map → picks.csv
4. Generate live signals + backtest
Single command: python batch_processor.py --tweets-file data/sample_tweets.csv
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from data_loader import CSVDataManager
from parser import HandicapperParser
from team_normalizer import CBBNormalizer
from game_mapper import GameMapper
from backtest_engine import BacktestEngine

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BatchTweetProcessor:
    """End-to-end tweet processing pipeline."""

    def __init__(self, data_dir: str = "data/handicapper"):
        self.data_dir = Path(data_dir)
        self.dm = CSVDataManager(data_dir)
        self.parser = HandicapperParser()
        self.normalizer = CBBNormalizer()
        self.mapper = GameMapper(data_dir)
        self.backtest = BacktestEngine(data_dir)

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def process_tweet_batch(self, tweets_df: pd.DataFrame) -> None:
        """Main pipeline: tweets DataFrame → full analysis."""
        print("🚀 BATCH TWEET PROCESSOR STARTED")
        print(f"📥 Processing {len(tweets_df)} tweets")

        # 1. Parse each tweet into raw picks
        new_raw_picks: list = []
        new_tweets: list = []

        for _, tweet in tweets_df.iterrows():
            raw_picks = self.parser.parse_tweet_to_raw_picks(
                str(tweet['text']),
                tweet['handicapper_id'],
                str(tweet['tweet_id']),
                str(tweet['created_at']),
            )

            for pick in raw_picks:
                pick.update({
                    'tweet_id': str(tweet['tweet_id']),
                    'handicapper_id': tweet['handicapper_id'],
                })
                new_raw_picks.append(pick)

            new_tweets.append({
                'tweet_id': str(tweet['tweet_id']),
                'handicapper_id': tweet['handicapper_id'],
                'created_at': str(tweet['created_at']),
                'text': str(tweet['text']),
                'tweet_url': str(tweet.get('tweet_url', '')),
                'ingested_at': datetime.now().isoformat(),
            })

        # 2. Persist raw tweets + raw picks
        self._append_to_csv('raw_tweets', new_tweets)
        self._append_to_csv('raw_picks', new_raw_picks)
        print(f"✅ Parsed {len(new_raw_picks)} raw picks")

        # 3. Map successful raw picks → games
        data = self.dm.load_app_data()
        picks_df = pd.DataFrame(new_raw_picks)
        successful = picks_df[picks_df['parse_status'] == 'success'] if not picks_df.empty else pd.DataFrame()

        if not successful.empty:
            mapping_results = self.mapper.batch_map_raw_picks(successful, data['games'])
            self.mapper.save_picks(mapping_results)
            print(f"✅ Mapped {len(mapping_results)} picks to games")
        else:
            print("⚠️  No successfully parsed picks to map")

        # 4. Backtest + live signals
        self.backtest.cmd_backtest(None)
        self.backtest.cmd_live_signals(None)

        print("\n🎉 PIPELINE COMPLETE!")
        print("💾 Updated: raw_tweets.csv, raw_picks.csv, picks.csv, backtest_results.csv")

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _append_to_csv(self, table_name: str, records: list) -> None:
        """Append records to a CSV table with auto-assigned IDs."""
        if not records:
            return

        id_col = f"{table_name.rstrip('s')}_id"
        next_id = self.dm.get_next_id(table_name)

        for i, record in enumerate(records):
            if id_col not in record:
                record[id_col] = next_id + i

        new_df = pd.DataFrame(records)
        csv_path = self.data_dir / f"{table_name}.csv"

        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
        else:
            combined = new_df

        combined.to_csv(csv_path, index=False)

    # ------------------------------------------------------------------
    # Mock Twitter fetch
    # ------------------------------------------------------------------

    def mock_twitter_fetch(self, handles: list, since_hours: int = 1) -> pd.DataFrame:
        """Mock Twitter API — replace with real scraping."""
        print(f"🔄 Mock fetching tweets from {len(handles)} handles...")

        base_time = datetime.now() - timedelta(hours=since_hours)

        tweet_templates = [
            "{} -{:.1f} ({:.1f}u) - 🔥 confidence",
            "{} vs {} OVER {:.1f} ({:.1f}u)",
            "FADE {} ML - overvalued",
            "{} +{:.1f} LIVE DOG ({:.1f}u)",
        ]

        teams = ['Illinois', 'UConn', 'Kansas', 'Duke', 'Gonzaga', 'Houston', 'Purdue']
        mock_tweets: list = []

        for handle in handles:
            for _ in range(random.randint(1, 4)):
                tweet_time = base_time - timedelta(minutes=random.randint(0, 60))
                template = random.choice(tweet_templates)
                team1 = random.choice(teams)
                team2 = random.choice([t for t in teams if t != team1])
                line = round(random.uniform(1.5, 8.5), 1)
                units = random.choice([1.0, 1.5, 2.0, 2.5])
                total = round(140.5 + random.uniform(-5, 5), 1)

                if 'vs' in template:
                    text = template.format(team1, team2, total, units)
                elif 'FADE' in template:
                    text = template.format(team1)
                else:
                    text = template.format(team1, line, units)

                mock_tweets.append({
                    'tweet_id': f"mock_{int(tweet_time.timestamp())}_{handle}",
                    'handle': handle,
                    'handicapper_id': 1,
                    'text': text,
                    'created_at': tweet_time.isoformat(),
                    'tweet_url': f"https://twitter.com/{handle}/status/mock",
                })

        mock_df = pd.DataFrame(mock_tweets)
        mock_path = self.data_dir / 'mock_tweets.csv'
        mock_df.to_csv(mock_path, index=False)
        print(f"✅ Generated {len(mock_tweets)} mock tweets → {mock_path}")

        return mock_df


# ---------------------------------------------------------------------------
# Sample data helper
# ---------------------------------------------------------------------------

def create_sample_tweet_batch(data_dir: str = "data/handicapper") -> pd.DataFrame:
    """Create sample_tweets.csv for testing."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    sample_tweets = pd.DataFrame({
        'tweet_id': ['1849999999999999999', '1850000000000000000'],
        'handicapper_id': [1, 2],
        'text': [
            "Illinois -3.5 (2u) vs Wisconsin | Gonzaga ML",
            "UConn/Kentucky OVER 142.5 (3u) - both teams shoot 40% from 3",
        ],
        'created_at': ['2026-03-01T14:23:00', '2026-03-01T15:10:00'],
        'tweet_url': [
            'https://twitter.com/CBB_Edge/status/184999',
            'https://twitter.com/HoopsLock/status/185000',
        ],
    })

    path = out / 'sample_tweets.csv'
    sample_tweets.to_csv(path, index=False)
    print(f"✅ Created {path} for testing")
    return sample_tweets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_batch_process(args) -> None:
    processor = BatchTweetProcessor(args.data_dir)

    if args.mock:
        data = processor.dm.load_app_data()
        handles = data['handicappers']['handle'].tolist()
        tweets_df = processor.mock_twitter_fetch(handles)
    else:
        tweets_df = pd.read_csv(args.tweets_file)

    processor.process_tweet_batch(tweets_df)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Batch Tweet Processor")
    arg_parser.add_argument('--tweets-file', help='CSV with tweets to process')
    arg_parser.add_argument('--mock', action='store_true', help='Generate mock tweets')
    arg_parser.add_argument('--sample', action='store_true', help='Create sample_tweets.csv')
    arg_parser.add_argument(
        '--data-dir',
        default='data/handicapper',
        help='Directory containing handicapper CSV files (default: data/handicapper)',
    )

    args = arg_parser.parse_args()

    if args.sample:
        create_sample_tweet_batch(args.data_dir)
    elif args.mock or args.tweets_file:
        cmd_batch_process(args)
    else:
        arg_parser.print_help()
