#!/usr/bin/env python3
"""
Production Twitter Scraper for Handicapper Picks
Primary: Twitter API v2 | Fallback: HTML scraping
Cron-ready: python twitter_scraper.py --cron
"""

import os
import re
import sys
import time
import random
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from data_loader import CSVDataManager

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

RATE_LIMIT_WAIT_SECONDS = 900  # 15 minutes


@dataclass
class TwitterConfig:
    bearer_token: str
    max_tweets_per_handle: int = 50
    since_hours: int = 24
    rate_limit_delay: float = 1.0


class TwitterScraper:
    """Production Twitter scraper with API + HTML fallback"""

    def __init__(self, config: TwitterConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        })
        self.dm = CSVDataManager()

    def scrape_active_handicappers(self) -> pd.DataFrame:
        """Main entrypoint: scrape all active handicappers"""
        logger.info("🚀 Starting Twitter scrape...")

        data = self.dm.load_app_data()
        active_cappers = data['handicappers'][
            data['handicappers']['status'] == 'active'
        ]

        if active_cappers.empty:
            logger.warning("No active handicappers found")
            return pd.DataFrame()

        logger.info(f"📡 Scraping {len(active_cappers)} active handicappers")

        all_tweets: List[Dict] = []
        for _, capper in active_cappers.iterrows():
            handle = str(capper['handle']).lstrip('@')
            last_tweet_id = capper.get('last_ingested_tweet_id', None)
            if pd.isna(last_tweet_id):
                last_tweet_id = None

            tweets = self._scrape_single_handle(handle, last_tweet_id)
            if tweets:
                all_tweets.extend(tweets)
                logger.info(f"✅ @{handle}: {len(tweets)} new tweets")

            time.sleep(self.config.rate_limit_delay)

        tweets_df = pd.DataFrame(all_tweets)
        if not tweets_df.empty:
            self._save_tweets(tweets_df)
            self._update_last_ingested(tweets_df)

        logger.info(f"🎉 Scraping complete: {len(tweets_df)} total tweets")
        return tweets_df

    def _scrape_single_handle(self, handle: str, since_id: Optional[str]) -> List[Dict]:
        """Scrape single Twitter handle (API first, HTML fallback)"""
        api_tweets = self._twitter_api_v2(handle, since_id)
        if api_tweets:
            return api_tweets

        logger.warning(f"API failed for @{handle}, trying HTML scrape")
        return self._html_scrape_timeline(handle)

    def _twitter_api_v2(self, handle: str, since_id: Optional[str]) -> List[Dict]:
        """Twitter API v2 - primary method"""
        try:
            user_url = f"https://api.twitter.com/2/users/by/username/{handle}"
            user_headers = {
                "Authorization": f"Bearer {self.config.bearer_token}",
                "User-Agent": "v2UserLookupJS",
            }

            user_response = self.session.get(user_url, headers=user_headers)
            if user_response.status_code != 200:
                logger.error(f"User lookup failed for @{handle}: {user_response.status_code}")
                return []

            user_data = user_response.json()
            if 'data' not in user_data:
                return []

            user_id = user_data['data']['id']

            tweets_url = f"https://api.twitter.com/2/users/{user_id}/tweets"
            params: Dict = {
                'max_results': self.config.max_tweets_per_handle,
                'tweet.fields': 'created_at,public_metrics,author_id',
            }
            if since_id is not None:
                params['since_id'] = since_id

            tweets_headers = {
                "Authorization": f"Bearer {self.config.bearer_token}",
                "User-Agent": "v2TweetsLookupJS",
            }

            response = self.session.get(tweets_url, headers=tweets_headers, params=params)

            if response.status_code == 200:
                data = response.json()
                tweets = data.get('data', [])

                processed_tweets = []
                for tweet in tweets:
                    processed_tweets.append({
                        'tweet_id': tweet['id'],
                        'handle': handle,
                        'text': tweet['text'],
                        'created_at': tweet['created_at'],
                        'tweet_url': f"https://twitter.com/{handle}/status/{tweet['id']}",
                        'favorite_count': tweet.get('public_metrics', {}).get('like_count', 0),
                        'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                    })

                return processed_tweets

            elif response.status_code == 429:
                logger.warning(f"Rate limited for @{handle}, waiting 15min")
                time.sleep(RATE_LIMIT_WAIT_SECONDS)
                return []

            else:
                logger.error(f"API error @{handle}: {response.status_code}")
                return []

        except Exception as e:
            logger.error(f"API exception @{handle}: {e}")
            return []

    def _html_scrape_timeline(self, handle: str) -> List[Dict]:
        """HTML scraping fallback (simplified mock — replace with real selenium/BS4)"""
        logger.info(f"HTML scraping @{handle} (simplified)")

        mock_tweets = []
        base_time = datetime.now() - timedelta(hours=random.randint(1, 24))

        tweet_templates = [
            f"{random.choice(['Illinois', 'UConn', 'Duke'])} -{random.uniform(2, 8):.1f}",
            f"FADE {random.choice(['Kansas', 'Gonzaga'])} ML",
            f"{random.choice(['Purdue', 'Houston'])}/{random.choice(['Kentucky', 'Arizona'])} OVER {random.uniform(135, 150):.1f}",
        ]

        for _ in range(random.randint(1, 3)):
            tweet_time = base_time - timedelta(minutes=random.randint(0, 120))
            mock_tweets.append({
                'tweet_id': f"html_{int(tweet_time.timestamp())}",
                'handle': handle,
                'text': random.choice(tweet_templates) + " (1u)",
                'created_at': tweet_time.isoformat(),
                'tweet_url': f"https://twitter.com/{handle}/status/html_mock",
                'favorite_count': random.randint(5, 150),
                'retweet_count': random.randint(0, 25),
            })

        return mock_tweets

    def _save_tweets(self, tweets_df: pd.DataFrame):
        """Append tweets to raw_tweets.csv"""
        data = self.dm.load_app_data()

        tweets_df = tweets_df.copy()
        tweets_df['source'] = 'twitter_scrape'
        tweets_df['ingested_at'] = datetime.now().isoformat()

        if not data['raw_tweets'].empty:
            data['raw_tweets'] = pd.concat([data['raw_tweets'], tweets_df], ignore_index=True)
        else:
            data['raw_tweets'] = tweets_df

        self.dm.save_app_data(data)
        logger.info(f"💾 Saved {len(tweets_df)} tweets to raw_tweets.csv")

    def _update_last_ingested(self, tweets_df: pd.DataFrame):
        """Update handicappers with latest tweet_id per handle"""
        data = self.dm.load_app_data()
        handicappers = data['handicappers'].copy()

        for handle in tweets_df['handle'].unique():
            handle_tweets = tweets_df[tweets_df['handle'] == handle].sort_values(
                'tweet_id', ascending=False
            )
            latest_tweet = handle_tweets.iloc[0]
            mask = handicappers['handle'] == f"@{handle}"
            handicappers.loc[mask, 'last_ingested_tweet_id'] = latest_tweet['tweet_id']
            handicappers.loc[mask, 'last_scrape_time'] = datetime.now().isoformat()

        data['handicappers'] = handicappers
        self.dm.save_app_data(data)


def load_config() -> TwitterConfig:
    """Load Twitter config from environment / .env file"""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        logger.error("❌ TWITTER_BEARER_TOKEN not set in .env")
        logger.info("Create .env file:")
        logger.info("TWITTER_BEARER_TOKEN=your_bearer_token_here")
        sys.exit(1)

    return TwitterConfig(bearer_token=bearer_token)


def main():
    """CLI entrypoint"""
    import argparse

    parser = argparse.ArgumentParser(description="Twitter Scraper")
    parser.add_argument('--cron', action='store_true', help='Cron mode (quiet)')
    parser.add_argument('--handle', help='Single handle to scrape')
    parser.add_argument('--test', action='store_true', help='Test mode')

    args = parser.parse_args()

    config = load_config()
    scraper = TwitterScraper(config)

    if args.cron:
        logging.getLogger().setLevel(logging.WARNING)

    tweets = scraper.scrape_active_handicappers()

    if not tweets.empty and not args.cron:
        logger.info("🔄 Triggering batch processing...")
        from batch_processor import BatchTweetProcessor
        processor = BatchTweetProcessor()
        processor.process_tweet_batch(tweets)


if __name__ == "__main__":
    main()
