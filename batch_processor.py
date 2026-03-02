#!/usr/bin/env python3
"""
Batch Tweet Processor for Handicapper Picks
Processes raw tweets through parse → map → backtest pipeline.
"""

import logging
import re
from typing import Optional, Tuple

import pandas as pd

from data_loader import CSVDataManager

logger = logging.getLogger(__name__)


class BatchTweetProcessor:
    """Processes batches of raw tweets into structured picks."""

    def __init__(self, data_dir: str = "./data"):
        self.dm = CSVDataManager(data_dir)

    def real_twitter_fetch(self) -> pd.DataFrame:
        """Use production scraper to fetch tweets from active handicappers."""
        from twitter_scraper import TwitterScraper, load_config
        config = load_config()
        scraper = TwitterScraper(config)
        return scraper.scrape_active_handicappers()

    def process_tweet_batch(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of raw tweets through the picks pipeline.

        Parameters
        ----------
        tweets_df:
            DataFrame of raw tweets as returned by :class:`TwitterScraper`.

        Returns
        -------
        pd.DataFrame
            Processed picks ready for backtesting.
        """
        if tweets_df.empty:
            logger.info("No tweets to process")
            return pd.DataFrame()

        logger.info(f"Processing {len(tweets_df)} tweets...")

        data = self.dm.load_app_data()

        new_picks = self._parse_tweets(tweets_df)
        if new_picks.empty:
            logger.info("No picks parsed from tweets")
            return pd.DataFrame()

        if not data['raw_picks'].empty:
            data['raw_picks'] = pd.concat([data['raw_picks'], new_picks], ignore_index=True)
        else:
            data['raw_picks'] = new_picks

        self.dm.save_app_data(data)
        logger.info(f"✅ Saved {len(new_picks)} new picks")
        return new_picks

    def _parse_tweets(self, tweets_df: pd.DataFrame) -> pd.DataFrame:
        """Parse raw tweet text into structured pick records."""
        picks = []
        for _, tweet in tweets_df.iterrows():
            text = str(tweet.get('text', ''))
            handle = tweet.get('handle', '')
            pick_datetime = tweet.get('created_at', '')

            pick_type, pick_spread, pick_total, pick_team = self._extract_pick(text)

            if pick_type:
                picks.append({
                    'handle': f"@{handle}" if not str(handle).startswith('@') else handle,
                    'tweet_id': tweet.get('tweet_id', ''),
                    'raw_text': text,
                    'pick_type': pick_type,
                    'pick_team': pick_team,
                    'pick_spread': pick_spread,
                    'pick_total': pick_total,
                    'pick_datetime_utc': pick_datetime,
                    'source': 'twitter_scrape',
                })

        return pd.DataFrame(picks)

    def _extract_pick(self, text: str) -> Tuple[Optional[str], Optional[float], Optional[float], Optional[str]]:
        """Extract pick details from tweet text.

        Returns
        -------
        tuple of (pick_type, pick_spread, pick_total, pick_team)
        """
        text_upper = text.upper()

        # FADE (fade = bet against) — check before ML to avoid mis-match
        fade_match = re.search(r'FADE\s+([A-Z][A-Z ]+?)(?:\s+ML|\s|$)', text_upper)
        if fade_match:
            team = fade_match.group(1).strip().title()
            return 'fade', None, None, team

        # OVER/UNDER total
        total_match = re.search(r'(OVER|UNDER)\s+([\d.]+)', text_upper)
        if total_match:
            direction = total_match.group(1)
            total = float(total_match.group(2))
            return 'total', None, total, direction

        # Spread (e.g. "Duke -5.5" or "UConn +3.5")
        spread_match = re.search(r'([A-Z][A-Z ]+?)\s+([+-][\d.]+)', text_upper)
        if spread_match:
            team = spread_match.group(1).strip().title()
            spread = float(spread_match.group(2))
            return 'spread', spread, None, team

        # Moneyline (ML)
        ml_match = re.search(r'([A-Z][A-Z ]+?)\s+ML', text_upper)
        if ml_match:
            team = ml_match.group(1).strip().title()
            return 'moneyline', None, None, team

        return None, None, None, None
