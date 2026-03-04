"""
Generate sample data for the CBB handicapper picks-tracking application.

Writes minimal CSV files to data/handicapper/ so the full pipeline can run
without a live data feed.

Usage:
    python generate_sample_data.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_sample_data(data_dir: str = "data/handicapper") -> None:
    """Write sample CSV files to *data_dir*."""
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # handicappers.csv
    # ------------------------------------------------------------------
    handicappers = pd.DataFrame([
        {
            'handicapper_id': 1,
            'handle': '@CBB_Edge',
            'tier': 'sharp',
            'status': 'active',
            'lifetime_roi': 0.032,
            'win_pct': 0.562,
            'total_picks': 245,
            'parse_template_id': 1,
            'notes': 'Consistent sharp, great with spreads',
            'created_at': '2026-01-15',
        },
        {
            'handicapper_id': 2,
            'handle': '@HoopsLock',
            'tier': 'sharp',
            'status': 'active',
            'lifetime_roi': 0.018,
            'win_pct': 0.548,
            'total_picks': 198,
            'parse_template_id': 2,
            'notes': 'Excellent totals specialist',
            'created_at': '2026-01-20',
        },
        {
            'handicapper_id': 3,
            'handle': '@MidMajorGuru',
            'tier': 'medium',
            'status': 'active',
            'lifetime_roi': -0.012,
            'win_pct': 0.521,
            'total_picks': 156,
            'parse_template_id': 3,
            'notes': 'Good with underdogs',
            'created_at': '2026-01-18',
        },
        {
            'handicapper_id': 4,
            'handle': '@PublicFader',
            'tier': 'public',
            'status': 'paused',
            'lifetime_roi': -0.045,
            'win_pct': 0.489,
            'total_picks': 89,
            'parse_template_id': 1,
            'notes': 'Reverse line movement tracker',
            'created_at': '2026-01-22',
        },
    ])
    handicappers.to_csv(out / 'handicappers.csv', index=False)

    # ------------------------------------------------------------------
    # games.csv
    # ------------------------------------------------------------------
    games = pd.DataFrame([
        {
            'game_id': 123,
            'date': '2026-02-28 19:00:00',
            'home_team': 'Illinois',
            'away_team': 'Wisconsin',
            'home_score': 78,
            'away_score': 72,
            'closing_spread': -4.5,
            'total_line': 138.5,
        },
        {
            'game_id': 124,
            'date': '2026-02-28 20:30:00',
            'home_team': 'Kentucky',
            'away_team': 'UConn',
            'home_score': 82,
            'away_score': 79,
            'closing_spread': None,
            'total_line': 142.0,
        },
        {
            'game_id': 125,
            'date': '2026-02-28 21:00:00',
            'home_team': 'Purdue',
            'away_team': 'Gonzaga',
            'home_score': 71,
            'away_score': 69,
            'closing_spread': -1.5,
            'total_line': 135.0,
        },
        {
            'game_id': 126,
            'date': '2026-02-28 22:15:00',
            'home_team': 'Houston',
            'away_team': 'Tennessee',
            'home_score': 65,
            'away_score': 58,
            'closing_spread': -6.5,
            'total_line': 124.0,
        },
        {
            'game_id': 127,
            'date': '2026-03-02 19:00:00',
            'home_team': 'Duke',
            'away_team': 'Kansas',
            'home_score': None,
            'away_score': None,
            'closing_spread': -3.5,
            'total_line': 144.0,
        },
        {
            'game_id': 128,
            'date': '2026-03-02 21:30:00',
            'home_team': 'Gonzaga',
            'away_team': 'Illinois',
            'home_score': None,
            'away_score': None,
            'closing_spread': -2.5,
            'total_line': 140.5,
        },
    ])
    games.to_csv(out / 'games.csv', index=False)

    # ------------------------------------------------------------------
    # raw_tweets.csv
    # ------------------------------------------------------------------
    raw_tweets = pd.DataFrame([
        {
            'tweet_id': '1845678901234567890',
            'handicapper_id': 1,
            'created_at': '2026-02-28 14:23:00',
            'text': 'Illinois -3.5 (2u) vs Wisconsin. Bulls are rolling.',
            'tweet_url': 'https://twitter.com/CBB_Edge/status/1845678901234567890',
            'ingested_at': '2026-02-28 14:25:00',
        },
        {
            'tweet_id': '1845689012345678901',
            'handicapper_id': 2,
            'created_at': '2026-02-28 15:10:00',
            'text': 'UConn/Kentucky OVER 142.5 (3u) - both shooting lights out',
            'tweet_url': 'https://twitter.com/HoopsLock/status/1845689012345678901',
            'ingested_at': '2026-02-28 15:12:00',
        },
        {
            'tweet_id': '1845690123456789012',
            'handicapper_id': 3,
            'created_at': '2026-02-28 16:45:00',
            'text': 'FADE Duke ML +120 vs UNC. Blue devils overrated.',
            'tweet_url': 'https://twitter.com/MidMajorGuru/status/1845690123456789012',
            'ingested_at': '2026-02-28 16:47:00',
        },
        {
            'tweet_id': '1845701234567890123',
            'handicapper_id': 1,
            'created_at': '2026-02-28 18:20:00',
            'text': 'Gonzaga +2.5 (1u) - Zags undervalued on road',
            'tweet_url': 'https://twitter.com/CBB_Edge/status/1845701234567890123',
            'ingested_at': '2026-02-28 18:22:00',
        },
    ])
    raw_tweets.to_csv(out / 'raw_tweets.csv', index=False)

    # ------------------------------------------------------------------
    # raw_picks.csv
    # ------------------------------------------------------------------
    raw_picks = pd.DataFrame([
        {
            'raw_pick_id': 1,
            'tweet_id': '1845678901234567890',
            'handicapper_id': 1,
            'market': 'spread',
            'team_raw': 'Illinois',
            'line': -3.5,
            'units': 2.0,
            'odds': None,
            'parse_status': 'success',
            'parsed_at': '2026-02-28 14:25:30',
        },
        {
            'raw_pick_id': 2,
            'tweet_id': '1845689012345678901',
            'handicapper_id': 2,
            'market': 'total',
            'team_raw': 'UConn/Kentucky',
            'line': 142.5,
            'units': 3.0,
            'odds': None,
            'parse_status': 'success',
            'parsed_at': '2026-02-28 15:12:45',
        },
        {
            'raw_pick_id': 3,
            'tweet_id': '1845690123456789012',
            'handicapper_id': 3,
            'market': 'moneyline',
            'team_raw': 'Duke',
            'line': None,
            'units': 1.0,
            'odds': '+120',
            'parse_status': 'fade_detected',
            'parsed_at': '2026-02-28 16:47:20',
        },
        {
            'raw_pick_id': 4,
            'tweet_id': '1845701234567890123',
            'handicapper_id': 1,
            'market': 'spread',
            'team_raw': 'Gonzaga',
            'line': 2.5,
            'units': 1.0,
            'odds': None,
            'parse_status': 'success',
            'parsed_at': '2026-02-28 18:22:15',
        },
    ])
    raw_picks.to_csv(out / 'raw_picks.csv', index=False)

    # ------------------------------------------------------------------
    # picks.csv
    # ------------------------------------------------------------------
    picks = pd.DataFrame([
        {
            'pick_id': 1,
            'raw_pick_id': 1,
            'handicapper_id': 1,
            'game_id': 123,
            'market': 'spread',
            'side': 'home',
            'line': -3.5,
            'units': 2.0,
            'mapping_status': 'ok',
            'created_at': '2026-02-28 14:25:30',
        },
        {
            'pick_id': 2,
            'raw_pick_id': 2,
            'handicapper_id': 2,
            'game_id': 124,
            'market': 'total',
            'side': 'over',
            'line': 142.5,
            'units': 3.0,
            'mapping_status': 'ok',
            'created_at': '2026-02-28 15:12:45',
        },
        {
            'pick_id': 3,
            'raw_pick_id': 4,
            'handicapper_id': 1,
            'game_id': 125,
            'market': 'spread',
            'side': 'away',
            'line': 2.5,
            'units': 1.0,
            'mapping_status': 'ok',
            'created_at': '2026-02-28 18:22:15',
        },
    ])
    picks.to_csv(out / 'picks.csv', index=False)

    print(f"✅ Sample data written to {out}/")
    print(f"   handicappers.csv : {len(handicappers)} rows")
    print(f"   games.csv        : {len(games)} rows")
    print(f"   raw_tweets.csv   : {len(raw_tweets)} rows")
    print(f"   raw_picks.csv    : {len(raw_picks)} rows")
    print(f"   picks.csv        : {len(picks)} rows")


if __name__ == "__main__":
    generate_sample_data()
