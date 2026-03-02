import re
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class HandicapperParser:
    """Parses Twitter handicapper tweets → raw_picks.csv records"""

    def __init__(self):
        self.handicapper_templates = {
            1: {  # @CBB_Edge style: "Illinois -3.5 (2u)"
                'spread': r'([A-Za-z\s&.\'\-]+?)\s*[-+]?(\d+\.?\d*)\s*\((\d+(?:\.\d+)?)u?\)',
                'ml': r'([A-Za-z\s&.\'\-]+?)\s*ML\s*\((\d+(?:\.\d+)?)u?\)',
                'total': r'(?:over|under|o|u)\s*(\d+\.?\d*)\s*\((\d+(?:\.\d+)?)u?\)'
            },
            2: {  # @HoopsLock style: "UConn/Kentucky OVER 142.5 (3u)"
                'total': r'([A-Za-z\s&\/\.\-]+?)\s+(?:OVER|UNDER|O|U)\s*(\d+\.?\d*)\s*\((\d+(?:\.\d+)?)u?\)',
                'spread': r'([A-Za-z\s&\/\.\-]+?)\s*[-+]?(\d+\.?\d*)\s*\((\d+(?:\.\d+)?)u?\)'
            },
            3: {  # @MidMajorGuru style: "FADE Duke ML +120"
                'fade': r'(?:FADE|NO)\s+([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+ML\b\s*(?:[-+]?(\d+\.?\d*))?',
                'ml': r'([A-Za-z\s&.\'\-]+?)\s*ML\s*[-+]?\d*'
            },
            4: {  # Generic fallback
                'spread': r'([A-Za-z\s&.\'\-]+?)\s*[-+]?(\d+\.?\d*)',
                'total': r'(?:over|under)\s*(\d+\.?\d*)',
                'ml': r'([A-Za-z\s&.\'\-]+?)\s*(?:ML|moneyline)'
            }
        }

    def normalize_text(self, text: str) -> str:
        """Clean tweet text for parsing"""
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'[^\w\s&./\-\+\(\)\d%]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip().lower()
        return text

    def extract_picks(self, tweet_text: str, handicapper_id: int) -> List[Dict]:
        """Main parsing function → raw_picks.csv format"""
        normalized = self.normalize_text(tweet_text)
        template = self.handicapper_templates.get(handicapper_id, self.handicapper_templates[4])

        picks = []

        for pattern_type, regex in template.items():
            matches = list(re.finditer(regex, normalized, re.IGNORECASE))

            for match in matches:
                pick = self._match_to_pick(match, pattern_type, handicapper_id)
                if pick:
                    picks.append(pick)

        # Detect fades as opposite side (fallback: only if no fades already found)
        if not any(p.get('market') == 'fade' for p in picks):
            fade_matches = re.findall(r'(?:fade|no|avoid)\s+([A-Za-z\s&.\'\-]+)', normalized, re.IGNORECASE)
            for team in fade_matches:
                picks.append({
                    'market': 'fade',
                    'team_raw': team.strip(),
                    'line': None,
                    'units': 1.0,
                    'odds': None,
                    'parse_status': 'fade_detected'
                })

        return picks if picks else [{'parse_status': 'failed', 'team_raw': normalized}]

    def _match_to_pick(self, match, pattern_type: str, handicapper_id: int) -> Optional[Dict]:
        """Convert regex match → structured pick"""
        groups = match.groups()

        try:
            if pattern_type == 'spread':
                team_raw, line, units = groups[:3]
                return {
                    'market': 'spread',
                    'team_raw': team_raw.strip(),
                    'line': float(line) if line else None,
                    'units': float(units) if units else 1.0,
                    'odds': None,
                    'parse_status': 'success'
                }
            elif pattern_type == 'total':
                team_raw, total, units = groups
                return {
                    'market': 'total',
                    'team_raw': team_raw.strip(),
                    'line': float(total),
                    'units': float(units) if units else 1.0,
                    'odds': None,
                    'parse_status': 'success'
                }
            elif pattern_type == 'ml':
                team_raw, units = groups
                return {
                    'market': 'moneyline',
                    'team_raw': team_raw.strip(),
                    'line': None,
                    'units': float(units) if units else 1.0,
                    'odds': None,
                    'parse_status': 'success'
                }
            elif pattern_type == 'fade':
                team_raw, odds = groups
                return {
                    'market': 'fade',
                    'team_raw': team_raw.strip(),
                    'line': None,
                    'units': 1.0,
                    'odds': odds,
                    'parse_status': 'fade_detected'
                }
        except (ValueError, IndexError):
            pass

        return None

    def parse_tweet_to_raw_picks(self, tweet_text: str, handicapper_id: int,
                                 tweet_id: str, created_at: str = "") -> List[Dict]:
        """Full pipeline: tweet → raw_picks.csv records"""
        picks = self.extract_picks(tweet_text, handicapper_id)

        for pick in picks:
            pick.update({
                'tweet_id': tweet_id,
                'handicapper_id': handicapper_id,
                'parsed_at': datetime.now().isoformat()
            })

        logger.info(f"Parsed tweet {tweet_id}: {len(picks)} picks")
        return picks

    def save_raw_picks(self, raw_picks: List[Dict], data_dir: str = "./data"):
        """Append parsed picks to raw_picks.csv"""
        from data_loader import CSVDataManager

        dm = CSVDataManager(data_dir)
        data = dm.load_app_data()

        next_id = dm.get_next_id('raw_picks')
        for i, pick in enumerate(raw_picks):
            pick['raw_pick_id'] = next_id + i

        new_df = pd.DataFrame(raw_picks)
        if not data['raw_picks'].empty:
            data['raw_picks'] = pd.concat([data['raw_picks'], new_df], ignore_index=True)
        else:
            data['raw_picks'] = new_df

        dm.save_app_data(data)
        logger.info(f"Saved {len(raw_picks)} new raw picks")


def test_parser():
    """Test with realistic tweet samples"""
    parser = HandicapperParser()

    test_tweets = [
        {
            'text': "Illinois -3.5 (2u) vs Wisconsin. Bulls are rolling. #CBB",
            'handicapper_id': 1,
            'tweet_id': '1845678901234567890'
        },
        {
            'text': "UConn/Kentucky OVER 142.5 (3u) - both shooting lights out LFG",
            'handicapper_id': 2,
            'tweet_id': '1845689012345678901'
        },
        {
            'text': "FADE Duke ML +120 vs UNC. Blue devils overrated this year",
            'handicapper_id': 3,
            'tweet_id': '1845690123456789012'
        },
        {
            'text': "Gonzaga +2.5 (1u) - Zags undervalued on road @CBB_Edge",
            'handicapper_id': 1,
            'tweet_id': '1845701234567890123'
        }
    ]

    print("=== TWEET PARSER TEST ===")
    all_picks = []

    for tweet in test_tweets:
        picks = parser.parse_tweet_to_raw_picks(
            tweet['text'], tweet['handicapper_id'], tweet['tweet_id']
        )
        print(f"\nTweet: {tweet['text'][:60]}...")
        print(f"Picks found: {len(picks)}")

        for pick in picks:
            print(f"  {pick['market']}: {pick['team_raw']} {pick['line']} ({pick['units']}u)")
            all_picks.append(pick)

    print(f"\n✅ Total picks parsed: {len(all_picks)}")
    return all_picks


if __name__ == "__main__":
    test_parser()
