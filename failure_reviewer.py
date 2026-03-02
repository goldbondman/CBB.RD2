#!/usr/bin/env python3
"""
Parse Failure Review CLI + Notebook Interface
Review failed parses/mappings and manually correct → picks.csv
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from data_loader import CSVDataManager
from team_normalizer import CBBNormalizer
from game_mapper import GameMapper
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FailureReviewer:
    """Interactive review and correction of parse/mapping failures"""

    def __init__(self, data_dir="./data/handicapper"):
        self.dm = CSVDataManager(data_dir)
        self.normalizer = CBBNormalizer()
        self.mapper = GameMapper()
        self.data = self.dm.load_app_data()

    def get_failed_picks(self) -> pd.DataFrame:
        """Get all failed/partial raw picks needing review"""
        raw_picks = self.data['raw_picks']

        # Picks that already mapped successfully
        mapped_ids = set(self.data['picks']['raw_pick_id'].tolist())

        # Failed parses OR unmapped successful parses
        failed_mask = (
            (raw_picks['parse_status'] != 'success') |
            (
                (raw_picks['parse_status'] == 'success') &
                (~raw_picks['raw_pick_id'].isin(mapped_ids))
            )
        )

        failed = raw_picks[failed_mask].copy()
        if 'parsed_at' in failed.columns:
            failed = failed.sort_values('parsed_at', ascending=False)

        return failed

    def review_single_failure(self, raw_pick_id: int) -> dict:
        """Interactive review of single failed pick"""
        failed_pick = self.data['raw_picks'][
            self.data['raw_picks']['raw_pick_id'] == raw_pick_id
        ]
        if failed_pick.empty:
            print(f"❌ Pick ID {raw_pick_id} not found")
            return {}

        pick = failed_pick.iloc[0]
        print(f"\n🔍 REVIEWING PICK ID: {raw_pick_id}")
        print(f"  Tweet: {pick.get('tweet_id', 'N/A')}")
        team_raw = pick.get('team_raw', 'N/A') or 'N/A'
        print(f"  Raw text context: {str(team_raw)[:100]}")
        print(f"  Auto-parse status: {pick['parse_status']}")

        # Interactive correction
        corrected = self._interactive_correct(pick)
        if corrected:
            corrected['raw_pick_id'] = raw_pick_id
            corrected['mapping_status'] = 'manual_override'
            print(f"✅ Corrected pick: {corrected}")
            return corrected

        return {}

    def _interactive_correct(self, pick: pd.Series) -> Optional[dict]:
        """Interactive form for manual correction"""
        print("\n📝 MANUAL CORRECTION:")

        market = input(
            f"Market (spread/total/ml/fade) [{pick.get('market', 'spread')}]: "
        ).strip() or pick.get('market', 'spread')
        if market not in ['spread', 'total', 'moneyline', 'fade', 'ml']:
            print("❌ Invalid market")
            return None

        team_raw = input(f"Team(s) [{pick.get('team_raw', '')}]: ").strip()
        if not team_raw:
            team_raw = str(pick.get('team_raw', ''))
        if not team_raw:
            print("❌ Team required")
            return None

        line_str = input(f"Line [{pick.get('line', '')}]: ").strip()
        line = float(line_str) if line_str else pick.get('line', None)

        units_str = input(f"Units [{pick.get('units', 1.0)}]: ").strip()
        units = float(units_str) if units_str else float(pick.get('units', 1.0) or 1.0)

        side = input("Side (home/away/over/under) [auto]: ").strip()

        print(f"\n📋 SUMMARY:")
        print(f"  Market: {market}")
        print(f"  Team: {team_raw}")
        print(f"  Line: {line}")
        print(f"  Units: {units}")
        print(f"  Side: {side or 'auto'}")

        confirm = input("✅ Save this correction? (y/n): ").strip().lower()
        if confirm != 'y':
            print("❌ Skipped")
            return None

        return {
            'market': market,
            'team_raw': team_raw,
            'line': line,
            'units': units,
            'side': side,
            'handicapper_id': pick['handicapper_id'],
            'tweet_id': pick.get('tweet_id', ''),
        }

    def fix_and_map(self, raw_pick_id: int) -> bool:
        """Full fix + map → picks.csv pipeline"""
        # 1. Interactive correction
        corrected = self.review_single_failure(raw_pick_id)
        if not corrected:
            return False

        # 2. Map to game
        games_df = self.data['games']
        mapping_result = self.mapper.map_raw_pick_to_game(
            pd.Series(corrected), games_df
        )

        if mapping_result['game_id'] is None:
            print(f"⚠️  Could not map to game: {mapping_result['mapping_status']}")
            return False

        # 3. Save as pick
        final_pick = {**corrected, **mapping_result}
        pick_id = self.dm.append_record('picks', final_pick)

        print(f"✅ SAVED PICK ID: {pick_id}")
        print(f"  Game: {mapping_result['game_id']}")
        print(f"  Status: {mapping_result['mapping_status']}")

        return True

    def bulk_review_summary(self):
        """Show summary of all failures"""
        failed = self.get_failed_picks()

        if failed.empty:
            print("✅ No failed picks to review!")
            return

        print(f"\n📊 FAILURE SUMMARY ({len(failed)} total):")
        display_cols = [c for c in ['raw_pick_id', 'handicapper_id', 'team_raw', 'parse_status']
                        if c in failed.columns]
        print(failed[display_cols].head(10).to_string(index=False))

        print(f"\nBy status:")
        print(failed['parse_status'].value_counts().to_string())

        # Show recent 5 for quick review
        recent = failed.head(5)
        print(f"\n🔍 TOP 5 RECENT FAILURES:")
        for _, row in recent.iterrows():
            team_raw = str(row.get('team_raw', 'N/A') or 'N/A')
            print(f"  ID {row['raw_pick_id']}: {team_raw[:50]}... [{row['parse_status']}]")


def cmd_review_failures(args):
    """CLI: Review failures"""
    reviewer = FailureReviewer()
    reviewer.bulk_review_summary()

    if hasattr(args, 'raw_pick_id') and args.raw_pick_id:
        success = reviewer.fix_and_map(args.raw_pick_id)
        if success:
            print("✅ Successfully fixed and mapped!")
        else:
            print("❌ Fix failed")


def cmd_list_failures(args):
    """CLI: List all failures"""
    reviewer = FailureReviewer()
    failed = reviewer.get_failed_picks()

    if failed.empty:
        print("✅ No failures!")
        return

    display_cols = [c for c in ['raw_pick_id', 'team_raw', 'parse_status', 'parsed_at']
                    if c in failed.columns]
    print(f"\n📋 {len(failed)} FAILED PICKS:")
    print(failed[display_cols].to_string(index=False))


# CLI Integration (add to handicapper_cli.py from Prompt 6)
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse Failure Reviewer")
    subparsers = parser.add_subparsers(dest='command')

    # List failures
    subparsers.add_parser('list', help='List all failures')

    # Review single
    p2 = subparsers.add_parser('review', help='Review single failure')
    p2.add_argument('--id', type=int, dest='raw_pick_id', help='Raw pick ID')

    # Bulk summary
    subparsers.add_parser('summary', help='Summary stats')

    args = parser.parse_args()

    if args.command == 'list':
        cmd_list_failures(args)
    elif args.command == 'review':
        cmd_review_failures(args)
    elif args.command == 'summary':
        reviewer = FailureReviewer()
        reviewer.bulk_review_summary()
    else:
        parser.print_help()
