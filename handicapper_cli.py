#!/usr/bin/env python3
"""
GitHub-Native Handicapper App CLI
Run: python handicapper_cli.py --help
No Streamlit/Vercel - pure terminal + notebooks
"""

import argparse
import logging
import sys

import pandas as pd

from data_loader import CSVDataManager
from parser import HandicapperParser
from game_mapper import GameMapper

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def print_banner() -> None:
    print(
        """
╔══════════════════════════════════════════════════════╗
║           Handicapper Wisdom of Crowds CLI           ║
║                CSV-Only • GitHub-Native              ║
╚══════════════════════════════════════════════════════╝
    """
    )


# ---------------------------------------------------------------------------
# Command implementations
# ---------------------------------------------------------------------------


def cmd_list_handicappers(args: argparse.Namespace) -> None:
    """List all handicappers."""
    dm = CSVDataManager()
    try:
        data = dm.load_app_data()
    except (FileNotFoundError, AssertionError) as exc:
        print(f"❌ Could not load data: {exc}")
        sys.exit(1)

    if data["handicappers"].empty:
        print("No handicappers found. Add with 'add-handicapper'")
        return

    df = data["handicappers"].copy()
    print(f"\n📊 {len(df)} Handicappers:")
    cols = [c for c in ["handle", "tier", "status", "lifetime_roi", "total_picks"] if c in df.columns]
    print(df[cols].to_string(index=False))

    active = len(df[df["status"] == "active"]) if "status" in df.columns else "?"
    print(f"\n✅ Active: {active}/{len(df)}")


def cmd_add_handicapper(args: argparse.Namespace) -> None:
    """Add a new handicapper."""
    dm = CSVDataManager()

    new_capper = {
        "handle": args.handle,
        "tier": args.tier,
        "status": "active",
        "lifetime_roi": 0.0,
        "win_pct": 0.5,
        "total_picks": 0,
        "parse_template_id": 1,
        "notes": args.notes or "",
        "created_at": pd.Timestamp.now().isoformat(),
    }

    handicapper_id = dm.append_record("handicappers", new_capper)
    print(f"✅ Added handicapper ID {handicapper_id}: {args.handle}")


def cmd_parse_tweet(args: argparse.Namespace) -> None:
    """Parse a single tweet and save the resulting raw picks."""
    dm = CSVDataManager()
    try:
        data = dm.load_app_data()
    except (FileNotFoundError, AssertionError) as exc:
        print(f"❌ Could not load data: {exc}")
        sys.exit(1)

    capper = data["handicappers"][data["handicappers"]["handle"] == args.handle]
    if capper.empty:
        print(f"❌ Handicapper @{args.handle} not found")
        return

    handicapper_id = capper.iloc[0]["handicapper_id"]
    parser = HandicapperParser(dm.data_dir)

    raw_picks = parser.parse_tweet_to_raw_picks(
        args.text,
        int(handicapper_id),
        args.tweet_id,
        args.created_at,
    )

    if not raw_picks:
        print("⚠️  No picks could be parsed from that tweet text.")
        return

    print(f"\n✅ Parsed {len(raw_picks)} picks:")
    for pick in raw_picks:
        line_str = pick.get("line")
        line_display = str(line_str) if line_str is not None else "N/A"
        print(
            f"  📍 {pick['market']}: {pick['team_raw']} "
            f"{line_display} ({pick['units']}u)"
        )

    parser.save_raw_picks(raw_picks)
    print("\n💾 Saved to raw_picks.csv")


def cmd_map_picks(args: argparse.Namespace) -> None:
    """Map raw picks to games."""
    dm = CSVDataManager()
    try:
        data = dm.load_app_data()
    except (FileNotFoundError, AssertionError) as exc:
        print(f"❌ Could not load data: {exc}")
        sys.exit(1)

    mapper = GameMapper(dm.data_dir)
    unmapped = data["raw_picks"][data["raw_picks"]["parse_status"] == "success"]

    if unmapped.empty:
        print("No unmapped picks found")
        return

    print(f"🔄 Mapping {len(unmapped)} raw picks...")
    mapping_results = mapper.batch_map_raw_picks(unmapped, data["games"])

    display_cols = [c for c in ["raw_pick_id", "team_raw", "game_id", "mapping_status"] if c in mapping_results.columns]
    print(f"\n✅ Mapping complete:")
    print(mapping_results[display_cols].head().to_string(index=False))

    print("\n📈 Summary:")
    print(mapping_results["mapping_status"].value_counts().to_string())

    mapper.save_picks(mapping_results)
    print("\n💾 Saved to picks.csv")


def cmd_backtest(args: argparse.Namespace) -> None:
    """Simple backtest summary."""
    dm = CSVDataManager()
    try:
        data = dm.load_app_data()
    except (FileNotFoundError, AssertionError) as exc:
        print(f"❌ Could not load data: {exc}")
        sys.exit(1)

    if data["picks"].empty or data["games"].empty:
        print("Need picks.csv and games.csv with results")
        return

    print("📊 BACKTEST SUMMARY")
    print("Handicapper | Picks | Win% | ROI")
    print("-" * 40)

    picks_with_results = data["picks"].merge(data["games"], on="game_id", how="left")

    roi_by_capper = (
        picks_with_results.groupby("handicapper_id")
        .agg(picks=("pick_id" if "pick_id" in picks_with_results.columns else "handicapper_id", "count"))
        .round(3)
    )

    print(roi_by_capper.to_string())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    print_banner()

    parser = argparse.ArgumentParser(description="Handicapper Wisdom CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # list
    subparsers.add_parser("list", help="List handicappers")

    # add-handicapper
    p2 = subparsers.add_parser("add-handicapper", help="Add handicapper")
    p2.add_argument("--handle", required=True, help="Twitter handle")
    p2.add_argument(
        "--tier", choices=["sharp", "medium", "public"], default="medium"
    )
    p2.add_argument("--notes", help="Notes")

    # parse
    p3 = subparsers.add_parser("parse", help="Parse tweet")
    p3.add_argument("--handle", required=True, help="Handicapper handle")
    p3.add_argument("--text", required=True, help="Tweet text")
    p3.add_argument("--tweet-id", default="manual", help="Tweet ID")
    p3.add_argument("--created-at", default=None, help="ISO-8601 timestamp (default: now)")

    # map
    subparsers.add_parser("map", help="Map raw picks to games")

    # backtest
    subparsers.add_parser("backtest", help="Run backtest")

    args = parser.parse_args()

    if args.command == "list":
        cmd_list_handicappers(args)
    elif args.command == "add-handicapper":
        cmd_add_handicapper(args)
    elif args.command == "parse":
        cmd_parse_tweet(args)
    elif args.command == "map":
        cmd_map_picks(args)
    elif args.command == "backtest":
        cmd_backtest(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
