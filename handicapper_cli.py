"""
CLI interface for the CBB handicapper picks-tracking application.

Usage:
    python handicapper_cli.py backtest
    python handicapper_cli.py signals
    python handicapper_cli.py picks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data_loader import load_app_data
from backtest_engine import BacktestEngine


def cmd_backtest(args) -> None:
    engine = BacktestEngine(args.data_dir)
    engine.cmd_backtest(args)


def cmd_signals(args) -> None:
    engine = BacktestEngine(args.data_dir)
    engine.cmd_live_signals(args)


def cmd_picks(args) -> None:
    data = load_app_data(args.data_dir)
    picks = data['picks']
    if picks.empty:
        print("No picks found.")
        return
    print(picks.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CBB Handicapper CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--data-dir',
        default='data/handicapper',
        help='Directory containing handicapper CSV files (default: data/handicapper)',
    )

    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('backtest', help='Run backtest on graded picks')
    subparsers.add_parser('signals', help='Show live signals for upcoming games')
    subparsers.add_parser('picks', help='List all mapped picks')

    args = parser.parse_args()

    if args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'signals':
        cmd_signals(args)
    elif args.command == 'picks':
        cmd_picks(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
