#!/usr/bin/env python3
"""
Handicapper analysis CLI: backtest and live signals.

Usage:
    python handicapper_cli.py backtest              # Full backtest analysis
    python handicapper_cli.py signals               # Today's live signals
    python handicapper_cli.py backtest --data-dir /path/to/data
"""

from __future__ import annotations

import argparse

from backtest_engine import cmd_backtest, cmd_live_signals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CBB Handicapper Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--data-dir', default='./data',
        help='Path to data directory (default: ./data)',
    )
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser(
        'backtest',
        help='Full backtest analysis: ROI, alignment patterns, top performers',
    )
    subparsers.add_parser(
        'signals',
        help="Live alignment signals for today's slate",
    )

    args = parser.parse_args()

    if args.command == 'backtest':
        cmd_backtest(args)
    elif args.command == 'signals':
        cmd_live_signals(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
