#!/usr/bin/env python3
"""
Handicapper CLI – main entry point for the CBB handicapper app.
"""

from __future__ import annotations

import argparse

from failure_reviewer import cmd_list_failures, cmd_review_failures, FailureReviewer


def main():
    parser = argparse.ArgumentParser(
        description="CBB Handicapper CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # review-failures subcommand
    p_review = subparsers.add_parser(
        "review-failures",
        help="Review and fix parse failures interactively",
    )
    p_review.add_argument(
        "--id",
        type=int,
        dest="raw_pick_id",
        help="Specific raw_pick_id to review and fix",
    )

    # list-failures subcommand
    subparsers.add_parser(
        "list-failures",
        help="List all failed/unmapped raw picks",
    )

    # summary subcommand
    subparsers.add_parser(
        "summary",
        help="Show failure summary statistics",
    )

    args = parser.parse_args()

    if args.command == "review-failures":
        cmd_review_failures(args)
    elif args.command == "list-failures":
        cmd_list_failures(args)
    elif args.command == "summary":
        reviewer = FailureReviewer()
        reviewer.bulk_review_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
