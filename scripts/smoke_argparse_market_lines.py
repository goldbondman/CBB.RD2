"""Smoke-check parser wiring for market_lines CLI."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingestion.market_lines import build_parser


def run() -> None:
    parser = build_parser(Path("data"))
    option_strings = [opt for action in parser._actions for opt in action.option_strings]
    assert option_strings.count("--days-back") == 1, "--days-back option should only be registered once"

    parsed = parser.parse_args(["--mode", "pregame"])
    assert parsed.days_back is None
    assert parsed.start_date is None
    assert parsed.end_date is None
    print("argparse smoke checks passed")


if __name__ == "__main__":
    run()
