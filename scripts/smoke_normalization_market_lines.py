"""Smoke checks for market_lines normalization helpers."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingestion.market_lines import _norm_bool, _norm_int


def run() -> None:
    assert _norm_int("") is None
    assert _norm_int(None) is None
    assert _norm_int("0") == 0
    assert _norm_bool("", default=True) is True
    assert _norm_bool("", default=False) is False
    assert _norm_bool(None, default=True) is True
    assert _norm_bool("false") is False
    print("normalization smoke checks passed")


if __name__ == "__main__":
    run()
