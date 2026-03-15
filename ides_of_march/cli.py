from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import pandas as pd

from .orchestrator import IDESOrchestrator


def _parse_as_of(value: str | None) -> pd.Timestamp:
    if value:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
        if pd.isna(ts):
            raise ValueError(f"Invalid --as-of value: {value}")
        return ts
    return pd.Timestamp.now(tz="UTC")


def _emit(result, as_json: bool) -> None:
    payload = {
        "ok": result.ok,
        "status": result.status,
        "error": result.error,
        "stages": [asdict(s) for s in result.stages],
        "outputs": result.outputs,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
        return

    print(f"Status: {result.status}")
    for stage in result.stages:
        print(f"- {stage.agent}: {stage.status}")
        for note in stage.notes:
            print(f"  - {note}")
    if result.outputs:
        print("Outputs:")
        for key, value in result.outputs.items():
            print(f"- {key}: {value}")
    if result.error:
        print(f"Error: {result.error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m ides_of_march.cli")
    parser.add_argument("--json", action="store_true", help="Emit structured JSON output")
    sub = parser.add_subparsers(dest="cmd", required=True)

    pred = sub.add_parser("predict", help="Run standalone prediction pipeline")
    pred.add_argument("--json", action="store_true", help="Emit structured JSON output")
    pred.add_argument("--as-of", default="", help="UTC timestamp override")
    pred.add_argument(
        "--mc-mode",
        default="confidence_only",
        choices=["confidence_only", "confidence_filter", "blended"],
        help="Monte Carlo operating mode",
    )
    pred.add_argument("--hours-ahead", type=int, default=48, help="Upcoming horizon in hours")
    pred.add_argument("--hours-back", type=int, default=1, help="Backward window in hours to include recently started games")

    backtest = sub.add_parser("backtest", help="Run variant backtests")
    backtest.add_argument("--json", action="store_true", help="Emit structured JSON output")
    backtest.add_argument("--as-of", default="", help="UTC timestamp override")
    backtest.add_argument("--start-date", default="", help="Optional UTC lower bound")
    backtest.add_argument("--end-date", default="", help="Optional UTC upper bound")
    backtest.add_argument(
        "--require-wagertalk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require WagerTalk historical odds to be present and matched in training rows (default: False).",
    )

    audit = sub.add_parser("audit", help="Run data-integrity audit for IDES pipeline")
    audit.add_argument("--json", action="store_true", help="Emit structured JSON output")
    audit.add_argument("--as-of", default="", help="UTC timestamp override")
    audit.add_argument("--hours-ahead", type=int, default=48, help="Upcoming horizon in hours")
    audit.add_argument("--hours-back", type=int, default=1, help="Backward window in hours to include recently started games")
    audit.add_argument("--strict", action="store_true", help="Exit non-zero on WARN")

    return parser


def cli_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    orch = IDESOrchestrator()

    as_of = _parse_as_of(getattr(args, "as_of", ""))

    if args.cmd == "predict":
        result = orch.predict(as_of=as_of, mc_mode=args.mc_mode, hours_ahead=args.hours_ahead, hours_back=args.hours_back)
        _emit(result, args.json)
        return 0 if result.ok else 1

    if args.cmd == "backtest":
        start = args.start_date.strip() or None
        end = args.end_date.strip() or None
        result = orch.backtest(
            as_of=as_of,
            start_date=start,
            end_date=end,
            require_wagertalk=bool(args.require_wagertalk),
        )
        _emit(result, args.json)
        return 0 if result.ok else 1

    if args.cmd == "audit":
        result = orch.audit(as_of=as_of, hours_ahead=args.hours_ahead, hours_back=args.hours_back)
        _emit(result, args.json)
        if args.strict and result.status == "FAIL":
            return 1
        return 0 if result.status in {"PASS", "WARN"} else 1

    parser.error(f"Unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(cli_main())
