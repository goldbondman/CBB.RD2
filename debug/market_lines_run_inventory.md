# Market Lines Run Inventory (2026-03-05)

## Workflow + Trigger Inventory
- Workflow file: `.github/workflows/market_lines.yml`
- Workflow name: `Market Lines Capture`
- Recent trigger observed: `schedule` run `22730197738` at `2026-03-05T18:14:07Z` (10:14 AM America/Los_Angeles)
- Job timezone assumptions before fix:
  - Runner clock: UTC
  - Ingestion date default: mixed (`resolve_date_range` UTC date, `run_capture` used `date.today()`)
  - No explicit `TZ` or `PIPELINE_TZ` set in workflow before fix

## Commands Executed in Action
- Market ingestion command template:
  - `python -m ingestion.market_lines --mode <mode> --append <append> --master-file <target_file> [--start-date ...] [--end-date ...] [--days-back ...]`
- Enrichment command:
  - `python -m enrichment.predictions_with_context`

## Environment Variables in Workflow (before fix)
- Inputs read from `github.event.inputs`:
  - `mode`, `start_date`, `end_date`, `days_back`, `append`, `target_file`
- Not explicitly set (before fix):
  - `TZ`
  - `PIPELINE_TZ`
  - `ESPN_SCOREBOARD_GROUPS`

## Files Touched / Produced
- Market pipeline outputs:
  - `data/market_lines_master.csv`
  - `data/market_lines_latest.csv`
  - `data/market_lines.csv`
  - `data/odds_snapshot.csv`
- Prediction context output:
  - `data/predictions_with_context.csv`

## Key Functions (Lineage)
- Enumeration + ingestion:
  - `ingestion.market_lines.fetch_espn_scoreboard`
  - `ingestion.market_lines.run_capture`
  - `ingestion.market_lines.build_market_row`
  - `ingestion.market_lines.write_master_market_file`
  - `ingestion.market_lines.regenerate_market_views`
- Context join:
  - `enrichment.predictions_with_context.build_predictions_with_context`
  - `enrichment.predictions_with_context._build_market_lines_fallback`

## Filters / Logic That Could Reduce Games
- Scoreboard enumeration source: ESPN scoreboard events
- Pre-fix request params were `dates=<yyyymmdd>&limit=200` (no `groups=50`)
- Pregame filtering in `run_capture` skipped games with status including `FINAL`
- Dedupe in master write path could collapse duplicates on key subset
- Latest view path had two implementations:
  - `write_latest_from_master` grouped by `event_id`
  - `regenerate_market_views` grouped by `game_id,book,market_type`

## Observed Failure Evidence from Run 22730197738
- In log: `Resolved date range: start=2026-03-05 end=2026-03-05`
- In log: `Market ingest summary: pulled=2 matched=2 ... inserted=2`
- Artifact outputs had only two games:
  - `401825560 Iowa Hawkeyes vs Michigan Wolverines`
  - `401825561 Michigan State Spartans vs Rutgers Scarlet Knights`
- `predictions_with_context` had 64 rows but only 2 market matches in log (`Market lines merged: 2/101 games matched`), causing many missing spread/total context values.

## Root-Cause Hypothesis from Discovery
- ESPN endpoint parameterization is the primary reducer:
  - `scoreboard?dates=20260305&limit=200` returns 2 events
  - `scoreboard?dates=20260305&groups=50&limit=1000` returns 46 events
- Secondary risk: mismatch between latest-view generation methods and mixed timezone defaults.
