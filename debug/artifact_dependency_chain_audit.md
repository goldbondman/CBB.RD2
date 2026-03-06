# Artifact Dependency Chain Audit

## Chain overview
1. `update_espn_cbb.yml` publishes `INFRA-espn-data`.
2. `market_lines.yml` publishes `INFRA-market-lines` and `INFRA-predictions-with-context`.
3. `cbb_predictions_rolling.yml` publishes `INFRA-predictions-rolling` and `INFRA-ready-cbb-predictions-rolling`.
4. `cbb_analytics.yml` downloads/merges these artifacts and runs ESPN fallback only when Tier A is unavailable.

## Shared downloader impact
- Central helper patched: `.github/scripts/download_latest_artifact.py`.
- Current helper consumers in chain:
  - `INFRA-espn-data` download in `cbb_analytics.yml`
  - `INFRA-predictions-rolling` download in `cbb_analytics.yml`
  - `INFRA-ready-cbb-predictions-rolling` download in `cbb_analytics.yml`
- `INFRA-market-lines` and `INFRA-predictions-with-context` continue via `actions/download-artifact@v4` and are unaffected by helper internals.

## Post-fix behavior expectations
- Lookup failures are emitted as `[ERROR][STAGE=LOOKUP_FAILED] ...`
- Metadata failures are emitted as `[ERROR][STAGE=METADATA_FAILED] ...`
- Download failures are emitted as `[ERROR][STAGE=DOWNLOAD_FAILED] ...`
- Extraction failures are emitted as `[ERROR][STAGE=EXTRACT_FAILED] ...`
- Validation failures are emitted as `[ERROR][STAGE=VALIDATION_FAILED] ...`

## Cross-workflow compatibility assessment
- No artifact names changed.
- No destination paths changed.
- No workflow dependency ordering changed.
- Error text in `cbb_analytics.yml` wrappers now reflects download-stage failures instead of forcing "lookup failed".

## Noted runtime observation
- Latest successful `cbb_predictions_rolling.yml` run (`22692149656`) currently exposes `INFRA-predictions-rolling` in artifact list.
- `INFRA-ready-cbb-predictions-rolling` did not appear in the same run's artifact list during this validation; helper now reports this correctly as `LOOKUP_FAILED` instead of misclassifying a downstream failure.
