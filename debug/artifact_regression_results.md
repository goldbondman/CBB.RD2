# Artifact Downloader Regression Results (Agent F)

## What was tested
1. Python syntax check:
   - `py -3 -m py_compile .github/scripts/download_latest_artifact.py`
2. Reproduction target path (`INFRA-predictions-rolling`):
   - `py -3 .github/scripts/download_latest_artifact.py --repo goldbondman/CBB.RD2 --workflow-file cbb_predictions_rolling.yml --artifact-name INFRA-predictions-rolling --branch main --max-runs 100 --dest debug/tmp_artifact_extract_validation2 --debug-json debug/artifact_payload_validation.json`
3. Shared-helper dependency path (`INFRA-espn-data`):
   - `py -3 .github/scripts/download_latest_artifact.py --repo goldbondman/CBB.RD2 --workflow-file update_espn_cbb.yml --artifact-name INFRA-espn-data --branch main --max-runs 50 --dest debug/tmp_artifact_extract_espn --debug-json debug/artifact_payload_validation_espn.json`
4. Missing-artifact classification path (`INFRA-ready-cbb-predictions-rolling`):
   - `py -3 .github/scripts/download_latest_artifact.py --repo goldbondman/CBB.RD2 --workflow-file cbb_predictions_rolling.yml --artifact-name INFRA-ready-cbb-predictions-rolling --branch main --max-runs 100 --dest debug/tmp_artifact_extract_ready --debug-json debug/artifact_payload_validation_ready.json`
5. Artifact publisher checks (non-helper workflows) via GitHub API:
   - `market_lines.yml` run `22773190092`: contains `INFRA-market-lines` and `INFRA-predictions-with-context`.

## Results
- Syntax check: PASS.
- `INFRA-predictions-rolling`: PASS, archive downloaded and extracted; diagnostic JSON written.
- `INFRA-espn-data`: PASS, archive downloaded and extracted; diagnostic JSON written.
- `INFRA-ready-cbb-predictions-rolling`: FAIL with classified `LOOKUP_FAILED` (expected classification behavior when artifact unavailable).
- Market-lines publishing artifacts: PASS (present in latest successful run).

## Expected success log snippet after fix
```text
[INFO] Found artifact name=INFRA-predictions-rolling run_id=... artifact_id=... size_bytes=... created_at=... searched_runs=...
[OK] Extracted artifact to data/staging
```

## Expected failure log snippet after fix
```text
[ERROR][STAGE=LOOKUP_FAILED] No matching non-expired artifact found. workflow='cbb_predictions_rolling.yml', artifact='INFRA-ready-cbb-predictions-rolling', branch='main', searched_runs=...
```

## Remaining risks
- If GitHub API media-type requirements change again, helper depends on centralized `_headers` and should be adjusted there.
- `INFRA-ready-cbb-predictions-rolling` publication behavior in `cbb_predictions_rolling.yml` may need separate investigation if it should always exist.
