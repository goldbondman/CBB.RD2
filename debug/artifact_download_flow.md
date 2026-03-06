# Artifact Download Flow (Agent A Discovery)

## Scope searched
- `.github/workflows/cbb_analytics.yml`
- `.github/workflows/cbb_predictions_rolling.yml`
- `.github/workflows/market_lines.yml`
- `.github/workflows/update_espn_cbb.yml`
- `.github/scripts/download_latest_artifact.py`

## Artifact dependency chain (as implemented)
- `update_espn_cbb.yml` uploads `INFRA-espn-data` at `.github/workflows/update_espn_cbb.yml:197-218`.
- `market_lines.yml` uploads:
  - `INFRA-market-lines` at `.github/workflows/market_lines.yml:182-194`
  - `INFRA-predictions-with-context` at `.github/workflows/market_lines.yml:197-203`
- `cbb_predictions_rolling.yml` uploads:
  - `INFRA-predictions-rolling` at `.github/workflows/cbb_predictions_rolling.yml:499-516`
  - `INFRA-ready-cbb-predictions-rolling` at `.github/workflows/cbb_predictions_rolling.yml:518-525`
- `cbb_analytics.yml` consumes:
  - best-effort `INFRA-espn-data` via helper script at `.github/workflows/cbb_analytics.yml:180-194`
  - `INFRA-market-lines` and `INFRA-predictions-with-context` via `actions/download-artifact@v4` at `.github/workflows/cbb_analytics.yml:195-209`
  - `INFRA-predictions-rolling` via helper script at `.github/workflows/cbb_analytics.yml:211-227`
  - `INFRA-ready-cbb-predictions-rolling` via helper script at `.github/workflows/cbb_analytics.yml:229-245`

## Shared helper vs duplicated logic
- Shared helper for cross-run latest-artifact resolution/download:
  - `.github/scripts/download_latest_artifact.py`
- It is invoked in `cbb_analytics.yml` at lines `187`, `217`, `235`.
- Other workflows mostly use `actions/download-artifact@v4` with known `run-id` and do not use this helper.

## Exact helper flow (`download_latest_artifact.py`)
1. Lookup successful workflow runs:
   - endpoint: `GET /repos/{repo}/actions/workflows/{workflow_ref}/runs?...`
   - code: `_find_artifact`, lines `77-83`
   - headers: `_headers(..., accept="application/vnd.github+json")`, lines `25-35`
2. Lookup artifacts for each run:
   - endpoint: `GET /repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100`
   - code: `_find_artifact`, line `91`
   - headers: JSON accept via `_api_json`, lines `34-43`
3. Select first matching non-expired artifact with positive size:
   - code: lines `92-100`
4. Read `archive_download_url`:
   - code: lines `154-158`
5. Download archive bytes:
   - code: `_download_binary`, lines `46-55`
   - current headers: `Accept: application/octet-stream` (line `47`)
   - urlopen follows redirects by default in urllib
6. Write temporary zip + extract:
   - temp file write lines `161-164`
   - extract lines `58-67`, invoked at `166`
7. Emit outputs (`run_id`, `artifact_id`) and success log:
   - lines `108-114`, `170-171`

## Current error handling path
- Helper raises `ArtifactDownloadError` for any phase:
  - lookup (`No successful runs found...`, lines `85-87`; `No matching non-expired artifact...`, lines `101-105`)
  - download (`Artifact download failed...`, line `53`)
  - extract (`Unsafe path...`, line `65`)
- In `cbb_analytics.yml` rolling downloads, all helper failures are wrapped with a hardcoded lookup message:
  - `.github/workflows/cbb_analytics.yml:223-226`
  - `.github/workflows/cbb_analytics.yml:241-244`
  - This currently conflates lookup failures with download/extract/validation failures.

## Suspicious area for Agent B
- `_download_binary` sends `Accept: application/octet-stream` at line `47`.
- Reported failure shows HTTP `415` requiring JSON `Accept`, indicating endpoint/header mismatch at download time.
- The workflow wrapper message ("Artifact lookup failed") is not stage-aware and likely misclassifies failures.

## Agent A handoff note
- Focus Agent B on:
  - verifying the exact endpoint hit during download and the exact `Accept` header.
  - proving whether lookup succeeded before download failed.
  - proving misclassification path in `cbb_analytics.yml` shell wrapper.
- Centralized fix surface likely includes:
  - `.github/scripts/download_latest_artifact.py` (header/endpoint/error taxonomy)
  - `.github/workflows/cbb_analytics.yml` (error message classification/logging).
