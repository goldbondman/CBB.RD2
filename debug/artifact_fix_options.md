# Artifact Fix Options (Agent C)

## Inputs from Agent A + Agent B
- Shared helper: `.github/scripts/download_latest_artifact.py` is the central downloader for `cbb_analytics.yml`.
- Proven failure: `Accept: application/octet-stream` on archive API URL returns `415` (`DOWNLOAD_FAILED`), while JSON accept works and returns redirect.
- Current wrapper in `.github/workflows/cbb_analytics.yml` mislabels all helper failures as lookup failures.

## Option 1 (Recommended): Patch shared helper + classify errors + workflow message cleanup
- Approach:
  - In helper, use JSON API Accept for archive request (`application/vnd.github+json`) instead of octet-stream.
  - Keep authenticated API call, follow redirect, write zip, validate bytes, extract.
  - Add explicit staged error taxonomy in helper output:
    - `LOOKUP_FAILED`, `METADATA_FAILED`, `DOWNLOAD_FAILED`, `EXTRACT_FAILED`, `VALIDATION_FAILED`
  - Emit machine-readable debug payload validation JSON from helper.
  - Update `cbb_analytics.yml` wrapper messages to say "artifact download failed" and surface helper stage.
- Pros:
  - Minimal code surface, centralized fix for all helper users.
  - Preserves current workflow intent and artifact chain.
  - Addresses both root cause and misclassification.
- Cons:
  - Requires small helper refactor for stage-aware errors.
- Risk:
  - Low; behavior remains API-driven and deterministic.
- Compatibility:
  - High; no change to artifact names, chain semantics, or downstream merge behavior.

## Option 2: Switch helper download to direct `/actions/artifacts/{id}/zip` with no custom Accept header
- Approach:
  - Ignore `archive_download_url`; build explicit zip endpoint from `artifact_id`.
  - Use default headers or JSON Accept, then follow redirects.
- Pros:
  - More explicit endpoint construction.
- Cons:
  - Slightly more invasive than needed; duplicates data already provided in metadata.
  - Still needs error taxonomy + workflow wrapper fix.
- Risk:
  - Medium-low; endpoint stable but introduces avoidable divergence from metadata contract.
- Compatibility:
  - Good, but less minimal than Option 1.

## Option 3: Replace helper with `gh api` + `actions/download-artifact` flow
- Approach:
  - Use `gh api` for discovery and then invoke `actions/download-artifact@v4` with computed run IDs.
- Pros:
  - Leverages existing Actions primitives.
- Cons:
  - Broad workflow churn across scripts/workflows.
  - Harder to preserve existing helper behavior and diagnostics.
- Risk:
  - Medium-high due to broader blast radius.
- Compatibility:
  - Lower; requires structural workflow changes.

## Recommendation
- Choose **Option 1**.
- Why:
  - It is the smallest robust fix for this repo.
  - It directly resolves the proven 415 cause.
  - It preserves shared helper usage and the existing dependency chain.
  - It enables correct error classification without redesigning workflows.

## Handoff to planning council
- Recommended option: Option 1.
- Fallback option: Option 2 if metadata URL behavior changes unexpectedly.
- Open questions:
  - Should helper write debug payload file by default or via optional flag?
  - Should wrapper messages include stage from helper output/env for concise triage?
