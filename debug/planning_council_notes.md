# Planning Council Notes (Agent A + Agent B + Agent C + Orchestrator)

## Participants
- Agent A (Discovery): mapped helper + workflow call sites.
- Agent B (Troubleshooting): proved endpoint/header mismatch and misclassification.
- Agent C (Brainstorming): generated and compared fix options.
- Orchestrator: validated chain impact and constrained scope.

## Disagreements
- Whether to change endpoint construction (`archive_download_url` vs manual `/artifacts/{id}/zip`).
- Whether to leave workflow wrappers unchanged and rely only on helper error text.

## Resolved decisions
- Keep `archive_download_url` (already correct and proven); do not add endpoint reconstruction complexity.
- Change archive request Accept header to JSON-compatible media type.
- Add staged error taxonomy in helper so lookup/download/extract failures are distinguishable.
- Update `cbb_analytics.yml` wrapper messages to avoid false "lookup failed" reporting.

## Tradeoffs considered
- Minimal patch vs broad workflow redesign:
  - Broad redesign (replace helper entirely) rejected due unnecessary blast radius.
- Logging verbosity:
  - Added compact machine-readable diagnostics instead of excessive log spam.

## Rejected alternatives and why
- Full migration to `actions/download-artifact` for all remote artifacts:
  - Rejected; requires significant workflow restructuring and run-id wiring changes.
- Keep octet-stream and rely on redirect:
  - Rejected; empirically fails with HTTP 415.

## Final implementation direction
- Apply central helper patch + targeted wrapper wording fix.
- Validate with known failing artifact and shared helper paths.
