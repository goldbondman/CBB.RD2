# Rolling Artifact Debug Notes

## Error addressed
`No successful cbb_predictions_rolling run with INFRA-predictions-rolling artifact found`

## Root cause pattern
The old resolver in `/C:/Users/brand/OneDrive/Desktop/CBB/CBB.RD2/.github/workflows/cbb_analytics.yml` scanned only the latest 20 successful runs and selected by artifact size. On busy repos this can miss valid artifacts that exist outside that short window.

## New behavior
- Consumer now resolves artifacts from `main` branch explicitly.
- Uses `.github/scripts/download_latest_artifact.py` with:
  - `workflow-file=cbb_predictions_rolling.yml`
  - `artifact-name=INFRA-predictions-rolling`
  - fallback marker artifact `INFRA-ready-cbb-predictions-rolling`
  - `max-runs=100`
- Workflow prints the last 15 successful rolling runs and whether each run has the artifact.

## Local repro commands
```bash
gh run list --workflow cbb_predictions_rolling.yml --branch main --status success --limit 15 --json databaseId,createdAt,headBranch,conclusion
python .github/scripts/download_latest_artifact.py --repo <owner/repo> --workflow-file cbb_predictions_rolling.yml --artifact-name INFRA-predictions-rolling --branch main --max-runs 100 --dest data/staging
```

## Failure diagnostics now emitted
When resolution fails, workflow prints:
- artifact name searched
- workflow file
- branch constraint
- max runs searched
- concrete next steps
